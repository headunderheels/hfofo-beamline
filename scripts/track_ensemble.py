#!/usr/bin/env python3
"""Ensemble version of track_full_channel.py: vmap N independent MCS seeds
through the same deterministic backbone (field + GH2 + presswall + windows +
wedges), to test whether the ENSEMBLE MEAN (not a single noisy realization)
moves closer to G4BL's true reference trace once multiple Coulomb scattering
is included. A single stochastic realization can't be validated pointwise
against a single deterministic reference -- real scattering noise makes any
one trajectory diverge from any other specific one regardless of whether the
underlying systematic physics improved.

Usage:
    HFOFO_N_ENSEMBLE=4 HFOFO_OUT=artifacts/ensemble.csv \
        uv run python scripts/track_ensemble.py 31
"""

from __future__ import annotations

import os
import sys
import time

import beamline.jax  # noqa: F401
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.kinematics import MuonStateDz
from hfofo.background import cavity_window_positions, rfc0_interior_centers, track_with_drag
from hfofo.build import build_channel_batched, build_wedges
from hfofo.load import load_lattice
from hfofo.union_material import build_union_material

DATA = "data/hfofo.yaml"
OUT = os.environ.get("HFOFO_OUT", "artifacts/full_channel_ensemble.csv")
SEED = int(os.environ.get("HFOFO_SEED", "0"))
N_ENSEMBLE = int(os.environ.get("HFOFO_N_ENSEMBLE", "4"))
BEAM_START = -700.0 * u.mm
REF_MOMENTUM = 247.5 * u.MeV
MUON_MASS = 105.6583715 * u.MeV
DZ = 15.0 * u.mm
CHUNK_PERIODS = 1


def make_initial_state(n: int) -> MuonStateDz:
    def _make_one(z, pz):
        return MuonStateDz.make(
            position=Cartesian4.make(z=z),
            momentum=Cartesian3.make(z=pz),
            q=1,
        )

    return jax.vmap(_make_one, in_axes=(0, 0))(
        jnp.full((n,), BEAM_START), jnp.full((n,), REF_MOMENTUM)
    )


def main() -> None:
    n_periods = int(sys.argv[1]) if len(sys.argv) > 1 else 12

    lattice = load_lattice(DATA)
    period = lattice.meta.period
    channel = build_channel_batched(lattice)
    wedges = build_union_material(build_wedges(lattice))
    window_z, window_thick = cavity_window_positions(lattice.cavities)
    rfc0_centers = rfc0_interior_centers(lattice.cavities)

    z_end = n_periods * period + BEAM_START

    os.makedirs("artifacts", exist_ok=True)
    resume = os.path.exists(OUT) and len(open(OUT).readlines()) > 1
    if resume:
        lines = open(OUT).readlines()
        last_rows = lines[-N_ENSEMBLE:]
        muon, z_last, x_last, y_last, ct_last, px_last, py_last, pz_last, E_last = (
            np.array(vals, dtype=float)
            for vals in zip(*(row.strip().split(",") for row in last_rows), strict=True)
        )
        print(f"resuming from z={z_last[0]:.1f}mm ({(len(lines) - 1) // N_ENSEMBLE} steps present)")
        state = MuonStateDz(
            kin=Tangent(
                p=Cartesian4.make(
                    x=x_last * u.mm, y=y_last * u.mm, z=z_last * u.mm, ct=ct_last * u.mm
                ),
                t=Cartesian4.make(
                    x=px_last * u.MeV, y=py_last * u.MeV, z=pz_last * u.MeV, ct=E_last * u.MeV
                ),
            ),
            q=1,
        )
        z0 = float(z_last[0]) * u.mm
    else:
        with open(OUT, "w") as f:
            f.write("muon,z_mm,x_mm,y_mm,ct_mm,px_MeV,py_MeV,pz_MeV,E_MeV\n")
            e0 = (REF_MOMENTUM**2 + MUON_MASS**2) ** 0.5
            for m in range(N_ENSEMBLE):
                f.write(f"{m},{BEAM_START/u.mm:.1f},0.0,0.0,0.0,0.0,0.0,{REF_MOMENTUM/u.MeV},{e0/u.MeV:.6f}\n")
        state = make_initial_state(N_ENSEMBLE)
        z0 = BEAM_START

    def batched(state, z0, z1, n, keys):
        def one(s, k):
            return track_with_drag(
                channel,
                s,
                z0,
                z1,
                dz=DZ,
                include_presswall=True,
                wedges=wedges,
                window_z=window_z,
                window_thick=window_thick,
                rfc0_centers=rfc0_centers,
                n_steps=n,
                key=k,
                rtol=1e-3,
                atol=1e-5,
            )

        return jax.vmap(one, in_axes=(0, 0))(state, keys)

    run = jax.jit(batched, static_argnums=(3,))

    t0 = time.time()
    i = 0
    base_key = jr.key(SEED)
    while z0 < z_end - 1e-6:
        z1 = min(z0 + CHUNK_PERIODS * period, z_end)
        n = int(round((z1 - z0) / DZ))
        keys = jr.split(jr.fold_in(base_key, i), N_ENSEMBLE)
        state, outs = run(state, z0, z1, n, keys)
        jax.block_until_ready(outs[0])
        zc, x, y, ct, px, py, pz, E = (np.asarray(o) for o in outs)  # each (N_ENSEMBLE, n)
        with open(OUT, "a") as f:
            for step in range(n):
                for m in range(N_ENSEMBLE):
                    row = (
                        zc[m, step], x[m, step], y[m, step], ct[m, step],
                        px[m, step], py[m, step], pz[m, step], E[m, step],
                    )
                    scales = (u.mm, u.mm, u.mm, u.mm, u.MeV, u.MeV, u.MeV, u.MeV)
                    f.write(f"{m}," + ",".join(f"{v/s:.6f}" for v, s in zip(row, scales)) + "\n")
        z0 = z1
        i += 1
        print(f"period {i}: z -> {z1 / u.mm:.0f}  ({time.time() - t0:.1f}s)", flush=True)
    print(f"done -- wrote {OUT}")


if __name__ == "__main__":
    main()
