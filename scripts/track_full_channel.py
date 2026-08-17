#!/usr/bin/env python3
"""Track the reference muon through the full HFOFO channel with the material
physics from hfofo.background folded in: GH2 world-material background drag
(correctly excluded from the 5 RFC0 cavities' interiors, which -- unlike
every other cavity variant -- don't set cavityMaterial=GH2 in the deck), the
presswall, the per-cavity Be windows, and the LiH wedge absorbers.

Validated against criggall/muon-cooling's true single-particle G4BL reference
trace (hfofo-latest/g4bl-output/ReferenceParticle_247pt5.txt, EventID=-1) --
NOT the wide/lossy 100-particle ensemble mean in hfofo-frozen/g4bl-output,
which is not a fair single-particle comparison target (its energy std is
comparable to its mean). Over the full 31 periods this reproduces the true
reference particle's energy oscillation and transverse helical envelope
closely (RMS energy diff ~6 MeV against a ~250 MeV typical scale); see the
project history for what's been ruled in/out chasing the residual.

Supports resuming from a partial run (reads the last row of the output CSV
and reconstructs state) since the full 31-period run needs several
invocations to complete within typical tool/session time limits.

Usage:
    uv run python scripts/track_full_channel.py            # 12 periods
    uv run python scripts/track_full_channel.py 31          # full channel
    uv run python scripts/track_full_channel.py 31          # (resumes automatically)
"""

from __future__ import annotations

import os
import sys
import time

import beamline.jax  # noqa: F401  (enables jax_enable_x64)
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.kinematics import MuonStateDz
from hfofo.background import cavity_window_positions, rfc0_interior_centers, track_with_drag
from hfofo.build import build_channel_batched, build_wedges
from hfofo.load import load_lattice
from hfofo.union_material import build_union_material

DATA = "data/hfofo.yaml"
OUT = "artifacts/full_channel_trajectory.csv"
BEAM_START = -700.0 * u.mm
REF_MOMENTUM = 247.5 * u.MeV
MUON_MASS = 105.6583715 * u.MeV

# Small step size for the field-only diffrax mini-step + analytic-loss loop
# (see track_with_drag's docstring for why this isn't done via
# stochastic_solve). 15mm balances performance against needing finer
# resolution than 25mm gave in the channel's low-energy troughs, where the
# dynamics get locally stiffer and diffrax's per-step max_steps was
# otherwise exceeded.
DZ = 15.0 * u.mm
CHUNK_PERIODS = 1  # periods tracked per resumable checkpoint


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
        last = lines[-1].strip().split(",")
        z_last, x_last, y_last, ct_last, px_last, py_last, pz_last, E_last = map(
            float, last
        )
        print(f"resuming from z={z_last:.1f}mm ({len(lines) - 1} rows already present)")
        state = MuonStateDz(
            kin=Tangent(
                p=Cartesian4.make(
                    x=x_last * u.mm, y=y_last * u.mm, z=z_last * u.mm, ct=ct_last * u.mm
                ),
                t=Cartesian4.make(
                    x=px_last * u.MeV, y=py_last * u.MeV, z=pz_last * u.MeV, ct=E_last * u.MeV
                ),
            ),
            q=jnp.array(1),
        )
        z0 = z_last * u.mm
    else:
        with open(OUT, "w") as f:
            f.write("z_mm,x_mm,y_mm,ct_mm,px_MeV,py_MeV,pz_MeV,E_MeV\n")
            e0 = (REF_MOMENTUM**2 + MUON_MASS**2) ** 0.5
            f.write(f"{BEAM_START/u.mm:.1f},0.0,0.0,0.0,0.0,0.0,{REF_MOMENTUM/u.MeV},{e0/u.MeV:.6f}\n")
        state = MuonStateDz.make(
            position=Cartesian4.make(z=BEAM_START),
            momentum=Cartesian3.make(z=REF_MOMENTUM),
            q=1,
        )
        z0 = BEAM_START

    run = jax.jit(
        lambda s, z0, z1, n: track_with_drag(
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
            rtol=1e-3,
            atol=1e-5,
        ),
        static_argnums=(3,),
    )

    t0 = time.time()
    i = 0
    while z0 < z_end - 1e-6:
        z1 = min(z0 + CHUNK_PERIODS * period, z_end)
        n = int(round((z1 - z0) / DZ))
        state, outs = run(state, z0, z1, n)
        jax.block_until_ready(outs[0])
        zc, x, y, ct, px, py, pz, E = (np.asarray(o) for o in outs)
        with open(OUT, "a") as f:
            for row in zip(zc, x, y, ct, px, py, pz, E):
                f.write(
                    ",".join(
                        f"{v / scale:.6f}"
                        for v, scale in zip(
                            row, (u.mm, u.mm, u.mm, u.mm, u.MeV, u.MeV, u.MeV, u.MeV)
                        )
                    )
                    + "\n"
                )
        z0 = z1
        i += 1
        print(f"period {i}: z -> {z1 / u.mm:.0f}  ({time.time() - t0:.1f}s)", flush=True)
    print(f"done -- wrote {OUT}")


if __name__ == "__main__":
    main()
