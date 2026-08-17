#!/usr/bin/env python3
"""Track the reference muon through field *and* wedge absorbers (milestone C).

Extends ``track_reference.py`` (solenoids + RF only) by adding the 171 LiH
wedge absorbers as a ``UnionMaterial`` and switching from ``diffrax_solve`` to
``stochastic_solve``, which applies a stochastic energy-loss kick each time the
particle traverses material. The deliverable (per the plan doc, milestone C) is
the energy **sawtooth**: loss in each wedge, restored by the RF that follows it.

Usage:
    uv run python scripts/track_stochastic.py                 # 1 period, 1 seed
    uv run python scripts/track_stochastic.py --periods 31 --n-ensemble 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import beamline.jax  # noqa: F401  (enables jax_enable_x64)
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from beamline.jax.absorber.straggling import landau_energy_loss_sampler
from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.integrate.stochastic import stochastic_solve
from beamline.jax.kinematics import MuonStateDz
from hfofo.build import build_channel_batched, build_wedges
from hfofo.load import load_lattice
from hfofo.union_material import build_union_material

DATA = Path(__file__).parent.parent / "data" / "hfofo.yaml"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# Deck reference-muon parameters (track_v7.in) -- same as track_reference.py.
REF_MOMENTUM = 247.5 * u.MeV
BEAM_START = -700.0 * u.mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--periods",
        type=int,
        default=1,
        help="number of HFOFO periods to track (default 1; full channel is 31)",
    )
    ap.add_argument(
        "--z-end", type=float, default=None, help="explicit end z [mm] (overrides --periods)"
    )
    ap.add_argument(
        "--n-save",
        type=int,
        default=200,
        help="number of save points (keep intervals well under a wedge's z-extent "
        "so material segmentation has room inside each interval)",
    )
    ap.add_argument(
        "--n-ensemble", type=int, default=1, help="number of PRNG-keyed muons to run"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-7)
    ap.add_argument("--max-substeps", type=int, default=256)
    args = ap.parse_args()

    lattice = load_lattice(DATA)
    period = lattice.meta.period

    z_end = args.z_end if args.z_end is not None else BEAM_START + args.periods * period
    zs = jnp.linspace(BEAM_START, z_end, args.n_save)
    print(
        f"tracking {args.n_ensemble} muon(s) through field + {len(lattice.wedges)} "
        f"wedges: z = {BEAM_START:.0f} -> {z_end:.0f} mm"
    )

    field = build_channel_batched(lattice)
    material = build_union_material(build_wedges(lattice))

    start = MuonStateDz.make(
        position=Cartesian4.make(z=BEAM_START),
        momentum=Cartesian3.make(z=REF_MOMENTUM),
        q=1,
    )

    def run(key):
        return stochastic_solve(
            field,
            material,
            start,
            zs,
            key,
            sampler=landau_energy_loss_sampler,
            rtol=args.rtol,
            atol=args.atol,
            max_substeps=args.max_substeps,
        )

    keys = jr.split(jr.key(args.seed), args.n_ensemble)
    t0 = time.time()
    tracks, stats = jax.jit(jax.vmap(run))(keys)
    jax.block_until_ready(tracks.kin.p.coords)
    dt = time.time() - t0
    print(
        f"tracked {args.n_ensemble} muon(s) in {dt:.1f}s; "
        f"mean accepted steps/interval-loop: "
        f"{float(jnp.mean(stats['num_accepted_steps'])):.0f}, "
        f"mean rejected: {float(jnp.mean(stats['num_rejected_steps'])):.0f}"
    )

    _save(tracks, zs, ARTIFACTS)


def _save(tracks: MuonStateDz, zs, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    n_ensemble = tracks.kin.p.z.shape[0]
    z = np.asarray(tracks.kin.p.z) / u.mm  # (n_ensemble, n_save)
    E = np.asarray(tracks.kin.t.ct) / u.MeV

    csv = outdir / "stochastic_trajectory.csv"
    with csv.open("w") as f:
        f.write("muon,z_mm,E_MeV\n")
        for i in range(n_ensemble):
            for zi, Ei in zip(z[i], E[i], strict=True):
                f.write(f"{i},{zi:.6f},{Ei:.6f}\n")
    print(f"wrote {csv}")

    e0 = E[:, 0]
    ef = E[:, -1]
    print(f"energy: start {e0.mean():.3f} MeV -> end {ef.mean():.3f} MeV "
          f"(mean net {ef.mean() - e0.mean():+.3f} MeV, std {ef.std():.3f} MeV)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_ensemble):
        ax.plot(z[i], E[i], alpha=0.6, lw=1.2 if n_ensemble == 1 else 0.7)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("E [MeV]")
    ax.set_title("Energy vs z: sawtooth from wedge loss + RF reacceleration")
    fig.tight_layout()
    png = outdir / "stochastic_trajectory.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
