#!/usr/bin/env python3
"""Track the reference muon through the HFOFO channel.

Milestone A deliverable (single-particle version): propagate the deck's
reference muon (mu+, 247.5 MeV/c, on axis, from z=beamstart=-700mm) through the
solenoid+RF channel with the boundary-aware solver, and save the trajectory plus
diagnostic plots.

By-product for milestone B: the arrival-time profile ct(z) at each cavity is the
reference timing the RF phases must be set against.

Usage:
    uv run python scripts/track_reference.py                 # first few periods
    uv run python scripts/track_reference.py --periods 31    # full channel
    uv run python scripts/track_reference.py --z-end 20000   # explicit end [mm]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import beamline.jax  # noqa: F401  (enables jax_enable_x64)
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.integrate.propagate import diffrax_solve
from beamline.jax.kinematics import MuonStateDz
from hfofo.build import build_channel_batched
from hfofo.load import load_lattice

DATA = Path(__file__).parent.parent / "data" / "hfofo.yaml"
ARTIFACTS = Path(__file__).parent.parent / "artifacts"

# Deck reference-muon parameters (track_v7.in).
REF_MOMENTUM = 247.5 * u.MeV
BEAM_START = -700.0 * u.mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--periods",
        type=int,
        default=2,
        help="number of HFOFO periods to track (default 2; full channel is 31)",
    )
    ap.add_argument(
        "--z-end", type=float, default=None, help="explicit end z [mm] (overrides --periods)"
    )
    ap.add_argument("--n-save", type=int, default=400, help="number of save points")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-7)
    args = ap.parse_args()

    lattice = load_lattice(DATA)
    period = lattice.meta.period

    z_end = args.z_end if args.z_end is not None else BEAM_START + args.periods * period
    zs = jnp.linspace(BEAM_START, z_end, args.n_save)
    print(
        f"tracking reference muon: z = {BEAM_START:.0f} -> {z_end:.0f} mm "
        f"({args.periods} periods)" if args.z_end is None else
        f"tracking reference muon: z = {BEAM_START:.0f} -> {z_end:.0f} mm"
    )

    channel = build_channel_batched(lattice)

    def run() -> MuonStateDz:
        start = MuonStateDz.make(
            position=Cartesian4.make(z=BEAM_START),
            momentum=Cartesian3.make(z=REF_MOMENTUM),
            q=1,
        )
        sol, stats = diffrax_solve(
            channel, start, zs, forward_mode=True, rtol=args.rtol, atol=args.atol
        )
        return sol, stats

    t0 = time.time()
    track, stats = jax.jit(run)()
    jax.block_until_ready(track.kin.p.coords)
    print(f"tracked in {time.time() - t0:.1f}s; solver steps: {stats.get('num_steps')}")

    _save(track, zs, ARTIFACTS)


def _save(track: MuonStateDz, zs, outdir: Path) -> None:
    outdir.mkdir(exist_ok=True)
    z = np.asarray(track.kin.p.z) / u.mm
    x = np.asarray(track.kin.p.x) / u.mm
    y = np.asarray(track.kin.p.y) / u.mm
    ct = np.asarray(track.kin.p.ct) / u.mm
    px = np.asarray(track.kin.t.x) / u.MeV
    py = np.asarray(track.kin.t.y) / u.MeV
    pz = np.asarray(track.kin.t.z) / u.MeV
    E = np.asarray(track.kin.t.ct) / u.MeV

    # CSV
    csv = outdir / "reference_trajectory.csv"
    with csv.open("w") as f:
        f.write("z_mm,x_mm,y_mm,ct_mm,px_MeV,py_MeV,pz_MeV,E_MeV\n")
        for row in zip(z, x, y, ct, px, py, pz, E, strict=True):
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"wrote {csv}")

    # Plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    axs[0, 0].plot(z, x, label="x")
    axs[0, 0].plot(z, y, label="y")
    axs[0, 0].set_xlabel("z [mm]")
    axs[0, 0].set_ylabel("transverse [mm]")
    axs[0, 0].legend()
    axs[0, 0].set_title("Transverse position")

    axs[0, 1].plot(z, E)
    axs[0, 1].set_xlabel("z [mm]")
    axs[0, 1].set_ylabel("E [MeV]")
    axs[0, 1].set_title("Energy vs z (RF acceleration)")

    axs[1, 0].plot(x, y)
    axs[1, 0].set_xlabel("x [mm]")
    axs[1, 0].set_ylabel("y [mm]")
    axs[1, 0].set_aspect("equal")
    axs[1, 0].set_title("Transverse trajectory (helix)")

    axs[1, 1].plot(z, pz, label="pz")
    axs[1, 1].plot(z, np.hypot(px, py), label="p_transverse")
    axs[1, 1].set_xlabel("z [mm]")
    axs[1, 1].set_ylabel("momentum [MeV]")
    axs[1, 1].legend()
    axs[1, 1].set_title("Momentum components")

    fig.tight_layout()
    png = outdir / "reference_trajectory.png"
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
