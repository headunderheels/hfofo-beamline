#!/usr/bin/env python3
"""Milestone D sandbox: track a real 24-particle sample from initial.dat
through the HFOFO channel and compare INPUT vs OUTPUT eigen-emittances.

Loads initial.dat, applies a |pz-247.5|<75 MeV/c cut (raw capture beam has a
huge pz tail -- 26.5 to 3781.7 MeV/c -- that would hit apertures/kill-volumes
in the real G4BL simulation that this simplified channel doesn't model; the
cut avoids feeding those outliers into a channel with no way to remove them),
draws a reproducible 24-particle sample, and computes the INPUT eigen-
emittances (reproduces docs/SESSION_HANDOFF numbers exactly: eps1=2929.06,
eps2=556.77, product=1,630,806 mm*MeV/c vs naive product 3,196,377 --
confirms eigen-emittances matter here by a factor of ~2x, not just in
principle).

initial.dat lives in the criggall/muon-cooling checkout, not this repo --
set HFOFO_MUON_COOLING to its path (exact file or a directory to search
under) if it isn't found automatically; see hfofo.reference_data.find_file.

--track: ensemble-tracks all 24 through 1 period with an aperture cut
(APERTURE_RADIUS, 200mm -- see track_with_drag's aperture_radius docstring)
and computes OUTPUT eigen-emittances from the survivors. This was blocked
for a while by a misdiagnosed "compile cost scales with N" issue -- see
docs/SESSION_HANDOFF_2026-08-17_aperture_cut.md for the full story; it's
resolved now (compile time is flat ~15-18s regardless of N).

--bisect: times compile/run separately across N=4/8/16/20/24, if the
N-scaling question ever needs re-checking.

Usage:
    uv run python scripts/emittance_sandbox.py           # input emittances only
    uv run python scripts/emittance_sandbox.py --track    # + ensemble tracking
    HFOFO_MUON_COOLING=/path/to/muon-cooling uv run python scripts/emittance_sandbox.py --track
"""

from __future__ import annotations

import argparse
import time

import beamline.jax  # noqa: F401
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.kinematics import MuonStateDz
from hfofo.background import (
    cavity_window_positions_windowed,
    rfc0_interior_centers,
    track_with_drag,
)
from hfofo.build import build_channel_batched_windowed, build_wedges_windowed
from hfofo.emittance import covariance4, eigen_emittances, naive_projected_emittances
from hfofo.load import load_lattice
from hfofo.reference_data import find_file
from hfofo.union_material import build_union_material

DATA = "data/hfofo.yaml"
BEAM_START = -700.0 * u.mm
PZ_CUT_MEV = 75.0
PZ_CENTER_MEV = 247.5
SAMPLE_SIZE = 24
SAMPLE_SEED = 0
DZ = 15.0 * u.mm  # REVERTED from 60mm -- see optimize_taper.py's d73df4b for why: a real,
           # reproduced failure ("max_steps was reached") at multi-period,
           # N>=24 scale that the original 60mm retuning was never tested
           # against (only verified for one particle over one period). This
           # script had not been re-tested at that scale either -- reverted
           # here as a precaution, not because THIS script was independently
           # confirmed to fail, matching the same reasoning.

# Tightest iris radius anywhere in the deck (RFC2's irisRadius=200mm; RFC0/RFC
# use 300mm, RFC1 uses 250mm) -- a simple, conservative, physically-motivated
# global aperture cut. This model has no z-dependent per-element aperture map
# (a real refinement would track which element a particle is actually near),
# so a uniform cut at the tightest value errs toward excluding a few more
# particles than the true per-element apertures would in the wider-iris
# regions, rather than under-cutting and letting a lost particle silently
# corrupt ensemble statistics. See track_with_drag's aperture_radius docstring
# for the full story (a genuinely lost/unstable particle -- confirmed by
# direct trajectory inspection, radius doubling every ~150-200mm -- otherwise
# either crashes the solver or, if tolerance is loosened to avoid the crash,
# silently produces a 100+ meter nonphysical excursion that corrupts any
# ensemble statistic computed from it).
APERTURE_RADIUS = 200.0 * u.mm


def load_sample(size: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> np.ndarray:
    """Reproducible sample of ``size`` mu+ rows from initial.dat, after the
    pz cut. Returns the raw (N, 12) rows (see initial.dat's column format in
    the docstring above / the handoff doc) -- caller picks out the columns
    it needs.
    """
    d = np.genfromtxt(find_file("initial.dat"), comments="#")
    mu = d[d[:, 7] == -13]  # PDGid == -13 is mu+
    pz = mu[:, 5]
    cut = np.abs(pz - PZ_CENTER_MEV) < PZ_CUT_MEV
    mu_cut = mu[cut]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(mu_cut), size=size, replace=False)
    return mu_cut[idx]


def input_eigen_emittances(sample: np.ndarray) -> None:
    sx, sy, spx, spy = sample[:, 0], sample[:, 1], sample[:, 3], sample[:, 4]
    sigma4 = covariance4(jnp.array(sx), jnp.array(spx), jnp.array(sy), jnp.array(spy))
    e1, e2 = eigen_emittances(sigma4)
    nx, ny = naive_projected_emittances(sigma4)
    print(f"INPUT eigen-emittances:  eps1={float(e1):.2f}  eps2={float(e2):.2f}"
          f"  product={float(e1*e2):.2f}")
    print(f"INPUT naive emittances:  eps_x={float(nx):.2f}  eps_y={float(ny):.2f}"
          f"  product={float(nx*ny):.2f}  (ratio {float(nx*ny)/float(e1*e2):.3f}x)")


def make_ensemble_state(sample: np.ndarray) -> MuonStateDz:
    """Build the vmapped initial MuonStateDz for the sampled ensemble.
    Transverse position/momentum come from the sample; ct is centered
    (mean subtracted) to match the reference-particle convention (mean
    ct=0 at BEAM_START), since G4BL's beamZ directive relocates the whole
    snapshot to the tracking start and only relative timing within the
    bunch is physically meaningful here.
    """
    sx, sy = sample[:, 0], sample[:, 1]
    spx, spy, spz = sample[:, 3], sample[:, 4], sample[:, 5]
    st = sample[:, 6]
    ct0 = st - st.mean()

    def make_one(x, y, px, py, pz, ct):
        return MuonStateDz.make(
            position=Cartesian4.make(x=x * u.mm, y=y * u.mm, z=BEAM_START, ct=ct * u.c_light),
            momentum=Cartesian3.make(x=px * u.MeV, y=py * u.MeV, z=pz * u.MeV),
            q=1,
        )

    return jax.vmap(make_one)(
        jnp.array(sx), jnp.array(sy), jnp.array(spx), jnp.array(spy),
        jnp.array(spz), jnp.array(ct0),
    )


def track_ensemble(n: int, n_periods: int = 1, rtol: float = 1e-3, atol: float = 1e-5):
    """Track an N-particle ensemble, with an aperture cut (APERTURE_RADIUS)
    so a genuinely lost/unstable particle is frozen at the aperture rather
    than either crashing the solver or (if tolerance were loosened instead)
    silently corrupting the ensemble with a nonphysical runaway trajectory.
    See track_with_drag's aperture_radius docstring for the full story --
    this was root-caused by testing every particle in a real 24-particle
    initial.dat sample individually: exactly one (of 24) was genuinely
    unstable (transverse radius doubling every ~150-200mm from an entirely
    unremarkable starting point), not a tolerance/resolution artifact
    (confirmed: making dz FINER made the crash worse, not better; only
    loosening tolerance masked it, and only by letting the trajectory run
    away to 100+ meters off-axis instead of erroring out).

    Uses the SAFE (tight) default tolerance -- rtol=1e-2 was tried first to
    work around the crash, but that's the wrong fix (see above); with the
    aperture cut in place the safe tolerance works for every particle.
    """
    sample = load_sample(size=n)
    state0 = make_ensemble_state(sample)

    lattice = load_lattice(DATA)
    period = lattice.meta.period
    z0 = BEAM_START
    z1 = z0 + n_periods * period
    n_steps = int(round(float((z1 - z0) / DZ)))

    z_center = float((z0 + z1) / 2)
    channel = build_channel_batched_windowed(lattice, z_center=z_center)
    wedges = build_union_material(build_wedges_windowed(lattice, z_center=z_center))
    window_z, window_thick = cavity_window_positions_windowed(lattice.cavities, z_center=z_center)
    rfc0_centers = rfc0_interior_centers(lattice.cavities)

    def track_one(state):
        final_state, outs = track_with_drag(
            channel, state, z0, z1, dz=DZ, include_presswall=True, wedges=wedges,
            window_z=window_z, window_thick=window_thick, rfc0_centers=rfc0_centers,
            n_steps=n_steps, key=None, rtol=rtol, atol=atol,
            aperture_radius=APERTURE_RADIUS,
        )
        return final_state

    track_vmapped = jax.jit(jax.vmap(track_one))

    print(f"N={n}, {n_periods} period(s), {n_steps} steps/particle, rtol={rtol}, "
          f"aperture={APERTURE_RADIUS/u.mm:.0f}mm: ", end="", flush=True)
    t0 = time.time()
    compiled = track_vmapped.lower(state0).compile()
    t_compile = time.time() - t0
    t0 = time.time()
    final_state = compiled(state0)
    jax.block_until_ready(final_state.kin.p.coords)
    t_run = time.time() - t0
    print(f"compile={t_compile:.1f}s  run={t_run:.1f}s  total={t_compile+t_run:.1f}s")

    r = jnp.hypot(final_state.kin.p.x, final_state.kin.p.y)
    n_lost = int(jnp.sum(r >= APERTURE_RADIUS - 1e-3 * u.mm))
    print(f"particles at/beyond aperture (treated as lost): {n_lost}/{n}")
    return final_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", action="store_true", help="also attempt ensemble tracking")
    ap.add_argument("--bisect", action="store_true", help="bisect N to find the compile-cost knee")
    args = ap.parse_args()

    sample = load_sample()
    print(f"loaded {SAMPLE_SIZE}-particle sample (seed={SAMPLE_SEED}) from initial.dat")
    input_eigen_emittances(sample)

    if args.bisect:
        for n in [4, 8, 12, 16, 20, 24]:
            track_ensemble(n, n_periods=1)
        return

    if args.track:
        final_state = track_ensemble(SAMPLE_SIZE, n_periods=1)
        p, t = final_state.kin.p, final_state.kin.t
        r = jnp.hypot(p.x, p.y)
        survived = r < APERTURE_RADIUS - 1e-3 * u.mm
        n_survived = int(jnp.sum(survived))
        print(f"\n{n_survived}/{SAMPLE_SIZE} particles survived to the aperture; "
              f"OUTPUT emittances computed from survivors only "
              f"(a frozen/lost particle's phase-space point isn't a meaningful "
              f"sample of the surviving beam).")
        # jnp.cov can't take a boolean mask directly under jit in a way that
        # changes array size -- select survivors as a concrete numpy op
        # instead (this whole path runs post-jit, on already-computed
        # concrete arrays, so this is fine).
        import numpy as np
        surv_np = np.asarray(survived)
        xs = np.asarray(p.x)[surv_np]
        pxs = np.asarray(t.x)[surv_np]
        ys = np.asarray(p.y)[surv_np]
        pys = np.asarray(t.y)[surv_np]
        sigma4 = covariance4(jnp.array(xs), jnp.array(pxs), jnp.array(ys), jnp.array(pys))
        e1, e2 = eigen_emittances(sigma4)
        nx, ny = naive_projected_emittances(sigma4)
        print(f"\nOUTPUT eigen-emittances: eps1={float(e1):.2f}  eps2={float(e2):.2f}"
              f"  product={float(e1*e2):.2f}")
        print(f"OUTPUT naive emittances: eps_x={float(nx):.2f}  eps_y={float(ny):.2f}"
              f"  product={float(nx*ny):.2f}")


if __name__ == "__main__":
    main()
