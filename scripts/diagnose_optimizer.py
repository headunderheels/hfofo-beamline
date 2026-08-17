#!/usr/bin/env python3
"""Diagnose what's actually driving optimize_lattice.py's observed merit
reduction: genuine phase-space tightening of the surviving beam, or the
optimizer gaming the aperture cut by simply changing which particles
survive (fewer/different survivors can lower an eigen-emittance product
without the channel doing any better physics).

Compares nominal vs. final (optimized) parameters from a completed
optimize_lattice.py checkpoint:
1. Survivor count at each (does the aperture cut let through a different
   NUMBER of particles?).
2. The actual (live) merit at each -- what optimize_lattice.py reports.
3. A FIXED-SURVIVOR-SET merit at the final params: apply the NOMINAL run's
   survivor mask (not the final run's own aperture weights) to the final
   params' resulting phase space. This isolates genuine tightening (how
   tight does the SAME fixed population of particles get under the
   improved fields) from selection effects (the live merit dropping partly
   or mostly because different/fewer particles are being counted at all).

If the fixed-survivor-set merit drops by roughly the same amount as the
live merit, the improvement is real. If the fixed-survivor-set merit barely
moves (or moves much less) while the live merit drops a lot, the optimizer
is mostly exploiting the aperture cut, not improving the channel.

IMPORTANT: at small N (e.g. the N=6 default), the aperture cut may simply
never trigger at all for either parameter set -- confirmed directly (N=6,
this project's initial check): 6/6 survive at both nominal and optimized
params, so that comparison couldn't have detected aperture-gaming even if
it existed, because the aperture was never actually in play. The real test
needs an N where the aperture cut is live (we know N=24 loses 8/24 = 33%
at nominal params) -- but a full jacfwd optimizer run at N=24 costs enough
wall-clock time per step (compile + K_PARAMS-times-longer tangent
propagation than a single-parameter run) that it doesn't fit a sandboxed
tool-call budget -- run it directly in your own environment instead, no
280s-per-call constraint there:

    uv run python scripts/optimize_lattice.py --n-ensemble 24 --steps 4 \\
        --skip-jacfwd-check --checkpoint artifacts/optimize_checkpoint_n24.txt
    # repeat the above (resumes automatically) for more steps if desired
    uv run python scripts/diagnose_optimizer.py --n-ensemble 24 \\
        --checkpoint artifacts/optimize_checkpoint_n24.txt

Usage:
    uv run python scripts/diagnose_optimizer.py [--n-ensemble N] [--checkpoint PATH]
"""

from __future__ import annotations

import argparse

import beamline.jax  # noqa: F401
import hepunits as u
import jax.numpy as jnp
import numpy as np

from emittance_sandbox import APERTURE_RADIUS
from hfofo.emittance import eigen_emittances, weighted_covariance4
from optimize_lattice import N_ENSEMBLE, build_pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ensemble", type=int, default=N_ENSEMBLE)
    ap.add_argument("--checkpoint", default="artifacts/optimize_checkpoint.txt",
                     help="a completed (or partial) optimize_lattice.py checkpoint; "
                          "the LAST row's params are used as the 'final' comparison point")
    args = ap.parse_args()

    checkpoint = np.loadtxt(args.checkpoint)
    if checkpoint.ndim == 1:
        checkpoint = checkpoint[None, :]
    k_params = checkpoint.shape[1] - 1  # last column is the recorded merit
    final_params = jnp.array(checkpoint[-1, :k_params])
    print(f"loaded {len(checkpoint)} step(s) from {args.checkpoint}; "
          f"using the last row as 'final' params: {final_params}")

    merit_fn, nominal_params, track_fn = build_pipeline(
        n_ensemble=args.n_ensemble, return_track_fn=True
    )
    print(f"nominal params: {nominal_params}")

    print("\ntracking at NOMINAL params...")
    final_nom = track_fn(nominal_params)
    p_nom, t_nom = final_nom.kin.p, final_nom.kin.t
    r_nom = jnp.hypot(p_nom.x, p_nom.y)
    weights_nom = jnp.where(r_nom >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
    n_survived_nom = int(jnp.sum(weights_nom))
    cov_nom = weighted_covariance4(p_nom.x, t_nom.x, p_nom.y, t_nom.y, weights_nom)
    e1_nom, e2_nom = eigen_emittances(cov_nom)
    merit_nom = float(e1_nom * e2_nom)
    print(f"  survivors: {n_survived_nom}/{args.n_ensemble}   live merit: {merit_nom:.4f}")
    print(f"  per-particle r [mm]: {np.asarray(r_nom)/u.mm}")

    print("\ntracking at FINAL (optimized) params...")
    final_opt = track_fn(final_params)
    p_opt, t_opt = final_opt.kin.p, final_opt.kin.t
    r_opt = jnp.hypot(p_opt.x, p_opt.y)
    weights_opt = jnp.where(r_opt >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
    n_survived_opt = int(jnp.sum(weights_opt))
    cov_opt = weighted_covariance4(p_opt.x, t_opt.x, p_opt.y, t_opt.y, weights_opt)
    e1_opt, e2_opt = eigen_emittances(cov_opt)
    merit_opt_live = float(e1_opt * e2_opt)
    print(f"  survivors: {n_survived_opt}/{args.n_ensemble}   live merit: {merit_opt_live:.4f}")
    print(f"  per-particle r [mm]: {np.asarray(r_opt)/u.mm}")

    # Fixed-survivor-set merit: apply the NOMINAL run's mask to the FINAL
    # params' phase space -- same particles counted both times, isolating
    # genuine tightening from selection effects.
    cov_fixed = weighted_covariance4(p_opt.x, t_opt.x, p_opt.y, t_opt.y, weights_nom)
    e1_fixed, e2_fixed = eigen_emittances(cov_fixed)
    merit_opt_fixed = float(e1_fixed * e2_fixed)

    print(f"\n{'='*60}")
    print(f"nominal:                          survivors={n_survived_nom}  merit={merit_nom:.4f}")
    print(f"final, live aperture weights:     survivors={n_survived_opt}  merit={merit_opt_live:.4f}")
    print(f"final, NOMINAL's fixed survivors: (same {n_survived_nom})    merit={merit_opt_fixed:.4f}")
    print(f"{'='*60}")

    live_drop = merit_nom - merit_opt_live
    fixed_drop = merit_nom - merit_opt_fixed
    print(f"\nlive-merit drop (what the optimizer reported):  {live_drop:+.1f}  ({live_drop/merit_nom*100:.1f}%)")
    print(f"fixed-survivor-set drop (genuine tightening):   {fixed_drop:+.1f}  ({fixed_drop/merit_nom*100:.1f}%)")
    if n_survived_opt != n_survived_nom:
        print(f"\n*** survivor count CHANGED ({n_survived_nom} -> {n_survived_opt}) -- "
              f"selection effects are in play; compare the live-drop and fixed-drop "
              f"percentages above to see how much of the reduction is genuine vs. "
              f"selection ***")
    else:
        print(f"\nsurvivor count unchanged ({n_survived_nom}) -- selection effect ruled out "
              f"for this comparison, drop is genuine tightening. NOTE: if this ran at a "
              f"small N (see module docstring), an unchanged survivor count may just mean "
              f"the aperture was never close to triggering for either parameter set, not "
              f"that gaming is impossible in general -- rerun at a larger N (e.g. 24) if "
              f"so, where the aperture is confirmed live at nominal params.")


if __name__ == "__main__":
    main()

