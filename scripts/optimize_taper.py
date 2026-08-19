#!/usr/bin/env python3
"""Multi-period taper-parameterized optimization with the full 6D
eigen-emittance as the merit, extending optimize_lattice.py's single-period,
individual-current, 4D-transverse demo along three axes the user asked for:

1. TAPER PARAMETERIZATION instead of individual solenoid currents. The deck's
   own steady-state solenoid currents (periods 1-29; period 0 is an entrance
   matching section, period 29 onward is an exit matching section + solenoid
   -- both excluded, held fixed at nominal, same reasoning as the entrance)
   are a smooth, decaying function of period index, not independent numbers
   -- fit quality at increasing polynomial degree (least-squares against the
   real data): degree 1 (2 params) RMS 0.88 out of a 9.55 range (~9%),
   degree 2 (3 params) RMS 0.22 (~2.3%), degree 3 (4 params) RMS 0.058
   (~0.6%). Default degree=2: good fit, still low-dimensional, and this is
   exactly the kind of family reverse-mode AD is suited for -- one backward
   pass gives the gradient w.r.t. however many taper coefficients you choose,
   at no extra cost per coefficient (unlike forward-mode/jacfwd, which costs
   one pass per parameter).

2. FULL 6D EIGEN-EMITTANCE (hfofo.emittance.eigen_emittances_6d) instead of
   the 4D transverse-only version -- includes (ct, E) as the longitudinal
   canonical pair, so this merit can actually see the RF/absorber energy
   balance and any longitudinal-transverse coupling from the wedges'
   dispersive placement, which a 1-period 4D merit structurally cannot.

3. MULTI-PERIOD tracking, with the period count as a first-class, freely
   configurable parameter (--n-periods, default 5) rather than a hardcoded
   constant -- see build_pipeline's docstring for how this scales to the
   full 31-period channel without the windowing cost blowing up.

WORKFLOW NOTE: this script generates curated diagnostic plots
(artifacts/optimize_taper_diagnostics.png via --plot), not just numbers --
per project convention, visual inspection of exactly this kind of
per-period trajectory has driven real performance/design judgement calls
in prior iterations (this session's own rank-deficiency and aperture-loss
findings would have been visible immediately in the transmission/
eigen-emittance-vs-period panels rather than needing separate ad hoc
debugging). Any future script in this vein should produce inspectable
plots as a normal part of its output, not an afterthought bolted on after
the numbers already look right.

Usage:
    uv run python scripts/optimize_taper.py --n-periods 5
    uv run python scripts/optimize_taper.py --n-periods 31 --reverse-mode
    uv run python scripts/optimize_taper.py --n-periods 5 --taper-degree 3 --verify-consistency
    uv run python scripts/optimize_taper.py --n-periods 5 --steps 5 --plot
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import time

import beamline.jax  # noqa: F401
import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import optax

from hfofo.background import (
    cavity_window_positions_windowed,
    rfc0_interior_centers,
    track_with_drag,
)
from hfofo.build import (
    AMP_TO_JPHI,
    K_SOLENOIDS_LOCAL,
    _nearest,
    build_cavity,
    build_channel_batched_windowed,
    build_wedges_windowed,
)
from hfofo.emittance import eigen_emittances_6d, weighted_covariance6
from hfofo.load import load_lattice
from hfofo.stacked import BatchedChannel, StackedField
from hfofo.union_material import build_union_material
from emittance_sandbox import APERTURE_RADIUS, load_sample, make_ensemble_state

DATA = "data/hfofo.yaml"
BEAM_START = -700.0 * u.mm
DZ = 15.0 * u.mm  # REVERTED from 60mm after a real failure -- see below
# The performance patch that retuned DZ project-wide to 60mm verified safety
# against exactly one known-unstable particle over one period near the
# channel start (see hfofo.background.track_with_drag's docstring). It was
# NOT tested against a full multi-period, N=24+ ensemble -- and when
# actually run at --n-periods 5 --n-ensemble 24, it hit "the maximum number
# of solver steps was reached". Reverting to --dz 15 (the pre-retuning
# value) fixed it cleanly: the jacfwd-vs-finite-difference check completed
# with a normal 0.07% match, right at the point the dz=60mm run crashed.
# This is real, direct evidence -- not a theoretical concern -- that dz=60mm
# is unsafe at the ensemble/period scale this script is actually meant to be
# used at. Use --dz 60 explicitly if you've separately verified it's safe
# for your specific n_periods/n_ensemble; don't assume it from the
# single-particle verification alone. The other scripts in this project
# (track_full_channel.py, gradient_check.py, emittance_sandbox.py,
# optimize_lattice.py) still default to 60mm and have NOT been re-tested at
# a comparably large multi-period ensemble scale -- this same risk likely
# applies to them too if pushed to a similar scale, not yet confirmed either way.
N_ENSEMBLE = 6

# Steady-state (taper-eligible) region: excludes the entrance matching
# section (period 0) and the exit matching section + exit solenoid (period
# 29 onward -- confirmed by direct inspection of the real data: the smooth
# taper holds cleanly through period ~30's early solenoids then breaks
# sharply, dropping from ~85 to ~45 over the last couple of entries; period
# 29 is used as a clean, slightly conservative boundary rather than chasing
# the exact breakpoint). Both excluded regions are held fixed at their
# nominal (deck) values -- not part of the optimization at all.
TAPER_MIN_PERIOD = 1
TAPER_MAX_PERIOD = 29  # exclusive: periods [1, 29) are taper-eligible


def is_taper_eligible(z: float, period: float) -> bool:
    n = (z - BEAM_START) / period
    return TAPER_MIN_PERIOD <= n < TAPER_MAX_PERIOD


def taper_magnitude(n_norm, theta) -> jnp.ndarray:
    """Polynomial in n_norm (period index normalized by the FULL lattice's
    n_periods, NOT by however many periods are currently being tracked --
    this is what lets --n-periods change without changing what the taper
    parameters mean; extending from 5 periods to the full 31 just evaluates
    the same function further along its existing domain, no refit needed).
    Degree = len(theta) - 1.
    """
    powers = jnp.stack([n_norm**k for k in range(len(theta))])
    return jnp.dot(theta, powers)


def fit_nominal_taper(lattice, degree: int) -> jnp.ndarray:
    """Least-squares fit of the real steady-state solenoid magnitudes to a
    degree-``degree`` polynomial in normalized period index -- gives a
    genuinely deck-derived nominal theta (not a guess), consistent with this
    project's practice throughout of starting optimizations from the real
    design rather than an arbitrary point.
    """
    period = lattice.meta.period
    n_total = lattice.meta.n_periods
    steady = [
        s for s in lattice.solenoids if is_taper_eligible(s.z, period)
    ]
    zs = np.array([s.z for s in steady])
    mags = np.abs(np.array([s.current for s in steady]))
    n_norm = ((zs - BEAM_START) / period) / n_total
    A = np.stack([n_norm**k for k in range(degree + 1)], axis=1)
    theta, _, _, _ = np.linalg.lstsq(A, mags, rcond=None)
    return jnp.array(theta)


def build_period_channel(lattice, period_idx: int, theta):
    """Windowed EM channel for one period, with taper-eligible solenoids'
    current magnitude overridden by taper_magnitude(theta) -- everything
    else (cavities, matching-section solenoids) at nominal deck values.

    Reuses build_channel_batched_windowed as-is, then overrides via
    eqx.tree_at (same pattern as optimize_lattice.py's track()), computing
    the override values from a SEPARATE call to _nearest with the identical
    (z_center, K_SOLENOIDS_LOCAL) arguments build_channel_batched_windowed
    uses internally. This is safe because _nearest is a deterministic pure
    function (sorted by a fixed key) -- identical arguments give identical
    order, not just identical membership -- but it IS a real coupling
    between this function and build.py's internals: if
    build_channel_batched_windowed's own default k_solenoids or selection
    logic ever changes, this must change with it. Flagged here rather than
    silently relied upon.
    """
    period = lattice.meta.period
    z_center = BEAM_START + (period_idx + 0.5) * period

    ch = build_channel_batched_windowed(lattice, z_center=z_center)
    near_sols = _nearest(lattice.solenoids, z_center, K_SOLENOIDS_LOCAL)

    def override_current(s):
        sign = 1.0 if s.current >= 0.0 else -1.0
        n_norm = ((s.z - BEAM_START) / period) / lattice.meta.n_periods
        eligible = is_taper_eligible(s.z, period)
        tapered = sign * taper_magnitude(n_norm, theta)
        return jnp.where(eligible, tapered, s.current)

    new_currents = jnp.stack([override_current(s) for s in near_sols])
    new_jphi = new_currents * AMP_TO_JPHI

    sol_group = ch.groups[0]
    new_stack = eqx.tree_at(lambda f: f.field.jphi, sol_group.stack, new_jphi)
    return BatchedChannel(groups=[StackedField(stack=new_stack), ch.groups[1]])


def build_pipeline(
    n_periods: int = 5,
    n_ensemble: int = N_ENSEMBLE,
    taper_degree: int = 2,
    forward_mode: bool = True,
    transmission_power: float = 1.0,
    dz: float | None = None,
):
    """Multi-period taper-parameterized pipeline. Returns (merit, nominal_theta).

    ``n_periods``: freely configurable, 1 up to the full lattice's
    ``n_periods`` (31 currently). Cost scales linearly with this -- each
    period gets its OWN freshly-built windowed channel/wedges/window-
    positions (K_SOLENOIDS_LOCAL=24 solenoids etc, same fixed budget
    regardless of n_periods), tracked in a plain Python for loop (unrolled
    at trace time, standard and safe for a loop bound known statically up
    to the low tens -- this is NOT a jax.lax.scan over periods, because each
    period's windowed-element SELECTION is a different, non-shape-changing
    but value-changing Python-level operation that doesn't fit a scan body
    cleanly). This is what makes going from 5 periods to 31 "just work"
    rather than needing the windowing K's to grow with n_periods -- if you
    ever see K_SOLENOIDS_LOCAL et al. needing to scale with n_periods to
    keep working, something in this design has been violated.

    ``taper_degree``: polynomial degree for the steady-state solenoid
    magnitude profile (see taper_magnitude/fit_nominal_taper). Default 2
    (3 free parameters) balances fit quality (~2.3% RMS against the real
    deck currents) against dimensionality; raise for a closer nominal match
    or to give the optimizer more shape freedom, at no AD-mode-switching
    cost either way (this is exactly the reverse-mode use case).

    ``forward_mode``: see hfofo.background.track_with_drag's docstring --
    controls which diffrax adjoint is used, and hence which AD transform
    works (jax.jvp/jacfwd for True, jax.grad for False). Cost comparison is
    empirical, not assumed -- see optimize_lattice.py's
    verify_grad_consistency-equivalent below and check it at whatever
    n_periods/taper_degree you actually run.

    ``transmission_power``: merit = eigen_emittance_product /
    survival_fraction^transmission_power. Guards against the optimizer
    trading transmission for a tighter surviving core (see the
    diagnose_optimizer.py-style concern this was built to address);
    power=1 is a reasonable starting point, not empirically tuned yet.

    ``dz``: overrides the module-level DZ (60mm). ADDED AFTER A REAL FAILURE:
    dz=60mm was verified safe against exactly one known-unstable particle
    over exactly one period near the channel start (see
    hfofo.background.track_with_drag's docstring) -- NOT against a full
    multi-period, N=24+ run, where a *different* particle can become
    marginal partway through and hit a much larger aperture overshoot at
    the coarser step size before the freeze catches it. A real run at
    --n-periods 5 --n-ensemble 24 hit exactly this ("maximum number of
    solver steps was reached") where the same command had not been tested
    at this scale before. If you hit this: try --dz 15 first (cheap, direct
    test of whether the retuned default is the actual cause) before
    assuming a deeper bug -- and please report back either way, since this
    determines whether dz=60mm needs to be walked back as the default.
    """
    if dz is None:
        dz = DZ
    lattice = load_lattice(DATA)
    period = lattice.meta.period
    nominal_theta = fit_nominal_taper(lattice, degree=taper_degree)

    sample = load_sample(size=n_ensemble)
    state0 = make_ensemble_state(sample)

    rfc0_centers = rfc0_interior_centers(lattice.cavities)

    def track(theta):
        state = state0
        for period_idx in range(n_periods):
            channel = build_period_channel(lattice, period_idx, theta)
            wedges = build_union_material(
                build_wedges_windowed(
                    lattice, z_center=BEAM_START + (period_idx + 0.5) * period
                )
            )
            window_z, window_thick = cavity_window_positions_windowed(
                lattice.cavities, z_center=BEAM_START + (period_idx + 0.5) * period
            )
            z0 = BEAM_START + period_idx * period
            z1 = z0 + period
            n_steps = int(round(float(period / dz)))

            def track_one(s):
                fs, _ = track_with_drag(
                    channel, s, z0, z1, dz=dz, include_presswall=True,
                    wedges=wedges, window_z=window_z, window_thick=window_thick,
                    rfc0_centers=rfc0_centers, n_steps=n_steps, key=None,
                    rtol=1e-3, atol=1e-5, aperture_radius=APERTURE_RADIUS,
                )
                return fs

            state = jax.vmap(track_one)(state)
        return state

    def merit(theta):
        final = track(theta)
        p, t = final.kin.p, final.kin.t
        r = jnp.hypot(p.x, p.y)
        weights = jnp.where(r >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
        survival = jnp.mean(weights)
        cov = weighted_covariance6(p.x, t.x, p.y, t.y, p.ct, t.ct, weights)
        eps = eigen_emittances_6d(cov)
        return jnp.prod(eps) / survival**transmission_power

    return merit, nominal_theta


def value_and_grad_fwd(f):
    """See optimize_lattice.py's function of the same name for the full
    derivation/caveats -- identical pattern, just against this module's
    merit (theta, not per-solenoid params)."""

    def f_with_aux(theta):
        v = f(theta)
        return v, v

    def vg(theta):
        grad, val = jax.jacfwd(f_with_aux, has_aux=True)(theta)
        return val, grad

    return vg


def value_and_grad_rev(f):
    """See optimize_lattice.py's function of the same name. This is the
    mode this module is actually built to exercise -- taper parameters are
    exactly the "few coefficients, want the full gradient cheaply" case
    reverse-mode is suited for, unlike optimize_lattice.py's K individual
    currents (K=3, empirically forward-mode-favorable at that pipeline's
    scale -- see that module's docstrings). Whether reverse-mode actually
    wins HERE, at whatever taper_degree/n_periods/n_ensemble you use, is
    still an empirical question -- check with --verify-consistency, don't
    assume the theoretical direction holds without looking.
    """
    return jax.value_and_grad(f)


def _verify_one_mode_worker(forward_mode: bool, pipeline_kwargs: dict, nominal_theta_list: list):
    """Top-level (picklable) worker for verify_grad_consistency_parallel's
    spawned subprocess -- must be a module-level function, not a closure or
    lambda, since 'spawn' re-imports this module in the child process and
    calls the function by reference rather than pickling a live object.

    Independently rebuilds the pipeline rather than receiving any JAX object
    across the process boundary -- JAX closures/compiled functions are not
    reliably picklable, and rebuilding from scratch in each process is also
    exactly what makes running this under 'spawn' safe (see the caller's
    docstring for why 'spawn', not the platform default 'fork', matters here).
    """
    import time

    import jax
    import jax.numpy as jnp

    from optimize_taper import build_pipeline

    nominal_theta = jnp.array(nominal_theta_list)
    merit, _ = build_pipeline(forward_mode=forward_mode, **pipeline_kwargs)
    t0 = time.time()
    grad = jax.jacfwd(merit)(nominal_theta) if forward_mode else jax.grad(merit)(nominal_theta)
    elapsed = time.time() - t0
    return forward_mode, [float(g) for g in grad], elapsed


def verify_grad_consistency_parallel(nominal_theta, **pipeline_kwargs):
    """Same check as verify_grad_consistency, but runs the forward- and
    reverse-mode gradient computations as two INDEPENDENT OS processes
    concurrently instead of sequentially -- they don't depend on each other
    at all (each rebuilds its own pipeline and computes its own gradient
    from scratch), so on a multi-core machine wall-clock time should drop to
    roughly the slower of the two rather than their sum. Measured
    sequentially in a real N=24, K=3 run: ~388s (forward) + ~498s (reverse)
    = ~886s; this should complete in roughly the ~498s of just the slower
    side, given at least 2 free cores.

    IMPORTANT -- uses multiprocessing's 'spawn' start method explicitly, NOT
    the platform default ('fork' on Linux). Forking a process that has
    already imported/initialized JAX is a well-documented hazard: XLA's
    runtime, thread pools, and device state generally do not survive fork()
    safely (can deadlock or silently corrupt results). 'spawn' creates a
    genuinely fresh Python process that re-imports everything cleanly,
    avoiding this entirely, at the cost of some per-worker startup overhead
    (each process re-imports jax/beamline/hfofo from scratch, on top of its
    own compile) -- worth it for correctness over the fork() shortcut.

    LIMITATION: verified only for CORRECTNESS (matches
    verify_grad_consistency's sequential result exactly), not for the
    speedup this is meant to provide -- the environment this was built and
    tested in has exactly 1 CPU core (confirmed via nproc/jax.local_device_count()
    both reporting 1), so two concurrent processes have no second core to
    actually run on there. Check wall-clock on your real multi-core machine
    before assuming the speedup materializes; don't assume it from this
    docstring alone.
    """
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    theta_list = nominal_theta.tolist() if hasattr(nominal_theta, "tolist") else list(nominal_theta)

    with ctx.Pool(processes=2) as pool:
        results = pool.starmap(
            _verify_one_mode_worker,
            [(True, pipeline_kwargs, theta_list), (False, pipeline_kwargs, theta_list)],
        )

    by_mode = {fwd: (jnp.array(g), t) for fwd, g, t in results}
    grad_fwd, t_fwd = by_mode[True]
    grad_rev, t_rev = by_mode[False]

    print(f"forward-mode (jax.jacfwd, ForwardMode adjoint): grad={grad_fwd}  ({t_fwd:.1f}s)")
    print(f"reverse-mode (jax.grad, RecursiveCheckpointAdjoint): grad={grad_rev}  ({t_rev:.1f}s)")
    rel_err = float(jnp.max(jnp.abs(grad_fwd - grad_rev) / jnp.abs(grad_fwd))) * 100
    print(f"max relative difference: {rel_err:.4f}%")
    print(
        f"wall-clock: ran concurrently -- max({t_fwd:.1f}s, {t_rev:.1f}s) on a machine with >=2 "
        f"free cores, rather than the ~{t_fwd + t_rev:.1f}s the sequential version would take"
    )
    return grad_fwd, grad_rev


def verify_grad_consistency(nominal_theta, **pipeline_kwargs):
    """Build the pipeline both AD-mode ways at the same nominal_theta and
    confirm jax.jacfwd and jax.grad agree -- see optimize_lattice.py's
    function of the same name for the full rationale. Also reports timing
    for both, since which mode is cheaper is empirical per this module's
    own docstrings, not a fixed rule inherited from optimize_lattice.py's
    K=3-individual-currents case.
    """
    merit_fwd, _ = build_pipeline(forward_mode=True, **pipeline_kwargs)
    merit_rev, _ = build_pipeline(forward_mode=False, **pipeline_kwargs)

    print("forward-mode (jax.jacfwd, ForwardMode adjoint)...")
    t0 = time.time()
    grad_fwd = jax.jacfwd(merit_fwd)(nominal_theta)
    print(f"  grad={grad_fwd}  ({time.time() - t0:.1f}s)")

    print("reverse-mode (jax.grad, RecursiveCheckpointAdjoint)...")
    t0 = time.time()
    grad_rev = jax.grad(merit_rev)(nominal_theta)
    print(f"  grad={grad_rev}  ({time.time() - t0:.1f}s)")

    rel_err = float(jnp.max(jnp.abs(grad_fwd - grad_rev) / jnp.abs(grad_fwd))) * 100
    print(f"max relative difference: {rel_err:.4f}%")
    return grad_fwd, grad_rev


def optimize(merit, nominal_theta, n_steps=3, lr=0.1, checkpoint_path=None, value_and_grad_fn=None):
    """See optimize_lattice.py's function of the same name -- identical
    resumable-checkpoint pattern, generalized to however many taper
    coefficients ``nominal_theta`` has (not hardcoded to K_PARAMS=3).
    """
    n_params = len(nominal_theta)
    print(f"\noptimizing {n_params} taper coefficient(s) for {n_steps} more step(s) (Adam, lr={lr})...")
    theta = nominal_theta
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        saved = np.loadtxt(checkpoint_path, ndmin=2)
        theta = jnp.array(saved[-1, :n_params])
        print(f"resuming from checkpoint: {len(saved)} step(s) already done, "
              f"last merit={saved[-1, n_params]:.4f}")

    if value_and_grad_fn is None:
        value_and_grad_fn = value_and_grad_fwd(merit)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta)
    grad_and_val = jax.jit(value_and_grad_fn)

    rows = []
    for step in range(n_steps):
        t0 = time.time()
        val, grad = grad_and_val(theta)
        updates, opt_state = optimizer.update(grad, opt_state)
        theta = optax.apply_updates(theta, updates)
        rows.append([*theta.tolist(), float(val)])
        print(f"  step {step}: merit={float(val):.4f}  theta={theta}  ({time.time() - t0:.1f}s)")

    if checkpoint_path is not None:
        prior = []
        if os.path.exists(checkpoint_path):
            prior = np.loadtxt(checkpoint_path, ndmin=2).tolist()
        np.savetxt(checkpoint_path, np.array(prior + rows))
        print(f"wrote {checkpoint_path} ({len(prior) + len(rows)} step(s) total)")

    return theta


def track_with_diagnostics(lattice, theta, n_periods: int, n_ensemble: int, sample=None, dz: float | None = None):
    """Track n_periods periods, recording per-period survivor fraction and
    eigen-emittances -- the trajectory data the diagnostic plots need, not
    just the final merit. Returns (per_period_survival, per_period_eps_product,
    per_period_eps_individual, final_state), where per_period_eps_individual
    is a list of (eps1, eps2, eps3) tuples (NaN-filled below the rank-7
    floor). Separated from build_pipeline's merit() because plotting needs
    the intermediate history, not just the final scalar.

    NOTE on eps1/eps2/eps3 identity: eigen_emittances_6d sorts by VALUE
    (eps1 >= eps2 >= eps3), not by physical mode identity. The reference
    paper (Alexahin 2018, Table 1) tracks 3 SPECIFIC physical modes (two
    transverse normal modes + one longitudinal) consistently period to
    period via a continuous optics calculation -- our eps1/2/3 have no such
    guarantee, and if two modes' emittances ever cross in value between
    periods, which slot ("eps1" vs "eps2") represents which physical mode
    would swap, which could look like a discontinuity that isn't physically
    real. In practice, for this channel's actual sampled beam, the
    longitudinal spread (driven by the raw ct/E scale) is consistently far
    larger than either transverse mode's, so eps1 is very likely always the
    longitudinal-dominated mode in practice -- but this has NOT been proven
    to hold at every period/parameter value tested, so panels using this
    breakdown are labeled by "largest/middle/smallest", not by physical
    identity, and this caveat should stay attached to that labeling rather
    than being quietly dropped.

    ``dz``: see build_pipeline's docstring -- kept consistent here since
    plot_diagnostics calls this directly, not through build_pipeline.
    """
    from emittance_sandbox import APERTURE_RADIUS, load_sample, make_ensemble_state
    from hfofo.background import cavity_window_positions_windowed, rfc0_interior_centers
    from hfofo.union_material import build_union_material
    from hfofo.build import build_wedges_windowed
    from hfofo.emittance import eigen_emittances_6d, weighted_covariance6

    if dz is None:
        dz = DZ
    if sample is None:
        sample = load_sample(size=n_ensemble)
    state = make_ensemble_state(sample)
    period = lattice.meta.period
    rfc0_centers = rfc0_interior_centers(lattice.cavities)

    survival_history = []
    eps_product_history = []
    eps_individual_history = []
    for period_idx in range(n_periods):
        channel = build_period_channel(lattice, period_idx, theta)
        z_center = BEAM_START + (period_idx + 0.5) * period
        wedges = build_union_material(build_wedges_windowed(lattice, z_center=z_center))
        window_z, window_thick = cavity_window_positions_windowed(lattice.cavities, z_center=z_center)
        z0 = BEAM_START + period_idx * period
        z1 = z0 + period
        n_steps = int(round(float(period / dz)))

        def track_one(s):
            fs, _ = track_with_drag(
                channel, s, z0, z1, dz=dz, include_presswall=True,
                wedges=wedges, window_z=window_z, window_thick=window_thick,
                rfc0_centers=rfc0_centers, n_steps=n_steps, key=None,
                rtol=1e-3, atol=1e-5, aperture_radius=APERTURE_RADIUS,
            )
            return fs

        state = jax.vmap(track_one)(state)
        p, t = state.kin.p, state.kin.t
        r = jnp.hypot(p.x, p.y)
        weights = jnp.where(r >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
        survival = float(jnp.mean(weights))
        n_survivors = int(jnp.sum(weights))
        if n_survivors >= 7:  # 6D covariance rank floor -- see build_pipeline's docstring
            cov = weighted_covariance6(p.x, t.x, p.y, t.y, p.ct, t.ct, weights)
            eps = eigen_emittances_6d(cov)
            eps_product = float(jnp.prod(eps))
            eps_individual = tuple(float(e) for e in eps)
        else:
            eps_product = float("nan")  # rank-deficient -- not a meaningful number, don't plot as if it were
            eps_individual = (float("nan"), float("nan"), float("nan"))
        survival_history.append(survival)
        eps_product_history.append(eps_product)
        eps_individual_history.append(eps_individual)
        eps_str = f"eps={eps_individual[0]:.3e}/{eps_individual[1]:.3e}/{eps_individual[2]:.3e}" if n_survivors >= 7 else "eps=N/A (below rank-7 floor)"
        print(f"  period {period_idx}: survival={survival:.3f} ({n_survivors} alive)  {eps_str}")

    return survival_history, eps_product_history, eps_individual_history, state


def plot_diagnostics(
    lattice, nominal_theta, final_theta, n_periods: int, n_ensemble: int,
    checkpoint_rows=None, outdir: str = "artifacts", dz: float | None = None,
):
    """Six-panel diagnostic figure, generated as a normal part of running
    this script (not an afterthought) -- per project convention, plots like
    this are what make judgement calls about optimizer behavior possible;
    several real findings this session (the aperture-loss trend, the
    rank-deficiency floor) would have been visible immediately here rather
    than needing separate ad hoc debugging.

    Benchmarked against Alexahin 2018 (JINST 13 P08013), the actual HFOFO
    design paper this channel is built from: initial/final 6D emittance
    ratio 112.8 (36.4 transverse, 3.1 longitudinal) at ~65% transmission
    over the real 124m (~30-period) channel. Two different ways these get
    used here, deliberately:
    - Transmission (panel 4): the 65% line is drawn directly on the partial
      run, because transmission only ever decreases along the channel --
      if you're already below 65% within the first few periods (out of
      ~30), you're guaranteed to finish below it too. This is exactly the
      "something to check" signal it's meant to be.
    - 6D cooling ratio (panel 5): NOT drawn as a flat 112.8 line, because
      that's a FULL ~30-period target and cooling compounds multiplicatively
      -- expecting anywhere near it after only n_periods periods would be
      misleading. Instead annotated with the equivalent per-period rate
      (112.8^(1/30), assuming uniform compounding -- a real simplification,
      the true design almost certainly doesn't cool at a perfectly uniform
      per-period rate, but it's a far more honest reference point for a
      partial run than the full-channel number would be on its own).

    Panel 1: taper profile -- nominal (deck-fit) vs final (optimized) current
      magnitude vs period index, with the real deck data points overlaid so
      it's visible whether the optimizer found something that still looks
      like a physically sensible taper or something wild.
    Panel 2: merit vs optimizer step, from the checkpoint file.
    Panel 3: reference benchmark text (this project's own convention, not
      from the paper) -- the Alexahin 2018 numbers restated here so they're
      visible alongside the plot rather than requiring the paper open
      separately.
    Panel 4: per-period survival fraction, nominal vs final, with the 65%
      full-channel target -- directly shows whether an optimized taper is
      winning by cooling or by losing more/fewer particles (the
      diagnose_optimizer.py-style concern).
    Panel 5: 6D cooling ratio (eps_product[0]/eps_product[period]) per
      period, nominal vs final -- see the compounding note above for why
      this is a ratio, not the raw absolute eigen-emittance product.
    Panel 6: individual eigen-emittances (largest/middle/smallest -- NOT
      guaranteed physical mode identity, see track_with_diagnostics's
      docstring for why), nominal vs final, mirroring the paper's Figure 6
      style (3 mode curves per beam).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    period = lattice.meta.period
    n_total = lattice.meta.n_periods
    steady = [s for s in lattice.solenoids if is_taper_eligible(s.z, period)]
    n_idx_real = np.array([(s.z - BEAM_START) / period for s in steady])
    mag_real = np.abs(np.array([s.current for s in steady]))

    n_idx_curve = np.linspace(TAPER_MIN_PERIOD, TAPER_MAX_PERIOD, 200)
    n_norm_curve = n_idx_curve / n_total
    nominal_curve = np.array([float(taper_magnitude(nn, nominal_theta)) for nn in n_norm_curve])
    final_curve = np.array([float(taper_magnitude(nn, final_theta)) for nn in n_norm_curve])

    print("tracking at nominal_theta for diagnostics...")
    surv_nom, eps_nom, eps_ind_nom, _ = track_with_diagnostics(lattice, nominal_theta, n_periods, n_ensemble, dz=dz)
    print("tracking at final_theta for diagnostics...")
    surv_fin, eps_fin, eps_ind_fin, _ = track_with_diagnostics(lattice, final_theta, n_periods, n_ensemble, dz=dz)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    periods_x = list(range(n_periods))

    ax = axs[0, 0]
    ax.scatter(n_idx_real, mag_real, s=8, alpha=0.4, color="gray", label="real deck currents")
    ax.plot(n_idx_curve, nominal_curve, label="nominal (fitted)", color="C0")
    ax.plot(n_idx_curve, final_curve, label="final (optimized)", color="C3")
    ax.set_xlabel("period index")
    ax.set_ylabel("|current| [A, engineering units]")
    ax.set_title("Solenoid current taper profile")
    ax.legend()

    ax = axs[0, 1]
    if checkpoint_rows is not None and len(checkpoint_rows) > 0:
        merits = [row[-1] for row in checkpoint_rows]
        ax.plot(range(len(merits)), merits, marker="o")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("merit (lower is better)")
        ax.set_title("Optimization convergence")
    else:
        ax.text(0.5, 0.5, "no checkpoint history", ha="center", va="center")
        ax.set_title("Optimization convergence (no data)")

    ax = axs[0, 2]
    ax.axis("off")
    per_period_rate = 112.8 ** (1.0 / 30.0)
    benchmark_text = (
        "Reference: Alexahin 2018\n(JINST 13 P08013)\n\n"
        "Real HFOFO design, ~30 periods (124m):\n"
        "  6D emittance ratio: 112.8x\n"
        "    transverse (4D): 36.4x\n"
        "    longitudinal:     3.1x\n"
        "  transmission: ~65%\n\n"
        f"Equivalent uniform per-period\n6D cooling rate: {per_period_rate:.3f}x/period\n"
        "(assumes constant rate -- a real\nsimplification, not from the paper)"
    )
    ax.text(0.05, 0.95, benchmark_text, transform=ax.transAxes, va="top", ha="left",
            fontsize=10, family="monospace")

    ax = axs[1, 0]
    ax.plot(periods_x, surv_nom, marker="o", label="nominal", color="C0")
    ax.plot(periods_x, surv_fin, marker="o", label="final", color="C3")
    ax.axhline(0.65, color="k", linestyle="--", linewidth=1.2, label="paper target: 65% (full channel)")
    ax.axhline(7 / n_ensemble, color="gray", linestyle=":", linewidth=1, label="rank-7 floor (this N)")
    ax.set_xlabel("period")
    ax.set_ylabel("survival fraction")
    ax.set_title(f"Transmission per period (N={n_ensemble})")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    eps0_nom = eps_nom[0] if eps_nom and eps_nom[0] == eps_nom[0] else float("nan")  # NaN-safe first value
    eps0_fin = eps_fin[0] if eps_fin and eps_fin[0] == eps_fin[0] else float("nan")
    ratio_nom = [eps0_nom / e if e == e and e != 0 else float("nan") for e in eps_nom]
    ratio_fin = [eps0_fin / e if e == e and e != 0 else float("nan") for e in eps_fin]
    ax.plot(periods_x, ratio_nom, marker="o", label="nominal", color="C0")
    ax.plot(periods_x, ratio_fin, marker="o", label="final", color="C3")
    ax.set_xlabel("period")
    ax.set_ylabel("eps_product[0] / eps_product[period]")
    ax.set_title("6D cooling ratio so far\n(NOT directly comparable to the full 112.8x target -- see docstring)")
    ax.legend()

    ax = axs[1, 2]
    labels = ["largest", "middle", "smallest"]
    colors = ["C4", "C5", "C6"]
    for mode_idx in range(3):
        nom_vals = [row[mode_idx] for row in eps_ind_nom]
        fin_vals = [row[mode_idx] for row in eps_ind_fin]
        ax.plot(periods_x, nom_vals, color=colors[mode_idx], linestyle="-", marker="o",
                 label=f"{labels[mode_idx]} (nominal)")
        ax.plot(periods_x, fin_vals, color=colors[mode_idx], linestyle="--", marker="s",
                 label=f"{labels[mode_idx]} (final)")
    ax.set_yscale("log")
    ax.set_xlabel("period")
    ax.set_ylabel("eigen-emittance (log scale)")
    ax.set_title("Individual eigen-emittances\n(largest/middle/smallest by VALUE, not physical mode identity)")
    ax.legend(fontsize=7)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "optimize_taper_diagnostics.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-periods", type=int, default=5,
        help="number of lattice periods to track (1 to the full 31) -- see "
             "build_pipeline's docstring for why this is cheap to change",
    )
    ap.add_argument("--n-ensemble", type=int, default=N_ENSEMBLE)
    ap.add_argument("--taper-degree", type=int, default=2)
    ap.add_argument("--transmission-power", type=float, default=1.0)
    ap.add_argument(
        "--dz", type=float, default=None,
        help="mm, overrides the module default (60mm). ADDED AFTER A REAL "
             "FAILURE: dz=60mm was only verified against one known-unstable "
             "particle over one period near the channel start, not a full "
             "multi-period/N=24+ run -- if you hit 'maximum number of solver "
             "steps was reached' at a scale not tested before, try --dz 15 "
             "first to check whether the retuned default is the cause",
    )
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--checkpoint", default="artifacts/optimize_taper_checkpoint.txt")
    ap.add_argument("--skip-jacfwd-check", action="store_true")
    ap.add_argument("--reverse-mode", action="store_true")
    ap.add_argument("--verify-consistency", action="store_true")
    ap.add_argument(
        "--parallel", action="store_true",
        help="with --verify-consistency, run the forward- and reverse-mode "
             "checks as two concurrent OS processes (multiprocessing, "
             "'spawn' start method) instead of sequentially -- see "
             "verify_grad_consistency_parallel's docstring; needs >=2 free "
             "CPU cores to actually help, has no effect without --verify-consistency",
    )
    ap.add_argument(
        "--plot", action="store_true",
        help="generate artifacts/optimize_taper_diagnostics.png (taper profile, "
             "merit convergence, per-period survival, per-period eigen-emittance) "
             "-- re-tracks at nominal and final theta, so this costs roughly 2x "
             "a normal run's tracking time",
    )
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    pipeline_kwargs = dict(
        n_periods=args.n_periods,
        n_ensemble=args.n_ensemble,
        taper_degree=args.taper_degree,
        transmission_power=args.transmission_power,
        dz=args.dz * u.mm if args.dz is not None else None,
    )
    merit, nominal_theta = build_pipeline(forward_mode=not args.reverse_mode, **pipeline_kwargs)
    print(f"nominal_theta (degree={args.taper_degree}): {nominal_theta}")

    if args.verify_consistency:
        if args.parallel:
            verify_grad_consistency_parallel(nominal_theta, **pipeline_kwargs)
        else:
            verify_grad_consistency(nominal_theta, **pipeline_kwargs)
    elif not args.skip_jacfwd_check:
        if args.reverse_mode:
            print("(--skip-jacfwd-check has no effect with --reverse-mode; "
                  "use --verify-consistency instead)")
        else:
            print("verifying jacfwd against finite-difference on one component...")
            grad = jax.jacfwd(merit)(nominal_theta)
            eps = 1e-2
            direction = jnp.zeros_like(nominal_theta).at[0].set(1.0)
            fd = (float(merit(nominal_theta + eps * direction))
                  - float(merit(nominal_theta - eps * direction))) / (2 * eps)
            print(f"  jacfwd[0]={float(grad[0]):.6f}  fd={fd:.6f}  "
                  f"diff={abs(float(grad[0]) - fd) / abs(fd) * 100:.4f}%")

    value_and_grad_fn = value_and_grad_rev(merit) if args.reverse_mode else value_and_grad_fwd(merit)
    final_theta = optimize(
        merit, nominal_theta, n_steps=args.steps, lr=args.lr,
        checkpoint_path=args.checkpoint, value_and_grad_fn=value_and_grad_fn,
    )

    if args.plot:
        lattice = load_lattice(DATA)
        checkpoint_rows = None
        if os.path.exists(args.checkpoint):
            checkpoint_rows = np.loadtxt(args.checkpoint, ndmin=2).tolist()
        plot_diagnostics(
            lattice, nominal_theta, final_theta, args.n_periods, args.n_ensemble,
            checkpoint_rows=checkpoint_rows, dz=pipeline_kwargs["dz"],
        )


if __name__ == "__main__":
    main()
