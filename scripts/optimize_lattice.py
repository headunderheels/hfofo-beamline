#!/usr/bin/env python3
"""Extends gradient_check.py from one design parameter to several, via
either forward-mode (jax.jacfwd) or reverse-mode (jax.grad) AD, then wires
up a real optax optimizer over that parameter set and confirms the merit
(output eigen-emittance product) genuinely decreases.

Two phases, run in order (see main()):
1. Gradient verification: for forward-mode, the full Jacobian
   d(merit)/d(params), spot-checked against finite-difference on one
   component. For reverse-mode, use --verify-consistency instead: build the
   pipeline both ways at the same nominal params and confirm jax.jacfwd and
   jax.grad agree (they compute the same analytic gradient via different AD
   transforms; any mismatch would mean something is actually wrong).
2. optax optimization: a handful of gradient-descent-family steps (Adam,
   small learning rate) using the chosen mode's gradient each step, printing
   the merit at every step to confirm it decreases.

AD mode: an earlier round of this project concluded "jax.grad fails on this
pipeline, always use jax.jvp" -- too broad a claim. The precise statement
(see hfofo.background.track_with_drag's docstring for the full derivation
and measured cost comparisons): diffrax's adjoint choice must match the AD
direction. forward_mode=True (this script's default) pairs with
jax.jvp/jax.jacfwd; forward_mode=False pairs with ordinary jax.grad. Which
mode is actually cheaper for THIS pipeline at a given ensemble size/parameter
count is an empirical question -- measured on this exact pipeline (N=6,
K=3): forward-mode ~138s, reverse-mode ~189s, i.e. reverse-mode is currently
SLOWER here, not faster (an earlier draft of this docstring claimed a
crossover "already around K=2" extrapolated from a much simpler toy problem;
that specific number did not hold up against this pipeline's actual timing
and has been removed). Use --verify-consistency to check both modes agree
AND compare their timing at whatever K/ensemble size you actually intend to
use, rather than assuming reverse-mode wins past some fixed K.

Uses a small ensemble (N=6) and few design parameters (K=3, the first 3
windowed solenoids' currents) to keep total runtime bounded -- NOT the
target scale for a real optimization campaign, which should use a larger
ensemble/parameter set once this mechanism is trusted (gradient_check.py
already established it is, for a single parameter; this extends that to
several and to an actual optimizer, not just gradient computation).

Usage:
    uv run python scripts/optimize_lattice.py
    uv run python scripts/optimize_lattice.py --reverse-mode --verify-consistency
"""

from __future__ import annotations

import argparse
import os
import time

import beamline.jax  # noqa: F401
import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import numpy as np
import optax

from emittance_sandbox import APERTURE_RADIUS, load_sample, make_ensemble_state
from hfofo.background import (
    cavity_window_positions_windowed,
    rfc0_interior_centers,
    track_with_drag,
)
from hfofo.build import AMP_TO_JPHI, build_channel_batched_windowed, build_wedges_windowed
from hfofo.emittance import eigen_emittances, weighted_covariance4
from hfofo.load import load_lattice
from hfofo.stacked import BatchedChannel, StackedField
from hfofo.union_material import build_union_material

DATA = "data/hfofo.yaml"
N_ENSEMBLE = 6  # overridable via --n-ensemble; also read by diagnose_optimizer.py
K_PARAMS = 3  # first K windowed solenoids' currents, as design parameters
BEAM_START = -700.0 * u.mm
DZ = 15.0 * u.mm  # REVERTED from 60mm -- see optimize_taper.py's d73df4b for why: a real,
           # reproduced failure ("max_steps was reached") at multi-period,
           # N>=24 scale that the original 60mm retuning was never tested
           # against (only verified for one particle over one period). This
           # script had not been re-tested at that scale either -- reverted
           # here as a precaution, not because THIS script was independently
           # confirmed to fail, matching the same reasoning.


def build_pipeline(
    n_ensemble: int = N_ENSEMBLE, return_track_fn: bool = False, forward_mode: bool = True
):
    """Shared setup: lattice, base channel/wedges/windows, ensemble state,
    and the merit(params) closure. Returns (merit, nominal_params), or
    (merit, nominal_params, track_fn) if ``return_track_fn`` -- track_fn
    (params) -> raw final ensemble MuonStateDz (pre-covariance/pre-merit),
    for diagnostics that need the individual particles' final phase space
    (see diagnose_optimizer.py) rather than just the scalar merit.

    ``forward_mode``: passed through to ``track_with_drag``/``diffrax_solve``.
    True (default) selects diffrax's ``ForwardMode`` adjoint, which pairs
    with ``jax.jvp``/``jax.jacfwd`` (``value_and_grad_fwd`` below) and BREAKS
    under ``jax.grad``. False selects ``RecursiveCheckpointAdjoint``, which
    pairs with ordinary ``jax.grad``/``jax.value_and_grad``
    (``value_and_grad_rev`` below). See ``hfofo.background.track_with_drag``'s
    docstring for the full derivation and measured cost comparisons --
    which mode is actually faster is empirical, not a fixed rule; measured
    on this exact pipeline at K=3, N=6, reverse-mode is currently SLOWER
    (~189s vs ~138s), not faster. Use ``verify_grad_consistency`` below to
    check both modes agree AND compare timing at whatever K/ensemble size
    you actually intend to use.
    """
    sample = load_sample(size=n_ensemble)
    state0 = make_ensemble_state(sample)

    lattice = load_lattice(DATA)
    period = lattice.meta.period
    z0 = BEAM_START
    z1 = z0 + period
    n_steps = int(round(float((z1 - z0) / DZ)))
    z_center = float((z0 + z1) / 2)

    base_channel = build_channel_batched_windowed(lattice, z_center=z_center)
    wedges = build_union_material(build_wedges_windowed(lattice, z_center=z_center))
    window_z, window_thick = cavity_window_positions_windowed(lattice.cavities, z_center=z_center)
    rfc0_centers = rfc0_interior_centers(lattice.cavities)

    nominal_params = base_channel.groups[0].stack.field.jphi[:K_PARAMS] / AMP_TO_JPHI

    def track(params):
        """params: (K_PARAMS,) vector of solenoid currents (design
        parameters) -> raw final ensemble state after 1 period. Overrides
        the first K_PARAMS windowed solenoids' jphi via eqx.tree_at, same
        pattern as gradient_check.py, just vectorized over K parameters
        instead of 1.
        """
        sol_group = base_channel.groups[0]
        new_jphi = sol_group.stack.field.jphi.at[:K_PARAMS].set(params * AMP_TO_JPHI)
        new_field = eqx.tree_at(lambda f: f.jphi, sol_group.stack.field, new_jphi)
        new_stack = eqx.tree_at(lambda s: s.field, sol_group.stack, new_field)
        channel = BatchedChannel(groups=[StackedField(stack=new_stack), base_channel.groups[1]])

        def track_one(state):
            final_state, _ = track_with_drag(
                channel, state, z0, z1, dz=DZ, include_presswall=True, wedges=wedges,
                window_z=window_z, window_thick=window_thick, rfc0_centers=rfc0_centers,
                n_steps=n_steps, key=None, rtol=1e-3, atol=1e-5,
                aperture_radius=APERTURE_RADIUS, forward_mode=forward_mode,
            )
            return final_state

        return jax.vmap(track_one)(state0)

    def merit(params):
        """params -> output eigen-emittance product for the surviving
        ensemble after 1 period."""
        final = track(params)
        p, t = final.kin.p, final.kin.t
        r = jnp.hypot(p.x, p.y)
        weights = jnp.where(r >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
        cov = weighted_covariance4(p.x, t.x, p.y, t.y, weights)
        e1, e2 = eigen_emittances(cov)
        return e1 * e2

    if return_track_fn:
        return merit, nominal_params, jax.jit(track)
    return merit, nominal_params


def verify_jacfwd(merit, nominal_params):
    print(f"nominal params: {nominal_params}")
    print("computing jax.jacfwd (K_PARAMS forward-mode passes, vectorized)...")
    t0 = time.time()
    grad_fn = jax.jacfwd(merit)
    grad = grad_fn(nominal_params)
    print(f"  jacobian = {grad}  ({time.time()-t0:.1f}s)")

    # spot-check component 0 against finite-difference
    eps = 0.1
    direction = jnp.zeros(K_PARAMS).at[0].set(1.0)
    t0 = time.time()
    fd = (merit(nominal_params + eps * direction) - merit(nominal_params - eps * direction)) / (2 * eps)
    rel_err = abs(float(fd) - float(grad[0])) / abs(float(grad[0])) * 100
    print(f"  component[0] FD check: jacfwd={float(grad[0]):.4f}  FD={float(fd):.4f}"
          f"  rel_err={rel_err:.2f}%  ({time.time()-t0:.1f}s)")
    return grad


def value_and_grad_fwd(f):
    """Forward-mode equivalent of jax.value_and_grad, for use with a merit
    built via ``build_pipeline(forward_mode=True)`` (the default).

    jax.value_and_grad itself defaults to REVERSE-mode AD (jax.grad). An
    earlier round of this project found jax.grad fails outright against a
    merit built with forward_mode=True (diffrax's ForwardMode adjoint is
    built for forward-mode AD only, and raises under jax.grad) and
    concluded from that "jax.grad always fails on this pipeline" -- too
    broad a claim. The precise statement: the adjoint must match the AD
    direction. This is the forward-mode side of that pairing; see
    ``value_and_grad_rev`` for the reverse-mode side (paired with a merit
    built via ``forward_mode=False``). Which side is actually cheaper is
    empirical, not a fixed rule -- see
    ``hfofo.background.track_with_drag``'s docstring for measured numbers
    (reverse-mode is currently the slower of the two on this pipeline at
    K=3, N=6) and ``verify_grad_consistency`` below for a direct check that
    both sides compute the same gradient.

    Gets the primal value "for free" alongside the Jacobian by having f
    return itself as an aux output (has_aux=True runs f normally within
    each jvp call, so this costs nothing beyond what jacfwd already does).
    """

    def f_with_aux(params):
        v = f(params)
        return v, v

    def vg(params):
        grad, val = jax.jacfwd(f_with_aux, has_aux=True)(params)
        return val, grad

    return vg


def value_and_grad_rev(f):
    """Reverse-mode value_and_grad, for use with a merit built via
    ``build_pipeline(forward_mode=False)`` -- see ``value_and_grad_fwd``'s
    docstring and ``hfofo.background.track_with_drag``'s docstring for why
    the pairing matters and the measured cost tradeoff between the two.

    This is just ``jax.value_and_grad`` (ordinary reverse-mode AD) -- unlike
    the forward-mode side, it needs no wrapper: a single reverse pass already
    yields both the primal value and the full gradient, which is the entire
    point of reverse-mode for a many-parameters/one-output function.
    """
    return jax.value_and_grad(f)


def verify_grad_consistency(nominal_params, n_ensemble: int = N_ENSEMBLE):
    """Build the pipeline BOTH ways (forward_mode=True and False) at the
    same nominal_params and confirm jax.jacfwd (forward-mode) and jax.grad
    (reverse-mode) agree. Both compute the exact same analytic gradient via
    different AD transforms/adjoints -- unlike a finite-difference spot
    check (which only bounds error to FD's own precision), a mismatch here
    would mean something is actually broken, not just "less precise." Run
    this once before trusting --reverse-mode for a real optimization
    campaign, the same way verify_jacfwd's FD check was used to establish
    trust in the forward-mode path originally.
    """
    merit_fwd, _ = build_pipeline(n_ensemble=n_ensemble, forward_mode=True)
    merit_rev, _ = build_pipeline(n_ensemble=n_ensemble, forward_mode=False)

    print("forward-mode (jax.jacfwd, ForwardMode adjoint)...")
    t0 = time.time()
    grad_fwd = jax.jacfwd(merit_fwd)(nominal_params)
    print(f"  grad={grad_fwd}  ({time.time() - t0:.1f}s)")

    print("reverse-mode (jax.grad, RecursiveCheckpointAdjoint)...")
    t0 = time.time()
    grad_rev = jax.grad(merit_rev)(nominal_params)
    print(f"  grad={grad_rev}  ({time.time() - t0:.1f}s)")

    rel_err = float(jnp.max(jnp.abs(grad_fwd - grad_rev) / jnp.abs(grad_fwd))) * 100
    print(f"max relative difference: {rel_err:.4f}%")
    return grad_fwd, grad_rev


def optimize(merit, nominal_params, n_steps=3, lr=0.5, checkpoint_path=None, value_and_grad_fn=None):
    """Runs Adam for n_steps additional steps, resuming from a checkpoint
    file if one exists (each step costs ~1-2 minutes wall-clock in
    forward-mode, more in reverse-mode for very few parameters -- see
    hfofo.background.track_with_drag's docstring -- a multi-step run
    generally needs several invocations to fit typical tool-call/session
    time budgets, same reason track_full_channel.py is resumable).

    ``value_and_grad_fn``: a ``params -> (value, grad)`` callable. Defaults
    to ``value_and_grad_fwd(merit)`` for backward compatibility (matches
    this function's original forward-mode-only behavior); pass
    ``value_and_grad_rev(merit)`` instead if ``merit`` was built via
    ``build_pipeline(forward_mode=False)`` (see main()'s ``--reverse-mode``).
    Passing a forward-mode value_and_grad_fn against a forward_mode=False
    merit (or vice versa) will not raise cleanly -- it depends on which
    adjoint the merit's own track_with_drag call was actually built with,
    not on this argument, so double-check the two agree with each other.

    NOTE: only ``params`` is checkpointed, not optax's internal Adam state
    (momentum/variance estimates) -- each resume reinitializes a fresh
    optimizer state. For a short run (a handful of total steps) this is a
    reasonable simplification, not a hidden correctness issue: Adam's
    first step from a fresh state is equivalent to plain gradient descent
    (zero-initialized moment estimates), only mildly different from a true
    warm-started step. For a long optimization campaign where momentum
    continuity actually matters, extend this to serialize opt_state too
    (e.g. via jax.tree_util.tree_flatten + np.savez).

    Writes the checkpoint after EVERY step, not once after the whole loop
    (a real bug, found the hard way in optimize_taper.py's twin function: a
    run requesting several steps that crashed on the LAST one -- e.g.
    hitting max_steps from the dz=60mm issue documented in
    track_with_drag's docstring -- silently discarded every successfully-
    completed earlier step from that invocation, since nothing had been
    written to disk yet when the exception propagated out of the loop).
    """
    print(f"\noptimizing {K_PARAMS} parameters for {n_steps} more step(s) (Adam, lr={lr})...")
    params = nominal_params
    prior = []
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        saved = np.loadtxt(checkpoint_path, ndmin=2)
        prior = saved.tolist()
        params = jnp.array(saved[-1, :K_PARAMS])
        print(f"resuming from checkpoint: {len(saved)} step(s) already done, "
              f"last merit={saved[-1, K_PARAMS]:.4f}")

    if value_and_grad_fn is None:
        value_and_grad_fn = value_and_grad_fwd(merit)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    grad_and_val = jax.jit(value_and_grad_fn)

    for step in range(n_steps):
        t0 = time.time()
        val, grad = grad_and_val(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        dt = time.time() - t0
        prior.append(list(np.asarray(params)) + [float(val)])
        print(f"  step {step}: merit={float(val):.4f}  params={params}  ({dt:.1f}s)")
        if checkpoint_path is not None:
            np.savetxt(checkpoint_path, np.array(prior))

    if checkpoint_path is not None:
        print(f"wrote {checkpoint_path} ({len(prior)} step(s) total)")

    final_val = float(merit(params))
    print(f"\ncurrent merit={final_val:.4f}")
    return params, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3, help="optimizer steps to run this invocation")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--n-ensemble", type=int, default=N_ENSEMBLE)
    ap.add_argument("--checkpoint", default="artifacts/optimize_checkpoint.txt")
    ap.add_argument("--skip-jacfwd-check", action="store_true")
    ap.add_argument(
        "--reverse-mode", action="store_true",
        help="use reverse-mode AD (jax.grad + diffrax's RecursiveCheckpointAdjoint) "
             "instead of forward-mode (jax.jacfwd + ForwardMode) -- NOT confirmed "
             "faster at this pipeline's default K=3/N=6 (measured slower there, "
             "~189s vs ~138s); use --verify-consistency to check timing at your "
             "actual K/ensemble size before assuming this helps, see "
             "hfofo.background.track_with_drag's docstring for the measured numbers",
    )
    ap.add_argument(
        "--verify-consistency", action="store_true",
        help="build the pipeline both AD-mode ways and confirm jax.jacfwd and "
             "jax.grad agree at nominal params, instead of the (forward-mode-only) "
             "jacfwd-vs-finite-difference spot check -- do this once before trusting "
             "--reverse-mode for a real optimization campaign",
    )
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    merit, nominal_params = build_pipeline(
        n_ensemble=args.n_ensemble, forward_mode=not args.reverse_mode
    )

    if args.verify_consistency:
        verify_grad_consistency(nominal_params, n_ensemble=args.n_ensemble)
    elif not args.skip_jacfwd_check:
        if args.reverse_mode:
            print(
                "(--skip-jacfwd-check has no effect with --reverse-mode, since the "
                "jacfwd-vs-finite-difference check only applies to the forward-mode "
                "path; pass --verify-consistency instead if you want a cross-check "
                "before optimizing)"
            )
        else:
            verify_jacfwd(merit, nominal_params)

    value_and_grad_fn = value_and_grad_rev(merit) if args.reverse_mode else value_and_grad_fwd(merit)
    optimize(
        merit, nominal_params, n_steps=args.steps, lr=args.lr,
        checkpoint_path=args.checkpoint, value_and_grad_fn=value_and_grad_fn,
    )


if __name__ == "__main__":
    main()
