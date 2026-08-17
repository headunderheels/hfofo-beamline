#!/usr/bin/env python3
"""Extends gradient_check.py from one design parameter to several via
jax.jacfwd, then wires up a real optax optimizer over that small parameter
set and confirms the merit (output eigen-emittance product) genuinely
decreases.

Two phases, run in order (see main()):
1. jacfwd verification: the full Jacobian d(merit)/d(params) for a small
   vector of solenoid currents, spot-checked against finite-difference on
   one component (checking every component would multiply the already
   substantial per-evaluation cost for little extra confidence, given
   jax.jvp -- which jacfwd is built from -- was already verified
   component-by-component in gradient_check.py/test_emittance.py).
2. optax optimization: a handful of gradient-descent-family steps
   (Adam, small learning rate) using jacfwd's gradient each step, printing
   the merit at every step to confirm it decreases.

Uses a small ensemble (N=6) and few design parameters (K=3, the first 3
windowed solenoids' currents) to keep total runtime bounded -- NOT the
target scale for a real optimization campaign, which should use a larger
ensemble/parameter set once this mechanism is trusted (gradient_check.py
already established it is, for a single parameter; this extends that to
several and to an actual optimizer, not just gradient computation).

Usage:
    uv run python scripts/optimize_lattice.py
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
from hfofo.background import cavity_window_positions, rfc0_interior_centers, track_with_drag
from hfofo.build import AMP_TO_JPHI, build_channel_batched_windowed, build_wedges_windowed
from hfofo.emittance import eigen_emittances, weighted_covariance4
from hfofo.load import load_lattice
from hfofo.stacked import BatchedChannel, StackedField
from hfofo.union_material import build_union_material

DATA = "data/hfofo.yaml"
N_ENSEMBLE = 6  # overridable via --n-ensemble; also read by diagnose_optimizer.py
K_PARAMS = 3  # first K windowed solenoids' currents, as design parameters
BEAM_START = -700.0 * u.mm
DZ = 15.0 * u.mm


def build_pipeline(n_ensemble: int = N_ENSEMBLE, return_track_fn: bool = False):
    """Shared setup: lattice, base channel/wedges/windows, ensemble state,
    and the merit(params) closure. Returns (merit, nominal_params), or
    (merit, nominal_params, track_fn) if ``return_track_fn`` -- track_fn
    (params) -> raw final ensemble MuonStateDz (pre-covariance/pre-merit),
    for diagnostics that need the individual particles' final phase space
    (see diagnose_optimizer.py) rather than just the scalar merit.
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
    window_z, window_thick = cavity_window_positions(lattice.cavities)
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
                aperture_radius=APERTURE_RADIUS,
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
    """Forward-mode equivalent of jax.value_and_grad. jax.value_and_grad
    itself defaults to REVERSE-mode AD (jax.grad), which fails outright on
    this pipeline -- diffrax's adaptive step-size control uses a
    lax.while_loop with a runtime-dependent length, and reverse-mode AD
    cannot differentiate through that (confirmed directly in an earlier
    session: ValueError, "Reverse-mode differentiation does not work for
    lax.while_loop..."). Every gradient in this project must go through
    forward-mode (jax.jvp/jax.jacfwd) instead -- see gradient_check.py and
    the handoff docs.

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


def optimize(merit, nominal_params, n_steps=3, lr=0.5, checkpoint_path=None):
    """Runs Adam for n_steps additional steps, resuming from a checkpoint
    file if one exists (each step costs ~2 minutes wall-clock -- a
    multi-step run generally needs several invocations to fit typical
    tool-call/session time budgets, same reason track_full_channel.py is
    resumable).

    NOTE: only ``params`` is checkpointed, not optax's internal Adam state
    (momentum/variance estimates) -- each resume reinitializes a fresh
    optimizer state. For a short run (a handful of total steps) this is a
    reasonable simplification, not a hidden correctness issue: Adam's
    first step from a fresh state is equivalent to plain gradient descent
    (zero-initialized moment estimates), only mildly different from a true
    warm-started step. For a long optimization campaign where momentum
    continuity actually matters, extend this to serialize opt_state too
    (e.g. via jax.tree_util.tree_flatten + np.savez).
    """
    print(f"\noptimizing {K_PARAMS} parameters for {n_steps} more step(s) (Adam, lr={lr})...")
    params = nominal_params
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        saved = np.loadtxt(checkpoint_path)
        params = jnp.array(saved[-1, :K_PARAMS])
        print(f"resuming from checkpoint: {len(saved)} step(s) already done, "
              f"last merit={saved[-1, K_PARAMS]:.4f}")

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    grad_and_val = jax.jit(value_and_grad_fwd(merit))

    rows = []
    for step in range(n_steps):
        t0 = time.time()
        val, grad = grad_and_val(params)
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        dt = time.time() - t0
        rows.append(list(np.asarray(params)) + [float(val)])
        print(f"  step {step}: merit={float(val):.4f}  params={params}  ({dt:.1f}s)")

    if checkpoint_path is not None:
        prior = []
        if os.path.exists(checkpoint_path):
            prior = np.loadtxt(checkpoint_path).tolist()
            if prior and not isinstance(prior[0], list):
                prior = [prior]
        all_rows = prior + rows
        np.savetxt(checkpoint_path, np.array(all_rows))
        print(f"wrote {checkpoint_path} ({len(all_rows)} step(s) total)")

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
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    merit, nominal_params = build_pipeline(n_ensemble=args.n_ensemble)
    if not args.skip_jacfwd_check:
        verify_jacfwd(merit, nominal_params)
    optimize(merit, nominal_params, n_steps=args.steps, lr=args.lr,
              checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
