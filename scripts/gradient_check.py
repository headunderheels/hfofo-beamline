#!/usr/bin/env python3
"""Verify jax.jvp (forward-mode AD) works correctly through the FULL
milestone-D pipeline: one solenoid's current -> ensemble tracking (with the
differentiable aperture-weighted covariance, not NumPy boolean exclusion) ->
output eigen-emittance product.

This is the last unverified link before real gradient-based optimization:
every piece has been checked in isolation (jvp through diffrax_solve alone,
jvp through track_with_drag alone, jvp through eigen_emittances/
weighted_covariance4 alone) but never all the way through together with a
design-parameter perturbation. Uses a small ensemble (N=8, 1 period) to keep
this verification step's runtime reasonable -- NOT the target scale for
actual optimization, which should use the full 24 (or larger).

Usage:
    uv run python scripts/gradient_check.py
"""

from __future__ import annotations

import time

import beamline.jax  # noqa: F401
import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp

from emittance_sandbox import load_sample, make_ensemble_state, APERTURE_RADIUS
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
N_ENSEMBLE = 8
BEAM_START = -700.0 * u.mm
DZ = 60.0 * u.mm  # retuned from 15mm -- see track_with_drag docstring for the measured tradeoff


def main() -> None:
    sample = load_sample(size=N_ENSEMBLE)
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

    # The nominal current of the FIRST windowed solenoid (index 0 in the
    # K-nearest-to-z_center selection) -- the design parameter we'll
    # perturb. jphi = current * AMP_TO_JPHI (see build.py); recover the
    # nominal "current" units by dividing back out.
    nominal_current0 = float(base_channel.groups[0].stack.field.jphi[0]) / AMP_TO_JPHI
    print(f"nominal current[0] = {nominal_current0:.4f}")

    def merit(current0):
        """current0 (the design parameter, traced) -> output eigen-emittance
        product for the surviving ensemble after 1 period. Rebuilds the
        solenoid group's jphi array with index 0 overridden via
        eqx.tree_at (proven pattern from the handoff), leaves everything
        else (including the cavity group) untouched.
        """
        sol_group = base_channel.groups[0]
        new_jphi = sol_group.stack.field.jphi.at[0].set(current0 * AMP_TO_JPHI)
        new_field = eqx.tree_at(lambda f: f.jphi, sol_group.stack.field, new_jphi)
        new_stack = eqx.tree_at(lambda s: s.field, sol_group.stack, new_field)
        new_sol_group = StackedField(stack=new_stack)
        channel = BatchedChannel(groups=[new_sol_group, base_channel.groups[1]])

        def track_one(state):
            final_state, _ = track_with_drag(
                channel, state, z0, z1, dz=DZ, include_presswall=True, wedges=wedges,
                window_z=window_z, window_thick=window_thick, rfc0_centers=rfc0_centers,
                n_steps=n_steps, key=None, rtol=1e-3, atol=1e-5,
                aperture_radius=APERTURE_RADIUS,
            )
            return final_state

        final = jax.vmap(track_one)(state0)
        p, t = final.kin.p, final.kin.t
        r = jnp.hypot(p.x, p.y)
        weights = jnp.where(r >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
        cov = weighted_covariance4(p.x, t.x, p.y, t.y, weights)
        e1, e2 = eigen_emittances(cov)
        return e1 * e2

    print("computing jax.jvp (primal + tangent in one forward-mode pass)...")
    t0 = time.time()
    val, tangent = jax.jvp(merit, (nominal_current0,), (1.0,))
    print(f"  merit={float(val):.4f}  d(merit)/d(current0)={float(tangent):.6f}"
          f"  ({time.time()-t0:.1f}s)")

    print("\nfinite-difference cross-check (separate eps values):")
    for eps in [1.0, 0.1]:
        t0 = time.time()
        v_plus = merit(nominal_current0 + eps)
        v_minus = merit(nominal_current0 - eps)
        fd = (v_plus - v_minus) / (2 * eps)
        rel_err = abs(float(fd) - float(tangent)) / abs(float(tangent)) * 100
        print(f"  eps={eps}: FD={float(fd):.6f}  rel_err={rel_err:.2f}%  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
