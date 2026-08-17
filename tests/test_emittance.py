"""Verification tests for hfofo.emittance.

Locks in the checks already done ad hoc while deriving/validating the
eigen-emittance formula (see emittance.py's module docstring for the
derivation) so they don't need re-deriving in a future session.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("beamline")

import hepunits as u

from hfofo.emittance import (
    covariance4,
    eigen_emittances,
    naive_projected_emittances,
    weighted_covariance4,
)
from hfofo.reference_data import find_file


def test_uncoupled_matches_analytic():
    """A block-diagonal (x,px)/(y,py) covariance -- no x-y coupling -- must
    reduce eigen_emittances exactly to the ordinary eps_x = sqrt(det(sigma_x)),
    eps_y = sqrt(det(sigma_y)) computed directly from the 2x2 blocks.
    """
    sigma_x = np.array([[4.0, 1.0], [1.0, 2.0]])
    sigma_y = np.array([[3.0, 0.5], [0.5, 1.5]])
    sigma4 = np.zeros((4, 4))
    sigma4[:2, :2] = sigma_x
    sigma4[2:, 2:] = sigma_y

    eps_x_analytic = np.sqrt(np.linalg.det(sigma_x))
    eps_y_analytic = np.sqrt(np.linalg.det(sigma_y))

    e1, e2 = eigen_emittances(jnp.array(sigma4))
    got = sorted([float(e1), float(e2)], reverse=True)
    expected = sorted([eps_x_analytic, eps_y_analytic], reverse=True)
    assert np.allclose(got, expected, rtol=1e-6)


def test_coupling_increases_naive_product_over_eigen():
    """For a genuinely x-y-coupled covariance, the naive projected-emittance
    product must be >= the true eigen-emittance product (the invariant is
    smaller than the naively-projected estimate whenever there's real
    coupling -- this is the whole reason to use eigen-emittances for a
    helically-coupled channel like HFOFO instead of naive eps_x*eps_y).
    """
    rng = np.random.default_rng(42)
    A = rng.standard_normal((4, 4))
    sigma4 = A @ A.T + 5 * np.eye(4)  # positive definite, generically coupled

    e1, e2 = eigen_emittances(jnp.array(sigma4))
    nx, ny = naive_projected_emittances(jnp.array(sigma4))
    assert float(nx * ny) >= float(e1 * e2)


def test_real_initial_dat_sample_shows_substantial_coupling():
    """On this channel's actual initial.dat sample (not a synthetic toy
    case), the naive projected-emittance product should be close to 2x the
    true eigen-emittance product -- confirms eigen-emittances matter
    numerically here, not just in principle. Skips gracefully if
    initial.dat isn't available in this environment.
    """
    try:
        path = find_file("initial.dat")
    except FileNotFoundError:
        pytest.skip("initial.dat not found in this environment")

    d = np.genfromtxt(path, comments="#")
    mu = d[d[:, 7] == -13]  # PDGid == -13 is mu+
    x, y, px, py, pz = mu[:, 0], mu[:, 1], mu[:, 3], mu[:, 4], mu[:, 5]
    cut = np.abs(pz - 247.5) < 75.0
    mu_cut = mu[cut]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(mu_cut), size=24, replace=False)
    sample = mu_cut[idx]
    sx, spx, sy, spy = sample[:, 0], sample[:, 3], sample[:, 1], sample[:, 4]

    sigma4 = covariance4(jnp.array(sx), jnp.array(spx), jnp.array(sy), jnp.array(spy))
    e1, e2 = eigen_emittances(sigma4)
    nx, ny = naive_projected_emittances(sigma4)
    ratio = float(nx * ny) / float(e1 * e2)
    assert 1.5 < ratio < 2.5, f"expected naive/eigen ratio near 2x, got {ratio:.3f}"


def test_differentiable_via_jvp():
    """eigen_emittances must differentiate correctly via forward-mode AD
    (jax.jvp) -- this is the entire point of avoiding jnp.linalg.eig (see
    the module docstring: eig is not differentiable in JAX for a general/
    non-symmetric matrix, which Sigma @ S is).

    Note: unlike differentiating through the FULL tracking pipeline (diffrax
    + wedges), this algebraic function (trace/matmul/det only) is very
    well-conditioned -- finite-difference agrees with jvp to near machine
    precision even at tiny step sizes, no "pick an appropriately-sized eps"
    caveat needed here specifically (that caveat applies to gradients through
    the ODE solve, not to this function in isolation).
    """
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 4))
    base = jnp.array(A @ A.T + 5 * np.eye(4))
    direction = rng.standard_normal((4, 4))
    direction = jnp.array(direction + direction.T)

    def merit(sigma4):
        e1, e2 = eigen_emittances(sigma4)
        return e1 * e2

    _, tangent = jax.jvp(merit, (base,), (direction,))

    eps = 1e-4
    fd = (merit(base + eps * direction) - merit(base - eps * direction)) / (2 * eps)
    rel_err = abs(float(fd) - float(tangent)) / abs(float(tangent))
    assert rel_err < 1e-3, f"jvp vs FD mismatch: {rel_err*100:.4f}%"


def test_weighted_covariance_hard_mask_matches_subset():
    """A hard 0/1 weight vector must reproduce covariance4's population-
    normalized equivalent computed directly on the kept subset -- i.e.
    weighted_covariance4 with a mask is a faithful (if differently
    normalized) stand-in for "exclude these particles" via boolean
    indexing, just without breaking the JAX trace.
    """
    rng = np.random.default_rng(3)
    n = 20
    x, px, y, py = (jnp.array(rng.standard_normal(n)) for _ in range(4))
    mask = jnp.array([1.0] * 15 + [0.0] * 5)

    cov_masked = weighted_covariance4(x, px, y, py, mask)

    idx = np.array(mask) > 0.5
    cov_subset_population = covariance4(x[idx], px[idx], y[idx], py[idx]) * (14 / 15)
    assert jnp.allclose(cov_masked, cov_subset_population, rtol=1e-6)


def test_weighted_covariance_differentiable_via_jvp():
    """weighted_covariance4 (and eigen_emittances built on top of it) must
    differentiate correctly w.r.t. the WEIGHTS themselves via jax.jvp --
    this is the mechanism that keeps aperture-cut particle exclusion inside
    the JAX trace for gradient-based optimization, instead of dropping to
    NumPy boolean indexing (which breaks jax.jvp for anything downstream).
    """
    rng = np.random.default_rng(3)
    n = 20
    x, px, y, py = (jnp.array(rng.standard_normal(n)) for _ in range(4))

    def merit(w):
        cov = weighted_covariance4(x, px, y, py, w)
        e1, e2 = eigen_emittances(cov)
        return e1 * e2

    base_w = jnp.ones(n)
    direction = jnp.array(rng.standard_normal(n))
    _, tangent = jax.jvp(merit, (base_w,), (direction,))

    eps = 1e-4
    fd = (merit(base_w + eps * direction) - merit(base_w - eps * direction)) / (2 * eps)
    rel_err = abs(float(fd) - float(tangent)) / abs(float(tangent))
    assert rel_err < 1e-3, f"jvp vs FD mismatch: {rel_err*100:.4f}%"


@pytest.mark.slow
def test_output_emittance_gradient_jvp_matches_finite_difference():
    """The FULL milestone-D pipeline (one solenoid's current -> ensemble
    tracking with the aperture cut -> weighted covariance -> output
    eigen-emittance product) must differentiate correctly via jax.jvp,
    matching finite-difference. This is the last link that hadn't been
    checked end to end -- every piece was verified in isolation (jvp
    through diffrax_solve alone, through track_with_drag alone, through
    eigen_emittances/weighted_covariance4 alone) but not all together with
    a real design-parameter perturbation until this test.

    Marked slow: ~130s for the jvp call plus ~65s per FD eps checked (small
    N=6 ensemble, 1 period, to keep this bounded -- NOT representative of
    real optimization scale, which should use a larger ensemble). Run with
    ``-m slow`` to include it.
    """
    import equinox as eqx

    from hfofo.background import cavity_window_positions, rfc0_interior_centers, track_with_drag
    from hfofo.build import AMP_TO_JPHI, build_channel_batched_windowed, build_wedges_windowed
    from hfofo.stacked import BatchedChannel, StackedField
    from hfofo.union_material import build_union_material

    try:
        path = find_file("initial.dat")
    except FileNotFoundError:
        pytest.skip("initial.dat not found in this environment")

    import numpy as np

    d = np.genfromtxt(path, comments="#")
    mu = d[d[:, 7] == -13]
    pz = mu[:, 5]
    cut = np.abs(pz - 247.5) < 75.0
    mu_cut = mu[cut]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mu_cut), size=6, replace=False)
    sample = mu_cut[idx]

    from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
    from beamline.jax.kinematics import MuonStateDz

    BEAM_START = -700.0 * u.mm
    DZ = 15.0 * u.mm
    APERTURE_RADIUS = 200.0 * u.mm

    def make_state(x, y, px, py, pz):
        return MuonStateDz.make(
            position=Cartesian4.make(x=x * u.mm, y=y * u.mm, z=BEAM_START),
            momentum=Cartesian3.make(x=px * u.MeV, y=py * u.MeV, z=pz * u.MeV),
            q=1,
        )

    state0 = jax.vmap(make_state)(
        jnp.array(sample[:, 0]), jnp.array(sample[:, 1]),
        jnp.array(sample[:, 3]), jnp.array(sample[:, 4]), jnp.array(sample[:, 5]),
    )

    lattice = load_lattice_fixture()
    period = lattice.meta.period
    z0, z1 = BEAM_START, BEAM_START + period
    n_steps = int(round(float((z1 - z0) / DZ)))
    z_center = float((z0 + z1) / 2)

    base_channel = build_channel_batched_windowed(lattice, z_center=z_center)
    wedges = build_union_material(build_wedges_windowed(lattice, z_center=z_center))
    window_z, window_thick = cavity_window_positions(lattice.cavities)
    rfc0_centers = rfc0_interior_centers(lattice.cavities)
    nominal_current0 = float(base_channel.groups[0].stack.field.jphi[0]) / AMP_TO_JPHI

    def merit(current0):
        sol_group = base_channel.groups[0]
        new_jphi = sol_group.stack.field.jphi.at[0].set(current0 * AMP_TO_JPHI)
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

        final = jax.vmap(track_one)(state0)
        p, t = final.kin.p, final.kin.t
        r = jnp.hypot(p.x, p.y)
        weights = jnp.where(r >= APERTURE_RADIUS - 1e-3 * u.mm, 0.0, 1.0)
        cov = weighted_covariance4(p.x, t.x, p.y, t.y, weights)
        e1, e2 = eigen_emittances(cov)
        return e1 * e2

    _, tangent = jax.jvp(merit, (nominal_current0,), (1.0,))
    eps = 0.1
    fd = (merit(nominal_current0 + eps) - merit(nominal_current0 - eps)) / (2 * eps)
    rel_err = abs(float(fd) - float(tangent)) / abs(float(tangent))
    assert rel_err < 0.05, f"jvp vs FD mismatch: {rel_err*100:.2f}%"


def load_lattice_fixture():
    from pathlib import Path

    from hfofo.load import load_lattice

    return load_lattice(Path(__file__).parent.parent / "data" / "hfofo.yaml")
