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

from hfofo.emittance import covariance4, eigen_emittances, naive_projected_emittances
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
