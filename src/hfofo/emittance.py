"""Eigenmode (normal-mode) transverse emittances, computed without eigendecomposition.

For a helically-coupled channel like HFOFO (tilted/rotating solenoids couple
the x and y planes), naive projected emittances (eps_x = sqrt(det(sigma_x)),
eps_y = sqrt(det(sigma_y)) from the uncoupled 2x2 blocks) ignore x-y
correlations and can substantially overstate the true invariant phase-space
volume. The correct quantities are the eigen-emittances from the symplectic
decomposition of the full 4x4 transverse covariance matrix.

The textbook approach computes these via the eigenvalues of Sigma @ S (S the
symplectic form) -- but jax.numpy.linalg.eig for a general (non-symmetric)
matrix is NOT differentiable in JAX at all (confirmed directly: it raises
NotImplementedError for both forward- and reverse-mode AD). Sigma @ S is not
symmetric, so this path is a dead end for anything that needs gradients.

This module instead uses Williamson's theorem directly: for a positive-
definite Sigma and antisymmetric symplectic S, the eigenvalues of M = Sigma@S
are purely imaginary pairs +-i*eps1, +-i*eps2. Since tr(M) = 0 always (a short
cyclic-trace argument: tr(Sigma@S) = tr((Sigma@S)^T) = tr(S^T @ Sigma^T)
= tr(-S @ Sigma) = -tr(Sigma@S) [cyclic property] => tr(Sigma@S) = 0), the
characteristic polynomial of M reduces to lambda^4 + c2*lambda^2 + det(M) = 0,
giving:

    eps1^2 + eps2^2 = c2 = -trace(M @ M) / 2
    eps1 * eps2      = sqrt(det(M))

Both sides use only trace, matmul, and det -- all differentiable via ordinary
forward-mode AD (jax.jvp), with no eigendecomposition anywhere.

Verified (see tests/test_emittance.py):
- Exact match against the known analytic uncoupled case (block-diagonal Sigma
  reduces to ordinary eps_x, eps_y).
- Correctly diverges from naive projected emittances under real coupling
  (confirmed on this channel's actual initial.dat sample: naive product is
  ~2x the true eigen-emittance product).
- Differentiates correctly via jax.jvp, matching finite-difference at an
  appropriately-sized step (a naive eps=1e-4 FD check showed ~1.4% mismatch
  purely from floating-point cancellation noise, NOT an AD bug -- confirmed
  by checking eps=1e-2 gives 0.015% match and the mismatch grows as eps
  shrinks further, the textbook FD-precision-floor signature).

Convention: phase-space coordinates ordered (x, px, y, py); sigma4 is the
4x4 covariance matrix in that order.
"""

from __future__ import annotations

import jax.numpy as jnp

from beamline.jax.types import SFloat

# Canonical symplectic form for (x, px, y, py) ordering.
_S = jnp.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0, 0.0],
    ]
)


def eigen_emittances(sigma4: jnp.ndarray) -> tuple[SFloat, SFloat]:
    """The two eigen-emittances (eps1, eps2) of a 4D transverse phase space.

    ``sigma4`` is the 4x4 covariance matrix of (x, px, y, py), in that order
    (e.g. from ``np.cov`` on an (N, 4) array of particle coordinates, or the
    JAX equivalent). Returns (eps1, eps2) with eps1 >= eps2 by construction
    of the quadratic solve below (no ordering is physically privileged; sort
    afterward if you need a specific convention).

    Fully differentiable via jax.jvp (forward-mode AD) -- see the module
    docstring for why this avoids jnp.linalg.eig entirely and does not
    support jax.grad (reverse-mode) any better or worse than the rest of
    this pipeline; use forward-mode for anything built on top of this.
    """
    S = _S.astype(sigma4.dtype)
    M = sigma4 @ S
    c2 = -jnp.trace(M @ M) / 2.0
    p = jnp.sqrt(jnp.maximum(jnp.linalg.det(M), 0.0))
    disc = jnp.maximum(c2**2 - 4 * p**2, 0.0)
    t1 = (c2 + jnp.sqrt(disc)) / 2.0
    t2 = (c2 - jnp.sqrt(disc)) / 2.0
    return jnp.sqrt(jnp.maximum(t1, 0.0)), jnp.sqrt(jnp.maximum(t2, 0.0))


def naive_projected_emittances(sigma4: jnp.ndarray) -> tuple[SFloat, SFloat]:
    """The naive (uncoupled) projected emittances eps_x, eps_y, ignoring any
    x-y correlation in sigma4. Provided only for comparison against
    ``eigen_emittances`` -- NOT the physically correct invariant for a
    coupled channel; see the module docstring.
    """
    eps_x = jnp.sqrt(jnp.maximum(jnp.linalg.det(sigma4[:2, :2]), 0.0))
    eps_y = jnp.sqrt(jnp.maximum(jnp.linalg.det(sigma4[2:, 2:]), 0.0))
    return eps_x, eps_y


def covariance4(x, px, y, py) -> jnp.ndarray:
    """4x4 covariance matrix of (x, px, y, py) from equal-length 1D arrays
    of per-particle coordinates (e.g. an ensemble's positions/momenta at a
    given z). Thin wrapper around jnp.cov for the (x,px,y,py) convention
    used throughout this module.
    """
    phase = jnp.stack([x, px, y, py], axis=0)  # (4, N)
    return jnp.cov(phase)
