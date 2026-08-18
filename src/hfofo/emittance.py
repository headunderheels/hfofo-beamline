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


def weighted_covariance4(x, px, y, py, weights: jnp.ndarray) -> jnp.ndarray:
    """Weighted 4x4 covariance of (x, px, y, py), differentiable in
    ``weights`` (unlike excluding particles via boolean/NumPy indexing,
    which drops out of the JAX trace and breaks jax.jvp through anything
    downstream).

    Intended use: a soft or hard aperture cut. ``weights = jnp.where(r >
    aperture_radius, 0.0, 1.0)`` reproduces "exclude lost particles from
    the covariance" while staying differentiable -- gradients flow
    correctly on either side of the cut, since the aperture condition
    itself is ordinary JAX control flow (`jnp.where`, not a Python `if` on
    a traced value). Right AT a particle's cut boundary the merit is
    genuinely non-smooth (a particle discretely enters/leaves the sample as
    a design parameter crosses the value that puts that particle exactly at
    the aperture) -- this is an inherent property of hard apertures, not a
    bug in this implementation; a gradient-based optimizer step from
    exactly that point is locally unreliable but nothing else is.

    Uses population (sum(weights)) normalization, not the N-1 (Bessel-
    corrected) normalization ``covariance4``/``jnp.cov`` use by default --
    a fixed, small, weight-independent-in-the-all-ones-case rescaling
    that matters for matching an exact reported number but not for
    gradients or relative comparisons. With all weights equal to 1, this
    differs from ``covariance4`` by exactly the N/(N-1) factor; it does NOT
    reduce to bit-identical output even in that limit -- compare
    the two intentionally if you need to know which convention a given
    reported number uses.
    """
    w = weights
    wsum = jnp.sum(w)
    phase = jnp.stack([x, px, y, py], axis=0)  # (4, N)
    mean = jnp.sum(phase * w[None, :], axis=1) / wsum  # (4,)
    centered = phase - mean[:, None]
    return (centered * w[None, :]) @ centered.T / wsum


# ---------------------------------------------------------------------------
# 6D (full transverse + longitudinal) eigen-emittances
# ---------------------------------------------------------------------------
# Same Williamson's-theorem approach as the 4D case above, generalized to
# three canonical pairs instead of two. Convention: (x, px, y, py, ct, E) --
# the longitudinal pair matches this codebase's existing position/momentum
# representation directly (Tangent[Cartesian4]'s p=(x,y,z,ct) and
# t=(px,py,pz,E), with z the independent tracking variable rather than a
# phase-space coordinate; ct and E are the natural remaining canonical pair).
#
# For a positive-definite 6x6 covariance Sigma and the canonical symplectic
# form S (block-diagonal, three [[0,1],[-1,0]] blocks), the eigenvalues of
# M = Sigma @ S come in three conjugate pairs +-i*eps1, +-i*eps2, +-i*eps3.
# tr(M) = 0 (same cyclic-trace argument as the 4D case: Sigma symmetric, S
# antisymmetric). For THIS eigenvalue structure (symmetric about 0), the
# characteristic polynomial is also EVEN in lambda, so tr(M^3) = 0 too (each
# +-i*eps pair contributes (i eps)^3 + (-i eps)^3 = 0). Using Newton's
# identities (relating power sums p_k=tr(M^k) to elementary symmetric
# polynomials e_k of the eigenvalues, with e1=p1=0 substituted in):
#     e2 = -p2/2               = -tr(M@M)/2
#     e3 = p3/3                = 0   (confirms the "even" structure, not used)
#     e4 = p2^2/8 - p4/4        = tr(M@M)^2/8 - tr(M@M@M@M)/4
# With a=eps1^2, b=eps2^2, c=eps3^2: a+b+c=e2, ab+ac+bc=e4, abc=det(M) (the
# standard elementary-symmetric/Vieta relation for a monic cubic's roots).
# Solved via the trigonometric (Cardano/Viete) closed form for a depressed
# cubic with three real roots -- no eigendecomposition anywhere, so this
# differentiates via ordinary jax.jvp/jax.grad exactly like the 4D case.
#
# Verified (see tests/test_emittance.py): exact match against the analytic
# uncoupled case (three independent 2x2 blocks reduce to the three ordinary
# 2D emittances); exact match against an independent, non-differentiable
# oracle (plain numpy.linalg.eig) on random genuinely-coupled 6x6
# covariances; forward-mode AD matches finite-difference at an
# appropriately-sized step (same FD-conditioning caveat as the 4D case --
# see test_emittance.py's docstring for why eps=1e-2 is the right FD step,
# not a smaller one).
_S6 = jnp.array(
    [
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ]
)


def eigen_emittances_6d(sigma6: jnp.ndarray) -> jnp.ndarray:
    """The three eigen-emittances of a 6D phase space (x, px, y, py, ct, E),
    as an array [eps1, eps2, eps3] with eps1 >= eps2 >= eps3 (no ordering is
    physically privileged; this is just the order the cubic solve produces).
    Fully differentiable via jax.jvp -- see the module note above.
    """
    S = _S6.astype(sigma6.dtype)
    M = sigma6 @ S
    p2 = jnp.trace(M @ M)
    p4 = jnp.trace(M @ M @ M @ M)
    c2 = -p2 / 2.0
    c4 = p2**2 / 8.0 - p4 / 4.0
    c6 = jnp.linalg.det(M)
    # cubic in t=eps^2: t^3 - c2 t^2 + c4 t - c6 = 0. Depress via t = s + c2/3.
    p = c4 - c2**2 / 3.0
    q = -2 * c2**3 / 27.0 + c2 * c4 / 3.0 - c6
    safe_p = jnp.where(p < 0.0, p, -1e-30)  # guard sqrt(-3/p); p<0 for 3 real roots
    arg = jnp.clip((3 * q) / (2 * safe_p) * jnp.sqrt(-3.0 / safe_p), -1.0, 1.0)
    theta = jnp.arccos(arg)
    roots = jnp.stack(
        [
            2 * jnp.sqrt(-safe_p / 3.0) * jnp.cos(theta / 3.0 - 2 * jnp.pi * k / 3.0) + c2 / 3.0
            for k in range(3)
        ]
    )
    return jnp.sort(jnp.sqrt(jnp.maximum(roots, 0.0)))[::-1]


def covariance6(x, px, y, py, ct, E) -> jnp.ndarray:
    """6x6 covariance of (x, px, y, py, ct, E) -- the 6D analogue of
    ``covariance4``. See that function's docstring for the convention note.
    """
    phase = jnp.stack([x, px, y, py, ct, E], axis=0)  # (6, N)
    return jnp.cov(phase)


def weighted_covariance6(x, px, y, py, ct, E, weights: jnp.ndarray) -> jnp.ndarray:
    """Weighted 6x6 covariance -- the 6D analogue of ``weighted_covariance4``.
    See that function's docstring for the aperture-cut usage pattern and the
    population- vs Bessel-normalization note (same convention here: population).
    """
    w = weights
    wsum = jnp.sum(w)
    phase = jnp.stack([x, px, y, py, ct, E], axis=0)  # (6, N)
    mean = jnp.sum(phase * w[None, :], axis=1) / wsum
    centered = phase - mean[:, None]
    return (centered * w[None, :]) @ centered.T / wsum
