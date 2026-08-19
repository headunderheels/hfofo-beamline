"""GH2 world-material background drag, combined with the wedge absorbers.

``track_v7.in`` sets ``param worldMaterial=GH2`` with
``material GH2 Z=1 A=1.01 density=0.014`` (20% of liquid-hydrogen density) --
this fills the *entire* simulated world, not just the 171 discrete wedges.
G4Beamline applies continuous ionization energy loss to the beam everywhere
it travels. The channel's RF gradients and per-cavity ``timeOffset`` phasing
were tuned in G4BL against a system that includes this drag; a model that
only accounts for the wedges is missing a large, systematic energy loss
(~25 MeV/period, ~768 MeV over the full 31-period channel per
``beamline``'s own Bethe-Bloch calculation at these momenta -- an order of
magnitude more than the wedges contribute) and picks up a large net energy
surplus relative to the real channel.
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import hepunits as u
import jax
import jax.numpy as jnp
import jax.random as jr

from beamline.jax.absorber.material import DensityCorrection, Material, StragglingParams
from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.kinematics import MuonStateDz, ParticleState
from beamline.jax.types import SBool, SFloat

from hfofo.union_material import UnionMaterial

# track_v7.in: `material GH2 Z=1 A=1.01 density=0.014 # 20% of LH2`.
# mean_excitation: standard PDG value for hydrogen gas (19.2 eV); the density
# here is a custom (very dilute) fill, not the tabulated STP value, but I is
# roughly density-independent. plasma_energy DOES depend on density (electron
# density scales with it) -- computed via the standard
# hbar*omega_p = 28.816 eV * sqrt(rho[g/cm^3] * Z/A) scaling.
_GH2_Z, _GH2_A_GMOL = 1, 1.01
_GH2_DENSITY = 0.014 * u.g / u.cm3
GH2 = Material(
    name="GH2 (worldMaterial, 20% LH2 density)",
    Z=_GH2_Z,
    mass=_GH2_A_GMOL * u.g / u.mol,
    density=_GH2_DENSITY,
    mean_excitation=19.2 * u.eV,
    plasma_energy=28.816 * (0.014 * (_GH2_Z / _GH2_A_GMOL)) ** 0.5 * u.eV,
    is_atomic=True,
    # Density effect is negligible for a gas this dilute at these momenta;
    # x0 set high enough that the correction never activates in our regime
    # (delta0=0 is the only value that matters below x0 -- see
    # DensityCorrection.__call__).
    density_correction=DensityCorrection(C=0.0, x0=3.0, x1=100.0, a=0.0, k=0.0, delta0=0.0),
)

# `place presswall z=850` (abs_place7_31.txt) + `tubs presswall length=4
# outerRadius=360 material=Stainless316` (track_v7.in) -- a single, one-time
# 4mm Stainless316 wall the beam crosses once, at z=850mm. Approximated with
# iron-like properties (Z=26, A=55.85, standard steel-alloy mean excitation);
# 316 stainless is Fe-Cr-Ni, close enough in Z/A for this thin a crossing.
PRESSWALL_Z = 850.0 * u.mm
PRESSWALL_THICKNESS = 4.0 * u.mm
STAINLESS_316 = Material(
    name="Stainless316 (presswall, iron-like approximation)",
    Z=26,
    mass=55.85 * u.g / u.mol,
    density=8.00 * u.g / u.cm3,
    mean_excitation=300.0 * u.eV,
    plasma_energy=28.816 * (8.00 * (26 / 55.85)) ** 0.5 * u.eV,
    is_atomic=True,
    density_correction=DensityCorrection(C=0.0, x0=3.0, x1=100.0, a=0.0, k=0.0, delta0=0.0),
)

# Each pillbox cavity has two thin Be "win2" windows (upstream and downstream,
# at z = cavity_z -+ (innerLength/2 + win2Thick/2)), covering r=[0, irisRadius]
# -- i.e. directly in the beam path (win1, r=[0, win1OuterRadius], is disabled
# in the deck via win1OuterRadius=0, so win2 fills all the way to r=0). wall/
# pipe are annular OUTSIDE irisRadius/innerRadius and do NOT intersect the
# beam -- see BLCMDpillbox.cc's construct(): win2 uses
# G4Tubs(win1OuterRadius, irisRadius, ...) while wall/pipe start at
# irisRadius+collarRadialThick / innerRadius respectively. win2Thick differs
# by cavity variant: 0.12mm (RFC0/RFC), 0.1mm (RFC1), 0.07mm (RFC2).
BERYLLIUM = Material(
    name="Be (cavity win2 windows)",
    Z=4,
    mass=9.012 * u.g / u.mol,
    density=1.848 * u.g / u.cm3,
    mean_excitation=63.7 * u.eV,
    plasma_energy=28.816 * (1.848 * (4 / 9.012)) ** 0.5 * u.eV,
    is_atomic=True,
    density_correction=DensityCorrection(C=0.0, x0=3.0, x1=100.0, a=0.0, k=0.0, delta0=0.0),
)
_WIN2THICK_BY_VARIANT = {"RFC0": 0.12, "RFC": 0.12, "RFC1": 0.1, "RFC2": 0.07}
_CAVITY_INNER_LENGTH = 249.0 * u.mm
_RFC0_HALF_LENGTH = _CAVITY_INNER_LENGTH / 2  # 124.5mm


def rfc0_interior_centers(cavities) -> jnp.ndarray:
    """z-centers of every RFC0 cavity's interior (the only variant that does
    NOT set ``cavityMaterial=GH2`` in the deck -- RFC/RFC1/RFC2 all do. Real
    G4BL has vacuum, not GH2, inside these 5 entrance-region cavities;
    confirmed by an ~7.3 MeV early-channel offset between our model and
    G4BL's true reference trace that matches 5 * innerLength(249mm) worth of
    wrongly-applied GH2 loss to within rounding.
    """
    return jnp.array([c.z * u.mm for c in cavities if c.variant == "RFC0"])


def cavity_window_positions(cavities) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (z_centers, thicknesses) for every cavity's two Be windows."""
    zs, thicks = [], []
    for c in cavities:
        t = _WIN2THICK_BY_VARIANT[c.variant] * u.mm
        half = _CAVITY_INNER_LENGTH / 2 + t / 2
        zs.append(c.z * u.mm - half)
        thicks.append(t)
        zs.append(c.z * u.mm + half)
        thicks.append(t)
    return jnp.array(zs), jnp.array(thicks)


def cavity_window_positions_windowed(
    cavities, z_center: float, k_cavities: int = 44
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Like ``cavity_window_positions``, but only the ``k_cavities`` nearest
    cavities' Be windows (see ``build.py``'s module-level windowing note --
    same principle, applied here for the first time: ``track_with_drag``'s
    per-step loop previously vmapped over every cavity's windows regardless
    of location, 758 entries total for the full lattice, even though a
    single 60mm step can cross at most one or two of them). Measured
    contribution of this specific vmap: negligible runtime effect (~1%,
    within noise) but a real ~25% reduction in one-time compile cost (8.4s
    vs 11.1s in a bare single-particle, one-period test) -- worth doing
    since it's free and consistent with the existing solenoid/cavity/wedge
    windowing, not because it's the dominant cost.

    ``k_cavities`` defaults to match ``K_CAVITIES_LOCAL`` (the windowed EM
    channel's own cavity count) since callers windowing the channel to the
    same ``z_center`` are already restricting to that same population of
    cavities -- keep the two consistent rather than picking an unrelated k.

    NOT yet applied to track_full_channel.py: that script's window_z/
    window_thick are currently built once, globally, outside its per-period
    loop (unlike channel/wedges, which it already rebuilds per period as
    ordinary traced arguments specifically so one compiled function is
    reused across periods rather than retracing every one). Windowing this
    too would need the same per-period-rebuild-as-traced-argument treatment
    to avoid regressing that reuse -- a real but separable follow-up, not
    done here to keep this change scoped and low-risk.
    """
    from hfofo.build import _nearest

    return cavity_window_positions(_nearest(cavities, z_center, k_cavities))


class EverywhereMaterial(MaterialVolume):
    """A material filling all of space (the ``worldMaterial`` background gas).

    KNOWN BROKEN with ``stochastic_solve``: its kick-application logic in
    ``substep`` decides whether a kick applies purely from the *sign* of
    ``signed_time_to_boundary`` (positive = outside, kick skipped), never
    calling ``contains()`` at all. Since this always reports "inside"
    (``contains()`` is always True) but ``BackgroundAndWedges.
    signed_time_to_boundary`` below reports the *wedges'* signed distance
    (positive whenever not inside a wedge), stochastic_solve silently never
    applies the background kick outside wedge crossings -- confirmed
    empirically (a full-channel energy profile with this wired in was
    indistinguishable from the wedges-only profile). Properly supporting an
    always-present material this way would need a real change to
    ``stochastic_solve``'s substep logic, not just a material implementation.

    ``track_with_drag`` in this module is the actual working implementation:
    a small deterministic step loop (diffrax mini-step + analytic loss),
    sidestepping this class and ``BackgroundAndWedges`` entirely. Kept here
    for context/history, not for use -- prefer ``track_with_drag``.
    """

    material: Material = eqx.field(static=True)
    char_length: SFloat = 100.0 * u.mm

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # Always inside, boundary infinitely far away.
        return -jnp.inf

    def characteristic_length(self) -> SFloat:
        return self.char_length

    def interaction_params(
        self, state: ParticleState, thickness: SFloat
    ) -> StragglingParams:
        return self.material.straggling_params(state, thickness)


class BackgroundAndWedges(MaterialVolume):
    """The GH2 background gas plus the 171 wedge absorbers, combined.

    See ``EverywhereMaterial``'s docstring: this combination does NOT work
    correctly with ``stochastic_solve`` (the background kick silently never
    fires outside a wedge crossing). Kept for context/history -- use
    ``track_with_drag`` instead, which sidesteps this entirely with a small
    deterministic step loop.
    """

    background: EverywhereMaterial
    wedges: UnionMaterial

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        return self.wedges.signed_time_to_boundary(ray)

    def characteristic_length(self) -> SFloat:
        return self.background.characteristic_length()

    def interaction_params(
        self, state: ParticleState, thickness: SFloat
    ) -> StragglingParams:
        bg = self.background.interaction_params(state, thickness)
        wg = self.wedges.interaction_params(state, thickness)
        fields = dataclasses.fields(StragglingParams)
        summed = {f.name: getattr(bg, f.name) + getattr(wg, f.name) for f in fields}
        return StragglingParams(**summed)


def build_background_and_wedges(wedges: UnionMaterial) -> BackgroundAndWedges:
    """Combine the GH2 world-material background with the wedge union.

    See ``EverywhereMaterial``'s docstring -- the result does NOT apply
    background kicks correctly under ``stochastic_solve``. Use
    ``track_with_drag`` instead.
    """
    return BackgroundAndWedges(background=EverywhereMaterial(material=GH2), wedges=wedges)


def apply_energy_loss(state: MuonStateDz, dE: SFloat) -> MuonStateDz:
    """Reduce a muon's total energy by ``dE``, preserving momentum direction.

    Clamps so E never drops below ``1.01 * MUON_MASS`` -- compounding loss
    mechanisms (GH2 + wedges) landing right at a deep RF-phase-oscillation
    trough can otherwise push the analytic correction below rest mass, which
    is unphysical and also numerically degenerate for z-parametrized tracking
    (beta->0 means dz/dct->0, needing unbounded steps to make any z-progress
    -- this is what actually breaks diffrax's max_steps, not a field/geometry
    issue). This is a real accuracy caveat at those specific troughs, not
    just a numerical nicety: it means the model is doing something wrong near
    its deepest energy dips, worth revisiting rather than treating the clamp
    as a final answer.
    """
    MUON_MASS = 105.6583715 * u.MeV
    p = state.kin.t
    p3 = jnp.array([p.x, p.y, p.z])
    pmag = jnp.linalg.norm(p3)
    E = p.ct
    E_new = jnp.maximum(E - dE, 1.01 * MUON_MASS)
    m2 = E**2 - pmag**2
    pmag_new = jnp.sqrt(jnp.maximum(E_new**2 - m2, 1e-6))
    scale = pmag_new / pmag
    new_t = Cartesian4.make(x=p.x * scale, y=p.y * scale, z=p.z * scale, ct=E_new)
    return MuonStateDz(kin=Tangent(p=state.kin.p, t=new_t), q=state.q)


# ---------------------------------------------------------------------------
# Multiple Coulomb scattering (Highland formula)
# ---------------------------------------------------------------------------
# beamline's absorber physics models only energy loss (mean + Landau
# straggling) -- see absorber/volume.py's own module docstring: "the
# stochastic interaction (energy straggling now, multiple scattering later)".
# MCS is the dominant transverse-emittance-growth mechanism in a real
# ionization-cooling channel (it's the physical reason such channels need
# continuous RF reacceleration at all: energy loss cools, MCS reheats). Its
# complete absence was flagged as a leading candidate for the tune-mismatch
# beat seen against G4BL's true reference trace (design doc S9 residual
# investigation) -- added here as an app-side kick, applied once per
# track_with_drag step from the combined variance of every material crossed
# in that step (Highland's thin-scatterer variances add linearly).
#
# Radiation lengths (X0): PDG mass radiation length divided by density, using
# the SAME density values already assumed above for GH2/STAINLESS_316/
# BERYLLIUM, plus a standard solid-LiH density for the wedges. Not derived
# from beamline's own Material class -- its density/mass fields are stored in
# an internal representation not straightforwardly convertible back to
# g/cm^3 without reading its source in detail; using independent PDG-table
# values here instead, the same approach background.py already takes for
# GH2/STAINLESS_316/BERYLLIUM's other constants (mean_excitation etc).
_LIH_DENSITY = 0.820 * u.g / u.cm3  # PDG-standard value for solid LiH
X0_GH2 = (63.05 * u.g / u.cm2) / _GH2_DENSITY  # hydrogen; mass X0 = 63.05 g/cm^2
X0_STAINLESS_316 = (13.84 * u.g / u.cm2) / (8.00 * u.g / u.cm3)  # iron-like
X0_BERYLLIUM = (65.19 * u.g / u.cm2) / (1.848 * u.g / u.cm3)
X0_LIH = (79.62 * u.g / u.cm2) / _LIH_DENSITY


def highland_theta0_squared(state: MuonStateDz, thickness: SFloat, X0: SFloat) -> SFloat:
    """Squared RMS projected multiple-scattering angle [rad^2] (PDG/Highland).

    theta0 = (13.6 MeV / (beta*p)) * sqrt(x/X0) * [1 + 0.038 ln(x/(X0*beta^2))]
    for a singly-charged particle (our muons, z=1). p [MeV], beta from the
    current state; x the traversed thickness, X0 the material's radiation
    length. Returns the SQUARED angle so multiple simultaneous crossings in
    one step can be combined by summing (independent-thin-scatterer variances
    add) before drawing a single kick. Guards thickness<=0 (returns 0, not
    NaN from log(0)/sqrt(negative)).
    """
    t = state.kin.t
    p = jnp.sqrt(t.x**2 + t.y**2 + t.z**2)
    beta = p / t.ct
    safe_x = jnp.where(thickness > 0.0, thickness, 1e-9 * u.mm)
    ratio = safe_x / X0
    theta0 = (13.6 * u.MeV / (beta * p)) * jnp.sqrt(ratio) * (
        1.0 + 0.038 * jnp.log(ratio / beta**2)
    )
    return jnp.where(thickness > 0.0, theta0**2, 0.0)


def apply_scattering_kick(state: MuonStateDz, theta0: SFloat, key) -> MuonStateDz:
    """Apply one Highland-style MCS kick: independent Gaussian draws for the
    two projected scattering angles (the standard small-angle treatment),
    then rescale pz so the momentum magnitude (and hence kinetic energy) is
    exactly conserved -- MCS changes direction, not energy. Rescaling exactly
    rather than relying on the small-angle approximation to hold matters here
    because many small kicks compound over hundreds of steps.
    """
    t = state.kin.t
    p = jnp.sqrt(t.x**2 + t.y**2 + t.z**2)
    kx, ky = jr.split(key)
    dthx = theta0 * jr.normal(kx)
    dthy = theta0 * jr.normal(ky)
    new_x = t.x + p * dthx
    new_y = t.y + p * dthy
    new_z_sq = p**2 - new_x**2 - new_y**2
    new_z = jnp.sign(t.z) * jnp.sqrt(jnp.maximum(new_z_sq, 0.0))
    new_t = Cartesian4.make(x=new_x, y=new_y, z=new_z, ct=t.ct)
    return MuonStateDz(kin=Tangent(p=state.kin.p, t=new_t), q=state.q)


def track_with_drag(
    field,
    start: MuonStateDz,
    z0: SFloat,
    z1: SFloat,
    dz: SFloat = 60.0 * u.mm,
    include_presswall: bool = True,
    wedges: UnionMaterial | None = None,
    window_z: jnp.ndarray | None = None,
    window_thick: jnp.ndarray | None = None,
    rfc0_centers: jnp.ndarray | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    n_steps: int | None = None,
    key: jax.Array | None = None,
    aperture_radius: SFloat | None = None,
    forward_mode: bool = True,
):
    """Track a muon through ``field`` (solenoids+RF) with GH2 background drag,
    the ``presswall``, cavity Be windows, and (optionally) the LiH wedge
    absorbers folded in self-consistently, one small ``dz`` step at a time: a
    field-only ``diffrax_solve`` mini-step, followed by analytic energy-loss
    kicks and (if ``key`` is given) a multiple-Coulomb-scattering kick, for
    that same step. Not implemented via ``stochastic_solve`` -- that
    machinery's kick-detection is keyed off the *sign* of
    ``signed_time_to_boundary``, which doesn't have a consistent meaning for a
    material that's always present (see ``BackgroundAndWedges``'s docstring
    history / the module-level notes); a small-step deterministic loop sidesteps
    that entirely.

    ``dz`` (default 60mm here, though the default in scripts/optimize_taper.py
    has since been reverted to 15mm -- see below): retuned after measuring the
    tradeoff directly, not by guessing. On a stable single-particle trajectory
    (1 lattice period): ~2.3x faster (2.1s -> 0.9s post-compile) with a
    ~0.25 MeV (~0.08%) difference in final energy, and forward-/reverse-mode
    AD remained EXACTLY self-consistent at either value (0.000% relative
    difference between jax.jvp and finite-difference at both dz=15mm and
    dz=60mm). Compile time is essentially unaffected by dz (jax.lax.scan
    compiles one step body regardless of how many times it iterates) --
    the win is purely in runtime, which is what dominates repeated
    gradient/ensemble evaluations. Also re-verified against the SPECIFIC
    unstable particle that motivated ``aperture_radius`` below (see
    docs/SESSION_HANDOFF_2026-08-17_aperture_cut.md): the freeze still
    triggers correctly and produces a sane result at dz=60mm, just with a
    slightly larger one-step overshoot past the aperture threshold (~9mm vs
    ~1mm at dz=15mm) -- nowhere near the catastrophic failure mode that
    document found from loosening *tolerance* instead (100+ km off-axis,
    negative energy). Do NOT use a large dz as a substitute for the
    aperture cut, and do not loosen rtol/atol to work around an unstable
    trajectory -- both of those were already tried and rejected for good
    reasons documented there; dz is a genuinely different, safe knob only
    because the aperture freeze re-checks every step regardless of its size.

    IMPORTANT CORRECTION, found the hard way: all of the above verification
    was against ONE particle over ONE period near the channel start -- NOT
    against a full multi-period, larger-ensemble run. When scripts/
    optimize_taper.py (multi-period, N=24 ensemble) was actually run at
    dz=60mm across 5 periods, it hit "the maximum number of solver steps was
    reached" -- a real failure, not a theoretical risk. Reverting to dz=15mm
    fixed it cleanly. Root cause not fully isolated (plausibly: a *different*
    particle than the one originally tested becomes marginal partway through
    a longer run, and the larger per-step aperture overshoot at dz=60mm is
    enough to push it into a harder-to-resolve regime before the freeze
    catches it) -- but the practical conclusion is firm regardless of the
    exact mechanism: dz=60mm's safety verification does NOT generalize to
    multi-period/larger-ensemble runs, and scripts/optimize_taper.py has
    reverted its own default to 15mm accordingly. The other scripts using
    this function (track_full_channel.py, gradient_check.py,
    emittance_sandbox.py, optimize_lattice.py) still default to 60mm and have
    NOT been re-tested at a comparably large multi-period ensemble scale --
    treat that as an open risk for them too, not a cleared one, until
    someone actually runs that test rather than assuming the single-particle
    verification above still applies.

    ``window_z``/``window_thick`` come from ``cavity_window_positions`` --
    pass both together, or neither (skips Be windows). ``rfc0_centers`` comes
    from ``rfc0_interior_centers`` -- excludes GH2 loss inside RFC0 cavities
    specifically (the deck's only cavity variant without
    ``cavityMaterial=GH2``); omit to apply GH2 everywhere uniformly.

    ``key``: if given, applies one multiple-Coulomb-scattering kick per step
    (Highland formula; see ``highland_theta0_squared``/``apply_scattering_kick``),
    combining the variance from every material crossed in that step (GH2,
    presswall, windows, wedge) before drawing a single Gaussian kick. Pass
    ``None`` to skip MCS entirely (energy-loss-only, matching this function's
    original behavior). ``beamline``'s own absorber physics does not model
    MCS at all (see the module-level note above ``highland_theta0_squared``);
    this is an app-side addition.

    ``aperture_radius``: if given, a particle whose transverse radius
    (sqrt(x^2+y^2)) exceeds this is treated as LOST -- its state is frozen
    from that point on (no further field evolution or material kicks; ``x``,
    ``y``, ``ct``, ``px``, ``py``, ``pz`` all held at their last valid values,
    ``z`` still advances to track the requested grid). This models the real
    channel's aperture/collimation (irises at 200-300mm depending on cavity
    variant, the abtube kill volume at r>500mm, wedge physical extents --
    none of which this simplified model otherwise represents at all).

    Without this, a genuinely unstable/lost particle (confirmed by direct
    inspection: transverse radius doubling every ~150-200mm, reaching
    hundreds of mm within a fraction of one period -- not a numerical
    artifact) drives the ODE into a regime the default tolerance correctly
    refuses to resolve (diffrax's max_steps safety check fires, as it
    should -- see track_full_channel.py's design-doc note on why that check
    isn't disabled). Loosening tolerance to avoid the crash does NOT fix
    this: it lets the solver push through into a wildly nonphysical state
    (verified directly: one such particle, from an entirely unremarkable
    starting point, ended up 100+ meters off-axis with negative kinetic
    energy after just one period) that then silently corrupts any ensemble
    statistic (covariance/emittance) computed from it. Freezing at the
    aperture is the physically-motivated fix: mirrors what the real
    channel's collimators do (remove the particle), rather than either
    crashing on it or quietly averaging in a nonphysical trajectory.

    Wedge crossing thickness uses the exact geometric formula
    ``stochastic_solve``'s substep uses (signed-distance at both step
    endpoints plus true displacement), not a containment guess -- both the
    energy-loss and (if enabled) the scattering-kick calculations for the
    wedge use this same thickness.

    ``n_steps`` must be a concrete Python int (not a traced value) when calling
    this under ``jax.jit`` -- pass it explicitly if ``z0``/``z1`` are traced;
    otherwise it's derived from ``z0``/``z1``/``dz`` directly.

    Returns (final_state, (z, x, y, ct, px, py, pz, E)) -- one row per dz step
    (not including the start point). If ``aperture_radius`` is given, also
    check the returned ``final_state``'s radius against it -- a particle at
    exactly ``aperture_radius`` may be a live particle or a frozen lost one;
    compare against the trajectory's earlier rows if you need to distinguish.

    ``forward_mode``: passed straight through to the internal
    ``diffrax_solve`` mini-step call, controlling which adjoint diffrax uses
    for AD -- ``ForwardMode()`` if True (the default), ``RecursiveCheckpointAdjoint()``
    (diffrax's own default, built via checkpointing) if False. This determines
    which JAX AD transform works, NOT whether this pipeline is differentiable
    at all -- an earlier round of this project concluded "jax.grad fails on
    this pipeline, always use jax.jvp," which was too broad. The precise
    statement, verified directly both ways on this exact function (bare
    channel and the full wedges+GH2 driver, matching to 6 significant figures
    either way): ``forward_mode=True`` pairs with forward-mode AD
    (``jax.jvp``/``jax.jacfwd``) and BREAKS under ``jax.grad``/``jax.vjp``;
    ``forward_mode=False`` pairs with reverse-mode AD (``jax.grad``) and
    should not be used with ``jax.jvp`` (untested, and there is no reason to
    -- ``ForwardMode`` exists precisely because it is cheaper per-parameter
    for forward-mode).

    Cost tradeoff: measured on a bare single-particle, no-aperture, no-emittance
    merit function (one design parameter, one lattice period): forward-mode
    ~66s, reverse-mode ~119s (reverse-mode ~1.8x slower there -- checkpointing
    recomputes parts of the forward pass during the backward pass). Measured
    again on the REAL optimize_lattice.py pipeline (N=6 ensemble, aperture
    cut, eigen-emittance merit, K=3 parameters): forward-mode ~138s,
    reverse-mode ~189s -- reverse-mode is STILL slower here (~1.37x), not
    faster. The naive theoretical argument (forward-mode cost scales linearly
    with parameter count K; reverse-mode cost is roughly independent of K, so
    there should be some crossover K beyond which reverse-mode wins) is
    probably still right in direction, but an earlier version of this
    docstring asserted a specific crossover ("already around K=2") extrapolated
    from the simple single-particle numbers above -- that specific claim does
    NOT hold on the real, more complex pipeline and has been removed. The
    crossover K for a given merit function is an empirical question, not a
    generic constant: always run scripts/optimize_lattice.py's
    ``--verify-consistency`` (which also reports both modes' timing) at
    whatever K and ensemble size you actually intend to use, rather than
    assuming a crossover point from a different problem's numbers.
    """
    from beamline.jax.integrate.propagate import diffrax_solve

    if n_steps is None:
        n_steps = int(round(float((z1 - z0) / dz)))

    def one_step(state, i, step_key, frozen_state, is_lost):
        zc0 = z0 + i * dz
        zc1 = zc0 + dz
        if aperture_radius is not None:
            # Feed the FROZEN (already-known-safe) state into diffrax for an
            # already-lost particle, not its own (potentially wildly
            # diverging) state -- otherwise re-integrating a runaway
            # trajectory on every subsequent step reintroduces the same
            # max_steps risk we're trying to avoid, just later.
            integrate_from = jax.tree.map(
                lambda f, s: jnp.where(is_lost, f, s), frozen_state, state
            )
        else:
            integrate_from = state
        zs = jnp.array([zc0, zc1])
        track, _ = diffrax_solve(
            field, integrate_from, zs, forward_mode=forward_mode, rtol=rtol, atol=atol
        )
        state = jax.tree.map(lambda a: a[-1], track)
        theta0_sq = 0.0
        if rfc0_centers is not None:
            # No GH2 inside an RFC0 cavity's interior (real deck has vacuum
            # there) -- step size is small relative to the 249mm cavity
            # length, so a midpoint-inside check is an adequate approximation.
            mid = (zc0 + zc1) / 2
            in_rfc0 = jnp.any(jnp.abs(mid - rfc0_centers) < _RFC0_HALF_LENGTH)
            gh2_dz = jnp.where(in_rfc0, 1e-9 * u.mm, dz)  # tiny, not 0 (avoid NaN in Bethe-Bloch log terms)
        else:
            gh2_dz = dz
        dE_gh2 = GH2.straggling_params(state, gh2_dz).mean_energy_loss
        theta0_sq = theta0_sq + highland_theta0_squared(state, gh2_dz, X0_GH2)
        state = apply_energy_loss(state, dE_gh2)
        if include_presswall:
            # PRESSWALL_Z falling within this step's [zc0, zc1) range is a
            # runtime (traced) condition, not a static Python index -- z0/z1
            # may themselves be traced values under an outer jax.jit.
            crosses_wall = (zc0 <= PRESSWALL_Z) & (PRESSWALL_Z < zc1)
            wall_thickness = jnp.where(crosses_wall, PRESSWALL_THICKNESS, 0.0)
            dE_wall = jnp.where(
                crosses_wall,
                STAINLESS_316.straggling_params(state, PRESSWALL_THICKNESS).mean_energy_loss,
                0.0,
            )
            theta0_sq = theta0_sq + highland_theta0_squared(state, wall_thickness, X0_STAINLESS_316)
            state = apply_energy_loss(state, dE_wall)
        if window_z is not None:
            crosses_win = (zc0 <= window_z) & (window_z < zc1)

            def one_window(thick):
                return BERYLLIUM.straggling_params(state, thick).mean_energy_loss

            def one_window_theta0sq(thick):
                return highland_theta0_squared(state, thick, X0_BERYLLIUM)

            dE_each = jax.vmap(one_window)(window_thick)
            theta0sq_each = jax.vmap(one_window_theta0sq)(window_thick)
            dE_win = jnp.sum(jnp.where(crosses_win, dE_each, 0.0))
            theta0_sq = theta0_sq + jnp.sum(jnp.where(crosses_win, theta0sq_each, 0.0))
            state = apply_energy_loss(state, dE_win)
        if wedges is not None:
            # Wedges are only ~26-60mm thick; naive containment checks against
            # the outer dz step (15mm default) are ambiguous right at a wedge
            # boundary -- confirmed empirically (a step-like pattern in the
            # ours-vs-G4BL energy diff lines up with wedge z positions).
            # A 2-point (start+end) containment average halved the error but
            # didn't close it (still assumes the crossing sits at the step's
            # midpoint). This instead uses the EXACT thickness formula
            # stochastic_solve's substep uses for its (correct, geometric)
            # boundary-crossing case -- signed_time_to_boundary at both
            # endpoints plus the true displacement, not a containment guess:
            #   both endpoints inside  -> thickness = displacement
            #   only start inside      -> thickness = -sdf0 (exit distance)
            #   only end inside        -> thickness = displacement - sdf0
            #   neither inside         -> thickness = 0
            # (see stochastic.py's substep -- this is that formula, just
            # evaluated directly instead of through the broken
            # always-present-material combination described above.)
            start_state = jax.tree.map(lambda a: a[0], track)
            end_state_raw = jax.tree.map(lambda a: a[-1], track)
            ray0 = start_state.ray()
            ray1 = end_state_raw.ray()
            sdf0 = wedges.signed_time_to_boundary(ray0)
            sdf1 = wedges.signed_time_to_boundary(ray1)
            displacement = abs(ray1.p - ray0.p)
            raw_thickness = jnp.where(
                (sdf0 < 0.0) & (sdf1 < 0.0),
                displacement,
                jnp.where(
                    sdf0 < 0.0,
                    -sdf0,
                    jnp.where(sdf1 < 0.0, displacement - sdf0, 0.0),
                ),
            )
            # Bethe-Bloch-style straggling formulas can have a log(thickness)
            # singularity at exactly zero -- substitute a tiny safe value and
            # gate the result, same pattern stochastic_solve's substep uses.
            safe_thickness = jnp.where(raw_thickness <= 0.0, 1e-6 * u.mm, raw_thickness)
            dE_raw = wedges.interaction_params(state, safe_thickness).mean_energy_loss
            dE_wedge = jnp.where(raw_thickness > 0.0, dE_raw, 0.0)
            theta0sq_wedge_raw = highland_theta0_squared(state, safe_thickness, X0_LIH)
            theta0_sq = theta0_sq + jnp.where(raw_thickness > 0.0, theta0sq_wedge_raw, 0.0)
            state = apply_energy_loss(state, dE_wedge)
        if step_key is not None:
            state = apply_scattering_kick(state, jnp.sqrt(theta0_sq), step_key)

        if aperture_radius is not None:
            r = jnp.sqrt(state.kin.p.x**2 + state.kin.p.y**2)
            newly_lost = r > aperture_radius
            is_lost_next = is_lost | newly_lost
            # First step a particle exceeds the aperture, its CURRENT state
            # becomes the frozen reference for every subsequent step;
            # already-lost particles keep their existing frozen_state.
            frozen_state_next = jax.tree.map(
                lambda f, s: jnp.where(is_lost, f, s), frozen_state, state
            )
            # Reported state: frozen if lost (before or newly), else the
            # real evolved state.
            reported = jax.tree.map(
                lambda f, s: jnp.where(is_lost_next, f, s), frozen_state_next, state
            )
            p, t = reported.kin.p, reported.kin.t
            return reported, frozen_state_next, is_lost_next, (zc1, p.x, p.y, p.ct, t.x, t.y, t.z, t.ct)

        p, t = state.kin.p, state.kin.t
        return state, (zc1, p.x, p.y, p.ct, t.x, t.y, t.z, t.ct)

    if aperture_radius is not None:
        def scan_body(carry, xi):
            state, frozen_state, is_lost = carry
            if key is None:
                i, step_key = xi, None
            else:
                i, step_key = xi
            new_state, new_frozen, new_lost, out = one_step(
                state, i, step_key, frozen_state, is_lost
            )
            return (new_state, new_frozen, new_lost), out

        init_carry = (start, start, jnp.array(False))
        if key is None:
            (final_state, _, _), outs = jax.lax.scan(
                scan_body, init_carry, jnp.arange(n_steps)
            )
        else:
            step_keys = jr.split(key, n_steps)
            (final_state, _, _), outs = jax.lax.scan(
                scan_body, init_carry, (jnp.arange(n_steps), step_keys)
            )
        return final_state, outs

    if key is None:

        def scan_body(state, i):
            return one_step(state, i, None, None, None)

        final_state, outs = jax.lax.scan(scan_body, start, jnp.arange(n_steps))
    else:
        step_keys = jr.split(key, n_steps)

        def scan_body(state, xi):
            i, step_key = xi
            return one_step(state, i, step_key, None, None)

        final_state, outs = jax.lax.scan(
            scan_body, start, (jnp.arange(n_steps), step_keys)
        )
    return final_state, outs
