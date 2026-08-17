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
import jax.numpy as jnp

from beamline.jax.absorber.material import DensityCorrection, Material, StragglingParams
from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.coordinates import Cartesian3, Tangent
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
    from beamline.jax.coordinates import Cartesian4

    new_t = Cartesian4.make(x=p.x * scale, y=p.y * scale, z=p.z * scale, ct=E_new)
    return MuonStateDz(kin=Tangent(p=state.kin.p, t=new_t), q=state.q)


def track_with_drag(
    field,
    start: MuonStateDz,
    z0: SFloat,
    z1: SFloat,
    dz: SFloat = 25.0 * u.mm,
    include_presswall: bool = True,
    wedges: UnionMaterial | None = None,
    window_z: jnp.ndarray | None = None,
    window_thick: jnp.ndarray | None = None,
    rfc0_centers: jnp.ndarray | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    n_steps: int | None = None,
):
    """Track a muon through ``field`` (solenoids+RF) with GH2 background drag,
    the ``presswall``, cavity Be windows, and (optionally) the LiH wedge
    absorbers folded in self-consistently, one small ``dz`` step at a time: a
    field-only ``diffrax_solve`` mini-step, followed by analytic energy-loss
    kicks for that same step. Not implemented via ``stochastic_solve`` -- that
    machinery's kick-detection is keyed off the *sign* of
    ``signed_time_to_boundary``, which doesn't have a consistent meaning for a
    material that's always present (see ``BackgroundAndWedges``'s docstring
    history / the module-level notes); a small-step deterministic loop sidesteps
    that entirely.

    ``window_z``/``window_thick`` come from ``cavity_window_positions`` --
    pass both together, or neither (skips Be windows). ``rfc0_centers`` comes
    from ``rfc0_interior_centers`` -- excludes GH2 loss inside RFC0 cavities
    specifically (the deck's only cavity variant without
    ``cavityMaterial=GH2``); omit to apply GH2 everywhere uniformly.

    Wedge handling here is simpler than ``stochastic_solve``'s boundary-aware
    crossing logic: at each step we just check whether the CURRENT position is
    inside any wedge (``wedges.contains``) and, if so, apply that wedge's
    straggling loss for the full step thickness ``dz``. Since ``dz`` (default
    25mm) is comparable to or a bit smaller than a wedge's thickness
    (~26-60mm), this resolves crossings to within about one step -- adequate
    for the wedges' small individual contribution, not precise enough to trust
    for anything wedge-crossing-dominated.

    ``n_steps`` must be a concrete Python int (not a traced value) when calling
    this under ``jax.jit`` -- pass it explicitly if ``z0``/``z1`` are traced;
    otherwise it's derived from ``z0``/``z1``/``dz`` directly.

    Returns (final_state, (z, x, y, ct, px, py, pz, E)) -- one row per dz step
    (not including the start point).
    """
    import jax
    from beamline.jax.integrate.propagate import diffrax_solve

    if n_steps is None:
        n_steps = int(round(float((z1 - z0) / dz)))

    def one_step(state, i):
        zc0 = z0 + i * dz
        zc1 = zc0 + dz
        zs = jnp.array([zc0, zc1])
        track, _ = diffrax_solve(field, state, zs, forward_mode=True, rtol=rtol, atol=atol)
        state = jax.tree.map(lambda a: a[-1], track)
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
        state = apply_energy_loss(state, dE_gh2)
        if include_presswall:
            # PRESSWALL_Z falling within this step's [zc0, zc1) range is a
            # runtime (traced) condition, not a static Python index -- z0/z1
            # may themselves be traced values under an outer jax.jit.
            crosses_wall = (zc0 <= PRESSWALL_Z) & (PRESSWALL_Z < zc1)
            dE_wall = jnp.where(
                crosses_wall,
                STAINLESS_316.straggling_params(state, PRESSWALL_THICKNESS).mean_energy_loss,
                0.0,
            )
            state = apply_energy_loss(state, dE_wall)
        if window_z is not None:
            crosses_win = (zc0 <= window_z) & (window_z < zc1)

            def one_window(thick):
                return BERYLLIUM.straggling_params(state, thick).mean_energy_loss

            dE_each = jax.vmap(one_window)(window_thick)
            dE_win = jnp.sum(jnp.where(crosses_win, dE_each, 0.0))
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
            state = apply_energy_loss(state, dE_wedge)
        p, t = state.kin.p, state.kin.t
        return state, (zc1, p.x, p.y, p.ct, t.x, t.y, t.z, t.ct)

    def scan_body(state, i):
        new_state, out = one_step(state, i)
        return new_state, out

    final_state, outs = jax.lax.scan(scan_body, start, jnp.arange(n_steps))
    return final_state, outs
