"""Build a ``beamline`` field channel from a typed HFOFO lattice.

Turns the loaded schema records into placed ``beamline`` field sources summed
into one ``SumField``. Each element is wrapped in a ``TransformEMField`` (for EM
sources) placing it from local to global coordinates.
"""

from __future__ import annotations

import equinox as eqx
import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Cylindric3, Tangent, Transform
from beamline.jax.emfield import EMTensorField, SumField, TransformEMField
from beamline.jax.magnet.solenoid import ThickSolenoid
from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberWedge, TransformMaterialVolume
from beamline.jax.rfcavity.pillbox import PillboxCavity

from hfofo.schema import Cavity, Lattice, Rotation, Solenoid, Wedge
from hfofo.stacked import BatchedChannel, StackedField, stack_components

# ---------------------------------------------------------------------------
# CALIBRATION -- solenoid current -> jphi
# ---------------------------------------------------------------------------
# G4Beamline's ``coil`` element is a uniform-current-density annulus (Rin..Rout)
# discretized into ``nSheets`` radially-spaced thin current sheets, each carrying
# a fraction of the total current proportional to its radial slice -- see
# BLCoil.cc's ``getSheetField``/``addField`` (G4beamline 3.08 source). This is
# EXACTLY ``beamline.jax.magnet.solenoid.ThickSolenoid.B_shells``: verified
# bit-for-bit against a standalone compile of G4BL's literal C++ code (same
# elliptic-integral sheet formula, differing only by the expected Amp<->e/ns
# unit conversion, matched to 9+ significant figures).
#
# The single-solenoid reference trace originally used to calibrate this
# (criggall/muon-cooling field-studies/trace/single-solenoid/
# ReferenceParticle_NoPitch.txt) turned out to use a DIFFERENT, smaller test
# coil (innerRadius=360, outerRadius=500) than kat11's real geometry
# (420, 600) -- found by walking the file's git history to the commit that
# generated it (criggall/muon-cooling commit 9eaba07,
# field-studies/g4bl-input/hfofo_sol.in). That geometry mismatch, not a
# modeling difference, was the entire source of the earlier ~1-19% mismatches
# seen with both a thin-shell heuristic and a naive thick-coil fit. Using
# ThickSolenoid with the CORRECT test-coil geometry and the exact derived
# AMP_TO_JPHI conversion (no fitting at all) reproduces that reference trace to
# <=0.03% at every tabulated point.
#
# AMP_TO_JPHI converts G4BL's ``current`` (Amp/mm^2, a real current density) to
# beamline's ``jphi`` (e/ns/mm^2): 1 Amp = 1 Coulomb/s = (1/e) elementary
# charges/s = 6.241509074e9 e/ns (e = 1.602176634e-19 C). This is a physical
# constant, not a fit.
AMP_TO_JPHI: float = 6.241509074e9  # e/ns per Amp (exact: 1/(e[C] * 1e9))

# Coil kat11 geometry (from track_v7.in): all solenoids share this. Modeled as
# a true thick (uniform-current-density) annulus, nSheets=10 pinned to match
# the deck's ``coil kat11 ... nSheets=10`` exactly (see CALIBRATION note).
SOLENOID_RIN = 420.0 * u.mm
SOLENOID_ROUT = 600.0 * u.mm
SOLENOID_LENGTH = 300.0 * u.mm
SOLENOID_NSHEETS = 10


class Kat11Solenoid(ThickSolenoid):
    """``ThickSolenoid`` with ``num_shells`` pinned to match kat11's nSheets=10.

    ``ThickSolenoid.field_strength`` calls ``B_shells`` with its default
    (num_shells=200, vmap=False/lax.scan) -- both wrong for us: 200 shells is
    20x the work for no accuracy gain here (10 already matches G4BL to
    <=0.03%), and under the outer ``StackedField`` vmap (187 solenoids at
    once), an inner ``vmap`` compiles better than a nested ``lax.scan``.
    """

    num_shells: int = eqx.field(static=True, default=SOLENOID_NSHEETS)

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        xcyl = point.to_cylindric3()
        Brho, Bz = self.B_shells(xcyl.rho, xcyl.z, num_shells=self.num_shells, vmap=True)
        Bphi = jnp.zeros_like(Brho)
        E = Tangent(p=point.to_cartesian3(), t=Cartesian3.make())
        B = Tangent(p=xcyl, t=Cylindric3.make(rho=Brho, phi=Bphi, z=Bz))
        return E, B.to_cartesian()

# Pillbox variants: (iris radius, gradient). Iris radius is a kill aperture in
# the deck, not a field shaper; for optics we use a single TM010 pillbox and
# ignore the iris. Gradient: RFC0 uses Grad0=20, the rest Grad=25 MV/m.
_CAVITY_GRADIENT = {
    "RFC0": 20.0 * u.MV / u.m,
    "RFC": 25.0 * u.MV / u.m,
    "RFC1": 25.0 * u.MV / u.m,
    "RFC2": 25.0 * u.MV / u.m,
}
_CAVITY_INNER_LENGTH = 249.0 * u.mm


# ---------------------------------------------------------------------------
# Rotation / transform helpers
# ---------------------------------------------------------------------------

_AXIS_VEC = {
    "X": lambda: Cartesian3.make(x=1.0),
    "Y": lambda: Cartesian3.make(y=1.0),
    "Z": lambda: Cartesian3.make(z=1.0),
}


def _rotation_matrix(rotations: list[Rotation]):
    """Compose a 4x4 rotation matrix from a sequence of axis-angle rotations.

    G4Beamline applies ``rotation=A..,B..`` in the listed order to the object.
    We build each single-axis rotation via ``Transform.make_axis_angle`` and
    compose by matrix product (later rotations left-multiply).
    """
    R = jnp.eye(4)
    for rot in rotations:
        axis = _AXIS_VEC[rot.axis]()
        single = Transform.make_axis_angle(
            axis=axis, angle=rot.angle, translation=Cartesian4.make()
        )
        R = single.rotation @ R
    return R


def _placement_transform(
    rotations: list[Rotation], x: float, y: float, z: float
) -> Transform:
    """A Transform combining a compound rotation with a translation."""
    return Transform(
        translation=Cartesian4.make(x=x, y=y, z=z),
        rotation=_rotation_matrix(rotations),
    )


# ---------------------------------------------------------------------------
# Element builders
# ---------------------------------------------------------------------------


def build_solenoid(s: Solenoid) -> EMTensorField:
    jphi = s.current * AMP_TO_JPHI
    coil = Kat11Solenoid(
        Rin=SOLENOID_RIN,
        Rout=SOLENOID_ROUT,
        jphi=jphi,
        L=SOLENOID_LENGTH,
    )
    tf = _placement_transform(s.rotations, 0.0, 0.0, s.z)
    return TransformEMField(transform=tf, field=coil)


def _cavity_phase(cav: Cavity, frequency: float) -> float:
    """Map the deck's per-place ``time_offset`` [ns] to a pillbox phase [rad].

    beamline's pillbox field is ``E0 cos(2*pi*f*t + phase)`` (t carried as the
    ct coordinate). G4Beamline's ``timeOffset`` shifts that cavity's time origin
    so the reference particle is on-crest at the cavity center, i.e. the field is
    effectively ``cos(2*pi*f*(t - timeOffset))``. Matching the two gives
    ``phase = -2*pi*f*timeOffset``. Transit time only scales the amplitude (a
    sinc factor over the gap), it does not shift the crest, so no extra offset is
    needed. Verified: this maximizes single-cavity energy gain (milestone B).
    """
    return -2.0 * jnp.pi * frequency * cav.time_offset


def build_cavity(cav: Cavity, frequency: float) -> EMTensorField:
    grad = _CAVITY_GRADIENT[cav.variant]
    cavity = PillboxCavity(
        length=_CAVITY_INNER_LENGTH,
        frequency=frequency * u.GHz,
        E0=grad,
        mode="TM",
        m=0,
        n=1,
        p=0,  # TM010: uniform Ez, no longitudinal variation (cf. repo RF benchmark)
        phase=_cavity_phase(cav, frequency),
    )
    tf = _placement_transform(cav.rotations, 0.0, 0.0, cav.z)
    return TransformEMField(transform=tf, field=cavity)


# ---------------------------------------------------------------------------
# Wedge absorbers (material volumes)
# ---------------------------------------------------------------------------
# G4Beamline ``trap`` friendly params -> G4Trap half-lengths (verified against
# the G4BL manual: "direct interface to G4Trap ... axis along z, symmetrical
# left-right"):
#   dz  = length / 2                 (z-extent)
#   dy1 = dy2 = height / 2           (y-extent)
#   dx1 = dx3 = lowerWidth / 2       (x half-width at the lower, -y, edge)
#   dx2 = dx4 = upperWidth / 2       (x half-width at the upper, +y, edge)
#   theta = phi = alpha1 = alpha2 = 0
# The deck does not use the Xul/Xur/Xll/Xlr corner offsets, so the cross-section
# is unsheared (alpha=0). ``lowerWidth`` is overridden per placement (the taper
# down the channel); height/length/upperWidth are shared (LatticeMeta.wedge_base).

WEDGE_MATERIAL = MATERIALS["lithium_hydride_LiH"]


def build_wedge(w: Wedge, base) -> TransformMaterialVolume:
    """Build one placed LiH wedge absorber from a schema Wedge + shared base.

    ``base`` is ``lattice.meta.wedge_base`` (full height/length/upper_width);
    ``w.lower_width`` is the per-place taper override.
    """
    wedge = AbsorberWedge(
        material=WEDGE_MATERIAL,
        dz=base.length / 2 * u.mm,
        theta=0.0,
        phi=0.0,
        dy1=base.height / 2 * u.mm,
        dx1=w.lower_width / 2 * u.mm,
        dx2=base.upper_width / 2 * u.mm,
        alpha1=0.0,
        dy2=base.height / 2 * u.mm,
        dx3=w.lower_width / 2 * u.mm,
        dx4=base.upper_width / 2 * u.mm,
        alpha2=0.0,
    )
    tf = _placement_transform(w.rotations, w.x * u.mm, w.y * u.mm, w.z * u.mm)
    return TransformMaterialVolume(transform=tf, material=wedge)


def build_wedges(lattice: Lattice) -> list[TransformMaterialVolume]:
    """Build all placed wedge absorbers."""
    return [build_wedge(w, lattice.meta.wedge_base) for w in lattice.wedges]


def build_channel(
    lattice: Lattice,
    *,
    include_solenoids: bool = True,
    include_cavities: bool = True,
) -> SumField:
    """Assemble the EM channel (solenoids + RF) as one SumField.

    Wedges are material volumes, not EM sources, and are built separately for
    the stochastic stepper (milestone C); they are not part of this SumField.

    NOTE: this loop-based ``SumField`` does not scale (566 components ->
    ~minutes eager, OOM under jit). It is kept as a correctness oracle and for
    small assemblies. Use ``build_channel_batched`` for the full channel.
    """
    components: list[EMTensorField] = []
    if include_solenoids:
        components.extend(build_solenoid(s) for s in lattice.solenoids)
    if include_cavities:
        components.extend(
            build_cavity(c, lattice.meta.frequency) for c in lattice.cavities
        )
    if not components:
        raise ValueError("channel has no components")
    return SumField(components)


def build_channel_batched(
    lattice: Lattice,
    *,
    include_solenoids: bool = True,
    include_cavities: bool = True,
) -> BatchedChannel:
    """Assemble the EM channel as a ``BatchedChannel`` (vmap-batched, scalable).

    Same physics as ``build_channel`` but groups same-typed components into
    ``StackedField``s evaluated with ``vmap`` -- compiles once, runs batched.
    Solenoids form one group; cavities are grouped by gradient (they share a
    single ``PillboxCavity`` shape, differing only in ``E0``, so all cavities
    can batch together once ``E0`` is a stacked leaf).
    """
    groups: list[StackedField] = []
    if include_solenoids and lattice.solenoids:
        sols = [build_solenoid(s) for s in lattice.solenoids]
        groups.append(StackedField(stack=stack_components(sols)))
    if include_cavities and lattice.cavities:
        cavs = [build_cavity(c, lattice.meta.frequency) for c in lattice.cavities]
        groups.append(StackedField(stack=stack_components(cavs)))
    if not groups:
        raise ValueError("channel has no components")
    return BatchedChannel(groups=groups)


# ---------------------------------------------------------------------------
# Verification helper -- check a (Rin, Rout, L, current) config against a
# known reference Bz(z), with NO fitting (AMP_TO_JPHI is a physical constant).
# ---------------------------------------------------------------------------


def predicted_bz_onaxis(
    current: float,
    z: float,
    *,
    Rin: float = SOLENOID_RIN,
    Rout: float = SOLENOID_ROUT,
    L: float = SOLENOID_LENGTH,
    num_shells: int = SOLENOID_NSHEETS,
) -> float:
    """On-axis Bz(z) [CLHEP field units] predicted for a coil at ``current``
    [Amp/mm^2], with no fitting -- see the CALIBRATION note above.
    """
    coil = ThickSolenoid(Rin=Rin, Rout=Rout, jphi=current * AMP_TO_JPHI, L=L)
    _, bz = coil.B_shells(1e-6, z, num_shells=num_shells, vmap=True)
    return float(bz)
