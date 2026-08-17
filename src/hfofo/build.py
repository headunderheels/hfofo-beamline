"""Build a ``beamline`` field channel from a typed HFOFO lattice.

Turns the loaded schema records into placed ``beamline`` field sources summed
into one ``SumField``. Each element is wrapped in a ``TransformEMField`` (for EM
sources) placing it from local to global coordinates.

The one physics-calibration knob -- the solenoid current -> ``jphi`` conversion
-- is isolated as ``CURRENT_TO_JPHI`` below and is currently a **placeholder**.
See the module note and ``calibrate_jphi`` before trusting solenoid field
magnitudes.
"""

from __future__ import annotations

import hepunits as u
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Transform
from beamline.jax.emfield import EMTensorField, SumField, TransformEMField
from beamline.jax.magnet.solenoid import ThinShellSolenoid
from beamline.jax.absorber.material import MATERIALS
from beamline.jax.absorber.volume import AbsorberWedge, TransformMaterialVolume
from beamline.jax.rfcavity.pillbox import PillboxCavity

from hfofo.schema import Cavity, Lattice, Rotation, Solenoid, Wedge
from hfofo.stacked import BatchedChannel, StackedField, stack_components

# ---------------------------------------------------------------------------
# CALIBRATION -- solenoid current -> jphi
# ---------------------------------------------------------------------------
# The deck's solenoid ``current`` is in engineering units (the ``4.421*BLS``
# scaling). Calibrated against the committed G4Beamline single-solenoid
# reference-particle trace (criggall/muon-cooling
# field-studies/trace/single-solenoid/ReferenceParticle_NoPitch.txt), which
# tabulates on-axis Bz(z) for the kat11 geometry at current=80.46: peak
# Bz = 4.69314 T at z=0.
#
# MODELING CHOICE (provisional -- flagged for later investigation):
# G4Beamline's ``coil`` field falls off faster in z than beamline's
# ``ThickSolenoid`` (uniform current across Rin..Rout) predicts -- a ~12%
# discrepancy at z=300mm. The falloff instead matches a THIN SHELL at the inner
# radius (420mm) to <=1.4% across the whole profile. This is a radial
# current-distribution modeling difference between the two codes, not a bug.
# Since the frozen channel was designed/tuned in G4BL, we match G4BL by using a
# thin shell at 420mm. Revisit if a thick-coil treatment is wanted (would
# require reconciling the two current models -- candidate to raise upstream).
CURRENT_TO_JPHI: float = 8.613692e11  # thin-shell@420 fit to G4BL reference
_CALIBRATED = True

# Coil kat11 geometry (from track_v7.in): all solenoids share this.
# We model each as a thin shell at the inner radius (see CALIBRATION note).
SOLENOID_SHELL_R = 420.0 * u.mm
SOLENOID_LENGTH = 300.0 * u.mm

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
    jphi = s.current * CURRENT_TO_JPHI
    coil = ThinShellSolenoid(
        R=SOLENOID_SHELL_R,
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
    if not _CALIBRATED and include_solenoids:
        import warnings

        warnings.warn(
            "Solenoid current->jphi conversion is not calibrated "
            "(CURRENT_TO_JPHI is a placeholder); solenoid field magnitudes are "
            "not physical. See build.calibrate_jphi.",
            stacklevel=2,
        )

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
    if not _CALIBRATED and include_solenoids:
        import warnings

        warnings.warn(
            "Solenoid current->jphi conversion is not calibrated "
            "(CURRENT_TO_JPHI is a placeholder); solenoid field magnitudes are "
            "not physical. See build.calibrate_jphi.",
            stacklevel=2,
        )

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
# Calibration helper (to be used once a reference field is known)
# ---------------------------------------------------------------------------


def calibrate_jphi(reference_bz_peak: float, at_current: float) -> float:
    """Return the CURRENT_TO_JPHI factor reproducing a known peak on-axis Bz.

    Given the known peak on-axis field ``reference_bz_peak`` [CLHEP field units]
    for a single solenoid at deck ``at_current``, solve for the jphi that
    produces it (the field is linear in jphi), then divide by the current. Uses
    the thin-shell-at-inner-radius model (see the CALIBRATION note above).
    """
    probe = ThinShellSolenoid(R=SOLENOID_SHELL_R, jphi=1.0, L=SOLENOID_LENGTH)
    _, bz_per_unit_jphi = probe.B_elliptic(1e-6, 0.0)
    jphi_needed = reference_bz_peak / bz_per_unit_jphi
    return float(jphi_needed / at_current)
