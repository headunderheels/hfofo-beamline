"""Typed schema for the HFOFO lattice.

Plain ``dataclasses`` (no third-party deps). These describe the *placement*
geometry of the channel -- the irregular, per-element data that genuinely varies
(positions, rotations, tapering currents and widths). The regular structure (how
many elements per period, the fixed transverse wedge positions, the base solid
dimensions) lives in the builder and the YAML defaults, not repeated here.

Conventions (enforced by ``load.py``):
- All angles are in **radians**.
- All lengths are in **mm** (CLHEP), consistent with ``beamline``/``hepunits``.
- ``z`` is the absolute longitudinal position in mm.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rotation:
    """A single rotation about a principal axis.

    ``axis`` is one of 'X', 'Y', 'Z'; ``angle`` is in radians. A list of these
    is applied in order (matching the G4Beamline ``rotation=X..,Z..`` sequence).
    """

    axis: str
    angle: float

    def __post_init__(self) -> None:
        if self.axis not in ("X", "Y", "Z"):
            raise ValueError(f"Rotation axis must be X/Y/Z, got {self.axis!r}")


@dataclass(frozen=True)
class Solenoid:
    """A placed solenoid.

    ``current`` is retained in the deck's engineering units (the ``4.421*BLS``
    scaling); the current->jphi physics conversion is applied at build time via
    a single documented factor (see ``build.AMP_TO_JPHI``).
    """

    z: float
    current: float
    rotations: list[Rotation] = field(default_factory=list)
    polarity: str = ""  # "pos" | "neg" | "" (informational; sign is in current)


@dataclass(frozen=True)
class Cavity:
    """A placed RF cavity.

    ``variant`` selects the pillbox definition (RFC0/RFC/RFC1/RFC2), which differ
    by iris radius and gradient. ``time_offset`` is the deck's per-place timing
    (ns), mapped to ``PillboxCavity.phase`` at build time.
    """

    z: float
    variant: str
    time_offset: float
    rotations: list[Rotation] = field(default_factory=list)


@dataclass(frozen=True)
class Wedge:
    """A placed LiH wedge absorber.

    ``lower_width`` is the one per-element geometric override that tapers along
    the channel; the base solid dimensions (height/length/upper_width) are shared
    and live in ``LatticeMeta.wedge_base``.
    """

    z: float
    x: float
    y: float
    lower_width: float
    rotations: list[Rotation] = field(default_factory=list)


@dataclass(frozen=True)
class WedgeBase:
    """Shared base dimensions for all wedges (full sizes, mm), from ``trap wedge0``."""

    height: float
    length: float
    upper_width: float
    material: str = "LITHIUM_HYDRIDE"


@dataclass(frozen=True)
class LatticeMeta:
    """Provenance and channel-wide parameters."""

    source: str  # description of where this was extracted from
    period: float  # HFOFO period length [mm]
    n_periods: int
    frequency: float  # RF frequency [GHz]
    wedge_base: WedgeBase
    notes: str = ""


@dataclass
class Lattice:
    """The full parsed HFOFO channel."""

    meta: LatticeMeta
    solenoids: list[Solenoid] = field(default_factory=list)
    cavities: list[Cavity] = field(default_factory=list)
    wedges: list[Wedge] = field(default_factory=list)
