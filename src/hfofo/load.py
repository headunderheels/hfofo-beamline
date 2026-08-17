"""Load ``hfofo.yaml`` into the typed schema.

Thin ``from_dict`` conversion with explicit checks. Because the data file is
fully explicit (no defaults to expand), this is a direct mapping -- the value
here is validation and unit-contract enforcement, not magic.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from hfofo.schema import (
    Cavity,
    Lattice,
    LatticeMeta,
    Rotation,
    Solenoid,
    Wedge,
    WedgeBase,
)

_VALID_VARIANTS = {"RFC0", "RFC", "RFC1", "RFC2"}


def _require(d: dict, key: str, ctx: str):
    if key not in d:
        raise ValueError(f"{ctx}: missing required field {key!r}")
    return d[key]


def _rotations(raw: list, ctx: str) -> list[Rotation]:
    out = []
    for i, r in enumerate(raw or []):
        axis = _require(r, "axis", f"{ctx}.rotations[{i}]")
        angle = _require(r, "angle", f"{ctx}.rotations[{i}]")
        if not isinstance(angle, (int, float)):
            raise ValueError(f"{ctx}.rotations[{i}].angle must be numeric (radians)")
        # NOTE: we do NOT range-check angles. Cumulative wedge rolls legitimately
        # exceed 2*pi (the deck advances 120deg/wedge: Z333, Z453, Z573 -> up to
        # ~10 rad), so "> 2pi" cannot distinguish a valid large angle from a
        # stray degree value. The radians contract is enforced at extraction
        # time (scripts/extract_from_g4bl.py converts every angle), not guessed
        # here.
        out.append(Rotation(axis=axis, angle=float(angle)))
    return out


def load_lattice(path: str | Path) -> Lattice:
    """Load and validate the HFOFO lattice YAML."""
    data = yaml.safe_load(Path(path).read_text())

    m = _require(data, "meta", "lattice")
    wb = _require(m, "wedge_base", "meta")
    meta = LatticeMeta(
        source=m.get("source", "unknown"),
        period=float(_require(m, "period", "meta")),
        n_periods=int(_require(m, "n_periods", "meta")),
        frequency=float(_require(m, "frequency", "meta")),
        wedge_base=WedgeBase(
            height=float(_require(wb, "height", "meta.wedge_base")),
            length=float(_require(wb, "length", "meta.wedge_base")),
            upper_width=float(_require(wb, "upper_width", "meta.wedge_base")),
            material=wb.get("material", "LITHIUM_HYDRIDE"),
        ),
        notes=m.get("notes", ""),
    )

    solenoids = []
    for i, s in enumerate(data.get("solenoids", [])):
        ctx = f"solenoids[{i}]"
        solenoids.append(
            Solenoid(
                z=float(_require(s, "z", ctx)),
                current=float(_require(s, "current", ctx)),
                rotations=_rotations(s.get("rotations"), ctx),
                polarity=s.get("polarity", ""),
            )
        )

    cavities = []
    for i, c in enumerate(data.get("cavities", [])):
        ctx = f"cavities[{i}]"
        variant = _require(c, "variant", ctx)
        if variant not in _VALID_VARIANTS:
            raise ValueError(f"{ctx}: unknown variant {variant!r}")
        cavities.append(
            Cavity(
                z=float(_require(c, "z", ctx)),
                variant=variant,
                time_offset=float(_require(c, "time_offset", ctx)),
                rotations=_rotations(c.get("rotations"), ctx),
            )
        )

    wedges = []
    for i, w in enumerate(data.get("wedges", [])):
        ctx = f"wedges[{i}]"
        wedges.append(
            Wedge(
                z=float(_require(w, "z", ctx)),
                x=float(_require(w, "x", ctx)),
                y=float(_require(w, "y", ctx)),
                lower_width=float(_require(w, "lower_width", ctx)),
                rotations=_rotations(w.get("rotations"), ctx),
            )
        )

    return Lattice(
        meta=meta, solenoids=solenoids, cavities=cavities, wedges=wedges
    )
