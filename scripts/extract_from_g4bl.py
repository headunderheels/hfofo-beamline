#!/usr/bin/env python3
"""One-time extraction: G4Beamline frozen deck -> hfofo.yaml.

**This is provenance, not a runtime dependency.** Nothing in ``src/hfofo``
imports it. It exists so the committed ``data/hfofo.yaml`` is reproducible from
the original G4Beamline input, and so the (mildly unpleasant) job of parsing the
G4BL DSL lives in exactly one disposable place.

Usage:
    python scripts/extract_from_g4bl.py \
        --input-dir /path/to/muon-cooling/hfofo-frozen/g4bl-input \
        --out data/hfofo.yaml

It resolves ``param`` definitions (with ``$var`` substitution and basic
arithmetic), follows the ``sol_place*/RFplace*/abs_place*`` placements, converts
all angles to **radians**, resolves ``z`` to absolute mm, and writes YAML that
conforms to ``hfofo.schema``.
"""

from __future__ import annotations

import argparse
import ast
import math
import operator
import re
from pathlib import Path

import yaml

# --- tiny safe arithmetic evaluator for param expressions -------------------

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _ev(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_ev(node.operand))
    raise ValueError(f"unsupported expr node: {ast.dump(node)}")


def evalexpr(expr: str, params: dict[str, float]) -> float:
    for name in sorted(params, key=len, reverse=True):
        expr = expr.replace(f"${name}", f"({params[name]!r})")
    if "$" in expr:
        raise KeyError(f"undefined param(s) in: {expr}")
    return _ev(ast.parse(expr.replace("^", "**"), mode="eval"))


# --- deck loading -----------------------------------------------------------

_ROT = re.compile(r"([XYZ])([-+0-9.$\w]+)")


def _rot_to_radians(value: str, params: dict[str, float]) -> list[dict]:
    """Parse ``X$pitch,Z120`` into [{axis,angle_rad}, ...].

    NOTE the deck's unit split: ``$pitch`` params are already radians
    (``pitch = -0.0025*180/pi`` is stored as *degrees* in the deck param, but the
    numeric Z-rolls like ``Z120`` are degrees too). We normalise *everything* to
    radians here. The deck's ``param pitch=-0.0025*180/pi`` yields a degree value,
    so all rotation angles are treated as degrees at this layer and converted.
    """
    out = []
    for part in value.split(","):
        m = _ROT.fullmatch(part.strip())
        if not m:
            raise ValueError(f"bad rotation token {part!r}")
        axis, ang = m.group(1), m.group(2)
        deg = evalexpr(ang, params)
        out.append({"axis": axis, "angle": math.radians(deg)})
    return out


def load_deck(path: Path, params: dict[str, float], out: list[dict]) -> None:
    text = path.read_text().replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\\\n", " ", text)
    for line in text.split("\n"):
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        if s.startswith("include "):
            load_deck(path.parent / s.split(None, 1)[1].strip(), params, out)
            continue
        tok = s.split()
        if tok[0] == "param":
            for kv in tok[1:]:
                if "=" in kv:
                    k, e = kv.split("=", 1)
                    try:
                        params[k] = evalexpr(e, params)
                    except (ValueError, KeyError):
                        pass  # non-numeric param, skip
        elif tok[0] == "place":
            out.append(_place(tok[1:], params, s))


def _place(tok: list[str], params: dict[str, float], raw: str) -> dict:
    rec: dict = {"name": tok[0], "raw": raw}
    for kv in tok[1:]:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        if k == "rotation":
            rec["rotations"] = _rot_to_radians(v, params)
        else:
            try:
                rec[k] = evalexpr(v, params)
            except (ValueError, KeyError):
                rec[k] = v  # keep string extras (format=ascii etc.)
    return rec


# --- classify placements into schema records --------------------------------

_RFC_VARIANTS = {"RFC0", "RFC", "RFC1", "RFC2"}


def build_lattice_dict(input_dir: Path) -> dict:
    deck = input_dir / "track_v7.in"
    params: dict[str, float] = {}
    placements: list[dict] = []
    load_deck(deck, params, placements)

    solenoids, cavities, wedges = [], [], []
    for p in placements:
        name = p["name"]
        if name in ("SolPos", "SolNeg"):
            solenoids.append(
                {
                    "z": p["z"],
                    "current": p.get(
                        "current",
                        params["curpl"] if name == "SolPos" else params["curmn"],
                    ),
                    "rotations": p.get("rotations", []),
                    "polarity": "pos" if name == "SolPos" else "neg",
                }
            )
        elif name in _RFC_VARIANTS:
            cavities.append(
                {
                    "z": p["z"],
                    "variant": name,
                    "time_offset": p.get("timeOffset", 0.0),
                    "rotations": p.get("rotations", []),
                }
            )
        elif name == "wedge0":
            wedges.append(
                {
                    "z": p["z"],
                    "x": p.get("x", 0.0),
                    "y": p.get("y", 0.0),
                    "lower_width": p.get("lowerWidth", params.get("lowerWidth", 59.5)),
                    "rotations": p.get("rotations", []),
                }
            )
        # presswall/abtube/terminus/Det* -> deferred, not extracted

    meta = {
        "source": "criggall/muon-cooling hfofo-frozen/g4bl-input (track_v7.in)",
        "period": params["period"],
        "n_periods": int(params["np"]),
        "frequency": 0.325,  # GHz, from pillbox defs
        "wedge_base": {
            "height": 350.0,
            "length": 700.0,
            "upper_width": 0.005,
            "material": "LITHIUM_HYDRIDE",
        },
        "notes": (
            "Angles in radians; z absolute in mm. Solenoid current in deck "
            "engineering units (4.421*BLS scaling); convert via build.AMP_TO_JPHI."
        ),
    }
    return {
        "meta": meta,
        "solenoids": solenoids,
        "cavities": cavities,
        "wedges": wedges,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    lattice = build_lattice_dict(args.input_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        yaml.safe_dump(lattice, f, sort_keys=False, default_flow_style=False)

    print(
        f"wrote {args.out}: "
        f"{len(lattice['solenoids'])} solenoids, "
        f"{len(lattice['cavities'])} cavities, "
        f"{len(lattice['wedges'])} wedges"
    )


if __name__ == "__main__":
    main()
