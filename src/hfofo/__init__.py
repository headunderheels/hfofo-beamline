"""HFOFO snake channel built on the ``beamline`` differentiable simulator.

An *application* of ``beamline``: it reconstructs the frozen HFOFO muon-cooling
channel (solenoids, RF, LiH wedges) from extracted lattice data and assembles it
into ``beamline`` field/material objects for tracking.
"""

from hfofo.build import build_channel, build_cavity, build_solenoid
from hfofo.load import load_lattice
from hfofo.schema import (
    Cavity,
    Lattice,
    LatticeMeta,
    Rotation,
    Solenoid,
    Wedge,
    WedgeBase,
)

__all__ = [
    "load_lattice",
    "build_channel",
    "build_solenoid",
    "build_cavity",
    "Lattice",
    "LatticeMeta",
    "Solenoid",
    "Cavity",
    "Wedge",
    "WedgeBase",
    "Rotation",
]
