#!/usr/bin/env python3
"""Smoke test: load the lattice, build the channel, evaluate the summed field.

Verifies the full pipeline (YAML -> typed records -> beamline SumField) works
end to end and that the summed field is finite and nonzero at a probe point.

Run:
    uv run python scripts/smoke_test.py
"""

import time
from pathlib import Path

import beamline.jax  # noqa: F401  (enables jax_enable_x64 on import)
import hepunits as u
import jax
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian4, Tangent
from hfofo.build import build_channel_batched
from hfofo.load import load_lattice

DATA = Path(__file__).parent.parent / "data" / "hfofo.yaml"


def main() -> None:
    lattice = load_lattice(DATA)
    print(
        f"loaded: {len(lattice.solenoids)} solenoids, "
        f"{len(lattice.cavities)} cavities, {len(lattice.wedges)} wedges"
    )

    channel = build_channel_batched(lattice)
    n = sum(g.stack and 1 for g in channel.groups)
    print(f"batched channel: {len(channel.groups)} stacked groups")

    pt = Cartesian4.make(x=1.0 * u.mm, z=-425.0 * u.mm)
    mom = Cartesian4.make(z=200.0 * u.MeV, ct=230.0 * u.MeV)
    vec = Tangent(p=pt, t=mom)

    field = jax.jit(lambda v: channel(v).t.coords)

    t = time.time()
    force = field(vec).block_until_ready()
    print(f"first eval (incl. compile): {time.time() - t:.1f}s")
    t = time.time()
    force = field(vec).block_until_ready()
    print(f"cached eval:                {time.time() - t:.4f}s")
    print(f"force at z=-425 mm: {force}")

    assert jnp.all(jnp.isfinite(force)), "field is non-finite!"
    assert jnp.any(force != 0.0), "field is identically zero!"

    print("\nSMOKE TEST PASSED: batched channel builds and evaluates fast.")


if __name__ == "__main__":
    main()
