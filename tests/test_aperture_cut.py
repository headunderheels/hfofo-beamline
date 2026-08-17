"""Tests for track_with_drag's aperture_radius (particle-loss) mechanism.

See hfofo/background.py's track_with_drag docstring for the full motivation:
a genuinely unstable/lost particle (confirmed by direct trajectory
inspection on a real initial.dat sample -- transverse radius doubling every
~150-200mm from an entirely unremarkable starting point) either crashes the
solver (correctly, at safe tolerance) or, if tolerance is loosened to avoid
the crash, produces a nonphysical 100+ meter excursion that silently
corrupts any ensemble statistic computed from it. aperture_radius freezes
such a particle at the aperture instead, mirroring what the real channel's
collimators would do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("beamline")

import hepunits as u
import jax
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4
from beamline.jax.kinematics import MuonStateDz
from hfofo.background import track_with_drag
from hfofo.build import build_channel_batched_windowed
from hfofo.load import load_lattice

DATA = Path(__file__).parent.parent / "data" / "hfofo.yaml"


@pytest.fixture(scope="module")
def channel():
    lattice = load_lattice(DATA)
    return build_channel_batched_windowed(lattice, z_center=0.0)


def _track(channel, x0, y0, aperture_radius, n_steps=20, dz=15.0 * u.mm):
    start = MuonStateDz.make(
        position=Cartesian4.make(x=x0, y=y0, z=-700.0 * u.mm),
        momentum=Cartesian3.make(z=247.5 * u.MeV),
        q=1,
    )
    z0 = -700.0 * u.mm
    z1 = z0 + n_steps * dz
    final_state, outs = track_with_drag(
        channel, start, z0, z1, dz=dz, include_presswall=False,
        n_steps=n_steps, aperture_radius=aperture_radius,
    )
    return final_state, outs


def test_particle_within_aperture_unaffected(channel):
    """A particle that never approaches the aperture should track identically
    whether aperture_radius is set or not (freezing logic must not perturb
    the ordinary case).
    """
    final_with, outs_with = _track(channel, 0.0, 0.0, aperture_radius=200.0 * u.mm)
    final_without, outs_without = _track(channel, 0.0, 0.0, aperture_radius=None)
    assert jnp.allclose(outs_with[1], outs_without[1], atol=1e-9 * u.mm)  # x
    assert jnp.allclose(outs_with[2], outs_without[2], atol=1e-9 * u.mm)  # y


def test_particle_outside_aperture_freezes_immediately():
    """A particle already started beyond the aperture must freeze on step 1
    and stay exactly frozen (unchanged x/y/px/py) for every subsequent step.
    """
    lattice = load_lattice(DATA)
    channel = build_channel_batched_windowed(lattice, z_center=0.0)
    final_state, outs = _track(
        channel, 500.0 * u.mm, 0.0, aperture_radius=200.0 * u.mm, n_steps=10
    )
    z, x, y, ct, px, py, pz, E = outs
    # every row after the first should be identical to the first (frozen)
    assert jnp.allclose(x, x[0], atol=1e-9 * u.mm)
    assert jnp.allclose(y, y[0], atol=1e-9 * u.mm)
    assert jnp.allclose(px, px[0], atol=1e-9 * u.MeV)
    assert jnp.allclose(py, py[0], atol=1e-9 * u.MeV)
    # z must still advance normally even though transverse state is frozen
    assert float(z[-1]) > float(z[0])


def test_aperture_cut_prevents_solver_crash_on_unstable_particle():
    """The real regression case: a particle from an actual initial.dat
    sample (index 20 of the reproducible 24-particle seed=0 draw, |pz-247.5|
    <75MeV/c cut) that hits diffrax's max_steps safety check without an
    aperture cut must complete successfully with one.
    """
    import glob

    import numpy as np

    candidates = glob.glob(
        "/home/claude/muon-cooling/**/initial.dat", recursive=True
    ) + glob.glob("/home/*/muon-cooling/**/initial.dat", recursive=True)
    if not candidates:
        pytest.skip("initial.dat not found in this environment")

    d = np.genfromtxt(candidates[0], comments="#")
    mu = d[d[:, 7] == -13]
    pz = mu[:, 5]
    cut = np.abs(pz - 247.5) < 75.0
    mu_cut = mu[cut]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(mu_cut), size=24, replace=False)
    sample = mu_cut[idx]
    row = sample[20]  # the confirmed-unstable particle

    lattice = load_lattice(DATA)
    period = lattice.meta.period
    z0 = -700.0 * u.mm
    z1 = z0 + period
    channel = build_channel_batched_windowed(lattice, z_center=float((z0 + z1) / 2))

    start = MuonStateDz.make(
        position=Cartesian4.make(x=row[0] * u.mm, y=row[1] * u.mm, z=z0),
        momentum=Cartesian3.make(x=row[3] * u.MeV, y=row[4] * u.MeV, z=row[5] * u.MeV),
        q=1,
    )
    dz = 15.0 * u.mm
    n_steps = int(round(float((z1 - z0) / dz)))
    # must not raise
    final_state, outs = track_with_drag(
        channel, start, z0, z1, dz=dz, include_presswall=True,
        n_steps=n_steps, rtol=1e-3, atol=1e-5, aperture_radius=200.0 * u.mm,
    )
    r_final = float(jnp.hypot(final_state.kin.p.x, final_state.kin.p.y)) / u.mm
    assert r_final >= 200.0 - 1e-3, "expected this particle to be flagged lost"
