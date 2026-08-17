"""Tests for loading and building the HFOFO lattice."""

import math
from pathlib import Path

import pytest

from hfofo.load import load_lattice
from hfofo.schema import Lattice

DATA = Path(__file__).parent.parent / "data" / "hfofo.yaml"


@pytest.fixture(scope="module")
def lattice() -> Lattice:
    return load_lattice(DATA)


def test_counts(lattice: Lattice):
    # Exact element counts from the frozen deck (track_v7.in, np=31).
    assert len(lattice.solenoids) == 187
    assert len(lattice.cavities) == 379
    assert len(lattice.wedges) == 171


def test_meta(lattice: Lattice):
    assert lattice.meta.period == 4200.0
    assert lattice.meta.n_periods == 31
    assert lattice.meta.frequency == 0.325
    assert lattice.meta.wedge_base.height == 350.0
    assert lattice.meta.wedge_base.upper_width == 0.005


def test_angles_are_radians(lattice: Lattice):
    # Wedges all have a Y rotation of exactly pi/2 (Y90 across the beam). This is
    # the cleanest radians check: if extraction had left Y90 as degrees it would
    # read 90.0, not pi/2. (We can't range-check other angles: cumulative wedge
    # rolls legitimately exceed 2*pi.)
    for w in lattice.wedges:
        yrot = [r for r in w.rotations if r.axis == "Y"]
        assert len(yrot) == 1
        assert yrot[0].angle == pytest.approx(math.pi / 2)


def test_wedge_positions(lattice: Lattice):
    # Three canonical transverse positions, evenly split.
    positions = {(round(w.x, 3), round(w.y, 3)) for w in lattice.wedges}
    assert positions == {(-79.11, -155.4), (174.2, 9.204), (-95.05, 146.2)}


def test_wedge_width_tapers(lattice: Lattice):
    widths = [w.lower_width for w in lattice.wedges]
    assert max(widths) == pytest.approx(59.326)
    assert min(widths) == pytest.approx(26.766)


def test_absolute_z(lattice: Lattice):
    # First wedge is at offset 2250 in period 1: 2250 + 4200 = 6450.
    assert lattice.wedges[0].z == pytest.approx(6450.0)


def test_validation_rejects_missing_field(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "meta:\n"
        "  period: 4200\n  n_periods: 31\n  frequency: 0.325\n"
        "  wedge_base: {height: 350, length: 700, upper_width: 0.005}\n"
        "solenoids:\n"
        "- z: 0.0\n"  # missing required 'current'
        "  rotations: []\n"
    )
    with pytest.raises(ValueError, match="missing required field 'current'"):
        load_lattice(bad)


# --- build tests (require beamline installed) ---


def test_build_channel(lattice: Lattice):
    pytest.importorskip("beamline")
    from hfofo.build import build_channel

    channel = build_channel(lattice)
    # 187 solenoids + 379 cavities summed
    assert len(channel.components) == 187 + 379


def test_cavities_only(lattice: Lattice):
    pytest.importorskip("beamline")
    from hfofo.build import build_channel

    channel = build_channel(lattice, include_solenoids=False)
    assert len(channel.components) == 379


def test_solenoid_field_formula_and_conversion():
    """The ThickSolenoid formula + AMP_TO_JPHI conversion match G4BL exactly.

    Reference: criggall/muon-cooling field-studies single-solenoid trace
    (field-studies/trace/single-solenoid/ReferenceParticle_NoPitch.txt),
    on-axis Bz(z) at current=80.46 Amp/mm^2. This trace's *own* deck (recovered
    from the muon-cooling git history at the commit that generated it, since
    the checked-in deck had since drifted -- see build.py's CALIBRATION note)
    used a test coil with innerRadius=360, outerRadius=500 -- NOT kat11's real
    420/600 -- so this validates the general model (uniform-current-density
    annulus, G4BL's exact nSheets algorithm) and the AMP_TO_JPHI unit
    conversion, not kat11's specific geometry. No fitting: AMP_TO_JPHI is a
    physical constant (Amp -> e/ns), and the match is <=0.03% at every point.
    """
    pytest.importorskip("beamline")
    import hepunits as u

    from hfofo.build import predicted_bz_onaxis

    # G4BL reference: on-axis Bz(z) [T] at current=80.46, test-coil geometry
    # (innerRadius=360, outerRadius=500, length=300 -- see docstring above).
    ref = {0: 4.69314, 100: 4.39501, 200: 3.65254, 300: 2.77530, 395: 2.03980}
    for zmm, bref in ref.items():
        bz = predicted_bz_onaxis(
            80.46, zmm * u.mm, Rin=360.0 * u.mm, Rout=500.0 * u.mm
        )
        got = bz / u.tesla
        assert abs(got - bref) / bref < 0.0005, f"z={zmm}: {got:.5f} vs {bref:.5f}"


# --- batched channel correctness (the key test: batched == loop oracle) ---


@pytest.mark.slow
def test_batched_matches_oracle_rf(lattice: Lattice):
    """StackedField (vmap) must equal the plain SumField loop, for RF.

    Marked ``slow``: the oracle side runs the eager loop over 379 cavities
    (~3 min) to have a trusted reference. Run with ``-m slow`` to include it.
    This guards against batching/broadcasting bugs producing wrong physics.
    """
    pytest.importorskip("beamline")
    import hepunits as u
    import jax
    import jax.numpy as jnp

    from beamline.jax.coordinates import Cartesian4, Tangent
    from hfofo.build import build_channel, build_channel_batched

    vec = Tangent(
        p=Cartesian4.make(x=1.0 * u.mm, z=-425.0 * u.mm),
        t=Cartesian4.make(z=200.0 * u.MeV, ct=230.0 * u.MeV),
    )

    oracle = build_channel(lattice, include_solenoids=False)(vec).t.coords
    batched = jax.jit(
        lambda v: build_channel_batched(lattice, include_solenoids=False)(v).t.coords
    )(vec)

    assert jnp.allclose(batched, oracle, rtol=1e-6, atol=1e-9)


def test_batched_channel_builds(lattice: Lattice):
    pytest.importorskip("beamline")
    from hfofo.build import build_channel_batched

    ch = build_channel_batched(lattice)
    assert len(ch.groups) == 2  # solenoids + cavities


# --- wedge absorbers + union material ---


def test_build_wedges(lattice: Lattice):
    pytest.importorskip("beamline")
    from hfofo.build import build_wedges

    wedges = build_wedges(lattice)
    assert len(wedges) == 171


def test_wedge_geometry_mapping(lattice: Lattice):
    """G4BL trap friendly params -> G4Trap half-lengths produce the right shape."""
    pytest.importorskip("beamline")
    import hepunits as u
    import numpy as np

    from hfofo.build import build_wedges

    w0 = build_wedges(lattice)[0].material  # AbsorberWedge inside the transform
    base = lattice.meta.wedge_base
    wd = lattice.wedges[0]
    # half-lengths
    assert float(w0.dz) == pytest.approx(base.length / 2)
    assert float(w0.dy1) == pytest.approx(base.height / 2)
    assert float(w0.dx1) == pytest.approx(wd.lower_width / 2)  # lower edge
    assert float(w0.dx2) == pytest.approx(base.upper_width / 2)  # upper edge


def test_union_material_matches_oracle(lattice: Lattice):
    """UnionMaterial (vmap) equals the per-wedge loop for contains/params."""
    pytest.importorskip("beamline")
    import hepunits as u
    import jax.numpy as jnp

    from beamline.jax.coordinates import Cartesian3, Cartesian4
    from beamline.jax.kinematics import MuonStateDz
    from hfofo.build import build_wedges
    from hfofo.union_material import build_union_material

    wedges = build_wedges(lattice)
    union = build_union_material(wedges)
    wd = lattice.wedges[0]

    center = Cartesian3.make(x=wd.x * u.mm, y=wd.y * u.mm, z=wd.z * u.mm)
    assert bool(union.contains(center)) == any(bool(w.contains(center)) for w in wedges)

    state = MuonStateDz.make(
        position=Cartesian4.make(x=wd.x * u.mm, y=wd.y * u.mm, z=wd.z * u.mm),
        momentum=Cartesian3.make(z=200 * u.MeV),
        q=1,
    )
    up = union.interaction_params(state, 10.0 * u.mm)
    containing = next(w for w in wedges if bool(w.contains(center)))
    op = containing.interaction_params(state, 10.0 * u.mm)
    assert float(up.mean_energy_loss) == pytest.approx(float(op.mean_energy_loss), rel=1e-6)

    # outside any wedge -> zero loss
    out = MuonStateDz.make(
        position=Cartesian4.make(z=-99999.0 * u.mm),
        momentum=Cartesian3.make(z=200 * u.MeV),
        q=1,
    )
    assert float(union.interaction_params(out, 10.0 * u.mm).mean_energy_loss) == 0.0
