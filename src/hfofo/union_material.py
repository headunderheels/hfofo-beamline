"""Batched union of many material volumes (the material analogue of StackedField).

``stochastic_solve`` takes a single ``MaterialVolume``, but the HFOFO channel has
171 wedge absorbers. ``UnionMaterial`` presents them as one volume, evaluated
with ``equinox.filter_vmap`` over a stacked PyTree (compiles once, runs batched)
rather than a Python loop.

The wedges are **non-overlapping** (distinct z, 700mm apart, ~30mm thick each),
so any point is inside at most one wedge. That makes the union unambiguous:

- ``contains``            = any wedge contains the point (logical OR)
- ``interaction_params``  = the containing wedge's params (exactly one, or a
                            zero-loss no-op if none)
- ``signed_time_to_boundary`` = nearest wedge surface (min |signed time|)
- ``characteristic_length``   = min across wedges

Prototyped here; intended to be lifted upstream into ``beamline`` alongside the
batched-field work once proven, since any multi-absorber channel needs it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from beamline.jax.absorber.material import StragglingParams
from beamline.jax.absorber.volume import MaterialVolume
from beamline.jax.coordinates import Cartesian3, Tangent
from beamline.jax.kinematics import ParticleState
from beamline.jax.types import SBool, SFloat

from hfofo.stacked import _batch_size, stack_components


class UnionMaterial(MaterialVolume):
    """A batched union of same-typed, non-overlapping material volumes.

    ``stack`` is a single ``MaterialVolume`` PyTree whose array leaves carry a
    leading batch axis (as produced by ``stack_components``).
    """

    stack: MaterialVolume

    def contains(self, point: Cartesian3) -> SBool:
        per = eqx.filter_vmap(lambda m: m.contains(point))(self.stack)
        return jnp.any(per)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        n = _batch_size(self.stack)

        def one(m) -> SFloat:
            return jnp.asarray(m.signed_time_to_boundary(ray))

        per = jnp.broadcast_to(eqx.filter_vmap(one)(self.stack), (n,))
        # nearest surface across the batch (smallest absolute signed time)
        return per[jnp.argmin(jnp.abs(per))]

    def characteristic_length(self) -> SFloat:
        n = _batch_size(self.stack)

        def one(m) -> SFloat:
            return jnp.asarray(m.characteristic_length())

        per = jnp.broadcast_to(eqx.filter_vmap(one)(self.stack), (n,))
        return jnp.min(per)

    def interaction_params(
        self, state: ParticleState, thickness: SFloat
    ) -> StragglingParams:
        """Straggling params of the wedge containing ``state`` (or a no-op).

        Wedges do not overlap, so at most one ``contains`` is true. We evaluate
        every wedge's (contains, params) in a single vmap and select the
        containing one by mask; if none contains the point, zero-loss params
        result (the stepper also guards thickness==0).

        ``StragglingParams`` is a plain frozen dataclass (not a registered
        PyTree), so the vmapped function returns an explicit tuple of arrays --
        keeping everything in one transform scope avoids tracer leaks.
        """
        import dataclasses

        point = state.kin.p.to_cartesian3()
        field_names = [f.name for f in dataclasses.fields(StragglingParams)]

        def per_wedge(m):
            inside = m.contains(point)
            p = m.interaction_params(state, thickness)
            vals = tuple(getattr(p, name) for name in field_names)
            return inside, vals

        inside, vals = eqx.filter_vmap(per_wedge)(self.stack)
        # select containing wedge's value per field (sum: at most one nonzero)
        selected = {
            name: jnp.sum(jnp.where(inside, v, 0.0))
            for name, v in zip(field_names, vals, strict=True)
        }
        return StragglingParams(**selected)


def build_union_material(wedges: list[MaterialVolume]) -> UnionMaterial:
    """Stack a list of (same-typed) placed wedge volumes into a UnionMaterial."""
    if not wedges:
        raise ValueError("no wedges to union")
    return UnionMaterial(stack=stack_components(wedges))
