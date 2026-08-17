"""Batched EM-field evaluation for large lattices.

``beamline.jax.emfield.SumField`` evaluates its components in a Python loop.
That is fine for a handful of sources but does not scale: the HFOFO channel has
566 components, and looping over them is ~5000x slower than necessary and blows
up ``jit`` compilation (each component unrolls into the XLA graph).

The fix exploits the fact that the components are not distinct *shapes*, only
distinct *parameters*: all 187 solenoids are the same ``TransformEMField``
wrapping a ``ThickSolenoid``, differing only in leaf values (jphi, rotation
matrix, translation). So we stack same-typed components into a single PyTree
(parameters batched along a leading axis) and evaluate the *one* template with
``equinox.filter_vmap``, summing the result. XLA compiles the template once and
runs it batched.

``StackedField`` holds one such stack; ``BatchedChannel`` holds several stacks
(e.g. solenoids + one stack per cavity gradient) and sums across them. Both are
``EMTensorField``s, so they drop into the existing tracking machinery unchanged.

This is prototyped here against the real channel; the intent is to lift it
upstream into ``beamline`` once proven (it is general infrastructure, not
HFOFO-specific).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from beamline.jax.coordinates import Cartesian3, Cartesian4, Tangent
from beamline.jax.emfield import EMTensorField
from beamline.jax.types import SBool, SFloat


def _batch_size(stack: EMTensorField) -> int:
    """Leading batch-axis length of a stacked component PyTree."""
    arrays = [x for x in jax.tree.leaves(stack) if eqx.is_array(x) and x.ndim > 0]
    if not arrays:
        raise ValueError("stacked component has no batched array leaves")
    return arrays[0].shape[0]


def stack_components[T: EMTensorField](components: list[T]) -> T:
    """Stack a list of same-typed components into one batched PyTree.

    Every component must be the same class with the same PyTree structure. Only
    the *array* leaves are stacked (leading batch axis of length
    ``len(components)``); *static* leaves (strings, ints like a cavity's
    ``mode``/``m``/``n``/``p``) must be identical across components and are kept
    shared. ``StackedField`` vmaps over the array leaves with the static ones
    held constant.
    """
    if not components:
        raise ValueError("cannot stack an empty component list")

    # Split each component into (dynamic array leaves, static everything-else).
    dynamics, statics = zip(
        *(eqx.partition(c, eqx.is_array) for c in components), strict=True
    )
    # Static parts must match across the batch (same mode/m/n/p/etc.).
    static0 = statics[0]
    stacked_dyn = jax.tree.map(lambda *xs: jnp.stack(xs), *dynamics)
    return eqx.combine(stacked_dyn, static0)


class StackedField(EMTensorField):
    """A batch of same-typed EM sources, evaluated with vmap and summed.

    ``stack`` is a single ``EMTensorField`` PyTree whose leaves carry a leading
    batch axis (as produced by ``stack_components``).
    """

    stack: EMTensorField

    def __call__(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        per = eqx.filter_vmap(lambda comp: comp(vec).t.coords)(self.stack)
        return Tangent(p=vec.p, t=Cartesian4(coords=per.sum(axis=0)))

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        # Components (TransformEMField) contract via __call__, not field_strength.
        raise RuntimeError("Use __call__ on StackedField, not field_strength")

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        # Force a batched result: components whose signed_time_to_boundary
        # returns a constant (e.g. solenoids always return inf -- field fills
        # space) make filter_vmap collapse to a scalar, which isn't indexable.
        # Broadcast against the batch size so `per` is always length-N.
        n = _batch_size(self.stack)

        def one(comp) -> SFloat:
            return jnp.asarray(comp.signed_time_to_boundary(ray))

        per = eqx.filter_vmap(one)(self.stack)
        per = jnp.broadcast_to(per, (n,))
        # nearest boundary across the batch (smallest absolute signed time)
        return per[jnp.argmin(jnp.abs(per))]


class BatchedChannel(EMTensorField):
    """Sum over several ``StackedField`` groups (plus optional loose components).

    Use one group per set of same-typed components (e.g. all solenoids, then one
    group per cavity gradient). ``loose`` holds any leftover one-off components
    that do not batch (evaluated individually).
    """

    groups: list[StackedField]
    loose: list[EMTensorField]

    def __init__(
        self,
        groups: list[StackedField],
        loose: list[EMTensorField] | None = None,
    ):
        self.groups = groups
        self.loose = loose or []

    def __call__(self, vec: Tangent[Cartesian4]) -> Tangent[Cartesian4]:
        total = jnp.zeros(4)
        for g in self.groups:
            total = total + g(vec).t.coords
        for c in self.loose:
            total = total + c(vec).t.coords
        return Tangent(p=vec.p, t=Cartesian4(coords=total))

    def field_strength(
        self, point: Cartesian4
    ) -> tuple[Tangent[Cartesian3], Tangent[Cartesian3]]:
        raise RuntimeError("Use __call__ on BatchedChannel, not field_strength")

    def contains(self, point: Cartesian3) -> SBool:
        return jnp.array(True)

    def signed_time_to_boundary(self, ray: Tangent[Cartesian3]) -> SFloat:
        ds = jnp.array(
            [g.signed_time_to_boundary(ray) for g in self.groups]
            + [c.signed_time_to_boundary(ray) for c in self.loose]
        )
        return ds[jnp.argmin(jnp.abs(ds))]
