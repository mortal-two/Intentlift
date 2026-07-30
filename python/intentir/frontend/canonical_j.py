"""Shared ``J -> intent MLIR`` entry for reference and target paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from intentir.ir.canonical_j import elaborate_j, j_elaboration
from intentir.ir.dimension_relation import resolve_dimension_relations
from intentir.ir.value_dimension import resolve_value_dimension_equalities

from .emit_mlir import BroadcastAxes, IndexBoundsProjection, emit_module

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from intentir.verify.gate1.model import SemanticArchive


def emit_j_module(
    j: SemanticArchive,
    dims: Mapping[str, int],
    *,
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> str:
    """Emit MLIR only after the unique deterministic J elaboration boundary."""

    j = elaborate_j(j)
    resolved_dims = resolve_dimension_relations(j, dims)
    value_dimension_equalities = resolve_value_dimension_equalities(j, resolved_dims)
    output_values = {
        binding.output: binding.value
        for binding in j_elaboration(j).output_bindings
    }
    return emit_module(
        j.intent,
        resolved_dims,
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
        value_dimension_equalities=value_dimension_equalities,
        output_values=output_values,
    )


__all__ = ["emit_j_module"]
