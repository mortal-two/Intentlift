"""Shared ``J -> intent MLIR`` entry for reference and target paths."""

from __future__ import annotations

from typing import Mapping

from intentir.ir.canonical_j import elaborate_j
from intentir.verify.gate1.model import SemanticArchive

from .emit_mlir import BroadcastAxes, IndexBoundsProjection, emit_module


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
    return emit_module(
        j.intent,
        dict(dims),
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    )


__all__ = ["emit_j_module"]
