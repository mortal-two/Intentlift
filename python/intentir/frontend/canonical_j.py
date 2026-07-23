"""Shared ``CanonicalJ -> intent MLIR`` entry for reference and target paths."""

from __future__ import annotations

from typing import Mapping, Sequence

from intentir.ir.canonical_j import CanonicalJ, elaborate_j
from intentir.verify.gate1.model import SemanticArchive

from .emit_mlir import BroadcastAxes, IndexBoundsProjection, emit_module


def emit_canonical_j_module(
    j: CanonicalJ | SemanticArchive,
    dims: Mapping[str, int],
    *,
    fold: int = 0,
    func_name: str | None = None,
    broadcast_axes: BroadcastAxes | None = None,
    proven_index_bounds: IndexBoundsProjection | None = None,
) -> str:
    """Emit MLIR only after the unique deterministic J elaboration boundary."""

    canonical = j if isinstance(j, CanonicalJ) else elaborate_j(j)
    return emit_module(
        canonical.archive.intent,
        dict(dims),
        fold=fold,
        func_name=func_name,
        broadcast_axes=broadcast_axes,
        proven_index_bounds=proven_index_bounds,
    )


__all__ = ["emit_canonical_j_module"]
