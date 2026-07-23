"""Front-end: lift IntentIR JSON into the `intent` MLIR dialect.

The research flow is ``LLM -> human-readable JSON -> intent MLIR dialect ->
lower to standard MLIR``. This package is the ``JSON -> intent MLIR`` leg: it
emits real `intent`-dialect text (``intent.loop`` / ``contraction`` / ``reduce``
/ ``map_binary`` / ``map_unary``) from an :class:`intentir.ir.model.IntentFunction`,
instantiated at concrete shapes. The downstream ``-convert-intent-to-linalg``
pass carries it to standard MLIR.
"""

from intentir.frontend.canonical_j import emit_j_module
from intentir.frontend.emit_mlir import emit_body_function, emit_loop_module, emit_module

__all__ = [
    "emit_body_function",
    "emit_j_module",
    "emit_loop_module",
    "emit_module",
]
