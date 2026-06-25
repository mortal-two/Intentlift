"""IntentIR in-memory model + JSON (de)serialization.

Mirrors the JSON specification in ``.trellis/spec/ir/op-signatures.md`` and is
the 1:1 bridge to the ``intent`` MLIR dialect.
"""

from .model import IntentFunction, IntentOp, Tensor

__all__ = ["IntentFunction", "IntentOp", "Tensor"]
