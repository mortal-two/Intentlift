"""IntentIR: certified semantic lifting.

Python side owns LLM extraction and verification; the Intent MLIR dialect is
the C++ side (see ``include/Intent`` / ``lib/Intent``). The link between them is
IntentIR JSON (see ``intentir.ir``), which maps 1:1 to the ``intent`` dialect.
"""

__version__ = "0.0.1"
