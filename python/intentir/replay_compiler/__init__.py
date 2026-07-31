"""J-driven, source-blind Replay compiler support.

The package exposes only a deterministic ``J -> S_J`` semantic view and
read-only query/mount helpers.  It contains no target lowering, pi consumer, or
production search policy.
"""

from .packet import SemanticProgramMount, build_semantic_program_mount
from .semantic_materializer import (
    SemanticMaterializationError,
    semantic_materialize,
)
from .semantic_program import (
    SEMANTIC_MATERIALIZER_PROTOCOL,
    SEMANTIC_PROGRAM_SCHEMA,
    SEMANTIC_RECEIPT_PROTOCOL,
    SemanticProgram,
)
from .semantic_query import SemanticQueryError, query_semantic_program

__all__ = [
    "SEMANTIC_MATERIALIZER_PROTOCOL",
    "SEMANTIC_PROGRAM_SCHEMA",
    "SEMANTIC_RECEIPT_PROTOCOL",
    "SemanticMaterializationError",
    "SemanticProgram",
    "SemanticProgramMount",
    "SemanticQueryError",
    "build_semantic_program_mount",
    "query_semantic_program",
    "semantic_materialize",
]
