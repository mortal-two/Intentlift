"""Resolve host MLIR/LLVM tools without importing a concrete backend."""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
INTENT_OPT = os.environ.get(
    "INTENT_OPT",
    str(REPO_ROOT / "build" / "tools" / "intent-opt" / "intent-opt"),
)
MLIR_OPT = os.environ.get("MLIR_OPT", "/usr/lib/llvm-20/bin/mlir-opt")
LLVM_BIN = Path(MLIR_OPT).parent
LLVM_LIB = Path(os.environ.get("LLVM_LIB", "/usr/lib/llvm-20/lib"))
MLIR_TRANSLATE = os.environ.get(
    "MLIR_TRANSLATE", str(LLVM_BIN / "mlir-translate")
)
CLANG = os.environ.get("CLANG", str(LLVM_BIN / "clang-20"))


__all__ = [
    "CLANG",
    "INTENT_OPT",
    "LLVM_BIN",
    "LLVM_LIB",
    "MLIR_OPT",
    "MLIR_TRANSLATE",
    "REPO_ROOT",
]
