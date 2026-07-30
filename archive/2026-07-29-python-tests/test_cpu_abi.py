"""Fail-closed ranked-memref ABI contract tests."""

from __future__ import annotations

import ast
import ctypes
from pathlib import Path

import numpy as np
import pytest

from intentir.backend.cpu.abi import (
    CPUABIValidationError,
    descriptor_for,
    ranked_memref_descriptor_type,
    validate_array,
)


@pytest.mark.parametrize("shape", [(), (5,), (2, 3), (2, 3, 4, 5)])
def test_ranked_memref_descriptor_matches_numpy_storage(shape) -> None:
    value = np.zeros(shape, dtype=np.float32)
    descriptor = descriptor_for(value)

    assert descriptor.allocated == value.ctypes.data
    assert descriptor.aligned == value.ctypes.data
    assert descriptor.offset == 0
    if shape:
        assert tuple(descriptor.sizes) == shape
        assert tuple(descriptor.strides) == tuple(
            stride // value.dtype.itemsize for stride in value.strides
        )
    assert ctypes.alignment(type(descriptor)) == ctypes.alignment(ctypes.c_void_p)
    assert isinstance(descriptor, ranked_memref_descriptor_type(len(shape)))


def test_abi_rejects_shape_dtype_and_layout_mismatch_without_copying() -> None:
    contiguous = np.arange(12, dtype=np.float32).reshape(3, 4)
    valid = validate_array(
        contiguous, shape=(3, 4), dtype="f32", context="input"
    )
    assert valid is contiguous

    with pytest.raises(CPUABIValidationError, match="shape mismatch"):
        validate_array(contiguous, shape=(4, 3), dtype="f32", context="input")
    with pytest.raises(CPUABIValidationError, match="dtype mismatch"):
        validate_array(contiguous, shape=(3, 4), dtype="f64", context="input")
    with pytest.raises(CPUABIValidationError, match="C-contiguous"):
        validate_array(
            contiguous.T, shape=(4, 3), dtype="f32", context="input"
        )


def test_abi_rejects_read_only_destination() -> None:
    output = np.empty((4,), dtype=np.float32)
    output.flags.writeable = False

    with pytest.raises(CPUABIValidationError, match="writable"):
        validate_array(
            output,
            shape=(4,),
            dtype="f32",
            context="output",
            writable=True,
        )


def test_active_scalar_path_has_no_source_or_target_backend_dependency() -> None:
    root = Path(__file__).resolve().parents[2]
    files = [
        *(root / "python/intentir/backend/cpu").glob("*.py"),
        root / "python/intentir/verify/gate2/scalar.py",
    ]
    imported: set[str] = set()
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "mlir-runner" not in source
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

    forbidden = ("intentir.backend.cuda", "torch", "triton", "cupy")
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in imported
        for prefix in forbidden
    )
