"""Shared host transport for dtypes NumPy does not provide natively.

Target computation remains compiler generated.  These helpers only preserve
BF16 storage bits while values cross NumPy, PyTorch, the CPU memref ABI, and
portable benchmark receipts.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class RuntimeDTypeUnavailableError(RuntimeError):
    """A declared runtime dtype lacks its required host storage package."""


def numpy_bfloat16_dtype() -> np.dtype[Any]:
    """Return the registered two-byte NumPy-compatible BF16 dtype."""

    try:
        import ml_dtypes
    except ImportError as exc:  # pragma: no cover - packaging failure
        raise RuntimeDTypeUnavailableError(
            "BF16 host transport requires the verify extra dependency ml_dtypes"
        ) from exc
    return np.dtype(ml_dtypes.bfloat16)


def is_numpy_bfloat16(dtype: Any) -> bool:
    """Recognize BF16 without treating generic two-byte void storage as BF16."""

    try:
        return np.dtype(dtype) == numpy_bfloat16_dtype()
    except (TypeError, RuntimeDTypeUnavailableError):
        return False


def torch_tensor_from_numpy(
    torch: Any,
    value: Any,
    *,
    dtype: Any,
    device: str,
    clone: bool = False,
) -> Any:
    """Create a Torch tensor while preserving exact NumPy BF16 storage bits."""

    array = np.asarray(value)
    if dtype == torch.bfloat16 and is_numpy_bfloat16(array.dtype):
        storage = np.ascontiguousarray(array).view(np.uint16)
        tensor = torch.from_numpy(storage).view(torch.bfloat16).to(device=device)
    else:
        tensor = torch.as_tensor(array, dtype=dtype, device=device)
    return tensor.clone() if clone else tensor


def torch_tensor_to_numpy(torch: Any, tensor: Any) -> np.ndarray:
    """Copy a Torch tensor to NumPy, retaining BF16 as BF16 rather than FP32."""

    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        bits = value.view(torch.uint16).numpy()
        return bits.view(numpy_bfloat16_dtype()).copy()
    return value.numpy().copy()


def portable_numpy_dtype_name(dtype: Any) -> str:
    """Return an unambiguous portable spelling for one NumPy dtype."""

    value = np.dtype(dtype)
    return "bfloat16" if is_numpy_bfloat16(value) else value.str


def portable_numpy_dtype(spelling: str) -> np.dtype[Any]:
    """Decode a dtype spelling produced by :func:`portable_numpy_dtype_name`."""

    if spelling in {"bf16", "bfloat16"}:
        return numpy_bfloat16_dtype()
    return np.dtype(spelling)


__all__ = [
    "RuntimeDTypeUnavailableError",
    "is_numpy_bfloat16",
    "numpy_bfloat16_dtype",
    "portable_numpy_dtype",
    "portable_numpy_dtype_name",
    "torch_tensor_from_numpy",
    "torch_tensor_to_numpy",
]
