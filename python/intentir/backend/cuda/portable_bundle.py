"""JSON-safe transport primitives for cross-machine CUDA benchmark bundles."""

from __future__ import annotations

import base64
import hashlib
import re
from typing import Any, Mapping

import numpy as np

from intentir.verify.gate2.triton import (
    ArgumentRole,
    TritonArgument,
    TritonExecutionSpec,
)


ARRAY_PROTOCOL = "intentir.transport.ndarray.v1"


def cuda_compute_capability_for_chip(chip: str) -> tuple[int, int]:
    """Decode the exact compute capability named by an NVVM cubin target."""

    match = re.fullmatch(r"sm_([0-9]+)a?", chip)
    if match is None:
        raise ValueError(f"unsupported CUDA chip spelling {chip!r}")
    digits = match.group(1)
    if len(digits) < 2:
        raise ValueError(f"CUDA chip lacks major/minor digits: {chip!r}")
    return int(digits[:-1]), int(digits[-1])


def encode_array(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    dense = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    if dense.dtype.hasobject:
        raise ValueError("portable array cannot contain Python objects")
    payload = dense.tobytes(order="C")
    return {
        "protocol": ARRAY_PROTOCOL,
        "dtype": dense.dtype.str,
        "shape": list(dense.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "data_base64": base64.b64encode(payload).decode("ascii"),
    }


def decode_array(value: Mapping[str, Any]) -> np.ndarray:
    if value.get("protocol") != ARRAY_PROTOCOL:
        raise ValueError("unsupported portable array protocol")
    payload = base64.b64decode(value["data_base64"], validate=True)
    if hashlib.sha256(payload).hexdigest() != value.get("sha256"):
        raise ValueError("portable array digest mismatch")
    dtype = np.dtype(value["dtype"])
    shape = tuple(int(item) for item in value["shape"])
    array = np.frombuffer(payload, dtype=dtype).copy()
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if array.size != expected:
        raise ValueError("portable array byte length differs from shape")
    return array.reshape(shape)


def encode_values(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: encode_array(value) for name, value in sorted(values.items())}


def decode_values(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: decode_array(value) for name, value in values.items()}


def encode_triton_spec(spec: TritonExecutionSpec) -> dict[str, Any]:
    return {
        "arguments": [
            {
                "name": item.name,
                "role": item.role.value,
                "value_key": item.value_key,
                "dtype": item.dtype,
                "shape": list(item.shape) if item.shape is not None else None,
                "value": item.value,
                "output_key": item.output_key,
            }
            for item in spec.arguments
        ],
        "grid": list(spec.grid),
        "schedule": dict(spec.schedule),
        "device": spec.device,
    }


def decode_triton_spec(value: Mapping[str, Any]) -> TritonExecutionSpec:
    return TritonExecutionSpec(
        arguments=tuple(
            TritonArgument(
                name=item["name"],
                role=ArgumentRole(item["role"]),
                value_key=item.get("value_key"),
                dtype=item.get("dtype"),
                shape=(
                    tuple(int(dim) for dim in item["shape"])
                    if item.get("shape") is not None
                    else None
                ),
                value=item.get("value"),
                output_key=item.get("output_key"),
            )
            for item in value["arguments"]
        ),
        grid=tuple(int(item) for item in value["grid"]),
        schedule=dict(value["schedule"]),
        device=str(value.get("device", "cuda")),
    )


__all__ = [
    "ARRAY_PROTOCOL",
    "cuda_compute_capability_for_chip",
    "decode_array",
    "decode_triton_spec",
    "decode_values",
    "encode_array",
    "encode_triton_spec",
    "encode_values",
]
