from __future__ import annotations

import pytest

from intentir.ir.layer_c import LayerCResolutionError, resolve_layer_c_scalar


def _pure_c(value: int = 4):
    return {
        "slots": [
            {
                "name": "c0",
                "kind": "compiler",
                "role": "num_warps",
                "value_type": "int",
                "required": True,
            }
        ],
        "values": {"c0": value},
    }


def test_resolve_layer_c_scalar_uses_canonical_role() -> None:
    resolved = resolve_layer_c_scalar(_pure_c(), "num_warps")
    assert resolved.slot == "c0"
    assert resolved.value == 4
    assert resolved.path == "$.values.c0"


@pytest.mark.parametrize(
    "legacy",
    [
        {"launch": {"num_warps": 4}},
        {"num_warps": 4},
        {"schedule_hints": [{"parameter": "num_warps", "value": 4}]},
    ],
)
def test_resolver_rejects_multiple_legacy_presentations(legacy) -> None:
    with pytest.raises(LayerCResolutionError, match="canonical"):
        resolve_layer_c_scalar(legacy, "num_warps")


def test_resolver_rejects_missing_role() -> None:
    with pytest.raises(LayerCResolutionError, match="absent"):
        resolve_layer_c_scalar(_pure_c(), "num_stages")
