from __future__ import annotations

import pytest

from intentir.backend.layer_c import LayerCResolutionError, resolve_layer_c_scalar


@pytest.mark.parametrize(
    "layer_c",
    [
        {"launch": {"num_warps": 4}},
        {"launch_parameters": {"num_warps": 4}},
        {"num_warps": {"hint_id": "hint.num_warps", "value": 4}},
        {
            "schedule_hints": [
                {
                    "parameter": "num_warps",
                    "value": 4,
                    "anchor_ids": ["anchor.schedule"],
                }
            ]
        },
    ],
)
def test_resolve_layer_c_scalar_accepts_equivalent_layouts(layer_c) -> None:
    resolved = resolve_layer_c_scalar(layer_c, "num_warps")

    assert resolved.value == 4
    assert resolved.paths


def test_resolve_layer_c_scalar_deduplicates_consistent_restatements() -> None:
    resolved = resolve_layer_c_scalar(
        {
            "launch": {"num_warps": 8},
            "schedule_hints": [{"parameter": "num_warps", "value": 8}],
        },
        "num_warps",
    )

    assert resolved.value == 8
    assert len(resolved.paths) == 2


def test_resolve_layer_c_scalar_rejects_conflicts() -> None:
    with pytest.raises(LayerCResolutionError, match="conflicts"):
        resolve_layer_c_scalar(
            {
                "launch": {"num_warps": 4},
                "schedule_hints": [{"parameter": "num_warps", "value": 8}],
            },
            "num_warps",
        )


def test_resolve_layer_c_scalar_rejects_missing_parameter() -> None:
    with pytest.raises(LayerCResolutionError, match="is absent"):
        resolve_layer_c_scalar({"launch": {"num_stages": 3}}, "num_warps")

