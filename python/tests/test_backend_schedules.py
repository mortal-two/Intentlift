from __future__ import annotations

import pytest

from intentir.backend.cuda.schedules import (
    ElementwiseSchedule,
    RowSchedule,
    pass_override,
    resolve_elementwise_schedule,
    resolve_row_schedule,
)


_RMSNORM = """
module {
  func.func @rmsnorm(
      %x: tensor<4x1024xf32>, %w: tensor<1024xf32>, %eps: tensor<f32>)
      -> tensor<4x1024xf32> {
    %sum = intent.reduce %x {axes = array<i64: 1>, fn = "sum", keepdims = true}
      : tensor<4x1024xf32> -> tensor<4x1xf32>
    %inv = intent.map_unary %sum {fn = "rsqrt"}
      : tensor<4x1xf32> -> tensor<4x1xf32>
    %out = intent.map_binary %x, %inv {fn = "mul"}
      : (tensor<4x1024xf32>, tensor<4x1xf32>) -> tensor<4x1024xf32>
    return %out : tensor<4x1024xf32>
  }
}
"""

_FLAT_ELEMENTWISE = """
module {
  func.func @activation(
      %x: tensor<512x256xf32>, %y: tensor<512x256xf32>)
      -> tensor<512x256xf32> {
    %out = intent.map_binary %x, %y {fn = "mul"}
      : (tensor<512x256xf32>, tensor<512x256xf32>)
        -> tensor<512x256xf32>
    return %out : tensor<512x256xf32>
  }
}
"""

_BROADCAST_ELEMENTWISE = """
module {
  func.func @activation(
      %x: tensor<512x256xf32>, %scale: tensor<f32>)
      -> tensor<512x256xf32> {
    %out = intent.map_binary %x, %scale {fn = "mul"}
      : (tensor<512x256xf32>, tensor<f32>) -> tensor<512x256xf32>
    return %out : tensor<512x256xf32>
  }
}
"""


def _pure_c(num_warps: int = 4):
    return {
        "slots": [
            {
                "name": "c0", "kind": "compiler", "role": "num_warps",
                "value_type": "int", "required": True,
            }
        ],
        "values": {"c0": num_warps},
    }


def test_small_rmsnorm_preserves_the_measured_preload_policy() -> None:
    decision = resolve_row_schedule(
        _RMSNORM,
        _pure_c(),
    )

    assert isinstance(decision.schedule, RowSchedule)
    assert decision.schedule.block_threads == 128
    assert decision.schedule.defer_epilogue_col_loads is False
    assert decision.source == "structural.small-register-rmsnorm.preload-col.v1"
    assert decision.optimization["policy_id"] == (
        "cuda.row.small-register-rmsnorm.preload-col.v1"
    )
    assert decision.pass_argument.endswith("defer-epilogue-col-loads=false")


def test_explicit_row_candidate_retains_all_ablation_axes() -> None:
    override = RowSchedule(
        block_threads=256,
        tile_cols=64,
        vector_width=1,
        fuse_fma=True,
        defer_epilogue_col_loads=False,
    )

    decision = resolve_row_schedule(
        _RMSNORM,
        _pure_c(),
        schedule_override=override,
    )

    assert decision.schedule == override
    assert decision.source == "offline.layer_c_retune.v1"
    assert decision.optimization["candidate"] == override.to_dict()
    assert pass_override(decision) == {
        decision.fast_path_id: override.pass_argument
    }


def test_large_equal_shape_elementwise_graph_selects_flat_vec4() -> None:
    decision = resolve_elementwise_schedule(
        _FLAT_ELEMENTWISE,
        _pure_c(),
    )

    assert decision.schedule == ElementwiseSchedule(128, True, 4)
    assert "flatten-contiguous=true" in decision.pass_argument
    assert decision.optimization["function_name_used"] is False


def test_broadcast_elementwise_graph_remains_on_the_row_path() -> None:
    decision = resolve_elementwise_schedule(
        _BROADCAST_ELEMENTWISE,
        _pure_c(),
    )

    assert decision.schedule == ElementwiseSchedule(128, False, 1)
    assert "flatten-contiguous" not in decision.pass_argument


def test_flat_override_must_be_structurally_eligible() -> None:
    with pytest.raises(ValueError, match="structurally eligible"):
        resolve_elementwise_schedule(
            _BROADCAST_ELEMENTWISE,
            _pure_c(),
            schedule_override=ElementwiseSchedule(256, True, 2),
        )


@pytest.mark.parametrize(
    "schedule",
    [
        lambda: RowSchedule(96, 32),
        lambda: RowSchedule(128, 48),
        lambda: ElementwiseSchedule(128, False, 4),
        lambda: ElementwiseSchedule(128, True, 1),
    ],
)
def test_invalid_schedule_candidates_fail_closed(schedule) -> None:
    with pytest.raises(ValueError):
        schedule()
