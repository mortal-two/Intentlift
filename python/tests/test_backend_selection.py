from __future__ import annotations

import re
from pathlib import Path

import pytest

import intentir.backend.cuda.compile as backend_compile
from intentir.backend.cuda.selection import (
    DEFAULT_PASSES,
    LoweringTarget,
    select_variant,
)


_SOURCE = """
module {
  func.func @map(%x: tensor<4xf32>) -> tensor<4xf32> {
    %0 = intent.map_unary %x {fn = "exp"} : tensor<4xf32> -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""


def _pass_name(argument: str) -> str:
    return re.match(r"^-([^=]+)", argument).group(1)  # type: ignore[union-attr]


def test_registry_covers_every_registered_direct_gpu_pass() -> None:
    expected = {
        "convert-intent-row-reduction-to-gpu",
        "convert-intent-padded-row-reduction-to-gpu",
        "convert-intent-elementwise-to-gpu",
        "convert-intent-gather-to-gpu",
        "convert-intent-layout-to-gpu",
        "convert-intent-rng-to-gpu",
        "convert-intent-arg-reduce-to-gpu",
        "convert-intent-sort-to-gpu",
        "convert-intent-pool-to-gpu",
        "convert-intent-upsample-to-gpu",
        "convert-intent-polar-to-gpu",
        "convert-intent-conv-to-gpu",
        "convert-intent-conv-to-gpu-mma",
        "convert-intent-compact-to-gpu",
        "convert-intent-contraction-to-gpu",
        "convert-intent-contraction-to-gpu-mma",
        "convert-intent-contraction-to-gpu-mma-ws",
        "convert-intent-loop-contraction-to-gpu",
        "convert-intent-flash-mma-to-gpu",
        "convert-intent-attention-backward-to-gpu",
        "convert-intent-splitk-reduce-to-gpu",
        "convert-intent-splitk-reduce-varargs-to-gpu",
        "convert-intent-splitk-reduce-backward-to-gpu",
    }

    assert {_pass_name(item.pass_argument) for item in DEFAULT_PASSES} == expected


def test_selector_uses_the_cpp_structural_probe_without_verdicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str | None]] = []

    def fake_run(command: list[str], stdin: str | None = None) -> str:
        calls.append((command, stdin))
        return "module { gpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) }"

    monkeypatch.setattr(backend_compile, "_run", fake_run)

    result = select_variant(_SOURCE, target=LoweringTarget(chip="sm_120a"))

    assert result.pipeline_kind == "specialized"
    assert result.attempts[-1].direct_hit is True
    assert calls[0][1] == _SOURCE
    record = result.to_dict()
    assert "receipt" not in record
    assert "verdict" not in record
    assert "P5" not in record


def test_selector_accepts_a_complete_compiler_emitted_launch_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_compile,
        "_run",
        lambda _command, stdin=None: (
            "module { gpu.launch blocks(%c1, %c1, %c1) "
            "threads(%c1, %c1, %c1) {} "
            "gpu.launch blocks(%c1, %c1, %c1) "
            "threads(%c1, %c1, %c1) {} }"
        ),
    )

    result = select_variant(_SOURCE, target=LoweringTarget(chip="sm_120a"))

    assert result.pipeline_kind == "specialized"
    assert result.attempts[-1].gpu_launch_count == 2
    assert result.attempts[-1].direct_hit


def test_selector_falls_back_when_no_pass_consumes_the_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend_compile,
        "_run",
        lambda _command, stdin=None: stdin or _SOURCE,
    )

    result = select_variant(_SOURCE)

    assert result.pipeline_kind == "generic"
    assert result.fast_path_id is None
    assert len(result.attempts) == len(DEFAULT_PASSES)
    assert all(not attempt.direct_hit for attempt in result.attempts)


def test_schedule_override_still_requires_a_direct_structural_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = "-convert-intent-elementwise-to-gpu=block-threads=256"
    seen: list[str] = []

    def fake_run(command: list[str], stdin: str | None = None) -> str:
        seen.append(command[1])
        return stdin or _SOURCE

    monkeypatch.setattr(backend_compile, "_run", fake_run)
    result = select_variant(
        _SOURCE,
        only_fast_path="cuda.elementwise.row.vectorized.v1",
        pass_overrides={"cuda.elementwise.row.vectorized.v1": override},
    )

    assert result.pipeline_kind == "generic"
    assert seen == [override]
    assert result.attempts[0].pass_argument == override


@pytest.mark.skipif(
    not Path(backend_compile.INTENT_OPT).exists(), reason="intent-opt not built"
)
def test_real_row_reduction_pass_is_selected_structurally() -> None:
    source = (
        Path(__file__).resolve().parents[2] / "test" / "lower-softmax-to-gpu.mlir"
    )
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"

    result = select_variant(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-row-reduction-to-gpu=block-threads=4"
        },
    )

    assert result.pipeline_kind == "specialized"
    assert result.fast_path_id == fast_path_id
    assert result.attempts[0].residual_ops == ()
    assert result.attempts[0].gpu_launch_count == 1


@pytest.mark.skipif(
    not Path(backend_compile.INTENT_OPT).exists(), reason="intent-opt not built"
)
def test_real_multistage_row_reduction_graph_is_selected_structurally() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "test"
        / "lower-row-multistage-to-gpu.mlir"
    )
    fast_path_id = "cuda.row_reduction.normalization_loss.read_once.v1"

    result = select_variant(
        source,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: (
                "-convert-intent-row-reduction-to-gpu=block-threads=128 "
                "vector-width=1 reduction-parts=4"
            )
        },
    )

    assert result.pipeline_kind == "specialized"
    assert result.attempts[0].residual_ops == ()
    assert result.attempts[0].gpu_launch_count == 5
