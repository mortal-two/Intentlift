from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import intentir.backend.cuda.compile as backend_compile
import intentir.backend.cuda.pipeline as backend_pipeline
from intentir.backend.cuda.pipeline import (
    LoweredProgram,
    _transformation_record,
    execute_cuda_artifact,
    lower_to_cuda,
)
from intentir.backend.cuda.selection import LoweringTarget, ProbeAttempt, VariantSelection
from intentir.backend.cuda.schedules import ElementwiseSchedule, ScheduleDecision


_SOURCE = "module { func.func @f() { return } }\n"
_SPECIALIZED = "module { gpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) }\n"


def _kernel(*, cubin: bytes = b"\x7fELFfake") -> backend_compile.CompiledKernel:
    return backend_compile.CompiledKernel(
        cubin=cubin,
        kernel_name="kernel",
        binary_name="kernel_module",
        binary_mlir="module { gpu.binary @kernel_module [] }\n",
    )


def _selection(kind: str) -> VariantSelection:
    if kind == "generic":
        return VariantSelection(
            pipeline_kind="generic",
            target=LoweringTarget(),
            fast_path_id=None,
            pass_argument=None,
            scf_to_cf=False,
            attempts=(),
        )
    attempt = ProbeAttempt(
        fast_path_id="cuda.test.v1",
        pass_argument="-convert-intent-test-to-gpu",
        target_match=True,
        status="selected",
        direct_hit=True,
        gpu_launch_count=1,
    )
    return VariantSelection(
        pipeline_kind="specialized",
        target=LoweringTarget(),
        fast_path_id="cuda.test.v1",
        pass_argument="-convert-intent-test-to-gpu",
        scf_to_cf=True,
        attempts=(attempt,),
    )


def test_specialized_pipeline_replays_the_exact_probe_before_codegen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("specialized")
    observed: dict[str, object] = {}
    monkeypatch.setattr(backend_pipeline, "select_variant", lambda *a, **k: selection)
    monkeypatch.setattr(
        backend_compile, "_run", lambda command, stdin=None: _SPECIALIZED
    )

    def fake_lower(module: str, *, scf_to_cf: bool = False):
        observed["module"] = module
        observed["scf_to_cf"] = scf_to_cf
        return [_kernel()]

    monkeypatch.setattr(
        backend_compile, "lower_memref_module_to_kernels", fake_lower
    )

    result = lower_to_cuda(_SOURCE)

    assert result.selection.pipeline_kind == "specialized"
    assert observed == {"module": _SPECIALIZED, "scf_to_cf": True}
    assert result.to_record()["kernels"][0]["cubin_is_elf"] is True


def test_cpp_authored_concrete_state_record_is_decoded_without_name_rules() -> None:
    record = _transformation_record(
        """
        func.func @opaque() attributes {
          intent.backend.all_reduce_count = 0 : i64,
          intent.backend.barrier_count = 2 : i64,
          intent.backend.block_threads = 64 : i64,
          intent.backend.rows_per_block = 2 : i64,
          intent.backend.stage_count = 3 : i64,
          intent.backend.reduction_parts = 16 : i64,
          intent.backend.fma_count = 4 : i64,
          intent.backend.approximate_div_count = 1 : i64,
          intent.backend.folded_constexpr_select_count = 0 : i64,
          intent.backend.predicated_value_if_count = 0 : i64,
          intent.backend.lowering_path_kind = "row_reduction",
          intent.backend.reduction_topology = "workgroup_hierarchical",
          intent.backend.shuffle_count = 20 : i64,
          intent.backend.transformation_trace = [
            "cuda.row_reduction.read_once.v1",
            "cuda.reduction.workgroup_hierarchical.v1",
            "cuda.numeric.fma.f32.v1",
            "cuda.numeric.div_reciprocal_approx.f32.v1"
          ],
          intent.launch_plan = {block = [64, 1, 1], grid = [7, 1, 1],
            per_thread_tile = 8 : i64, pipeline_depth = 1 : i64,
            smem = 32 : i64, split_factor = 1 : i64,
            vector_width = 4 : i64}
        } { return }
        """
    )

    assert record is not None
    assert record.fma_count == 4
    assert record.approximate_div_count == 1
    assert record.predicated_value_if_count == 0
    assert record.folded_constexpr_select_count == 0
    assert record.rows_per_block == 2
    assert record.to_dict()["rows_per_block"] == 2
    assert record.warps_per_row == 1
    assert record.stage_count == 3
    assert record.reduction_parts == 16
    assert record.observation_changing
    assert record.reduction_topology == "workgroup_hierarchical"
    canonical_collective = replace(
        record,
        reduction_topology="gpu_all_reduce",
        fma_count=0,
        approximate_div_count=0,
        approximation_mode="none",
    )
    assert not canonical_collective.observation_changing
    multirow_prefix = replace(
        canonical_collective,
        reduction_topology="sorted_prefix_multirow_warp_early_stop",
    )
    assert multirow_prefix.observation_changing
    wide_prefix = replace(
        canonical_collective,
        reduction_topology=(
            "sorted_prefix_warp_scan_wide_epilogue_cross_multiply"
        ),
    )
    assert wide_prefix.observation_changing
    assert [item.to_dict() for item in record.launch_plans] == [
        {
            "grid": [7, 1, 1],
            "block": [64, 1, 1],
            "smem_bytes": 32,
            "per_thread_tile": 8,
            "vector_width": 4,
            "pipeline_depth": 1,
            "split_factor": 1,
        }
    ]


def test_generic_pipeline_uses_the_real_generic_compiler_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("generic")
    observed: list[Path] = []
    monkeypatch.setattr(backend_pipeline, "select_variant", lambda *a, **k: selection)

    def fake_compile(path: Path):
        path = Path(path)
        assert path.read_text(encoding="utf-8") == _SOURCE
        observed.append(path)
        return [_kernel()]

    monkeypatch.setattr(backend_compile, "compile_all_kernels", fake_compile)

    result = lower_to_cuda(_SOURCE)

    assert result.selection.pipeline_kind == "generic"
    assert len(result.kernels) == 1
    assert len(observed) == 1


def test_pipeline_rejects_a_target_different_from_the_configured_backend() -> None:
    with pytest.raises(ValueError, match="configured CUDA backend"):
        lower_to_cuda(_SOURCE, target=LoweringTarget(chip="sm_90"))


def test_pipeline_threads_a_structural_schedule_into_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("generic")
    observed: dict[str, object] = {}

    def fake_select(*args, **kwargs):
        observed.update(kwargs)
        return selection

    monkeypatch.setattr(backend_pipeline, "select_variant", fake_select)
    monkeypatch.setattr(
        backend_compile, "compile_all_kernels", lambda path: [_kernel()]
    )
    decision = ScheduleDecision(
        fast_path_id="cuda.elementwise.row.vectorized.v1",
        schedule=ElementwiseSchedule(128, True, 2),
        observed_block_threads=128,
        source="test",
    )

    lower_to_cuda(_SOURCE, schedule_decision=decision)

    assert observed["only_fast_path"] == decision.fast_path_id
    assert observed["pass_overrides"] == {
        decision.fast_path_id: decision.pass_argument
    }


def test_lowered_program_rejects_non_elf_device_artifacts() -> None:
    selection = _selection("generic")
    with pytest.raises(ValueError, match="not an ELF cubin"):
        LoweredProgram(
            selection=selection,
            kernels=(_kernel(cubin=b"not-elf"),),
        )


def test_execute_artifact_reuses_persisted_kernels_without_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("generic")
    kernel = _kernel()
    program = LoweredProgram(
        selection=selection,
        kernels=(kernel,),
    )
    observed: dict[str, object] = {}

    def fake_run(path, inputs, compile_all, output_aliases=None):
        observed["source"] = Path(path).read_text(encoding="utf-8")
        observed["inputs"] = inputs
        observed["kernels"] = compile_all(path)
        observed["aliases"] = output_aliases
        return {"out0": "executed"}

    monkeypatch.setattr(backend_pipeline.backend_orchestrate, "run_program", fake_run)

    result = execute_cuda_artifact(
        program,
        _SOURCE,
        {"x": "input"},
        output_aliases={"out0": "x"},
    )

    assert result == {"out0": "executed"}
    assert observed == {
        "source": _SOURCE,
        "inputs": {"x": "input"},
        "kernels": [kernel],
        "aliases": {"out0": "x"},
    }


@pytest.mark.skipif(
    not Path(backend_compile.INTENT_OPT).exists(), reason="intent-opt not built"
)
@pytest.mark.skipif(
    not (Path(backend_compile.CUDA_BIN) / "ptxas").exists(),
    reason="ptxas not available",
)
def test_neutral_bridge_really_compiles_generic_and_specialized_cubins() -> None:
    add_module = Path(__file__).resolve().parents[2] / "test" / "add.mlir"
    generic = lower_to_cuda(add_module, allow_specialized=False)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    specialized = lower_to_cuda(
        add_module,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=16"
        },
    )

    assert generic.selection.pipeline_kind == "generic"
    assert specialized.selection.pipeline_kind == "specialized"
    assert generic.kernels[0].cubin.startswith(b"\x7fELF")
    assert specialized.kernels[0].cubin.startswith(b"\x7fELF")
