from __future__ import annotations

import json
from dataclasses import replace

import pytest

from intentir.extraction.lifters.scrc import (
    check_scrc_lift_candidate,
    decode_scrc_lift_artifact,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)
from intentir.extraction.providers.provider import ProviderConfig
from intentir.ir.model import IntentFunction
from intentir.verify.frontend_workloads import (
    FRONTEND_DATASET_HARNESSES,
    build_frontend_dataset_gate2_suite,
    prepare_frontend_dataset_workload,
    project_frontend_dataset_harnesses,
)
from intentir.verify.gate1.cuda_compiler import CudaCompileCase, CudaKernelSource
from intentir.verify.gate1.model import (
    Boundary,
    Gate1Artifact,
    ImplementationConfig,
    LogicalSignature,
    ParameterKind,
    ParameterMode,
    ResultRelationKind,
    ResultRelationSpec,
    ScheduleSlot,
    ScheduleSlotKind,
    SemanticArchive,
    SignatureParameter,
)
from intentir.verify.gate1.pi import RealizationWitness
from intentir.verify.gate1.prompt import build_rewriter_payload
from intentir.verify.gate1.native_orchestrate import (
    CudaGate1Runner,
    TileLangGate1Runner,
)
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangKernelSource,
)
from intentir.verify.gate2.native import NativeExecutionSpec
from intentir.verify.adapters import NativeGate2ExecutionAdapter
from intentir.verify.scrc_workload_run import _build_workload_runtime
from intentir.verify.workload_registry import (
    prepare_registered_scrc_workload,
    registered_scrc_case_ids,
)


def _add_artifact(*, cuda: bool) -> Gate1Artifact:
    intent = IntentFunction.from_json(
        {
            "name": "scrc_j0",
            "inputs": [
                {"name": "t0", "dtype": "f32", "shape": ["x0", 16]},
                {"name": "t1", "dtype": "f32", "shape": ["x0", 16]},
            ],
            "outputs": [
                {"name": "t2", "dtype": "f32", "shape": ["x0", 16]}
            ],
            "body": [
                {
                    "id": "n0",
                    "op": "map_binary",
                    "result": "t2",
                    "operands": ["t0", "t1"],
                    "attrs": {"fn": "add"},
                }
            ],
        }
    )
    parameters = [
        SignatureParameter(
            "p0", ParameterKind.POINTER, ParameterMode.INPUT, "f32", 2,
            logical_ref="t0",
        ),
        SignatureParameter(
            "p1", ParameterKind.POINTER, ParameterMode.INPUT, "f32", 2,
            logical_ref="t1",
        ),
        SignatureParameter(
            "p2", ParameterKind.POINTER, ParameterMode.OUTPUT, "f32", 2,
            logical_ref="t2",
        ),
    ]
    if cuda:
        parameters.append(
            SignatureParameter(
                "p3", ParameterKind.SCALAR, ParameterMode.VALUE, "i32",
                logical_ref="x0",
            )
        )
    return Gate1Artifact(
        SemanticArchive(
            "scrc_k0",
            intent,
            {"n0": "body[0]"},
            Boundary(
                result_relations=[
                    ResultRelationSpec("t2", ResultRelationKind.TOLERANCE)
                ]
            ),
            signature=LogicalSignature(parameters),
        ),
        ImplementationConfig(
            [
                ScheduleSlot(
                    "c0",
                    ScheduleSlotKind.COMPILER,
                    "launch_block_x",
                )
            ],
            {"c0": 16},
        ),
        RealizationWitness.from_json(
            {
                "regions": [
                    {"id": "r0", "parent": None, "placement": "root"}
                ],
                "instances": [
                    {
                        "id": "i0",
                        "semantic_ref": "n0",
                        "region": "r0",
                        "mechanism": "realize",
                    }
                ],
                "routes": [],
                "order": [],
                "sharing": [],
                "refinements": [],
            }
        ),
    )


def test_tilelang_dataset_workload_materializes_real_tir_for_lift() -> None:
    prepared = prepare_frontend_dataset_workload(
        FRONTEND_DATASET_HARNESSES["tilelang:add2d"]
    )

    assert prepared.request.frontend == "tilelang"
    assert isinstance(prepared.request.source.gate_source, TileLangKernelSource)
    assert isinstance(prepared.request.binding.gate1_case, TileLangCompileCase)
    assert prepared.request.binding.values == {"c0": 16}
    assert prepared.request.binding.gate1_case.schedule == {"c0": 16}
    assert prepared.source_slice.source_kind == "materialized_tir"
    assert "T.launch_thread(\"blockIdx.x\", M)" in prepared.source_slice.code
    assert prepared.source_slice.parameters == ("A_handle", "B_handle", "C_handle")
    assert prepared.source_to_opaque == {
        "A_handle": "p0",
        "B_handle": "p1",
        "C_handle": "p2",
    }
    request = prepare_scrc_lift(
        prepared.source_slice.code,
        compiler_view=prepared.compiler_view,
        source_symbol=prepared.source_slice.function,
        frontend="tilelang",
    )
    assert request.anonymous_compiler_view is not None
    assert request.anonymous_compiler_view["parameter_order"] == ["p0", "p1", "p2"]
    view = request.anonymous_compiler_view
    assert view["schedule_values"] == {"c0": 16}
    assert view["compiler_slots"] == ["c0"]
    assert view["compiler_roles"] == {"c0": "launch_block_x"}
    source_extent = view["grid_contract"][0]
    assert source_extent.startswith("v")
    assert [item["shape"] for item in view["arguments"]] == [[4, 16]] * 3
    assert [item["shape_contract"] for item in view["arguments"]] == [
        ["h0", 16]
    ] * 3
    assert view["shape_coordinates"] == {"h0": 4}
    assert view["grid_shape_coordinates"] == ["h0", None, None]
    assert "M" not in json.dumps(view, sort_keys=True)


def test_tilelang_factory_execution_requires_the_pinned_dataset_source() -> None:
    adapter = replace(
        FRONTEND_DATASET_HARNESSES["tilelang:add2d"],
        source_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="source digest differs"):
        prepare_frontend_dataset_workload(adapter)


def test_cuda_dataset_workload_preserves_real_generated_source_and_abi() -> None:
    prepared = prepare_frontend_dataset_workload(
        FRONTEND_DATASET_HARNESSES["cuda:tilelang_add2d"]
    )

    assert prepared.request.frontend == "cuda"
    assert isinstance(prepared.request.source.gate_source, CudaKernelSource)
    assert isinstance(prepared.request.binding.gate1_case, CudaCompileCase)
    assert prepared.request.binding.values == {"c0": 16}
    assert prepared.request.binding.gate1_case.schedule == {"c0": 16}
    assert "__launch_bounds__(16, 1) main_kernel" in prepared.source_slice.code
    assert prepared.source_slice.parameters == ("A", "B", "C", "M")
    assert prepared.source_to_opaque == {
        "A": "p0",
        "B": "p1",
        "C": "p2",
        "M": "p3",
    }
    request = prepare_scrc_lift(
        prepared.source_slice.code,
        compiler_view=prepared.compiler_view,
        source_symbol=prepared.source_slice.function,
        frontend="cuda",
    )
    assert request.anonymous_compiler_view is not None
    assert request.anonymous_compiler_view["schedule_values"] == {"c0": 16}
    assert request.anonymous_compiler_view["compiler_slots"] == ["c0"]
    assert request.anonymous_compiler_view["parameter_order"] == [
        "p0", "p1", "p2", "p3"
    ]
    assert request.anonymous_compiler_view["grid_contract"] == ["p3", "1", "1"]
    assert [
        item["shape_contract"]
        for item in request.anonymous_compiler_view["arguments"][:3]
    ] == [["h0", 16]] * 3
    assert request.anonymous_compiler_view["arguments"][3][
        "shape_coordinate"
    ] == "h0"
    assert request.anonymous_compiler_view["shape_coordinates"] == {"h0": 4}
    assert request.anonymous_compiler_view["grid_shape_coordinates"] == [
        "h0",
        None,
        None,
    ]


def test_dataset_shape_contract_blocks_dynamic_to_static_specialization() -> None:
    prepared = prepare_frontend_dataset_workload(
        FRONTEND_DATASET_HARNESSES["tilelang:add2d"]
    )
    request = prepare_scrc_lift(
        prepared.source_slice.code,
        compiler_view=prepared.compiler_view,
        source_symbol=prepared.source_slice.function,
        frontend="tilelang",
    )
    dynamic = _add_artifact(cuda=False)

    accepted = decode_scrc_lift_artifact(json.dumps(dynamic.to_json()), request)
    assert accepted.j.intent.inputs[0].shape == ["x0", 16]

    specialized = dynamic.to_json()
    function = specialized["j"]["intent"]["intent_function"]
    for tensor in [*function["inputs"], *function["outputs"]]:
        tensor["shape"][0] = 4
    checked = check_scrc_lift_candidate(
        json.dumps(specialized), scrc_candidate_check_context(request)
    )

    assert checked["status"] == "fail"
    assert any(
        item["rule_code"] == "construction_shape_contract_mismatch"
        for item in checked["diagnostics"]
    )
    replay_packet = build_rewriter_payload(dynamic)
    assert "shape_contract" not in json.dumps(replay_packet, sort_keys=True)
    assert request.anonymous_compiler_view is not None
    anonymous_source_extent = request.anonymous_compiler_view["grid_contract"][0]
    assert anonymous_source_extent not in json.dumps(replay_packet, sort_keys=True)


def test_static_source_shape_contract_cannot_be_generalized_by_current_value() -> None:
    prepared = prepare_frontend_dataset_workload(
        FRONTEND_DATASET_HARNESSES["tilelang:add2d"]
    )
    static_view = replace(
        prepared.compiler_view,
        arguments=tuple(
            replace(argument, shape_contract=(4, 16))
            for argument in prepared.compiler_view.arguments
        ),
        grid_shape_coordinates=(None, None, None),
    )
    request = prepare_scrc_lift(
        prepared.source_slice.code,
        compiler_view=static_view,
        source_symbol=prepared.source_slice.function,
        frontend="tilelang",
    )
    static_payload = _add_artifact(cuda=False).to_json()
    function = static_payload["j"]["intent"]["intent_function"]
    for tensor in [*function["inputs"], *function["outputs"]]:
        tensor["shape"][0] = 4

    accepted = decode_scrc_lift_artifact(json.dumps(static_payload), request)
    assert accepted.j.intent.inputs[0].shape == [4, 16]

    generalized = _add_artifact(cuda=False)
    checked = check_scrc_lift_candidate(
        json.dumps(generalized.to_json()), scrc_candidate_check_context(request)
    )
    assert checked["status"] == "fail"
    assert any(
        item["rule_code"] == "construction_shape_contract_mismatch"
        for item in checked["diagnostics"]
    )


def test_native_gate2_suite_uses_one_dataset_binding_for_both_frontends() -> None:
    for case_id, cuda in (
        ("tilelang:add2d", False),
        ("cuda:tilelang_add2d", True),
    ):
        prepared = prepare_frontend_dataset_workload(
            FRONTEND_DATASET_HARNESSES[case_id]
        )
        suite = build_frontend_dataset_gate2_suite(
            prepared, _add_artifact(cuda=cuda)
        )

        assert suite.suite_id == case_id
        assert suite.c_values == (("c0", 16),)
        assert len(suite.cases) == 1
        execution = suite.cases[0].triton
        assert isinstance(execution, NativeExecutionSpec)
        assert execution.grid == (4, 1, 1)
        assert execution.block == (16, 1, 1)
        assert suite.cases[0].refeval.dims_map == {"x0": 4}
        assert suite.cases[0].output_keys == ("output_p2",)


def test_native_gate2_rejects_conflicting_shape_and_scalar_dimensions() -> None:
    prepared = prepare_frontend_dataset_workload(
        FRONTEND_DATASET_HARNESSES["cuda:tilelang_add2d"]
    )
    parameters = (
        *prepared.runtime_parameters[:-1],
        replace(prepared.runtime_parameters[-1], value=5),
    )
    inconsistent = replace(prepared, runtime_parameters=parameters)

    with pytest.raises(ValueError, match="scalar is inconsistent"):
        build_frontend_dataset_gate2_suite(
            inconsistent, _add_artifact(cuda=True)
        )


def test_frontend_neutral_registry_and_runtime_dispatch_native_cases() -> None:
    assert "tilelang:add2d" in registered_scrc_case_ids(frontend="tilelang")
    assert "cuda:tilelang_add2d" in registered_scrc_case_ids(frontend="cuda")
    provider = ProviderConfig(
        "test", "https://provider.invalid/v1", "unused", "test-model"
    )
    for case_id, runner_type in (
        ("tilelang:add2d", TileLangGate1Runner),
        ("cuda:tilelang_add2d", CudaGate1Runner),
    ):
        prepared = prepare_registered_scrc_workload(case_id)
        runner, lift, writers = _build_workload_runtime(
            prepared,
            provider=provider,
            run_masked_j=False,
            lift_backend="miniswe",
            replay_backend="miniswe",
            replay_width=2,
        )
        gate1 = runner.dependencies.gate1_factory(1)

        assert isinstance(gate1.runner, runner_type)
        assert len(gate1.runner.writers) == 2
        assert len({id(writer) for writer in gate1.runner.writers}) == 2
        assert all(
            writer.frontend.value == prepared.request.frontend
            for writer in gate1.runner.writers
        )
        assert lift.frontend == prepared.request.frontend
        assert isinstance(runner.dependencies.gate2, NativeGate2ExecutionAdapter)
        assert len(writers) == 1
        assert writers[0].writers == gate1.runner.writers


def test_dataset_harness_projection_is_wide_but_fail_closed_for_opaque_abi() -> None:
    projection = project_frontend_dataset_harnesses()

    assert sum(item.frontend == "tilelang" for item in projection.harnesses.values()) == 26
    assert sum(item.frontend == "cuda" for item in projection.harnesses.values()) == 30
    assert set(item.case_id for item in projection.diagnostics) == {
        "tilelang:gemm",
        "tilelang:matmul_bias_relu2d",
        "tilelang:matmul_fused_epilogue2d",
        "tilelang:matmul_relu2d",
        "tilelang:mlp2d",
        "cuda:tilelang_gemm",
        "cuda:tilelang_matmul_bias_relu2d",
        "cuda:tilelang_matmul_fused_epilogue2d",
        "cuda:tilelang_matmul_relu2d",
        "cuda:tilelang_mlp2d",
    }
    projected_canary = projection.harnesses["tilelang:add2d"]
    # Even the legacy canary IDs go through the corpus-wide projection policy;
    # they are not replaced by the hand-authored finite smoke fixtures below.
    assert projected_canary.dimensions == (("M", 16),)
    assert projected_canary.factory_kwargs == (("n", 16), ("threads", 128))
    assert projected_canary.seed != FRONTEND_DATASET_HARNESSES["tilelang:add2d"].seed
    assert projected_canary.source_sha256 == FRONTEND_DATASET_HARNESSES[
        "tilelang:add2d"
    ].source_sha256
    assert len(projected_canary.descriptor_sha256) == 64
    encoded = json.dumps(projection.to_json(), sort_keys=True)
    assert "intent_seed" not in encoded
    assert "source.py" not in encoded
    assert "/home/" not in encoded


def test_projected_dataset_harnesses_prepare_every_nonopaque_case() -> None:
    projection = project_frontend_dataset_harnesses()

    prepared = {
        case_id: prepare_frontend_dataset_workload(harness)
        for case_id, harness in projection.harnesses.items()
    }

    assert len(prepared) == 56
    assert prepared["tilelang:add_bias2d"].compiler_view.parameter_order == (
        "inp_handle",
        "bias_handle",
        "out_handle",
    )
    assert prepared["cuda:naive_gemm"].compiler_view.grid_shape_coordinates == (
        None,
        None,
        None,
    )
    tilelang_mask = prepared["tilelang:masked_softmax2d"]
    cuda_mask = prepared["cuda:tilelang_masked_softmax2d"]
    assert next(
        item.dtype
        for item in tilelang_mask.runtime_parameters
        if item.semantic_name == "mask"
    ) == "i1"
    assert next(
        item.dtype
        for item in cuda_mask.runtime_parameters
        if item.semantic_name == "mask"
    ) == "i8"
    assert tilelang_mask.compiler_view.arguments[1].dtype == "i1"
    assert tilelang_mask.request.binding.gate1_case.parameters[1].dtype == "i1"


def test_dataset_shape_coordinates_remain_injective_when_values_coincide() -> None:
    prepared = prepare_frontend_dataset_workload(
        project_frontend_dataset_harnesses().harnesses["cuda:naive_gemm"]
    )
    request = prepare_scrc_lift(
        prepared.source_slice.code,
        compiler_view=prepared.compiler_view,
        source_symbol=prepared.source_slice.function,
        frontend="cuda",
    )

    assert request.anonymous_compiler_view is not None
    view = request.anonymous_compiler_view
    assert view["shape_coordinates"] == {"h0": 4, "h1": 4, "h2": 4}
    assert [item["shape_contract"] for item in view["arguments"][:3]] == [
        ["h0", "h1"],
        ["h1", "h2"],
        ["h0", "h2"],
    ]
    assert all(name not in json.dumps(view, sort_keys=True) for name in ("A", "B", "M", "N", "K"))


@pytest.mark.skipif(
    __import__("os").environ.get("INTENTIR_RUN_NATIVE_DATASET_SMOKE") != "1",
    reason="set INTENTIR_RUN_NATIVE_DATASET_SMOKE=1 for real GPU execution",
)
def test_native_dataset_add2d_closes_real_gate1_and_gate2() -> None:
    import importlib.util
    from pathlib import Path

    import torch

    from intentir.backend.toolchain.tools import INTENT_OPT
    from intentir.ir.canonical_j import canonicalize_gate1_artifact
    from intentir.verify.gate1.cuda_compiler import CudaCompiler, find_nvcc
    from intentir.verify.gate1.frontend import frontend_gate1_capability
    from intentir.verify.gate1.model import Frontend, Gate1Status
    from intentir.verify.gate1.orchestrate import Gate1Config
    from intentir.verify.gate1.replay import StaticReplayWriter
    from intentir.verify.gate1.tilelang_compiler import TileLangCompiler
    from intentir.verify.gate2.native import CudaRunner, TileLangRunner
    from intentir.verify.gate2.orchestrate import Gate2Runner

    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if find_nvcc() is None:
        pytest.skip("nvcc unavailable")
    if importlib.util.find_spec("tilelang") is None:
        pytest.skip("TileLang unavailable")
    if not Path(INTENT_OPT).is_file():
        pytest.skip("matching intent-opt build unavailable")

    tilelang_replay = '''import tilelang.language as T
X0 = T.dynamic("x0")
@T.prim_func
def scrc_k0(p0: T.Tensor((X0, 16), "float32"), p1: T.Tensor((X0, 16), "float32"), p2: T.Tensor((X0, 16), "float32")):
    with T.Kernel(X0, threads=16) as (v0,):
        for v1 in T.Parallel(16):
            p2[v0, v1] = p0[v0, v1] + p1[v0, v1]
'''
    cuda_replay = '''#include <stdint.h>
extern "C" __global__ void __launch_bounds__(16, 1) scrc_k0(const float* __restrict__ p0, const float* __restrict__ p1, float* __restrict__ p2, int p3) {
  p2[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))] = (p0[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))] + p1[((((int64_t)((int)blockIdx.x)) * (int64_t)16) + ((int64_t)((int)threadIdx.x)))]);
}
'''
    for case_id, cuda, replay in (
        ("tilelang:add2d", False, tilelang_replay),
        ("cuda:tilelang_add2d", True, cuda_replay),
    ):
        prepared = prepare_frontend_dataset_workload(
            FRONTEND_DATASET_HARNESSES[case_id]
        )
        artifact = canonicalize_gate1_artifact(_add_artifact(cuda=cuda))
        frontend = Frontend(prepared.request.frontend)
        compiler = (
            CudaCompiler(timeout=240)
            if cuda
            else TileLangCompiler(timeout=300)
        )
        gate1 = frontend_gate1_capability(frontend).build_runner(
            compiler=compiler,
            writer=StaticReplayWriter(replay, frontend=frontend),
            config=Gate1Config(run_masked_j=False),
        ).run(
            artifact,
            source=prepared.request.source.gate_source,
            source_case=prepared.request.binding.gate1_case,
        )
        assert gate1.status is Gate1Status.REPLAY_CLOSED
        if cuda:
            source_runner = CudaRunner(CudaCompiler(timeout=240))
            replay_runner = CudaRunner(CudaCompiler(timeout=240))
        else:
            source_runner = TileLangRunner()
            replay_runner = TileLangRunner()
        gate2 = NativeGate2ExecutionAdapter(
            frontend.value,
            Gate2Runner(
                source_runner=source_runner,
                replay_runner=replay_runner,
            ),
            lambda candidate, _gate1, _binding: (
                build_frontend_dataset_gate2_suite(prepared, candidate)
            ),
        ).execute(artifact, gate1, prepared.request)
        assert gate2.passed, gate2.to_json()
