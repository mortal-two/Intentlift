from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1.codegen import UnsafeReplayCodeError
from intentir.verify.gate1.compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.diagnose import RepairFeedback
from intentir.verify.gate1.compare import compare_ttir
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    Frontend,
    Gate1Artifact,
    Gate1Status,
    KernelInterface,
    KernelParameter,
    LocationMap,
    ParameterKind,
    ParameterMode,
    ReplayPlan,
    ReplayStep,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SemanticArchive,
)
from intentir.verify.gate1.orchestrate import CandidateRecord, Gate1Config, Gate1Runner
from intentir.verify.gate1.replay import (
    ReplayGeneration,
    StaticReplayWriter,
    restore_masked_replay_module,
)
from intentir.verify.gate1.rules import default_triton_rule_registry


GOOD_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

BAD_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %one = arith.constant 1.000000e+00 : f32
    %value = tt.load %x : !tt.ptr<f32>
    %wrong = arith.addf %value, %one : f32
    tt.store %out, %wrong : !tt.ptr<f32>
    tt.return
  }
}
'''

GOOD_CODE = '''import triton
import triton.language as tl

@triton.jit
def kernel(x, out, n, BLOCK: tl.constexpr):
    return
'''

REAL_COPY_CODE = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["n"])
def kernel(x, out, n, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < n
    value = tl.load(x + offset, mask=mask, other=0.0)
    tl.store(out + offset, value, mask=mask)
'''

REAL_MASKED_FAILURE_CODE = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["scrc_p2"])
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    offset = tl.program_id(0) * scrc_p3 + tl.arange(0, scrc_p3)
    mask = offset < scrc_p2
    value = tl.gate1_masked_ablation_unknown(scrc_p0 + offset)
    tl.store(scrc_p1 + offset, value, mask=mask)
'''

OPAQUE_MASKED_COMPILE_FAIL = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["scrc_p2"])
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    "MASKED_FAIL"
    return
'''


def _artifact() -> Gate1Artifact:
    intent = IntentFunction.from_json(
        {
            "name": "identity_semantics",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["N"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["N"]}],
            "body": [
                {
                    "op": "cast",
                    "result": "out",
                    "operands": ["x"],
                    "attrs": {"dtype": "f32"},
                }
            ],
        }
    )
    return Gate1Artifact(
        j=SemanticArchive(
            "kernel",
            intent,
            {"identity_node": "body[0]"},
            Boundary(
                domains={"lane_domain": "lane < N"},
                accesses=[
                    BoundaryAccess(
                        "load_x",
                        AccessKind.LOAD,
                        "x",
                        predicate="lane < N",
                        fill_value=0.0,
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                    BoundaryAccess(
                        "store_out",
                        AccessKind.STORE,
                        "out",
                        predicate="lane < N",
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                ],
            ),
        ),
        interface=KernelInterface(
            kernel_id="kernel",
            function_name="kernel",
            frontend=Frontend.TRITON,
            parameters=[
                KernelParameter(
                    "x",
                    ParameterKind.POINTER,
                    ParameterMode.INPUT,
                    "f32",
                    1,
                    tensor_ref="x",
                ),
                KernelParameter(
                    "out",
                    ParameterKind.POINTER,
                    ParameterMode.OUTPUT,
                    "f32",
                    1,
                    tensor_ref="out",
                ),
                KernelParameter(
                    "n", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"
                ),
                KernelParameter("BLOCK", ParameterKind.CONSTEXPR),
            ],
            grid=["ceil_div(n, BLOCK)"],
            compiler_slots=["num_warps"],
            do_not_specialize=["n"],
        ),
        schedule=ScheduleSpec(
            slots=[
                ScheduleSlot("BLOCK", ScheduleSlotKind.CONSTEXPR),
                ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER),
            ],
            values={"BLOCK": 64, "num_warps": 4},
        ),
        plan=ReplayPlan(
            [
                ReplayStep("bind_program_domain", refs=["lane_domain"]),
                ReplayStep("realize_masked_access", refs=["load_x", "store_out"]),
                ReplayStep("realize_elementwise", node_ref="identity_node"),
            ]
        ),
        location_map=LocationMap("private.py", []),
        source="SOURCE_CANARY_91F7",
    )


def _case() -> CompileCase:
    return CompileCase(
        arguments={
            "x": RuntimeArgument.tensor((65,), "float32"),
            "out": RuntimeArgument.tensor((65,), "float32"),
            "n": RuntimeArgument.scalar(65, "int32"),
        },
        grid=(2,),
        schedule={"BLOCK": 64, "num_warps": 4},
    )


class FakeCompiler:
    def __init__(self, *, target: str = "sm_test") -> None:
        self.target = target
        self.calls: list[str] = []

    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        self.calls.append(code)
        if "COMPILE_FAIL" in code or "MASKED_FAIL" in code:
            return CompileResult(ok=False, diagnostic='File "/secret/worker.py", line 7: bad replay')
        ttir = BAD_TTIR if "BAD" in code else GOOD_TTIR
        return CompileResult(
            ok=True,
            ttir=ttir,
            ttgir="ttgir",
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": self.target},
        )


class CaptureWriter:
    def __init__(
        self,
        normal: list[str],
        *,
        masked: str = OPAQUE_MASKED_COMPILE_FAIL,
        repeated_id: bool = False,
    ) -> None:
        self.normal = normal
        self.masked = masked
        self.repeated_id = repeated_id
        self.normal_index = 0
        self.masked_index = 0
        self.calls: list[tuple[RepairFeedback | None, ReplayGeneration | None, bool]] = []

    def generate(
        self,
        artifact: Gate1Artifact,
        registry,
        *,
        feedback: RepairFeedback | None = None,
        base_candidate: ReplayGeneration | None = None,
        masked: bool = False,
    ) -> ReplayGeneration:
        self.calls.append((feedback, base_candidate, masked))
        if masked:
            code = self.masked
            index = self.masked_index
            self.masked_index += 1
            generation_id = f"masked:{index}"
        else:
            index = self.normal_index
            code = self.normal[min(index, len(self.normal) - 1)]
            self.normal_index += 1
            generation_id = "reused" if self.repeated_id else f"normal:{index}"
        digest = hashlib.sha256(code.encode()).hexdigest()
        return ReplayGeneration(
            code=code,
            prompt_sha256="prompt",
            requested_model="capture",
            generation_id=generation_id,
            code_sha256=digest,
            response_model="capture",
            masked=masked,
            feedback=feedback,
        )


def _runner(**config) -> Gate1Runner:
    return Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(**config),
    )


def _source(function: str = "kernel") -> KernelSource:
    return KernelSource.from_code("SOURCE", function, trusted=True)


def test_static_masked_writer_preserves_opaque_evidence_for_runner() -> None:
    writer = StaticReplayWriter(GOOD_CODE, masked=REAL_MASKED_FAILURE_CODE)
    generation = writer.generate(
        _artifact(), default_triton_rule_registry(), masked=True
    )

    assert generation.masked
    assert "def scrc_k0(scrc_p0, scrc_p1, scrc_p2" in generation.code
    restored = restore_masked_replay_module(generation.code, _artifact())
    assert "def kernel(x, out, n, BLOCK: tl.constexpr):" in restored
    assert 'do_not_specialize=[\'n\']' in restored


def test_masked_abi_restore_renames_model_locals_that_collide() -> None:
    opaque = '''import triton
import triton.language as tl

@triton.jit
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    x = tl.load(scrc_p0)
    tl.static_assert(True, "x", scrc_p0=1)
    tl.store(scrc_p1, x)
'''
    restored = restore_masked_replay_module(opaque, _artifact())

    assert "def kernel(x, out, n, BLOCK: tl.constexpr):" in restored
    assert "scrc_local_0 = tl.load(x)" in restored
    assert "tl.static_assert(True, 'x', scrc_p0=1)" in restored
    assert "tl.store(out, scrc_local_0)" in restored


def test_masked_abi_restore_rejects_a_wrong_public_signature() -> None:
    wrong_signature = REAL_MASKED_FAILURE_CODE.replace(
        "scrc_p2, scrc_p3: tl.constexpr",
        "scrc_p3: tl.constexpr, scrc_p2",
    )

    with pytest.raises(UnsafeReplayCodeError, match="opaque ABI"):
        restore_masked_replay_module(wrong_signature, _artifact())


def test_bounded_repair_stability_and_masked_ablation_close() -> None:
    writer = CaptureWriter(["BAD", "GOOD", "GOOD"])
    result = _runner(stable_closures=2).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert len(result.candidates) == 3
    assert [item.comparison is not None and item.comparison.hard_pass for item in result.candidates] == [False, True, True]
    assert writer.calls[0][0] is None and writer.calls[0][1] is None
    feedback, base, masked = writer.calls[1]
    assert not masked and feedback is not None and base is not None
    feedback_text = str(feedback.to_json())
    assert "SOURCE_CANARY_91F7" not in feedback_text
    assert "源=" not in feedback_text
    assert "arith.addf" not in feedback_text
    assert writer.calls[2][0] is None and writer.calls[2][1] is None
    assert writer.calls[3][2]
    assert result.masked_compile is not None and not result.masked_compile.ok
    assert result.masked_generation is not None
    assert result.masked_generation.masked
    assert result.to_json()["masked_generation"]["generation_id"].startswith(
        "masked:"
    )
    assert result.leakage_ratio == 0.0


def test_runner_restores_opaque_masked_code_and_rejects_real_leakage() -> None:
    compiler = FakeCompiler()
    writer = CaptureWriter(["GOOD"], masked=REAL_MASKED_FAILURE_CODE)
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(stable_closures=1),
    ).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert result.leakage_ratio == 1.0
    assert result.masked_generation is not None
    assert "scrc_p0" in result.masked_generation.code
    assert "def kernel(x, out, n, BLOCK: tl.constexpr):" in compiler.calls[-1]


def test_initial_plus_two_repairs_is_exactly_three_candidates() -> None:
    writer = CaptureWriter(["COMPILE_FAIL"])
    result = _runner(stable_closures=1, run_masked_ablation=False).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert len(result.candidates) == 3
    assert len(writer.calls) == 3
    assert result.best_candidate is not None


def test_regressed_candidate_does_not_replace_best_repair_base() -> None:
    writer = CaptureWriter(["BAD", "COMPILE_FAIL", "COMPILE_FAIL"])
    result = _runner(stable_closures=1, run_masked_ablation=False).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.best_candidate is result.candidates[0]
    assert result.candidates[1].regressed
    assert writer.calls[2][1] is result.candidates[0].generation


def test_reused_generation_id_cannot_satisfy_stability() -> None:
    writer = CaptureWriter(["GOOD"], repeated_id=True)
    result = _runner(stable_closures=2).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert any(
        candidate.diagnosis
        and "generation_id" in " ".join(candidate.diagnosis.mismatches)
        for candidate in result.candidates
    )


def test_missing_masked_generation_fails_closed() -> None:
    writer = StaticReplayWriter(GOOD_CODE)
    result = _runner().run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert any("ablation was unavailable" in item for item in result.diagnostics)


def test_declared_function_and_alias_contracts_are_rejected_before_writer() -> None:
    writer = CaptureWriter(["GOOD"])
    function_mismatch = _runner().run(
        _source("other"),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )
    assert function_mismatch.status is Gate1Status.LIFT_REJECTED
    assert writer.calls == []

    artifact = _artifact()
    aliased = replace(
        artifact,
        interface=replace(
            artifact.interface,
            parameters=[
                artifact.interface.parameters[0],
                replace(artifact.interface.parameters[1], alias_of="x"),
                *artifact.interface.parameters[2:],
            ],
        ),
    )
    writer2 = CaptureWriter(["GOOD"])
    alias_result = _runner().run(
        _source(), aliased, _case(), default_triton_rule_registry(), writer2
    )
    assert alias_result.status is Gate1Status.LIFT_REJECTED
    assert writer2.calls == []


def test_rule_registry_is_versioned_and_has_a_stable_digest() -> None:
    first = default_triton_rule_registry()
    second = default_triton_rule_registry()
    assert first.digest == second.digest
    assert all(card.version for card in first.cards)
    assert {
        "bind_program_domain",
        "materialize_index_tile",
        "realize_masked_access",
        "realize_elementwise",
        "realize_reduce",
        "realize_k_tiled_dot",
    } <= {card.id for card in first.cards}


def test_best_candidate_rank_keeps_interface_ahead_of_semantic_score() -> None:
    interface_mismatch = compare_ttir(
        GOOD_TTIR,
        GOOD_TTIR.replace("%x: !tt.ptr<f32>", "%x: !tt.ptr<f16>"),
    )
    semantic_mismatch = compare_ttir(GOOD_TTIR, BAD_TTIR)
    compiled = CompileResult(ok=True, ttir=GOOD_TTIR)
    interface_record = CandidateRecord(
        0, 0, None, compiled, interface_mismatch, None
    )
    semantic_record = CandidateRecord(
        0, 1, None, compiled, semantic_mismatch, None
    )

    assert interface_mismatch.score > semantic_mismatch.score
    assert semantic_record.rank_tuple > interface_record.rank_tuple


def test_real_triton_runner_closes_without_launching(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real Gate1 compile integration")

    writer = StaticReplayWriter(
        REAL_COPY_CODE,
        masked=REAL_MASKED_FAILURE_CODE,
    )
    runner = Gate1Runner(
        compiler=TritonCompiler(cache_dir=tmp_path, timeout=180.0),
        config=Gate1Config(),
    )
    result = runner.run(
        KernelSource.from_code(REAL_COPY_CODE, "kernel", trusted=True),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.source_compile is not None and result.source_compile.ok
    assert result.best_candidate is not None
    assert result.best_candidate.comparison is not None
    assert result.best_candidate.comparison.hard_pass
    assert result.masked_compile is not None and not result.masked_compile.ok
