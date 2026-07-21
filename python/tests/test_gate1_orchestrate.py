from __future__ import annotations

import hashlib
import inspect
import random
from dataclasses import replace

import pytest

from intentir.extraction.providers.client import ProviderError
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.codegen import UnsafeReplayCodeError
from intentir.verify.gate1.compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.diagnose import sanitize_replay_diagnostic
from intentir.verify.gate1.interventions import (
    PAIRED_INTERVENTION_FAMILY_POLICY,
    InterventionStatus,
)
from intentir.verify.gate1.leakage import (
    LeakageKind,
    LeakagePolicy,
    scan_static_leakage,
)
from intentir.verify.gate1.compare import compare_ttir
from intentir.verify.gate1.composer import materialize_replay_plan
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
from intentir.verify.gate1.orchestrate import (
    CandidateRecord,
    Gate1Config,
    Gate1Runner,
    MAX_EXACT_COMPOSED_PLAN_STEPS,
    MASKED_J_FAMILY_POLICY,
    ReliftTrigger,
    RELIFT_STRUCTURAL_GUIDANCE_POLICY,
    _build_structural_relift_guidance,
    _exact_composed_leakage_policy,
    _validate_compiled_interface,
    canonical_artifact_digest,
)
from intentir.verify.gate1.pi_ablations import _case_with_artifact_schedule
from intentir.verify.gate1.replay import (
    ReplayGeneration,
    ResponsesReplayWriter,
    StaticReplayWriter,
    restore_masked_replay_module,
)
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate1.specialization import (
    Gate1SpecializationFamily,
    Gate1SpecializationProfile,
    specialization_family_contract_digest,
)
from intentir.verify.gate1.ttir import fingerprint_ttir


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

LOOP_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = tt.load %x : !tt.ptr<f32>
    %value = scf.for %iv = %c0 to %n step %c1 iter_args(%carry = %init) -> (f32) : i32 {
      %next = arith.addf %carry, %init : f32
      scf.yield %next : f32
    }
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

ADD_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    %sum = arith.addf %value, %value : f32
    tt.store %out, %sum : !tt.ptr<f32>
    tt.return
  }
}
'''

SUB_TTIR = ADD_TTIR.replace("arith.addf", "arith.subf")

SPECIALIZED_SCALAR_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

UNUSED_POINTER_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %label: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

WRITTEN_LABEL_TTIR = r'''
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %label: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %value = tt.load %x : !tt.ptr<f32>
    tt.store %label, %value : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
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

OPAQUE_MASKED_EMPTY = '''import triton
import triton.language as tl

@triton.jit
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    pass
'''

OPAQUE_MASKED_SEMANTIC_DRIFT = '''import triton
import triton.language as tl

@triton.jit(do_not_specialize=["scrc_p2"])
def scrc_k0(scrc_p0, scrc_p1, scrc_p2, scrc_p3: tl.constexpr):
    offset = tl.program_id(0) * scrc_p3 + tl.arange(0, scrc_p3)
    mask = offset < scrc_p2
    value = tl.load(scrc_p0 + offset, mask=mask, other=0.0)
    BAD = value + 1.0
    tl.store(scrc_p1 + offset, BAD, mask=mask)
'''


def _artifact() -> Gate1Artifact:
    intent = IntentFunction.from_json(
        {
            "name": "identity_semantics",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["n"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["n"]}],
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
    artifact = Gate1Artifact(
        j=SemanticArchive(
            "kernel",
            intent,
            {"identity_node": "body[0]"},
            Boundary(
                domains={"lane_domain": "lane < n"},
                accesses=[
                    BoundaryAccess(
                        "load_x",
                        AccessKind.LOAD,
                        "x",
                        predicate="lane < n",
                        fill_value=0.0,
                        node_ref="identity_node",
                        index_expression="lane",
                    ),
                    BoundaryAccess(
                        "store_out",
                        AccessKind.STORE,
                        "out",
                        predicate="lane < n",
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
                ReplayStep("bind_runtime_scalar", refs=["n"]),
                ReplayStep("declare_schedule_slot", refs=["BLOCK"]),
                ReplayStep("declare_schedule_slot", refs=["num_warps"]),
                ReplayStep("materialize_index_tile", refs=["load_x"]),
                ReplayStep("realize_masked_access", refs=["load_x"]),
                ReplayStep("realize_cast", node_ref="identity_node"),
                ReplayStep("materialize_index_tile", refs=["store_out"]),
                ReplayStep("realize_masked_access", refs=["store_out"]),
            ]
        ),
        location_map=LocationMap("private.py", []),
        source="SOURCE_CANARY_91F7",
    )
    return materialize_replay_plan(
        artifact, default_triton_rule_registry()
    )[0]


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


def _profile_case(n: int) -> CompileCase:
    return CompileCase(
        arguments={
            "x": RuntimeArgument.tensor((n,), "float32"),
            "out": RuntimeArgument.tensor((n,), "float32"),
            "n": RuntimeArgument.scalar(n, "int32"),
        },
        grid=((n + 63) // 64,),
        schedule={"BLOCK": 64, "num_warps": 4},
    )


def _fixed_c_family(
    extents: tuple[int, ...] = (65, 1),
) -> Gate1SpecializationFamily:
    profiles = tuple(
        Gate1SpecializationProfile(f"n{n}", _profile_case(n))
        for n in extents
    )
    return Gate1SpecializationFamily(
        "anti-leakage-fixed-c-family",
        "n65",
        profiles,
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
        if "REPLAY_SUB" in code:
            ttir = SUB_TTIR
        elif "SOURCE_ADD" in code or "REPLAY_ADD" in code:
            ttir = ADD_TTIR
        else:
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


class FamilySelectiveCompiler(FakeCompiler):
    def __init__(
        self,
        *,
        ignored_intervention_extent: int | None = None,
        intervention_compile_failure_extent: int | None = None,
        masked_hard_closure_extent: int | None = None,
        masked_compile_failure_extent: int | None = None,
    ) -> None:
        super().__init__()
        self.ignored_intervention_extent = ignored_intervention_extent
        self.intervention_compile_failure_extent = (
            intervention_compile_failure_extent
        )
        self.masked_hard_closure_extent = masked_hard_closure_extent
        self.masked_compile_failure_extent = masked_compile_failure_extent
        self.profile_calls: list[tuple[str, int]] = []

    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        n = int(case.arguments["n"].value)
        self.calls.append(code)
        self.profile_calls.append((code, n))
        masked = "BAD = value + 1.0" in code
        if masked and n == self.masked_compile_failure_extent:
            return CompileResult(
                ok=False,
                diagnostic="masked profile compile failure",
            )
        if (
            "REPLAY_SUB" in code
            and n == self.intervention_compile_failure_extent
        ):
            return CompileResult(
                ok=False,
                diagnostic="intervention profile compile failure",
            )
        if "REPLAY_SUB" in code:
            ttir = (
                ADD_TTIR
                if n == self.ignored_intervention_extent
                else SUB_TTIR
            )
        elif "SOURCE_ADD" in code or "REPLAY_ADD" in code:
            ttir = ADD_TTIR
        elif masked:
            ttir = (
                GOOD_TTIR
                if n == self.masked_hard_closure_extent
                else BAD_TTIR
            )
        else:
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
            target={"arch": "sm_test"},
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
        self.calls: list[bool] = []
        self.artifacts: list[Gate1Artifact] = []

    def generate(
        self,
        artifact: Gate1Artifact,
        registry,
        *,
        masked: bool = False,
        plan_override: ReplayPlan | None = None,
        ablation_kind: str | None = None,
    ) -> ReplayGeneration:
        self.calls.append(masked)
        self.artifacts.append(artifact)
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
            ablation_kind=ablation_kind,
        )


def test_active_replay_writer_exposes_no_inner_repair_channel() -> None:
    parameters = inspect.signature(ResponsesReplayWriter.generate).parameters

    assert "base_candidate" not in parameters
    assert "repair_candidate" not in parameters
    assert "control_candidate" not in parameters


def test_gate1_defaults_exclude_paired_semantic_interventions() -> None:
    config = Gate1Config()

    assert not config.run_semantic_interventions
    assert config.run_masked_ablation


def test_pi_ablation_compile_case_uses_the_frozen_artifact_schedule() -> None:
    unscheduled = replace(_case(), schedule={})
    scheduled = _case_with_artifact_schedule(unscheduled, _artifact())
    assert scheduled.schedule == {"BLOCK": 64, "num_warps": 4}

    conflicting = replace(_case(), schedule={"BLOCK": 32, "num_warps": 4})
    with pytest.raises(ValueError, match="differs from the frozen artifact"):
        _case_with_artifact_schedule(conflicting, _artifact())


def _runner(**config) -> Gate1Runner:
    config.setdefault("run_semantic_interventions", False)
    # Generic orchestration tests opt into the terminal masked-J attack only
    # when that behavior is under test. Production Gate1Config remains
    # default-on.
    config.setdefault("run_masked_ablation", False)
    return Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(**config),
    )


def _source(function: str = "kernel") -> KernelSource:
    return KernelSource.from_code("SOURCE", function, trusted=True)


def test_composer_shadow_is_reported_but_does_not_change_gate1_verdict() -> None:
    registry = default_triton_rule_registry()
    enabled = _runner(run_masked_ablation=False).run(
        _source(), _artifact(), _case(), registry, CaptureWriter(["GOOD"])
    )
    disabled = _runner(
        run_masked_ablation=False,
        run_composer_shadow=False,
    ).run(
        _source(), _artifact(), _case(), registry, CaptureWriter(["GOOD"])
    )

    assert enabled.status is disabled.status is Gate1Status.REPLAY_CLOSED
    assert len(enabled.composer_shadow) == 1
    assert disabled.composer_shadow == ()
    shadow_json = enabled.to_json()["composer_shadow"][0]
    assert shadow_json["status"] in {"match", "mismatch", "failed"}


@pytest.mark.parametrize(
    ("steps", "expected_limit", "extended"),
    [
        (64, 64, False),
        (65, 65, True),
        (87, 87, True),
        (MAX_EXACT_COMPOSED_PLAN_STEPS + 1,
         MAX_EXACT_COMPOSED_PLAN_STEPS, False),
    ],
)
def test_exact_composed_step_budget_is_bounded(
    steps: int, expected_limit: int, extended: bool
) -> None:
    plan = ReplayPlan(
        [ReplayStep("realize_const", refs=[f"node_{index}"]) for index in range(steps)]
    )

    policy, was_extended = _exact_composed_leakage_policy(
        LeakagePolicy(), plan
    )

    assert policy.max_steps == expected_limit
    assert was_extended is extended
    assert policy.max_literal_chars == LeakagePolicy().max_literal_chars
    assert policy.max_total_literal_chars == LeakagePolicy().max_total_literal_chars

    if steps > MAX_EXACT_COMPOSED_PLAN_STEPS:
        oversized_policy, oversized_extended = _exact_composed_leakage_policy(
            LeakagePolicy(max_steps=1024), plan
        )
        assert oversized_policy.max_steps == MAX_EXACT_COMPOSED_PLAN_STEPS
        assert oversized_extended is False


def test_runner_extends_only_exact_composer_mechanical_step_budget() -> None:
    artifact = _artifact()
    registry = default_triton_rule_registry()
    configured = LeakagePolicy(max_steps=1)
    standalone = scan_static_leakage(
        artifact.plan,
        source="SOURCE",
        j=artifact.j,
        policy=configured,
        registry=registry,
    )

    assert not standalone.passed
    assert any(item.kind is LeakageKind.BUDGET for item in standalone.violations)

    result = Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
        leakage_policy=configured,
    ).run(
        _source(), artifact, _case(), registry, CaptureWriter(["GOOD"])
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.leakage_report is not None
    metrics = result.leakage_report.metrics
    assert metrics["configured_step_budget"] == 1
    assert metrics["deterministic_plan_steps"] == len(artifact.plan.steps)
    assert metrics["effective_step_budget"] == len(artifact.plan.steps)
    assert metrics["exact_composer_step_extension"] == 1
    assert any(
        "extended only the mechanical plan step budget" in item
        for item in result.diagnostics
    )


def test_plan_mismatch_cannot_receive_exact_composer_step_extension() -> None:
    artifact = _artifact()
    tampered = replace(
        artifact,
        plan=ReplayPlan(
            [
                *artifact.plan.steps,
                artifact.plan.steps[-1],
            ]
        ),
    )
    result = Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
        leakage_policy=LeakagePolicy(max_steps=1),
    ).run(
        _source(),
        tampered,
        _case(),
        default_triton_rule_registry(),
        CaptureWriter(["GOOD"]),
    )

    assert result.status is Gate1Status.LIFT_REJECTED
    assert result.leakage_report is None
    assert any("artifact validation failed" in item for item in result.diagnostics)


def test_gate1_rejects_incomplete_lift_artifact_instead_of_generating_pi() -> None:
    incomplete = replace(
        _artifact(), plan=ReplayPlan([]), compose_provenance=None
    )
    writer = CaptureWriter(["GOOD"])

    result = _runner(run_masked_ablation=False).run(
        _source(),
        incomplete,
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.LIFT_REJECTED
    assert writer.calls == []


def test_gate1_readmission_rejects_mutated_nonportable_layer_b() -> None:
    artifact = _artifact()
    invalid = "lane = grid_axis(0), lane < n"
    artifact.j.boundary.domains["lane_domain"] = invalid
    writer = CaptureWriter(["GOOD"])

    result = _runner(run_masked_ablation=False).run(
        _source(),
        artifact,
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.LIFT_REJECTED
    assert writer.calls == []
    assert artifact.j.boundary.domains["lane_domain"] == invalid
    assert any(
        "Boundary.domains.lane_domain must be a portable boolean" in item
        for item in result.diagnostics
    )


def _binary_artifact() -> Gate1Artifact:
    artifact = _artifact()
    intent = IntentFunction.from_json(
        {
            "name": "binary_semantics",
            "inputs": [{"name": "x", "dtype": "f32", "shape": ["n"]}],
            "outputs": [{"name": "out", "dtype": "f32", "shape": ["n"]}],
            "body": [
                {
                    "op": "map_binary",
                    "result": "out",
                    "operands": ["x", "x"],
                    "attrs": {"fn": "add"},
                }
            ],
        }
    )
    changed = replace(
        artifact,
        j=replace(artifact.j, intent=intent),
        plan=ReplayPlan(
            [
                replace(step, rule="realize_elementwise")
                if step.rule == "realize_cast"
                else step
                for step in artifact.plan.steps
            ]
        ),
    )
    return materialize_replay_plan(
        changed, default_triton_rule_registry()
    )[0]


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


def test_paired_intervention_is_the_deterministic_anti_leakage_basis() -> None:
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"])
    result = _runner(
        stable_closures=1, run_semantic_interventions=True
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.anti_leakage_basis == (
        "compose_equality+paired_intervention"
    )
    assert len(result.interventions) == 1
    assert result.interventions[0].status is InterventionStatus.PASS
    assert [item.j.intent.body[0].attrs["fn"] for item in writer.artifacts[:2]] == [
        "add",
        "sub",
    ]
    assert writer.artifacts[0].plan.to_json() == writer.artifacts[1].plan.to_json()
    assert writer.artifacts[0].schedule.to_json() == (
        writer.artifacts[1].schedule.to_json()
    )
    assert writer.calls == [False, False]
    assert result.interventions[0].generation is not None
    assert result.interventions[0].generation.generation_id == "normal:1"


def test_paired_intervention_rejects_a_writer_that_ignores_changed_j() -> None:
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_ADD"])
    result = _runner(
        stable_closures=1, run_semantic_interventions=True
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert result.anti_leakage_basis == "paired_intervention"
    assert result.interventions[0].status is InterventionStatus.IGNORED_J
    assert writer.calls == [False, False]


def test_paired_intervention_uses_one_mutant_replay_for_full_fixed_c_family() -> None:
    rng = random.Random(20260718)
    extents = tuple(
        dict.fromkeys(
            (65, 1, 63, 64, 66, 127, 128, 129)
            + tuple(rng.randint(2, 2048) for _ in range(4))
        )
    )
    compiler = FamilySelectiveCompiler()
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"])
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=True,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family(extents),
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert writer.calls == [False, False]
    assert len(result.interventions) == 1
    intervention = result.interventions[0]
    assert intervention.family_policy == PAIRED_INTERVENTION_FAMILY_POLICY
    assert intervention.passed
    assert [item.profile_id for item in intervention.profile_records] == [
        f"n{n}" for n in extents
    ]
    assert all(item.passed for item in intervention.profile_records)
    assert [
        n for code, n in compiler.profile_calls if "REPLAY_SUB" in code
    ] == list(extents)
    assert result.anti_leakage_basis == (
        "compose_equality+paired_intervention_fixed_c_family"
    )


def test_secondary_intervention_delta_failure_rejects_family_after_full_sweep() -> None:
    compiler = FamilySelectiveCompiler(ignored_intervention_extent=1)
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"])
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=True,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family((65, 1, 129)),
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert writer.calls == [False, False]
    intervention = result.interventions[0]
    assert intervention.status is InterventionStatus.IGNORED_J
    assert [item.status for item in intervention.profile_records] == [
        InterventionStatus.PASS,
        InterventionStatus.IGNORED_J,
        InterventionStatus.PASS,
    ]
    assert [
        n for code, n in compiler.profile_calls if "REPLAY_SUB" in code
    ] == [65, 1, 129]
    assert result.anti_leakage_basis == (
        "paired_intervention_fixed_c_family"
    )


def test_family_intervention_semantic_failure_precedes_incomplete_evidence() -> None:
    compiler = FamilySelectiveCompiler(
        ignored_intervention_extent=1,
        intervention_compile_failure_extent=129,
    )
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=True,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"]),
        specialization_family=_fixed_c_family((65, 1, 129)),
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    intervention = result.interventions[0]
    assert intervention.status is InterventionStatus.IGNORED_J
    assert intervention.family_outcome == (
        "semantic_failure_and_incomplete"
    )
    assert intervention.verdict is not None
    assert intervention.verdict.status is InterventionStatus.IGNORED_J
    assert intervention.family_evidence_complete is False
    assert intervention.family_digest is not None
    assert [item.status for item in intervention.profile_records] == [
        InterventionStatus.PASS,
        InterventionStatus.IGNORED_J,
        InterventionStatus.COMPILE_FAILED,
    ]


def test_intervention_family_compile_failure_is_incomplete_not_leakage() -> None:
    compiler = FamilySelectiveCompiler(
        intervention_compile_failure_extent=1
    )
    writer = CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"])
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=True,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family((65, 1, 129)),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "intervention_evidence_incomplete"
    assert writer.calls == [False, False]
    assert result.interventions[0].family_evidence_complete is False
    assert [item.status for item in result.interventions[0].profile_records] == [
        InterventionStatus.PASS,
        InterventionStatus.COMPILE_FAILED,
        InterventionStatus.PASS,
    ]
    assert [
        n for code, n in compiler.profile_calls if "REPLAY_SUB" in code
    ] == [65, 1, 129]
    assert result.anti_leakage_basis == (
        "paired_intervention_fixed_c_family_incomplete"
    )


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


def test_failed_replay_relifts_then_restarts_stability_and_ablation() -> None:
    class RecordingRelifter:
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.requests.append(request)
            return artifact

    writer = CaptureWriter(
        ["BAD", "GOOD", "GOOD"],
        masked=OPAQUE_MASKED_SEMANTIC_DRIFT,
    )
    relifter = RecordingRelifter()
    runner = _runner(
        stable_closures=2,
        max_relifts=2,
        run_masked_ablation=True,
    )
    result = runner.run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert len(result.candidates) == 3
    assert [item.comparison is not None and item.comparison.hard_pass for item in result.candidates] == [False, True, True]
    assert relifter.requests[0].trigger is ReliftTrigger.NORMAL_NO_CLOSURE
    assert [item.relift_round for item in result.candidates] == [0, 1, 1]
    assert writer.calls[:3] == [False, False, False]
    assert writer.calls[3] is True
    assert len(writer.calls) == 4
    assert runner.compiler.calls[:5] == [
        "SOURCE",
        "BAD",
        "SOURCE",
        "GOOD",
        "GOOD",
    ]
    assert result.masked_generation is not None
    assert result.masked_generation.masked
    assert result.masked_compile is not None and result.masked_compile.ok
    assert result.masked_comparison is not None
    assert not result.masked_comparison.hard_pass
    assert result.masked_outcome == "hard_mismatch"
    assert any("masked-J/full-pi attack" in item for item in result.diagnostics)


def test_masked_j_with_actual_lift_pi_hard_closure_rejects_gate1() -> None:
    class ForbiddenLeakageRelifter:
        def relift(self, *args, **kwargs):
            raise AssertionError(
                "masked-J leakage is not a normal source/replay relift trigger"
            )

    compiler = FakeCompiler()
    writer = CaptureWriter(["GOOD"], masked=REAL_MASKED_FAILURE_CODE)
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            max_relifts=2,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=ForbiddenLeakageRelifter(),
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert result.anti_leakage_basis == "masked_j_full_pi_hard_closure"
    assert result.masked_generation is not None
    assert result.masked_generation.masked
    assert result.masked_compile is not None and result.masked_compile.ok
    assert result.masked_comparison is not None
    assert result.masked_comparison.hard_pass
    assert result.masked_outcome == "hard_closure"
    assert result.to_json()["masked_outcome"] == "hard_closure"
    assert writer.calls == [False, True]
    assert writer.artifacts[0].plan.to_json() == writer.artifacts[1].plan.to_json()
    assert any(
        "masked-J/full-pi attack hard-closed" in item
        for item in result.diagnostics
    )


def test_masked_j_actual_pi_hard_mismatch_allows_normal_closure() -> None:
    writer = CaptureWriter(
        ["GOOD"], masked=OPAQUE_MASKED_SEMANTIC_DRIFT
    )
    result = Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.anti_leakage_basis == (
        "compose_equality+masked_j_full_pi_nonclosure"
    )
    assert result.masked_generation is not None
    assert result.masked_compile is not None and result.masked_compile.ok
    assert result.masked_comparison is not None
    assert not result.masked_comparison.hard_pass
    assert result.masked_outcome == "hard_mismatch"
    assert "leakage_ratio" not in result.to_json()
    assert all("score ratio=" not in item for item in result.diagnostics)
    assert any(
        "masked-J/full-pi attack compiled but did not hard-close" in item
        for item in result.diagnostics
    )


def test_masked_j_valid_candidate_compile_failure_is_hard_nonclosure() -> None:
    writer = CaptureWriter(
        ["GOOD"],
        masked=OPAQUE_MASKED_COMPILE_FAIL,
    )
    result = Gate1Runner(
        compiler=FakeCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.anti_leakage_basis == (
        "compose_equality+masked_j_full_pi_nonclosure"
    )
    assert result.masked_generation is not None
    assert result.masked_compile is not None
    assert not result.masked_compile.ok
    assert result.masked_comparison is None
    assert result.masked_outcome == "compile_nonclosure"


def test_masked_j_one_generation_hard_mismatches_every_fixed_c_profile() -> None:
    extents = (65, 1, 63, 64, 66, 127, 128, 129)
    compiler = FamilySelectiveCompiler()
    writer = CaptureWriter(
        ["GOOD"], masked=OPAQUE_MASKED_SEMANTIC_DRIFT
    )
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family(extents),
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert writer.calls == [False, True]
    assert result.masked_outcome == "family_hard_mismatch"
    evidence = result.masked_family_evidence
    assert evidence is not None
    assert evidence["policy"] == MASKED_J_FAMILY_POLICY
    assert isinstance(evidence["actual_pi_digest"], str)
    assert len(evidence["actual_pi_digest"]) == 64
    assert evidence["family_id"] == "anti-leakage-fixed-c-family"
    assert evidence["family_contract_digest"] == (
        specialization_family_contract_digest(_fixed_c_family(extents))
    )
    assert isinstance(evidence["masked_code_sha256"], str)
    assert isinstance(evidence["restored_code_sha256"], str)
    assert evidence["all_profiles_hard_mismatch"] is True
    assert evidence["hard_closed_profile_ids"] == []
    assert evidence["incomplete_profile_ids"] == []
    assert [item["profile_id"] for item in evidence["records"]] == [
        f"n{n}" for n in sorted(extents, key=lambda value: f"n{value}")
    ]
    assert all(
        isinstance(item["source_fingerprint_digest"], str)
        and isinstance(item["replay_fingerprint_digest"], str)
        for item in evidence["records"]
    )
    assert [
        n
        for code, n in compiler.profile_calls
        if "BAD = value + 1.0" in code
    ] == list(extents)
    assert result.anti_leakage_basis == (
        "compose_equality+masked_j_full_pi_fixed_c_family_hard_mismatch"
    )


def test_masked_j_secondary_hard_closure_rejects_after_complete_family_sweep() -> None:
    compiler = FamilySelectiveCompiler(masked_hard_closure_extent=1)
    writer = CaptureWriter(
        ["GOOD"], masked=OPAQUE_MASKED_SEMANTIC_DRIFT
    )
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family((65, 1, 129)),
    )

    assert result.status is Gate1Status.LEAKAGE_REJECTED
    assert writer.calls == [False, True]
    assert result.masked_outcome == "family_hard_closure"
    evidence = result.masked_family_evidence
    assert evidence is not None
    assert evidence["hard_closed_profile_ids"] == ["n1"]
    assert evidence["incomplete_profile_ids"] == []
    assert [
        n
        for code, n in compiler.profile_calls
        if "BAD = value + 1.0" in code
    ] == [65, 1, 129]
    assert result.anti_leakage_basis == (
        "masked_j_full_pi_fixed_c_family_hard_closure"
    )


def test_masked_j_family_compile_failure_is_incomplete_not_nonclosure_pass() -> None:
    compiler = FamilySelectiveCompiler(masked_compile_failure_extent=1)
    writer = CaptureWriter(
        ["GOOD"], masked=OPAQUE_MASKED_SEMANTIC_DRIFT
    )
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family((65, 1, 129)),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "masked_evidence_incomplete"
    assert writer.calls == [False, True]
    assert result.masked_outcome == "family_incomplete"
    evidence = result.masked_family_evidence
    assert evidence is not None
    assert evidence["all_profiles_hard_mismatch"] is False
    assert evidence["hard_closed_profile_ids"] == []
    assert evidence["incomplete_profile_ids"] == ["n1"]
    assert [
        n
        for code, n in compiler.profile_calls
        if "BAD = value + 1.0" in code
    ] == [65, 1, 129]
    assert result.anti_leakage_basis == (
        "masked_j_full_pi_fixed_c_family_incomplete"
    )


def test_masked_j_family_invalid_abi_reports_every_profile_incomplete() -> None:
    invalid = REAL_MASKED_FAILURE_CODE.replace(
        "scrc_p2, scrc_p3: tl.constexpr",
        "scrc_p3: tl.constexpr, scrc_p2",
    )
    writer = CaptureWriter(["GOOD"], masked=invalid)
    result = Gate1Runner(
        compiler=FamilySelectiveCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=False,
        ),
    ).run(
        _source(),
        _artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        writer,
        specialization_family=_fixed_c_family((65, 1, 129)),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "masked_evidence_incomplete"
    evidence = result.masked_family_evidence
    assert evidence is not None
    assert evidence["evidence_complete"] is False
    assert evidence["incomplete_profile_ids"] == ["n1", "n129", "n65"]
    assert {item["outcome"] for item in evidence["records"]} == {
        "invalid_replay"
    }


def test_family_anti_leakage_evidence_is_canonical_across_profile_order() -> None:
    registry = default_triton_rule_registry()

    def run(extents: tuple[int, ...]):
        writer = CaptureWriter(
            ["REPLAY_ADD", "REPLAY_SUB"],
            masked=OPAQUE_MASKED_SEMANTIC_DRIFT,
        )
        result = Gate1Runner(
            compiler=FamilySelectiveCompiler(),
            config=Gate1Config(
                stable_closures=1,
                run_semantic_interventions=True,
                run_masked_ablation=True,
            ),
        ).run(
            KernelSource.from_code(
                "SOURCE_ADD", "kernel", trusted=True
            ),
            _binary_artifact(),
            _profile_case(65),
            registry,
            writer,
            specialization_family=_fixed_c_family(extents),
        )
        return result, writer

    forward, forward_writer = run((65, 1, 129))
    reversed_result, reversed_writer = run((129, 1, 65))

    assert forward.status is reversed_result.status is Gate1Status.REPLAY_CLOSED
    assert forward.interventions[0].family_digest is not None
    assert forward.interventions[0].family_digest == (
        reversed_result.interventions[0].family_digest
    )
    assert forward.masked_family_evidence["family_digest"] == (
        reversed_result.masked_family_evidence["family_digest"]
    )
    assert forward.interventions[0].to_json()["profile_records"] == (
        reversed_result.interventions[0].to_json()["profile_records"]
    )
    assert forward.masked_family_evidence["records"] == (
        reversed_result.masked_family_evidence["records"]
    )
    assert forward_writer.calls == reversed_writer.calls == [
        False,
        False,
        True,
    ]
    assert forward_writer.artifacts[0].plan.to_json() == (
        forward_writer.artifacts[-1].plan.to_json()
    )


def test_family_anti_leakage_digests_bind_the_complete_family_contract() -> None:
    registry = default_triton_rule_registry()

    def run(extents: tuple[int, ...]):
        return Gate1Runner(
            compiler=FamilySelectiveCompiler(),
            config=Gate1Config(
                stable_closures=1,
                run_semantic_interventions=True,
                run_masked_ablation=True,
            ),
        ).run(
            KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
            _binary_artifact(),
            _profile_case(65),
            registry,
            CaptureWriter(
                ["REPLAY_ADD", "REPLAY_SUB"],
                masked=OPAQUE_MASKED_SEMANTIC_DRIFT,
            ),
            specialization_family=_fixed_c_family(extents),
        )

    left = run((65, 1, 129))
    right = run((65, 1, 128))

    assert left.status is right.status is Gate1Status.REPLAY_CLOSED
    assert left.interventions[0].family_digest != (
        right.interventions[0].family_digest
    )
    assert left.masked_family_evidence["family_digest"] != (
        right.masked_family_evidence["family_digest"]
    )


def test_family_intervention_digest_excludes_score_and_diagnostic_prose() -> None:
    result = Gate1Runner(
        compiler=FamilySelectiveCompiler(),
        config=Gate1Config(
            stable_closures=1,
            run_semantic_interventions=True,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE_ADD", "kernel", trusted=True),
        _binary_artifact(),
        _profile_case(65),
        default_triton_rule_registry(),
        CaptureWriter(["REPLAY_ADD", "REPLAY_SUB"]),
        specialization_family=_fixed_c_family((65, 1, 129)),
    )
    original = result.interventions[0]
    rewritten_profiles = []
    for profile in original.profile_records:
        assert profile.verdict is not None
        comparison = replace(
            profile.verdict.comparison,
            score=profile.verdict.comparison.score / 2,
        )
        rewritten_profiles.append(
            replace(
                profile,
                diagnostic="arbitrary human prose",
                verdict=replace(
                    profile.verdict,
                    comparison=comparison,
                    diagnostic="different human prose",
                ),
            )
        )
    rewritten = replace(
        original,
        diagnostic="different aggregate prose",
        profile_records=tuple(rewritten_profiles),
    )

    assert original.family_digest == rewritten.family_digest


def test_candidate_binds_the_current_canonical_artifact_digest() -> None:
    registry = default_triton_rule_registry()
    artifact = _artifact()
    result = _runner(stable_closures=1).run(
        _source(), artifact, _case(), registry, CaptureWriter(["GOOD"])
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.candidates[0].artifact_digest == (
        canonical_artifact_digest(artifact, registry)
    )
    assert result.candidates[0].to_json()["artifact_digest"] == (
        result.candidates[0].artifact_digest
    )


def test_structural_relift_guidance_surfaces_a_loop_carry_gap() -> None:
    source = fingerprint_ttir(LOOP_TTIR)
    replay = fingerprint_ttir(GOOD_TTIR)

    guidance = _build_structural_relift_guidance(source, replay)

    assert guidance["policy"] == RELIFT_STRUCTURAL_GUIDANCE_POLICY
    assert "loop_structure" in guidance["mismatch_coordinates"]
    assert guidance["source"]["loop_count"] == 1
    assert guidance["replay"]["loop_count"] == 0
    loop = guidance["source"]["loops"][0]
    assert len(loop["init_values"]) == 1
    assert len(loop["yielded_values"]) == 1
    assert loop["carry_pairs"] == [
        {
            "init": loop["init_values"][0],
            "yield": loop["yielded_values"][0],
        }
    ]
    assert loop["block_argument_count"] == 2
    assert loop["carried_argument_use_counts"] == [1]
    assert loop["body_op_multiset"] == {"arith.addf": 1}
    assert len(loop["captured_values"]) == 1


def test_compile_failure_has_no_replay_side_retry() -> None:
    writer = CaptureWriter(["COMPILE_FAIL"])
    result = _runner(stable_closures=1, run_masked_ablation=False).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "relifter_unavailable"
    assert result.termination is not None
    assert result.termination.hard_difference_categories == ()
    assert len(result.candidates) == 1
    assert len(writer.calls) == 1
    assert result.best_candidate is not None
    assert result.best_candidate.diagnosis is not None
    assert result.best_candidate.diagnosis.mismatches == ("bad replay",)


def test_compile_diagnostic_guides_lift_without_replay_source() -> None:
    summary = sanitize_replay_diagnostic(
        '''Traceback (most recent call last):
  File "/secret/worker.py", line 7
    output_value = tl.load(error_ptr)
                   ^
NameError: output_value is not defined'''
    )

    assert summary == "NameError: output_value is not defined"
    assert "/secret/worker.py" not in summary
    assert "tl.load" not in summary


def test_last_artifact_round_reports_relift_budget_exhaustion() -> None:
    result = _runner(
        max_relifts=1,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        CaptureWriter(["BAD"]),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "relift_budget_exhausted"
    assert result.termination is not None
    assert result.termination.artifact_round == 0
    assert result.termination.remaining_artifact_round_budget == 0
    assert result.termination.hard_difference_categories == (
        "op_multiset",
        "dataflow",
    )


def test_semantic_relift_failure_is_audited_as_no_closure() -> None:
    class FailingRelifter:
        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            assert source_fingerprint is not None
            assert history
            assert request.trigger is ReliftTrigger.NORMAL_NO_CLOSURE
            assert request.hard_difference_categories
            raise RuntimeError("relift provider unavailable")

    writer = CaptureWriter(["BAD"])
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=FailingRelifter(),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "relift_failed"
    assert result.termination is not None
    assert result.termination.normal_candidate_count == 1
    assert result.termination.remaining_artifact_round_budget == 1
    assert result.termination.hard_difference_categories == (
        "op_multiset",
        "dataflow",
    )
    assert len(result.candidates) == 1
    assert any(
        "semantic relift failed: RuntimeError: relift provider unavailable"
        in diagnostic
        for diagnostic in result.diagnostics
    )


def test_replay_policy_failure_triggers_fresh_source_aware_relift() -> None:
    class PolicyRejectingWriter:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            assert "base_candidate" not in kwargs
            if self.calls == 1:
                raise UnsafeReplayCodeError("libdevice call is not allowlisted")
            code = GOOD_CODE
            digest = hashlib.sha256(code.encode()).hexdigest()
            return ReplayGeneration(
                code=code,
                prompt_sha256="prompt",
                requested_model="capture",
                generation_id=f"normal:{self.calls}",
                code_sha256=digest,
            )

    class RecordingRelifter:
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.requests.append(request)
            assert not hasattr(history[-1], "generation")
            assert not history[-1].generation_available
            return artifact

    writer = PolicyRejectingWriter()
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert len(result.candidates) == 2
    assert result.candidates[0].comparison is None
    assert relifter.requests[0].trigger is ReliftTrigger.REPLAY_GENERATION_FAILED
    assert writer.calls == 2


def test_outer_relift_round_replays_the_revised_artifact() -> None:
    class RecordingRelifter:
        calls = 0
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.calls += 1
            assert source_fingerprint is not None
            assert [item.relift_round for item in history] == [0]
            assert history[0].comparison is not None
            assert history[0].comparison["differences"]
            assert not hasattr(history[0], "generation")
            self.requests.append(request)
            return artifact

    writer = CaptureWriter(["BAD", "GOOD"])
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert relifter.calls == 1
    assert len(relifter.requests) == 1
    request = relifter.requests[0]
    assert request.trigger is ReliftTrigger.NORMAL_NO_CLOSURE
    assert request.evaluation_round == 0
    assert request.candidate_count == 1
    assert request.compiled_candidate_count == 1
    assert request.best_generation_round == 0
    assert request.hard_difference_categories == (
        "op_multiset",
        "dataflow",
    )
    assert request.semantic_guidance is not None
    guidance = request.semantic_guidance
    assert guidance.policy == "deterministic_gate1_hard_diff_guidance_v2"
    assert guidance.failed_hard_categories == (
        "op_multiset",
        "dataflow",
    )
    assert "interface" in guidance.confirmed_hard_categories
    assert guidance.hard_difference_digest
    assert guidance.current_j_summary["intent_op_multiset"] == {"cast": 1}
    assert any(
        item["category"] == "dataflow"
        and "j.intent.body[*].operands/results"
        in item["inspect_j_surfaces"]
        for item in guidance.category_guidance
    )
    assert (
        "return_unchanged_editable_fields_when_no_semantic_change_is_justified"
        in guidance.decision_contract
    )
    assert [item.relift_round for item in result.candidates] == [0, 1]
    assert result.stagnation is None


def test_identical_artifact_and_hard_diff_stop_outer_loop_deterministically() -> None:
    class RecordingRelifter:
        calls = 0

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.calls += 1
            assert source_fingerprint is not None
            assert history[-1].comparison is not None
            assert not hasattr(history[-1], "generation")
            return artifact

    writer = CaptureWriter(["BAD", "BAD", "GOOD"])
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=4,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "deterministic_stagnation"
    assert result.termination is not None
    assert result.termination.normal_candidate_count == 2
    assert result.termination.remaining_artifact_round_budget == 2
    assert relifter.calls == 1
    assert writer.calls == [False, False]
    assert [item.relift_round for item in result.candidates] == [0, 1]
    assert result.stagnation is not None
    assert result.stagnation.first_relift_round == 0
    assert result.stagnation.repeated_relift_round == 1
    assert result.stagnation.hard_difference_categories == (
        "op_multiset",
        "dataflow",
    )
    serialized = result.to_json()["stagnation"]
    assert serialized["policy"] == "exact_artifact_and_hard_diff_v1"
    assert serialized["consecutive_observations"] == 2
    assert "code" not in serialized


def test_artifact_change_prevents_false_stagnation_on_the_same_hard_diff() -> None:
    class AlternatingRelifter:
        calls = 0

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.calls += 1
            return _binary_artifact() if self.calls % 2 else _artifact()

    writer = CaptureWriter(["BAD", "BAD", "BAD"])
    relifter = AlternatingRelifter()
    result = _runner(
        max_relifts=3,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "relift_budget_exhausted"
    assert result.stagnation is None
    assert relifter.calls == 2
    assert writer.calls == [False, False, False]
    artifact_digests = [item.artifact_digest for item in result.candidates]
    assert artifact_digests[0] != artifact_digests[1]
    assert artifact_digests[0] == artifact_digests[2]
    hard_diff_digests = {
        item.effective_hard_difference_digest for item in result.candidates
    }
    assert len(hard_diff_digests) == 1


def test_changed_hard_diff_resets_stagnation_chain() -> None:
    class RecordingRelifter:
        calls = 0

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.calls += 1
            return artifact

    writer = CaptureWriter(["BAD", "REPLAY_SUB", "REPLAY_SUB", "GOOD"])
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=4,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert relifter.calls == 2
    assert writer.calls == [False, False, False]
    assert result.stagnation is not None
    assert result.stagnation.first_relift_round == 1
    assert result.stagnation.repeated_relift_round == 2


def test_compile_failure_breaks_hard_diff_stagnation_chain() -> None:
    class RecordingRelifter:
        calls = 0

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            self.calls += 1
            return artifact

    writer = CaptureWriter(
        ["BAD", "COMPILE_FAIL", "BAD", "BAD", "GOOD"]
    )
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=5,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert relifter.calls == 3
    assert writer.calls == [False, False, False, False]
    assert result.stagnation is not None
    assert result.stagnation.first_relift_round == 2
    assert result.stagnation.repeated_relift_round == 3


def test_compile_failure_relifts_and_never_reuses_failed_replay_as_base() -> None:
    class RecordingRelifter:
        requests = []

        def relift(
            self, artifact, *, source_fingerprint, history, request
        ):
            assert not hasattr(history[-1], "generation")
            assert history[-1].generation_available
            assert history[-1].diagnosis is not None
            self.requests.append(request)
            return artifact

    writer = CaptureWriter(["COMPILE_FAIL", "GOOD"])
    relifter = RecordingRelifter()
    result = _runner(
        max_relifts=2,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert relifter.requests[0].trigger is ReliftTrigger.REPLAY_COMPILE_FAILED
    assert relifter.requests[0].semantic_guidance is None
    assert writer.calls == [False, False]


def test_provider_transport_failure_does_not_trigger_semantic_relift() -> None:
    class CoolingWriter:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            raise ProviderError("HTTP 429: model_cooldown")

    class ForbiddenRelifter:
        def relift(self, *args, **kwargs):
            raise AssertionError(
                "provider transport failure has no semantic Relift evidence"
            )

    writer = CoolingWriter()
    result = _runner(
        max_relifts=3,
        stable_closures=1,
        run_masked_ablation=False,
    ).run(
        _source(),
        _artifact(),
        _case(),
        default_triton_rule_registry(),
        writer,
        relifter=ForbiddenRelifter(),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "replay_provider_unavailable"
    assert result.termination is not None
    assert result.termination.remaining_artifact_round_budget == 2
    assert writer.calls == 1
    assert len(result.candidates) == 1
    diagnosis = result.candidates[0].diagnosis
    assert diagnosis is not None
    assert diagnosis.responsibility.value == "unknown"
    assert diagnosis.allowed_modifications == ()
    assert any(
        "no Gate1 hard difference exists" in item
        for item in result.diagnostics
    )


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


def test_missing_masked_generation_makes_anti_leakage_unevaluable() -> None:
    writer = StaticReplayWriter(GOOD_CODE)
    result = _runner(run_masked_ablation=True).run(
        _source(), _artifact(), _case(), default_triton_rule_registry(), writer
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "masked_evidence_incomplete"
    assert result.anti_leakage_basis == "masked_j_full_pi_unevaluated"
    assert result.masked_outcome == "generation_unavailable"
    assert result.masked_generation is None
    assert result.masked_compile is None
    assert any(
        "anti-leakage evidence is incomplete" in item
        for item in result.diagnostics
    )


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
        "bind_wide_program_domain",
        "materialize_index_tile",
        "materialize_pointer_tile",
        "realize_masked_access",
        "materialize_mask_tile",
        "realize_elementwise",
        "realize_reduce",
        "realize_flattened_reduce",
        "realize_multi_tile_reduce",
        "stage_reduction_preheader",
        "fuse_reduction_tile_outputs",
        "realize_k_tiled_dot",
        "realize_outer_product",
    } <= {card.id for card in first.cards}
    cards = {card.id: card for card in first.cards}
    assert {
        card.id for card in first.cards if card.selection_mode.value == "override"
    } == {
        "bind_wide_program_domain",
        "materialize_pointer_tile",
        "materialize_mask_tile",
        "realize_flattened_reduce",
        "realize_multi_tile_reduce",
        "stage_reduction_preheader",
        "fuse_reduction_tile_outputs",
    }
    assert all(
        card.override_family is not None
        for card in first.cards
        if card.selection_mode.value == "override"
    )
    assert cards["realize_k_tiled_dot"].selection_mode.value == "derived"
    assert cards["materialize_pointer_tile"].version == "4"
    assert any(
        "one shared staged base per buffer" in item
        for item in cards["materialize_pointer_tile"].pitfalls
    )
    assert "maximal common row/base term" in cards[
        "materialize_pointer_tile"
    ].example
    assert cards["realize_masked_access"].version == "3"
    assert cards["realize_masked_access"].api_mapping["vector_conjunction"] == "&"
    assert cards["realize_broadcast"].version == "5"
    assert cards["realize_elementwise"].version == "5"
    assert cards["realize_elementwise"].api_mapping["recip"] == "1.0 / operand"
    assert cards["realize_elementwise"].api_mapping["abs"] == "tl.abs"
    assert "libdevice.pow" in cards["realize_elementwise"].api_mapping["pow"]
    assert "libdevice.tanh" in cards["realize_elementwise"].api_mapping["tanh"]
    assert cards["realize_synchronize"].api_mapping["program_barrier"] == (
        "tl.debug_barrier"
    )
    assert cards["realize_reduce"].version == "4"
    assert "tile_axis" in cards["realize_reduce"].reads
    assert "semantic_extent" in cards["realize_reduce"].api_mapping["mean"]
    assert "tl.reduce" in cards["realize_reduce"].api_mapping["prod"]
    assert "multiply combiner" in cards["realize_reduce"].api_mapping["prod"]
    assert cards["bind_wide_program_domain"].api_mapping[
        "program_coordinate_i64"
    ].endswith("to(tl.int64)")
    assert cards["realize_flattened_reduce"].api_mapping["sum"] == "tl.sum(tile)"
    assert cards["realize_multi_tile_reduce"].override_family == (
        "reduction_tiling"
    )
    assert "C.strategy.tile_slot" in cards[
        "realize_multi_tile_reduce"
    ].api_mapping["loop"]
    assert any(
        "runtime loop" in item
        for item in cards["realize_multi_tile_reduce"].pitfalls
    )
    assert cards["stage_reduction_preheader"].override_family == (
        "reduction_preheader_staging"
    )
    assert "raw_pointer + outer_index * stride" in cards[
        "stage_reduction_preheader"
    ].api_mapping["loop_capture"]
    assert any(
        "not general LICM" in item
        for item in cards["stage_reduction_preheader"].pitfalls
    )
    assert cards["fuse_reduction_tile_outputs"].override_family == (
        "reduction_tile_fusion"
    )
    assert "shared per-tile producer closure once" in cards[
        "fuse_reduction_tile_outputs"
    ].api_mapping["single_tile_body"]
    assert any(
        "separate loop" in item
        for item in cards["fuse_reduction_tile_outputs"].pitfalls
    )
    assert cards["bind_runtime_scalar"].version == "2"
    assert "tensor_ref" in cards["bind_runtime_scalar"].reads
    assert cards["declare_schedule_slot"].version == "2"
    assert any(
        "compiler slot is not a kernel parameter" in item
        for item in cards["declare_schedule_slot"].pitfalls
    )
    assert "tl.sum" in first.api_tokens(Frontend.TRITON)
    assert "+" not in first.api_tokens(Frontend.TRITON)
    assert "*" not in first.api_tokens(Frontend.TRITON)
    assert cards["realize_k_tiled_dot"].version == "3"
    assert "result dtype" in cards["realize_k_tiled_dot"].reads
    assert cards["realize_k_tiled_dot"].api_mapping["float32"] == "tl.float32"
    assert cards["realize_outer_product"].api_mapping["combine"] == "*"
    assert "subscripts" in cards["realize_outer_product"].reads
    assert cards["materialize_mask_tile"].api_mapping[
        "vector_conjunction"
    ] == "&"
    assert any(
        "Do not recompute component comparisons" in item
        for item in cards["materialize_mask_tile"].pitfalls
    )
    assert cards["realize_broadcast"].api_mapping["scalar_to_tile"] == (
        "use a rank-0 scalar operand directly; scalar broadcasting never "
        "requires synthetic tl.expand_dims"
    )
    assert "global axis number is never" in cards[
        "realize_broadcast"
    ].api_mapping["global_to_local_axis_map"]
    assert "explicit_expand_dims" in cards[
        "realize_broadcast"
    ].api_mapping["local_realization_modes"]
    assert "rank" in cards["realize_broadcast"].api_mapping[
        "shape_postcondition"
    ]
    assert any(
        "redundant domain facts" in item
        for item in cards["realize_masked_access"].pitfalls
    )


def test_compiled_interface_allows_specialized_nonbinding_scalars_only() -> None:
    artifact = _artifact()
    python_interface = {
        "arg_names": ["x", "out", "n", "BLOCK"],
        "constexprs": ["BLOCK"],
        "constexpr_indices": [3],
    }
    result = CompileResult(
        ok=True,
        ttir=SPECIALIZED_SCALAR_TTIR,
        python_interface=python_interface,
    )
    fingerprint = fingerprint_ttir(
        SPECIALIZED_SCALAR_TTIR,
        python_interface=python_interface,
    )
    _validate_compiled_interface(result, fingerprint, artifact)

    semantic_intent = IntentFunction.from_json(
        {
            "name": "semantic_scalar",
            "inputs": [
                {"name": "x", "dtype": "f32", "shape": ["N"]},
                {"name": "eps", "dtype": "f32", "shape": []},
            ],
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
    semantic = replace(
        artifact,
        j=replace(artifact.j, intent=semantic_intent),
        interface=replace(
            artifact.interface,
            parameters=[
                *artifact.interface.parameters[:2],
                replace(
                    artifact.interface.parameters[2],
                    dtype="f32",
                    tensor_ref="eps",
                ),
                artifact.interface.parameters[3],
            ],
        ),
    )
    with pytest.raises(ValueError, match="omitted binding runtime argument.*n"):
        _validate_compiled_interface(result, fingerprint, semantic)


def test_compiled_interface_accepts_unused_as_static_input_subset() -> None:
    artifact = _artifact()
    intent_payload = artifact.j.intent.to_json()["intent_function"]
    intent_payload["inputs"].append(
        {"name": "label", "dtype": "f32", "shape": [1]}
    )
    intent = IntentFunction.from_json(intent_payload)
    parameters = list(artifact.interface.parameters)
    parameters.insert(
        1,
        KernelParameter(
            "label",
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            "f32",
            1,
            tensor_ref="label",
        ),
    )
    artifact = replace(
        artifact,
        j=replace(artifact.j, intent=intent),
        interface=replace(artifact.interface, parameters=parameters),
    )
    python_interface = {
        "arg_names": ["x", "label", "out", "n", "BLOCK"],
        "constexprs": ["BLOCK"],
        "constexpr_indices": [4],
    }
    result = CompileResult(
        ok=True,
        ttir=UNUSED_POINTER_TTIR,
        python_interface=python_interface,
    )
    fingerprint = fingerprint_ttir(
        UNUSED_POINTER_TTIR,
        python_interface=python_interface,
    )

    _validate_compiled_interface(result, fingerprint, artifact)

    incompatible = CompileResult(
        ok=True,
        ttir=WRITTEN_LABEL_TTIR,
        python_interface=python_interface,
    )
    incompatible_fingerprint = fingerprint_ttir(
        WRITTEN_LABEL_TTIR,
        python_interface=python_interface,
    )
    with pytest.raises(
        ValueError,
        match="specialized access 'output'.*declared mode 'input'",
    ):
        _validate_compiled_interface(
            incompatible, incompatible_fingerprint, artifact
        )
    comparison = compare_ttir(UNUSED_POINTER_TTIR, WRITTEN_LABEL_TTIR)
    assert comparison.hard_passes["interface"] is False


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
        masked=OPAQUE_MASKED_EMPTY,
    )
    runner = Gate1Runner(
        compiler=TritonCompiler(cache_dir=tmp_path, timeout=180.0),
            config=Gate1Config(run_semantic_interventions=False),
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
    assert result.masked_generation is not None
    assert result.masked_generation.masked
    assert result.masked_compile is not None and result.masked_compile.ok
    assert result.masked_comparison is not None
    assert not result.masked_comparison.hard_pass
    assert result.masked_outcome == "hard_mismatch"
