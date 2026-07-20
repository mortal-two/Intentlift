from __future__ import annotations

import hashlib
import random

import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.gate1 import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    Frontend,
    Gate1Artifact,
    Gate1Config,
    Gate1FamilyRunner,
    Gate1Runner,
    Gate1SpecializationFamily,
    Gate1SpecializationProfile,
    Gate1Status,
    PiAblationKind,
    PiAblationStatus,
    KernelInterface,
    KernelParameter,
    LocationMap,
    ParameterKind,
    ParameterMode,
    ReplayGeneration,
    ReplayPlan,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SemanticArchive,
    ReliftTrigger,
    default_triton_rule_registry,
    materialize_replay_plan,
    run_pi_ablations,
)
from intentir.verify.gate1.specialization import (
    GATE1_FAMILY_CONTRACT_DIGEST_SCHEMA,
    GATE1_FAMILY_DIFF_POLICY,
    GATE1_SPECIALIZATION_POLICY,
    build_family_relift_guidance,
    family_hard_difference_digest,
    specialization_family_contract_digest,
    specialization_family_contract_payload,
)
from intentir.verify.gate1.compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
)
from intentir.verify.gate1.ttir import fingerprint_ttir


GOOD_TTIR = """
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %v = tt.load %x : !tt.ptr<f32>
    tt.store %out, %v : !tt.ptr<f32>
    tt.return
  }
}
"""

BAD_TTIR = GOOD_TTIR.replace(
    "%v = tt.load %x : !tt.ptr<f32>",
    "%v0 = tt.load %x : !tt.ptr<f32>\n"
    "    %one = arith.constant 1.0 : f32\n"
    "    %v = arith.addf %v0, %one : f32",
)

LOOP_TTIR = """
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
"""


def _artifact() -> Gate1Artifact:
    artifact = Gate1Artifact(
        j=SemanticArchive(
            "kernel",
            IntentFunction.from_json(
                {
                    "name": "identity",
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
            ),
            {"n0": "body[0]"},
            Boundary(
                domains={"d0": "i < n"},
                accesses=[
                    BoundaryAccess(
                        "a0", AccessKind.LOAD, "x", predicate="i < n",
                        fill_value=0.0, node_ref="n0", index_expression="i",
                    ),
                    BoundaryAccess(
                        "a1", AccessKind.STORE, "out", predicate="i < n",
                        node_ref="n0", index_expression="i",
                    ),
                ],
            ),
        ),
        interface=KernelInterface(
            "kernel",
            Frontend.TRITON,
            [
                KernelParameter("x", ParameterKind.POINTER, ParameterMode.INPUT, "f32", 1, tensor_ref="x"),
                KernelParameter("out", ParameterKind.POINTER, ParameterMode.OUTPUT, "f32", 1, tensor_ref="out"),
                KernelParameter("n", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"),
                KernelParameter("BLOCK", ParameterKind.CONSTEXPR),
            ],
            ["ceil_div(n, BLOCK)"],
            function_name="kernel",
            compiler_slots=["num_warps"],
            do_not_specialize=["n"],
        ),
        schedule=ScheduleSpec(
            [
                ScheduleSlot("BLOCK", ScheduleSlotKind.CONSTEXPR),
                ScheduleSlot("num_warps", ScheduleSlotKind.COMPILER),
            ],
            {"BLOCK": 64, "num_warps": 4},
        ),
        plan=ReplayPlan([]),
        location_map=LocationMap("private.py", []),
    )
    return materialize_replay_plan(
        artifact, default_triton_rule_registry()
    )[0]


def _profile(profile_id: str, n: int) -> Gate1SpecializationProfile:
    return Gate1SpecializationProfile(
        profile_id,
        CompileCase(
            {
                "x": RuntimeArgument.tensor((n,), "float32"),
                "out": RuntimeArgument.tensor((n,), "float32"),
                "n": RuntimeArgument.scalar(n, "int32"),
            },
            ((n + 63) // 64,),
            {"BLOCK": 64, "num_warps": 4},
        ),
    )


class _Compiler:
    def __init__(self, *, drift_profile: int | None = None) -> None:
        self.drift_profile = drift_profile
        self.calls: list[tuple[str, int]] = []

    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        n = int(case.arguments["n"].value)
        self.calls.append((code, n))
        drift = "REPLAY" in code and n == self.drift_profile
        return CompileResult(
            True,
            BAD_TTIR if drift else GOOD_TTIR,
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": "test"},
        )


class _AllDriftCompiler(_Compiler):
    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        n = int(case.arguments["n"].value)
        self.calls.append((code, n))
        return CompileResult(
            True,
            BAD_TTIR if "REPLAY" in code else GOOD_TTIR,
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": "test"},
        )


class _LoopGapCompiler(_Compiler):
    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        n = int(case.arguments["n"].value)
        self.calls.append((code, n))
        return CompileResult(
            True,
            GOOD_TTIR if "REPLAY" in code else LOOP_TTIR,
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": "test"},
        )


class _Writer:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: list[dict[str, object]] = []

    def generate(self, artifact, registry, **kwargs) -> ReplayGeneration:
        self.calls += 1
        self.kwargs.append(dict(kwargs))
        code = "REPLAY"
        return ReplayGeneration(
            code,
            "prompt",
            "test",
            f"generation:{self.calls}",
            hashlib.sha256(code.encode()).hexdigest(),
            ablation_kind=kwargs.get("ablation_kind"),
        )


class _RoundWriter:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, artifact, registry, **kwargs) -> ReplayGeneration:
        self.calls += 1
        code = f"REPLAY:{self.calls}"
        return ReplayGeneration(
            code,
            "prompt",
            "test",
            f"generation:{self.calls}",
            hashlib.sha256(code.encode()).hexdigest(),
        )


class _RoundCompiler(_Compiler):
    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        n = int(case.arguments["n"].value)
        self.calls.append((code, n))
        drift = code == "REPLAY:1" and n == 65
        return CompileResult(
            True,
            BAD_TTIR if drift else GOOD_TTIR,
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": "test"},
        )


class _FamilyCompileFailureCompiler(_Compiler):
    def compile(self, source: KernelSource, case: CompileCase) -> CompileResult:
        code, _ = source.read_code()
        n = int(case.arguments["n"].value)
        self.calls.append((code, n))
        if "REPLAY" in code and n == 65:
            return CompileResult(
                False,
                None,
                diagnostic="CompileError: secondary profile failed",
                python_interface={
                    "arg_names": ["x", "out", "n", "BLOCK"],
                    "constexprs": ["BLOCK"],
                    "constexpr_indices": [3],
                    "do_not_specialize": ["n"],
                    "do_not_specialize_raw": [2],
                },
                compiler={"triton": "test"},
                target={"arch": "test"},
            )
        return CompileResult(
            True,
            GOOD_TTIR,
            python_interface={
                "arg_names": ["x", "out", "n", "BLOCK"],
                "constexprs": ["BLOCK"],
                "constexpr_indices": [3],
                "do_not_specialize": ["n"],
                "do_not_specialize_raw": [2],
            },
            compiler={"triton": "test"},
            target={"arch": "test"},
        )


class _Relifter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def relift(
        self,
        artifact,
        *,
        source_fingerprint,
        history,
        request,
    ):
        self.calls.append(
            {
                "source_fingerprint": source_fingerprint,
                "history": history,
                "request": request,
            }
        )
        return artifact


def _family() -> Gate1SpecializationFamily:
    return Gate1SpecializationFamily(
        "runtime-tail-family", "n1", (_profile("n1", 1), _profile("n65", 65))
    )


def test_fixed_c_family_contract_digest_is_order_canonical_and_case_bound() -> None:
    forward = _family()
    reversed_family = Gate1SpecializationFamily(
        "runtime-tail-family",
        "n1",
        (_profile("n65", 65), _profile("n1", 1)),
    )
    changed_case = Gate1SpecializationFamily(
        "runtime-tail-family",
        "n1",
        (_profile("n1", 1), _profile("n66", 66)),
    )

    payload = specialization_family_contract_payload(forward)
    assert payload["digest_schema"] == GATE1_FAMILY_CONTRACT_DIGEST_SCHEMA
    assert [item["profile_id"] for item in payload["profiles"]] == [
        "n1",
        "n65",
    ]
    assert specialization_family_contract_digest(forward) == (
        specialization_family_contract_digest(reversed_family)
    )
    assert specialization_family_contract_digest(forward) != (
        specialization_family_contract_digest(changed_case)
    )


def test_fixed_c_family_uses_one_replay_and_requires_all_profiles() -> None:
    compiler = _Compiler()
    writer = _Writer()
    result = Gate1FamilyRunner(compiler=compiler).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _family(),
        default_triton_rule_registry(),
        writer,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.all_passed
    assert writer.calls == 1
    assert compiler.calls == [
        ("SOURCE", 1), ("SOURCE", 65), ("REPLAY", 1), ("REPLAY", 65)
    ]


def test_fixed_c_family_reuses_an_exact_gate1_generation_without_writer() -> None:
    compiler = _Compiler()
    generation = ReplayGeneration(
        "REPLAY",
        "shared-prompt",
        "gpt-5.6-sol",
        "responses:shared-generation",
        hashlib.sha256(b"REPLAY").hexdigest(),
        response_model="gpt-5.6-sol",
        response_id="shared-generation",
    )
    result = Gate1FamilyRunner(compiler=compiler).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _family(),
        default_triton_rule_registry(),
        generation=generation,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert result.generation is generation
    assert result.to_json()["generation"]["generation_id"] == (
        "responses:shared-generation"
    )
    assert [record["passed"] for record in result.to_json()["records"]] == [
        True,
        True,
    ]


def test_fixed_c_family_preserves_order_across_random_runtime_extents() -> None:
    rng = random.Random(20260717)
    extents = tuple(dict.fromkeys(rng.randint(1, 4096) for _ in range(16)))
    profiles = tuple(
        _profile(f"random_{index:02d}", extent)
        for index, extent in enumerate(extents)
    )
    family = Gate1SpecializationFamily(
        "random-runtime-family",
        profiles[0].profile_id,
        profiles,
    )
    writer = _Writer()
    result = Gate1FamilyRunner(compiler=_Compiler()).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        family,
        default_triton_rule_registry(),
        writer,
    )

    assert result.all_passed
    assert writer.calls == 1
    assert [record.profile_id for record in result.records] == [
        profile.profile_id for profile in profiles
    ]


def test_fixed_c_family_rejects_ambiguous_generation_sources() -> None:
    generation = ReplayGeneration(
        "REPLAY", "prompt", "test", "generation", "digest"
    )
    with pytest.raises(ValueError, match="exactly one"):
        Gate1FamilyRunner(compiler=_Compiler()).run(
            KernelSource.from_code("SOURCE", "kernel", trusted=True),
            _artifact(),
            _family(),
            default_triton_rule_registry(),
            _Writer(),
            generation=generation,
        )


def test_one_profile_drift_rejects_the_whole_family_without_average_score() -> None:
    result = Gate1FamilyRunner(compiler=_Compiler(drift_profile=65)).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _family(),
        default_triton_rule_registry(),
        _Writer(),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert [item.passed for item in result.records] == [True, False]


def test_family_failure_builds_deterministic_member_scoped_relift_guidance() -> None:
    result = Gate1FamilyRunner(compiler=_Compiler(drift_profile=65)).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _family(),
        default_triton_rule_registry(),
        _Writer(),
    )

    evidence = result.relift_evidence_json()
    guidance = build_family_relift_guidance(_artifact(), result)

    assert evidence["policy"] == GATE1_SPECIALIZATION_POLICY
    assert evidence["diff_policy"] == GATE1_FAMILY_DIFF_POLICY
    assert evidence["hard_comparison_complete"]
    assert evidence["failed_profile_ids"] == ["n65"]
    assert evidence["family_hard_difference_digest"] == (
        family_hard_difference_digest(result)
    )
    assert guidance is not None
    assert guidance.policy == (
        "deterministic_fixed_c_family_hard_diff_guidance_v2"
    )
    assert guidance.hard_difference_digest == (
        evidence["family_hard_difference_digest"]
    )
    assert guidance.current_j_summary["fixed_c_family"] == {
        "family_id": "runtime-tail-family",
        "primary_profile_id": "n1",
        "profiles": ["n1", "n65"],
        "hard_closed_profiles": ["n1"],
        "failed_profiles": ["n65"],
    }
    affected = {
        entry["category"]: entry["affected_profiles"]
        for entry in guidance.category_guidance
    }
    assert affected
    assert set().union(*map(set, affected.values())) == {"n65"}
    assert "preserve_hard_closed_fixed_c_profiles" in (
        guidance.decision_contract
    )


def test_family_diff_digest_and_guidance_are_profile_order_canonical() -> None:
    artifact = _artifact()
    source = KernelSource.from_code("SOURCE", "kernel", trusted=True)
    registry = default_triton_rule_registry()
    forward = Gate1FamilyRunner(compiler=_Compiler(drift_profile=65)).run(
        source,
        artifact,
        _family(),
        registry,
        _Writer(),
    )
    reversed_family = Gate1SpecializationFamily(
        "runtime-tail-family",
        "n1",
        (_profile("n65", 65), _profile("n1", 1)),
    )
    reversed_result = Gate1FamilyRunner(
        compiler=_Compiler(drift_profile=65)
    ).run(
        source,
        artifact,
        reversed_family,
        registry,
        _Writer(),
    )

    assert family_hard_difference_digest(forward) == (
        family_hard_difference_digest(reversed_result)
    )
    assert build_family_relift_guidance(
        artifact, forward
    ).to_json() == build_family_relift_guidance(
        artifact, reversed_result
    ).to_json()


def test_family_guidance_deduplicates_consensus_exact_differences() -> None:
    result = Gate1FamilyRunner(compiler=_AllDriftCompiler()).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _family(),
        default_triton_rule_registry(),
        _Writer(),
    )
    guidance = build_family_relift_guidance(_artifact(), result)

    assert guidance is not None
    for category in guidance.category_guidance:
        assert "profile_differences" not in category
        assert category["raw_difference_count_by_profile"]
        assert category["consensus_differences"]
        assert all(
            item["profile_ids"] == ["n1", "n65"]
            for item in category["consensus_differences"]
        )
        assert category["profile_specific_differences"] == []


def test_family_guidance_groups_shared_loop_structure_gap() -> None:
    result = Gate1FamilyRunner(compiler=_LoopGapCompiler()).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _family(),
        default_triton_rule_registry(),
        _Writer(),
    )
    guidance = build_family_relift_guidance(_artifact(), result)

    assert guidance is not None
    structural = guidance.structural_guidance
    assert structural is not None
    assert structural["policy"] == (
        "deterministic_fixed_c_family_structure_guidance_v1"
    )
    assert len(structural["groups"]) == 1
    group = structural["groups"][0]
    assert group["profile_ids"] == ["n1", "n65"]
    assert "loop_structure" in group["mismatch_coordinates"]
    assert group["source"]["loop_count"] == 1
    assert group["replay"]["loop_count"] == 0


def test_gate1_family_failure_relifts_once_then_rechecks_every_profile() -> None:
    compiler = _RoundCompiler()
    writer = _RoundWriter()
    relifter = _Relifter()
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            max_relifts=3,
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _profile("n1", 1).case,
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
        specialization_family=_family(),
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    assert writer.calls == 2
    assert len(relifter.calls) == 1
    assert len(result.candidates) == 2
    first, second = result.candidates
    assert first.comparison is not None and first.comparison.hard_pass
    assert not first.overall_hard_pass
    assert first.specialization_evidence["failed_profile_ids"] == ["n65"]
    by_profile = {
        item["profile_id"]: item
        for item in first.specialization_evidence["records"]
    }
    assert by_profile["n1"]["source_compile_reused"] is True
    assert by_profile["n1"]["replay_compile_reused"] is True
    assert by_profile["n65"]["source_compile_reused"] is False
    assert by_profile["n65"]["replay_compile_reused"] is False
    assert first.relift_guidance is not None
    assert first.relift_guidance.policy == (
        "deterministic_fixed_c_family_hard_diff_guidance_v2"
    )
    assert second.overall_hard_pass
    request = relifter.calls[0]["request"]
    assert request.trigger is ReliftTrigger.NORMAL_NO_CLOSURE
    guidance = request.semantic_guidance
    assert guidance is not None
    family_summary = guidance.current_j_summary["fixed_c_family"]
    assert family_summary["hard_closed_profiles"] == ["n1"]
    assert family_summary["failed_profiles"] == ["n65"]
    evidence = relifter.calls[0]["history"][-1]
    assert evidence.specialization_evidence["failed_profile_ids"] == ["n65"]
    assert all(
        "comparison" not in record
        for record in evidence.specialization_evidence["records"]
    )
    assert evidence.overall_hard_pass is False
    assert [
        n
        for code, n in compiler.calls
        if code == "REPLAY:1"
    ] == [1, 65]
    assert [
        n
        for code, n in compiler.calls
        if code == "REPLAY:2"
    ] == [1, 65]


def test_primary_hard_mismatch_still_completes_secondary_family_profiles() -> None:
    compiler = _Compiler(drift_profile=1)
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            max_relifts=1,
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _profile("n1", 1).case,
        default_triton_rule_registry(),
        _Writer(),
        specialization_family=_family(),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    candidate = result.candidates[0]
    assert candidate.specialization_evidence["failed_profile_ids"] == ["n1"]
    assert [
        record["profile_id"]
        for record in candidate.specialization_evidence["records"]
    ] == ["n1", "n65"]
    assert [n for code, n in compiler.calls if code == "REPLAY"] == [1, 65]


def test_gate1_family_stagnation_uses_complete_family_diff_digest() -> None:
    compiler = _Compiler(drift_profile=65)
    writer = _Writer()
    relifter = _Relifter()
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            max_relifts=3,
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _profile("n1", 1).case,
        default_triton_rule_registry(),
        writer,
        relifter=relifter,
        specialization_family=_family(),
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "deterministic_stagnation"
    assert result.stagnation is not None
    assert writer.calls == 2
    assert len(relifter.calls) == 1
    assert len(result.candidates) == 2
    assert all(
        candidate.specialization_evidence["hard_comparison_complete"]
        for candidate in result.candidates
    )
    assert result.stagnation.hard_difference_digest == (
        result.candidates[0].effective_hard_difference_digest
    )
    assert result.candidates[0].effective_hard_difference_digest == (
        result.candidates[1].effective_hard_difference_digest
    )


def test_gate1_integrated_family_closes_random_runtime_profiles() -> None:
    rng = random.Random(20260718)
    extents = tuple(dict.fromkeys(rng.randint(1, 4096) for _ in range(12)))
    profiles = tuple(
        _profile(f"random_{index:02d}", extent)
        for index, extent in enumerate(extents)
    )
    family = Gate1SpecializationFamily(
        "integrated-random-family",
        profiles[0].profile_id,
        profiles,
    )
    result = Gate1Runner(
        compiler=_Compiler(),
        config=Gate1Config(
            max_relifts=1,
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        profiles[0].case,
        default_triton_rule_registry(),
        _Writer(),
        specialization_family=family,
    )

    assert result.status is Gate1Status.REPLAY_CLOSED
    evidence = result.candidates[0].specialization_evidence
    assert evidence["all_passed"] is True
    assert evidence["hard_comparison_complete"] is True
    assert [record["profile_id"] for record in evidence["records"]] == [
        profile.profile_id for profile in profiles
    ]


def test_family_replay_compile_failure_finishes_sweep_before_relift_exit() -> None:
    compiler = _FamilyCompileFailureCompiler()
    family = Gate1SpecializationFamily(
        "three-profile-family",
        "n1",
        (
            _profile("n1", 1),
            _profile("n65", 65),
            _profile("n129", 129),
        ),
    )
    result = Gate1Runner(
        compiler=compiler,
        config=Gate1Config(
            max_relifts=1,
            stable_closures=1,
            run_semantic_interventions=False,
            run_masked_ablation=False,
        ),
    ).run(
        KernelSource.from_code("SOURCE", "kernel", trusted=True),
        _artifact(),
        _profile("n1", 1).case,
        default_triton_rule_registry(),
        _Writer(),
        specialization_family=family,
    )

    assert result.status is Gate1Status.NO_CLOSURE
    assert result.termination_reason == "relift_budget_exhausted"
    candidate = result.candidates[0]
    assert candidate.relift_trigger is ReliftTrigger.REPLAY_COMPILE_FAILED
    assert candidate.relift_guidance is None
    assert candidate.effective_hard_difference_digest is None
    records = candidate.specialization_evidence["records"]
    assert [record["profile_id"] for record in records] == [
        "n1",
        "n65",
        "n129",
    ]
    assert [record["failure_stage"] for record in records] == [
        "passed",
        "replay_compile",
        "passed",
    ]
    assert [
        n for code, n in compiler.calls if "REPLAY" in code
    ] == [1, 65, 129]


def test_family_rejects_c_drift_and_static_shape_drift_before_generation() -> None:
    with pytest.raises(ValueError, match="fixed-C"):
        Gate1SpecializationFamily(
            "bad-c",
            "a",
            (
                _profile("a", 1),
                Gate1SpecializationProfile(
                    "b",
                    CompileCase(
                        _profile("b0", 2).case.arguments,
                        (1,),
                        {"BLOCK": 128, "num_warps": 4},
                    ),
                ),
            ),
        )

    artifact = _artifact()
    payload = artifact.to_json()
    payload["j"]["intent"]["intent_function"]["inputs"][0]["shape"] = [17]
    payload["j"]["intent"]["intent_function"]["outputs"][0]["shape"] = [17]
    static = Gate1Artifact.from_json(payload)
    writer = _Writer()
    with pytest.raises(ValueError, match="static J extent"):
        Gate1FamilyRunner(compiler=_Compiler()).run(
            KernelSource.from_code("SOURCE", "kernel", trusted=True),
            static,
            _family(),
            default_triton_rule_registry(),
            writer,
        )
    assert writer.calls == 0


def test_pi_necessity_reaches_fresh_writer_compile_and_hard_compare() -> None:
    artifact = _artifact()
    profile = _profile("n65", 65)
    interface = {
        "arg_names": ["x", "out", "n", "BLOCK"],
        "constexprs": ["BLOCK"],
        "constexpr_indices": [3],
        "do_not_specialize": ["n"],
        "do_not_specialize_raw": [2],
        "grid": list(profile.case.grid),
        "grid_contract": list(artifact.interface.grid),
    }
    source_fp = fingerprint_ttir(
        GOOD_TTIR,
        python_interface=interface,
        schedule=profile.case.schedule,
    )
    writer = _Writer()
    records = run_pi_ablations(
        artifact,
        source_fp,
        profile.case,
        default_triton_rule_registry(),
        writer,
        compiler=_Compiler(),
        limit=2,
    )

    assert [item.status for item in records] == [
        PiAblationStatus.UNEXPECTED_CLOSURE,
        PiAblationStatus.UNEXPECTED_CLOSURE,
    ]
    assert all(item.attempts == 1 for item in records)
    assert all(item.generation is not None for item in records)
    assert all(item.compile is not None for item in records)
    assert all(item.comparison is not None for item in records)
    assert not any(item.supports_pi_necessity for item in records)
    assert writer.calls == 2
    assert writer.kwargs[0]["ablation_kind"] == "j_only"
    assert writer.kwargs[0]["plan_override"] == ReplayPlan([])
    assert writer.kwargs[1]["ablation_kind"] == "plan_step_dropout"
    assert writer.kwargs[1]["plan_override"] == records[1].spec.plan

    killed = run_pi_ablations(
        artifact,
        source_fp,
        profile.case,
        default_triton_rule_registry(),
        _Writer(),
        compiler=_AllDriftCompiler(),
        limit=1,
    )
    assert [item.status for item in killed] == [PiAblationStatus.DEGRADED]
    assert killed[0].supports_pi_necessity
    assert killed[0].comparison is not None
    assert not killed[0].comparison.hard_pass

    strategy_writer = _Writer()
    strategy_records = run_pi_ablations(
        artifact,
        source_fp,
        profile.case,
        default_triton_rule_registry(),
        strategy_writer,
        compiler=_Compiler(),
        limit=1,
        kinds=(PiAblationKind.PLAN_MECHANISM_SUBSTITUTION,),
    )
    assert [item.status for item in strategy_records] == [
        PiAblationStatus.SEMANTICS_PRESERVED
    ]
    strategy_spec = strategy_records[0].spec
    assert strategy_spec.artifact is not None
    assert strategy_spec.artifact.j == artifact.j
    assert strategy_spec.strategy_before != strategy_spec.strategy_after
    assert strategy_spec.artifact.plan != artifact.plan
    assert strategy_writer.kwargs == [
        {
            "masked": False,
            "plan_override": None,
            "ablation_kind": None,
        }
    ]
