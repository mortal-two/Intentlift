from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from intentir.verify.gate1.codegen import validate_triton_module
from intentir.verify.gate1.evaluation import FailureCause, summarize_capability
from intentir.verify.gate1.leakage import LeakageKind, scan_static_leakage


ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_SCRIPT = ROOT / "test" / "scrc_gate1_capability.py"


def _load_capability_module():
    name = "_intentir_test_scrc_gate1_capability"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CAPABILITY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_capability_module()


REQUIRED_CASES = {
    "add.correct",
    "row_sum.correct",
    "softmax.correct",
    "rmsnorm.correct",
    "matmul.correct",
    "add.commuted",
    "add.redundant_zero",
    "row_sum.reversed_predicate",
    "row_sum.wrong_fill",
    "softmax.missing_stabilization",
    "softmax.wrong_fill",
    "softmax.reversed_sub",
    "rmsnorm.sqrt_reciprocal",
    "rmsnorm.wrong_divisor",
    "matmul.wrong_dot_operands",
    "matmul.wrong_b_address",
    "rmsnorm.missing_do_not_specialize",
    "add.pi_api_leakage",
}


def test_capability_catalog_covers_baselines_equivalences_and_hard_mutants() -> None:
    cases = probe.capability_cases()
    ids = [case.case_id for case in cases]
    assert len(ids) == len(set(ids))
    assert set(ids) == REQUIRED_CASES
    assert {case.kernel for case in cases} == {
        "add",
        "row_sum",
        "softmax",
        "rmsnorm",
        "matmul",
    }
    assert {case.family for case in cases} == {
        "baseline",
        "equivalent",
        "semantic_mutant",
        "metadata_variant",
        "leakage_mutant",
    }


@pytest.mark.parametrize("case", probe.capability_cases(), ids=lambda case: case.case_id)
def test_every_candidate_is_a_valid_self_contained_triton_module(case) -> None:
    code = case.candidate_code()
    validation = validate_triton_module(code, required_function=case.function_name)
    assert case.function_name in validation.function_names


def test_all_text_mutants_change_exactly_the_reviewed_replay_fragment() -> None:
    base = probe.REPLAY_MODULE.read_text(encoding="utf-8")
    for case in probe.capability_cases():
        candidate = case.candidate_code(base)
        if case.mutations:
            assert candidate != base, case.case_id
        else:
            assert candidate == base, case.case_id


def test_compile_specializations_use_same_c_values_for_source_and_replay() -> None:
    for kernel in ("add", "row_sum", "softmax", "rmsnorm", "matmul"):
        case = probe.compile_case(kernel)
        assert case.device == "cuda"
        assert case.grid
        assert case.schedule["num_warps"] == 4
        assert case.schedule["num_stages"] == 2
        assert case.schedule["num_ctas"] == 1
    assert probe.compile_case("add").schedule["BLOCK"] == 64
    assert probe.compile_case("row_sum").schedule["BLOCK_N"] == 64
    assert probe.compile_case("matmul").schedule["BLOCK_K"] == 16


def test_launch_grid_is_hard_but_jit_specialization_policy_is_not() -> None:
    specialization = probe.compile_case("rmsnorm")
    metadata = probe._hard_interface_metadata(
        {
            "arg_names": ["x_ptr", "eps", "BLOCK_N"],
            "constexprs": ["BLOCK_N"],
            "do_not_specialize": ["eps"],
        },
        specialization,
        "rmsnorm_kernel",
    )
    assert metadata["function_name"] == "rmsnorm_kernel"
    assert metadata["launch_grid"] == [3]
    # The compiler may report this field, but the Gate1 fingerprint filters it
    # out because it is not part of the callable signature or J.A/J.B.
    assert metadata["do_not_specialize"] == ["eps"]
    # C values are deliberately absent here; they are compared in the soft
    # schedule component rather than being smuggled into the hard ABI.
    assert "BLOCK_N" not in metadata


def test_pi_api_leakage_is_rejected_before_compilation() -> None:
    case = next(
        case
        for case in probe.capability_cases()
        if case.case_id == "add.pi_api_leakage"
    )
    report = scan_static_leakage(case.leaked_plan)
    assert not report.passed
    assert any(
        violation.kind is LeakageKind.API_TOKEN and violation.evidence == "tl.load"
        for violation in report.violations
    )
    assert case.expected_cause is FailureCause.PLAN_LEAKAGE


def test_full_gate2_matrix_has_required_boundary_shapes_values_and_seeds() -> None:
    required_seeds = {0, 1, 2}

    add = probe.gate2_trials("add", "full")
    assert {trial.dimensions[0] for trial in add} == {1, 63, 64, 65, 113}
    assert {trial.seed for trial in add} == required_seeds
    assert {trial.distribution for trial in add} == {"normal", "special"}

    for kernel in ("row_sum", "softmax", "rmsnorm"):
        trials = probe.gate2_trials(kernel, "full")
        assert {trial.dimensions[-1] for trial in trials} == {1, 63, 64, 65}
        assert {trial.seed for trial in trials} == required_seeds

    softmax = probe.gate2_trials("softmax", "full")
    assert {trial.distribution for trial in softmax} == {
        "normal",
        "negative",
        "extreme",
    }
    assert "special" in {
        trial.distribution for trial in probe.gate2_trials("row_sum", "full")
    }
    assert "special" in {
        trial.distribution for trial in probe.gate2_trials("rmsnorm", "full")
    }

    matmul = probe.gate2_trials("matmul", "full")
    dimensions = {trial.dimensions for trial in matmul}
    assert (1, 1, 1) in dimensions
    assert (31, 31, 31) in dimensions
    assert (32, 32, 32) in dimensions
    assert (33, 33, 33) in dimensions
    assert any(53 in shape for shape in dimensions)
    assert any(shape[2] > 32 for shape in dimensions)
    assert {trial.seed for trial in matmul} == required_seeds
    assert "zero" in {trial.distribution for trial in matmul}


def test_design_target_preserves_precision_and_exposes_coverage_tradeoff() -> None:
    observations = probe.expected_observations()
    summary = summarize_capability(observations)
    assert summary.total == 18
    assert summary.true_accept == 10
    assert summary.false_accept == 0
    assert summary.false_reject == 1
    assert summary.true_reject == 7
    assert summary.gate2_precision_given_closed == 1.0
    assert summary.gate1_coverage_of_gate2_pass == pytest.approx(10 / 11)
    assert probe._gate2_pass_given_rejected(observations) == pytest.approx(1 / 8)


@pytest.mark.parametrize(
    ("category", "cause"),
    [
        ("interface", FailureCause.INTERFACE),
        ("op_multiset", FailureCause.OPERATOR),
        ("dataflow", FailureCause.DATAFLOW),
        ("reduction", FailureCause.REDUCTION),
        ("predicate_effect", FailureCause.PREDICATE_EFFECT),
        ("schedule-only", FailureCause.SCHEDULE),
        ("future-category", FailureCause.UNKNOWN),
    ],
)
def test_comparator_categories_map_to_stable_failure_taxonomy(category, cause) -> None:
    assert probe._comparison_cause(category) is cause
