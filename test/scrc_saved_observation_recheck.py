"""Recheck saved real-LLM Gate1 observations with the current deterministic stack.

This is a diagnostic migration tool, not a fresh SCRC experiment.  It reuses a
persisted LiftArtifact and persisted replay source, then reruns current artifact
validation, source/replay Triton compilation and normal hard comparison.  It
never calls Lift, Relift or a replay model.  An optional anti-leakage migration
reuses the saved intervention and actual-pi masked-J replay sources, without
repairing or regenerating either. Gate2 is optional, but it runs only when the
saved artifact is re-accepted by the current Gate1 and can be bound to a typed
IntentProgram plus the current replay/scalar protocol. Fixed-C family execution
is also optional; when enabled, each saved replay arm is evaluated by the
current in-loop full-family policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from intentir.verify.gate1.compiler import KernelSource
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import (
    Gate1Config,
    Gate1Runner,
    canonical_artifact_digest,
)
from intentir.verify.gate1.composer import assert_replay_plan_matches
from intentir.verify.gate1.program_adapter import intent_program_from_gate1
from intentir.verify.gate1.replay import StaticReplayWriter
from intentir.verify.gate1.rule_release import load_rule_release
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate2 import Gate2Runner

from scrc_heldout_batch import READY_CASES, _compile_case, build_heldout_case
from scrc_live_e2e import (
    _gate1_specialization_family,
    _gate2_case,
    _gate2_evidence_json,
    _gate2_suite,
    _gate2_suite_evidence_json,
    _select_gate1_specialization_evidence,
    _select_gate2_replay_observation,
)


RECHECK_PROTOCOL = (
    "saved_real_llm_gate1_gate2_family_antileak_recheck_v4"
)


def saved_anti_leakage_sources(
    gate1: dict[str, Any],
) -> tuple[tuple[str, ...], str]:
    """Return saved mutant arms and the saved actual-pi masked-J arm."""

    interventions = gate1.get("interventions")
    if not isinstance(interventions, list):
        raise ValueError("saved report has no intervention observation list")
    mutant_sources: list[str] = []
    for index, record in enumerate(interventions):
        generation = (
            record.get("generation") if isinstance(record, dict) else None
        )
        code = (
            generation.get("code")
            if isinstance(generation, dict)
            else None
        )
        if not isinstance(code, str):
            raise ValueError(
                f"saved intervention {index} has no replay source"
            )
        mutant_sources.append(code)
    masked_generation = gate1.get("masked_generation")
    masked_code = (
        masked_generation.get("code")
        if isinstance(masked_generation, dict)
        else None
    )
    if not isinstance(masked_code, str):
        raise ValueError("saved report has no actual-pi masked-J replay source")
    return tuple(mutant_sources), masked_code


def _saved_generation_identity(
    generation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(generation, dict):
        return None
    return {
        key: generation.get(key)
        for key in (
            "requested_model",
            "response_model",
            "response_id",
            "generation_id",
            "code_sha256",
        )
    }


def _artifact_digest_from_compose_provenance(
    provenance: Mapping[str, Any],
) -> str | None:
    """Recover the canonical artifact digest from legacy shadow evidence."""

    keys = (
        "composer_version",
        "j_digest",
        "interface_digest",
        "schedule_digest",
        "strategy_digest",
        "registry_digest",
        "plan_digest",
    )
    if any(
        not isinstance(provenance.get(key), str) or not provenance.get(key)
        for key in keys
    ):
        return None
    payload = {
        "digest_schema": "scrc-artifact-semantic-v1",
        **{key: provenance[key] for key in keys},
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _legacy_round_artifact_digests(
    composer_shadow: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    """Return only unambiguous match-status round bindings."""

    by_round: dict[int, set[str]] = {}
    for record in composer_shadow:
        if not isinstance(record, Mapping) or record.get("status") != "match":
            continue
        relift_round = record.get("relift_round")
        provenance = record.get("compose_provenance")
        if not isinstance(relift_round, int) or not isinstance(
            provenance, Mapping
        ):
            continue
        digest = _artifact_digest_from_compose_provenance(provenance)
        if digest is not None:
            by_round.setdefault(relift_round, set()).add(digest)
    return {
        relift_round: next(iter(digests))
        for relift_round, digests in by_round.items()
        if len(digests) == 1
    }


def select_saved_candidate(
    candidates: list[dict[str, Any]],
    *,
    final_artifact_digest: str,
    composer_shadow: Sequence[Mapping[str, Any]] = (),
) -> tuple[int, dict[str, Any]]:
    """Select replay code only when its artifact binding is provable.

    Schema31 candidates carry ``artifact_digest`` directly.  Older saved
    observations are accepted only when a same-round, match-status composer
    shadow contains complete canonical provenance that hashes to the final
    artifact digest.  Missing evidence is a coverage gap, not permission to
    pair an older replay with the final Lift artifact.
    """

    if not isinstance(final_artifact_digest, str) or not final_artifact_digest:
        raise ValueError("final artifact digest must be non-empty")
    legacy_digests = _legacy_round_artifact_digests(composer_shadow)
    generated = [
        (index, candidate)
        for index, candidate in enumerate(candidates)
        if isinstance(candidate.get("generation"), dict)
        and isinstance(candidate["generation"].get("code"), str)
    ]
    if not generated:
        raise ValueError("saved report has no replay source observation")
    aligned = [
        (index, candidate)
        for index, candidate in generated
        if (
            candidate.get("artifact_digest")
            or legacy_digests.get(candidate.get("relift_round"))
        )
        == final_artifact_digest
    ]
    if not aligned:
        raise ValueError(
            "saved report has no replay source bound to the final artifact"
        )
    hard_closed = [
        item
        for item in aligned
        if isinstance(item[1].get("comparison"), dict)
        and item[1]["comparison"].get("hard_pass") is True
    ]
    if hard_closed:
        return max(
            hard_closed,
            key=lambda item: (
                int(item[1].get("relift_round", -1)),
                int(item[1].get("generation_round", -1)),
                item[0],
            ),
        )
    return max(
        aligned,
        key=lambda item: (
            int(item[1].get("relift_round", -1)),
            int(item[1].get("generation_round", -1)),
            item[0],
        ),
    )


def rehydrate_saved_artifact(
    payload: dict[str, Any], *, source: str
) -> Gate1Artifact:
    artifact = dict(payload)
    artifact.pop("source_sha256", None)
    artifact.pop("source_bytes", None)
    artifact["source"] = source
    return Gate1Artifact.from_json(artifact)


def run_saved_gate2(
    case: Any,
    artifact: Gate1Artifact,
    replay_code: str,
    *,
    seed: int,
    final_artifact_digest: str,
    gate1_j_digest: str,
    program: Any,
    runner: Gate2Runner | None = None,
) -> dict[str, Any]:
    """Run current Gate2 for a saved replay re-accepted by current Gate1."""

    replay = KernelSource.from_code(
        replay_code, artifact.interface.function_name
    )
    result = (runner or Gate2Runner()).run(
        replay,
        _gate2_case(
            artifact,
            case,
            seed,
            program=program,
            final_artifact_digest=final_artifact_digest,
            gate1_j_digest=gate1_j_digest,
            replay_artifact_digest=final_artifact_digest,
        ),
    )
    return {
        "saved_observation_recheck": True,
        "required_gate1_state": "replay_closed",
        **_gate2_evidence_json(result),
    }


def run_saved_gate2_suite(
    case: Any,
    artifact: Gate1Artifact,
    replay_code: str,
    *,
    seeds: tuple[int, ...],
    final_artifact_digest: str,
    gate1_j_digest: str,
    program: Any,
    runner: Gate2Runner | None = None,
) -> dict[str, Any]:
    """Run the current declared Gate2 matrix as independent evidence."""

    replay = KernelSource.from_code(
        replay_code, artifact.interface.function_name
    )
    result = (runner or Gate2Runner()).run_suite(
        replay,
        _gate2_suite(
            artifact,
            case,
            seeds,
            program=program,
            final_artifact_digest=final_artifact_digest,
            gate1_j_digest=gate1_j_digest,
            replay_artifact_digest=final_artifact_digest,
        ),
    )
    return {
        "saved_observation_recheck": True,
        "required_gate1_state": "replay_closed",
        **_gate2_suite_evidence_json(result),
    }


def run_saved_recheck(
    case_name: str,
    report_path: Path,
    *,
    run_gate2: bool = False,
    run_gate2_matrix: bool = False,
    run_family: bool = False,
    run_anti_leakage: bool = False,
    seed: int = 20260715,
    rule_release: Path | None = None,
) -> dict[str, Any]:
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    case = build_heldout_case(case_name)
    gate1 = saved.get("gate1")
    if not isinstance(gate1, dict):
        raise ValueError("saved report has no Gate1 result")
    artifact_payload = gate1.get("artifact")
    if not isinstance(artifact_payload, dict):
        raise ValueError("saved report has no complete Gate1 artifact")
    candidates = gate1.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("saved report has no Gate1 candidate list")
    artifact = rehydrate_saved_artifact(
        artifact_payload, source=case.source_text
    )
    registry = default_triton_rule_registry()
    if rule_release is not None:
        registry = load_rule_release(rule_release).apply(registry)
    registry = registry.frozen_snapshot()
    final_saved_artifact_digest = canonical_artifact_digest(
        artifact,
        registry,
    )
    composer_shadow = gate1.get("composer_shadow")
    selected_index, selected = select_saved_candidate(
        candidates,
        final_artifact_digest=final_saved_artifact_digest,
        composer_shadow=(
            composer_shadow if isinstance(composer_shadow, list) else ()
        ),
    )
    generation = selected["generation"]
    specialization_family = (
        _gate1_specialization_family(case)
        if run_family or run_anti_leakage
        else None
    )
    if run_anti_leakage:
        mutant_sources, masked_source = saved_anti_leakage_sources(gate1)
        writer = StaticReplayWriter(
            [generation["code"], *mutant_sources],
            masked=masked_source,
        )
    else:
        mutant_sources = ()
        writer = StaticReplayWriter(generation["code"])
    saved_anti_leakage_identity = (
        {
            "interventions": [
                {
                    "intervention_id": (
                        record["spec"].get("intervention_id")
                        if isinstance(record.get("spec"), dict)
                        else None
                    ),
                    "generation": _saved_generation_identity(
                        record.get("generation")
                    ),
                }
                for record in gate1.get("interventions", [])
                if isinstance(record, dict)
            ],
            "masked_generation": _saved_generation_identity(
                gate1.get("masked_generation")
            ),
        }
        if run_anti_leakage
        else None
    )
    result = Gate1Runner(
        config=Gate1Config(
            max_relifts=1,
            stable_closures=1,
            run_semantic_interventions=run_anti_leakage,
            max_semantic_interventions=len(mutant_sources),
            run_masked_ablation=run_anti_leakage,
        )
    ).run(
        case.kernel_source,
        artifact,
        _compile_case(case),
        registry,
        writer,
        specialization_family=specialization_family,
    )
    current = result.to_json()
    current["artifact"].pop("source", None)
    try:
        current_artifact_digest = canonical_artifact_digest(
            result.artifact,
            registry,
        )
        current_composed = assert_replay_plan_matches(
            result.artifact, registry
        )
        current_gate1_j_digest = current_composed.provenance.j_digest
        current_program = intent_program_from_gate1(result.artifact)
    except Exception as exc:
        current_artifact_digest = None
        current_gate1_j_digest = None
        current_program = None
        current_artifact_identity_error = f"{type(exc).__name__}: {exc}"
    else:
        current_artifact_identity_error = None
    aligned_gate2_candidate = _select_gate2_replay_observation(
        result,
        final_artifact_digest=current_artifact_digest,
    )
    family_evidence = _select_gate1_specialization_evidence(
        result,
        final_artifact_digest=current_artifact_digest,
    )
    aligned_generation = (
        None
        if aligned_gate2_candidate is None
        else aligned_gate2_candidate.generation
    )
    comparison = (
        None
        if aligned_gate2_candidate is None
        else aligned_gate2_candidate.comparison
    )
    gate2_suite = (
        run_saved_gate2_suite(
            case,
            result.artifact,
            aligned_generation.code,
            seeds=(seed, seed + 1, seed + 2),
            final_artifact_digest=current_artifact_digest,
            gate1_j_digest=current_gate1_j_digest,
            program=current_program,
        )
        if (
            run_gate2_matrix
            and result.replay_closed
            and aligned_generation is not None
            and current_artifact_digest is not None
            and current_gate1_j_digest is not None
            and current_program is not None
        )
        else None
    )
    gate2 = (
        run_saved_gate2(
            case,
            result.artifact,
            aligned_generation.code,
            seed=seed,
            final_artifact_digest=current_artifact_digest,
            gate1_j_digest=current_gate1_j_digest,
            program=current_program,
        )
        if (
            run_gate2
            and gate2_suite is None
            and result.replay_closed
            and aligned_generation is not None
            and current_artifact_digest is not None
            and current_gate1_j_digest is not None
            and current_program is not None
        )
        else None
    )
    gate2_status = (
        ("gate2_pass" if gate2_suite["passed"] else "gate2_suite_failed")
        if gate2_suite is not None
        else (
            "coverage_gap"
            if (run_gate2 or run_gate2_matrix)
            and aligned_generation is None
            else (None if gate2 is None else gate2["status"])
        )
    )
    return {
        "protocol": RECHECK_PROTOCOL,
        "role": "diagnostic_not_fresh_lift_or_replay",
        "case": case_name,
        "source_report": str(report_path),
        "historical_gate1_status": gate1.get("status"),
        "selected_saved_candidate": {
            "index": selected_index,
            "relift_round": selected.get("relift_round"),
            "historical_hard_pass": (
                selected.get("comparison", {}).get("hard_pass")
                if isinstance(selected.get("comparison"), dict)
                else None
            ),
            "code_sha256": generation.get("code_sha256"),
            "generation": _saved_generation_identity(generation),
            "final_artifact_digest": final_saved_artifact_digest,
            "candidate_artifact_digest": (
                selected.get("artifact_digest")
                or _legacy_round_artifact_digests(
                    composer_shadow
                    if isinstance(composer_shadow, list)
                    else ()
                ).get(selected.get("relift_round"))
            ),
        },
        "selected_saved_anti_leakage": saved_anti_leakage_identity,
        "current_summary": {
            "status": result.status.value,
            "termination_reason": result.termination_reason,
            "source_compile_ok": bool(
                result.source_compile and result.source_compile.ok
            ),
            "replay_compile_ok": bool(
                result.candidates and result.candidates[-1].compiled
            ),
            "hard_pass": None if comparison is None else comparison.hard_pass,
            "failed_hard_categories": (
                []
                if comparison is None
                else [
                    component.category
                    for component in comparison.components
                    if component.hard and not component.passed
                ]
            ),
            "gate2_status": gate2_status,
            "family_enabled": specialization_family is not None,
            "anti_leakage_replayed": run_anti_leakage,
            "intervention_family_passed": (
                None
                if not run_anti_leakage
                else bool(
                    result.interventions
                    and all(item.passed for item in result.interventions)
                )
            ),
            "masked_family_outcome": (
                None
                if result.masked_family_evidence is None
                else result.masked_family_evidence["outcome"]
            ),
            "family_all_passed": (
                None
                if family_evidence is None
                else family_evidence["all_passed"]
            ),
            "family_failed_profile_ids": (
                []
                if family_evidence is None
                else family_evidence["failed_profile_ids"]
            ),
        },
        "current_gate1": current,
        "current_gate1_specialization_family": family_evidence,
        "current_gate2_replay_alignment": {
            "policy": "final_gate1_artifact_digest_exact_match_v1",
            "status": (
                "aligned"
                if aligned_gate2_candidate is not None
                else "coverage_gap"
            ),
            "final_artifact_digest": current_artifact_digest,
            "candidate_artifact_digest": (
                None
                if aligned_gate2_candidate is None
                else aligned_gate2_candidate.artifact_digest
            ),
            "reason": (
                None
                if aligned_gate2_candidate is not None
                else (
                    "final_gate1_artifact_digest_unavailable"
                    if current_artifact_digest is None
                    else "no_executable_candidate_bound_to_final_gate1_artifact"
                )
            ),
            "diagnostic": current_artifact_identity_error,
        },
        "current_gate2": gate2,
        "current_gate2_suite": gate2_suite,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=READY_CASES)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gate2", action="store_true")
    parser.add_argument("--gate2-matrix", action="store_true")
    parser.add_argument("--family", action="store_true")
    parser.add_argument("--anti-leakage", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--rule-release", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report_path = args.reports_dir / f"{args.case}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_saved_recheck(
            args.case,
            report_path,
            run_gate2=args.gate2,
            run_gate2_matrix=args.gate2_matrix,
            run_family=args.family,
            run_anti_leakage=args.anti_leakage,
            seed=args.seed,
            rule_release=args.rule_release,
        )
    except Exception as exc:
        result = {
            "protocol": RECHECK_PROTOCOL,
            "role": "diagnostic_not_fresh_lift_or_replay",
            "case": args.case,
            "source_report": str(report_path),
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 2
    else:
        code = 0
    output = args.out_dir / f"{args.case}.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.get("current_summary", result), ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
