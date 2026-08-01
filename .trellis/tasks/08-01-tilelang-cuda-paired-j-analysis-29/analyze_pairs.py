"""Deterministically compare accepted TileLang/generated-CUDA J artifacts.

The analysis consumes only Task28's exact accepted evidence.  It never invokes
Lift or Replay, never edits either artifact, and keeps frontend ABI/C deltas
separate from semantic-J correspondence.
"""

from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import canonical_artifact_digest


TASK_ROOT = Path(__file__).resolve().parent
SOURCE_TASK = TASK_ROOT.parent / "08-01-j-only-replay-multifrontend-batch-28"
SOURCE_RESULTS = SOURCE_TASK / "results_current.json"
SOURCE_COHORTS = SOURCE_TASK / "cohorts.json"
MANIFEST = TASK_ROOT / "cohort20.json"
SELECTION_FREEZE_COMMIT = "00e1f8db"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _accepted(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("typecheck") == "pass"
        and summary.get("gate1") == "replay_closed"
        and summary.get("gate2_passed") is True
        and summary.get("lift_agent_calls") == 0
        and summary.get("pi_visible_to_replay") is False
    )


def _primary_selection(
    manifest: Mapping[str, Any], cohorts: Mapping[str, Sequence[str]]
) -> tuple[list[str], dict[str, int]]:
    anchors = manifest.get("anchors")
    if not isinstance(anchors, Mapping):
        raise ValueError("cohort20 anchors must be an object")
    selected = [str(stem) for stem in anchors.values()]
    if len(selected) != len(set(selected)):
        raise ValueError("cohort20 anchors must be unique")
    universe = sorted({stem for members in cohorts.values() for stem in members})
    unknown = sorted(set(selected) - set(universe))
    if unknown:
        raise ValueError(f"cohort20 anchors are absent from Task28: {unknown}")
    counts = {
        name: sum(stem in members for stem in selected)
        for name, members in cohorts.items()
    }
    stop = int(manifest["fill_rule"]["stop_at_unique_pairs"])
    while len(selected) < stop:

        def rank(stem: str) -> tuple[float, int, str]:
            memberships = [
                name for name, members in cohorts.items() if stem in members
            ]
            gain = sum(1.0 / (1 + counts[name]) for name in memberships)
            return (-gain, -len(memberships), stem)

        chosen = min((stem for stem in universe if stem not in selected), key=rank)
        selected.append(chosen)
        for name, members in cohorts.items():
            counts[name] += int(chosen in members)
    return selected, counts


_NODE_REFERENCE_FIELDS = frozenset(
    {"id", "node_ref", "consumer_ref", "update_node_ref"}
)


def _replace_node_ids(
    value: Any,
    node_paths: Mapping[str, str],
    *,
    field: str | None = None,
) -> Any:
    if isinstance(value, str) and field in _NODE_REFERENCE_FIELDS:
        path = node_paths.get(value)
        return f"@node:{path}" if path is not None else value
    if isinstance(value, list):
        return [_replace_node_ids(item, node_paths, field=field) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_node_ids(item, node_paths, field=key)
            for key, item in value.items()
        }
    return value


def _identity_normalized_j(j: Mapping[str, Any]) -> dict[str, Any]:
    """Remove frontend identity while preserving every typed semantic fact."""

    payload = copy.deepcopy(dict(j))
    raw_nodes = payload.pop("node_ids", {})
    if not isinstance(raw_nodes, Mapping):
        raise TypeError("J.node_ids must be an object")
    node_paths = {str(node): str(path) for node, path in raw_nodes.items()}
    payload = _replace_node_ids(payload, node_paths)
    payload.pop("kernel_id", None)
    intent = payload.get("intent")
    if isinstance(intent, dict):
        function = intent.get("intent_function", intent)
        if isinstance(function, dict) and "name" in function:
            function["name"] = "@frontend-neutral-function"
    return payload


def _semantic_core(j: Mapping[str, Any]) -> dict[str, Any]:
    payload = _identity_normalized_j(j)
    payload.pop("signature", None)
    return payload


def _iter_ops(items: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for operation in items:
        yield operation
        body = operation.get("body", [])
        if isinstance(body, list):
            yield from _iter_ops(item for item in body if isinstance(item, Mapping))
        for arm_name in ("positive", "negative"):
            arm = operation.get(arm_name)
            if not isinstance(arm, Mapping):
                continue
            arm_body = arm.get("body", [])
            if isinstance(arm_body, list):
                yield from _iter_ops(
                    item for item in arm_body if isinstance(item, Mapping)
                )


def _intent_metrics(tile: Mapping[str, Any], cuda: Mapping[str, Any]) -> dict[str, Any]:
    tile_intent = _identity_normalized_j(tile)["intent"]
    cuda_intent = _identity_normalized_j(cuda)["intent"]
    tile_function = tile_intent.get("intent_function", tile_intent)
    cuda_function = cuda_intent.get("intent_function", cuda_intent)
    tile_ops = list(_iter_ops(tile_function.get("body", [])))
    cuda_ops = list(_iter_ops(cuda_function.get("body", [])))
    tile_sequence = [str(item.get("op")) for item in tile_ops]
    cuda_sequence = [str(item.get("op")) for item in cuda_ops]
    tile_set = set(tile_sequence)
    cuda_set = set(cuda_sequence)
    union = tile_set | cuda_set
    return {
        "exact": tile_intent == cuda_intent,
        "tilelang_node_count": len(tile_ops),
        "cuda_node_count": len(cuda_ops),
        "node_count_equal": len(tile_ops) == len(cuda_ops),
        "op_sequence_equal": tile_sequence == cuda_sequence,
        "op_kind_set_equal": tile_set == cuda_set,
        "op_kind_set_jaccard": (
            len(tile_set & cuda_set) / len(union) if union else 1.0
        ),
        "tilelang_op_sequence": tile_sequence,
        "cuda_op_sequence": cuda_sequence,
    }


def _buffer_profile(parameters: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
    return sorted(
        (
            str(parameter.get("mode")),
            str(parameter.get("dtype")),
            int(parameter.get("rank", 0)),
        )
        for parameter in parameters
        if parameter.get("kind") in {"tensor", "pointer"}
    )


def _ordered_buffer_profile(
    parameters: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str, int]]:
    return [
        (
            str(parameter.get("mode")),
            str(parameter.get("dtype")),
            int(parameter.get("rank", 0)),
        )
        for parameter in parameters
        if parameter.get("kind") in {"tensor", "pointer"}
    ]


def _scalar_profile(parameters: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, int]]:
    return [
        (
            str(parameter.get("mode")),
            str(parameter.get("dtype")),
            int(parameter.get("rank", 0)),
        )
        for parameter in parameters
        if parameter.get("kind") == "scalar"
    ]


def _signature_metrics(
    tile: Mapping[str, Any], cuda: Mapping[str, Any]
) -> dict[str, Any]:
    tile_parameters = list(tile.get("parameters", []))
    cuda_parameters = list(cuda.get("parameters", []))
    tile_buffers = _buffer_profile(tile_parameters)
    cuda_buffers = _buffer_profile(cuda_parameters)
    tile_ordered = _ordered_buffer_profile(tile_parameters)
    cuda_ordered = _ordered_buffer_profile(cuda_parameters)
    tile_scalars = _scalar_profile(tile_parameters)
    cuda_scalars = _scalar_profile(cuda_parameters)
    tile_kinds = Counter(str(item.get("kind")) for item in tile_parameters)
    cuda_kinds = Counter(str(item.get("kind")) for item in cuda_parameters)
    return {
        "raw_exact": tile == cuda,
        "buffer_profile_exact": tile_buffers == cuda_buffers,
        "ordered_buffer_profile_exact": tile_ordered == cuda_ordered,
        "tilelang_buffer_profile": tile_buffers,
        "cuda_buffer_profile": cuda_buffers,
        "tilelang_scalar_profile": tile_scalars,
        "cuda_scalar_profile": cuda_scalars,
        "extra_cuda_scalar_count": len(cuda_scalars) - len(tile_scalars),
        "tilelang_kind_counts": dict(sorted(tile_kinds.items())),
        "cuda_kind_counts": dict(sorted(cuda_kinds.items())),
        "carrier_kind_delta": tile_kinds != cuda_kinds,
        "tilelang_i1_buffers": sum(dtype == "i1" for _mode, dtype, _rank in tile_buffers),
        "cuda_i8_buffers": sum(dtype == "i8" for _mode, dtype, _rank in cuda_buffers),
    }


def _boundary_metrics(
    tile: Mapping[str, Any], cuda: Mapping[str, Any]
) -> dict[str, Any]:
    tile_boundary = _identity_normalized_j(tile)["boundary"]
    cuda_boundary = _identity_normalized_j(cuda)["boundary"]
    fields = sorted(set(tile_boundary) | set(cuda_boundary))
    field_exact = {
        field: tile_boundary.get(field) == cuda_boundary.get(field)
        for field in fields
    }

    def count(payload: Mapping[str, Any], field: str) -> int:
        value = payload.get(field, [])
        return len(value) if isinstance(value, (list, dict)) else int(value is not None)

    return {
        "exact": tile_boundary == cuda_boundary,
        "field_exact": field_exact,
        "tilelang_counts": {field: count(tile_boundary, field) for field in fields},
        "cuda_counts": {field: count(cuda_boundary, field) for field in fields},
    }


def _config_metrics(
    tile: Mapping[str, Any], cuda: Mapping[str, Any]
) -> dict[str, Any]:
    tile_slots = list(tile.get("slots", []))
    cuda_slots = list(cuda.get("slots", []))
    return {
        "exact": tile == cuda,
        "slots_exact": tile_slots == cuda_slots,
        "values_exact": tile.get("values", {}) == cuda.get("values", {}),
        "tilelang_slot_count": len(tile_slots),
        "cuda_slot_count": len(cuda_slots),
    }


def _load_frontend(
    stem: str,
    frontend: str,
    pair_evidence: Mapping[str, Any],
) -> tuple[Gate1Artifact, dict[str, Any]]:
    evidence = pair_evidence[frontend]
    if evidence.get("accepted") is not True:
        raise ValueError(f"Task28 pair {stem}/{frontend} is not accepted")
    selected = evidence.get("selected")
    if not isinstance(selected, Mapping) or not _accepted(selected):
        raise ValueError(f"Task28 selected evidence is not accepted: {stem}/{frontend}")
    expected_case = (
        f"tilelang:{stem}" if frontend == "tilelang" else f"cuda:tilelang_{stem}"
    )
    if selected.get("case") != expected_case:
        raise ValueError(f"Task28 selected case is stale: {stem}/{frontend}")
    run_dir = Path(str(selected["run_dir"])).resolve()
    expected_root = SOURCE_TASK.joinpath("runs").resolve()
    try:
        run_dir.relative_to(expected_root)
    except ValueError as exc:
        raise ValueError(f"run_dir escapes Task28 evidence root: {run_dir}") from exc
    summary = _read(run_dir / "summary.json")
    if selected.get("run") != run_dir.name:
        raise ValueError(f"Task28 selected run name disagrees with run_dir: {run_dir}")
    if not _accepted(summary) or summary.get("case") != expected_case:
        raise ValueError(f"run-local summary disagrees with Task28: {run_dir}")
    visibility = _read(run_dir / "pi_visibility.json")
    if visibility.get("packet_root") != ["c", "j", "u_f"]:
        raise ValueError(f"Replay packet is not J-only: {run_dir}")
    if visibility.get("pi_visible_to_replay") is not False:
        raise ValueError(f"pi was visible to Replay: {run_dir}")
    artifact_payload = _read(run_dir / "artifact.json")
    artifact = canonicalize_gate1_artifact(Gate1Artifact.from_json(artifact_payload))
    gate1 = _read(run_dir / "gate1.json")
    if gate1.get("artifact_digest") != canonical_artifact_digest(artifact):
        raise ValueError(f"Gate1 digest does not bind canonical artifact: {run_dir}")
    gate2 = _read(run_dir / "gate2.json")
    if gate2.get("passed") is not True:
        raise ValueError(f"run-local Gate2 is not passing: {run_dir}")
    authoring = _read(run_dir / "authoring.json")
    expected_pair = [f"tilelang:{stem}", f"cuda:tilelang_{stem}"]
    if authoring.get("stem") != stem or authoring.get("cases") != expected_pair:
        raise ValueError(f"authoring receipt does not bind the same-origin pair: {run_dir}")
    replay_path = run_dir / "replay.source"
    if not replay_path.is_file():
        raise ValueError(f"accepted run lacks exact Replay source: {run_dir}")
    return artifact, {
        "case": summary["case"],
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "artifact_sha256": _digest(artifact.to_json()),
        "gate1_artifact_digest": gate1["artifact_digest"],
        "typecheck": summary["typecheck"],
        "gate1": summary["gate1"],
        "gate2": summary["gate2"],
        "gate2_passed": summary["gate2_passed"],
        "lift_agent_calls": summary["lift_agent_calls"],
        "pi_visible_to_replay": summary["pi_visible_to_replay"],
        "gate1_elapsed_ms": float(summary["gate1_elapsed_ms"]),
        "replay_source_bytes": len(replay_path.read_bytes()),
        "replay_provider_calls": int(summary["replay_provider_calls"]),
        "task28_selected_run_bound": True,
        "same_origin_authoring_bound": True,
    }


def _pair_metrics(
    stem: str,
    pair_evidence: Mapping[str, Any],
    *,
    in_primary: bool,
) -> dict[str, Any]:
    tile_artifact, tile_provenance = _load_frontend(
        stem, "tilelang", pair_evidence
    )
    cuda_artifact, cuda_provenance = _load_frontend(stem, "cuda", pair_evidence)
    tile_payload = tile_artifact.to_json()
    cuda_payload = cuda_artifact.to_json()
    tile_j = tile_payload["j"]
    cuda_j = cuda_payload["j"]
    tile_identity = _identity_normalized_j(tile_j)
    cuda_identity = _identity_normalized_j(cuda_j)
    tile_core = _semantic_core(tile_j)
    cuda_core = _semantic_core(cuda_j)
    intent = _intent_metrics(tile_j, cuda_j)
    boundary = _boundary_metrics(tile_j, cuda_j)
    signature = _signature_metrics(tile_j["signature"], cuda_j["signature"])
    config = _config_metrics(tile_payload["c"], cuda_payload["c"])
    dimension_fields = {
        field: tile_j.get(field) == cuda_j.get(field)
        for field in (
            "dimension_constraints",
            "value_dimension_relations",
            "dimension_relations",
        )
    }
    section_exact = {
        "intent": intent["exact"],
        "boundary": boundary["exact"],
        "scopes": tile_identity.get("scopes") == cuda_identity.get("scopes"),
        "storage": tile_identity.get("storage") == cuda_identity.get("storage"),
        **dimension_fields,
    }
    categories: list[str] = []
    if signature["carrier_kind_delta"]:
        categories.append("tensor_pointer_carrier")
    if signature["extra_cuda_scalar_count"] != 0:
        categories.append("runtime_scalar_abi")
    if not signature["buffer_profile_exact"]:
        categories.append("typed_buffer_profile")
    elif not signature["ordered_buffer_profile_exact"]:
        categories.append("buffer_order")
    if not intent["exact"]:
        categories.append("layer_a_typed_representation")
    if not boundary["exact"]:
        categories.append("boundary")
    if not boundary["field_exact"].get("accesses", True):
        categories.append("access")
    if not boundary["field_exact"].get("effects", True):
        categories.append("effect")
    if not all(dimension_fields.values()):
        categories.append("dimension_contract")
    if not config["exact"]:
        categories.append("C")
    if tile_payload["pi"] != cuda_payload["pi"]:
        categories.append("pi")
    return {
        "stem": stem,
        "in_primary": in_primary,
        "cohort_memberships": list(pair_evidence["cohort_memberships"]),
        "provenance": {
            "tilelang": tile_provenance,
            "cuda": cuda_provenance,
        },
        "canonical_j_bytes": {
            "tilelang": len(_json_bytes(tile_j)),
            "cuda": len(_json_bytes(cuda_j)),
        },
        "full_j_exact": tile_j == cuda_j,
        "identity_normalized_j_exact": tile_identity == cuda_identity,
        "semantic_core_exact": tile_core == cuda_core,
        "section_exact": section_exact,
        "intent": intent,
        "boundary": boundary,
        "signature": signature,
        "c": config,
        "pi_exact": tile_payload["pi"] == cuda_payload["pi"],
        "difference_categories": categories,
    }


def _path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        current = current[key]
    return current


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    boolean_metrics = (
        "full_j_exact",
        "identity_normalized_j_exact",
        "semantic_core_exact",
        "intent.exact",
        "intent.node_count_equal",
        "intent.op_sequence_equal",
        "intent.op_kind_set_equal",
        "boundary.exact",
        "signature.raw_exact",
        "signature.buffer_profile_exact",
        "signature.ordered_buffer_profile_exact",
        "c.exact",
        "c.slots_exact",
        "c.values_exact",
        "pi_exact",
    )
    exact_counts = {
        metric: sum(bool(_path(row, metric)) for row in rows)
        for metric in boolean_metrics
    }
    boundary_fields = sorted(
        {
            field
            for row in rows
            for field in row["boundary"]["field_exact"]
        }
    )
    boundary_exact_counts = {
        field: sum(
            bool(row["boundary"]["field_exact"].get(field, False))
            for row in rows
        )
        for field in boundary_fields
    }
    dimension_fields = (
        "dimension_constraints",
        "value_dimension_relations",
        "dimension_relations",
    )
    dimension_exact_counts = {
        field: sum(bool(row["section_exact"][field]) for row in rows)
        for field in dimension_fields
    }
    category_counts = Counter(
        category for row in rows for category in row["difference_categories"]
    )
    op_jaccards = [float(row["intent"]["op_kind_set_jaccard"]) for row in rows]
    tile_bytes = [int(row["canonical_j_bytes"]["tilelang"]) for row in rows]
    cuda_bytes = [int(row["canonical_j_bytes"]["cuda"]) for row in rows]
    extra_scalars = [int(row["signature"]["extra_cuda_scalar_count"]) for row in rows]
    count = len(rows)

    def replay_cost(frontends: Sequence[str]) -> dict[str, Any]:
        records = [
            row["provenance"][frontend]
            for row in rows
            for frontend in frontends
        ]
        elapsed = [float(record["gate1_elapsed_ms"]) for record in records]
        source_bytes = [int(record["replay_source_bytes"]) for record in records]
        calls = [int(record["replay_provider_calls"]) for record in records]
        return {
            "run_count": len(records),
            "gate1_elapsed_ms": {
                "min": min(elapsed),
                "median": median(elapsed),
                "mean": mean(elapsed),
                "max": max(elapsed),
            },
            "replay_source_bytes": {
                "min": min(source_bytes),
                "median": median(source_bytes),
                "mean": mean(source_bytes),
                "max": max(source_bytes),
            },
            "provider_calls": {
                "total": sum(calls),
                "all_one_per_run": all(call == 1 for call in calls),
            },
        }

    return {
        "pair_count": count,
        "frontend_artifact_count": count * 2,
        "all_exact_accepted_provenance": all(
            _accepted(row["provenance"][frontend])
            and row["provenance"][frontend]["task28_selected_run_bound"] is True
            and row["provenance"][frontend]["same_origin_authoring_bound"] is True
            for row in rows
            for frontend in ("tilelang", "cuda")
        ),
        "exact_counts": exact_counts,
        "exact_rates": {
            metric: exact_counts[metric] / count for metric in boolean_metrics
        },
        "boundary_field_exact_counts": boundary_exact_counts,
        "dimension_field_exact_counts": dimension_exact_counts,
        "difference_category_counts": dict(sorted(category_counts.items())),
        "op_kind_set_jaccard": {
            "min": min(op_jaccards),
            "median": median(op_jaccards),
            "mean": mean(op_jaccards),
        },
        "canonical_j_bytes": {
            "tilelang_median": median(tile_bytes),
            "cuda_median": median(cuda_bytes),
            "tilelang_total": sum(tile_bytes),
            "cuda_total": sum(cuda_bytes),
        },
        "extra_cuda_scalars": {
            "pair_count_nonzero": sum(value != 0 for value in extra_scalars),
            "total": sum(extra_scalars),
            "median": median(extra_scalars),
            "mean": mean(extra_scalars),
        },
        "replay_cost": {
            "all": replay_cost(("tilelang", "cuda")),
            "tilelang": replay_cost(("tilelang",)),
            "cuda": replay_cost(("cuda",)),
        },
    }


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def _render_markdown(result: Mapping[str, Any]) -> str:
    primary = result["primary"]
    robust = result["robustness"]
    rows = result["pairs"]
    labels = [
        ("Full canonical J exact", "full_j_exact"),
        ("Identity-normalized J exact", "identity_normalized_j_exact"),
        ("Signature-free semantic core exact", "semantic_core_exact"),
        ("Layer-A intent exact", "intent.exact"),
        ("Op-kind set exact", "intent.op_kind_set_equal"),
        ("Boundary exact", "boundary.exact"),
        ("Raw logical signature exact", "signature.raw_exact"),
        ("Carrier-normalized typed buffer profile exact", "signature.buffer_profile_exact"),
        ("C exact", "c.exact"),
        ("Compact pi exact", "pi_exact"),
    ]
    lines = [
        "# Task29 TileLang / generated-CUDA paired-J results",
        "",
        f"- primary detailed cohort: {primary['pair_count']} unique pairs / "
        f"{primary['frontend_artifact_count']} accepted artifacts",
        f"- full robustness cohort: {robust['pair_count']} unique pairs / "
        f"{robust['frontend_artifact_count']} accepted artifacts",
        "- analysis calls Lift/Replay: 0",
        "- full-J equality is descriptive, never an acceptance gate",
        "",
        "| Metric | Primary 20 | Full 31 |",
        "|---|---:|---:|",
    ]
    for label, metric in labels:
        lines.append(
            f"| {label} | {primary['exact_counts'][metric]}/20 | "
            f"{robust['exact_counts'][metric]}/31 |"
        )
    lines.extend(
        [
            "",
            "## Exact accepted Replay cost",
            "",
            "| Cohort / frontend | Runs | Gate1 median (s) | Gate1 mean (s) | Replay bytes median | Provider calls |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cohort_name, aggregate in (("Primary 20", primary), ("Full 31", robust)):
        for frontend in ("tilelang", "cuda"):
            cost = aggregate["replay_cost"][frontend]
            lines.append(
                f"| {cohort_name} / {frontend} | {cost['run_count']} | "
                f"{cost['gate1_elapsed_ms']['median'] / 1000.0:.3f} | "
                f"{cost['gate1_elapsed_ms']['mean'] / 1000.0:.3f} | "
                f"{cost['replay_source_bytes']['median']:.1f} | "
                f"{cost['provider_calls']['total']} |"
            )
    lines.extend(
        [
            "",
            "## Difference taxonomy",
            "",
            "| Category | Primary 20 | Full 31 |",
            "|---|---:|---:|",
        ]
    )
    categories = sorted(
        set(primary["difference_category_counts"])
        | set(robust["difference_category_counts"])
    )
    for category in categories:
        lines.append(
            f"| `{category}` | "
            f"{primary['difference_category_counts'].get(category, 0)} | "
            f"{robust['difference_category_counts'].get(category, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Boundary field exactness",
            "",
            "| Boundary field | Primary 20 | Full 31 |",
            "|---|---:|---:|",
        ]
    )
    fields = sorted(
        set(primary["boundary_field_exact_counts"])
        | set(robust["boundary_field_exact_counts"])
    )
    for field in fields:
        lines.append(
            f"| `{field}` | {primary['boundary_field_exact_counts'].get(field, 0)}/20 | "
            f"{robust['boundary_field_exact_counts'].get(field, 0)}/31 |"
        )
    lines.extend(
        [
            "",
            "## Primary pair details",
            "",
            "| Stem | Core | Intent | Ops | Boundary | Buffer profile | Extra CUDA scalars | C | Difference categories |",
            "|---|---|---|---|---|---|---:|---|---|",
        ]
    )
    for row in rows:
        if not row["in_primary"]:
            continue
        lines.append(
            f"| `{row['stem']}` | {_yes(row['semantic_core_exact'])} | "
            f"{_yes(row['intent']['exact'])} | "
            f"{_yes(row['intent']['op_kind_set_equal'])} | "
            f"{_yes(row['boundary']['exact'])} | "
            f"{_yes(row['signature']['buffer_profile_exact'])} | "
            f"{row['signature']['extra_cuda_scalar_count']} | "
            f"{_yes(row['c']['exact'])} | "
            f"{', '.join(row['difference_categories']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "All rows dereference Task28's latest accepted `selected.run_dir` and verify the run-local "
            "summary, J-only visibility receipt, Gate1 artifact digest, and passing Gate2 certificate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest = _read(MANIFEST)
    source_manifest = _read(SOURCE_COHORTS)
    source_results = _read(SOURCE_RESULTS)
    cohorts = source_manifest["cohorts"]
    selected, coverage = _primary_selection(manifest, cohorts)
    if selected != manifest["primary_stems"]:
        raise ValueError("frozen primary_stems do not match the declared selection rule")
    if coverage != manifest["source_cohort_coverage"]:
        raise ValueError("frozen source_cohort_coverage is stale")
    all_stems = sorted(source_results["pairs"])
    if len(all_stems) != manifest["robustness_unique_pair_count"]:
        raise ValueError("Task28 robustness pair count changed")
    primary_set = set(selected)
    rows = [
        _pair_metrics(
            stem,
            source_results["pairs"][stem],
            in_primary=stem in primary_set,
        )
        for stem in all_stems
    ]
    primary_rows = [row for row in rows if row["in_primary"]]
    primary = _aggregate(primary_rows)
    robustness = _aggregate(rows)
    if not primary["all_exact_accepted_provenance"]:
        raise ValueError("primary cohort contains non-accepted provenance")
    if not robustness["all_exact_accepted_provenance"]:
        raise ValueError("robustness cohort contains non-accepted provenance")
    result = {
        "protocol": "intentir.experiment.cross-frontend-j-analysis.v1",
        "source_task": SOURCE_TASK.name,
        "selection_freeze_commit": SELECTION_FREEZE_COMMIT,
        "source_results_sha256": hashlib.sha256(SOURCE_RESULTS.read_bytes()).hexdigest(),
        "cohort_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "analysis_agent_calls": 0,
        "selection": {
            "primary_stems": selected,
            "source_cohort_coverage": coverage,
            "selection_uses_j_or_outcomes": False,
        },
        "primary": primary,
        "robustness": robustness,
        "pairs": rows,
    }
    TASK_ROOT.joinpath("results_current.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    TASK_ROOT.joinpath("results_current.md").write_text(
        _render_markdown(result), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "primary_pairs": result["primary"]["pair_count"],
                "robustness_pairs": result["robustness"]["pair_count"],
                "primary_semantic_core_exact": result["primary"]["exact_counts"][
                    "semantic_core_exact"
                ],
                "robustness_semantic_core_exact": result["robustness"][
                    "exact_counts"
                ]["semantic_core_exact"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
