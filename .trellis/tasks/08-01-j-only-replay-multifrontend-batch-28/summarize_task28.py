"""Aggregate exact Task28 run evidence without double-counting cohort overlap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
RUN_ROOT = TASK_ROOT / "runs"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted(row: dict[str, Any]) -> bool:
    return (
        row.get("typecheck") == "pass"
        and row.get("gate1") == "replay_closed"
        and row.get("gate2_passed") is True
        and row.get("lift_agent_calls") == 0
        and row.get("pi_visible_to_replay") is False
    )


def _case_rows(case_id: str) -> list[dict[str, Any]]:
    case_dir = RUN_ROOT / case_id.replace(":", "_")
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(case_dir.glob("run-[0-9][0-9][0-9]")):
        summary = run_dir / "summary.json"
        if not summary.is_file():
            continue
        row = _read(summary)
        row["run"] = run_dir.name
        rows.append(row)
    return rows


def _selected_evidence(case_id: str) -> dict[str, Any]:
    rows = _case_rows(case_id)
    accepted = [row for row in rows if _accepted(row)]
    selected = accepted[-1] if accepted else (rows[-1] if rows else None)
    return {
        "case": case_id,
        "attempt_count": len(rows),
        "accepted_attempt_count": len(accepted),
        "accepted": bool(accepted),
        "selected": selected,
        "attempts": [
            {
                "run": row["run"],
                "typecheck": row.get("typecheck"),
                "gate1": row.get("gate1"),
                "gate2": row.get("gate2"),
                "accepted": _accepted(row),
                "total_elapsed_ms": row.get("total_elapsed_ms"),
            }
            for row in rows
        ],
    }


def main() -> int:
    manifest = _read(TASK_ROOT / "cohorts.json")
    stems = sorted({stem for members in manifest["cohorts"].values() for stem in members})
    pairs: dict[str, Any] = {}
    accepted_frontend_runs = 0
    accepted_pairs = 0
    for stem in stems:
        tilelang = _selected_evidence(f"tilelang:{stem}")
        cuda = _selected_evidence(f"cuda:tilelang_{stem}")
        pair_accepted = tilelang["accepted"] and cuda["accepted"]
        accepted_frontend_runs += int(tilelang["accepted"]) + int(cuda["accepted"])
        accepted_pairs += int(pair_accepted)
        pairs[stem] = {
            "accepted": pair_accepted,
            "tilelang": tilelang,
            "cuda": cuda,
            "cohort_memberships": [
                name for name, members in manifest["cohorts"].items() if stem in members
            ],
        }

    cohorts: dict[str, Any] = {}
    for name, members in manifest["cohorts"].items():
        accepted = [stem for stem in members if pairs[stem]["accepted"]]
        cohorts[name] = {
            "members": members,
            "member_count": len(members),
            "accepted": accepted,
            "accepted_count": len(accepted),
            "complete": len(accepted) == len(members),
        }

    result = {
        "protocol": "intentir.experiment.task28-results.v1",
        "counting": manifest["counting"],
        "unique_pair_count": len(stems),
        "accepted_unique_pair_count": accepted_pairs,
        "accepted_unique_frontend_run_count": accepted_frontend_runs,
        "all_lift_agent_calls_zero": all(
            evidence[frontend]["selected"] is None
            or evidence[frontend]["selected"].get("lift_agent_calls") == 0
            for evidence in pairs.values()
            for frontend in ("tilelang", "cuda")
        ),
        "pairs": pairs,
        "cohorts": cohorts,
    }
    (TASK_ROOT / "results_current.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Task28 current results",
        "",
        f"- accepted unique pairs: {accepted_pairs}/{len(stems)}",
        f"- accepted unique frontend runs: {accepted_frontend_runs}/{len(stems) * 2}",
        "- counting: 70 cohort memberships / 31 unique pairs / 62 unique frontend runs",
        "",
        "| Cohort | Accepted | Complete |",
        "|---|---:|---|",
    ]
    for name, row in cohorts.items():
        lines.append(
            f"| `{name}` | {row['accepted_count']}/{row['member_count']} | "
            f"{'yes' if row['complete'] else 'no'} |"
        )
    lines.extend(["", "| Stem | TileLang | CUDA | Pair |", "|---|---|---|---|"])
    for stem, row in pairs.items():
        lines.append(
            f"| `{stem}` | {'PASS' if row['tilelang']['accepted'] else '—'} | "
            f"{'PASS' if row['cuda']['accepted'] else '—'} | "
            f"{'PASS' if row['accepted'] else '—'} |"
        )
    (TASK_ROOT / "results_current.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "accepted_unique_pair_count": accepted_pairs,
        "accepted_unique_frontend_run_count": accepted_frontend_runs,
        "g1": cohorts["g1_pointwise_selection"],
    }, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
