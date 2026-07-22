#!/usr/bin/env python3
"""List, validate, or dry-run the active 28-kernel SCRC workload catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from intentir.verify.scrc_batch import (
    SequentialSCRCBatchRunner,
    dry_run_executor,
)
from intentir.verify.workload_manifest import build_workload_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true")
    action.add_argument("--validate-manifest", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", default=None)
    parser.add_argument("--tier", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def _emit(value: dict, *, output: Path | None, json_mode: bool) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    if json_mode or output is None:
        print(encoded, end="")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_workload_manifest()
    if args.validate_manifest and (args.case is not None or args.tier is not None):
        print(
            "--validate-manifest always validates the complete 28-item projection; "
            "--case/--tier are not accepted",
            file=sys.stderr,
        )
        return 2
    try:
        entries = manifest.select(case_id=args.case, tier=args.tier)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.list:
        value = {
            "protocol": manifest.to_json()["protocol"],
            "total": len(entries),
            "entries": [entry.to_json() for entry in entries],
        }
        if args.json or args.output is not None:
            _emit(value, output=args.output, json_mode=args.json)
        else:
            for entry in entries:
                print(
                    f"{entry.ordinal:02d}\t{entry.case_id}\t"
                    f"{entry.representative_tier}\t{entry.source_module}:"
                    f"{entry.symbol}"
                )
        return 0

    if args.validate_manifest:
        validation = manifest.validate(dataset_root=args.dataset_root)
        value = validation.to_json()
        _emit(value, output=args.output, json_mode=args.json)
        return 0 if validation.passed else 2

    report = SequentialSCRCBatchRunner().run(
        entries,
        dry_run_executor(dataset_root=args.dataset_root),
        mode="dry_run",
    )
    value = report.to_json()
    _emit(value, output=args.output, json_mode=args.json)
    return 0 if set(report.counts) <= {"ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
