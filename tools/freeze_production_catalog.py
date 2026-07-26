#!/usr/bin/env python3
"""Freeze acceptance-exported J+C snapshots into a content-addressed catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intentir.backend.cuda.production_cases import PRODUCTION_CASE_IDS
from intentir.ir.production_snapshot import (
    PRODUCTION_CATALOG_SCHEMA,
    ProductionSnapshot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, case_id in sorted(PRODUCTION_CASE_IDS.items()):
        snapshot_path = args.snapshot_dir / f"{name}.json"
        receipt_path = args.snapshot_dir / f"{name}.receipt.json"
        snapshot = ProductionSnapshot.from_json(
            snapshot_path.read_text(encoding="utf-8")
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        for field, expected in (
            ("snapshot_digest", snapshot.snapshot_digest),
            ("j_digest", snapshot.j_digest),
            ("c_digest", snapshot.c_digest),
        ):
            if receipt.get(field) != expected:
                raise ValueError(f"{name} receipt {field} mismatch")
        filename = f"{snapshot.snapshot_digest}.json"
        destination = args.out_dir / filename
        destination.write_text(snapshot.to_json(indent=2), encoding="utf-8")
        entries.append(
            {
                "name": name,
                "case_id": case_id,
                "snapshot_file": filename,
                "snapshot_digest": snapshot.snapshot_digest,
                "j_digest": snapshot.j_digest,
                "c_digest": snapshot.c_digest,
                "receipt_digest": receipt["receipt_digest"],
                "evidence_kind": receipt["evidence_kind"],
            }
        )
    catalog = {
        "protocol": PRODUCTION_CATALOG_SCHEMA,
        "entry_count": len(entries),
        "entries": entries,
    }
    (args.out_dir / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"froze {len(entries)} snapshots in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
