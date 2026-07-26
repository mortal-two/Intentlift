#!/usr/bin/env python3
"""Append one certified SCRC evidence envelope to the frozen J+C catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intentir.verify.production_export import append_production_export


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args(argv)

    snapshot, receipt, entry = append_production_export(
        args.catalog_dir,
        name=args.name,
        case_id=args.case_id,
        evidence=args.evidence,
    )
    print(
        json.dumps(
            {
                "name": entry.name,
                "case_id": entry.case_id,
                "snapshot_digest": snapshot.snapshot_digest,
                "j_digest": snapshot.j_digest,
                "c_digest": snapshot.c_digest,
                "receipt_digest": receipt.receipt_digest,
                "evidence_kind": receipt.evidence_kind,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
