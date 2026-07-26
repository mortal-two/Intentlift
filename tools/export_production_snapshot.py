#!/usr/bin/env python3
"""Export one certified SCRC evidence file as a strict J+C snapshot."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from intentir.verify.production_export import export_production_snapshot


def _write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    snapshot, receipt = export_production_snapshot(args.evidence)
    _write_atomic(args.output, snapshot.to_json(indent=2))
    if args.receipt is not None:
        receipt_payload = {
            **receipt.to_dict(),
            "receipt_digest": receipt.receipt_digest,
        }
        _write_atomic(
            args.receipt,
            json.dumps(
                receipt_payload,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
        )
    print(snapshot.snapshot_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
