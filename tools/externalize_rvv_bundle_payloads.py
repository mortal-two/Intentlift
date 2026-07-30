#!/usr/bin/env python3
"""Move large RVV input/reference arrays from JSON into adjacent raw files."""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any, Mapping


BUNDLE_PROTOCOL = "intentir.production.rvv-search-bundle.v4"


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "value"


def _externalize_values(
    values: Mapping[str, Any],
    *,
    output_root: Path,
    relative_root: Path,
    group: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, (name, raw_record) in enumerate(values.items()):
        record = dict(raw_record)
        if "data_file" in record:
            if "data_base64" in record:
                raise ValueError("RVV payload mixes embedded and file transports")
            result[name] = record
            continue
        payload = base64.b64decode(record.pop("data_base64"), validate=True)
        relative = (
            relative_root
            / group
            / f"{index:03d}-{_safe_component(str(name))}.bin"
        )
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        record["data_file"] = relative.as_posix()
        result[name] = record
    return result


def externalize(
    bundle: Mapping[str, Any], *, output_root: Path, relative_root: Path
) -> dict[str, Any]:
    if bundle.get("protocol") != BUNDLE_PROTOCOL:
        raise ValueError("unsupported RVV benchmark bundle")
    result = dict(bundle)
    entries = []
    for index, raw_entry in enumerate(bundle["kernels"]):
        entry = dict(raw_entry)
        kernel_root = relative_root / (
            f"{index:02d}-{_safe_component(str(entry['kernel']))}"
        )
        entry["values"] = _externalize_values(
            entry["values"],
            output_root=output_root,
            relative_root=kernel_root,
            group="inputs",
        )
        entry["reference_outputs"] = _externalize_values(
            entry["reference_outputs"],
            output_root=output_root,
            relative_root=kernel_root,
            group="reference",
        )
        entries.append(entry)
    result["kernels"] = entries
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    relative_root = Path(f"{args.out.stem}.payload")
    result = externalize(
        bundle,
        output_root=args.out.resolve().parent,
        relative_root=relative_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(
        f"wrote {len(result['kernels'])} RVV workloads to {args.out} "
        f"with raw arrays below {args.out.parent / relative_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
