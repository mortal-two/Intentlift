#!/usr/bin/env python3
"""Create one auditable backend-development J variant from an exact recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.dev_j_variant import (
    JDevEdit,
    create_j_dev_variant,
)
from intentir.ir.production_snapshot import load_production_snapshot


RECIPE_PROTOCOL = "intentir.backend.dev-j-recipe.v1"


def _load_recipe(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read development J recipe {path}: {exc}") from exc
    expected = {
        "protocol",
        "base_snapshot_digest",
        "source_kernel_id",
        "shape_config_id",
        "dtype_config_id",
        "base_certification_status",
        "edits",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(
            f"development J recipe must contain exactly {sorted(expected)}"
        )
    if value["protocol"] != RECIPE_PROTOCOL:
        raise ValueError("unsupported development J recipe protocol")
    if not isinstance(value["edits"], list) or not value["edits"]:
        raise ValueError("development J recipe edits must be a non-empty list")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base, _path = load_production_snapshot(args.snapshot_dir, args.name)
    recipe = _load_recipe(args.recipe)
    if recipe["base_snapshot_digest"] != base.snapshot_digest:
        raise ValueError(
            "development J recipe base snapshot changed: expected "
            f"{recipe['base_snapshot_digest']}, observed {base.snapshot_digest}"
        )
    variant = create_j_dev_variant(
        base,
        source_kernel_id=recipe["source_kernel_id"],
        shape_config_id=recipe["shape_config_id"],
        dtype_config_id=recipe["dtype_config_id"],
        base_certification_status=recipe["base_certification_status"],
        edits=tuple(JDevEdit.from_dict(item) for item in recipe["edits"]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(variant.to_json(indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "base_j_digest": variant.base_j_digest,
                "variant_j_digest": variant.variant_j_digest,
                "receipt_digest": variant.receipt_digest,
                "certification_status": variant.to_dict()["certification_status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
