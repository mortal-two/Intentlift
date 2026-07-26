from __future__ import annotations

import json
from pathlib import Path

from intentir.ir.production_snapshot import (
    ProductionSnapshot,
    load_production_snapshot,
)


_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _ROOT / "corpus" / "production-jc-v1"


def test_frozen_production_catalog_contains_exactly_the_closed_sixteen() -> None:
    catalog = json.loads((_CATALOG_DIR / "catalog.json").read_text())

    assert catalog["protocol"] == "intentir.production.jc-catalog.v1"
    assert catalog["entry_count"] == 16
    assert {item["name"] for item in catalog["entries"]} == {
        "dyt",
        "flash_bwd_preprocess",
        "flash_layer_norm_1pass",
        "fused_add_rms_norm",
        "geglu",
        "jsd",
        "kl_div",
        "layer_norm",
        "liger_rms_norm",
        "poly_norm",
        "softmax",
        "sparsemax",
        "swiglu",
        "tvd",
        "xformers_rms_norm",
        "xformers_rms_norm_add",
    }


def test_catalog_snapshots_are_content_addressed_j_plus_optional_c_only() -> None:
    catalog = json.loads((_CATALOG_DIR / "catalog.json").read_text())
    for entry in catalog["entries"]:
        path = _CATALOG_DIR / entry["snapshot_file"]
        raw = json.loads(path.read_text())
        snapshot = ProductionSnapshot.from_dict(raw)

        assert path.name == f"{snapshot.snapshot_digest}.json"
        assert entry["snapshot_digest"] == snapshot.snapshot_digest
        assert entry["j_digest"] == snapshot.j_digest
        assert entry["c_digest"] == snapshot.c_digest
        assert set(raw) == {
            "schema_version",
            "j",
            "c",
            "j_digest",
            "c_digest",
            "snapshot_digest",
        }


def test_frozen_catalog_is_directly_consumable_by_stable_name() -> None:
    catalog = json.loads((_CATALOG_DIR / "catalog.json").read_text())

    for entry in catalog["entries"]:
        snapshot, path = load_production_snapshot(
            _CATALOG_DIR, entry["name"]
        )
        assert path.name == entry["snapshot_file"]
        assert snapshot.snapshot_digest == entry["snapshot_digest"]
