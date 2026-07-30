from __future__ import annotations

import json
from pathlib import Path

import pytest

from intentir.ir.production_snapshot import (
    ProductionSnapshot,
    load_production_snapshot,
)


_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _ROOT / "corpus" / "production-jc-v1"
_FRESH14_CATALOG_DIR = _ROOT / "corpus" / "production-jc-v2"
_STAGE2_CATALOG_DIR = _ROOT / "corpus" / "production-jc-v3"


def test_frozen_production_catalog_contains_exactly_the_closed_eighteen() -> None:
    catalog = json.loads((_CATALOG_DIR / "catalog.json").read_text())

    assert catalog["protocol"] == "intentir.production.jc-catalog.v1"
    assert catalog["entry_count"] == 18
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
        "masked_copy",
        "masked_row_square_sum",
        "poly_norm",
        "softmax",
        "sparsemax",
        "swiglu",
        "tvd",
        "xformers_rms_norm",
        "xformers_rms_norm_add",
    }


def test_fresh14_catalog_contains_exactly_the_new_backend_intake() -> None:
    catalog = json.loads((_FRESH14_CATALOG_DIR / "catalog.json").read_text())

    assert catalog["protocol"] == "intentir.production.jc-catalog.v1"
    assert catalog["entry_count"] == 14
    assert {item["name"] for item in catalog["entries"]} == {
        "dyt",
        "flash_bwd_preprocess",
        "flash_layer_norm_1pass",
        "fused_add_rms_norm",
        "geglu",
        "kl_div",
        "layer_norm",
        "liger_rms_norm",
        "poly_norm",
        "softmax",
        "sparsemax",
        "splitk_reduce_varargs",
        "swiglu",
        "xformers_rms_norm_add",
    }


def test_stage2_catalog_extends_fresh14_with_exactly_eight_new_j() -> None:
    catalog = json.loads((_STAGE2_CATALOG_DIR / "catalog.json").read_text())
    fresh14 = json.loads((_FRESH14_CATALOG_DIR / "catalog.json").read_text())

    assert catalog["protocol"] == "intentir.production.jc-catalog.v1"
    assert catalog["entry_count"] == 22
    assert {item["name"] for item in catalog["entries"]} - {
        item["name"] for item in fresh14["entries"]
    } == {
        "cross_entropy",
        "flash_linear",
        "group_norm",
        "jsd",
        "liger_rope",
        "qwen2vl_mrope",
        "tvd",
        "xformers_rms_norm",
    }


@pytest.mark.parametrize(
    "catalog_dir", (_CATALOG_DIR, _FRESH14_CATALOG_DIR, _STAGE2_CATALOG_DIR)
)
def test_catalog_snapshots_are_content_addressed_j_plus_optional_c_only(
    catalog_dir: Path,
) -> None:
    catalog = json.loads((catalog_dir / "catalog.json").read_text())
    for entry in catalog["entries"]:
        path = catalog_dir / entry["snapshot_file"]
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


@pytest.mark.parametrize(
    "catalog_dir", (_CATALOG_DIR, _FRESH14_CATALOG_DIR, _STAGE2_CATALOG_DIR)
)
def test_frozen_catalog_is_directly_consumable_by_stable_name(
    catalog_dir: Path,
) -> None:
    catalog = json.loads((catalog_dir / "catalog.json").read_text())

    for entry in catalog["entries"]:
        snapshot, path = load_production_snapshot(
            catalog_dir, entry["name"]
        )
        assert path.name == entry["snapshot_file"]
        assert snapshot.snapshot_digest == entry["snapshot_digest"]
