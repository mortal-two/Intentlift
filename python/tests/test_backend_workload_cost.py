from __future__ import annotations

import json
from pathlib import Path

from intentir.backend.cuda.production_cases import (
    constexpr_input_bindings,
    prepare_production_workload,
)
from intentir.backend.materialize import materialize_j
from intentir.backend.workload_cost import derive_j_workload_cost
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.scrc_workload_run import build_workload_jc_suite


_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS = _ROOT / "corpus" / "production-jc-v1"


def _cost(name: str):
    snapshot, _path = load_production_snapshot(_SNAPSHOTS, name)
    case = build_workload_jc_suite(
        prepare_production_workload(name), snapshot.j, snapshot.c
    ).cases[0]
    materialized = materialize_j(
        snapshot.j,
        case.refeval.dims_map,
        constexpr_bindings=constexpr_input_bindings(
            snapshot.j,
            case.refeval.input_binding_map,
            case.values,
        ),
    )
    return derive_j_workload_cost(materialized)


def test_softmax_cost_separates_flops_comparisons_and_special_functions() -> None:
    cost = _cost("softmax")

    assert cost.boundary_input_bytes == 1016
    assert cost.boundary_output_bytes == 1016
    assert cost.floating_point_ops == 760
    assert cost.comparison_ops == 252
    assert cost.special_function_ops == 254
    assert cost.complete
    assert cost.arithmetic_intensity_flop_per_byte == 760 / 2032


def test_closed_18_cost_model_never_silently_drops_an_unknown_op() -> None:
    catalog = json.loads((_SNAPSHOTS / "catalog.json").read_text(encoding="utf-8"))
    costs = {entry["name"]: _cost(entry["name"]) for entry in catalog["entries"]}

    assert len(costs) == 18
    assert all(not item.unsupported_semantic_ops for item in costs.values())
    assert {
        name for name, item in costs.items() if item.dynamic_branches
    } == {"flash_layer_norm_1pass"}
    assert costs["masked_copy"].floating_point_ops == 0
    assert costs["masked_copy"].boundary_io_bytes == 8000
