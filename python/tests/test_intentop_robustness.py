"""Regression tests for the post-reset IntentOp ownership boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from intentir.backend.cpu.abi import CPUBackendUnsupportedError
from intentir.backend.cpu.pipeline import emit_cpu_module, execute_on_cpu_outputs
from intentir.backend.cuda.compile import INTENT_OPT
from intentir.backend.cuda.program import analyze_j_for_cuda
from intentir.extraction.lifters.scrc_schema import scrc_lift_core_json_schema
from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    Boundary,
    BoundaryAccess,
    EffectSpec,
    SchemaError,
    SemanticArchive,
)
from intentir.verify.gate1.pi import RealizationWitness


_ROOT = Path(__file__).resolve().parents[2]
_INVENTORY = _ROOT / "test" / "intentop_real_sources.json"


def _function_properties() -> dict:
    schema = scrc_lift_core_json_schema()
    return (
        schema["properties"]["j"]["properties"]["intent"]["properties"]
        ["intent_function"]["properties"]
    )


def _op_variants() -> tuple[dict, dict]:
    non_loop, loop = scrc_lift_core_json_schema()["$defs"]["intent_op"]["anyOf"]
    return non_loop, loop


@pytest.mark.parametrize(
    "field,value",
    [
        ("layer_b", {"index_in_bounds": True}),
        ("out", "Y"),
    ],
)
def test_direct_intent_model_rejects_legacy_per_op_channels(field, value) -> None:
    op = {
        "op": "cast",
        "result": "Y",
        "operands": ["X"],
        "attrs": {"to_dtype": "f32"},
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        IntentFunction.from_json(
            {
                "name": "closed_op",
                "inputs": [{"name": "X", "dtype": "f32", "shape": [4]}],
                "outputs": [{"name": "Y", "dtype": "f32", "shape": [4]}],
                "body": [op],
            }
        )


def test_lift_schema_has_one_b_channel_and_no_schedule_in_j() -> None:
    non_loop, loop = _op_variants()
    for variant in (non_loop, loop):
        assert "layer_b" not in variant["properties"]
        assert "out" not in variant["properties"]
    assert "returns" in _function_properties()
    trip = loop["properties"]["trip"]
    assert set(trip["properties"]) == {"axis"}
    assert trip["additionalProperties"] is False


@pytest.mark.parametrize("kind", ["atomic", "synchronize"])
def test_atomic_and_synchronize_spellings_are_not_j_access_or_effect_kinds(kind) -> None:
    with pytest.raises(SchemaError, match="must be one of"):
        BoundaryAccess.from_json(
            {
                "access_id": "a0",
                "kind": kind,
                "tensor": "t0",
                "index_expression": "0",
            }
        )
    with pytest.raises(SchemaError, match="must be one of"):
        EffectSpec.from_json({"kind": kind, "target": "t0"})


def test_j_uses_logical_update_spelling_without_fixing_a_realization() -> None:
    access = BoundaryAccess.from_json(
        {
            "access_id": "a0",
            "kind": "update",
            "tensor": "t0",
            "index_expression": "v0",
        }
    )
    effect = EffectSpec.from_json(
        {
            "effect_id": "e0",
            "kind": "update",
            "target": "t0",
            "access_ref": "a0",
        }
    )
    assert access.kind.value == "update"
    assert effect.kind.value == "update"


def test_atomic_async_and_barrier_remain_pi_realization_mechanisms() -> None:
    pi = RealizationWitness.from_json(
        {
            "regions": [{"id": "r0", "parent": None, "placement": "root"}],
            "instances": [
                {
                    "id": "i0",
                    "semantic_ref": "n0",
                    "region": "r0",
                    "mechanism": "transfer_async",
                },
                {
                    "id": "i1",
                    "semantic_ref": "n1",
                    "region": "r0",
                    "mechanism": "update_atomic",
                },
                {
                    "id": "i2",
                    "semantic_ref": "n2",
                    "region": "r0",
                    "mechanism": "synchronize",
                },
            ],
            "routes": [{"source": "i0", "target": "i1", "slots": [0]}],
            "order": [
                {"before": "i0", "after": "i1"},
                {"before": "i1", "after": "i2"},
            ],
            "sharing": [],
        }
    )
    assert [item["mechanism"] for item in pi.instances] == [
        "transfer_async",
        "update_atomic",
        "synchronize",
    ]


@pytest.mark.skipif(not Path(INTENT_OPT).exists(), reason="intent-opt not built")
def test_j_only_cpu_executes_copy_fusion_and_dynamic_gather() -> None:
    copy_fn = IntentFunction.from_json(
        {
            "name": "copy",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [4]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [4]}],
            "returns": [{"output": "Y", "value": "X"}],
            "body": [],
        }
    )
    x = np.arange(4, dtype=np.float32)
    copy_result = execute_on_cpu_outputs(copy_fn, {}, [x]).outputs["Y"]
    np.testing.assert_array_equal(copy_result, x)

    fused_fn = IntentFunction.from_json(
        {
            "name": "fused_square_reduce",
            "inputs": [{"name": "X", "dtype": "f32", "shape": [2, 4]}],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [2]}],
            "body": [
                {
                    "op": "map_binary",
                    "result": "square",
                    "operands": ["X", "X"],
                    "attrs": {"fn": "mul"},
                },
                {
                    "op": "reduce",
                    "result": "Y",
                    "operands": ["square"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [1],
                        "assoc": "float_nondeterministic",
                        "keepdims": False,
                    },
                },
            ],
        }
    )
    matrix = np.arange(8, dtype=np.float32).reshape(2, 4)
    fused_result = execute_on_cpu_outputs(fused_fn, {}, [matrix]).outputs["Y"]
    np.testing.assert_array_equal(fused_result, np.sum(matrix * matrix, axis=1))

    gather_fn = IntentFunction.from_json(
        {
            "name": "dynamic_gather",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": [4]},
                {"name": "I", "dtype": "i32", "shape": [3]},
            ],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [3]}],
            "body": [
                {
                    "op": "gather",
                    "result": "Y",
                    "operands": ["X", "I"],
                    "attrs": {"axis": 0, "mode": "take"},
                }
            ],
        }
    )
    indices = np.asarray([3, 0, 2], dtype=np.int32)
    gather_result = execute_on_cpu_outputs(
        gather_fn, {}, [x, indices]
    ).outputs["Y"]
    np.testing.assert_array_equal(gather_result, x[indices])


@pytest.mark.parametrize("op", ["flip", "scan", "scatter"])
def test_cpu_e_j_fails_closed_for_unlowered_or_relation_valued_constructs(op) -> None:
    if op == "scatter":
        inputs = [
            {"name": "X", "dtype": "f32", "shape": [4]},
            {"name": "I", "dtype": "i32", "shape": [2]},
            {"name": "V", "dtype": "f32", "shape": [2]},
        ]
        operation = {
            "op": "scatter",
            "result": "Y",
            "operands": ["X", "I", "V"],
            "attrs": {"axis": 0, "mode": "along_axis", "reduce": "add"},
        }
    elif op == "scan":
        inputs = [{"name": "X", "dtype": "f32", "shape": [4]}]
        operation = {
            "op": "scan",
            "result": "Y",
            "operands": ["X"],
            "attrs": {"axis": 0, "fn": "sum"},
        }
    else:
        inputs = [{"name": "X", "dtype": "f32", "shape": [4]}]
        operation = {
            "op": "flip",
            "result": "Y",
            "operands": ["X"],
            "attrs": {"axes": [0]},
        }
    fn = IntentFunction.from_json(
        {
            "name": f"unsupported_{op}",
            "inputs": inputs,
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [4]}],
            "body": [operation],
        }
    )
    with pytest.raises(CPUBackendUnsupportedError, match="complete lowering|logical update"):
        emit_cpu_module(fn, {})


def test_canonical_j_rejects_scatter_without_update_relation() -> None:
    fn = IntentFunction.from_json(
        {
            "name": "update_without_relation",
            "inputs": [
                {"name": "X", "dtype": "f32", "shape": [4]},
                {"name": "I", "dtype": "i32", "shape": [2]},
                {"name": "V", "dtype": "f32", "shape": [2]},
            ],
            "outputs": [{"name": "Y", "dtype": "f32", "shape": [4]}],
            "body": [
                {
                    "id": "n0",
                    "op": "scatter",
                    "result": "Y",
                    "operands": ["X", "I", "V"],
                    "attrs": {"axis": 0, "mode": "along_axis", "reduce": "add"},
                }
            ],
        }
    )
    with pytest.raises(SchemaError, match="requires a typed LogicalUpdateSpec"):
        SemanticArchive(
            "update_without_relation",
            fn,
            {"n0": "body[0]"},
            Boundary(),
        )


def test_real_source_inventory_is_unique_and_digest_pinned() -> None:
    inventory = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    assert inventory["schema"] == "intentir.intentop-source-inventory.v2"
    cases = inventory["cases"]
    assert len(cases) == 9
    assert len({item["case_id"] for item in cases}) == len(cases)
    assert len({item["path"] for item in cases}) == len(cases)
    source_root = Path(inventory["source_root_hint"])
    for item in cases:
        assert len(item["sha256"]) == 64
        assert item["mechanisms"]
        source = source_root / item["path"]
        if source.is_file():
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]
