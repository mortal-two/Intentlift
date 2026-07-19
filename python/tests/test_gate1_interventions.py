from __future__ import annotations

from intentir.verify.gate1 import (
    Gate1Artifact,
    InterventionStatus,
    build_semantic_interventions,
    compose_replay_plan,
    default_triton_rule_registry,
    evaluate_semantic_intervention,
    fingerprint_ttir,
)


def _artifact() -> Gate1Artifact:
    return Gate1Artifact.from_json(
        {
            "j": {
                "kernel_id": "add_kernel",
                "intent": {
                    "intent_function": {
                        "name": "add",
                        "inputs": [
                            {"name": "a", "dtype": "f32", "shape": ["N"]},
                            {"name": "b", "dtype": "f32", "shape": ["N"]},
                        ],
                        "outputs": [
                            {"name": "out", "dtype": "f32", "shape": ["N"]}
                        ],
                        "body": [
                            {
                                "op": "map_binary",
                                "result": "out",
                                "operands": ["a", "b"],
                                "attrs": {"fn": "add"},
                            }
                        ],
                    }
                },
                "node_ids": {"n0": "body[0]"},
                "boundary": {
                    "domains": {"d0": "i < N"},
                    "accesses": [
                        {
                            "access_id": "a0",
                            "kind": "load",
                            "tensor": "a",
                            "predicate": "i < N",
                            "fill_value": 0,
                            "node_ref": "n0",
                            "index_expression": "i",
                        },
                        {
                            "access_id": "a1",
                            "kind": "load",
                            "tensor": "b",
                            "predicate": "i < N",
                            "fill_value": 0,
                            "node_ref": "n0",
                            "index_expression": "i",
                        },
                        {
                            "access_id": "a2",
                            "kind": "store",
                            "tensor": "out",
                            "predicate": "i < N",
                            "node_ref": "n0",
                            "index_expression": "i",
                        },
                    ],
                    "effects": [
                        {"kind": "read", "target": "a", "node_ref": "n0"},
                        {"kind": "read", "target": "b", "node_ref": "n0"},
                        {"kind": "write", "target": "out", "node_ref": "n0"},
                    ],
                    "output_mappings": [
                        {"output": "out", "value": "out", "node_ref": "n0"}
                    ],
                },
            },
            "interface": {
                "kernel_id": "add_kernel",
                "frontend": "triton",
                "function_name": "add_kernel",
                "parameters": [
                    {
                        "name": "a_ptr",
                        "kind": "pointer",
                        "mode": "input",
                        "dtype": "f32",
                        "rank": 1,
                        "tensor_ref": "a",
                    },
                    {
                        "name": "b_ptr",
                        "kind": "pointer",
                        "mode": "input",
                        "dtype": "f32",
                        "rank": 1,
                        "tensor_ref": "b",
                    },
                    {
                        "name": "out_ptr",
                        "kind": "pointer",
                        "mode": "output",
                        "dtype": "f32",
                        "rank": 1,
                        "tensor_ref": "out",
                    },
                    {
                        "name": "N",
                        "kind": "scalar",
                        "mode": "value",
                        "dtype": "i32",
                    },
                    {"name": "BLOCK", "kind": "constexpr", "mode": "value"},
                ],
                "grid": ["ceil_div(N, BLOCK)"],
                "compiler_slots": ["num_warps"],
            },
            "schedule": {
                "slots": [
                    {"name": "BLOCK", "kind": "constexpr"},
                    {"name": "num_warps", "kind": "compiler"},
                ],
                "values": {"BLOCK": 64, "num_warps": 4},
            },
            "plan": {
                "steps": [
                    {"rule": "bind_program_domain", "refs": ["d0"]},
                    {"rule": "bind_runtime_scalar", "refs": ["N"]},
                    {"rule": "declare_schedule_slot", "refs": ["BLOCK"]},
                    {"rule": "declare_schedule_slot", "refs": ["num_warps"]},
                    {"rule": "materialize_index_tile", "refs": ["a0"]},
                    {"rule": "realize_masked_access", "refs": ["a0"]},
                    {"rule": "materialize_index_tile", "refs": ["a1"]},
                    {"rule": "realize_masked_access", "refs": ["a1"]},
                    {"rule": "realize_elementwise", "node_ref": "n0"},
                    {"rule": "materialize_index_tile", "refs": ["a2"]},
                    {"rule": "realize_masked_access", "refs": ["a2"]},
                    {"rule": "realize_output_mapping", "node_ref": "n0"}
                ]
            },
            "location_map": {"source_name": "add.py", "entries": []},
            "source": "SECRET_SOURCE",
        }
    )


BASELINE_TTIR = r"""
module {
  tt.func public @kernel(%a: !tt.ptr<f32>, %b: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %av = tt.load %a : !tt.ptr<f32>
    %bv = tt.load %b : !tt.ptr<f32>
    %value = arith.addf %av, %bv : f32
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
"""

SUB_TTIR = BASELINE_TTIR.replace("arith.addf", "arith.subf")
MUL_TTIR = BASELINE_TTIR.replace("arith.addf", "arith.mulf")
SUB_WITH_EXTRA_TTIR = SUB_TTIR.replace(
    "    tt.store %out, %value : !tt.ptr<f32>",
    "    %zero = arith.constant 0.000000e+00 : f32\n"
    "    tt.store %out, %value : !tt.ptr<f32>",
)

FILL_ZERO_TTIR = r"""
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %offset = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %limit = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offset, %limit : tensor<8xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %value = tt.load %x, %mask, %zero : !tt.ptr<f32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
"""

FILL_ONE_TTIR = FILL_ZERO_TTIR.replace(
    "dense<0.000000e+00>", "dense<1.000000e+00>"
)
FILL_TWO_TTIR = FILL_ZERO_TTIR.replace(
    "dense<0.000000e+00>", "dense<2.000000e+00>"
)

SHARED_FILL_ZERO_TTIR = r"""
module {
  tt.func public @kernel(%x: !tt.ptr<f32>, %y: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %offset = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %limit = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offset, %limit : tensor<8xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %left = tt.load %x, %mask, %zero : !tt.ptr<f32>
    %right = tt.load %y, %mask, %zero : !tt.ptr<f32>
    %value = arith.addf %left, %right : tensor<8xf32>
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
"""

SHARED_FILL_ONE_TTIR = SHARED_FILL_ZERO_TTIR.replace(
    "    %left = tt.load %x, %mask, %zero : !tt.ptr<f32>",
    "    %one = arith.constant dense<1.000000e+00> : tensor<8xf32>\n"
    "    %left = tt.load %x, %mask, %one : !tt.ptr<f32>",
)

REDUCTION_SUM_TTIR = r"""
module {
  tt.func public @row_sum(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %lanes = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %limit = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %lanes, %limit : tensor<8xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lanes : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %values = tt.load %ptrs, %mask, %zero : tensor<8x!tt.ptr<f32>>
    %total = "tt.reduce"(%values) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<8xf32>) -> f32
    tt.store %out, %total : !tt.ptr<f32>
    tt.return
  }
}
"""

REDUCTION_MAX_TTIR = REDUCTION_SUM_TTIR.replace(
    "dense<0.000000e+00>", "dense<0xFF800000>"
).replace("arith.addf", "arith.maxnumf")

CONST_ONE_TTIR = r"""
module {
  tt.func public @kernel(%out: !tt.ptr<f32>) {
    %value = arith.constant 1.000000e+00 : f32
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
"""

CONST_ZERO_TTIR = CONST_ONE_TTIR.replace("1.000000e+00", "0.000000e+00")
CONST_TWO_TTIR = CONST_ONE_TTIR.replace("1.000000e+00", "2.000000e+00")


def _constant_artifact() -> Gate1Artifact:
    payload = _artifact().to_json()
    function = payload["j"]["intent"]["intent_function"]
    function["inputs"] = []
    function["body"] = [
        {
            "op": "const",
            "result": "out",
            "attrs": {"value": 1.0, "dtype": "f32", "shape": ["N"]},
        }
    ]
    payload["j"]["boundary"]["accesses"] = [
        {
            "access_id": "a0",
            "kind": "store",
            "tensor": "out",
            "predicate": "i < N",
            "node_ref": "n0",
            "index_expression": "i",
        }
    ]
    payload["j"]["boundary"]["effects"] = [
        {"kind": "write", "target": "out", "node_ref": "n0"}
    ]
    payload["j"]["boundary"]["output_mappings"] = [
        {"output": "out", "value": "out", "node_ref": "n0"}
    ]
    payload["interface"]["parameters"] = [
        parameter
        for parameter in payload["interface"]["parameters"]
        if parameter["name"] not in {"a_ptr", "b_ptr"}
    ]
    payload["plan"] = {
        "steps": [
            {"rule": "bind_program_domain", "refs": ["d0"]},
            {"rule": "bind_runtime_scalar", "refs": ["N"]},
            {"rule": "declare_schedule_slot", "refs": ["BLOCK"]},
            {"rule": "declare_schedule_slot", "refs": ["num_warps"]},
            {"rule": "materialize_index_tile", "refs": ["a0"]},
            {"rule": "realize_masked_access", "refs": ["a0"]},
            {"rule": "realize_const", "node_ref": "n0"},
            {"rule": "realize_output_mapping", "node_ref": "n0"},
        ]
    }
    return Gate1Artifact.from_json(payload)


def test_binary_intervention_changes_only_the_registered_j_coordinate() -> None:
    artifact = _artifact()
    interventions = build_semantic_interventions(artifact)

    assert len(interventions) == 1
    spec = interventions[0]
    assert spec.target_ref == "n0"
    assert (spec.before, spec.after) == ("add", "sub")
    assert spec.artifact.interface == artifact.interface
    assert spec.artifact.schedule == artifact.schedule
    assert spec.artifact.plan == compose_replay_plan(
        spec.artifact, default_triton_rule_registry()
    ).plan
    assert spec.artifact.compose_provenance is not None
    assert spec.artifact.j.boundary == artifact.j.boundary
    assert spec.artifact.j.intent.body[0].attrs["fn"] == "sub"
    assert spec.baseline_j_digest != spec.intervention_j_digest


def test_exact_registered_operator_and_dataflow_delta_passes() -> None:
    spec = build_semantic_interventions(_artifact())[0]
    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(BASELINE_TTIR),
        fingerprint_ttir(SUB_TTIR),
    )

    assert verdict.status is InterventionStatus.PASS
    assert verdict.changed_categories == ("dataflow", "op_multiset")
    assert dict(verdict.operator_delta) == {"arith.addf": -1, "arith.subf": 1}


def test_output_constant_intervention_is_unique_and_changes_only_j_value() -> None:
    artifact = _constant_artifact()
    spec = build_semantic_interventions(artifact)[0]

    assert spec.kind.value == "output_constant_replacement"
    assert spec.target_ref == "n0"
    assert (spec.before, spec.after) == ("1", "0")
    assert spec.artifact.interface == artifact.interface
    assert spec.artifact.schedule == artifact.schedule
    assert spec.artifact.plan == compose_replay_plan(
        spec.artifact, default_triton_rule_registry()
    ).plan
    assert spec.artifact.compose_provenance is not None
    assert spec.artifact.j.boundary == artifact.j.boundary
    assert spec.artifact.j.intent.body[0].attrs["value"] == 0.0


def test_output_constant_intervention_requires_exact_typed_one_to_zero_store() -> None:
    spec = build_semantic_interventions(_constant_artifact())[0]
    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(CONST_ONE_TTIR),
        fingerprint_ttir(CONST_ZERO_TTIR),
    )

    assert verdict.status is InterventionStatus.PASS
    assert verdict.changed_categories == ("dataflow", "predicate_effect")
    assert dict(verdict.operator_delta) == {}
    assert len(verdict.store_value_delta) == 2

    wrong = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(CONST_ONE_TTIR),
        fingerprint_ttir(CONST_TWO_TTIR),
    )
    assert wrong.status is InterventionStatus.WRONG_CONSTANT_DELTA


def test_unchanged_replay_is_classified_as_ignored_j() -> None:
    spec = build_semantic_interventions(_artifact())[0]
    baseline = fingerprint_ttir(BASELINE_TTIR)
    verdict = evaluate_semantic_intervention(spec, baseline, baseline)

    assert verdict.status is InterventionStatus.IGNORED_J


def test_wrong_operator_direction_is_rejected_without_a_score_threshold() -> None:
    spec = build_semantic_interventions(_artifact())[0]
    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(BASELINE_TTIR),
        fingerprint_ttir(MUL_TTIR),
    )

    assert verdict.status is InterventionStatus.WRONG_OPERATOR_DELTA


def test_correct_target_with_an_extra_operator_is_off_target_drift() -> None:
    spec = build_semantic_interventions(_artifact())[0]
    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(BASELINE_TTIR),
        fingerprint_ttir(SUB_WITH_EXTRA_TTIR),
    )

    assert verdict.status is InterventionStatus.OFF_TARGET_DRIFT


def test_masked_fill_intervention_requires_an_exact_typed_zero_to_one_delta() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"][0] = {
        "op": "cast",
        "result": "out",
        "operands": ["a"],
        "attrs": {"dtype": "f32"},
    }
    artifact = Gate1Artifact.from_json(payload)
    spec = build_semantic_interventions(artifact)[0]

    assert spec.kind.value == "masked_load_fill_replacement"
    assert spec.target_ref == "a0"
    assert spec.artifact.j.boundary.accesses[0].fill_value == 1.0
    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(FILL_ZERO_TTIR),
        fingerprint_ttir(FILL_ONE_TTIR),
    )
    assert verdict.status is InterventionStatus.PASS
    assert verdict.changed_categories == ("dataflow", "predicate_effect")
    assert dict(verdict.operator_delta) == {}
    assert len(verdict.memory_fill_delta) == 2

    wrong = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(FILL_ZERO_TTIR),
        fingerprint_ttir(FILL_TWO_TTIR),
    )
    assert wrong.status is InterventionStatus.WRONG_FILL_DELTA


def test_fill_intervention_allows_one_new_constant_when_zero_is_shared() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"][0] = {
        "op": "cast",
        "result": "out",
        "operands": ["a"],
        "attrs": {"dtype": "f32"},
    }
    spec = build_semantic_interventions(Gate1Artifact.from_json(payload))[0]
    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(SHARED_FILL_ZERO_TTIR),
        fingerprint_ttir(SHARED_FILL_ONE_TTIR),
    )

    assert verdict.status is InterventionStatus.PASS
    assert dict(verdict.operator_delta) == {"arith.constant": 1}
    assert verdict.changed_categories == (
        "dataflow",
        "op_multiset",
        "predicate_effect",
    )


def test_fill_intervention_skips_a_reduction_masked_contribution_contract() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"][0] = {
        "op": "cast",
        "result": "out",
        "operands": ["a"],
        "attrs": {"dtype": "f32"},
    }
    payload["j"]["boundary"]["reductions"] = [
        {
            "node_ref": "n0",
            "tile_axis": 0,
            "identity": 0,
            "masked_contribution": 0,
        }
    ]

    assert build_semantic_interventions(Gate1Artifact.from_json(payload)) == ()


def test_fill_intervention_skips_indirect_inputs_to_a_reduction_contract() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"] = [
        {
            "op": "map_unary",
            "result": "prepared",
            "operands": ["a"],
            "attrs": {"fn": "abs"},
        },
        {
            "op": "reduce",
            "result": "out",
            "operands": ["prepared"],
            "attrs": {"fn": "mean", "axes": [0]},
        },
    ]
    payload["j"]["node_ids"] = {"n0": "body[0]", "n1": "body[1]"}
    payload["j"]["boundary"]["accesses"][0]["node_ref"] = "n0"
    payload["j"]["boundary"]["accesses"][1].pop("predicate", None)
    payload["j"]["boundary"]["accesses"][1].pop("fill_value", None)
    payload["j"]["boundary"]["accesses"][1]["node_ref"] = "n1"
    payload["j"]["boundary"]["accesses"][2]["node_ref"] = "n1"
    payload["j"]["boundary"]["effects"] = [
        {"kind": "read", "target": "a", "node_ref": "n0"},
        {"kind": "write", "target": "out", "node_ref": "n1"},
    ]
    payload["j"]["boundary"]["output_mappings"] = [
        {"output": "out", "value": "out", "node_ref": "n1"}
    ]
    payload["j"]["boundary"]["reductions"] = [
        {
            "node_ref": "n1",
            "tile_axis": 0,
            "identity": 0,
            "masked_contribution": 0,
        }
    ]

    artifact = Gate1Artifact.from_json(payload)
    assert build_semantic_interventions(artifact, limit=10) == ()


def test_sum_to_max_intervention_updates_reduction_and_masked_identity_together() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"][0] = {
        "op": "reduce",
        "result": "out",
        "operands": ["a"],
        "attrs": {"fn": "sum", "axes": [0]},
    }
    payload["j"]["boundary"]["accesses"][1]["node_ref"] = None
    payload["j"]["boundary"]["reductions"] = [
        {
            "node_ref": "n0",
            "tile_axis": 0,
            "identity": 0,
            "masked_contribution": 0,
        }
    ]
    artifact = Gate1Artifact.from_json(payload)
    spec = build_semantic_interventions(artifact)[0]

    assert spec.kind.value == "reduction_sum_to_max"
    assert spec.artifact.j.intent.body[0].attrs["fn"] == "max"
    assert spec.artifact.j.boundary.accesses[0].fill_value == "-inf"
    assert spec.artifact.j.boundary.reductions[0].identity == "-inf"
    assert (
        spec.artifact.j.boundary.reductions[0].masked_contribution == "-inf"
    )

    verdict = evaluate_semantic_intervention(
        spec,
        fingerprint_ttir(REDUCTION_SUM_TTIR),
        fingerprint_ttir(REDUCTION_MAX_TTIR),
    )
    assert verdict.status is InterventionStatus.PASS
    assert dict(verdict.operator_delta) == {
        "arith.addf": -1,
        "arith.maxnumf": 1,
    }
    assert len(verdict.reduction_delta) == 2


def test_no_supported_binary_coordinate_is_explicitly_not_applicable() -> None:
    payload = _artifact().to_json()
    payload["j"]["intent"]["intent_function"]["body"][0] = {
        "op": "cast",
        "result": "out",
        "operands": ["a"],
        "attrs": {"dtype": "f32"},
    }
    for access in payload["j"]["boundary"]["accesses"]:
        access.pop("fill_value", None)
        access.pop("predicate", None)
    artifact = Gate1Artifact.from_json(payload)

    assert build_semantic_interventions(artifact) == ()
