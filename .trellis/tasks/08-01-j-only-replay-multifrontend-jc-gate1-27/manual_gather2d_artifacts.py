"""Source-audited manual J/C/pi authoring for the gather2d pair.

The primary agent authored these two artifacts from the current Dataset Source,
Harness binding, and compiler observation.  This module is experiment evidence,
not a Lift agent or a production lowering rule.  Fresh Replay never imports it
and receives only J + MaskC(C) + public frontend context; pi remains Gate-only.
"""

from __future__ import annotations

import copy
from typing import Any


ARTIFACT_SCHEMA = "intentir.scrc.artifact.v13"
PI_SCHEMA = "intentir.pi.normalization-witness.v1"


def _compact_pi() -> dict[str, Any]:
    return {
        "schema": PI_SCHEMA,
        "program_axes": [{"domain_ref": "d0", "axis": 0}],
    }


def _tilelang_j() -> dict[str, Any]:
    """J for the unguarded TileLang TIR gather under typed index bounds."""

    return {
        "kernel_id": "scrc_k0",
        "intent": {
            "intent_function": {
                "name": "scrc_j0",
                "inputs": [
                    {"name": "t0", "dtype": "f32", "shape": ["x0", 64]},
                    {"name": "t1", "dtype": "i32", "shape": [256]},
                    {"name": "t2", "dtype": "i32", "shape": [256]},
                ],
                "outputs": [
                    {"name": "t3", "dtype": "f32", "shape": [256]},
                ],
                "body": [
                    {
                        "op": "gather",
                        "operands": ["t0", "t1"],
                        "result": "t4",
                        "type": {"dtype": "f32", "shape": [256, 64]},
                        "attrs": {"axis": 0, "mode": "take"},
                    },
                    {
                        "op": "gather",
                        "operands": ["t4", "t2"],
                        "result": "t3",
                        "type": {"dtype": "f32", "shape": [256]},
                        "attrs": {"axis": 1, "mode": "along_axis"},
                    },
                ],
                "returns": [{"output": "t3", "value": "t3"}],
            }
        },
        "node_ids": {"n0": "body[0]", "n1": "body[1]"},
        "boundary": {
            "domains": {
                "d0": "(0 <= k0) and (k0 < 256)",
                "d1": "(0 <= k1) and (k1 < x0)",
                "d2": "(0 <= k2) and (k2 < 64)",
            },
            "accesses": [
                {
                    "access_id": "a0",
                    "kind": "load",
                    "tensor": "t1",
                    "index_expression": "[k0]",
                    "node_ref": "n0",
                },
                {
                    "access_id": "a1",
                    "kind": "load",
                    "tensor": "t0",
                    "index_expression": "[k1, k2]",
                    "node_ref": "n0",
                },
                {
                    "access_id": "a2",
                    "kind": "load",
                    "tensor": "t2",
                    "index_expression": "[k0]",
                    "node_ref": "n1",
                },
                {
                    "access_id": "a3",
                    "kind": "store",
                    "tensor": "t3",
                    "index_expression": "[k0]",
                    "node_ref": "n1",
                },
            ],
            "effects": [
                {
                    "effect_id": "e0",
                    "kind": "read",
                    "target": "t1",
                    "node_ref": "n0",
                    "access_ref": "a0",
                },
                {
                    "effect_id": "e1",
                    "kind": "read",
                    "target": "t0",
                    "node_ref": "n0",
                    "access_ref": "a1",
                },
                {
                    "effect_id": "e2",
                    "kind": "read",
                    "target": "t2",
                    "node_ref": "n1",
                    "access_ref": "a2",
                },
                {
                    "effect_id": "e3",
                    "kind": "write",
                    "target": "t3",
                    "node_ref": "n1",
                    "access_ref": "a3",
                },
            ],
            "aliases": [],
            "broadcasts": [],
            "reductions": [],
            "updates": [],
            "output_mappings": [],
            "index_bounds": [
                {"node_ref": "n0", "data": "t0", "indices": "t1", "axis": 0},
                {"node_ref": "n1", "data": "t4", "indices": "t2", "axis": 1},
            ],
            "consumer_extensions": [],
            "access_guard_relations": [],
            "result_relations": [{"output": "t3", "kind": "exact"}],
            "realization_authorizations": [],
        },
        "signature": {
            "parameters": [
                {
                    "name": "p0",
                    "kind": "tensor",
                    "dtype": "f32",
                    "rank": 2,
                    "mode": "input",
                    "logical_ref": "t0",
                },
                {
                    "name": "p1",
                    "kind": "tensor",
                    "dtype": "i32",
                    "rank": 1,
                    "mode": "input",
                    "logical_ref": "t1",
                },
                {
                    "name": "p2",
                    "kind": "tensor",
                    "dtype": "i32",
                    "rank": 1,
                    "mode": "input",
                    "logical_ref": "t2",
                },
                {
                    "name": "p3",
                    "kind": "tensor",
                    "dtype": "f32",
                    "rank": 1,
                    "mode": "output",
                    "logical_ref": "t3",
                },
            ]
        },
        "scopes": [
            {
                "scope_id": "q0",
                "kind": "function",
                "parent_scope": None,
                "captures": [],
                "definitions": ["t0", "t1", "t2", "t4", "t3"],
                "effects": ["e0", "e1", "e2", "e3"],
            }
        ],
        "storage": [],
        "dimension_constraints": ["x0 >= 1"],
        "value_dimension_relations": [],
        "dimension_relations": [],
    }


def _cuda_j() -> dict[str, Any]:
    """J for generated CUDA's explicit bounds-to-zero gather semantics."""

    body: list[dict[str, Any]] = [
        {
            "op": "const",
            "result": "v0",
            "type": {"dtype": "i32", "shape": []},
            "attrs": {"dtype": "i32", "shape": [], "value": 0},
        },
        {
            "op": "map_binary",
            "operands": ["t0", "v0"],
            "result": "v1",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "ge"},
        },
        {
            "op": "const",
            "result": "v2",
            "type": {"dtype": "i32", "shape": []},
            "attrs": {"dtype": "i32", "shape": [], "value": 64},
        },
        {
            "op": "map_binary",
            "operands": ["t0", "v2"],
            "result": "v3",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "lt"},
        },
        {
            "op": "map_binary",
            "operands": ["v1", "v3"],
            "result": "v4",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "logical_and"},
        },
        {
            "op": "map_binary",
            "operands": ["t2", "v0"],
            "result": "v5",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "ge"},
        },
        {
            "op": "map_binary",
            "operands": ["v4", "v5"],
            "result": "v6",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "logical_and"},
        },
        {
            "op": "map_binary",
            "operands": ["t2", "t3"],
            "result": "v7",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "lt"},
        },
        {
            "op": "map_binary",
            "operands": ["v6", "v7"],
            "result": "v8",
            "type": {"dtype": "i1", "shape": [256]},
            "attrs": {"fn": "logical_and"},
        },
        {
            "op": "gather",
            "operands": ["t1", "t2"],
            "result": "v9",
            "type": {"dtype": "f32", "shape": [256, 64]},
            "attrs": {"axis": 0, "mode": "take"},
        },
        {
            "op": "gather",
            "operands": ["v9", "t0"],
            "result": "v10",
            "type": {"dtype": "f32", "shape": [256]},
            "attrs": {"axis": 1, "mode": "along_axis"},
        },
        {
            "op": "const",
            "result": "v11",
            "type": {"dtype": "f32", "shape": []},
            "attrs": {"dtype": "f32", "shape": [], "value": 0.0},
        },
        {
            "op": "select",
            "operands": ["v8", "v10", "v11"],
            "result": "v12",
            "type": {"dtype": "f32", "shape": [256]},
        },
    ]
    return {
        "kernel_id": "scrc_k0",
        "intent": {
            "intent_function": {
                "name": "scrc_j0",
                "inputs": [
                    {"name": "t0", "dtype": "i32", "shape": [256]},
                    {"name": "t1", "dtype": "f32", "shape": ["x0", 64]},
                    {"name": "t2", "dtype": "i32", "shape": [256]},
                    {"name": "t3", "dtype": "i32", "shape": []},
                ],
                "outputs": [
                    {"name": "t4", "dtype": "f32", "shape": [256]},
                ],
                "body": body,
                "returns": [{"output": "t4", "value": "v12"}],
            }
        },
        "node_ids": {f"n{index}": f"body[{index}]" for index in range(len(body))},
        "boundary": {
            "domains": {
                "d0": "(0 <= k0) and (k0 < 256)",
                "d1": "(0 <= k1) and (k1 < x0)",
                "d2": "(0 <= k2) and (k2 < 64)",
            },
            "accesses": [
                {
                    "access_id": "a0",
                    "kind": "load",
                    "tensor": "t2",
                    "index_expression": "[k0]",
                },
                {
                    "access_id": "a1",
                    "kind": "load",
                    "tensor": "t0",
                    "index_expression": "[k0]",
                },
                {
                    "access_id": "a2",
                    "kind": "load",
                    "tensor": "t1",
                    "index_expression": "[k1, k2]",
                    "node_ref": "n9",
                    "predicate": (
                        "(0 <= k2) and (k2 < 64) and "
                        "(0 <= k1) and (k1 < x0)"
                    ),
                    "fill_value": 0.0,
                },
                {
                    "access_id": "a3",
                    "kind": "store",
                    "tensor": "t4",
                    "index_expression": "[k0]",
                    "node_ref": "n12",
                },
            ],
            "effects": [
                {"effect_id": "e0", "kind": "read", "target": "t2", "access_ref": "a0"},
                {"effect_id": "e1", "kind": "read", "target": "t0", "access_ref": "a1"},
                {
                    "effect_id": "e2",
                    "kind": "read",
                    "target": "t1",
                    "node_ref": "n9",
                    "access_ref": "a2",
                },
                {
                    "effect_id": "e3",
                    "kind": "write",
                    "target": "t4",
                    "node_ref": "n12",
                    "access_ref": "a3",
                },
            ],
            "aliases": [],
            "broadcasts": [],
            "reductions": [],
            "updates": [],
            "output_mappings": [],
            "index_bounds": [
                {"node_ref": "n9", "data": "t1", "indices": "t2", "axis": 0},
                {"node_ref": "n10", "data": "v9", "indices": "t0", "axis": 1},
            ],
            "consumer_extensions": [],
            "access_guard_relations": [],
            "result_relations": [{"output": "t4", "kind": "exact"}],
            "realization_authorizations": [],
        },
        "signature": {
            "parameters": [
                {
                    "name": "p0",
                    "kind": "pointer",
                    "dtype": "i32",
                    "rank": 1,
                    "mode": "input",
                    "logical_ref": "t0",
                },
                {
                    "name": "p1",
                    "kind": "pointer",
                    "dtype": "f32",
                    "rank": 2,
                    "mode": "input",
                    "logical_ref": "t1",
                },
                {
                    "name": "p2",
                    "kind": "pointer",
                    "dtype": "f32",
                    "rank": 1,
                    "mode": "output",
                    "logical_ref": "t4",
                },
                {
                    "name": "p3",
                    "kind": "pointer",
                    "dtype": "i32",
                    "rank": 1,
                    "mode": "input",
                    "logical_ref": "t2",
                },
                {
                    "name": "p4",
                    "kind": "scalar",
                    "dtype": "i32",
                    "rank": 0,
                    "mode": "value",
                    "logical_ref": "t3",
                },
            ]
        },
        "scopes": [
            {
                "scope_id": "q0",
                "kind": "function",
                "parent_scope": None,
                "captures": [],
                "definitions": [
                    "t0", "t1", "t2", "t3",
                    *[f"v{index}" for index in range(13)],
                ],
                "effects": ["e0", "e1", "e2", "e3"],
            }
        ],
        "storage": [],
        "dimension_constraints": ["x0 >= 1"],
        "value_dimension_relations": [
            {"value_ref": "t3", "dimension_ref": "x0", "relation": "eq"}
        ],
        "dimension_relations": [],
    }


def tilelang_gather2d_artifact() -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "j": _tilelang_j(),
        "c": {
            "slots": [
                {
                    "name": "c0",
                    "kind": "compiler",
                    "value_type": "int32",
                    "role": "launch_block_x",
                    "required": True,
                }
            ],
            "values": {"c0": 128},
        },
        "pi": _compact_pi(),
    }


def cuda_gather2d_artifact() -> dict[str, Any]:
    return {
        "schema": ARTIFACT_SCHEMA,
        "j": _cuda_j(),
        "c": {
            "slots": [
                {
                    "name": "c0",
                    "kind": "compiler",
                    "value_type": "i32",
                    "role": "launch_block_x",
                    "required": True,
                }
            ],
            "values": {"c0": 128},
        },
        "pi": _compact_pi(),
    }


def paired_gather_authoring_receipt() -> dict[str, Any]:
    tilelang = tilelang_gather2d_artifact()
    cuda = cuda_gather2d_artifact()
    return {
        "protocol": "intentir.experiment.manual-gather-paired-authoring.v1",
        "cases": ["tilelang:gather2d", "cuda:tilelang_gather2d"],
        "source_aware_author": "primary Codex agent",
        "lift_agent_calls": 0,
        "shared_mathematical_contract": {
            "input": ["f32[x0,64]", "row_idx:i32[256]", "col_idx:i32[256]"],
            "output": "f32[256]",
            "relation": "out[k0] = input[row_idx[k0], col_idx[k0]]",
            "declared_index_bounds": ["0 <= row < x0", "0 <= col < 64"],
        },
        "frontend_deltas": {
            "tilelang": {
                "source_semantics": "unguarded TIR access under typed index bounds",
                "signature": copy.deepcopy(tilelang["j"]["signature"]),
                "c": copy.deepcopy(tilelang["c"]),
            },
            "cuda": {
                "source_semantics": "explicit row/col guard with zero fill",
                "signature": copy.deepcopy(cuda["j"]["signature"]),
                "c": copy.deepcopy(cuda["c"]),
            },
        },
        "pi": _compact_pi(),
        "claim_scope": (
            "The pair shares the valid-index mathematical gather contract, while "
            "the full J truthfully retains CUDA's explicit bounds-to-zero control "
            "and each frontend's real ABI."
        ),
    }


__all__ = [
    "cuda_gather2d_artifact",
    "paired_gather_authoring_receipt",
    "tilelang_gather2d_artifact",
]
