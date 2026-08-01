"""Source-audited manual J/C/pi authoring for the first paired experiment.

This module is deliberately small and explicit.  It is not a Lift agent, a
frontend parser, or a kernel lowering rule.  The shared dictionary below records
the mathematical add2d contract observed in both current Dataset sources; the two
constructors expose only the real ABI/C differences of TileLang and generated
CUDA.  Fresh Replay never imports this module and never receives pi.
"""

from __future__ import annotations

import copy
from typing import Any


ARTIFACT_SCHEMA = "intentir.scrc.artifact.v13"
PI_SCHEMA = "intentir.pi.normalization-witness.v1"


def _shared_j() -> dict[str, Any]:
    return {
        "kernel_id": "scrc_k0",
        "intent": {
            "intent_function": {
                "name": "scrc_j0",
                "inputs": [
                    {"name": "t0", "dtype": "f32", "shape": ["x0", 16]},
                    {"name": "t1", "dtype": "f32", "shape": ["x0", 16]},
                ],
                "outputs": [
                    {"name": "t2", "dtype": "f32", "shape": ["x0", 16]},
                ],
                "body": [
                    {
                        "op": "map_binary",
                        "operands": ["t0", "t1"],
                        "result": "t2",
                        "type": {"dtype": "f32", "shape": ["x0", 16]},
                        "attrs": {"fn": "add"},
                    }
                ],
            }
        },
        "node_ids": {"n0": "body[0]"},
        "boundary": {
            "domains": {
                "d0": "(0 <= k0) and (k0 < x0)",
                "d1": "(0 <= k1) and (k1 < 16)",
            },
            "accesses": [
                {
                    "access_id": "a0",
                    "kind": "load",
                    "tensor": "t0",
                    "index_expression": "[k0, k1]",
                    "node_ref": "n0",
                },
                {
                    "access_id": "a1",
                    "kind": "load",
                    "tensor": "t1",
                    "index_expression": "[k0, k1]",
                    "node_ref": "n0",
                },
                {
                    "access_id": "a2",
                    "kind": "store",
                    "tensor": "t2",
                    "index_expression": "[k0, k1]",
                    "node_ref": "n0",
                },
            ],
            "effects": [
                {
                    "effect_id": "e0",
                    "kind": "read",
                    "target": "t0",
                    "node_ref": "n0",
                    "access_ref": "a0",
                },
                {
                    "effect_id": "e1",
                    "kind": "read",
                    "target": "t1",
                    "node_ref": "n0",
                    "access_ref": "a1",
                },
                {
                    "effect_id": "e2",
                    "kind": "write",
                    "target": "t2",
                    "node_ref": "n0",
                    "access_ref": "a2",
                },
            ],
            "aliases": [],
            "broadcasts": [],
            "reductions": [],
            "updates": [],
            "output_mappings": [],
            "index_bounds": [],
            "consumer_extensions": [],
            "access_guard_relations": [],
            "result_relations": [{"output": "t2", "kind": "tolerance"}],
            "realization_authorizations": [],
        },
        "scopes": [
            {
                "scope_id": "q0",
                "kind": "function",
                "parent_scope": None,
                "captures": [],
                "definitions": ["t0", "t1", "t2"],
                "effects": ["e0", "e1", "e2"],
            }
        ],
        "storage": [],
        "dimension_constraints": ["x0 >= 1"],
        "value_dimension_relations": [],
        "dimension_relations": [],
    }


def _compact_pi() -> dict[str, Any]:
    # Both current Sources directly bind the outer row domain to block/program
    # axis x.  The active v2 normalizer keeps this row hard; it authorizes no
    # axis rewrite and supplies no instruction to Replay.
    return {
        "schema": PI_SCHEMA,
        "program_axes": [{"domain_ref": "d0", "axis": 0}],
    }


def tilelang_add2d_artifact() -> dict[str, Any]:
    j = _shared_j()
    j["signature"] = {
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
                "dtype": "f32",
                "rank": 2,
                "mode": "input",
                "logical_ref": "t1",
            },
            {
                "name": "p2",
                "kind": "tensor",
                "dtype": "f32",
                "rank": 2,
                "mode": "output",
                "logical_ref": "t2",
            },
        ]
    }
    return {
        "schema": ARTIFACT_SCHEMA,
        "j": j,
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
            "values": {"c0": 16},
        },
        "pi": _compact_pi(),
    }


def cuda_add2d_artifact() -> dict[str, Any]:
    j = _shared_j()
    j["signature"] = {
        "parameters": [
            {
                "name": "p0",
                "kind": "pointer",
                "dtype": "f32",
                "rank": 2,
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
                "rank": 2,
                "mode": "output",
                "logical_ref": "t2",
            },
            {
                "name": "p3",
                "kind": "scalar",
                "dtype": "i32",
                "rank": 0,
                "mode": "value",
                "logical_ref": "x0",
            },
        ]
    }
    return {
        "schema": ARTIFACT_SCHEMA,
        "j": j,
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
            "values": {"c0": 16},
        },
        "pi": _compact_pi(),
    }


def paired_authoring_receipt() -> dict[str, Any]:
    tilelang = tilelang_add2d_artifact()
    cuda = cuda_add2d_artifact()
    shared_j = _shared_j()
    return {
        "protocol": "intentir.experiment.manual-paired-authoring.v1",
        "cases": ["tilelang:add2d", "cuda:tilelang_add2d"],
        "source_aware_author": "primary Codex agent",
        "lift_agent_calls": 0,
        "shared_mathematical_j": copy.deepcopy(shared_j),
        "shared_sections": [
            "intent",
            "node_ids",
            "boundary",
            "scopes",
            "storage",
            "dimension_constraints",
            "value_dimension_relations",
            "dimension_relations",
        ],
        "frontend_deltas": {
            "tilelang": {
                "signature": copy.deepcopy(tilelang["j"]["signature"]),
                "c": copy.deepcopy(tilelang["c"]),
            },
            "cuda": {
                "signature": copy.deepcopy(cuda["j"]["signature"]),
                "c": copy.deepcopy(cuda["c"]),
            },
        },
        "pi": _compact_pi(),
        "claim_scope": (
            "The pair shares mathematical J structure but not byte-identical ABI; "
            "each artifact is independently TypeChecked and Gate-certified."
        ),
    }


__all__ = [
    "cuda_add2d_artifact",
    "paired_authoring_receipt",
    "tilelang_add2d_artifact",
]
