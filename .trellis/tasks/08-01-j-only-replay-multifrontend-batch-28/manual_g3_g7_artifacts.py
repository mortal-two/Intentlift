"""Manual J/C/pi authoring for the runnable Task28 G3--G7 remainder.

The primary experiment author audited both frontend Sources and writes the exact
logical program directly.  No Lift agent participates.  Despite its historical
name, ``upsample_bicubic2d_aa`` currently performs exact 2x nearest-neighbor
replication in both Dataset Sources; J records that observed computation rather
than inventing bicubic semantics from the filename.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from manual_g1_artifacts import (
    ARTIFACT_SCHEMA,
    PI_SCHEMA,
    _access,
    _make_j,
    _parameter,
    _tensor_parameter,
    _value,
)
from manual_g2_artifacts import _Graph, _reduce, _reduction


G3_G7_RUNNABLE_STEMS = (
    "gather2d",
    "transpose2d",
    "upsample_bicubic2d_aa",
    "_attn_fwd",
    "flash_attention2d",
    "masked_attention2d",
)


def _p(
    frontend: str,
    ref: str,
    index: int,
    dtype: str,
    rank: int,
    mode: str,
) -> dict[str, Any]:
    return _tensor_parameter(
        ref,
        name=f"p{index}",
        dtype=dtype,
        rank=rank,
        mode=mode,
        frontend=frontend,
    )


def _gather_tilelang_j() -> dict[str, Any]:
    graph = _Graph()
    graph.op(
        "gather",
        ["t0", "t1"],
        "vrows",
        "f32",
        [256, 64],
        attrs={"axis": 0, "mode": "take"},
    )
    graph.op(
        "gather",
        ["vrows", "t2"],
        "t3",
        "f32",
        [256],
        attrs={"axis": 1, "mode": "along_axis"},
    )
    j = _make_j(
        inputs=[
            _value("t0", "f32", ["x0", 64]),
            _value("t1", "i32", [256]),
            _value("t2", "i32", [256]),
        ],
        outputs=[_value("t3", "f32", [256])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t1", "[k0]", graph.node("vrows")),
            _access("a1", "load", "t0", "[k1, k2]", graph.node("vrows")),
            _access("a2", "load", "t2", "[k0]", graph.node("t3")),
            _access("a3", "store", "t3", "[k0]", graph.node("t3")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < 256)",
            "d1": "(0 <= k1) and (k1 < x0)",
            "d2": "(0 <= k2) and (k2 < 64)",
        },
        result_kind="exact",
    )
    j["boundary"]["index_bounds"] = [
        {"node_ref": graph.node("vrows"), "data": "t0", "indices": "t1", "axis": 0},
        {"node_ref": graph.node("t3"), "data": "vrows", "indices": "t2", "axis": 1},
    ]
    return j


def _gather_cuda_j() -> dict[str, Any]:
    # CUDA ABI: col_idx, input, output, row_idx, M.
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    zero_i = graph.const("vzero_i", "i32", 0)
    col_ge = graph.op("map_binary", ["t0", zero_i], "vcol_ge", "i1", [256], attrs={"fn": "ge"})
    row_ge = graph.op("map_binary", ["t2", zero_i], "vrow_ge", "i1", [256], attrs={"fn": "ge"})
    broadcasts.extend(
        [
            {"operand": zero_i, "node_ref": graph.node(col_ge), "axes": [0]},
            {"operand": zero_i, "node_ref": graph.node(row_ge), "axes": [0]},
        ]
    )
    sixty_four = graph.const("vsixty_four", "i32", 64)
    col_lt = graph.op("map_binary", ["t0", sixty_four], "vcol_lt", "i1", [256], attrs={"fn": "lt"})
    row_lt = graph.op("map_binary", ["t2", "t3"], "vrow_lt", "i1", [256], attrs={"fn": "lt"})
    broadcasts.extend(
        [
            {"operand": sixty_four, "node_ref": graph.node(col_lt), "axes": [0]},
            {"operand": "t3", "node_ref": graph.node(row_lt), "axes": [0]},
        ]
    )
    col_ok = graph.op("map_binary", [col_ge, col_lt], "vcol_ok", "i1", [256], attrs={"fn": "logical_and"})
    row_ok = graph.op("map_binary", [row_ge, row_lt], "vrow_ok", "i1", [256], attrs={"fn": "logical_and"})
    valid = graph.op("map_binary", [col_ok, row_ok], "vvalid", "i1", [256], attrs={"fn": "logical_and"})
    graph.op(
        "gather", ["t1", "t2"], "vrows", "f32", [256, 64],
        attrs={"axis": 0, "mode": "take"},
    )
    graph.op(
        "gather", ["vrows", "t0"], "vgathered", "f32", [256],
        attrs={"axis": 1, "mode": "along_axis"},
    )
    zero_f = graph.const("vzero_f", "f32", 0.0)
    graph.op("select", [valid, "vgathered", zero_f], "t4", "f32", [256])
    broadcasts.append({"operand": zero_f, "node_ref": graph.node("t4"), "axes": [0]})
    j = _make_j(
        inputs=[
            _value("t0", "i32", [256]),
            _value("t1", "f32", ["x0", 64]),
            _value("t2", "i32", [256]),
            _value("t3", "i32", []),
        ],
        outputs=[_value("t4", "f32", [256])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0]", graph.node(col_ge)),
            _access("a1", "load", "t2", "[k0]", graph.node(row_ge)),
            {
                **_access("a2", "load", "t1", "[k1, k2]", graph.node("vrows")),
                "predicate": "(0 <= k1) and (k1 < x0) and (0 <= k2) and (k2 < 64)",
                "fill_value": 0.0,
            },
            _access("a3", "store", "t4", "[k0]", graph.node("t4")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < 256)",
            "d1": "(0 <= k1) and (k1 < x0)",
            "d2": "(0 <= k2) and (k2 < 64)",
        },
        broadcasts=broadcasts,
        result_kind="exact",
    )
    j["boundary"]["index_bounds"] = [
        {"node_ref": graph.node("vrows"), "data": "t1", "indices": "t2", "axis": 0},
        {"node_ref": graph.node("vgathered"), "data": "vrows", "indices": "t0", "axis": 1},
    ]
    j["value_dimension_relations"] = [
        {"value_ref": "t3", "dimension_ref": "x0", "relation": "eq"}
    ]
    return j


def _transpose_j() -> dict[str, Any]:
    graph = _Graph()
    graph.op(
        "transpose", ["t0"], "t1", "f32", [16, "x0"], attrs={"perm": [1, 0]}
    )
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16])],
        outputs=[_value("t1", "f32", [16, "x0"])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", graph.node("t1")),
            _access("a1", "store", "t1", "[k1, k0]", graph.node("t1")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 16)",
        },
        result_kind="exact",
    )


def _nearest_upsample_j() -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    oh = graph.op("iota", [], "voh", "i32", [8], attrs={"n": 8, "dtype": "i32"})
    two = graph.const("vtwo", "i32", 2)
    ih = graph.op("map_binary", [oh, two], "vih", "i32", [8], attrs={"fn": "div"})
    broadcasts.append({"operand": two, "node_ref": graph.node(ih), "axes": [0]})
    ow = graph.op("iota", [], "vow", "i32", [8], attrs={"n": 8, "dtype": "i32"})
    iw = graph.op("map_binary", [ow, two], "viw", "i32", [8], attrs={"fn": "div"})
    broadcasts.append({"operand": two, "node_ref": graph.node(iw), "axes": [0]})
    graph.op(
        "gather", ["t0", ih], "vheight", "f32", [1, 1, 8, 4],
        attrs={"axis": 2, "mode": "take"},
    )
    graph.op(
        "gather", ["vheight", iw], "t1", "f32", [1, 1, 8, 8],
        attrs={"axis": 3, "mode": "take"},
    )
    j = _make_j(
        inputs=[_value("t0", "f32", [1, 1, 4, 4])],
        outputs=[_value("t1", "f32", [1, 1, 8, 8])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[0, 0, k0, k1]", graph.node("vheight")),
            _access("a1", "store", "t1", "[0, 0, k2, k3]", graph.node("t1")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < 4)",
            "d1": "(0 <= k1) and (k1 < 4)",
            "d2": "(0 <= k2) and (k2 < 8)",
            "d3": "(0 <= k3) and (k3 < 8)",
        },
        broadcasts=broadcasts,
        result_kind="exact",
    )
    j["dimension_constraints"] = []
    j["boundary"]["index_bounds"] = [
        {"node_ref": graph.node("vheight"), "data": "t0", "indices": ih, "axis": 2},
        {"node_ref": graph.node("t1"), "data": "vheight", "indices": iw, "axis": 3},
    ]
    return j


def _attention_2d_j(*, size: int) -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    graph.op(
        "contraction", ["t0", "t1"], "vscores0", "f32", [size, size],
        attrs={"subscripts": "qd,kd->qk"},
    )
    graph.op("map_binary", ["vscores0", "t3"], "vscaled", "f32", [size, size], attrs={"fn": "mul"})
    broadcasts.append({"operand": "t3", "node_ref": graph.node("vscaled"), "axes": [0, 1]})
    query = graph.op("iota", [], "vquery", "i32", [size], attrs={"n": size, "dtype": "i32"})
    key = graph.op("iota", [], "vkey", "i32", [size], attrs={"n": size, "dtype": "i32"})
    causal = graph.op("map_binary", [query, key], "vcausal", "i1", [size, size], attrs={"fn": "ge"})
    broadcasts.extend(
        [
            {"operand": query, "node_ref": graph.node(causal), "axes": [1]},
            {"operand": key, "node_ref": graph.node(causal), "axes": [0]},
        ]
    )
    neg = graph.const("vneg", "f32", -1.0e9)
    graph.op("select", [causal, "vscaled", neg], "vmasked", "f32", [size, size])
    broadcasts.append({"operand": neg, "node_ref": graph.node("vmasked"), "axes": [0, 1]})
    _reduce(graph, "vmasked", "vmax", [size], fn="max", axes=[1])
    graph.op("map_binary", ["vmasked", "vmax"], "vcentered", "f32", [size, size], attrs={"fn": "sub"})
    broadcasts.append({"operand": "vmax", "node_ref": graph.node("vcentered"), "axes": [1]})
    graph.op("map_unary", ["vcentered"], "vexp", "f32", [size, size], attrs={"fn": "exp"})
    _reduce(graph, "vexp", "vsum", [size], fn="sum", axes=[1])
    graph.op("map_binary", ["vexp", "vsum"], "vprob", "f32", [size, size], attrs={"fn": "div"})
    broadcasts.append({"operand": "vsum", "node_ref": graph.node("vprob"), "axes": [1]})
    graph.op(
        "contraction", ["vprob", "t2"], "t4", "f32", [size, size],
        attrs={"subscripts": "qk,kd->qd"},
    )
    j = _make_j(
        inputs=[
            _value("t0", "f32", [size, size]),
            _value("t1", "f32", [size, size]),
            _value("t2", "f32", [size, size]),
            _value("t3", "f32", [1]),
        ],
        outputs=[_value("t4", "f32", [size, size])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k2]", graph.node("vscores0")),
            _access("a1", "load", "t1", "[k1, k2]", graph.node("vscores0")),
            _access("a2", "load", "t3", "[0]", graph.node("vscaled")),
            _access("a3", "load", "t2", "[k1, k2]", graph.node("t4")),
            _access("a4", "store", "t4", "[k0, k2]", graph.node("t4")),
        ],
        domains={
            "d0": f"(0 <= k0) and (k0 < {size})",
            "d1": f"(0 <= k1) and (k1 < {size})",
            "d2": f"(0 <= k2) and (k2 < {size})",
        },
        broadcasts=broadcasts,
        reductions=[
            _reduction(graph, "vmax", "-inf"),
            _reduction(graph, "vsum", 0.0),
        ],
    )
    j["dimension_constraints"] = []
    return j


def _attn_fwd_j() -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    score_shape = [1, 1, 16, 16]
    graph.op(
        "contraction", ["t0", "t1"], "vscores0", "f32", score_shape,
        attrs={"subscripts": "zhqd,zhkd->zhqk"},
    )
    graph.op("map_binary", ["vscores0", "t3"], "vbiased", "f32", score_shape, attrs={"fn": "add"})
    graph.op("map_binary", ["vbiased", "t4"], "vscaled", "f32", score_shape, attrs={"fn": "mul"})
    broadcasts.append({"operand": "t4", "node_ref": graph.node("vscaled"), "axes": [0, 1, 2, 3]})
    _reduce(graph, "vscaled", "vmax", [1, 1, 16], fn="max", axes=[3])
    graph.op("map_binary", ["vscaled", "vmax"], "vcentered", "f32", score_shape, attrs={"fn": "sub"})
    broadcasts.append({"operand": "vmax", "node_ref": graph.node("vcentered"), "axes": [3]})
    graph.op("map_unary", ["vcentered"], "vexp", "f32", score_shape, attrs={"fn": "exp"})
    _reduce(graph, "vexp", "vsum", [1, 1, 16], fn="sum", axes=[3])
    graph.op("map_binary", ["vexp", "vsum"], "vprob", "f32", score_shape, attrs={"fn": "div"})
    broadcasts.append({"operand": "vsum", "node_ref": graph.node("vprob"), "axes": [3]})
    graph.op(
        "contraction", ["vprob", "t2"], "t5", "f32", [1, 1, 16, 16],
        attrs={"subscripts": "zhqk,zhkd->zhqd"},
    )
    j = _make_j(
        inputs=[
            _value("t0", "f32", [1, 1, 16, 16]),
            _value("t1", "f32", [1, 1, 16, 16]),
            _value("t2", "f32", [1, 1, 16, 16]),
            _value("t3", "f32", [1, 1, 16, 16]),
            _value("t4", "f32", [1]),
        ],
        outputs=[_value("t5", "f32", [1, 1, 16, 16])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1, k2, k4]", graph.node("vscores0")),
            _access("a1", "load", "t1", "[k0, k1, k3, k4]", graph.node("vscores0")),
            _access("a2", "load", "t3", "[k0, k1, k2, k3]", graph.node("vbiased")),
            _access("a3", "load", "t4", "[0]", graph.node("vscaled")),
            _access("a4", "load", "t2", "[k0, k1, k3, k4]", graph.node("t5")),
            _access("a5", "store", "t5", "[k0, k1, k2, k4]", graph.node("t5")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < 1)",
            "d1": "(0 <= k1) and (k1 < 1)",
            "d2": "(0 <= k2) and (k2 < 16)",
            "d3": "(0 <= k3) and (k3 < 16)",
            "d4": "(0 <= k4) and (k4 < 16)",
        },
        broadcasts=broadcasts,
        reductions=[
            _reduction(graph, "vmax", "-inf"),
            _reduction(graph, "vsum", 0.0),
        ],
    )
    j["dimension_constraints"] = []
    return j


_J_BUILDERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "gather2d": lambda frontend: _gather_tilelang_j() if frontend == "tilelang" else _gather_cuda_j(),
    "transpose2d": lambda _frontend: _transpose_j(),
    "upsample_bicubic2d_aa": lambda _frontend: _nearest_upsample_j(),
    "_attn_fwd": lambda _frontend: _attn_fwd_j(),
    "flash_attention2d": lambda _frontend: _attention_2d_j(size=64),
    "masked_attention2d": lambda _frontend: _attention_2d_j(size=16),
}


_BLOCK_X = {
    "gather2d": 128,
    "transpose2d": 128,
    "upsample_bicubic2d_aa": 128,
    "_attn_fwd": 128,
    "flash_attention2d": 1,
    "masked_attention2d": 1,
}


def _signature(stem: str, frontend: str) -> dict[str, Any]:
    if stem == "gather2d":
        if frontend == "tilelang":
            params = [
                _p(frontend, "t0", 0, "f32", 2, "input"),
                _p(frontend, "t1", 1, "i32", 1, "input"),
                _p(frontend, "t2", 2, "i32", 1, "input"),
                _p(frontend, "t3", 3, "f32", 1, "output"),
            ]
        else:
            params = [
                _p(frontend, "t0", 0, "i32", 1, "input"),
                _p(frontend, "t1", 1, "f32", 2, "input"),
                _p(frontend, "t4", 2, "f32", 1, "output"),
                _p(frontend, "t2", 3, "i32", 1, "input"),
                _parameter("t3", name="p4", kind="scalar", dtype="i32", rank=0, mode="value"),
            ]
    elif stem == "transpose2d":
        params = [
            _p(frontend, "t0", 0, "f32", 2, "input"),
            _p(frontend, "t1", 1, "f32", 2, "output"),
        ]
        if frontend == "cuda":
            params.append(_parameter("x0", name="p2", kind="scalar", dtype="i32", rank=0, mode="value"))
    elif stem == "upsample_bicubic2d_aa":
        params = [
            _p(frontend, "t0", 0, "f32", 4, "input"),
            _p(frontend, "t1", 1, "f32", 4, "output"),
        ]
    elif stem == "_attn_fwd":
        logical = {
            "t0": ("f32", 4, "input"),
            "t1": ("f32", 4, "input"),
            "t2": ("f32", 4, "input"),
            "t3": ("f32", 4, "input"),
            "t4": ("f32", 1, "input"),
            "t5": ("f32", 4, "output"),
        }
        order = ["t0", "t1", "t2", "t3", "t4", "t5"] if frontend == "tilelang" else ["t1", "t5", "t0", "t2", "t3", "t4"]
        params = [_p(frontend, ref, i, *logical[ref]) for i, ref in enumerate(order)]
    else:
        logical = {
            "t0": ("f32", 2, "input"),
            "t1": ("f32", 2, "input"),
            "t2": ("f32", 2, "input"),
            "t3": ("f32", 1, "input"),
            "t4": ("f32", 2, "output"),
        }
        order = ["t0", "t1", "t2", "t3", "t4"] if frontend == "tilelang" else ["t1", "t4", "t0", "t2", "t3"]
        params = [_p(frontend, ref, i, *logical[ref]) for i, ref in enumerate(order)]
    return {"parameters": params}


def _compact_pi(stem: str) -> dict[str, Any]:
    if stem == "upsample_bicubic2d_aa":
        axes = [{"domain_ref": "d2", "axis": 0}, {"domain_ref": "d3", "axis": 1}]
    elif stem == "_attn_fwd":
        axes = [{"domain_ref": "d2", "axis": 0}]
    else:
        axes = [{"domain_ref": "d0", "axis": 0}]
    return {"schema": PI_SCHEMA, "program_axes": axes}


def build_artifact(case_id: str) -> dict[str, Any]:
    frontend, raw_stem = case_id.split(":", 1)
    stem = raw_stem.removeprefix("tilelang_") if frontend == "cuda" else raw_stem
    if frontend not in {"tilelang", "cuda"} or stem not in _J_BUILDERS:
        raise KeyError(case_id)
    j = _J_BUILDERS[stem](frontend)
    j["signature"] = _signature(stem, frontend)
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
            "values": {"c0": _BLOCK_X[stem]},
        },
        "pi": _compact_pi(stem),
    }


def authoring_receipt(case_id: str) -> dict[str, Any]:
    frontend, raw_stem = case_id.split(":", 1)
    stem = raw_stem.removeprefix("tilelang_") if frontend == "cuda" else raw_stem
    pair = (f"tilelang:{stem}", f"cuda:tilelang_{stem}")
    tilelang = build_artifact(pair[0])
    cuda = build_artifact(pair[1])
    shared_sections = [
        key
        for key in (
            "intent",
            "node_ids",
            "boundary",
            "scopes",
            "storage",
            "dimension_constraints",
            "value_dimension_relations",
            "dimension_relations",
        )
        if tilelang["j"].get(key) == cuda["j"].get(key)
    ]
    return {
        "protocol": "intentir.experiment.manual-paired-authoring.v2",
        "cases": list(pair),
        "source_aware_author": "primary Codex agent",
        "lift_agent_calls": 0,
        "stem": stem,
        "shared_sections": shared_sections,
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
        "pi": _compact_pi(stem),
        "claim_scope": (
            "Shared sections are measured equality after independent Source audit. "
            "The historical upsample name does not override its observed nearest "
            "neighbor computation."
        ),
    }


CASE_IDS = tuple(
    case_id
    for stem in G3_G7_RUNNABLE_STEMS
    for case_id in (f"tilelang:{stem}", f"cuda:tilelang_{stem}")
)
CASE_BUILDERS = {
    case_id: (lambda selected=case_id: build_artifact(selected)) for case_id in CASE_IDS
}
CASE_RECEIPTS = {
    case_id: (lambda selected=case_id: authoring_receipt(selected)) for case_id in CASE_IDS
}


__all__ = [
    "CASE_BUILDERS",
    "CASE_IDS",
    "CASE_RECEIPTS",
    "G3_G7_RUNNABLE_STEMS",
    "authoring_receipt",
    "build_artifact",
]
