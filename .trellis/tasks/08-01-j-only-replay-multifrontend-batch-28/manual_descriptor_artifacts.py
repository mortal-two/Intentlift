"""Manual J/C/pi for the five Task28 TensorMap-backed contraction pairs.

The CUDA Sources expose CUtensorMap values only as a physical execution ABI.
Their sidecars project those slots to the same logical tensors consumed by J
and source-blind Replay.  No Lift agent, Source parser, or kernel recipe is used
to author these artifacts.
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
from manual_g2_artifacts import _Graph


DESCRIPTOR_STEMS = (
    "gemm",
    "matmul_relu2d",
    "matmul_bias_relu2d",
    "matmul_fused_epilogue2d",
    "mlp2d",
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


def _domains(*, hidden: bool = False) -> dict[str, str]:
    result = {
        "d0": "(0 <= k0) and (k0 < x0)",
        "d1": "(0 <= k1) and (k1 < x1)",
        "d2": "(0 <= k2) and (k2 < x2)",
    }
    if hidden:
        result["d3"] = "(0 <= k3) and (k3 < x3)"
    return result


def _finish_dimensions(j: dict[str, Any], *, hidden: bool = False) -> dict[str, Any]:
    j["dimension_constraints"] = ["x0 >= 1", "x1 >= 1", "x2 >= 1"]
    if hidden:
        j["dimension_constraints"].append("x3 >= 1")
    return j


def _matmul_relu_j(*, f16_io: bool) -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    lhs = "t0"
    rhs = "t1"
    input_dtype = "f16" if f16_io else "f32"
    if f16_io:
        lhs = graph.op(
            "cast", [lhs], "va", "f32", ["x0", "x2"],
            attrs={"to_dtype": "f32"},
        )
        rhs = graph.op(
            "cast", [rhs], "vb", "f32", ["x2", "x1"],
            attrs={"to_dtype": "f32"},
        )
    mm = graph.op(
        "contraction", [lhs, rhs], "vmm", "f32", ["x0", "x1"],
        attrs={"subscripts": "mk,kn->mn"},
    )
    zero = graph.const("vzero", "f32", 0.0)
    relu_result = "vrelu" if f16_io else "t2"
    graph.op(
        "map_binary", [mm, zero], relu_result, "f32", ["x0", "x1"],
        attrs={"fn": "max"},
    )
    broadcasts.append(
        {"operand": zero, "node_ref": graph.node(relu_result), "axes": [0, 1]}
    )
    if f16_io:
        graph.op(
            "cast", [relu_result], "t2", "f16", ["x0", "x1"],
            attrs={"to_dtype": "f16"},
        )
    j = _make_j(
        inputs=[
            _value("t0", input_dtype, ["x0", "x2"]),
            _value("t1", input_dtype, ["x2", "x1"]),
        ],
        outputs=[_value("t2", input_dtype, ["x0", "x1"])],
        body=graph.body,
        accesses=[
            _access(
                "a0", "load", "t0", "[k0, k2]",
                graph.node(lhs) if lhs != "t0" else graph.node(mm),
            ),
            _access(
                "a1", "load", "t1", "[k2, k1]",
                graph.node(rhs) if rhs != "t1" else graph.node(mm),
            ),
            _access("a2", "store", "t2", "[k0, k1]", graph.node("t2")),
        ],
        domains=_domains(),
        broadcasts=broadcasts,
    )
    return _finish_dimensions(j)


def _matmul_bias_relu_j() -> dict[str, Any]:
    graph = _Graph()
    mm = graph.op(
        "contraction", ["t0", "t1"], "vmm", "f32", ["x0", "x1"],
        attrs={"subscripts": "mk,kn->mn"},
    )
    biased = graph.op(
        "map_binary", [mm, "t2"], "vbiased", "f32", ["x0", "x1"],
        attrs={"fn": "add"},
    )
    zero = graph.const("vzero", "f32", 0.0)
    graph.op(
        "map_binary", [biased, zero], "t3", "f32", ["x0", "x1"],
        attrs={"fn": "max"},
    )
    j = _make_j(
        inputs=[
            _value("t0", "f32", ["x0", "x2"]),
            _value("t1", "f32", ["x2", "x1"]),
            _value("t2", "f32", ["x1"]),
        ],
        outputs=[_value("t3", "f32", ["x0", "x1"])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k2]", graph.node(mm)),
            _access("a1", "load", "t1", "[k2, k1]", graph.node(mm)),
            _access("a2", "load", "t2", "[k1]", graph.node(biased)),
            _access("a3", "store", "t3", "[k0, k1]", graph.node("t3")),
        ],
        domains=_domains(),
        broadcasts=[
            {"operand": "t2", "node_ref": graph.node(biased), "axes": [0]},
            {"operand": zero, "node_ref": graph.node("t3"), "axes": [0, 1]},
        ],
    )
    return _finish_dimensions(j)


def _fused_epilogue_j(frontend: str) -> dict[str, Any]:
    graph = _Graph()
    mask_dtype = "i1" if frontend == "tilelang" else "i8"
    mm = graph.op(
        "contraction", ["t0", "t1"], "vmm", "f32", ["x0", "x1"],
        attrs={"subscripts": "mk,kn->mn"},
    )
    biased = graph.op(
        "map_binary", [mm, "t2"], "vbiased", "f32", ["x0", "x1"],
        attrs={"fn": "add"},
    )
    row = "t3"
    col = "t4"
    if mask_dtype != "i1":
        row = graph.op(
            "cast", [row], "vrow", "i1", ["x0"], attrs={"to_dtype": "i1"}
        )
        col = graph.op(
            "cast", [col], "vcol", "i1", ["x1"], attrs={"to_dtype": "i1"}
        )
    condition = graph.op(
        "map_binary", [row, col], "vcondition", "i1", ["x0", "x1"],
        attrs={"fn": "logical_and"},
    )
    mask = graph.op(
        "cast", [condition], "vmask", "f32", ["x0", "x1"],
        attrs={"to_dtype": "f32"},
    )
    graph.op(
        "map_binary", [biased, mask], "t5", "f32", ["x0", "x1"],
        attrs={"fn": "mul"},
    )
    j = _make_j(
        inputs=[
            _value("t0", "f32", ["x0", "x2"]),
            _value("t1", "f32", ["x2", "x1"]),
            _value("t2", "f32", ["x1"]),
            _value("t3", mask_dtype, ["x0"]),
            _value("t4", mask_dtype, ["x1"]),
        ],
        outputs=[_value("t5", "f32", ["x0", "x1"])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k2]", graph.node(mm)),
            _access("a1", "load", "t1", "[k2, k1]", graph.node(mm)),
            _access("a2", "load", "t2", "[k1]", graph.node(biased)),
            _access(
                "a3", "load", "t3", "[k0]",
                graph.node(row) if row != "t3" else graph.node(condition),
            ),
            _access(
                "a4", "load", "t4", "[k1]",
                graph.node(col) if col != "t4" else graph.node(condition),
            ),
            _access("a5", "store", "t5", "[k0, k1]", graph.node("t5")),
        ],
        domains=_domains(),
        broadcasts=[
            {"operand": "t2", "node_ref": graph.node(biased), "axes": [0]},
            {"operand": row, "node_ref": graph.node(condition), "axes": [1]},
            {"operand": col, "node_ref": graph.node(condition), "axes": [0]},
        ],
    )
    return _finish_dimensions(j)


def _mlp_j() -> dict[str, Any]:
    graph = _Graph()
    hidden = graph.op(
        "contraction", ["t0", "t1"], "vhidden0", "f32", ["x0", "x3"],
        attrs={"subscripts": "mk,kh->mh"},
    )
    hidden_bias = graph.op(
        "map_binary", [hidden, "t2"], "vhidden1", "f32", ["x0", "x3"],
        attrs={"fn": "add"},
    )
    zero = graph.const("vzero", "f32", 0.0)
    activated = graph.op(
        "map_binary", [hidden_bias, zero], "vhidden", "f32", ["x0", "x3"],
        attrs={"fn": "max"},
    )
    projected = graph.op(
        "contraction", [activated, "t3"], "vout", "f32", ["x0", "x1"],
        attrs={"subscripts": "mh,hn->mn"},
    )
    graph.op(
        "map_binary", [projected, "t4"], "t5", "f32", ["x0", "x1"],
        attrs={"fn": "add"},
    )
    j = _make_j(
        inputs=[
            _value("t0", "f32", ["x0", "x2"]),
            _value("t1", "f32", ["x2", "x3"]),
            _value("t2", "f32", ["x3"]),
            _value("t3", "f32", ["x3", "x1"]),
            _value("t4", "f32", ["x1"]),
        ],
        outputs=[_value("t5", "f32", ["x0", "x1"])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k2]", graph.node(hidden)),
            _access("a1", "load", "t1", "[k2, k3]", graph.node(hidden)),
            _access("a2", "load", "t2", "[k3]", graph.node(hidden_bias)),
            _access("a3", "load", "t3", "[k3, k1]", graph.node(projected)),
            _access("a4", "load", "t4", "[k1]", graph.node("t5")),
            _access("a5", "store", "t5", "[k0, k1]", graph.node("t5")),
        ],
        domains=_domains(hidden=True),
        broadcasts=[
            {"operand": "t2", "node_ref": graph.node(hidden_bias), "axes": [0]},
            {"operand": zero, "node_ref": graph.node(activated), "axes": [0, 1]},
            {"operand": "t4", "node_ref": graph.node("t5"), "axes": [0]},
        ],
    )
    return _finish_dimensions(j, hidden=True)


_J_BUILDERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "gemm": lambda _frontend: _matmul_relu_j(f16_io=True),
    "matmul_relu2d": lambda _frontend: _matmul_relu_j(f16_io=False),
    "matmul_bias_relu2d": lambda _frontend: _matmul_bias_relu_j(),
    "matmul_fused_epilogue2d": _fused_epilogue_j,
    "mlp2d": lambda _frontend: _mlp_j(),
}


def _signature(stem: str, frontend: str) -> dict[str, Any]:
    if stem in {"gemm", "matmul_relu2d"}:
        dtype = "f16" if stem == "gemm" else "f32"
        order = [("t0", dtype, 2, "input"), ("t1", dtype, 2, "input"), ("t2", dtype, 2, "output")]
    elif stem == "matmul_bias_relu2d":
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 2, "input"),
            "t2": ("f32", 1, "input"), "t3": ("f32", 2, "output"),
        }
        refs = ["t0", "t1", "t2", "t3"] if frontend == "tilelang" else ["t0", "t1", "t3", "t2"]
        order = [(ref, *logical[ref]) for ref in refs]
    elif stem == "matmul_fused_epilogue2d":
        mask_dtype = "i1" if frontend == "tilelang" else "i8"
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 2, "input"),
            "t2": ("f32", 1, "input"), "t3": (mask_dtype, 1, "input"),
            "t4": (mask_dtype, 1, "input"), "t5": ("f32", 2, "output"),
        }
        refs = ["t0", "t1", "t2", "t3", "t4", "t5"] if frontend == "tilelang" else ["t0", "t1", "t5", "t2", "t4", "t3"]
        order = [(ref, *logical[ref]) for ref in refs]
    else:
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 2, "input"),
            "t2": ("f32", 1, "input"), "t3": ("f32", 2, "input"),
            "t4": ("f32", 1, "input"), "t5": ("f32", 2, "output"),
        }
        refs = ["t0", "t1", "t2", "t3", "t4", "t5"] if frontend == "tilelang" else ["t0", "t5", "t1", "t3", "t2", "t4"]
        order = [(ref, *logical[ref]) for ref in refs]
    parameters = [
        _p(frontend, ref, index, dtype, rank, mode)
        for index, (ref, dtype, rank, mode) in enumerate(order)
    ]
    if frontend == "cuda":
        dimension_order = ["x2", "x0", "x1"]
        if stem == "mlp2d":
            dimension_order = ["x3", "x2", "x0", "x1"]
        for dimension in dimension_order:
            parameters.append(
                _parameter(
                    dimension,
                    name=f"p{len(parameters)}",
                    kind="scalar",
                    dtype="i32",
                    rank=0,
                    mode="value",
                )
            )
    return {"parameters": parameters}


def _compact_pi() -> dict[str, Any]:
    return {
        "schema": PI_SCHEMA,
        "program_axes": [
            {"domain_ref": "d1", "axis": 0},
            {"domain_ref": "d0", "axis": 1},
        ],
    }


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
                    "name": "c0", "kind": "compiler", "value_type": "i32",
                    "role": "launch_block_x", "required": True,
                },
                {
                    "name": "c1", "kind": "compiler", "value_type": "i1",
                    "role": "launch_grid_y", "required": True,
                },
            ],
            "values": {"c0": 128, "c1": True},
        },
        "pi": _compact_pi(),
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
            "intent", "node_ids", "boundary", "scopes", "storage",
            "dimension_constraints", "value_dimension_relations", "dimension_relations",
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
            "tilelang": {"signature": copy.deepcopy(tilelang["j"]["signature"]), "c": copy.deepcopy(tilelang["c"])},
            "cuda": {"signature": copy.deepcopy(cuda["j"]["signature"]), "c": copy.deepcopy(cuda["c"])},
        },
        "pi": _compact_pi(),
        "physical_abi_isolation": {
            "j_contains_tensor_map": False,
            "replay_contains_tensor_map": False,
            "cuda_source_uses_sidecar_tensor_map": True,
        },
        "claim_scope": (
            "Shared sections are measured after independent Source audit. CUDA "
            "TensorMap, swizzle, warp-specialized launch, and dynamic shared memory "
            "remain physical Source execution facts and are absent from J/Replay."
        ),
    }


CASE_IDS = tuple(
    case_id
    for stem in DESCRIPTOR_STEMS
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
    "DESCRIPTOR_STEMS",
    "authoring_receipt",
    "build_artifact",
]
