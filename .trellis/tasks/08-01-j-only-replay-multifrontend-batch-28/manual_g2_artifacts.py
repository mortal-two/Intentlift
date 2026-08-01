"""Manual, Source-audited J/C/pi authoring for Task28 cohort G2.

No Lift agent or persistent kernel lowering recipe participates here.  The
primary experiment author writes exact J semantics after auditing both Sources;
fresh Replay receives J + MaskC(C) only.  Grouped reductions are represented in
ordinary J with an authored one-hot selector and contraction, so physical ABI
ranks remain exact and pi never carries a hidden reshape or formula.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable

from manual_g1_artifacts import (
    ARTIFACT_SCHEMA,
    PI_SCHEMA,
    _access,
    _const,
    _make_j,
    _op,
    _parameter,
    _tensor_parameter,
    _value,
)


G2_STEMS = (
    "row_sum",
    "row_max",
    "grouped_row_sum2d",
    "softmax_inner",
    "masked_softmax2d",
    "groupnorm",
    "layernorm",
    "rms_norm2d",
    "layer_norm_residual2d",
    "rms_norm_residual2d",
)


class _Graph:
    """Small authoring aid that allocates only stable J node references."""

    def __init__(self) -> None:
        self.body: list[dict[str, Any]] = []
        self.refs: dict[str, str] = {}

    def add(self, row: dict[str, Any]) -> str:
        result = row.get("result")
        if not isinstance(result, str) or not result:
            raise ValueError("G2 manual graph requires one named result per operation")
        if result in self.refs:
            raise ValueError(f"duplicate G2 J result {result!r}")
        node_ref = f"n{len(self.body)}"
        self.body.append(row)
        self.refs[result] = node_ref
        return result

    def op(
        self,
        op: str,
        operands: Iterable[str],
        result: str,
        dtype: str,
        shape: Iterable[int | str],
        *,
        attrs: dict[str, Any] | None = None,
    ) -> str:
        return self.add(_op(op, operands, result, dtype, shape, attrs=attrs))

    def const(self, result: str, dtype: str, value: Any) -> str:
        return self.add(_const(result, dtype, value))

    def node(self, result: str) -> str:
        return self.refs[result]


def _reduce(
    graph: _Graph,
    operand: str,
    result: str,
    shape: Iterable[int | str],
    *,
    fn: str,
    axes: list[int],
) -> str:
    return graph.op(
        "reduce",
        [operand],
        result,
        "f32",
        shape,
        attrs={"fn": fn, "axes": axes, "assoc": "float_nondeterministic"},
    )


def _reduction(graph: _Graph, result: str, identity: Any) -> dict[str, Any]:
    return {"node_ref": graph.node(result), "tile_axis": 0, "identity": identity}


def _row_reduce_j(fn: str) -> dict[str, Any]:
    graph = _Graph()
    _reduce(graph, "t0", "t1", ["x0"], fn=fn, axes=[1])
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16])],
        outputs=[_value("t1", "f32", ["x0"])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", graph.node("t1")),
            _access("a1", "store", "t1", "[k0]", graph.node("t1")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 16)",
        },
        reductions=[_reduction(graph, "t1", 0.0 if fn == "sum" else "-inf")],
    )


def _group_selector(
    graph: _Graph,
    *,
    width: int,
    groups: int,
    group_size: int,
    prefix: str,
    broadcasts: list[dict[str, Any]],
) -> str:
    channel = graph.op(
        "iota", [], f"{prefix}c", "i32", [width], attrs={"n": width, "dtype": "i32"}
    )
    divisor = graph.const(f"{prefix}divisor", "i32", group_size)
    grouped = graph.op(
        "map_binary",
        [channel, divisor],
        f"{prefix}group_of_c",
        "i32",
        [width],
        attrs={"fn": "div"},
    )
    broadcasts.append(
        {"operand": divisor, "node_ref": graph.node(grouped), "axes": [0]}
    )
    group = graph.op(
        "iota", [], f"{prefix}g", "i32", [groups], attrs={"n": groups, "dtype": "i32"}
    )
    predicate = graph.op(
        "map_binary",
        [grouped, group],
        f"{prefix}selector_i1",
        "i1",
        [width, groups],
        attrs={"fn": "eq"},
    )
    broadcasts.extend(
        [
            {"operand": grouped, "node_ref": graph.node(predicate), "axes": [1]},
            {"operand": group, "node_ref": graph.node(predicate), "axes": [0]},
        ]
    )
    return graph.op(
        "cast",
        [predicate],
        f"{prefix}selector",
        "f32",
        [width, groups],
        attrs={"to_dtype": "f32"},
    )


def _grouped_row_sum_j() -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    selector = _group_selector(
        graph,
        width=64,
        groups=16,
        group_size=4,
        prefix="v",
        broadcasts=broadcasts,
    )
    graph.op(
        "contraction",
        ["t0", selector],
        "t1",
        "f32",
        ["x0", 16],
        attrs={"subscripts": "mc,cg->mg"},
    )
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 64])],
        outputs=[_value("t1", "f32", ["x0", 16])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", graph.node("t1")),
            _access("a1", "store", "t1", "[k0, k2]", graph.node("t1")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 64)",
            "d2": "(0 <= k2) and (k2 < 16)",
        },
        broadcasts=broadcasts,
    )


def _softmax_inner_j() -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    _reduce(graph, "t0", "t2", ["x0"], fn="max", axes=[1])
    graph.op("map_binary", ["t0", "t2"], "v0", "f32", ["x0", 16], attrs={"fn": "sub"})
    broadcasts.append({"operand": "t2", "node_ref": graph.node("v0"), "axes": [1]})
    graph.op("map_unary", ["v0"], "v1", "f32", ["x0", 16], attrs={"fn": "exp"})
    _reduce(graph, "v1", "t3", ["x0"], fn="sum", axes=[1])
    graph.op("map_binary", ["v1", "t3"], "t1", "f32", ["x0", 16], attrs={"fn": "div"})
    broadcasts.append({"operand": "t3", "node_ref": graph.node("t1"), "axes": [1]})
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16])],
        outputs=[
            _value("t1", "f32", ["x0", 16]),
            _value("t2", "f32", ["x0"]),
            _value("t3", "f32", ["x0"]),
        ],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", graph.node("t2")),
            _access("a1", "store", "t2", "[k0]", graph.node("t2")),
            _access("a2", "store", "t3", "[k0]", graph.node("t3")),
            _access("a3", "store", "t1", "[k0, k1]", graph.node("t1")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 16)",
        },
        broadcasts=broadcasts,
        reductions=[
            _reduction(graph, "t2", "-inf"),
            _reduction(graph, "t3", 0.0),
        ],
    )


def _masked_softmax_j(frontend: str) -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    mask_dtype = "i1" if frontend == "tilelang" else "i8"
    mask = "t1"
    if mask_dtype != "i1":
        mask = graph.op("cast", [mask], "v0", "i1", [64], attrs={"to_dtype": "i1"})
    mask_f32 = graph.op("cast", [mask], "v1", "f32", [64], attrs={"to_dtype": "f32"})
    mask_load_node = graph.node(mask) if mask != "t1" else graph.node(mask_f32)
    one = graph.const("v2", "f32", 1.0)
    complement = graph.op("map_binary", [one, mask_f32], "v3", "f32", [64], attrs={"fn": "sub"})
    broadcasts.append({"operand": one, "node_ref": graph.node(complement), "axes": [0]})
    kept = graph.op("map_binary", ["t0", mask_f32], "v4", "f32", ["x0", 64], attrs={"fn": "mul"})
    broadcasts.append({"operand": mask_f32, "node_ref": graph.node(kept), "axes": [0]})
    neg = graph.const("v5", "f32", -1.0e9)
    rejected = graph.op("map_binary", [neg, complement], "v6", "f32", [64], attrs={"fn": "mul"})
    broadcasts.append({"operand": neg, "node_ref": graph.node(rejected), "axes": [0]})
    masked = graph.op("map_binary", [kept, rejected], "v7", "f32", ["x0", 64], attrs={"fn": "add"})
    broadcasts.append({"operand": rejected, "node_ref": graph.node(masked), "axes": [0]})
    _reduce(graph, masked, "v8", ["x0"], fn="max", axes=[1])
    centered = graph.op("map_binary", [masked, "v8"], "v9", "f32", ["x0", 64], attrs={"fn": "sub"})
    broadcasts.append({"operand": "v8", "node_ref": graph.node(centered), "axes": [1]})
    exp = graph.op("map_unary", [centered], "v10", "f32", ["x0", 64], attrs={"fn": "exp"})
    _reduce(graph, exp, "v11", ["x0"], fn="sum", axes=[1])
    graph.op("map_binary", [exp, "v11"], "t2", "f32", ["x0", 64], attrs={"fn": "div"})
    broadcasts.append({"operand": "v11", "node_ref": graph.node("t2"), "axes": [1]})
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 64]), _value("t1", mask_dtype, [64])],
        outputs=[_value("t2", "f32", ["x0", 64])],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t1", "[k1]", mask_load_node),
            _access("a1", "load", "t0", "[k0, k1]", graph.node(kept)),
            _access("a2", "store", "t2", "[k0, k1]", graph.node("t2")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 64)",
        },
        broadcasts=broadcasts,
        reductions=[
            _reduction(graph, "v8", "-inf"),
            _reduction(graph, "v11", 0.0),
        ],
    )


def _groupnorm_j() -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    selector = _group_selector(
        graph,
        width=64,
        groups=4,
        group_size=16,
        prefix="v",
        broadcasts=broadcasts,
    )
    graph.op(
        "contraction", ["t0", selector], "vsum", "f32", ["x0", 4],
        attrs={"subscripts": "nch,cg->ng"},
    )
    graph.op("map_binary", ["t0", "t0"], "vsq", "f32", ["x0", 64, 16], attrs={"fn": "mul"})
    graph.op(
        "contraction", ["vsq", selector], "vsum2", "f32", ["x0", 4],
        attrs={"subscripts": "nch,cg->ng"},
    )
    inv = graph.const("vinv", "f32", 1.0 / 256.0)
    graph.op("map_binary", ["vsum", inv], "t4", "f32", ["x0", 4], attrs={"fn": "mul"})
    broadcasts.append({"operand": inv, "node_ref": graph.node("t4"), "axes": [0, 1]})
    graph.op("map_binary", ["vsum2", inv], "vmean2", "f32", ["x0", 4], attrs={"fn": "mul"})
    broadcasts.append({"operand": inv, "node_ref": graph.node("vmean2"), "axes": [0, 1]})
    graph.op("map_binary", ["t4", "t4"], "vmean_sq", "f32", ["x0", 4], attrs={"fn": "mul"})
    graph.op("map_binary", ["vmean2", "vmean_sq"], "vvar", "f32", ["x0", 4], attrs={"fn": "sub"})
    eps = graph.const("veps", "f32", 1.0e-5)
    graph.op("map_binary", ["vvar", eps], "vvar_eps", "f32", ["x0", 4], attrs={"fn": "add"})
    broadcasts.append({"operand": eps, "node_ref": graph.node("vvar_eps"), "axes": [0, 1]})
    graph.op("map_unary", ["vvar_eps"], "t5", "f32", ["x0", 4], attrs={"fn": "rsqrt"})
    graph.op(
        "contraction", ["t4", selector], "vmean_c", "f32", ["x0", 64],
        attrs={"subscripts": "ng,cg->nc"},
    )
    graph.op(
        "contraction", ["t5", selector], "vrstd_c", "f32", ["x0", 64],
        attrs={"subscripts": "ng,cg->nc"},
    )
    graph.op("map_binary", ["t0", "vmean_c"], "vdiff", "f32", ["x0", 64, 16], attrs={"fn": "sub"})
    broadcasts.append({"operand": "vmean_c", "node_ref": graph.node("vdiff"), "axes": [2]})
    graph.op("map_binary", ["vdiff", "vrstd_c"], "vnorm", "f32", ["x0", 64, 16], attrs={"fn": "mul"})
    broadcasts.append({"operand": "vrstd_c", "node_ref": graph.node("vnorm"), "axes": [2]})
    graph.op("map_binary", ["vnorm", "t1"], "vscaled", "f32", ["x0", 64, 16], attrs={"fn": "mul"})
    broadcasts.append({"operand": "t1", "node_ref": graph.node("vscaled"), "axes": [0, 2]})
    graph.op("map_binary", ["vscaled", "t2"], "t3", "f32", ["x0", 64, 16], attrs={"fn": "add"})
    broadcasts.append({"operand": "t2", "node_ref": graph.node("t3"), "axes": [0, 2]})
    return _make_j(
        inputs=[
            _value("t0", "f32", ["x0", 64, 16]),
            _value("t1", "f32", [64]),
            _value("t2", "f32", [64]),
        ],
        outputs=[
            _value("t3", "f32", ["x0", 64, 16]),
            _value("t4", "f32", ["x0", 4]),
            _value("t5", "f32", ["x0", 4]),
        ],
        body=graph.body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1, k2]", graph.node("vsum")),
            _access("a1", "load", "t1", "[k1]", graph.node("vscaled")),
            _access("a2", "load", "t2", "[k1]", graph.node("t3")),
            _access("a3", "store", "t4", "[k0, k3]", graph.node("t4")),
            _access("a4", "store", "t5", "[k0, k3]", graph.node("t5")),
            _access("a5", "store", "t3", "[k0, k1, k2]", graph.node("t3")),
        ],
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 64)",
            "d2": "(0 <= k2) and (k2 < 16)",
            "d3": "(0 <= k3) and (k3 < 4)",
        },
        broadcasts=broadcasts,
    )


def _layernorm_j(*, width: int, residual: bool) -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    if residual:
        input_refs = [
            _value("t0", "f32", ["x0", width]),
            _value("t1", "f32", ["x0", width]),
            _value("t2", "f32", [width]),
            _value("t3", "f32", [width]),
        ]
        output_refs = [
            _value("t4", "f32", ["x0", width]),
            _value("t5", "f32", ["x0"]),
            _value("t6", "f32", ["x0"]),
        ]
        data = graph.op("map_binary", ["t0", "t1"], "vdata", "f32", ["x0", width], attrs={"fn": "add"})
        weight, bias, out, mean, rstd = "t2", "t3", "t4", "t5", "t6"
    else:
        input_refs = [
            _value("t0", "f32", ["x0", width]),
            _value("t1", "f32", [width]),
            _value("t2", "f32", [width]),
        ]
        output_refs = [
            _value("t3", "f32", ["x0", width]),
            _value("t4", "f32", ["x0"]),
            _value("t5", "f32", ["x0"]),
        ]
        data = "t0"
        weight, bias, out, mean, rstd = "t1", "t2", "t3", "t4", "t5"
    _reduce(graph, data, "vsum", ["x0"], fn="sum", axes=[1])
    inv = graph.const("vinv", "f32", 1.0 / float(width))
    graph.op("map_binary", ["vsum", inv], mean, "f32", ["x0"], attrs={"fn": "mul"})
    broadcasts.append({"operand": inv, "node_ref": graph.node(mean), "axes": [0]})
    graph.op("map_binary", [data, mean], "vdiff", "f32", ["x0", width], attrs={"fn": "sub"})
    broadcasts.append({"operand": mean, "node_ref": graph.node("vdiff"), "axes": [1]})
    graph.op("map_binary", ["vdiff", "vdiff"], "vsq", "f32", ["x0", width], attrs={"fn": "mul"})
    _reduce(graph, "vsq", "vsum2", ["x0"], fn="sum", axes=[1])
    graph.op("map_binary", ["vsum2", inv], "vvar", "f32", ["x0"], attrs={"fn": "mul"})
    broadcasts.append({"operand": inv, "node_ref": graph.node("vvar"), "axes": [0]})
    eps = graph.const("veps", "f32", 1.0e-5)
    graph.op("map_binary", ["vvar", eps], "vvar_eps", "f32", ["x0"], attrs={"fn": "add"})
    broadcasts.append({"operand": eps, "node_ref": graph.node("vvar_eps"), "axes": [0]})
    graph.op("map_unary", ["vvar_eps"], rstd, "f32", ["x0"], attrs={"fn": "rsqrt"})
    graph.op("map_binary", ["vdiff", rstd], "vnorm", "f32", ["x0", width], attrs={"fn": "mul"})
    broadcasts.append({"operand": rstd, "node_ref": graph.node("vnorm"), "axes": [1]})
    graph.op("map_binary", ["vnorm", weight], "vscaled", "f32", ["x0", width], attrs={"fn": "mul"})
    broadcasts.append({"operand": weight, "node_ref": graph.node("vscaled"), "axes": [0]})
    graph.op("map_binary", ["vscaled", bias], out, "f32", ["x0", width], attrs={"fn": "add"})
    broadcasts.append({"operand": bias, "node_ref": graph.node(out), "axes": [0]})
    accesses = [
        _access("a0", "load", "t0", "[k0, k1]", graph.node(data) if residual else graph.node("vsum")),
    ]
    if residual:
        accesses.append(_access("a1", "load", "t1", "[k0, k1]", graph.node(data)))
    accesses.extend(
        [
            _access(f"a{len(accesses)}", "load", weight, "[k1]", graph.node("vscaled")),
            _access(f"a{len(accesses) + 1}", "load", bias, "[k1]", graph.node(out)),
            _access(f"a{len(accesses) + 2}", "store", mean, "[k0]", graph.node(mean)),
            _access(f"a{len(accesses) + 3}", "store", rstd, "[k0]", graph.node(rstd)),
            _access(f"a{len(accesses) + 4}", "store", out, "[k0, k1]", graph.node(out)),
        ]
    )
    return _make_j(
        inputs=input_refs,
        outputs=output_refs,
        body=graph.body,
        accesses=accesses,
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": f"(0 <= k1) and (k1 < {width})",
        },
        broadcasts=broadcasts,
        reductions=[
            _reduction(graph, "vsum", 0.0),
            _reduction(graph, "vsum2", 0.0),
        ],
    )


def _rmsnorm_j(*, residual: bool) -> dict[str, Any]:
    graph = _Graph()
    broadcasts: list[dict[str, Any]] = []
    if residual:
        inputs = [
            _value("t0", "f32", ["x0", 64]),
            _value("t1", "f32", ["x0", 64]),
            _value("t2", "f32", [64]),
            _value("t3", "f32", [64]),
        ]
        outputs = [_value("t4", "f32", ["x0", 64]), _value("t5", "f32", ["x0"])]
        first = graph.op("map_binary", ["t0", "t1"], "vsum_input", "f32", ["x0", 64], attrs={"fn": "add"})
        data = graph.op("map_binary", [first, "t3"], "vdata", "f32", ["x0", 64], attrs={"fn": "add"})
        broadcasts.append({"operand": "t3", "node_ref": graph.node(data), "axes": [0]})
        weight, out, rstd = "t2", "t4", "t5"
    else:
        inputs = [_value("t0", "f32", ["x0", 64]), _value("t1", "f32", [64])]
        outputs = [_value("t2", "f32", ["x0", 64]), _value("t3", "f32", ["x0"])]
        data, weight, out, rstd = "t0", "t1", "t2", "t3"
    graph.op("map_binary", [data, data], "vsq", "f32", ["x0", 64], attrs={"fn": "mul"})
    _reduce(graph, "vsq", "vsum_sq", ["x0"], fn="sum", axes=[1])
    inv = graph.const("vinv", "f32", 1.0 / 64.0)
    graph.op("map_binary", ["vsum_sq", inv], "vmean_sq", "f32", ["x0"], attrs={"fn": "mul"})
    broadcasts.append({"operand": inv, "node_ref": graph.node("vmean_sq"), "axes": [0]})
    eps = graph.const("veps", "f32", 1.0e-5)
    graph.op("map_binary", ["vmean_sq", eps], "vmean_eps", "f32", ["x0"], attrs={"fn": "add"})
    broadcasts.append({"operand": eps, "node_ref": graph.node("vmean_eps"), "axes": [0]})
    graph.op("map_unary", ["vmean_eps"], rstd, "f32", ["x0"], attrs={"fn": "rsqrt"})
    graph.op("map_binary", [data, rstd], "vnorm", "f32", ["x0", 64], attrs={"fn": "mul"})
    broadcasts.append({"operand": rstd, "node_ref": graph.node("vnorm"), "axes": [1]})
    graph.op("map_binary", ["vnorm", weight], out, "f32", ["x0", 64], attrs={"fn": "mul"})
    broadcasts.append({"operand": weight, "node_ref": graph.node(out), "axes": [0]})
    accesses = [_access("a0", "load", "t0", "[k0, k1]", graph.node(data) if residual else graph.node("vsq"))]
    if residual:
        accesses.extend(
            [
                _access("a1", "load", "t1", "[k0, k1]", graph.node("vsum_input")),
                _access("a2", "load", "t3", "[k1]", graph.node(data)),
            ]
        )
    accesses.extend(
        [
            _access(f"a{len(accesses)}", "load", weight, "[k1]", graph.node(out)),
            _access(f"a{len(accesses) + 1}", "store", rstd, "[k0]", graph.node(rstd)),
            _access(f"a{len(accesses) + 2}", "store", out, "[k0, k1]", graph.node(out)),
        ]
    )
    return _make_j(
        inputs=inputs,
        outputs=outputs,
        body=graph.body,
        accesses=accesses,
        domains={
            "d0": "(0 <= k0) and (k0 < x0)",
            "d1": "(0 <= k1) and (k1 < 64)",
        },
        broadcasts=broadcasts,
        reductions=[_reduction(graph, "vsum_sq", 0.0)],
    )


_J_BUILDERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "row_sum": lambda _frontend: _row_reduce_j("sum"),
    "row_max": lambda _frontend: _row_reduce_j("max"),
    "grouped_row_sum2d": lambda _frontend: _grouped_row_sum_j(),
    "softmax_inner": lambda _frontend: _softmax_inner_j(),
    "masked_softmax2d": _masked_softmax_j,
    "groupnorm": lambda _frontend: _groupnorm_j(),
    "layernorm": lambda _frontend: _layernorm_j(width=16, residual=False),
    "rms_norm2d": lambda _frontend: _rmsnorm_j(residual=False),
    "layer_norm_residual2d": lambda _frontend: _layernorm_j(width=64, residual=True),
    "rms_norm_residual2d": lambda _frontend: _rmsnorm_j(residual=True),
}


_BLOCK_X = {
    "row_sum": 128,
    "row_max": 128,
    "grouped_row_sum2d": 128,
    "softmax_inner": 128,
    "masked_softmax2d": 64,
    "groupnorm": 128,
    "layernorm": 128,
    "rms_norm2d": 64,
    "layer_norm_residual2d": 64,
    "rms_norm_residual2d": 64,
}


def _p(frontend: str, ref: str, index: int, dtype: str, rank: int, mode: str) -> dict[str, Any]:
    return _tensor_parameter(
        ref, name=f"p{index}", dtype=dtype, rank=rank, mode=mode, frontend=frontend
    )


def _signature(stem: str, frontend: str) -> dict[str, Any]:
    logical: dict[str, tuple[str, int, str]]
    order: list[str]
    if stem in {"row_sum", "row_max", "grouped_row_sum2d"}:
        output_rank = 1 if stem in {"row_sum", "row_max"} else 2
        logical = {"t0": ("f32", 2, "input"), "t1": ("f32", output_rank, "output")}
        order = ["t0", "t1"]
    elif stem == "softmax_inner":
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 2, "output"),
            "t2": ("f32", 1, "output"), "t3": ("f32", 1, "output"),
        }
        order = ["t0", "t1", "t2", "t3"]
    elif stem == "masked_softmax2d":
        mask_dtype = "i1" if frontend == "tilelang" else "i8"
        logical = {
            "t0": ("f32", 2, "input"), "t1": (mask_dtype, 1, "input"),
            "t2": ("f32", 2, "output"),
        }
        order = ["t0", "t1", "t2"]
    elif stem == "groupnorm":
        logical = {
            "t0": ("f32", 3, "input"), "t1": ("f32", 1, "input"),
            "t2": ("f32", 1, "input"), "t3": ("f32", 3, "output"),
            "t4": ("f32", 2, "output"), "t5": ("f32", 2, "output"),
        }
        order = ["t0", "t3", "t1", "t2", "t4", "t5"] if frontend == "tilelang" else ["t2", "t4", "t5", "t1", "t0", "t3"]
    elif stem == "layernorm":
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 1, "input"),
            "t2": ("f32", 1, "input"), "t3": ("f32", 2, "output"),
            "t4": ("f32", 1, "output"), "t5": ("f32", 1, "output"),
        }
        order = ["t0", "t3", "t1", "t2", "t4", "t5"] if frontend == "tilelang" else ["t2", "t0", "t4", "t3", "t5", "t1"]
    elif stem == "rms_norm2d":
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 1, "input"),
            "t2": ("f32", 2, "output"), "t3": ("f32", 1, "output"),
        }
        order = ["t0", "t1", "t2", "t3"] if frontend == "tilelang" else ["t0", "t2", "t3", "t1"]
    elif stem == "layer_norm_residual2d":
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 2, "input"),
            "t2": ("f32", 1, "input"), "t3": ("f32", 1, "input"),
            "t4": ("f32", 2, "output"), "t5": ("f32", 1, "output"),
            "t6": ("f32", 1, "output"),
        }
        order = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] if frontend == "tilelang" else ["t3", "t0", "t5", "t4", "t1", "t6", "t2"]
    elif stem == "rms_norm_residual2d":
        logical = {
            "t0": ("f32", 2, "input"), "t1": ("f32", 2, "input"),
            "t2": ("f32", 1, "input"), "t3": ("f32", 1, "input"),
            "t4": ("f32", 2, "output"), "t5": ("f32", 1, "output"),
        }
        order = ["t0", "t1", "t2", "t3", "t4", "t5"] if frontend == "tilelang" else ["t3", "t0", "t4", "t1", "t5", "t2"]
    else:  # pragma: no cover - closed by G2_STEMS
        raise KeyError(stem)
    parameters = [
        _p(frontend, ref, index, *logical[ref]) for index, ref in enumerate(order)
    ]
    if frontend == "cuda":
        parameters.append(
            _parameter(
                "x0", name=f"p{len(parameters)}", kind="scalar", dtype="i32",
                rank=0, mode="value",
            )
        )
    return {"parameters": parameters}


def _compact_pi(stem: str) -> dict[str, Any]:
    if stem == "grouped_row_sum2d":
        axes = [{"domain_ref": "d2", "axis": 0}, {"domain_ref": "d0", "axis": 1}]
    elif stem == "groupnorm":
        axes = [{"domain_ref": "d0", "axis": 0}, {"domain_ref": "d3", "axis": 1}]
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
            "slots": [{
                "name": "c0", "kind": "compiler", "value_type": "i32",
                "role": "launch_block_x", "required": True,
            }],
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
        "pi": _compact_pi(stem),
        "claim_scope": (
            "Shared sections are measured structural equality, not an a-priori "
            "requirement. Each frontend artifact retains its exact ABI and is "
            "independently TypeChecked and Gate-certified."
        ),
    }


CASE_IDS = tuple(
    case_id
    for stem in G2_STEMS
    for case_id in (f"tilelang:{stem}", f"cuda:tilelang_{stem}")
)
CASE_BUILDERS = {
    case_id: (lambda selected=case_id: build_artifact(selected)) for case_id in CASE_IDS
}
CASE_RECEIPTS = {
    case_id: (lambda selected=case_id: authoring_receipt(selected)) for case_id in CASE_IDS
}


__all__ = [
    "CASE_BUILDERS", "CASE_IDS", "CASE_RECEIPTS", "G2_STEMS",
    "authoring_receipt", "build_artifact",
]
