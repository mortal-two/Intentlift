"""Manual, Source-audited J/C/pi authoring for Task28 cohort G1.

This module is experiment evidence, not a parser, a Lift agent, or a persistent
kernel lowering table.  It factors only the closed J wire mechanics that are
identical across these manually authored cases.  Replay never imports this file
and receives exactly J + MaskC(C) + public frontend context; pi remains Gate-only.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable


ARTIFACT_SCHEMA = "intentir.scrc.artifact.v13"
PI_SCHEMA = "intentir.pi.normalization-witness.v1"
G1_STEMS = (
    "add2d",
    "relu2d",
    "floor2d",
    "exp2d",
    "where2d",
    "add_bias2d",
    "clamp2d",
    "rowmask_where2d",
    "any_kernel_dim",
    "copy2d_divmod",
)


def _value(name: str, dtype: str, shape: Iterable[int | str]) -> dict[str, Any]:
    return {"name": name, "dtype": dtype, "shape": list(shape)}


def _op(
    op: str,
    operands: Iterable[str],
    result: str,
    dtype: str,
    shape: Iterable[int | str],
    *,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "op": op,
        "operands": list(operands),
        "result": result,
        "type": {"dtype": dtype, "shape": list(shape)},
    }
    if attrs is not None:
        row["attrs"] = attrs
    return row


def _const(result: str, dtype: str, value: Any) -> dict[str, Any]:
    return {
        "op": "const",
        "result": result,
        "type": {"dtype": dtype, "shape": []},
        "attrs": {"dtype": dtype, "shape": [], "value": value},
    }


def _access(
    access_id: str,
    kind: str,
    tensor: str,
    index_expression: str,
    node_ref: str,
) -> dict[str, Any]:
    return {
        "access_id": access_id,
        "kind": kind,
        "tensor": tensor,
        "index_expression": index_expression,
        "node_ref": node_ref,
    }


def _make_j(
    *,
    inputs: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    body: list[dict[str, Any]],
    accesses: list[dict[str, Any]],
    domains: dict[str, str],
    broadcasts: list[dict[str, Any]] | None = None,
    reductions: list[dict[str, Any]] | None = None,
    result_kind: str = "tolerance",
) -> dict[str, Any]:
    node_ids = {f"n{index}": f"body[{index}]" for index in range(len(body))}
    effects = [
        {
            "effect_id": f"e{index}",
            "kind": "read" if access["kind"] == "load" else "write",
            "target": access["tensor"],
            "node_ref": access["node_ref"],
            "access_ref": access["access_id"],
        }
        for index, access in enumerate(accesses)
    ]
    definitions: list[str] = []
    for name in [
        *[item["name"] for item in inputs],
        *[item["name"] for item in outputs],
        *[item["result"] for item in body if isinstance(item.get("result"), str)],
    ]:
        if name not in definitions:
            definitions.append(name)
    return {
        "kernel_id": "scrc_k0",
        "intent": {
            "intent_function": {
                "name": "scrc_j0",
                "inputs": inputs,
                "outputs": outputs,
                "body": body,
            }
        },
        "node_ids": node_ids,
        "boundary": {
            "domains": domains,
            "accesses": accesses,
            "effects": effects,
            "aliases": [],
            "broadcasts": broadcasts or [],
            "reductions": reductions or [],
            "updates": [],
            "output_mappings": [],
            "index_bounds": [],
            "consumer_extensions": [],
            "access_guard_relations": [],
            "result_relations": [
                {"output": output["name"], "kind": result_kind}
                for output in outputs
            ],
            "realization_authorizations": [],
        },
        "scopes": [
            {
                "scope_id": "q0",
                "kind": "function",
                "parent_scope": None,
                "captures": [],
                "definitions": definitions,
                "effects": [item["effect_id"] for item in effects],
            }
        ],
        "storage": [],
        "dimension_constraints": ["x0 >= 1"],
        "value_dimension_relations": [],
        "dimension_relations": [],
    }


def _domains_2d(width: int) -> dict[str, str]:
    return {
        "d0": "(0 <= k0) and (k0 < x0)",
        "d1": f"(0 <= k1) and (k1 < {width})",
    }


def _simple_2d_accesses(
    input_nodes: list[tuple[str, str]], output: str, output_node: str, width: int
) -> list[dict[str, Any]]:
    rows = [
        _access(f"a{index}", "load", tensor, "[k0, k1]", node_ref)
        for index, (tensor, node_ref) in enumerate(input_nodes)
    ]
    rows.append(
        _access(f"a{len(rows)}", "store", output, "[k0, k1]", output_node)
    )
    return rows


def _add2d_j() -> dict[str, Any]:
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16]), _value("t1", "f32", ["x0", 16])],
        outputs=[_value("t2", "f32", ["x0", 16])],
        body=[_op("map_binary", ["t0", "t1"], "t2", "f32", ["x0", 16], attrs={"fn": "add"})],
        accesses=_simple_2d_accesses([("t0", "n0"), ("t1", "n0")], "t2", "n0", 16),
        domains=_domains_2d(16),
    )


def _unary_2d_j(fn: str) -> dict[str, Any]:
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16])],
        outputs=[_value("t1", "f32", ["x0", 16])],
        body=[_op("map_unary", ["t0"], "t1", "f32", ["x0", 16], attrs={"fn": fn})],
        accesses=_simple_2d_accesses([("t0", "n0")], "t1", "n0", 16),
        domains=_domains_2d(16),
    )


def _relu2d_j() -> dict[str, Any]:
    body = [
        _const("v0", "f32", 0.0),
        _op("map_binary", ["t0", "v0"], "t1", "f32", ["x0", 16], attrs={"fn": "max"}),
    ]
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16])],
        outputs=[_value("t1", "f32", ["x0", 16])],
        body=body,
        accesses=_simple_2d_accesses([("t0", "n1")], "t1", "n1", 16),
        domains=_domains_2d(16),
        broadcasts=[{"operand": "v0", "node_ref": "n1", "axes": [0, 1]}],
    )


def _where2d_j() -> dict[str, Any]:
    body = [
        _op("map_binary", ["t0", "t1"], "v0", "i1", ["x0", 16], attrs={"fn": "gt"}),
        _op("select", ["v0", "t0", "t1"], "t2", "f32", ["x0", 16]),
    ]
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16]), _value("t1", "f32", ["x0", 16])],
        outputs=[_value("t2", "f32", ["x0", 16])],
        body=body,
        accesses=_simple_2d_accesses([("t0", "n0"), ("t1", "n0")], "t2", "n1", 16),
        domains=_domains_2d(16),
    )


def _add_bias2d_j() -> dict[str, Any]:
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16]), _value("t1", "f32", [16])],
        outputs=[_value("t2", "f32", ["x0", 16])],
        body=[_op("map_binary", ["t0", "t1"], "t2", "f32", ["x0", 16], attrs={"fn": "add"})],
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", "n0"),
            _access("a1", "load", "t1", "[k1]", "n0"),
            _access("a2", "store", "t2", "[k0, k1]", "n0"),
        ],
        domains=_domains_2d(16),
        broadcasts=[{"operand": "t1", "node_ref": "n0", "axes": [0]}],
    )


def _clamp2d_j() -> dict[str, Any]:
    body = [
        _op("map_binary", ["t0", "t1"], "v0", "f32", ["x0", 16], attrs={"fn": "max"}),
        _op("map_binary", ["v0", "t2"], "t3", "f32", ["x0", 16], attrs={"fn": "min"}),
    ]
    return _make_j(
        inputs=[
            _value("t0", "f32", ["x0", 16]),
            _value("t1", "f32", [1]),
            _value("t2", "f32", [1]),
        ],
        outputs=[_value("t3", "f32", ["x0", 16])],
        body=body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", "n0"),
            _access("a1", "load", "t1", "[0]", "n0"),
            _access("a2", "load", "t2", "[0]", "n1"),
            _access("a3", "store", "t3", "[k0, k1]", "n1"),
        ],
        domains=_domains_2d(16),
        broadcasts=[
            {"operand": "t1", "node_ref": "n0", "axes": [0, 1]},
            {"operand": "t2", "node_ref": "n1", "axes": [0, 1]},
        ],
    )


def _rowmask_j(frontend: str) -> dict[str, Any]:
    mask_dtype = "i1" if frontend == "tilelang" else "i8"
    body: list[dict[str, Any]] = []
    mask_value = "t1"
    if mask_dtype != "i1":
        body.append(_op("cast", ["t1"], "v0", "i1", ["x0"], attrs={"to_dtype": "i1"}))
        mask_value = "v0"
    cast_result = f"v{len(body)}"
    body.append(_op("cast", [mask_value], cast_result, "f32", ["x0"], attrs={"to_dtype": "f32"}))
    multiply_node = f"n{len(body)}"
    body.append(
        _op("map_binary", ["t0", cast_result], "t2", "f32", ["x0", 64], attrs={"fn": "mul"})
    )
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 64]), _value("t1", mask_dtype, ["x0"])],
        outputs=[_value("t2", "f32", ["x0", 64])],
        body=body,
        accesses=[
            _access("a0", "load", "t1", "[k0]", "n0"),
            _access("a1", "load", "t0", "[k0, k1]", multiply_node),
            _access("a2", "store", "t2", "[k0, k1]", multiply_node),
        ],
        domains=_domains_2d(64),
        broadcasts=[{"operand": cast_result, "node_ref": multiply_node, "axes": [1]}],
    )


def _any_kernel_dim_j(frontend: str) -> dict[str, Any]:
    body = [
        _op(
            "reduce",
            ["t0"],
            "v0",
            "f32",
            ["x0"],
            attrs={"fn": "max", "axes": [1], "assoc": "float_nondeterministic"},
        ),
        _const("v1", "f32", 0.0),
        _op("map_binary", ["v0", "v1"], "v2", "i1", ["x0"], attrs={"fn": "ne"}),
    ]
    output_dtype = "i1"
    output_value = "v2"
    if frontend == "cuda":
        output_dtype = "i8"
        output_value = "t1"
        body.append(_op("cast", ["v2"], output_value, "i8", ["x0"], attrs={"to_dtype": "i8"}))
    else:
        body[-1]["result"] = "t1"
        output_value = "t1"
    output_node = f"n{len(body) - 1}"
    return _make_j(
        inputs=[_value("t0", "f32", ["x0", 16])],
        outputs=[_value("t1", output_dtype, ["x0"])],
        body=body,
        accesses=[
            _access("a0", "load", "t0", "[k0, k1]", "n0"),
            _access("a1", "store", "t1", "[k0]", output_node),
        ],
        domains=_domains_2d(16),
        broadcasts=[{"operand": "v1", "node_ref": "n2", "axes": [0]}],
        reductions=[{"node_ref": "n0", "tile_axis": 0, "identity": "-inf"}],
        result_kind="exact",
    )


_J_BUILDERS: dict[str, Callable[[str], dict[str, Any]]] = {
    "add2d": lambda _frontend: _add2d_j(),
    "relu2d": lambda _frontend: _relu2d_j(),
    "floor2d": lambda _frontend: _unary_2d_j("floor"),
    "exp2d": lambda _frontend: _unary_2d_j("exp"),
    "where2d": lambda _frontend: _where2d_j(),
    "add_bias2d": lambda _frontend: _add_bias2d_j(),
    "clamp2d": lambda _frontend: _clamp2d_j(),
    "rowmask_where2d": _rowmask_j,
    "any_kernel_dim": _any_kernel_dim_j,
    "copy2d_divmod": lambda _frontend: _unary_2d_j("identity"),
}


_BLOCK_X = {
    "add2d": 16,
    "relu2d": 16,
    "floor2d": 16,
    "exp2d": 16,
    "where2d": 16,
    "add_bias2d": 16,
    "clamp2d": 16,
    "rowmask_where2d": 64,
    "any_kernel_dim": 128,
    "copy2d_divmod": 128,
}


def _parameter(
    logical_ref: str,
    *,
    name: str,
    kind: str,
    dtype: str,
    rank: int,
    mode: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "dtype": dtype,
        "rank": rank,
        "mode": mode,
        "logical_ref": logical_ref,
    }


def _tensor_parameter(
    logical_ref: str, *, name: str, dtype: str, rank: int, mode: str, frontend: str
) -> dict[str, Any]:
    return _parameter(
        logical_ref,
        name=name,
        kind="tensor" if frontend == "tilelang" else "pointer",
        dtype=dtype,
        rank=rank,
        mode=mode,
    )


def _signature(stem: str, frontend: str) -> dict[str, Any]:
    p = lambda ref, index, dtype, rank, mode: _tensor_parameter(  # noqa: E731
        ref, name=f"p{index}", dtype=dtype, rank=rank, mode=mode, frontend=frontend
    )
    if stem == "add2d":
        parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, "f32", 2, "input"), p("t2", 2, "f32", 2, "output")]
    elif stem in {"relu2d", "floor2d", "exp2d", "copy2d_divmod"}:
        parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, "f32", 2, "output")]
    elif stem == "where2d":
        parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, "f32", 2, "input"), p("t2", 2, "f32", 2, "output")]
    elif stem == "add_bias2d":
        if frontend == "tilelang":
            parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, "f32", 1, "input"), p("t2", 2, "f32", 2, "output")]
        else:
            parameters = [p("t1", 0, "f32", 1, "input"), p("t0", 1, "f32", 2, "input"), p("t2", 2, "f32", 2, "output")]
    elif stem == "clamp2d":
        if frontend == "tilelang":
            parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, "f32", 1, "input"), p("t2", 2, "f32", 1, "input"), p("t3", 3, "f32", 2, "output")]
        else:
            parameters = [p("t2", 0, "f32", 1, "input"), p("t0", 1, "f32", 2, "input"), p("t1", 2, "f32", 1, "input"), p("t3", 3, "f32", 2, "output")]
    elif stem == "rowmask_where2d":
        mask_dtype = "i1" if frontend == "tilelang" else "i8"
        if frontend == "tilelang":
            parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, mask_dtype, 1, "input"), p("t2", 2, "f32", 2, "output")]
        else:
            parameters = [p("t0", 0, "f32", 2, "input"), p("t2", 1, "f32", 2, "output"), p("t1", 2, mask_dtype, 1, "input")]
    elif stem == "any_kernel_dim":
        output_dtype = "i1" if frontend == "tilelang" else "i8"
        parameters = [p("t0", 0, "f32", 2, "input"), p("t1", 1, output_dtype, 1, "output")]
    else:  # pragma: no cover - closed by G1_STEMS and _J_BUILDERS
        raise KeyError(stem)
    if frontend == "cuda":
        parameters.append(
            _parameter("x0", name=f"p{len(parameters)}", kind="scalar", dtype="i32", rank=0, mode="value")
        )
    return {"parameters": parameters}


def _compact_pi() -> dict[str, Any]:
    return {
        "schema": PI_SCHEMA,
        "program_axes": [{"domain_ref": "d0", "axis": 0}],
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
                    "name": "c0",
                    "kind": "compiler",
                    "value_type": "i32",
                    "role": "launch_block_x",
                    "required": True,
                }
            ],
            "values": {"c0": _BLOCK_X[stem]},
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
        "pi": _compact_pi(),
        "claim_scope": (
            "Shared sections are measured structural equality, not an a-priori "
            "requirement. Each frontend artifact retains its exact ABI and is "
            "independently TypeChecked and Gate-certified."
        ),
    }


CASE_IDS = tuple(
    case_id
    for stem in G1_STEMS
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
    "G1_STEMS",
    "authoring_receipt",
    "build_artifact",
]
