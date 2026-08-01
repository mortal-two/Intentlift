"""Deterministic, parameterized, target-independent ``J -> S_J`` core.

Only the exact source-isolated J view is accepted.  The implementation never
receives pi, C, Source, Gate observations, a frontend identity, or historical
Replay state.  It reorganizes J into a declaration-order typed program that a
normal compiler agent can consume without reconstructing the graph from a flat
archive.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Mapping, Sequence

from .semantic_program import (
    SEMANTIC_MATERIALIZER_PROTOCOL,
    SEMANTIC_PROGRAM_SCHEMA,
    SemanticProgram,
)


_J_FIELDS = frozenset(
    {
        "kernel_id",
        "intent",
        "node_ids",
        "boundary",
        "signature",
        "scopes",
        "storage",
        "dimension_constraints",
        "value_dimension_relations",
        "dimension_relations",
    }
)
_DIMENSION_REF = re.compile(r"x[0-9]+")
_NODE_SCOPED_BOUNDARY_RELATIONS = (
    "accesses",
    "effects",
    "reductions",
    "updates",
    "broadcasts",
    "index_bounds",
    "output_mappings",
    "consumer_extensions",
    "access_guard_relations",
)


class SemanticMaterializationError(ValueError):
    """Fail-closed unsupported or malformed J semantic materialization."""

    code = "semantic_materializer_unsupported"

    def __init__(self, detail: str, *, path: str = "j") -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"{self.code}@{path}: {detail}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticMaterializationError("expected an object", path=path)
    return copy.deepcopy(dict(value))


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise SemanticMaterializationError("expected an array", path=path)
    return copy.deepcopy(value)


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise SemanticMaterializationError("expected a non-empty string", path=path)
    return value


def _results(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    if value is None:
        return []
    raise SemanticMaterializationError(
        "operation result must be a string, string array, or absent",
        path="j.intent.operation.result",
    )


def _contains_masked_semantics(j: Mapping[str, Any]) -> bool:
    intent = j.get("intent")
    function = intent.get("intent_function") if isinstance(intent, Mapping) else None
    body = function.get("body") if isinstance(function, Mapping) else None

    def walk(operations: Any) -> bool:
        if not isinstance(operations, list):
            return False
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            if operation.get("op") == "<masked>":
                return True
            if walk(operation.get("body")):
                return True
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if isinstance(arm, Mapping) and walk(arm.get("body")):
                    return True
        return False

    return walk(body)


def _shape_type(tensor: Mapping[str, Any], path: str) -> dict[str, Any]:
    name = _string(tensor.get("name"), f"{path}.name")
    dtype = tensor.get("dtype")
    shape = tensor.get("shape")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        raise SemanticMaterializationError(
            f"tensor {name!r} requires dtype and shape",
            path=path,
        )
    return {"dtype": dtype, "shape": copy.deepcopy(shape)}


def _inline(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _render_text(payload: Mapping[str, Any]) -> str:
    function = payload["function"]
    lines = [
        f"semantic_program {function['name']} mode={payload['mode']}",
        f"schema {payload['schema']}",
        "parameterized true",
        "logical_operations_imply_target_materialization false",
        "physical_materialization_authority pi only",
        "",
        "dimensions:",
    ]
    dimensions = payload["dimensions"]
    if dimensions:
        for dimension in dimensions:
            lines.append(f"  {dimension['ref']}: symbolic")
    else:
        lines.append("  <none>")

    lines.extend(["", "signature:"])
    parameters = function["signature"].get("parameters", [])
    if parameters:
        for parameter in parameters:
            lines.append(
                "  "
                + str(parameter.get("name"))
                + ": "
                + str(parameter.get("mode", "value"))
                + " "
                + str(parameter.get("kind"))
                + " dtype="
                + str(parameter.get("dtype"))
                + " rank="
                + str(parameter.get("rank"))
                + " logical_ref="
                + str(parameter.get("logical_ref"))
            )
    else:
        lines.append("  <none>")

    lines.extend(["", "inputs:"])
    for tensor in function["inputs"]:
        lines.append(
            f"  {tensor['name']}: tensor<{tensor['dtype']}"
            f"{_inline(tensor['shape'])}>"
        )
    if not function["inputs"]:
        lines.append("  <none>")

    lines.extend(["", "operations (declaration order):"])
    operations = payload["operations"]
    for operation in operations:
        result = ", ".join(operation["results"]) or "_"
        operands = ", ".join(operation["operands"])
        lines.append(
            f"  [{operation['ordinal']}] {operation['node_ref']} "
            f"@ {operation['path']} in {operation['region']}: "
            f"{result} = {operation['op']}({operands})"
        )
        if operation.get("result_type") is not None:
            lines.append(f"      type {_inline(operation['result_type'])}")
        if operation.get("attrs"):
            lines.append(f"      attrs {_inline(operation['attrs'])}")
        if operation.get("semantic_fields"):
            lines.append(
                f"      structure {_inline(operation['semantic_fields'])}"
            )
        if operation.get("operand_types"):
            lines.append(f"      operand_types {_inline(operation['operand_types'])}")
        if operation.get("result_types"):
            lines.append(f"      result_types {_inline(operation['result_types'])}")
        contract = operation.get("semantic_contract")
        if isinstance(contract, Mapping):
            axis_semantics = contract.get("axis_semantics")
            if axis_semantics:
                lines.append(f"      axis_semantics {_inline(axis_semantics)}")
            relation_refs = contract.get("boundary_relation_refs")
            if isinstance(relation_refs, Mapping):
                for relation_name, relation_indexes in relation_refs.items():
                    relation_table = payload["boundary"].get(relation_name, [])
                    resolved = [
                        relation_table[index]
                        for index in relation_indexes
                        if isinstance(relation_table, list)
                        and isinstance(index, int)
                        and 0 <= index < len(relation_table)
                    ]
                    if resolved:
                        lines.append(
                            f"      {relation_name} {_inline(resolved)}"
                        )
            access_effect_chains = contract.get("access_effect_chains")
            if access_effect_chains:
                lines.append(
                    f"      access_effect_chains {_inline(access_effect_chains)}"
                )
        if operation.get("child_regions"):
            lines.append(
                "      child_regions " + ", ".join(operation["child_regions"])
            )
    if not operations:
        lines.append("  <none>")

    lines.extend(["", "outputs:"])
    for output in function["outputs"]:
        lines.append(
            f"  {output['name']}: tensor<{output['dtype']}"
            f"{_inline(output['shape'])}> <- {output.get('value_ref')}"
        )
    if not function["outputs"]:
        lines.append("  <none>")

    lines.extend(["", "global J-only contracts:"])
    global_contracts = payload["global_contracts"]
    for name in sorted(global_contracts):
        value = global_contracts[name]
        if value in ([], {}):
            continue
        if isinstance(value, list):
            lines.append(f"  {name} [{len(value)}]")
            for index, item in enumerate(value):
                lines.append(f"    [{index}] {_inline(item)}")
        elif isinstance(value, Mapping):
            lines.append(f"  {name} [{len(value)}]")
            for key, item in value.items():
                lines.append(f"    {key} = {_inline(item)}")
        else:
            lines.append(f"  {name} = {_inline(value)}")

    lines.extend(["", "J-only contracts:"])
    lines.append(
        "  dimension_constraints "
        + _inline(payload["dimension_constraints"])
    )
    lines.append(
        "  value_dimension_relations "
        + _inline(payload["value_dimension_relations"])
    )
    lines.append(
        "  dimension_relations " + _inline(payload["dimension_relations"])
    )
    lines.append("  scopes " + _inline(payload["scopes"]))
    lines.append("  storage " + _inline(payload["storage"]))
    return "\n".join(lines) + "\n"


def semantic_materialize(
    j: Mapping[str, Any],
    *,
    masked: bool = False,
) -> SemanticProgram:
    """Materialize one exact source-isolated J view into parameterized ``S_J``.

    ``masked=True`` is deliberately strict: a normal J cannot accidentally be
    supplied to the masked arm.  Conversely, a masked J cannot enter the normal
    arm.  There is no cross-arm cache.
    """

    j_value = _mapping(j, "j")
    if set(j_value) != _J_FIELDS:
        missing = sorted(_J_FIELDS - set(j_value))
        unknown = sorted(set(j_value) - _J_FIELDS)
        raise SemanticMaterializationError(
            f"active J view field mismatch; missing={missing}, unknown={unknown}",
            path="j",
        )
    has_mask = _contains_masked_semantics(j_value)
    if masked and not has_mask:
        raise SemanticMaterializationError(
            "masked arm requires a freshly generated MaskJ(J) view",
            path="j.intent",
        )
    if not masked and has_mask:
        raise SemanticMaterializationError(
            "normal arm cannot consume a masked J view",
            path="j.intent",
        )

    intent = _mapping(j_value["intent"], "j.intent")
    function = _mapping(
        intent.get("intent_function"), "j.intent.intent_function"
    )
    function_name = _string(
        function.get("name"), "j.intent.intent_function.name"
    )
    inputs = _array(function.get("inputs"), "j.intent.intent_function.inputs")
    outputs = _array(function.get("outputs"), "j.intent.intent_function.outputs")
    body = _array(function.get("body"), "j.intent.intent_function.body")
    node_ids = _mapping(j_value["node_ids"], "j.node_ids")
    path_to_node = {str(path): str(node) for node, path in node_ids.items()}

    input_records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    values: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(inputs):
        tensor = _mapping(item, f"j.intent.inputs[{index}]")
        tensor_type = _shape_type(tensor, f"j.intent.inputs[{index}]")
        name = str(tensor["name"])
        record = {"name": name, **tensor_type}
        input_records.append(record)
        values[name] = {
            "ref": name,
            "role": "input",
            "type": tensor_type,
            "definition": None,
            "uses": [],
        }
    for index, item in enumerate(outputs):
        tensor = _mapping(item, f"j.intent.outputs[{index}]")
        tensor_type = _shape_type(tensor, f"j.intent.outputs[{index}]")
        output_records.append({"name": str(tensor["name"]), **tensor_type})

    operations: list[dict[str, Any]] = []
    regions: list[dict[str, Any]] = [
        {
            "id": "function",
            "kind": "function",
            "parent": None,
            "owner": function_name,
            "operations": [],
        }
    ]
    region_by_id = {"function": regions[0]}
    dataflow: list[dict[str, Any]] = []

    def add_value(
        ref: str,
        *,
        role: str,
        value_type: Any,
        definition: str,
    ) -> None:
        if ref in values and values[ref]["definition"] is not None:
            raise SemanticMaterializationError(
                f"value {ref!r} has multiple definitions",
                path=definition,
            )
        values.setdefault(
            ref,
            {
                "ref": ref,
                "role": role,
                "type": copy.deepcopy(value_type),
                "definition": definition,
                "uses": [],
            },
        )

    def add_use(ref: str, node_ref: str, kind: str, index: int | None) -> None:
        values.setdefault(
            ref,
            {
                "ref": ref,
                "role": "external_or_region_value",
                "type": None,
                "definition": None,
                "uses": [],
            },
        )
        use = {"node_ref": node_ref, "kind": kind}
        if index is not None:
            use["index"] = index
        values[ref]["uses"].append(use)
        edge = {"source": ref, "target": node_ref, "kind": kind}
        if index is not None:
            edge["index"] = index
        dataflow.append(edge)

    def create_region(
        region_id: str,
        *,
        kind: str,
        parent: str,
        owner: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if region_id in region_by_id:
            raise SemanticMaterializationError(
                f"duplicate region {region_id!r}", path="j.intent"
            )
        record = {
            "id": region_id,
            "kind": kind,
            "parent": parent,
            "owner": owner,
            "operations": [],
            **({"metadata": copy.deepcopy(dict(metadata))} if metadata else {}),
        }
        regions.append(record)
        region_by_id[region_id] = record

    def walk(ops: Sequence[Any], prefix: str, region_id: str) -> None:
        for index, raw_operation in enumerate(ops):
            path = f"{prefix}[{index}]"
            operation = _mapping(raw_operation, f"j.intent.{path}")
            op_name = _string(operation.get("op"), f"j.intent.{path}.op")
            op_results = _results(operation.get("result"))
            node_ref = path_to_node.get(path)
            if node_ref is None and op_name == "<masked>" and op_results:
                node_ref = op_results[0]
            if node_ref is None:
                node_ref = path
            raw_operands = operation.get("operands", [])
            operands = (
                list(raw_operands)
                if isinstance(raw_operands, list)
                and all(isinstance(item, str) for item in raw_operands)
                else []
            )
            if "operands" in operation and len(operands) != len(raw_operands):
                raise SemanticMaterializationError(
                    "operation operands must be a string array",
                    path=f"j.intent.{path}.operands",
                )
            excluded = {
                "op", "result", "operands", "attrs", "type", "body",
                "positive", "negative",
            }
            semantic_fields = {
                key: copy.deepcopy(value)
                for key, value in operation.items()
                if key not in excluded
            }
            record: dict[str, Any] = {
                "ordinal": len(operations),
                "node_ref": node_ref,
                "path": path,
                "region": region_id,
                "op": op_name,
                "operands": operands,
                "results": op_results,
                "result_type": copy.deepcopy(operation.get("type")),
                "attrs": copy.deepcopy(operation.get("attrs", {})),
                "semantic_fields": semantic_fields,
                "child_regions": [],
            }
            operations.append(record)
            region_by_id[region_id]["operations"].append(node_ref)
            for operand_index, operand in enumerate(operands):
                add_use(operand, node_ref, "operand", operand_index)
            predicate = operation.get("predicate")
            if isinstance(predicate, str):
                add_use(predicate, node_ref, "predicate", None)
            for result in op_results:
                add_value(
                    result,
                    role="operation_result",
                    value_type=operation.get("type"),
                    definition=node_ref,
                )

            scans = operation.get("scan", [])
            if isinstance(scans, list):
                for scan_index, scan_value in enumerate(scans):
                    if not isinstance(scan_value, Mapping):
                        continue
                    name = scan_value.get("name")
                    source = scan_value.get("source", scan_value.get("from"))
                    if isinstance(name, str):
                        add_value(
                            name,
                            role="loop_scan",
                            value_type=scan_value.get("type"),
                            definition=node_ref,
                        )
                    if isinstance(source, str):
                        add_use(source, node_ref, "scan_source", scan_index)
            carries = operation.get("carry", [])
            if isinstance(carries, list):
                for carry_index, carry_value in enumerate(carries):
                    if not isinstance(carry_value, Mapping):
                        continue
                    carry_type = carry_value.get("type")
                    for key, role in (
                        ("body_argument", "loop_state"),
                        ("result_value", "loop_result"),
                    ):
                        ref = carry_value.get(key)
                        if isinstance(ref, str):
                            add_value(
                                ref,
                                role=role,
                                value_type=carry_type,
                                definition=node_ref,
                            )
                    for key, kind in (
                        ("init_value", "carry_init"),
                        ("init", "carry_init"),
                        ("next_value", "carry_next"),
                    ):
                        ref = carry_value.get(key)
                        if isinstance(ref, str):
                            add_use(ref, node_ref, kind, carry_index)

            nested = operation.get("body")
            if isinstance(nested, list) and nested:
                child_id = f"{node_ref}.body"
                create_region(
                    child_id,
                    kind="body",
                    parent=region_id,
                    owner=node_ref,
                )
                record["child_regions"].append(child_id)
                walk(nested, f"{path}.body", child_id)
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if not isinstance(arm, Mapping):
                    continue
                arm_body = arm.get("body")
                if not isinstance(arm_body, list):
                    raise SemanticMaterializationError(
                        "conditional arm body must be an array",
                        path=f"j.intent.{path}.{polarity}.body",
                    )
                child_id = f"{node_ref}.{polarity}"
                arm_metadata = {
                    key: copy.deepcopy(value)
                    for key, value in arm.items()
                    if key != "body"
                }
                create_region(
                    child_id,
                    kind=f"conditional_{polarity}",
                    parent=region_id,
                    owner=node_ref,
                    metadata=arm_metadata,
                )
                record["child_regions"].append(child_id)
                walk(arm_body, f"{path}.{polarity}.body", child_id)

    walk(body, "body", "function")

    boundary = _mapping(j_value["boundary"], "j.boundary")

    # Build a node-local compiler view by exact reference joins only.  This is
    # deliberately not target lowering: the records below are already authored
    # in J, and grouping them by node/access/value removes repeated graph joins
    # from every fresh Replay episode without adding a semantic decision.
    node_relation_refs: dict[str, dict[str, list[int]]] = {}
    for relation_name in _NODE_SCOPED_BOUNDARY_RELATIONS:
        raw_relations = boundary.get(relation_name, [])
        if not isinstance(raw_relations, list):
            continue
        for relation_index, relation in enumerate(raw_relations):
            if not isinstance(relation, Mapping):
                continue
            node_ref = relation.get("node_ref")
            if not isinstance(node_ref, str):
                continue
            node_relation_refs.setdefault(node_ref, {}).setdefault(
                relation_name, []
            ).append(relation_index)

    value_types = {
        str(value["ref"]): copy.deepcopy(value.get("type"))
        for value in values.values()
        if isinstance(value.get("ref"), str)
    }
    for operation in operations:
        operands = operation["operands"]
        results = operation["results"]
        operation["operand_types"] = [
            {"ref": ref, "type": copy.deepcopy(value_types.get(ref))}
            for ref in operands
        ]
        operation["result_types"] = [
            {"ref": ref, "type": copy.deepcopy(value_types.get(ref))}
            for ref in results
        ]
        relation_refs = node_relation_refs.get(operation["node_ref"], {})
        accesses = [
            boundary["accesses"][index]
            for index in relation_refs.get("accesses", [])
        ]
        effects = [
            boundary["effects"][index]
            for index in relation_refs.get("effects", [])
        ]
        access_effect_chains = []
        for access in accesses:
            access_id = access.get("access_id")
            access_effect_chains.append(
                {
                    "access_id": access_id,
                    "effect_ids": [
                        effect.get("effect_id")
                        for effect in effects
                        if access_id is not None
                        and effect.get("access_ref") == access_id
                    ],
                }
            )

        axis_semantics: list[dict[str, Any]] = []
        attrs = operation.get("attrs")
        reduction_axes = (
            attrs.get("axes", [])
            if operation["op"] == "reduce" and isinstance(attrs, Mapping)
            else []
        )
        reduction_axis_set = {
            axis
            for axis in reduction_axes
            if isinstance(axis, int) and not isinstance(axis, bool)
        }
        for operand_index, operand in enumerate(operation["operand_types"]):
            if not reduction_axis_set:
                break
            operand_type = operand.get("type")
            shape = operand_type.get("shape") if isinstance(operand_type, Mapping) else None
            if not isinstance(shape, list):
                continue
            axis_semantics.append(
                {
                    "operand_index": operand_index,
                    "value_ref": operand["ref"],
                    "axes": [
                        {
                            "axis": axis,
                            "extent": copy.deepcopy(extent),
                            "role": (
                                "reduction"
                                if axis in reduction_axis_set
                                else "retained"
                            ),
                        }
                        for axis, extent in enumerate(shape)
                    ],
                }
            )
        operation["semantic_contract"] = {
            "boundary_relation_refs": copy.deepcopy(relation_refs),
            "access_effect_chains": access_effect_chains,
            "axis_semantics": axis_semantics,
        }

    output_mappings = boundary.get("output_mappings", [])
    mapping_by_output = {
        item.get("output"): item.get("value")
        for item in output_mappings
        if isinstance(item, Mapping) and isinstance(item.get("output"), str)
    }
    for output in output_records:
        output["value_ref"] = mapping_by_output.get(output["name"], output["name"])

    signature = _mapping(j_value["signature"], "j.signature")
    dimensions: list[str] = []

    def dimension(ref: Any) -> None:
        if isinstance(ref, str) and _DIMENSION_REF.fullmatch(ref) and ref not in dimensions:
            dimensions.append(ref)

    for tensor in [*input_records, *output_records]:
        for extent in tensor["shape"]:
            dimension(extent)
    for parameter in signature.get("parameters", []):
        if isinstance(parameter, Mapping):
            dimension(parameter.get("logical_ref"))
    dimension_relations = _array(
        j_value["dimension_relations"], "j.dimension_relations"
    )
    for relation in dimension_relations:
        if isinstance(relation, Mapping):
            dimension(relation.get("result"))
            expr = relation.get("expr")
            if isinstance(expr, Mapping):
                dimension(expr.get("arg"))

    payload = {
        "schema": SEMANTIC_PROGRAM_SCHEMA,
        "materializer": SEMANTIC_MATERIALIZER_PROTOCOL,
        "mode": "masked" if masked else "normal",
        "parameterized": True,
        "authority": {
            "semantic_input": "exact J only",
            "logical_program_only": True,
            "target_materialization_implied": False,
            "physical_materialization_authority": "pi only",
            "excluded": [
                "pi",
                "C valuation",
                "Source",
                "Gate observation",
                "frontend identity",
                "target API",
                "historical Replay",
            ],
        },
        "j_handle": _string(j_value["kernel_id"], "j.kernel_id"),
        "function": {
            "name": function_name,
            "signature": signature,
            "inputs": input_records,
            "outputs": output_records,
        },
        "dimensions": [{"ref": ref, "kind": "symbolic"} for ref in dimensions],
        "regions": regions,
        "operations": operations,
        "values": list(values.values()),
        "dataflow": dataflow,
        "boundary": boundary,
        "global_contracts": {
            name: copy.deepcopy(
                [
                    item
                    for item in value
                    if not isinstance(item, Mapping)
                    or not isinstance(item.get("node_ref"), str)
                ]
                if name in _NODE_SCOPED_BOUNDARY_RELATIONS
                and isinstance(value, list)
                else value
            )
            for name, value in boundary.items()
            if name not in _NODE_SCOPED_BOUNDARY_RELATIONS
            or not isinstance(value, list)
            or any(
                not isinstance(item, Mapping)
                or not isinstance(item.get("node_ref"), str)
                for item in value
            )
        },
        "scopes": _array(j_value["scopes"], "j.scopes"),
        "storage": _array(j_value["storage"], "j.storage"),
        "dimension_constraints": _array(
            j_value["dimension_constraints"], "j.dimension_constraints"
        ),
        "value_dimension_relations": _array(
            j_value["value_dimension_relations"],
            "j.value_dimension_relations",
        ),
        "dimension_relations": dimension_relations,
    }
    # Encoding now also rejects non-JSON and non-finite data before S_J can be
    # mounted into a Replay episode.
    json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False)
    counts = {
        "dimensions": len(dimensions),
        "inputs": len(input_records),
        "outputs": len(output_records),
        "regions": len(regions),
        "operations": len(operations),
        "values": len(values),
        "dataflow_edges": len(dataflow),
        "node_semantic_contracts": len(operations),
        **{
            f"boundary_{name}": len(value)
            for name, value in boundary.items()
            if isinstance(value, (list, dict))
        },
    }
    return SemanticProgram.from_payload(
        payload,
        mode="masked" if masked else "normal",
        text=_render_text(payload),
        counts=counts,
    )


__all__ = ["SemanticMaterializationError", "semantic_materialize"]
