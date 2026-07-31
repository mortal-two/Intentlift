"""Responses schema for the LLM-direct ``J + pure C + pi`` Lift result.

The local :class:`Gate1Artifact` parser remains authoritative.  This schema is
the wire contract given to the model so closed Gate1 objects do not drift into
invented field names.  The model, rather than a construction engine, produces
the complete realization witness.  The project parser performs final
cross-field and opaque-reference validation.
"""

from __future__ import annotations

import copy
from typing import Any

from intentir.ir.intent_op_contract import intent_op_json_schema
from intentir.verify.gate1.model import (
    BOUNDARY_ACCESS_REQUIRED_FIELDS,
    BOUNDARY_WIRE_FIELDS,
    GATE1_ARTIFACT_MODEL_FIELDS,
    SEMANTIC_ARCHIVE_WIRE_FIELDS,
    Gate1Artifact,
    render_boundary_access_wire_grammar,
    render_gate1_artifact_envelope_grammar,
)
from intentir.verify.gate1.pi import (
    ITERATION_BASES,
    NORMALIZATION_WITNESS_SCHEMA,
    PHYSICAL_ITERATION_DIRECTIONS,
    REALIZATION_MECHANISMS,
    REPRESENTATION_REFINEMENTS,
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)


def _closed_object(
    properties: dict[str, Any],
    required: tuple[str, ...] = (),
    *,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }
    if description is not None:
        schema["description"] = description
    return schema


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_STRING = {"type": "string", "minLength": 1}
_TENSOR_HANDLE = {"type": "string", "pattern": "^t[0-9]+$"}
_NODE_HANDLE = {"type": "string", "pattern": "^n[0-9]+$"}
_C_HANDLE = {"type": "string", "pattern": "^c[0-9]+$"}
_C_SUBJECT_HANDLE = {"type": "string", "pattern": "^[ndaqb][0-9]+$"}
_PARAMETER_HANDLE = {"type": "string", "pattern": "^p[0-9]+$"}
_PORTABLE_PREDICATE_STRING = {
    "type": "string",
    "minLength": 1,
    "description": (
        "One boolean ContractExpr v1 expression: symbols/literals, arithmetic, "
        "comparisons, and/or/not, and only ceil_div/cdiv/min/max calls. No "
        "assignment, prose, range notation, frontend calls, attributes, or "
        "program/grid/tile/lane coordinate construction."
    ),
}
_DIMENSION_CONSTRAINT_STRING = {
    "type": "string",
    "minLength": 1,
    "description": (
        "One boolean affine ContractExpr over declared x* dimensions. The "
        "v4 checker accepts integer literals, +/-, constant multiplication, "
        "comparisons, and conjunction. It rejects unknown references, "
        "non-affine arithmetic, disjunction, and negation. This J-owned "
        "legal-domain relation is never filled from a Harness valuation. "
        "When Source exposes logical and rounded extents separately, declare "
        "both x* handles and their relation instead of substituting the "
        "current rounded integer."
    ),
}
_STRING_ARRAY = _array(_STRING)
_INT_OR_STRING = {"anyOf": [{"type": "integer"}, _STRING]}


def render_semantic_archive_wire_grammar() -> str:
    """Render global archive invariants already enforced by the strict parser."""

    return (
        "J scopes are an exact derived dominance archive, not a copy of pi "
        "regions. Straight-line J uses one parentless function scope. Its "
        "definitions are exactly J input handles plus results actually produced "
        "by operations in that scope; a declared logical output is not a "
        "definition unless an operation really produces that same handle. "
        "captures and effects are likewise exact, with no convenience extras. "
        "dimension_constraints explicitly declares the closed affine legal "
        "domain of symbolic x* dimensions. Shape-dependent operations such "
        "as pad must state symbolic lo/hi expressions such as x1-x0 in Layer A; "
        "both dimensions remain declared when Source exposes both. Fixed code "
        "only parses and proves the authored relation and never reads a "
        "Harness dimension to repair it. "
        "When one Source scalar is both logical rank-0 data and a shape extent, "
        "keep the t* value and x* dimension distinct and explicitly declare "
        "their equality in value_dimension_relations; fixed code never creates it. "
        "Every semantic loop has exactly one corresponding loop_region scope. "
        "Every effect_if has two explicit positive/negative arms bound to "
        "distinct sibling branch_region scopes. Either arm body may be empty; "
        "the conditional has no result/yield, branch-local SSA never escapes, "
        "and scope.effects plus matching EffectSpec/access_ref is the sole "
        "effect membership authority. "
        "Every value_if has one explicit result/type and two explicit "
        "positive/negative {scope_id,body,yield_value} arms bound to distinct "
        "sibling branch_region scopes. Either pure body may be empty; both "
        "yields match the result type, and only the enclosing result escapes. "
        "value_if is neither select nor effect_if and is never reconstructed "
        "by fixed code. "
        "The storage array may be empty: ordinary SSA temporaries do not have "
        "semantic storage or lifetime objects. Add StorageObject only when J "
        "semantics requires storage identity, such as an alias/destination or "
        "loop-carried state. write_once requires exactly one matching boundary "
        "write event; producing an SSA value is not such an event. Every loop "
        "state requires one storage object binding its state, body, next and "
        "result values plus a non-literal init value, if any, across iterations."
    )


def _tensor_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "name": _TENSOR_HANDLE,
            "dtype": _STRING,
            "shape": _array(_INT_OR_STRING),
        },
        ("name", "dtype", "shape"),
    )


def _value_dimension_relation_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "value_ref": _TENSOR_HANDLE,
            "dimension_ref": {
                "type": "string",
                "pattern": r"^x[0-9]+$",
            },
            "relation": {"type": "string", "enum": ["eq"]},
        },
        ("value_ref", "dimension_ref", "relation"),
        description=(
            "LLM-authored equality between one integral rank-0 logical J input "
            "t* and one declared symbolic shape dimension x*. Fixed code checks "
            "the equality but never infers or inserts this relation."
        ),
    )


def _dimension_relation_schema() -> dict[str, Any]:
    dimension = {"type": "string", "pattern": r"^x[0-9]+$"}
    identity = _closed_object(
        {
            "op": {"type": "string", "enum": ["identity"]},
            "arg": dimension,
        },
        ("op", "arg"),
    )
    const_expr = _closed_object(
        {
            "op": {
                "type": "string",
                "enum": [
                    "add_const",
                    "mul_const",
                    "ceil_div_const",
                    "round_up_const",
                ],
            },
            "arg": dimension,
            "value": {"type": "integer", "minimum": 1},
        },
        ("op", "arg", "value"),
    )
    return _closed_object(
        {
            "result": dimension,
            "expr": {"oneOf": [identity, const_expr]},
        },
        ("result", "expr"),
        description=(
            "LLM-authored closed definition of one declared logical dimension "
            "from another. Fixed code validates and consumes it but never infers "
            "the relation from Harness values, grid, C, pi, or compiler output."
        ),
    )


def _intent_schema() -> dict[str, Any]:
    logical_return = _closed_object(
        {"output": _STRING, "value": _STRING},
        ("output", "value"),
        description=(
            "Layer-A value-to-result binding. Destination aliasing and partial "
            "write behavior belong only to the canonical J boundary."
        ),
    )
    function = _closed_object(
        {
            "name": {"type": "string", "enum": ["scrc_j0"]},
            "inputs": _array(_tensor_schema()),
            "outputs": _array(_tensor_schema()),
            "returns": _array(logical_return),
            "body": _array({"$ref": "#/$defs/intent_op"}),
        },
        ("name", "inputs", "outputs", "body"),
    )
    return _closed_object(
        {"intent_function": function}, ("intent_function",)
    )


def _boundary_schema() -> dict[str, Any]:
    access = _closed_object(
        {
            "access_id": _STRING,
            "kind": {
                "type": "string",
                "enum": ["load", "store", "update"],
            },
            "tensor": _STRING,
            "predicate": _nullable(_PORTABLE_PREDICATE_STRING),
            "fill_value": {},
            "node_ref": _nullable(_STRING),
            "index_expression": _STRING,
        },
        BOUNDARY_ACCESS_REQUIRED_FIELDS,
        description=(
            "Portable logical access in ContractExpr v1. Direct dense surface "
            "forms k0/[k0] and k1,k0/[k1,k0] are deterministically elaborated "
            "to the same structured ordered axes. General non-direct affine "
            "relations remain typed expressions but can be unsupported by a "
            "particular RefEval capability profile. "
            "A node_ref makes the access consumer-local: the relation is not "
            "implicitly inherited by sibling uses of the same tensor. "
            "fill_value is legal only on a load with a non-null predicate. An "
            "unconditional access uses the unique canonical spelling with both "
            "predicate and fill_value absent/null; consumers reject rather than "
            "silently erase an orphan fill_value."
        ),
    )
    access["allOf"] = [
        {
            "if": {
                "required": ["fill_value"],
                "properties": {
                    "fill_value": {"not": {"type": "null"}},
                },
            },
            "then": {
                "required": ["kind", "predicate"],
                "properties": {
                    "kind": {"const": "load"},
                    "predicate": {"type": "string", "minLength": 1},
                },
            },
        }
    ]
    broadcast = _closed_object(
        {
            "node_ref": _STRING,
            "operand": _STRING,
            "axes": _array(_INT_OR_STRING),
        },
        ("node_ref", "operand", "axes"),
    )
    access_guard_relation = _closed_object(
        {
            "access_ref": _STRING,
            "value_ref": _STRING,
            "axes": _array({"type": "integer", "minimum": 0}),
            "polarity": {
                "type": "string",
                "enum": ["positive", "negative"],
            },
        },
        ("access_ref", "value_ref", "axes", "polarity"),
        description=(
            "LLM-authored J relation from one declared boundary access to one "
            "declared Layer-A guard value. axes contains only JSON integer "
            "access-axis ordinals and polarity is closed. Fixed code never "
            "infers or inserts this relation."
        ),
    )
    alias = _closed_object(
        {
            "left": _STRING,
            "right": _STRING,
            "kind": {
                "type": "string",
                "enum": ["must_alias", "may_alias", "no_alias"],
            },
        },
        ("left", "right", "kind"),
    )
    effect = _closed_object(
        {
            "effect_id": _STRING,
            "kind": {
                "type": "string",
                "enum": [
                    "read",
                    "write",
                    "read_write",
                    "update",
                ],
            },
            "target": _STRING,
            "node_ref": _nullable(_STRING),
            "access_ref": _nullable(_STRING),
        },
        ("effect_id", "kind", "target", "access_ref"),
    )
    output_mapping = _closed_object(
        {
            "output": _STRING,
            "value": _STRING,
            "node_ref": _nullable(_STRING),
        },
        ("output", "value"),
        description=(
            "One authored canonical output binding. It may be the sole "
            "non-identity destination relation; when IntentFunction.returns "
            "also binds this output, both values must agree exactly. When an "
            "op directly produces the same output handle, leave "
            "output_mappings empty."
        ),
    )
    reduction = _closed_object(
        {
            "node_ref": _STRING,
            "tile_axis": _INT_OR_STRING,
            "identity": {},
        },
        ("node_ref", "tile_axis", "identity"),
        description=(
            "Reduce or inclusive-scan algebra and frontend-local tile-axis "
            "metadata. artifact.v4+ forbids the historical "
            "masked_contribution field: consumer_extensions is the unique "
            "authority for physical outside-domain contributions."
        ),
    )
    typed_outside = _closed_object(
        {
            "dtype": _STRING,
            "value": {
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "integer"},
                    {"type": "number"},
                    {"type": "string"},
                ]
            },
        },
        ("dtype", "value"),
    )
    consumer_extension = _closed_object(
        {
            "consumer_ref": _NODE_HANDLE,
            "operand_index": {"type": "integer", "minimum": 0},
            "access_ref": {"type": "string", "pattern": "^a[0-9]+$"},
            "tile_axis": {"type": "integer", "minimum": 0},
            "outside": typed_outside,
            "identity_law": {
                "type": "string",
                "enum": ["two_sided"],
            },
        },
        (
            "consumer_ref",
            "operand_index",
            "access_ref",
            "tile_axis",
            "outside",
            "identity_law",
        ),
        description=(
            "One LLM-authored Layer-B relation from a predicate-bearing load "
            "access to one reduce/scan operand. The referenced BoundaryAccess "
            "continues to own the predicate; no predicate is promoted to "
            "Layer-A SSA. v4 closes only two-sided sum/max/min identities."
        ),
    )
    index_bounds = _closed_object(
        {
            "node_ref": _STRING,
            "data": _STRING,
            "indices": _STRING,
            "axis": {"type": "integer", "minimum": 0},
        },
        ("node_ref", "data", "indices", "axis"),
        description=(
            "Typed precondition 0 <= indices < dim(data, axis), exactly bound "
            "to one gather/scatter node. Never use a proof string or proven flag."
        ),
    )
    update = _closed_object(
        {
            "node_ref": _STRING,
            "base": _STRING,
            "indices": _STRING,
            "values": _STRING,
            "axis": {"type": "integer", "minimum": 0},
            "combiner": {
                "type": "string", "enum": ["overwrite", "add"],
            },
            "duplicate_policy": {
                "type": "string", "enum": ["forbid", "combine", "one_of"],
            },
            "order": {
                "type": "string", "enum": ["input_order", "unordered"],
            },
        },
        (
            "node_ref", "base", "indices", "values", "axis", "combiner",
            "duplicate_policy", "order",
        ),
        description=(
            "Logical update only. Atomic/CAS/thread ownership are realization "
            "choices and must not appear here."
        ),
    )
    result_relation = _closed_object(
        {
            "output": _STRING,
            "kind": {
                "type": "string",
                "enum": ["exact", "tolerance", "one_of_updates"],
            },
            "update_node_ref": _STRING,
        },
        ("output", "kind"),
        description=(
            "Comparison class owned by J. Numeric tolerance values are forbidden "
            "and come only from external Gate2 policy Q. one_of_updates requires "
            "one non-empty update_node_ref. exact/tolerance must omit the "
            "update_node_ref field entirely; empty string or null is invalid."
        ),
    )
    authorization = _closed_object(
        {
            "auth_id": {"type": "string", "pattern": "^u[0-9]+$"},
            "kind": {
                "type": "string",
                "enum": [
                    "signed_widen",
                    "unsigned_widen",
                    "numeric_cast",
                    "accumulation_dtype",
                    "reassociation",
                    "native_atomic",
                    "privatized_update",
                ],
            },
            "subject_ref": _STRING,
            "source_dtype": _STRING,
            "target_dtype": _STRING,
        },
        ("auth_id", "kind", "subject_ref"),
        description=(
            "J/B-owned authorization for one sensitive representation choice. "
            "Compact pi cannot cite, extend, or redefine this semantic contract."
        ),
    )
    return _closed_object(
        {
            "domains": {
                "type": "object",
                "propertyNames": {"pattern": "^d[0-9]+$"},
                "additionalProperties": _PORTABLE_PREDICATE_STRING,
                "description": (
                    "Map each d* handle directly to one ContractExpr string. "
                    "Never emit an axes/members/domain object here."
                ),
            },
            "accesses": _array(access),
            "broadcasts": _array(broadcast),
            "aliases": _array(alias),
            "effects": _array(effect),
            "output_mappings": _array(output_mapping),
            "reductions": _array(reduction),
            "index_bounds": _array(index_bounds),
            "updates": _array(update),
            "result_relations": _array(result_relation),
            "realization_authorizations": _array(authorization),
            "consumer_extensions": _array(consumer_extension),
            "access_guard_relations": _array(access_guard_relation),
        },
        BOUNDARY_WIRE_FIELDS,
        description=render_boundary_access_wire_grammar(),
    )


def _signature_schema() -> dict[str, Any]:
    parameter = _closed_object(
        {
            "name": _PARAMETER_HANDLE,
            "kind": {
                "type": "string",
                "enum": ["tensor", "pointer", "scalar", "opaque", "constexpr"],
            },
            "mode": {
                "type": "string",
                "enum": ["input", "output", "inout", "value"],
            },
            "dtype": _nullable(_STRING),
            "rank": _nullable({"type": "integer", "minimum": 0}),
            "alias_of": _nullable(_STRING),
            "logical_ref": _nullable(_STRING),
        },
        ("name", "kind", "mode"),
        description=(
            "One J-owned logical callable handle. logical_ref links a "
            "pointer/tensor/scalar/semantic-constexpr handle only to a "
            "declared J input/output tensor or a symbolic J dimension; it "
            "never names a v* SSA value. Logical scalar data and constexpr "
            "values used as runtime-like logical data use a dtype-matching "
            "rank-0 J input and must "
            "contribute to an output dependency or explicit boundary/shape "
            "semantics. A runtime scalar used only for representation/address "
            "realization is kind=scalar, mode=value, rank=0, has public dtype "
            "but no logical_ref/J input; it remains an explicit ABI carrier and "
            "compact pi never assigns it semantic ownership. "
            "Integral "
            "extent-only handles instead bind a symbolic dimension. Every trusted "
            "compiler-view tensor argument, including one unused by the frozen "
            "specialization, keeps a logical_ref whose J tensor dtype, rank and "
            "shape exactly match that view. The inactive tensor remains a typed "
            "ABI declaration but has no invented operation, access or effect. "
            "Implementation-selection constexprs are C slots and never appear "
            "in this array. A constexpr used anywhere in the active graph as "
            "logical data, extent, shape, index, mask, predicate, or result binds "
            "t*/x* even when constant-folded. Only a Harness-fixed constexpr whose "
            "Source uses are exclusively compile-time graph discrimination may be "
            "a specialization carrier with kind=constexpr, mode=value, rank=0, "
            "public dtype and no logical_ref; its concrete value stays only in "
            "Harness, while Lift authors the complete active J graph. Lift, not "
            "fixed code, owns that semantic classification. The carrier is neither "
            "Layer-A data nor a C/backend choice. An opaque parameter is "
            "representation-only frontend plumbing such as cute.CopyAtom; it "
            "has dtype but no logical_ref and contributes no J semantics."
        ),
    )
    return _closed_object(
        {"parameters": _array(parameter)},
        ("parameters",),
    )


def _c_schema() -> dict[str, Any]:
    slot = _closed_object(
        {
            "name": _C_HANDLE,
            "kind": {
                "type": "string",
                "enum": ["constexpr", "compiler"],
                "description": (
                    "constexpr is a callable frontend compile-time parameter; "
                    "compiler is an external compiler-invocation option and never "
                    "a callable or kernel-body carrier."
                ),
            },
            "role": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*$",
            },
            "value_type": _STRING,
            "required": {"type": "boolean"},
            "subject_ref": {
                **_C_SUBJECT_HANDLE,
                "description": (
                    "Optional source-neutral attachment to one declared J "
                    "node/domain/access/scope/storage subject. It is not a "
                    "transformation permission or source binding. Compiler-kind "
                    "slots must omit subject_ref and axis_ref."
                ),
            },
            "axis_ref": {
                "anyOf": [
                    {"type": "integer", "minimum": 0},
                    {"type": "string", "pattern": "^x[0-9]+$"},
                ],
                "description": (
                    "Optional logical axis in AxisRefs_v1(J,subject_ref); "
                    "requires subject_ref and never names a physical axis."
                ),
            },
        },
        ("name", "kind", "role"),
    )
    return _closed_object(
        {
            "slots": _array(slot),
            "values": {
                "type": "object",
                "propertyNames": {"pattern": "^c[0-9]+$"},
                "additionalProperties": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "integer"},
                        {"type": "number"},
                        {"type": "string"},
                    ]
                },
            },
        },
        ("slots", "values"),
    )


def _legacy_pi_schema() -> dict[str, Any]:
    opaque_ref = {
        "type": "string",
        "pattern": "^(t|n|a|d|e|q|b|s|v|p|c|x)[0-9]+$",
    }
    region = _closed_object(
        {
            "id": {"type": "string", "pattern": "^r[0-9]+$"},
            "parent": _nullable({"type": "string", "pattern": "^r[0-9]+$"}),
            "placement": {
                "type": "string", "enum": ["root", "program", "loop", "branch"],
            },
        },
        ("id", "parent", "placement"),
    )
    instance = _closed_object(
        {
            "id": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Declaration-order pi graph-node ID. This is the only ID "
                    "namespace allowed at route/order/sharing endpoints."
                ),
            },
            "semantic_ref": {
                **opaque_ref,
                "description": (
                    "Anchor into a declared anonymous J/C handle. "
                    "It is not a pi graph-node ID and never appears directly "
                    "at route/order/sharing endpoints."
                ),
            },
            "region": {"type": "string", "pattern": "^r[0-9]+$"},
            "mechanism": {
                "type": "string",
                "enum": list(REALIZATION_MECHANISMS),
                "description": render_realization_mechanism_grammar(),
            },
        },
        ("id", "semantic_ref", "region", "mechanism"),
    )
    route = _closed_object(
        {
            "source": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A previously declared pi instance ID, never a raw anchor handle.",
            },
            "target": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A previously declared pi instance ID, never a raw anchor handle.",
            },
            "slots": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
                "uniqueItems": True,
            },
        },
        ("source", "target", "slots"),
    )
    program_axis = _closed_object(
        {
            "domain_ref": {
                "type": "string",
                "pattern": "^d[0-9]+$",
                "description": (
                    "J boundary domain anchoring one compiler-visible program "
                    "coordinate."
                ),
            },
            "axis": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
                "description": (
                    "Exact physical launch-axis identity; never inferred from "
                    "grid extent, declaration order, C, or instance multiplicity."
                ),
            },
        },
        ("domain_ref", "axis"),
        description=(
            "A closed pi-owned binding from one J domain program-coordinate "
            "identity to physical launch axis 0, 1, or 2. It carries no grid "
            "extent, Source spelling, frontend API, or logical J semantics."
        ),
    )
    program_linearization = _closed_object(
        {
            "domain_ref": {
                "type": "string",
                "pattern": "^d[0-9]+$",
                "description": "The J domain whose bound physical program coordinate is decoded.",
            },
            "dimension_refs": {
                "type": "array",
                "items": {"type": "string", "pattern": "^x[0-9]+$"},
                "minItems": 1,
                "uniqueItems": True,
                "description": (
                    "Outer-to-inner J dimension handles for the fixed row-major "
                    "mixed-radix quotient/remainder projection."
                ),
            },
        },
        ("domain_ref", "dimension_refs"),
        description=(
            "A source-neutral mixed-radix realization of one bound program "
            "coordinate. It contains no formula, value, predicate, Source spelling, "
            "frontend API, or target recipe."
        ),
    )
    route_extent_transform = _closed_object(
        {
            "source": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Instance anchored to one required positive-integer C extent slot."
                ),
            },
            "target": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Instance anchored to the C-attached J domain realization."
                ),
            },
            "slot": {"type": "integer", "minimum": 0},
            "divisor": {"type": "integer", "minimum": 2},
            "rounding": {"type": "string", "enum": ["floor", "ceil"]},
        },
        ("source", "target", "slot", "divisor", "rounding"),
        description=(
            "A closed representation-only extent partition on one existing pi "
            "route slot. It does not create J shape/domain/predicate semantics and "
            "contains no Source spelling or frontend API."
        ),
    )
    order = _closed_object(
        {
            "before": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A declared pi instance ID.",
            },
            "after": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "A declared pi instance ID.",
            },
        },
        ("before", "after"),
    )
    sharing = _closed_object(
        {
            "source": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": "The declared pi instance reused by all consumers.",
            },
            "consumers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "pattern": "^i[0-9]+$",
                    "description": "A declared pi consumer instance ID.",
                },
                "minItems": 1,
                "uniqueItems": True,
            },
        },
        ("source", "consumers"),
    )
    refinement = _closed_object(
        {
            "instance": {"type": "string", "pattern": "^i[0-9]+$"},
            "kind": {
                "type": "string",
                "enum": list(REPRESENTATION_REFINEMENTS),
            },
            "auth_ref": _nullable(
                {"type": "string", "pattern": "^u[0-9]+$"}
            ),
        },
        ("instance", "kind", "auth_ref"),
        description=(
            "A representation refinement. Sensitive kinds cite one J/B-owned "
            "authorization; identity materialization uses the verifier's closed "
            "global-safe rule."
        ),
    )
    iteration = _closed_object(
        {
            "region": {"type": "string", "pattern": "^r[0-9]+$"},
            "loop_instance": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Instance anchored to the J structured loop realized by "
                    "this physical loop region."
                ),
            },
            "tile_instance": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Instance anchored to the matching integer C tile slot."
                ),
            },
            "induction_basis": {
                "type": "string",
                "enum": list(ITERATION_BASES),
            },
        },
        ("region", "loop_instance", "tile_instance", "induction_basis"),
        description=(
            "A source-neutral physical induction-space relation. It contains "
            "no bound expression, numeric value, operator, source text, or "
            "frontend API."
        ),
    )
    loop_carrier = _closed_object(
        {
            "region": {"type": "string", "pattern": "^r[0-9]+$"},
            "loop_instance": {"type": "string", "pattern": "^i[0-9]+$"},
            "init": {"type": "string", "pattern": "^i[0-9]+$"},
            "state": {"type": "string", "pattern": "^i[0-9]+$"},
            "next": {"type": "string", "pattern": "^i[0-9]+$"},
            "result": {"type": "string", "pattern": "^i[0-9]+$"},
        },
        ("region", "loop_instance", "init", "state", "next", "result"),
        description=(
            "One declaration-ordered physical loop-carrier slot. Its temporal "
            "denotation is state[0]=init, state[k+1]=next[k], and "
            "result=state[T], including T=0. state and next are distinct physical "
            "occurrences joined by an ordinary route; next-to-state temporal and "
            "loop-exit result edges are implicit in this relation."
        ),
    )
    physical_iteration = _closed_object(
        {
            "region": {"type": "string", "pattern": "^r[0-9]+$"},
            "body_instance": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Instance anchored to one existing non-loop J "
                    "reduce/contraction or boundary access/domain realized by "
                    "this physical-only loop region."
                ),
            },
            "tile_instance": {
                "type": "string",
                "pattern": "^i[0-9]+$",
                "description": (
                    "Instance anchored to the required positive integer C tile "
                    "attached to the same J node and axis."
                ),
            },
            "induction_basis": {
                "type": "string",
                "enum": list(ITERATION_BASES),
            },
            "direction": {
                "type": "string",
                "enum": list(PHYSICAL_ITERATION_DIRECTIONS),
            },
        },
        (
            "region",
            "body_instance",
            "tile_instance",
            "induction_basis",
            "direction",
        ),
        description=(
            "A physical iteration shell for an existing logical operation or "
            "boundary traversal with a closed dependent axis. It adds no J loop, "
            "bound formula, numeric value, operator, source text, or frontend API."
        ),
    )
    physical_loop_carrier = _closed_object(
        {
            "region": {"type": "string", "pattern": "^r[0-9]+$"},
            "init": {"type": "string", "pattern": "^i[0-9]+$"},
            "state": {"type": "string", "pattern": "^i[0-9]+$"},
            "next": {"type": "string", "pattern": "^i[0-9]+$"},
            "result": {"type": "string", "pattern": "^i[0-9]+$"},
        },
        ("region", "init", "state", "next", "result"),
        description=(
            "One typed physical carrier in a physical_iterations region. "
            "init/state/next/result are distinct temporal occurrences; init -> state "
            "and state -> next are ordinary pi routes, while next-to-state temporal "
            "and loop-exit result edges are implicit. Roles may anchor the same J "
            "semantic value and carry no update formula or frontend spelling."
        ),
    )
    effect_terminal_control = _closed_object(
        {
            "terminal_instance": {
                "type": "string",
                "pattern": "^i[0-9]+$",
            },
            "continuation_instance": {
                "type": "string",
                "pattern": "^i[0-9]+$",
            },
        },
        ("terminal_instance", "continuation_instance"),
        description=(
            "One source-neutral terminal-control topology relation. The two "
            "instances must anchor distinct sibling effect_if arm scopes; the "
            "terminal arm ends the current kernel invocation and the continuation "
            "arm carries the remaining lexical computation. No predicate, "
            "polarity, return value, source spelling, or frontend API is present."
        ),
    )
    domain_terminal_control = _closed_object(
        {
            "domain_ref": {
                "type": "string",
                "pattern": "^d[0-9]+$",
            },
            "continuation_region": {
                "type": "string",
                "pattern": "^r[0-9]+$",
            },
        },
        ("domain_ref", "continuation_region"),
        description=(
            "One source-neutral program-domain terminal relation. The named "
            "placement=branch continuation region must already be controlled by "
            "an empty-slot route from the same J domain realization. Coordinates "
            "outside that J domain terminate the current invocation. The relation "
            "contains no predicate, polarity, return value, source spelling, or "
            "frontend API."
        ),
    )
    terminal_control = {
        "oneOf": [effect_terminal_control, domain_terminal_control],
        "description": (
            "Either terminal-versus-continuation topology for an authored J "
            "effect_if, or the terminal complement of an existing J-domain-owned "
            "continuation region. Both forms depend on J and carry no condition."
        ),
    }
    return _closed_object(
        {
            "regions": _array(region),
            "instances": _array(instance),
            "routes": _array(route),
            "program_axes": _array(program_axis),
            "program_linearizations": _array(program_linearization),
            "route_extent_transforms": _array(route_extent_transform),
            "order": _array(order),
            "sharing": _array(sharing),
            "refinements": _array(refinement),
            "iterations": _array(iteration),
            "loop_carriers": _array(loop_carrier),
            "physical_iterations": _array(physical_iteration),
            "physical_loop_carriers": _array(physical_loop_carrier),
            "terminal_controls": _array(terminal_control),
        },
        (
            "regions",
            "instances",
            "routes",
            "program_axes",
            "route_extent_transforms",
            "order",
            "sharing",
            "refinements",
            "iterations",
            "loop_carriers",
            "physical_iterations",
            "physical_loop_carriers",
            "terminal_controls",
        ),
        description=(
            render_realization_witness_wire_grammar()
            + " "
            + render_realization_mechanism_grammar()
            + " Preserve route/order/share/region multiplicity. Apart from the "
            "closed route-extent divisor/rounding, no operator or value; never a "
            "predicate, dtype, frontend API, rule identity or source text."
        ),
    )


def _pi_schema() -> dict[str, Any]:
    """Active, certification-only pi subwire offered to Lift.

    Every accepted field has a closed typed meaning.  A profile may normalize it
    only after implementing a sound unary anchor matcher; the current profile
    keeps program axes hard.  The legacy realization graph is intentionally
    absent: new Lift candidates do not serialize regions, instances, routes,
    carriers, or target recipes.
    """

    program_axis = _closed_object(
        {
            "domain_ref": {
                "type": "string",
                "pattern": "^d[0-9]+$",
                "description": "Declared J boundary domain directly observed on this physical launch axis; the current profile keeps the binding hard.",
            },
            "axis": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
                "description": "Observed Triton launch axis; it does not enter Replay.",
            },
        },
        ("domain_ref", "axis"),
    )
    return _closed_object(
        {
            "schema": {"const": NORMALIZATION_WITNESS_SCHEMA},
            "program_axes": _array(program_axis),
        },
        ("schema", "program_axes"),
        description=(
            "Compact Gate1-only normalization witness. It never enters Replay. "
            "Author only directly observed, unambiguous program-axis labels anchored "
            "in J; use an empty array otherwise. A row does not itself authorize "
            "normalization, and the current profile keeps it hard."
        ),
    )


def _portable_scope_schema() -> dict[str, Any]:
    return _closed_object(
        {
            "scope_id": _STRING,
            "kind": {
                "type": "string",
                "enum": [
                    "function", "program_instance", "loop_region",
                    "branch_region",
                ],
            },
            "parent_scope": _nullable(_STRING),
            "captures": _STRING_ARRAY,
            "definitions": _STRING_ARRAY,
            "effects": _STRING_ARRAY,
        },
        (
            "scope_id", "kind", "parent_scope", "captures", "definitions",
            "effects",
        ),
        description=(
            "Exact portable dominance/capture/effect scope derived from J. "
            "definitions/captures/effects must contain neither missing nor extra "
            "handles. Contains no source AST, helper name or frontend spelling."
        ),
    )


def _storage_schema() -> dict[str, Any]:
    event = _closed_object(
        {
            "kind": {
                "type": "string",
                "enum": [
                    "scope_entry", "scope_exit", "node", "access", "state",
                    "effect",
                ],
            },
            "ref": _STRING,
        },
        ("kind", "ref"),
    )
    return _closed_object(
        {
            "storage_id": _STRING,
            "members": _STRING_ARRAY,
            "scope_id": _STRING,
            "mutability": {
                "type": "string",
                "enum": ["immutable", "write_once", "mutable"],
            },
            "init_event": event,
            "last_use_event": event,
            "escapes": {"type": "boolean"},
            "crosses_iteration": {"type": "boolean"},
        },
        (
            "storage_id", "members", "scope_id", "mutability", "init_event",
            "last_use_event", "escapes", "crosses_iteration",
        ),
        description=(
            "Optional semantic storage identity/lifetime only; ordinary SSA "
            "temporaries require no entry. Physical allocation, register/shared/"
            "global placement and double buffering belong to C."
        ),
    )


def _location_map_schema() -> dict[str, Any]:
    span = _closed_object(
        {
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
            "start_column": _nullable({"type": "integer", "minimum": 0}),
            "end_column": _nullable({"type": "integer", "minimum": 0}),
        },
        ("start_line", "end_line"),
    )
    entry = _closed_object(
        {
            "span": span,
            "node_refs": _STRING_ARRAY,
            "access_refs": _STRING_ARRAY,
        },
        ("span", "node_refs", "access_refs"),
    )
    return _closed_object(
        {
            "source_name": {
                "type": "string", "enum": ["anonymous_source"]
            },
            "entries": _array(entry),
        },
        ("source_name", "entries"),
    )


def scrc_lift_core_json_schema() -> dict[str, Any]:
    """Return the complete source-aware LLM response schema."""

    semantic_archive = _closed_object(
        {
            "kernel_id": {"type": "string", "enum": ["scrc_k0"]},
            "intent": _intent_schema(),
            "node_ids": {
                "type": "object",
                "propertyNames": {"pattern": "^n[0-9]+$"},
                "additionalProperties": {
                    "type": "string",
                    "pattern": (
                        "^(intent_function\\.)?body\\[[0-9]+\\]"
                        "((\\.body|\\.positive\\.body|\\.negative\\.body)"
                        "\\[[0-9]+\\])*$"
                    ),
                },
                "description": (
                    "Map n0,n1,... to structural paths such as body[0] or "
                    "body[1].positive.body[0], "
                    "not to SSA/result handles such as v0."
                ),
            },
            "boundary": _boundary_schema(),
            "signature": _signature_schema(),
            "scopes": {
                "type": "array",
                "minItems": 1,
                "items": _portable_scope_schema(),
            },
            "storage": _array(_storage_schema()),
            "dimension_constraints": _array(
                _DIMENSION_CONSTRAINT_STRING
            ),
            "value_dimension_relations": _array(
                _value_dimension_relation_schema()
            ),
            "dimension_relations": _array(_dimension_relation_schema()),
        },
        SEMANTIC_ARCHIVE_WIRE_FIELDS,
        description=render_semantic_archive_wire_grammar(),
    )
    schema = _closed_object(
        {
            "schema": {
                "type": "string",
                "enum": [Gate1Artifact.SCHEMA],
            },
            "j": semantic_archive,
            "c": _c_schema(),
            "pi": _pi_schema(),
        },
        GATE1_ARTIFACT_MODEL_FIELDS,
        description=render_gate1_artifact_envelope_grammar(),
    )
    schema["$defs"] = {"intent_op": intent_op_json_schema()}
    return copy.deepcopy(schema)


def scrc_lift_core_text_format() -> dict[str, Any]:
    """Return the Responses format for Lift's source-aware LLM substage."""

    return {
        "type": "json_schema",
        "name": "scrc_lift_core",
        # Operation-specific semantic attrs remain open during the IntentOp
        # robustness audit. B is the one closed top-level boundary; C and pi
        # are separate closed objects.
        "strict": False,
        "schema": scrc_lift_core_json_schema(),
    }


__all__ = [
    "render_boundary_access_wire_grammar",
    "render_gate1_artifact_envelope_grammar",
    "render_semantic_archive_wire_grammar",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_text_format",
]
