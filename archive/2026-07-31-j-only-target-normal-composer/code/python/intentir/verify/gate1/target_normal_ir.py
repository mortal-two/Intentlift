"""Closed J-only Target-Normal IR used inside one source-blind Replay episode.

The model authors only the target-normal grouping/selection manifest.  Exact
operand, result, type, control and boundary facts remain the immutable ``S_J``
payload referenced by that manifest.  The checker below joins the two without
consulting pi, C, Source, a Harness valuation, or Gate evidence.  The checked
object is therefore a complete episode-local ``F_f`` rather than a fourth SCRC
artifact or a target-source template.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .replay_semantic_core import REPLAY_SEMANTIC_PROGRAM_PROTOCOL


TARGET_NORMAL_IR_PROTOCOL = "intentir.replay-target-normal-ir.v1"
TARGET_NORMAL_CHECK_PROTOCOL = "intentir.replay-target-normal-check.v1"


class TargetNormalError(ValueError):
    """The model-authored target-normal manifest is not closed over this J."""

    def __init__(self, rule_code: str, path: str, detail: str) -> None:
        self.rule_code = rule_code
        self.path = path
        self.detail = detail
        super().__init__(f"{rule_code} at {path}: {detail}")

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "rule_code": self.rule_code,
            "path": self.path,
            "detail": self.detail,
        }


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise TargetNormalError("object_required", path, "must be an object")
    return dict(value)


def _closed_object(
    value: Any,
    *,
    path: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    result = _object(value, path)
    missing = sorted(required - set(result))
    unknown = sorted(set(result) - required - optional)
    if missing:
        raise TargetNormalError(
            "missing_field", path, "missing " + ", ".join(missing)
        )
    if unknown:
        raise TargetNormalError(
            "unknown_field", path, "unknown " + ", ".join(unknown)
        )
    return result


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise TargetNormalError("string_required", path, "must be non-empty")
    return value


def _strings(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
    unique: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TargetNormalError("array_required", path, "must be an array")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if nonempty and not result:
        raise TargetNormalError("nonempty_required", path, "must not be empty")
    if unique and len(result) != len(set(result)):
        raise TargetNormalError("duplicate_reference", path, "must be unique")
    return result


def _node_results(node: Mapping[str, Any]) -> tuple[str, ...]:
    result = node.get("result")
    if isinstance(result, str):
        return (result,)
    if isinstance(result, list):
        return tuple(item for item in result if isinstance(item, str))
    return ()


_ELEMENTWISE_OPS = frozenset(
    {
        "const",
        "map_unary",
        "map_binary",
        "cast",
        "select",
        "broadcast",
        "transpose",
        "reshape",
        "gather",
        "scatter",
        "iota",
    }
)
_REDUCTION_OPS = frozenset({"reduce", "scan"})
_CONTRACTION_OPS = frozenset({"contraction"})
_CONTROL_OPS = frozenset({"loop", "effect_if", "value_if"})
_PRIMITIVES = frozenset(
    {
        "direct",
        "fused_elementwise",
        "reduction_pipeline",
        "contraction_pipeline",
        "structured_control",
        "unknown",
    }
)


def _check_primitive(
    primitive: str,
    nodes: Sequence[Mapping[str, Any]],
    path: str,
    *,
    masked: bool,
) -> None:
    if masked:
        if primitive != "unknown":
            raise TargetNormalError(
                "masked_primitive",
                path,
                "MaskJ nodes have no semantic opcode; only unknown is legal",
            )
        return
    if primitive == "unknown":
        raise TargetNormalError(
            "unknown_primitive", path, "unknown is legal only for MaskJ"
        )
    operations = tuple(str(node.get("op", "")) for node in nodes)
    if primitive == "direct":
        if len(nodes) != 1:
            raise TargetNormalError(
                "primitive_shape", path, "direct must cover exactly one J node"
            )
        return
    if primitive == "fused_elementwise":
        if not nodes or any(op not in _ELEMENTWISE_OPS for op in operations):
            raise TargetNormalError(
                "primitive_shape",
                path,
                "fused_elementwise may cover only elementwise/data-movement J nodes",
            )
        return
    if primitive == "reduction_pipeline":
        allowed = _ELEMENTWISE_OPS | _REDUCTION_OPS
        if (
            not any(op in _REDUCTION_OPS for op in operations)
            or any(op not in allowed for op in operations)
        ):
            raise TargetNormalError(
                "primitive_shape",
                path,
                "reduction_pipeline requires a reduction and only reduction/elementwise nodes",
            )
        return
    if primitive == "contraction_pipeline":
        allowed = _ELEMENTWISE_OPS | _CONTRACTION_OPS
        if (
            not any(op in _CONTRACTION_OPS for op in operations)
            or any(op not in allowed for op in operations)
        ):
            raise TargetNormalError(
                "primitive_shape",
                path,
                "contraction_pipeline requires a contraction and only contraction/elementwise nodes",
            )
        return
    if primitive == "structured_control":
        if not any(op in _CONTROL_OPS for op in operations):
            raise TargetNormalError(
                "primitive_shape", path, "structured_control requires a control J node"
            )
        return
    raise TargetNormalError("unknown_primitive", path, primitive)


@dataclass(frozen=True)
class TargetNormalGroup:
    id: str
    primitive: str
    semantic_refs: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "primitive": self.primitive,
            "semantic_refs": list(self.semantic_refs),
        }


@dataclass(frozen=True)
class CheckedTargetNormalProgram:
    """Complete, checked ``F_f``: model choices plus immutable exact ``S_J``."""

    frontend: str
    entry: str
    groups: tuple[TargetNormalGroup, ...]
    access_order: tuple[str, ...]
    outputs: tuple[str, ...]
    semantic_program: Mapping[str, Any]

    @property
    def protocol(self) -> str:
        return TARGET_NORMAL_IR_PROTOCOL

    def authored_json(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "frontend": self.frontend,
            "entry": self.entry,
            "groups": [group.to_json() for group in self.groups],
            "access_order": list(self.access_order),
            "outputs": list(self.outputs),
        }

    def to_evidence(self) -> dict[str, Any]:
        nodes = self.semantic_program.get("nodes", [])
        return {
            **self.authored_json(),
            "checked_semantic_node_count": len(nodes) if isinstance(nodes, list) else 0,
            "authority": (
                "proof-irrelevant Replay-internal F_f; exact semantics are the attached "
                "read-only S_J; no pi, C, Source, Harness valuation, Gate, or target source"
            ),
        }


def check_target_normal_program(
    value: Mapping[str, Any],
    semantic_program: Mapping[str, Any],
    *,
    frontend: str,
    entry: str,
) -> CheckedTargetNormalProgram:
    """Check one model-authored manifest against exactly one immutable ``S_J``."""

    if semantic_program.get("protocol") != REPLAY_SEMANTIC_PROGRAM_PROTOCOL:
        raise TargetNormalError(
            "semantic_protocol", "S_J.protocol", "unsupported semantic program"
        )
    semantic_status = semantic_program.get("status")
    if semantic_status not in {"normal", "masked"}:
        raise TargetNormalError(
            "semantic_status",
            "S_J.status",
            "must be normal or masked",
        )
    masked = semantic_status == "masked"
    data = _closed_object(
        value,
        path="F_f",
        required=frozenset(
            {"protocol", "frontend", "entry", "groups", "access_order", "outputs"}
        ),
    )
    if data["protocol"] != TARGET_NORMAL_IR_PROTOCOL:
        raise TargetNormalError(
            "protocol", "F_f.protocol", f"expected {TARGET_NORMAL_IR_PROTOCOL}"
        )
    if _string(data["frontend"], "F_f.frontend") != frontend:
        raise TargetNormalError(
            "frontend", "F_f.frontend", f"must equal {frontend!r}"
        )
    if _string(data["entry"], "F_f.entry") != entry:
        raise TargetNormalError("entry", "F_f.entry", f"must equal {entry!r}")

    raw_nodes = semantic_program.get("nodes")
    if not isinstance(raw_nodes, list):
        raise TargetNormalError("semantic_nodes", "S_J.nodes", "must be an array")
    nodes: list[Mapping[str, Any]] = []
    by_node: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_nodes):
        node = _object(raw, f"S_J.nodes[{index}]")
        node_id = _string(node.get("id"), f"S_J.nodes[{index}].id")
        if node_id in by_node:
            raise TargetNormalError("semantic_duplicate", "S_J.nodes", node_id)
        nodes.append(node)
        by_node[node_id] = node
    expected_nodes = tuple(str(node["id"]) for node in nodes)

    raw_groups = data["groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise TargetNormalError("groups", "F_f.groups", "must be a non-empty array")
    groups: list[TargetNormalGroup] = []
    seen_group_ids: set[str] = set()
    flattened: list[str] = []
    for index, raw in enumerate(raw_groups):
        path = f"F_f.groups[{index}]"
        group = _closed_object(
            raw,
            path=path,
            required=frozenset({"id", "primitive", "semantic_refs"}),
        )
        group_id = _string(group["id"], f"{path}.id")
        if group_id != f"g{index}" or re.fullmatch(r"g[0-9]+", group_id) is None:
            raise TargetNormalError(
                "group_id", f"{path}.id", "must be declaration-order g0, g1, ..."
            )
        if group_id in seen_group_ids:
            raise TargetNormalError("duplicate_group", f"{path}.id", group_id)
        seen_group_ids.add(group_id)
        primitive = _string(group["primitive"], f"{path}.primitive")
        if primitive not in _PRIMITIVES:
            raise TargetNormalError(
                "unknown_primitive", f"{path}.primitive", primitive
            )
        refs = _strings(
            group["semantic_refs"],
            f"{path}.semantic_refs",
            nonempty=True,
            unique=True,
        )
        unknown = [ref for ref in refs if ref not in by_node]
        if unknown:
            raise TargetNormalError(
                "unknown_semantic_ref",
                f"{path}.semantic_refs",
                ", ".join(unknown),
            )
        _check_primitive(
            primitive,
            [by_node[ref] for ref in refs],
            f"{path}.primitive",
            masked=masked,
        )
        flattened.extend(refs)
        groups.append(TargetNormalGroup(group_id, primitive, refs))
    if tuple(flattened) != expected_nodes:
        missing = [ref for ref in expected_nodes if ref not in flattened]
        duplicates = sorted({ref for ref in flattened if flattened.count(ref) > 1})
        detail = "groups must partition S_J nodes in exact lexical order"
        if missing:
            detail += "; missing=" + ",".join(missing)
        if duplicates:
            detail += "; duplicate=" + ",".join(duplicates)
        raise TargetNormalError("node_partition", "F_f.groups", detail)

    boundary = _object(semantic_program.get("boundary", {}), "S_J.boundary")
    raw_accesses = boundary.get("accesses", [])
    if not isinstance(raw_accesses, list):
        raise TargetNormalError("semantic_accesses", "S_J.boundary.accesses", "must be an array")
    expected_accesses = tuple(
        _string(_object(item, "S_J.boundary.accesses[]").get("access_id"), "access_id")
        for item in raw_accesses
    )
    access_order = _strings(
        data["access_order"], "F_f.access_order", unique=True
    )
    if access_order != expected_accesses:
        raise TargetNormalError(
            "access_partition",
            "F_f.access_order",
            "must contain every J access exactly once in authored boundary order",
        )

    raw_outputs = semantic_program.get("output_bindings", [])
    if not isinstance(raw_outputs, list):
        raise TargetNormalError("semantic_outputs", "S_J.output_bindings", "must be an array")
    expected_outputs: list[str] = []
    for index, raw in enumerate(raw_outputs):
        item = _object(raw, f"S_J.output_bindings[{index}]")
        output = item.get("output")
        value_ref = item.get("value_ref", item.get("value", output))
        expected_outputs.append(_string(value_ref, f"S_J.output_bindings[{index}]"))
    outputs = _strings(data["outputs"], "F_f.outputs", unique=True)
    if outputs != tuple(expected_outputs):
        raise TargetNormalError(
            "outputs", "F_f.outputs", "must equal S_J output bindings in order"
        )

    return CheckedTargetNormalProgram(
        frontend=frontend,
        entry=entry,
        groups=tuple(groups),
        access_order=access_order,
        outputs=outputs,
        semantic_program=copy.deepcopy(dict(semantic_program)),
    )


def check_target_normal_json(
    text: str,
    semantic_program: Mapping[str, Any],
    *,
    frontend: str,
    entry: str,
) -> tuple[CheckedTargetNormalProgram | None, dict[str, Any]]:
    """Parse/check exact bytes and return a closed diagnostic-only receipt."""

    try:
        value = json.loads(text)
        if not isinstance(value, Mapping):
            raise TargetNormalError("object_required", "F_f", "must be an object")
        checked = check_target_normal_program(
            value, semantic_program, frontend=frontend, entry=entry
        )
    except json.JSONDecodeError as exc:
        checked = None
        diagnostic = {
            "rule_code": "invalid_json",
            "path": "F_f",
            "detail": f"line {exc.lineno} column {exc.colno}",
        }
    except TargetNormalError as exc:
        checked = None
        diagnostic = exc.to_diagnostic()
    if checked is None:
        return None, {
            "protocol": TARGET_NORMAL_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [diagnostic],
            "authority": "J-only rejection; no pi/C/Source/Gate lookup and no repair",
        }
    return checked, {
        "protocol": TARGET_NORMAL_CHECK_PROTOCOL,
        "status": "pass",
        "diagnostics": [],
        "group_count": len(checked.groups),
        "semantic_node_count": sum(len(group.semantic_refs) for group in checked.groups),
        "access_count": len(checked.access_order),
        "authority": "J-only closure check; no pi/C/Source/Gate lookup and no repair",
    }


def render_target_normal_contract() -> str:
    """Return the short public authoring contract mounted in the Agent sandbox."""

    return f"""# Target-Normal IR v1

Write exactly `output/target-normal.json`.  The root has exactly:
`protocol`, `frontend`, `entry`, `groups`, `access_order`, `outputs`.

* `protocol` is `{TARGET_NORMAL_IR_PROTOCOL}`.
* `frontend` and `entry` equal `input/frontend.json`.
* `groups` partitions every `S_J.nodes[].id` once, in exact lexical order.
  Group ids are `g0,g1,...`.  Each group has exactly `id`, `primitive`,
  `semantic_refs`.
* Normal-J primitives: `direct` (one node), `fused_elementwise`,
  `reduction_pipeline`, `contraction_pipeline`, `structured_control`.
  If `S_J.status` is `masked`, every group must instead use `unknown`; this is
  an explicit bottom/unknown negative-control instance, never guessed semantics.
* `access_order` copies every `S_J.boundary.accesses[].access_id` once in the
  authored J boundary order.
* `outputs` copies the ordered value refs in `S_J.output_bindings`.

This is a complete semantic lowering instance because every ref resolves into
the immutable typed S_J.  Do not copy operands/types, create holes, emit target
source, or mention pi, C, Source, Harness, Gate, program ids, pointers, tiles,
layouts, APIs, or physical placement.  `target_normal_check` only checks these
J-owned facts and never supplies or repairs semantics.
"""


__all__ = [
    "TARGET_NORMAL_CHECK_PROTOCOL",
    "TARGET_NORMAL_IR_PROTOCOL",
    "CheckedTargetNormalProgram",
    "TargetNormalError",
    "TargetNormalGroup",
    "check_target_normal_json",
    "check_target_normal_program",
    "render_target_normal_contract",
]
