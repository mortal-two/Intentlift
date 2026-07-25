"""Source-blind Replay writers and the shared compiler-free ingress."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .codegen import extract_python_module
from .cuda_codegen import extract_cuda_module
from .cute_codegen import extract_cute_module
from .model import Frontend, Gate1Artifact
from .prompt import (
    assert_source_isolated_payload,
    build_public_frontend_context_receipt,
    build_rewriter_prompt,
    build_rewriter_payload,
    replay_packet_digest,
)
from .tilelang_codegen import extract_tilelang_module


MINISWE_REPLAY_BACKEND = "minisweagent.intentir-replay.v1"
RESPONSES_REPLAY_BACKEND = "openai.responses.intentir-replay.v1"
STATIC_REPLAY_BACKEND = "intentir.static-replay-test.v1"
REPLAY_CHECK_PROTOCOL = "intentir.scrc.replay-check.v1"
_ANONYMOUS_HANDLE = re.compile(r"^[pc][0-9]+$")


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    function_name: str
    model: str
    response_id: str | None
    packet_digest: str
    prompt_digest: str
    context_receipt: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)
    response_model: str | None = None
    entry_function_name: str | None = None
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    response_text: str = field(default="", repr=False)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
    episode_contract_sha256: str = ""
    physical_provider_calls: int = 0
    transport_retries: int = 0


@dataclass(frozen=True)
class ReplayAttempt:
    model: str
    diagnostic: str
    response_text: str = field(repr=False)
    response_id: str | None = None
    response_model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    request_sha256: str = ""
    response_sha256: str = ""
    raw_response_sha256: str = ""
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
    episode_contract_sha256: str = ""
    physical_provider_calls: int = 0
    transport_retries: int = 0


class ReplayGenerationError(RuntimeError):
    def __init__(self, attempt: ReplayAttempt) -> None:
        self.attempt = attempt
        super().__init__(f"Replay response rejected: {attempt.diagnostic}")


class ReplayWriter(Protocol):
    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration: ...


class RecordedReplayWriter(ReplayWriter, Protocol):
    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]: ...


@dataclass(frozen=True)
class ReplayIngressResult:
    """Result of fixed, compiler-free Replay ingress validation."""

    status: str
    code: str | None = field(default=None, repr=False)
    entry_function_name: str | None = None
    diagnostics: tuple[Mapping[str, Any], ...] = ()

    def to_check_response(self) -> dict[str, Any]:
        return {
            "protocol": REPLAY_CHECK_PROTOCOL,
            "status": self.status,
            "diagnostics": [dict(item) for item in self.diagnostics],
        }


class _ReplayABIViolation(ValueError):
    def __init__(
        self,
        rule_code: str,
        *,
        path: str,
        handles: tuple[str, ...] = (),
        line: int | None = None,
    ) -> None:
        self.rule_code = rule_code
        self.path = path
        self.handles = handles
        self.line = line
        super().__init__(rule_code)


def _check_triton_exact_abi(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
) -> None:
    """Check the exact packet-derived ABI where Triton syntax is unambiguous."""

    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    j_parameters = packet["j"]["signature"]["parameters"]
    c_slots = packet["c"]["slots"]
    expected_names = [
        *(str(item["name"]) for item in j_parameters),
        *(str(item["name"]) for item in c_slots),
    ]
    actual_names = [item.arg for item in function.args.args]
    if len(actual_names) != len(expected_names):
        raise _ReplayABIViolation(
            "abi_parameter_count",
            path="<entry-signature>",
            handles=tuple(expected_names[:32]),
            line=function.lineno,
        )
    if actual_names != expected_names:
        visible = tuple(
            dict.fromkeys(
                item
                for item in [*expected_names, *actual_names]
                if _ANONYMOUS_HANDLE.fullmatch(item)
            )
        )[:32]
        raise _ReplayABIViolation(
            "abi_parameter_order",
            path="<entry-signature>",
            handles=visible,
            line=function.lineno,
        )
    j_constexpr = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "constexpr"
    }
    expected_constexpr = j_constexpr | {str(item["name"]) for item in c_slots}
    for argument in function.args.args:
        is_constexpr = (
            isinstance(argument.annotation, ast.Attribute)
            and isinstance(argument.annotation.value, ast.Name)
            and argument.annotation.value.id == "tl"
            and argument.annotation.attr == "constexpr"
        )
        if argument.arg in expected_constexpr and not is_constexpr:
            raise _ReplayABIViolation(
                "constexpr_annotation_required",
                path=f"<parameter:{argument.arg}>",
                handles=(argument.arg,),
                line=argument.lineno,
            )
        if argument.arg not in expected_constexpr and is_constexpr:
            raise _ReplayABIViolation(
                "constexpr_annotation_forbidden",
                path=f"<parameter:{argument.arg}>",
                handles=(argument.arg,),
                line=argument.lineno,
            )


@dataclass(frozen=True)
class _TypedControlContract:
    kind: str
    positive_nonempty: bool
    negative_nonempty: bool


def _packet_typed_controls(packet: Mapping[str, Any]) -> list[_TypedControlContract]:
    """Return explicitly typed J controls in lexical preorder."""

    result: list[_TypedControlContract] = []

    def visit(body: Any) -> None:
        if not isinstance(body, list):
            return
        for operation in body:
            if not isinstance(operation, Mapping):
                continue
            positive = operation.get("positive")
            negative = operation.get("negative")
            kind = operation.get("op")
            if kind in {"effect_if", "value_if"}:
                positive_body = positive.get("body", [])
                negative_body = negative.get("body", [])
                result.append(
                    _TypedControlContract(
                        kind=str(kind),
                        positive_nonempty=bool(positive_body),
                        negative_nonempty=bool(negative_body),
                    )
                )
                visit(positive_body)
                visit(negative_body)
            visit(operation.get("body"))

    intent = packet.get("j", {}).get("intent", {})
    function = intent.get("intent_function", {}) if isinstance(intent, Mapping) else {}
    visit(function.get("body") if isinstance(function, Mapping) else None)
    return result


def _expression_is_runtime(
    expression: ast.AST,
    dependencies: Mapping[str, bool],
) -> bool:
    if isinstance(expression, ast.Name):
        return dependencies.get(expression.id, True)
    if isinstance(expression, ast.Constant):
        return False
    if isinstance(expression, ast.Call):
        function_name = (
            expression.func.attr
            if isinstance(expression.func, ast.Attribute)
            else expression.func.id
            if isinstance(expression.func, ast.Name)
            else ""
        )
        if function_name in {"load", "program_id", "atomic_add", "atomic_cas"}:
            return True
    return any(
        _expression_is_runtime(child, dependencies)
        for child in ast.iter_child_nodes(expression)
    )


def _runtime_if_nodes(
    function: ast.FunctionDef,
    *,
    runtime_parameters: set[str],
    constexpr_parameters: set[str],
) -> list[ast.If]:
    """Collect runtime ``if`` nodes while excluding closed constexpr control."""

    initial = {
        **{name: True for name in runtime_parameters},
        **{name: False for name in constexpr_parameters},
    }
    result: list[ast.If] = []

    def bind_target(target: ast.AST, value: bool, env: dict[str, bool]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind_target(item, value, env)

    def visit(statements: list[ast.stmt], env: dict[str, bool]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                dependency = (
                    _expression_is_runtime(value, env)
                    if value is not None
                    else False
                )
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                )
                for target in targets:
                    bind_target(target, dependency, env)
                continue
            if isinstance(statement, ast.If):
                if _expression_is_runtime(statement.test, env):
                    result.append(statement)
                visit(statement.body, dict(env))
                visit(statement.orelse, dict(env))
                continue
            if isinstance(statement, (ast.For, ast.While)):
                visit(statement.body, dict(env))
                visit(statement.orelse, dict(env))

    visit(function.body, initial)
    return result


def _arm_is_explicitly_empty(statements: list[ast.stmt]) -> bool:
    return bool(statements) and all(isinstance(item, ast.Pass) for item in statements)


def _final_simple_assignment_target(statements: list[ast.stmt]) -> str | None:
    if not statements:
        return None
    final = statements[-1]
    if (
        isinstance(final, ast.Assign)
        and len(final.targets) == 1
        and isinstance(final.targets[0], ast.Name)
    ):
        return final.targets[0].id
    if isinstance(final, ast.AnnAssign) and isinstance(final.target, ast.Name):
        return final.target.id
    return None


def _check_triton_typed_control_contract(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
) -> None:
    """Check only packet-declared runtime typed-control shells.

    Gate1 remains authoritative for operation/dataflow/effect identity and order;
    this compiler-free preflight prevents Replay from erasing or conflating an
    authored ``effect_if``/``value_if`` with select or another control kind.
    """

    expected = _packet_typed_controls(packet)
    if not expected:
        return
    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    j_parameters = packet["j"]["signature"]["parameters"]
    runtime_parameters = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") != "constexpr"
    }
    constexpr_parameters = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "constexpr"
    } | {str(item["name"]) for item in packet["c"]["slots"]}
    actual = _runtime_if_nodes(
        function,
        runtime_parameters=runtime_parameters,
        constexpr_parameters=constexpr_parameters,
    )
    if len(actual) != len(expected):
        kinds = {contract.kind for contract in expected}
        rule_code = (
            f"{next(iter(kinds))}_runtime_count"
            if len(kinds) == 1
            else "typed_control_runtime_count"
        )
        raise _ReplayABIViolation(
            rule_code,
            path="<typed-control>",
            line=(actual[0].lineno if actual else function.lineno),
        )
    for index, (node, contract) in enumerate(
        zip(actual, expected, strict=True)
    ):
        if not node.orelse:
            raise _ReplayABIViolation(
                f"{contract.kind}_explicit_negative_arm",
                path=f"<{contract.kind}:{index}>",
                line=node.lineno,
            )
        if contract.kind == "effect_if":
            positive_empty = _arm_is_explicitly_empty(node.body)
            negative_empty = _arm_is_explicitly_empty(node.orelse)
            if (
                positive_empty == contract.positive_nonempty
                or negative_empty == contract.negative_nonempty
            ):
                raise _ReplayABIViolation(
                    "effect_if_arm_polarity",
                    path=f"<effect_if:{index}>",
                    line=node.lineno,
                )
            continue
        positive_target = _final_simple_assignment_target(node.body)
        negative_target = _final_simple_assignment_target(node.orelse)
        if (
            positive_target is None
            or negative_target is None
            or positive_target != negative_target
        ):
            raise _ReplayABIViolation(
                "value_if_merge_target",
                path=f"<value_if:{index}>",
                line=node.lineno,
            )


def _packet_intent_ops_by_path(packet: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}

    def visit(body: Any, prefix: str) -> None:
        if not isinstance(body, list):
            return
        for index, operation in enumerate(body):
            if not isinstance(operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            result[path] = operation
            visit(operation.get("body"), f"{path}.body")
            for polarity in ("positive", "negative"):
                arm = operation.get(polarity)
                if isinstance(arm, Mapping):
                    visit(arm.get("body"), f"{path}.{polarity}.body")

    intent = packet.get("j", {}).get("intent", {})
    function = intent.get("intent_function", {}) if isinstance(intent, Mapping) else {}
    visit(function.get("body") if isinstance(function, Mapping) else None, "body")
    return result


def _packet_rounded_iteration_contracts(
    packet: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    """Resolve source-neutral rounded relations to packet ABI handles."""

    pi = packet.get("pi")
    if not isinstance(pi, Mapping):
        return []
    instances = {
        str(item.get("id")): str(item.get("semantic_ref"))
        for item in pi.get("instances", [])
        if isinstance(item, Mapping)
    }
    node_paths = packet.get("j", {}).get("node_ids", {})
    operations = _packet_intent_ops_by_path(packet)
    signature = packet.get("j", {}).get("signature", {})
    parameters = (
        signature.get("parameters", [])
        if isinstance(signature, Mapping)
        else []
    )
    relations = packet.get("j", {}).get("value_dimension_relations", [])
    c_slots = {
        str(item.get("name"))
        for item in packet.get("c", {}).get("slots", [])
        if isinstance(item, Mapping)
    }
    result: list[tuple[str, str, str]] = []
    for index, relation in enumerate(pi.get("iterations", [])):
        if not isinstance(relation, Mapping) or relation.get(
            "induction_basis"
        ) != "rounded_extent_step":
            continue
        loop_ref = instances.get(str(relation.get("loop_instance")))
        tile_ref = instances.get(str(relation.get("tile_instance")))
        path = node_paths.get(loop_ref) if isinstance(node_paths, Mapping) else None
        if isinstance(path, str) and path.startswith("intent_function."):
            path = path.removeprefix("intent_function.")
        operation = operations.get(path) if isinstance(path, str) else None
        trip = operation.get("trip") if isinstance(operation, Mapping) else None
        axis = trip.get("axis") if isinstance(trip, Mapping) else None
        if not isinstance(axis, str) or tile_ref not in c_slots:
            # Masked-J deliberately removes the trip semantics.  Do not guess
            # them from shape, C, or pi; the ablation may then fail Gate1.
            continue
        extent_refs = {axis}
        for item in relations if isinstance(relations, list) else []:
            if (
                isinstance(item, Mapping)
                and item.get("relation") == "eq"
                and item.get("dimension_ref") == axis
                and isinstance(item.get("value_ref"), str)
            ):
                extent_refs.add(str(item["value_ref"]))
        extent_parameters = [
            str(item["name"])
            for item in parameters
            if isinstance(item, Mapping)
            and item.get("logical_ref") in extent_refs
            and item.get("kind") in {"scalar", "constexpr"}
        ]
        if len(extent_parameters) != 1:
            continue
        result.append(
            (extent_parameters[0], str(tile_ref), f"pi.iterations[{index}]")
        )
    return result


def _for_nodes_with_assignments(
    function: ast.FunctionDef,
) -> list[tuple[ast.For, dict[str, ast.AST]]]:
    result: list[tuple[ast.For, dict[str, ast.AST]]] = []

    def bind(target: ast.AST, value: ast.AST, env: dict[str, ast.AST]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind(item, value, env)

    def visit(statements: list[ast.stmt], env: dict[str, ast.AST]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    bind(target, statement.value, env)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                bind(statement.target, statement.value, env)
            elif isinstance(statement, ast.For):
                result.append((statement, dict(env)))
                visit(statement.body, dict(env))
                visit(statement.orelse, dict(env))
            elif isinstance(statement, ast.If):
                visit(statement.body, dict(env))
                visit(statement.orelse, dict(env))

    visit(function.body, {})
    return result


def _resolved_expression(
    expression: ast.AST,
    assignments: Mapping[str, ast.AST],
    active: frozenset[str] = frozenset(),
) -> ast.AST:
    if (
        isinstance(expression, ast.Name)
        and expression.id in assignments
        and expression.id not in active
    ):
        return _resolved_expression(
            assignments[expression.id], assignments, active | {expression.id}
        )
    return expression


def _is_named(
    expression: ast.AST,
    name: str,
    assignments: Mapping[str, ast.AST],
) -> bool:
    expression = _resolved_expression(expression, assignments)
    return isinstance(expression, ast.Name) and expression.id == name


def _is_ceil_div(
    expression: ast.AST,
    extent: str,
    tile: str,
    assignments: Mapping[str, ast.AST],
) -> bool:
    expression = _resolved_expression(expression, assignments)
    if isinstance(expression, ast.Call) and len(expression.args) == 2:
        function = expression.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else ""
        )
        return name in {"cdiv", "ceil_div"} and _is_named(
            expression.args[0], extent, assignments
        ) and _is_named(expression.args[1], tile, assignments)
    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.FloorDiv
    ):
        return False
    if not _is_named(expression.right, tile, assignments):
        return False
    numerator = _resolved_expression(expression.left, assignments)
    if not isinstance(numerator, ast.BinOp) or not isinstance(
        numerator.op, ast.Sub
    ):
        return False
    if not isinstance(numerator.right, ast.Constant) or numerator.right.value != 1:
        return False
    summed = _resolved_expression(numerator.left, assignments)
    return (
        isinstance(summed, ast.BinOp)
        and isinstance(summed.op, ast.Add)
        and (
            _is_named(summed.left, extent, assignments)
            and _is_named(summed.right, tile, assignments)
            or _is_named(summed.left, tile, assignments)
            and _is_named(summed.right, extent, assignments)
        )
    )


def _is_rounded_upper(
    expression: ast.AST,
    extent: str,
    tile: str,
    assignments: Mapping[str, ast.AST],
) -> bool:
    expression = _resolved_expression(expression, assignments)
    if not isinstance(expression, ast.BinOp) or not isinstance(
        expression.op, ast.Mult
    ):
        return False
    return (
        _is_ceil_div(expression.left, extent, tile, assignments)
        and _is_named(expression.right, tile, assignments)
        or _is_named(expression.left, tile, assignments)
        and _is_ceil_div(expression.right, extent, tile, assignments)
    )


def _for_matches_rounded_contract(
    node: ast.For,
    assignments: Mapping[str, ast.AST],
    *,
    extent: str,
    tile: str,
) -> bool:
    iterator = _resolved_expression(node.iter, assignments)
    if not isinstance(iterator, ast.Call) or len(iterator.args) != 3:
        return False
    function = iterator.func
    name = (
        function.attr
        if isinstance(function, ast.Attribute)
        else function.id
        if isinstance(function, ast.Name)
        else ""
    )
    return (
        name == "range"
        and isinstance(iterator.args[0], ast.Constant)
        and iterator.args[0].value == 0
        and _is_rounded_upper(iterator.args[1], extent, tile, assignments)
        and _is_named(iterator.args[2], tile, assignments)
    )


def _check_triton_rounded_iteration_contract(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
) -> None:
    expected = _packet_rounded_iteration_contracts(packet)
    if not expected:
        return
    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    available = _for_nodes_with_assignments(function)
    consumed: set[int] = set()
    for extent, tile, path in expected:
        match = next(
            (
                index
                for index, (node, assignments) in enumerate(available)
                if index not in consumed
                and _for_matches_rounded_contract(
                    node, assignments, extent=extent, tile=tile
                )
            ),
            None,
        )
        if match is None:
            raise _ReplayABIViolation(
                "rounded_extent_step_contract",
                path=path,
                handles=(extent, tile),
                line=function.lineno,
            )
        consumed.add(match)


def _closed_ingress_diagnostic(exc: Exception) -> dict[str, Any]:
    """Project an exception to bounded source-free rule/path metadata."""

    if isinstance(exc, _ReplayABIViolation):
        return {
            "rule_code": exc.rule_code,
            "path": exc.path,
            "anonymous_handles": list(exc.handles),
            "line": exc.line,
        }
    line_match = re.search(r"\bline\s+(\d+)\b", str(exc))
    line = int(line_match.group(1)) if line_match is not None else None
    return {
        "rule_code": "replay_static_policy_rejected",
        "path": f"<line:{line}>" if line is not None else "<replay>",
        "anonymous_handles": [],
        "line": line,
    }


def _extract_replay_code(
    response: str,
    *,
    frontend: Frontend,
    function_name: str,
) -> tuple[str, str | None]:
    if frontend is Frontend.TRITON:
        return (
            extract_python_module(
                response,
                required_function=function_name,
            ),
            None,
        )
    if frontend is Frontend.CUTE:
        entry = "scrc_entry"
        return (
            extract_cute_module(
                response,
                required_kernel=function_name,
                required_entry=entry,
            ),
            entry,
        )
    if frontend is Frontend.TILELANG:
        return (
            extract_tilelang_module(
                response,
                required_function=function_name,
            ),
            None,
        )
    if frontend is Frontend.CUDA:
        return (
            extract_cuda_module(
                response,
                required_kernel=function_name,
            ),
            None,
        )
    raise ValueError(
        f"source-blind Replay has no audited validator for {frontend.value!r}"
    )


def replay_source_ingress(
    source: str,
    *,
    artifact: Gate1Artifact,
    frontend: Frontend | str,
    function_name: str = "scrc_k0",
    packet: Mapping[str, Any] | None = None,
) -> ReplayIngressResult:
    """Run the one host-owned static ingress used by check and submission.

    This path deliberately does not invoke a frontend compiler, Gate1 observer,
    execution harness, or any source-side view.
    """

    try:
        selected_frontend = Frontend(frontend)
    except (TypeError, ValueError):
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "frontend_not_supported",
                    "path": "<frontend>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    if selected_frontend is not Frontend.TRITON:
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "frontend_not_supported",
                    "path": "<frontend>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    if packet is None:
        packet = build_rewriter_payload(
            artifact,
            frontend=selected_frontend,
            masked=False,
        )
    assert_source_isolated_payload(packet)
    try:
        code, entry = _extract_replay_code(
            source,
            frontend=selected_frontend,
            function_name=function_name,
        )
        _check_triton_exact_abi(code, packet, function_name)
        _check_triton_typed_control_contract(code, packet, function_name)
        _check_triton_rounded_iteration_contract(code, packet, function_name)
    except Exception as exc:
        return ReplayIngressResult(
            "fail", diagnostics=(_closed_ingress_diagnostic(exc),)
        )
    return ReplayIngressResult(
        "pass",
        code=code,
        entry_function_name=entry,
    )


def _usage(raw: Mapping[str, Any]) -> dict[str, int]:
    value = raw.get("usage")
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[key] = item
    return result


def _context_receipt(raw: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    trace: list[dict[str, str]] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and "search" in item_type:
                trace.append(
                    {
                        "type": item_type,
                        "id": item.get("id") if isinstance(item.get("id"), str) else "",
                    }
                )
    receipt = build_public_frontend_context_receipt(context)
    return {
        "replay_input_protocol": str(context.get("protocol", "")),
        **receipt,
        "retrieval_trace": trace,
    }


def _agent_usage_and_identity(
    trajectory: Mapping[str, Any],
) -> tuple[dict[str, int], str | None, str | None]:
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    seen = False
    response_id: str | None = None
    response_model: str | None = None
    messages = trajectory.get("messages", ())
    if not isinstance(messages, list):
        return {}, None, None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("object") != "response":
            continue
        if isinstance(message.get("id"), str):
            response_id = message["id"]
        if isinstance(message.get("model"), str):
            response_model = message["model"]
        usage = message.get("usage")
        if not isinstance(usage, Mapping):
            continue
        seen = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                totals[key] += value
    return (totals if seen else {}), response_id, response_model


def _sanitize_agent_trajectory(value: Any, secret: str, *, key: str = "") -> Any:
    """Apply the same recursive secret-key and literal redaction as Lift."""

    normalized_key = key.casefold().replace("-", "_")
    if normalized_key in {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    } or normalized_key.endswith("_api_key"):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_agent_trajectory(
                item, secret, key=str(item_key)
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_agent_trajectory(item, secret) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_agent_trajectory(item, secret) for item in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, "<redacted>")
    return value


def _trajectory_bytes(trajectory: Mapping[str, Any]) -> bytes:
    return json.dumps(
        trajectory,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _replay_input_files(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend,
    masked: bool,
) -> tuple[dict[str, str], dict[str, Any], str]:
    """Build the complete, closed allowlist mounted for one agent episode."""

    instructions, user = build_rewriter_prompt(
        artifact, frontend=frontend, masked=masked
    )
    packet = json.loads(user)
    assert_source_isolated_payload(packet)
    packet_text = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    readme = """# IntentIR source-blind Replay input

This is the complete immutable input allowlist for one fresh Replay episode.

- `packet.json`: exactly source-isolated J + pi + MaskC(C) + U_f.
- `REPLAY_CONTRACT.md`: fixed public Replay construction contract.
- `TOOL.md`: fixed bounded check and submission interface.
- `intentir_packet_query`: bounded read-only packet inventory, path lookup, and
  anonymous-handle search plus paged graph/realization views. It exposes only
  packet.json and never writes Replay.

No Source or source slice, Harness/compiler view, actual C values, CompileCase,
grid, Gate diff/TTIR, previous Replay, sibling trajectory, or network is
available. Generate the entire source yourself. Fixed code only checks and
rejects; it never generates, patches, or certifies source.

Start with `intentir_packet_query overview`; then follow its exact `abi`,
`graph`, `realization`, and paged `edges` commands. These views expose the
complete packet facts in bounded pages. Use `section PATH`
only for one named omitted field; never concatenate inputs because the transport
may elide its middle.

Before ReplayCheck, inspect packet control mechanically: every J `effect_if`
must remain one runtime `if` with explicit polarity-preserving arms (including
an explicit empty arm); every J `value_if` must remain a distinct runtime `if`
whose explicit arms finally assign their yielded values to one common merge
local; and every pi `rounded_extent_step` must use its declared J trip extent
and matching C tile as `range(0, ceil_div(extent,tile)*tile, tile)`. `effect_if`,
`value_if`, and select are hard-distinct. Never substitute select/predicated
unconditional execution, current-shape guesses, or a second rounding formula.
"""
    tool = """# Replay tool contract

Write exactly `/output/replay.source`. To request bounded static diagnostics,
run exactly `intentir_replay_check /output/replay.source` as the complete shell
action. The seven ordinary evaluations share one budget; submission has one
reserved evaluation when exact current bytes have no prior PASS. Results contain
only a closed protocol/status/rule/path plus anonymous handles or line number.
No compiler, Gate1 observer, execution, or source oracle is run. A PASS has no
acceptance authority. Submit with a separate `printf 'INTENTIR_REPLAY_SUBMIT\\n'`.
The writer independently reruns the same ingress callable after submission.
"""
    return (
        {
            "README.md": readme,
            "packet.json": packet_text,
            "REPLAY_CONTRACT.md": instructions + "\n",
            "TOOL.md": tool,
        },
        packet,
        instructions,
    )


@dataclass
class ResponsesReplayWriter:
    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        instructions, user = build_rewriter_prompt(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        packet = json.loads(user)
        u_f = packet["u_f"]
        function_name = u_f["replay_function"]
        frontend = Frontend(u_f["frontend"])
        if frontend not in {
            Frontend.TRITON,
            Frontend.CUTE,
            Frontend.TILELANG,
            Frontend.CUDA,
        }:
            raise ValueError(
                f"source-blind Replay has no audited validator for {frontend.value!r}"
            )
        reply = call_responses(
            instructions,
            user,
            provider=self.provider,
            effort=self.effort,
        )
        prompt_digest = hashlib.sha256(
            (instructions + "\0" + user).encode()
        ).hexdigest()
        try:
            code, entry_function_name = _extract_replay_code(
                reply.text,
                frontend=frontend,
                function_name=function_name,
            )
        except Exception as exc:
            attempt = ReplayAttempt(
                model=reply.model,
                diagnostic=f"{type(exc).__name__}: {exc}",
                response_text=reply.text,
                response_id=(
                    reply.raw.get("id")
                    if isinstance(reply.raw.get("id"), str)
                    else None
                ),
                response_model=(
                    reply.raw.get("model")
                    if isinstance(reply.raw.get("model"), str)
                    else None
                ),
                usage=_usage(reply.raw),
                request_sha256=reply.request_sha256,
                response_sha256=(
                    reply.response_sha256
                    or hashlib.sha256(reply.text.encode()).hexdigest()
                ),
                raw_response_sha256=reply.raw_response_sha256,
                raw_response=reply.raw_response,
                attempts=reply.attempts,
                physical_provider_calls=reply.attempts,
                transport_retries=max(0, reply.attempts - 1),
                elapsed_ms=reply.elapsed_ms,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        generation = ReplayGeneration(
            code=code,
            function_name=function_name,
            model=reply.model,
            response_id=(
                reply.raw.get("id")
                if isinstance(reply.raw.get("id"), str) else None
            ),
            packet_digest=replay_packet_digest(packet),
            prompt_digest=prompt_digest,
            context_receipt=_context_receipt(
                reply.raw, packet.get("u_f", {})
            ),
            usage=_usage(reply.raw),
            response_model=(
                reply.raw.get("model")
                if isinstance(reply.raw.get("model"), str)
                else None
            ),
            entry_function_name=entry_function_name,
            request_sha256=reply.request_sha256,
            response_sha256=(
                reply.response_sha256
                or hashlib.sha256(reply.text.encode()).hexdigest()
            ),
            raw_response_sha256=reply.raw_response_sha256,
            response_text=reply.text,
            raw_response=reply.raw_response,
            attempts=reply.attempts,
            physical_provider_calls=reply.attempts,
            transport_retries=max(0, reply.attempts - 1),
            elapsed_ms=reply.elapsed_ms,
        )
        self._records.append(generation)
        return generation


@dataclass
class MiniSWEReplayWriter:
    """Fresh bounded multi-turn Replay agent; v1 is Triton-only."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    _records: list[ReplayGeneration | ReplayAttempt] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def records(self) -> tuple[ReplayGeneration | ReplayAttempt, ...]:
        return tuple(self._records)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        if self.frontend is not Frontend.TRITON:
            raise ValueError(
                "MiniSWE Replay v1 fails closed for non-Triton frontends"
            )
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = self.model_timeout_seconds
        if timeout is None:
            timeout = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
        files, packet, instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        packet_digest = replay_packet_digest(packet)
        function_name = str(packet["u_f"]["replay_function"])
        prompt_digest = hashlib.sha256(
            (
                instructions
                + "\0"
                + json.dumps(
                    packet,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            ).encode()
        ).hexdigest()
        try:
            from minisweagent.environments.intentir_replay import (
                IntentIRReplayWorkspace,
            )
            from minisweagent.run.intentir_replay import (
                IntentIRReplayRunError,
                create_intentir_replay_model,
                intentir_replay_episode_contract_sha256,
                run_intentir_replay,
                validated_intentir_replay_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MiniSWE Replay backend requires the lift-agent dependencies"
            ) from exc

        effective_worker_config = validated_intentir_replay_config(
            self.worker_config
        )
        episode_contract_sha256 = intentir_replay_episode_contract_sha256(
            files,
            effective_worker_config,
        )

        # This exact callable is used by every virtual check and once again by
        # the writer after submission. It has no compiler or Source dependency.
        def ingress(source_text: str) -> ReplayIngressResult:
            return replay_source_ingress(
                source_text,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )

        workspace = IntentIRReplayWorkspace.create(files)
        result = None
        formal_ingress: ReplayIngressResult | None = None
        try:
            model = create_intentir_replay_model(
                model_name=provider.model,
                api_base=provider.base_url,
                api_key=provider.api_key,
                reasoning_effort=effective_effort,
                timeout_seconds=timeout,
            )
            result = run_intentir_replay(
                model=model,
                workspace=workspace,
                config_overrides=self.worker_config,
                replay_checker=lambda source_text: ingress(
                    source_text
                ).to_check_response(),
            )
            assert result.source_text is not None
            formal_ingress = ingress(result.source_text)
        except IntentIRReplayRunError as exc:
            result = exc.result
            safe_trajectory = _sanitize_agent_trajectory(
                result.trajectory, provider.api_key
            )
            raw_trajectory = _trajectory_bytes(safe_trajectory)
            usage, response_id, response_model = _agent_usage_and_identity(
                safe_trajectory
            )
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=(
                    "Replay agent terminated without a formal submission: "
                    + result.exit_status
                ),
                response_text=result.source_text or "",
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                request_sha256=prompt_digest,
                response_sha256=(
                    hashlib.sha256((result.source_text or "").encode()).hexdigest()
                    if result.source_text is not None
                    else ""
                ),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                physical_provider_calls=result.physical_provider_calls,
                transport_retries=result.transport_retries,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt) from exc
        except Exception as exc:
            infrastructure_trajectory = {
                "info": {
                    "backend": MINISWE_REPLAY_BACKEND,
                    "fresh_session": True,
                    "exit_status": "InfrastructureError",
                    "exception_type": type(exc).__name__,
                }
            }
            raw_trajectory = _trajectory_bytes(infrastructure_trajectory)
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=(
                    "Replay infrastructure failure: " + type(exc).__name__
                ),
                response_text=(
                    result.source_text
                    if result is not None and result.source_text is not None
                    else ""
                ),
                request_sha256=prompt_digest,
                response_sha256=(
                    hashlib.sha256(result.source_text.encode()).hexdigest()
                    if result is not None and result.source_text is not None
                    else ""
                ),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls if result is not None else 0,
                physical_provider_calls=(
                    result.physical_provider_calls if result is not None else 0
                ),
                transport_retries=(
                    result.transport_retries if result is not None else 0
                ),
                elapsed_ms=result.elapsed_ms if result is not None else 0.0,
                trajectory=infrastructure_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(attempt)
            raise
        finally:
            workspace.cleanup()

        assert result is not None and result.source_text is not None
        assert formal_ingress is not None
        safe_trajectory = _sanitize_agent_trajectory(
            result.trajectory, provider.api_key
        )
        raw_trajectory = _trajectory_bytes(safe_trajectory)
        usage, response_id, response_model = _agent_usage_and_identity(
            safe_trajectory
        )
        if formal_ingress.status != "pass" or formal_ingress.code is None:
            attempt = ReplayAttempt(
                model=provider.model,
                diagnostic=json.dumps(
                    formal_ingress.to_check_response(), sort_keys=True
                ),
                response_text=result.source_text,
                response_id=response_id,
                response_model=response_model,
                usage=usage,
                request_sha256=prompt_digest,
                response_sha256=hashlib.sha256(
                    result.source_text.encode()
                ).hexdigest(),
                raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                physical_provider_calls=result.physical_provider_calls,
                transport_retries=result.transport_retries,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
                episode_contract_sha256=episode_contract_sha256,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt)
        info = safe_trajectory.get("info", {})
        replay_check = (
            info.get("replay_check", {}) if isinstance(info, Mapping) else {}
        )
        context_receipt = _context_receipt({}, packet["u_f"])
        context_receipt.update(
            {
                "backend_protocol": MINISWE_REPLAY_BACKEND,
                "fresh_session": True,
                "model_calls": result.model_calls,
                "physical_provider_calls": result.physical_provider_calls,
                "transport_retries": result.transport_retries,
                "packet_digest": packet_digest,
                "replay_check": (
                    dict(replay_check)
                    if isinstance(replay_check, Mapping)
                    else {}
                ),
                "formal_ingress_recheck": True,
                "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                "episode_contract_sha256": episode_contract_sha256,
            }
        )
        generation = ReplayGeneration(
            code=formal_ingress.code,
            function_name=function_name,
            model=provider.model,
            response_id=response_id,
            packet_digest=packet_digest,
            prompt_digest=prompt_digest,
            context_receipt=context_receipt,
            usage=usage,
            response_model=response_model,
            entry_function_name=formal_ingress.entry_function_name,
            request_sha256=prompt_digest,
            response_sha256=hashlib.sha256(
                result.source_text.encode()
            ).hexdigest(),
            raw_response_sha256=hashlib.sha256(raw_trajectory).hexdigest(),
            response_text=result.source_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            physical_provider_calls=result.physical_provider_calls,
            transport_retries=result.transport_retries,
            elapsed_ms=result.elapsed_ms,
            trajectory=safe_trajectory,
            backend_protocol=MINISWE_REPLAY_BACKEND,
            episode_contract_sha256=episode_contract_sha256,
        )
        self._records.append(generation)
        return generation


@dataclass
class StaticReplayWriter:
    code: str
    model: str = "static-test-writer"
    frontend: Frontend = Frontend.TRITON
    calls: list[bool] = field(default_factory=list, init=False)

    def generate(
        self, artifact: Gate1Artifact, *, masked: bool = False
    ) -> ReplayGeneration:
        instructions, user = build_rewriter_prompt(
            artifact, frontend=self.frontend, masked=masked
        )
        packet = json.loads(user)
        u_f = packet["u_f"]
        function_name = u_f["replay_function"]
        frontend = Frontend(u_f["frontend"])
        code, entry_function_name = _extract_replay_code(
            self.code,
            frontend=frontend,
            function_name=function_name,
        )
        self.calls.append(masked)
        return ReplayGeneration(
            code=code,
            function_name=function_name,
            model=self.model,
            response_id=None,
            entry_function_name=entry_function_name,
            packet_digest=replay_packet_digest(packet),
            prompt_digest=hashlib.sha256(
                (instructions + "\0" + user).encode()
            ).hexdigest(),
            context_receipt=_context_receipt({}, packet.get("u_f", {})),
            backend_protocol=STATIC_REPLAY_BACKEND,
        )


__all__ = [
    "MINISWE_REPLAY_BACKEND", "RESPONSES_REPLAY_BACKEND",
    "STATIC_REPLAY_BACKEND", "MiniSWEReplayWriter", "RecordedReplayWriter",
    "ReplayAttempt", "ReplayGeneration", "ReplayGenerationError",
    "ReplayIngressResult", "ReplayWriter", "ResponsesReplayWriter",
    "StaticReplayWriter", "replay_source_ingress",
]
