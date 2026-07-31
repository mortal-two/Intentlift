"""Source-blind Replay writers, ingress, and public synthetic compile probe."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from intentir.extraction.providers.client import (
    DEFAULT_RESPONSES_TIMEOUT_SECONDS,
    XHIGH_RESPONSES_TIMEOUT_SECONDS,
)
from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .codegen import extract_python_module
from .compiler import (
    CompileCase,
    CompileResult,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from .cuda_codegen import extract_cuda_module
from .cute_codegen import extract_cute_module
from .model import Frontend, Gate1Artifact
from .pi import NORMALIZATION_WITNESS_SCHEMA
from .prompt import (
    assert_source_isolated_payload,
    build_rewriter_prompt,
    build_rewriter_payload,
)
from .replay_semantic_core import materialize_replay_semantics
from .tilelang_codegen import extract_tilelang_module
from .ttir import (
    TTIRFingerprint,
    TTIRParseError,
    parse_ttir,
    runtime_argument_names,
)


MINISWE_REPLAY_BACKEND = "minisweagent.intentir-replay.v32"
MINISWE_REPLAY_TURN_TIMEOUT_SECONDS = 300.0
MINISWE_REPLAY_XHIGH_TURN_TIMEOUT_SECONDS = 900.0
RESPONSES_REPLAY_BACKEND = "openai.responses.intentir-replay.v1"
STATIC_REPLAY_BACKEND = "intentir.static-replay-test.v1"
REPLAY_CHECK_PROTOCOL = "intentir.scrc.replay-check.v15"
_ANONYMOUS_HANDLE = re.compile(r"^[pc][0-9]+$")


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    function_name: str
    model: str
    response_id: str | None
    context_receipt: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int] = field(default_factory=dict)
    response_model: str | None = None
    entry_function_name: str | None = None
    response_text: str = field(default="", repr=False)
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
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
    raw_response: bytes = field(default=b"", repr=False)
    attempts: int = 1
    elapsed_ms: float = 0.0
    trajectory: Mapping[str, Any] = field(default_factory=dict, repr=False)
    backend_protocol: str = RESPONSES_REPLAY_BACKEND
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
    """Closed result of Replay ingress or its source-blind compile probe."""

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
        expected: Mapping[str, Any] | None = None,
        observed: Mapping[str, Any] | None = None,
    ) -> None:
        self.rule_code = rule_code
        self.path = path
        self.handles = handles
        self.line = line
        self.expected = dict(expected) if expected is not None else None
        self.observed = dict(observed) if observed is not None else None
        super().__init__(rule_code)


def _check_triton_exact_abi(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
) -> tuple[_ReplayABIViolation, ...]:
    """Check the exact packet-derived ABI where Triton syntax is unambiguous."""

    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    j_parameters = packet["j"]["signature"]["parameters"]
    c_slots = [
        item
        for item in packet["c"]["slots"]
        if item.get("kind") == "constexpr"
    ]
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
    specialization_carriers = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "constexpr"
        and item.get("logical_ref") is None
    }
    absent_carriers = {
        str(item["name"])
        for item in j_parameters
        if item.get("kind") == "opaque"
        and str(item.get("dtype", "")).casefold() == "none"
    }
    forbidden_carriers = {
        **{
            name: "specialization_carrier_use_forbidden"
            for name in specialization_carriers
        },
        **{name: "absent_carrier_use_forbidden" for name in absent_carriers},
    }
    expected_constexpr = j_constexpr | {str(item["name"]) for item in c_slots}
    violations: list[_ReplayABIViolation] = []
    for argument in function.args.args:
        is_constexpr = (
            isinstance(argument.annotation, ast.Attribute)
            and isinstance(argument.annotation.value, ast.Name)
            and argument.annotation.value.id == "tl"
            and argument.annotation.attr == "constexpr"
        )
        if argument.arg in expected_constexpr and not is_constexpr:
            violations.append(
                _ReplayABIViolation(
                    "constexpr_annotation_required",
                    path=f"<parameter:{argument.arg}>",
                    handles=(argument.arg,),
                    line=argument.lineno,
                )
            )
        if argument.arg not in expected_constexpr and is_constexpr:
            violations.append(
                _ReplayABIViolation(
                    "constexpr_annotation_forbidden",
                    path=f"<parameter:{argument.arg}>",
                    handles=(argument.arg,),
                    line=argument.lineno,
                )
            )
    first_carrier_load: dict[str, ast.Name] = {}
    for statement in function.body:
        for node in ast.walk(statement):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in forbidden_carriers
            ):
                first_carrier_load.setdefault(node.id, node)
    for name, node in sorted(
        first_carrier_load.items(),
        key=lambda item: (item[1].lineno, item[1].col_offset, item[0]),
    ):
        violations.append(
            _ReplayABIViolation(
                forbidden_carriers[name],
                path=f"<parameter:{name}>",
                handles=(name,),
                line=node.lineno,
            )
        )
    return tuple(violations[:20])


@dataclass(frozen=True)
class _TypedControlContract:
    path: str
    kind: str
    positive_nonempty: bool
    negative_nonempty: bool
    terminal_polarity: str | None = None
    positive_scope_id: str | None = None
    negative_scope_id: str | None = None
    positive_effect_kinds: tuple[str, ...] = ()
    negative_effect_kinds: tuple[str, ...] = ()
    required_loop_depth: int | None = None


def _packet_typed_controls(packet: Mapping[str, Any]) -> list[_TypedControlContract]:
    """Return explicitly typed J controls in lexical preorder."""

    result: list[_TypedControlContract] = []
    j = packet.get("j")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    raw_scopes = (
        j.get("scopes", [])
        if isinstance(j, Mapping) and isinstance(j.get("scopes"), list)
        else boundary.get("scopes", [])
        if isinstance(boundary, Mapping)
        else []
    )
    scopes = (
        {
            str(item.get("scope_id")): item
            for item in raw_scopes
            if isinstance(item, Mapping) and isinstance(item.get("scope_id"), str)
        }
    )
    boundary_effect_kinds = (
        {
            str(item.get("effect_id")): str(item.get("kind"))
            for item in boundary.get("effects", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("effect_id"), str)
            and isinstance(item.get("kind"), str)
        }
        if isinstance(boundary, Mapping)
        else {}
    )
    pi = packet.get("pi")
    regions = (
        {
            str(item.get("id")): item
            for item in pi.get("regions", [])
            if isinstance(item, Mapping) and isinstance(item.get("id"), str)
        }
        if isinstance(pi, Mapping)
        else {}
    )
    scope_regions: dict[str, set[str]] = {}
    if isinstance(pi, Mapping):
        for item in pi.get("instances", []):
            if not isinstance(item, Mapping):
                continue
            semantic_ref = item.get("semantic_ref")
            region = item.get("region")
            if isinstance(semantic_ref, str) and isinstance(region, str):
                scope_regions.setdefault(semantic_ref, set()).add(region)
    instances = (
        {
            str(item.get("id")): str(item.get("semantic_ref"))
            for item in pi.get("instances", [])
            if isinstance(item, Mapping)
        }
        if isinstance(pi, Mapping)
        else {}
    )
    terminal_scopes = (
        {
            instances.get(str(item.get("terminal_instance"))): instances.get(
                str(item.get("continuation_instance"))
            )
            for item in pi.get("terminal_controls", [])
            if isinstance(item, Mapping)
            and instances.get(str(item.get("terminal_instance"))) is not None
            and instances.get(str(item.get("continuation_instance"))) is not None
        }
        if isinstance(pi, Mapping)
        else {}
    )

    def loop_ancestors(region_id: str) -> tuple[str, ...] | None:
        ancestors: list[str] = []
        seen: set[str] = set()
        current: str | None = region_id
        while current is not None:
            if current in seen:
                return None
            seen.add(current)
            region = regions.get(current)
            if region is None:
                return None
            if region.get("placement") == "loop":
                ancestors.append(current)
            parent = region.get("parent")
            current = str(parent) if isinstance(parent, str) else None
        return tuple(reversed(ancestors))

    def required_loop_depth(
        positive_scope: object,
        negative_scope: object,
    ) -> int | None:
        if not isinstance(positive_scope, str) or not isinstance(negative_scope, str):
            return None
        positive_regions = scope_regions.get(positive_scope, set())
        negative_regions = scope_regions.get(negative_scope, set())
        depths = {
            ancestors
            for region_id in positive_regions | negative_regions
            if (ancestors := loop_ancestors(region_id)) is not None
        }
        if len(depths) != 1:
            return None
        return len(next(iter(depths)))

    def arm_has_authored_activity(arm: Mapping[str, Any], body: object) -> bool:
        """Include boundary-owned effects/definitions in an arm's activity.

        An ``effect_if`` arm may own observable boundary effects through its typed
        scope even when its Layer-A body is empty.  Those effects must still remain
        lexically inside the Replay arm; captures alone do not require a statement.
        """

        if isinstance(body, list) and body:
            return True
        scope_id = arm.get("scope_id")
        scope = scopes.get(str(scope_id)) if isinstance(scope_id, str) else None
        return bool(
            isinstance(scope, Mapping)
            and (scope.get("definitions") or scope.get("effects"))
        )

    def arm_effect_kinds(arm: Mapping[str, Any]) -> tuple[str, ...]:
        scope_id = arm.get("scope_id")
        scope = scopes.get(str(scope_id)) if isinstance(scope_id, str) else None
        if not isinstance(scope, Mapping):
            return ()
        return tuple(
            boundary_effect_kinds[str(effect_ref)]
            for effect_ref in scope.get("effects", [])
            if str(effect_ref) in boundary_effect_kinds
        )

    def visit(body: Any, prefix: str) -> None:
        if not isinstance(body, list):
            return
        for index, operation in enumerate(body):
            if not isinstance(operation, Mapping):
                continue
            path = f"{prefix}[{index}]"
            positive = operation.get("positive")
            negative = operation.get("negative")
            kind = operation.get("op")
            if (
                kind in {"effect_if", "value_if"}
                and isinstance(positive, Mapping)
                and isinstance(negative, Mapping)
            ):
                positive_body = positive.get("body", [])
                negative_body = negative.get("body", [])
                terminal_polarity = next(
                    (
                        polarity
                        for polarity, arm in (
                            ("positive", positive),
                            ("negative", negative),
                        )
                        if arm.get("scope_id") in terminal_scopes
                        and terminal_scopes[arm.get("scope_id")]
                        == (
                            negative.get("scope_id")
                            if polarity == "positive"
                            else positive.get("scope_id")
                        )
                    ),
                    None,
                )
                result.append(
                    _TypedControlContract(
                        path=path,
                        kind=str(kind),
                        positive_nonempty=arm_has_authored_activity(
                            positive, positive_body
                        ),
                        negative_nonempty=arm_has_authored_activity(
                            negative, negative_body
                        ),
                        terminal_polarity=terminal_polarity,
                        positive_scope_id=(
                            str(positive.get("scope_id"))
                            if positive.get("scope_id") is not None
                            else None
                        ),
                        negative_scope_id=(
                            str(negative.get("scope_id"))
                            if negative.get("scope_id") is not None
                            else None
                        ),
                        positive_effect_kinds=arm_effect_kinds(positive),
                        negative_effect_kinds=arm_effect_kinds(negative),
                        required_loop_depth=required_loop_depth(
                            positive.get("scope_id"),
                            negative.get("scope_id"),
                        ),
                    )
                )
                if terminal_polarity == "negative":
                    visit(negative_body, f"{path}.negative.body")
                    visit(positive_body, f"{path}.positive.body")
                else:
                    visit(positive_body, f"{path}.positive.body")
                    visit(negative_body, f"{path}.negative.body")
            visit(operation.get("body"), f"{path}.body")

    intent = packet.get("j", {}).get("intent", {})
    function = intent.get("intent_function", {}) if isinstance(intent, Mapping) else {}
    visit(
        function.get("body") if isinstance(function, Mapping) else None,
        "j.intent.intent_function.body",
    )
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


def _check_triton_program_id_axes(code: str, function_name: str) -> None:
    """Reject Replay code that cannot name a real Triton launch-grid axis."""

    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else ""
        )
        if called != "program_id":
            continue
        keyword_axes = [item.value for item in node.keywords if item.arg == "axis"]
        axis = (
            node.args[0]
            if len(node.args) == 1 and not keyword_axes
            else keyword_axes[0]
            if not node.args and len(keyword_axes) == 1
            else None
        )
        value = axis.value if isinstance(axis, ast.Constant) else None
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value not in {0, 1, 2}
        ):
            raise _ReplayABIViolation(
                "program_id_axis_range",
                path="<program_id-axis>",
                line=node.lineno,
            )


@dataclass(frozen=True)
class _RuntimeIfSite:
    node: ast.If
    following: tuple[ast.stmt, ...]
    loop_depth: int
    helper_local: bool = False


def _runtime_if_nodes(
    function: ast.FunctionDef,
    *,
    runtime_parameters: set[str],
    constexpr_parameters: set[str],
    base_loop_depth: int = 0,
    helper_local: bool = False,
) -> list[_RuntimeIfSite]:
    """Collect runtime ``if`` nodes while excluding closed constexpr control."""

    initial = {
        **{name: True for name in runtime_parameters},
        **{name: False for name in constexpr_parameters},
    }
    result: list[_RuntimeIfSite] = []

    def compiler_visible_loop(statement: ast.For) -> bool:
        iterator = statement.iter
        if not isinstance(iterator, ast.Call):
            return False
        function_name = (
            iterator.func.attr
            if isinstance(iterator.func, ast.Attribute)
            else iterator.func.id
            if isinstance(iterator.func, ast.Name)
            else ""
        )
        return function_name == "range"

    def bind_target(target: ast.AST, value: bool, env: dict[str, bool]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                bind_target(item, value, env)

    def literal_truth_value(expression: ast.AST) -> bool | None:
        try:
            value = ast.literal_eval(expression)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
        if isinstance(value, (bool, int, float, complex, str, bytes)) or value is None:
            return bool(value)
        return None

    def visit(
        statements: list[ast.stmt],
        env: dict[str, bool],
        loop_depth: int,
    ) -> None:
        for index, statement in enumerate(statements):
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
                literal_truth = literal_truth_value(statement.test)
                if literal_truth is not None:
                    visit(
                        statement.body if literal_truth else statement.orelse,
                        dict(env),
                        loop_depth,
                    )
                    continue
                if _expression_is_runtime(statement.test, env):
                    result.append(
                        _RuntimeIfSite(
                            statement,
                            tuple(statements[index + 1 :]),
                            loop_depth,
                            helper_local,
                        )
                    )
                visit(statement.body, dict(env), loop_depth)
                visit(statement.orelse, dict(env), loop_depth)
                continue
            if isinstance(statement, ast.For):
                visit(
                    statement.body,
                    dict(env),
                    loop_depth + int(compiler_visible_loop(statement)),
                )
                visit(statement.orelse, dict(env), loop_depth)
                continue
            if isinstance(statement, ast.While):
                visit(statement.body, dict(env), loop_depth + 1)
                visit(statement.orelse, dict(env), loop_depth)

    visit(function.body, initial, base_loop_depth)
    return result


def _direct_helper_call_depths(
    function: ast.FunctionDef,
    *,
    helper_names: set[str],
    base_loop_depth: int,
) -> list[tuple[str, int]]:
    """Return direct sibling-JIT calls with their compiler-visible loop depth."""

    result: list[tuple[str, int]] = []

    def compiler_visible_loop(statement: ast.For) -> bool:
        iterator = statement.iter
        if not isinstance(iterator, ast.Call):
            return False
        called = (
            iterator.func.attr
            if isinstance(iterator.func, ast.Attribute)
            else iterator.func.id
            if isinstance(iterator.func, ast.Name)
            else ""
        )
        return called == "range"

    def calls_in(expression: ast.AST | None, depth: int) -> None:
        if expression is None:
            return
        for node in ast.walk(expression):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in helper_names
            ):
                result.append((node.func.id, depth))

    def visit(statements: list[ast.stmt], depth: int) -> None:
        for statement in statements:
            if isinstance(statement, ast.If):
                calls_in(statement.test, depth)
                visit(statement.body, depth)
                visit(statement.orelse, depth)
            elif isinstance(statement, ast.For):
                calls_in(statement.iter, depth)
                nested_depth = depth + int(compiler_visible_loop(statement))
                visit(statement.body, nested_depth)
                visit(statement.orelse, depth)
            elif isinstance(statement, ast.While):
                calls_in(statement.test, depth)
                visit(statement.body, depth + 1)
                visit(statement.orelse, depth)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                calls_in(statement.value, depth)
            elif isinstance(statement, (ast.Expr, ast.Return)):
                calls_in(statement.value, depth)

    visit(function.body, base_loop_depth)
    return result


def _arm_is_explicitly_empty(statements: list[ast.stmt]) -> bool:
    return bool(statements) and all(isinstance(item, ast.Pass) for item in statements)


def _statements_are_nonempty(statements: list[ast.stmt] | tuple[ast.stmt, ...]) -> bool:
    return any(not isinstance(item, ast.Pass) for item in statements)


def _statement_effect_counts(
    statements: list[ast.stmt] | tuple[ast.stmt, ...],
) -> dict[str, int]:
    counts = {"read": 0, "write": 0}
    for statement in statements:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if called == "load":
                counts["read"] += 1
            elif called == "store":
                counts["write"] += 1
    return counts


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
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    function = functions[function_name]
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
    } | {
        str(item["name"])
        for item in packet["c"]["slots"]
        if item.get("kind") == "constexpr"
    }
    actual = _runtime_if_nodes(
        function,
        runtime_parameters=runtime_parameters,
        constexpr_parameters=constexpr_parameters,
    )
    helper_names = set(functions) - {function_name}
    pending = _direct_helper_call_depths(
        function,
        helper_names=helper_names,
        base_loop_depth=0,
    )
    visited_calls: set[tuple[str, int]] = set()
    while pending:
        helper_name, call_depth = pending.pop(0)
        call_key = (helper_name, call_depth)
        if call_key in visited_calls:
            continue
        visited_calls.add(call_key)
        helper = functions[helper_name]
        helper_constexpr = {
            argument.arg
            for argument in helper.args.args
            if isinstance(argument.annotation, ast.Attribute)
            and isinstance(argument.annotation.value, ast.Name)
            and argument.annotation.value.id == "tl"
            and argument.annotation.attr == "constexpr"
        }
        helper_runtime = {
            argument.arg for argument in helper.args.args
        } - helper_constexpr
        actual.extend(
            _runtime_if_nodes(
                helper,
                runtime_parameters=helper_runtime,
                constexpr_parameters=helper_constexpr,
                base_loop_depth=call_depth,
                helper_local=True,
            )
        )
        pending.extend(
            _direct_helper_call_depths(
                helper,
                helper_names=helper_names,
                base_loop_depth=call_depth,
            )
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
            line=(actual[0].node.lineno if actual else function.lineno),
            expected={"runtime_control_count": len(expected)},
            observed={"runtime_control_count": len(actual)},
        )
    for index, (site, contract) in enumerate(
        zip(actual, expected, strict=True)
    ):
        node = site.node
        if (
            contract.required_loop_depth is not None
            and site.loop_depth != contract.required_loop_depth
        ):
            raise _ReplayABIViolation(
                "typed_control_loop_depth",
                path=f"{contract.path} -> <{contract.kind}:{index}>",
                handles=tuple(
                    item
                    for item in (
                        contract.positive_scope_id,
                        contract.negative_scope_id,
                    )
                    if item is not None
                ),
                line=node.lineno,
                expected={"authored_loop_depth": contract.required_loop_depth},
                observed={"replay_loop_depth": site.loop_depth},
            )
        if contract.kind == "effect_if" and contract.terminal_polarity is not None:
            terminal_nonempty = (
                contract.positive_nonempty
                if contract.terminal_polarity == "positive"
                else contract.negative_nonempty
            )
            continuation_nonempty = (
                contract.negative_nonempty
                if contract.terminal_polarity == "positive"
                else contract.positive_nonempty
            )
            terminal_effect_kinds = (
                contract.positive_effect_kinds
                if contract.terminal_polarity == "positive"
                else contract.negative_effect_kinds
            )
            terminates_loop_iteration = bool(
                contract.required_loop_depth is not None
                and contract.required_loop_depth > 0
            )
            helper_return = bool(terminates_loop_iteration and site.helper_local)
            terminal_statement_kind = (
                ast.Return
                if helper_return or not terminates_loop_iteration
                else ast.Continue
            )
            has_terminal_statement = bool(
                node.body
                and isinstance(node.body[-1], terminal_statement_kind)
                and (
                    not isinstance(node.body[-1], ast.Return)
                    or node.body[-1].value is None
                )
            )
            terminal_prefix = (
                node.body[:-1] if has_terminal_statement else node.body
            )
            predicate_negated = isinstance(node.test, ast.UnaryOp) and isinstance(
                node.test.op, ast.Not
            )
            actual_terminal_nonempty = _statements_are_nonempty(terminal_prefix)
            actual_continuation_nonempty = _statements_are_nonempty(site.following)
            expected_terminal_effect_counts = {
                kind: terminal_effect_kinds.count(kind)
                for kind in ("read", "write")
                if kind in terminal_effect_kinds
            }
            actual_terminal_effect_counts = _statement_effect_counts(terminal_prefix)
            expected_shape = {
                "j_control_path": contract.path,
                "terminal_polarity": contract.terminal_polarity,
                "forbids_else": True,
                "terminal_arm_ends_in": (
                    "helper_bare_return"
                    if helper_return
                    else "continue"
                    if terminates_loop_iteration
                    else "bare_return"
                ),
                "terminal_arm_nonempty_before_terminator": terminal_nonempty,
                "terminal_arm_effect_counts": expected_terminal_effect_counts,
                "continuation_nonempty_after_if": continuation_nonempty,
                "continuation_contains_return": False,
            }
            observed_shape = {
                "has_else": bool(node.orelse),
                "terminal_arm_ends_in": (
                    type(node.body[-1]).__name__ if node.body else None
                ),
                "predicate_negated": predicate_negated,
                "terminal_arm_nonempty_before_terminator": actual_terminal_nonempty,
                "terminal_arm_effect_counts": actual_terminal_effect_counts,
                "continuation_nonempty_after_if": actual_continuation_nonempty,
                "continuation_contains_return": any(
                    isinstance(item, ast.Return) for item in site.following
                ),
                "continuation_statement_kinds": [
                    type(item).__name__ for item in site.following[:16]
                ],
            }
            diagnostic_path = f"{contract.path} -> <effect_if:{index}>"
            scope_handles = tuple(
                item
                for item in (
                    contract.positive_scope_id,
                    contract.negative_scope_id,
                )
                if item is not None
            )
            if node.orelse:
                raise _ReplayABIViolation(
                    "terminal_effect_if_forbids_else",
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            if not has_terminal_statement:
                raise _ReplayABIViolation(
                    (
                        "terminal_effect_if_helper_bare_return"
                        if helper_return
                        else "terminal_effect_if_continue"
                        if terminates_loop_iteration
                        else "terminal_effect_if_bare_return"
                    ),
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            if predicate_negated != (contract.terminal_polarity == "negative"):
                raise _ReplayABIViolation(
                    "terminal_effect_if_polarity",
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            if actual_terminal_nonempty != terminal_nonempty:
                raise _ReplayABIViolation(
                    "terminal_effect_if_terminal_arm",
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            if any(
                actual_terminal_effect_counts.get(kind, 0) < count
                for kind, count in expected_terminal_effect_counts.items()
            ):
                raise _ReplayABIViolation(
                    "terminal_effect_if_scope_effects",
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            if actual_continuation_nonempty != continuation_nonempty:
                raise _ReplayABIViolation(
                    "terminal_effect_if_continuation",
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            if any(isinstance(item, ast.Return) for item in site.following):
                raise _ReplayABIViolation(
                    "terminal_effect_if_continuation_return",
                    path=diagnostic_path,
                    handles=scope_handles,
                    line=node.lineno,
                    expected=expected_shape,
                    observed=observed_shape,
                )
            continue
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


@dataclass(frozen=True)
class _CompiledIterationContract:
    block_arity: int
    basis: str
    extent: str
    tile: str
    tile_value: int
    path: str


def _packet_compiled_iteration_contracts(
    packet: Mapping[str, Any],
) -> tuple[_CompiledIterationContract, ...]:
    """Resolve authored loop bases to anonymous ABI handles when unambiguous."""

    j = packet.get("j")
    c = packet.get("c")
    pi = packet.get("pi")
    if not all(isinstance(value, Mapping) for value in (j, c, pi)):
        return ()
    assert isinstance(j, Mapping) and isinstance(c, Mapping) and isinstance(pi, Mapping)
    operations = _packet_intent_ops_by_path(packet)
    node_paths = j.get("node_ids")
    node_paths = node_paths if isinstance(node_paths, Mapping) else {}
    operation_by_node: dict[str, Mapping[str, Any]] = {}
    for node, raw_path in node_paths.items():
        if not isinstance(raw_path, str):
            continue
        path = raw_path.removeprefix("intent_function.")
        operation = operations.get(path)
        if operation is not None:
            operation_by_node[str(node)] = operation

    intent = j.get("intent")
    intent = intent if isinstance(intent, Mapping) else {}
    intent_function = intent.get("intent_function")
    intent_function = (
        intent_function if isinstance(intent_function, Mapping) else {}
    )
    value_shapes: dict[str, tuple[int | str, ...]] = {}
    for key in ("inputs", "outputs"):
        for item in intent_function.get(key, []):
            if not isinstance(item, Mapping):
                continue
            name = item.get("name")
            shape = item.get("shape")
            if isinstance(name, str) and isinstance(shape, list):
                value_shapes[name] = tuple(
                    value for value in shape if isinstance(value, (int, str))
                )
    for operation in operations.values():
        result = operation.get("result")
        result_type = operation.get("type")
        shape = result_type.get("shape") if isinstance(result_type, Mapping) else None
        if isinstance(result, str) and isinstance(shape, list):
            value_shapes[result] = tuple(
                value for value in shape if isinstance(value, (int, str))
            )

    boundary = j.get("boundary")
    boundary = boundary if isinstance(boundary, Mapping) else {}
    accesses = {
        str(item.get("access_id")): item
        for item in boundary.get("accesses", [])
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }
    signature = j.get("signature")
    parameters = (
        signature.get("parameters", []) if isinstance(signature, Mapping) else []
    )
    value_dimension_relations = j.get("value_dimension_relations")
    dimension_relations = j.get("dimension_relations")

    def extent_parameter(dimension: str) -> str | None:
        extent_refs = {dimension}
        sources = {
            str(item.get("result")): str(item.get("expr", {}).get("arg"))
            for item in (
                dimension_relations if isinstance(dimension_relations, list) else []
            )
            if isinstance(item, Mapping)
            and isinstance(item.get("result"), str)
            and isinstance(item.get("expr"), Mapping)
            and isinstance(item.get("expr", {}).get("arg"), str)
        }
        frontier = [dimension]
        while frontier:
            source = sources.get(frontier.pop())
            if source is not None and source not in extent_refs:
                extent_refs.add(source)
                frontier.append(source)
        value_refs = {
            str(item.get("value_ref"))
            for item in (
                value_dimension_relations
                if isinstance(value_dimension_relations, list)
                else []
            )
            if isinstance(item, Mapping)
            and item.get("relation") == "eq"
            and item.get("dimension_ref") in extent_refs
            and isinstance(item.get("value_ref"), str)
        }
        logical_refs = extent_refs | value_refs
        matches = [
            str(item.get("name"))
            for item in parameters
            if isinstance(item, Mapping)
            and item.get("kind") in {"scalar", "constexpr"}
            and item.get("logical_ref") in logical_refs
            and isinstance(item.get("name"), str)
        ]
        return matches[0] if len(matches) == 1 else None

    def subject_dimension(subject: str, axis: int | str | None) -> str | None:
        if isinstance(axis, str) and re.fullmatch(r"x[0-9]+", axis):
            return axis
        if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
            return None
        access = accesses.get(subject)
        if access is not None:
            tensor = access.get("tensor")
            shape = value_shapes.get(str(tensor))
            if shape is not None and axis < len(shape) and isinstance(shape[axis], str):
                return shape[axis]
            return None
        operation = operation_by_node.get(subject)
        if operation is None:
            return None
        operands = operation.get("operands")
        if not isinstance(operands, list) or not operands:
            return None
        if operation.get("op") == "reduce":
            shape = value_shapes.get(str(operands[0]))
            attrs = operation.get("attrs")
            axes = attrs.get("axes") if isinstance(attrs, Mapping) else None
            if (
                shape is not None
                and isinstance(axes, list)
                and axis in axes
                and axis < len(shape)
                and isinstance(shape[axis], str)
            ):
                return shape[axis]
            return None
        if operation.get("op") != "contraction":
            return None
        attrs = operation.get("attrs")
        subscripts = attrs.get("subscripts") if isinstance(attrs, Mapping) else None
        if not isinstance(subscripts, str) or "->" not in subscripts:
            return None
        inputs_text, output_text = subscripts.split("->", 1)
        input_terms = inputs_text.split(",")
        if len(input_terms) != len(operands):
            return None
        contracted: list[str] = []
        for term in input_terms:
            for label in term:
                if label not in output_text and label not in contracted:
                    contracted.append(label)
        if axis >= len(contracted):
            return None
        label = contracted[axis]
        dimensions: set[str] = set()
        for term, operand in zip(input_terms, operands, strict=True):
            shape = value_shapes.get(str(operand))
            if shape is None or len(shape) != len(term):
                continue
            for current, extent in zip(term, shape, strict=True):
                if current == label and isinstance(extent, str):
                    dimensions.add(extent)
        return next(iter(dimensions)) if len(dimensions) == 1 else None

    instances = {
        str(item.get("id")): item
        for item in pi.get("instances", [])
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    slots = {
        str(item.get("name")): item
        for item in c.get("slots", [])
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    values = c.get("values")
    values = values if isinstance(values, Mapping) else {}
    logical_carriers = [
        item for item in pi.get("loop_carriers", []) if isinstance(item, Mapping)
    ]
    physical_carriers = [
        item
        for item in pi.get("physical_loop_carriers", [])
        if isinstance(item, Mapping)
    ]
    result: list[_CompiledIterationContract] = []

    def append(
        *,
        basis: Any,
        dimension: str | None,
        tile_ref: str | None,
        block_arity: int,
        path: str,
    ) -> None:
        if basis not in {"extent_step", "rounded_extent_step", "tile_ordinal"}:
            return
        if dimension is None or tile_ref is None:
            return
        slot = slots.get(tile_ref)
        tile_value = values.get(tile_ref)
        if (
            slot is None
            or slot.get("kind") != "constexpr"
            or not isinstance(tile_value, int)
            or isinstance(tile_value, bool)
            or tile_value <= 0
        ):
            return
        extent = extent_parameter(dimension)
        if extent is None:
            return
        result.append(
            _CompiledIterationContract(
                block_arity=block_arity,
                basis=str(basis),
                extent=extent,
                tile=tile_ref,
                tile_value=tile_value,
                path=path,
            )
        )

    for index, relation in enumerate(pi.get("iterations", [])):
        if not isinstance(relation, Mapping):
            continue
        loop_instance = instances.get(str(relation.get("loop_instance")))
        tile_instance = instances.get(str(relation.get("tile_instance")))
        loop_ref = (
            str(loop_instance.get("semantic_ref")) if loop_instance is not None else None
        )
        tile_ref = (
            str(tile_instance.get("semantic_ref")) if tile_instance is not None else None
        )
        operation = operation_by_node.get(loop_ref or "")
        trip = operation.get("trip") if isinstance(operation, Mapping) else None
        dimension = trip.get("axis") if isinstance(trip, Mapping) else None
        carrier_count = sum(
            item.get("loop_instance") == relation.get("loop_instance")
            for item in logical_carriers
        )
        append(
            basis=relation.get("induction_basis"),
            dimension=dimension if isinstance(dimension, str) else None,
            tile_ref=tile_ref,
            block_arity=1 + carrier_count,
            path=f"pi.iterations[{index}]",
        )

    for index, relation in enumerate(pi.get("physical_iterations", [])):
        if not isinstance(relation, Mapping) or relation.get("direction") != "forward":
            continue
        body_instance = instances.get(str(relation.get("body_instance")))
        tile_instance = instances.get(str(relation.get("tile_instance")))
        body_ref = (
            str(body_instance.get("semantic_ref")) if body_instance is not None else None
        )
        tile_ref = (
            str(tile_instance.get("semantic_ref")) if tile_instance is not None else None
        )
        slot = slots.get(tile_ref or "")
        dimension = (
            subject_dimension(body_ref, slot.get("axis_ref"))
            if body_ref is not None and slot is not None
            else None
        )
        region = relation.get("region")
        carrier_count = sum(
            item.get("region") == region for item in physical_carriers
        )
        append(
            basis=relation.get("induction_basis"),
            dimension=dimension,
            tile_ref=tile_ref,
            block_arity=1 + carrier_count,
            path=f"pi.physical_iterations[{index}]",
        )
    return tuple(result)


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
    dimension_relations = packet.get("j", {}).get("dimension_relations", [])
    c_slots = {
        str(item.get("name"))
        for item in packet.get("c", {}).get("slots", [])
        if isinstance(item, Mapping) and item.get("kind") == "constexpr"
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
        dimension_sources = {
            str(item.get("result")): str(item.get("expr", {}).get("arg"))
            for item in dimension_relations
            if isinstance(item, Mapping)
            and isinstance(item.get("result"), str)
            and isinstance(item.get("expr"), Mapping)
            and isinstance(item.get("expr", {}).get("arg"), str)
        } if isinstance(dimension_relations, list) else {}
        frontier = [axis]
        while frontier:
            current = frontier.pop()
            source = dimension_sources.get(current)
            if source is not None and source not in extent_refs:
                extent_refs.add(source)
                frontier.append(source)
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


def _packet_route_extent_contracts(
    packet: Mapping[str, Any],
) -> list[tuple[str, int, str, str]]:
    """Resolve authored target-local extent partitions to anonymous C handles."""

    pi = packet.get("pi")
    if not isinstance(pi, Mapping):
        return []
    instances = {
        str(item.get("id")): str(item.get("semantic_ref"))
        for item in pi.get("instances", [])
        if isinstance(item, Mapping)
    }
    result: list[tuple[str, int, str, str]] = []
    for index, relation in enumerate(pi.get("route_extent_transforms", [])):
        path = f"pi.route_extent_transforms[{index}]"
        if not isinstance(relation, Mapping):
            raise _ReplayABIViolation(
                "route_extent_transform_contract",
                path=path,
            )
        source = instances.get(str(relation.get("source")))
        divisor = relation.get("divisor")
        rounding = relation.get("rounding")
        if not (
            isinstance(source, str)
            and isinstance(divisor, int)
            and not isinstance(divisor, bool)
            and divisor > 1
            and rounding in {"floor", "ceil"}
        ):
            raise _ReplayABIViolation(
                "route_extent_transform_contract",
                path=path,
            )
        result.append((source, divisor, rounding, path))
    return result


def _is_literal_integer(expression: ast.AST, value: int) -> bool:
    return (
        isinstance(expression, ast.Constant)
        and isinstance(expression.value, int)
        and not isinstance(expression.value, bool)
        and expression.value == value
    )


def _matches_route_extent_transform(
    expression: ast.AST,
    *,
    source: str,
    divisor: int,
    rounding: str,
) -> bool:
    if rounding == "floor":
        return (
            isinstance(expression, ast.BinOp)
            and isinstance(expression.op, ast.FloorDiv)
            and isinstance(expression.left, ast.Name)
            and expression.left.id == source
            and _is_literal_integer(expression.right, divisor)
        )
    if rounding != "ceil":
        return False
    if isinstance(expression, ast.Call) and len(expression.args) == 2:
        function = expression.func
        name = (
            function.attr
            if isinstance(function, ast.Attribute)
            else function.id
            if isinstance(function, ast.Name)
            else ""
        )
        if (
            name in {"cdiv", "ceil_div"}
            and isinstance(expression.args[0], ast.Name)
            and expression.args[0].id == source
            and _is_literal_integer(expression.args[1], divisor)
        ):
            return True
    if not (
        isinstance(expression, ast.BinOp)
        and isinstance(expression.op, ast.FloorDiv)
        and _is_literal_integer(expression.right, divisor)
    ):
        return False
    numerator = expression.left
    if isinstance(numerator, ast.BinOp) and isinstance(numerator.op, ast.Add):
        return (
            isinstance(numerator.left, ast.Name)
            and numerator.left.id == source
            and _is_literal_integer(numerator.right, divisor - 1)
        ) or (
            isinstance(numerator.right, ast.Name)
            and numerator.right.id == source
            and _is_literal_integer(numerator.left, divisor - 1)
        )
    return False


def _check_triton_route_extent_contract(
    code: str,
    packet: Mapping[str, Any],
    function_name: str,
) -> None:
    expected = _packet_route_extent_contracts(packet)
    if not expected:
        return
    tree = ast.parse(code, mode="exec")
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    expressions = [node for node in ast.walk(function) if isinstance(node, ast.expr)]
    for source, divisor, rounding, path in expected:
        if not any(
            _matches_route_extent_transform(
                expression,
                source=source,
                divisor=divisor,
                rounding=rounding,
            )
            for expression in expressions
        ):
            raise _ReplayABIViolation(
                "route_extent_transform_contract",
                path=path,
                handles=(source,),
                line=function.lineno,
            )


def _closed_ingress_diagnostic(exc: Exception) -> dict[str, Any]:
    """Project an exception to bounded source-free rule/path metadata."""

    if isinstance(exc, _ReplayABIViolation):
        return {
            "rule_code": exc.rule_code,
            "path": exc.path,
            "anonymous_handles": list(exc.handles),
            "line": exc.line,
            **({"expected": exc.expected} if exc.expected is not None else {}),
            **({"observed": exc.observed} if exc.observed is not None else {}),
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
        abi_violations = _check_triton_exact_abi(code, packet, function_name)
        if abi_violations:
            return ReplayIngressResult(
                "fail",
                diagnostics=tuple(
                    _closed_ingress_diagnostic(item)
                    for item in abi_violations
                ),
            )
        _check_triton_program_id_axes(code, function_name)
        # Replay construction preflight owns syntax and the exact J/C ABI only.
        # Pi is deliberately absent from the Replay packet; realization
        # differences are handled later by Gate1's fixed normalizer.
    except Exception as exc:
        return ReplayIngressResult(
            "fail", diagnostics=(_closed_ingress_diagnostic(exc),)
        )
    return ReplayIngressResult(
        "pass",
        code=code,
        entry_function_name=entry,
    )


_SYNTHETIC_PROBE_DTYPES = {
    "bf16": "bfloat16",
    "f16": "float16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}
_COMPILE_LINE = re.compile(r"(?:CompilationError:\s*)?at\s+(\d+):\d+")
_DIAGNOSTIC_HANDLE = re.compile(r"\b[pc][0-9]+\b")


def _synthetic_probe_dtype(dtype: str) -> str:
    normalized = dtype.removeprefix("torch.").casefold()
    return _SYNTHETIC_PROBE_DTYPES.get(normalized, normalized)


def _synthetic_probe_scalar(dtype: str) -> bool | int | float | str:
    normalized = dtype.removeprefix("torch.").casefold()
    if normalized in {"bool", "i1"}:
        return True
    if normalized.startswith("f") or normalized.startswith("float") or normalized in {
        "bf16",
        "bfloat16",
        "half",
        "double",
    }:
        return 1.0
    if normalized in {"str", "string"}:
        return "synthetic"
    return 16


def build_source_blind_replay_compile_case(
    artifact: Gate1Artifact,
) -> CompileCase:
    """Build a public deterministic compile profile from J/C declarations.

    The profile deliberately never reads ``C.values``, Source, Harness values,
    the launch grid, or prior Gate evidence.  Tensor storage size is irrelevant
    to Triton's pointer type during warmup, so every pointer uses one element;
    public scalar/constexpr representatives are fixed by declared dtype.
    """

    arguments: dict[str, RuntimeArgument] = {}
    schedule: dict[str, bool | int | float | str] = {}
    for parameter in artifact.j.signature.parameters:
        kind = parameter.kind.value
        dtype = parameter.dtype or "i64"
        if kind in {"pointer", "tensor"}:
            arguments[parameter.name] = RuntimeArgument.tensor(
                (1,), _synthetic_probe_dtype(dtype)
            )
        elif kind == "scalar":
            arguments[parameter.name] = RuntimeArgument.scalar(
                _synthetic_probe_scalar(dtype),
                _synthetic_probe_dtype(dtype),
            )
        elif kind in {"none", "opaque"}:
            arguments[parameter.name] = RuntimeArgument.none()
        elif kind == "constexpr":
            schedule[parameter.name] = _synthetic_probe_scalar(dtype)
        else:  # pragma: no cover - closed enum defensive boundary
            raise ValueError(f"unsupported Replay probe parameter kind {kind!r}")
    for slot in artifact.c.slots:
        if slot.kind.value != "constexpr":
            continue
        schedule[slot.name] = _synthetic_probe_scalar(slot.value_type)
    return CompileCase(arguments, (1, 1, 1), schedule)


def _source_blind_compile_failure(result: CompileResult) -> ReplayIngressResult:
    diagnostic = result.diagnostic or "Triton frontend rejected Replay"
    lines = [line.strip() for line in diagnostic.splitlines() if line.strip()]
    summary = re.sub(r"\s+", " ", lines[-1] if lines else diagnostic)[:2048]
    line_matches = _COMPILE_LINE.findall(diagnostic)
    handles = tuple(dict.fromkeys(_DIAGNOSTIC_HANDLE.findall(summary)))[:32]
    return ReplayIngressResult(
        "fail",
        diagnostics=(
            {
                "rule_code": "synthetic_frontend_compile_failed",
                "path": f"<synthetic-frontend-compile> {summary}",
                "anonymous_handles": list(handles),
                "line": int(line_matches[-1]) if line_matches else None,
            },
        ),
    )


def _packet_transfer_direct_counts(
    packet: Mapping[str, Any],
) -> tuple[dict[str, int] | None, tuple[str, ...]]:
    """Count only legacy-pi-authored direct boundary transfer occurrences.

    This historical projection is available only when the old occurrence graph
    is actually present.  Compact pi absence is not a zero-transfer claim.
    """

    j = packet.get("j")
    pi = packet.get("pi")
    boundary = j.get("boundary") if isinstance(j, Mapping) else None
    accesses = boundary.get("accesses") if isinstance(boundary, Mapping) else None
    instances = pi.get("instances") if isinstance(pi, Mapping) else None
    if not isinstance(accesses, list) or not isinstance(instances, list):
        # The active compact pi intentionally has no occurrence graph.  Missing
        # legacy ``instances`` therefore means that direct-transfer multiplicity
        # is not projectable, not that the compiled program must contain zero
        # loads/stores.  Formal Gate1 remains responsible for those hard facts.
        return None, ()
    access_by_id = {
        str(item.get("access_id")): item
        for item in accesses
        if isinstance(item, Mapping) and isinstance(item.get("access_id"), str)
    }
    counts: dict[str, int] = {}
    refs: list[str] = []
    for instance in instances:
        if not isinstance(instance, Mapping) or instance.get(
            "mechanism"
        ) != "transfer_direct":
            continue
        semantic_ref = instance.get("semantic_ref")
        access = access_by_id.get(str(semantic_ref))
        if access is None:
            continue
        kind = access.get("kind")
        if kind not in {"load", "store"}:
            continue
        operation = f"tt.{kind}"
        if access.get("predicate") is not None:
            operation += ".masked"
        counts[operation] = counts.get(operation, 0) + 1
        refs.append(str(semantic_ref))
    return counts, tuple(dict.fromkeys(refs))


def _packet_transfer_order_constraints(
    packet: Mapping[str, Any],
) -> tuple[
    tuple[str, str, str, str, str, str, str, str],
    ...,
]:
    """Project unambiguous, same-region transfer order already authored in pi.

    Each row is ``(before_instance, after_instance, before_access,
    after_access, before_operation, after_operation, before_pointer,
    after_pointer)``.  Pointer
    roots are discovered only by walking authored pi routes and carrier edges
    back to pointer/tensor signature instances.  Ambiguous or incomplete
    topology is ignored here and remains the responsibility of formal Gate1.
    """

    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return ()
    signature = j.get("signature")
    boundary = j.get("boundary")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    accesses_value = (
        boundary.get("accesses") if isinstance(boundary, Mapping) else None
    )
    instances_value = pi.get("instances")
    routes_value = pi.get("routes")
    order_value = pi.get("order")
    if not all(
        isinstance(value, list)
        for value in (
            parameters,
            accesses_value,
            instances_value,
            routes_value,
            order_value,
        )
    ):
        return ()

    pointer_parameters = {
        str(item.get("name"))
        for item in parameters
        if isinstance(item, Mapping)
        and item.get("kind") in {"pointer", "tensor"}
        and isinstance(item.get("name"), str)
    }
    accesses = {
        str(item.get("access_id")): item
        for item in accesses_value
        if isinstance(item, Mapping)
        and isinstance(item.get("access_id"), str)
    }
    instances = {
        str(item.get("id")): item
        for item in instances_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    if not pointer_parameters or not instances:
        return ()

    incoming: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
    for route in routes_value:
        if not isinstance(route, Mapping):
            continue
        source = route.get("source")
        target = route.get("target")
        if isinstance(source, str) and isinstance(target, str) and target in incoming:
            incoming[target].add(source)
    for key in ("loop_carriers", "physical_loop_carriers"):
        relations = pi.get(key)
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, Mapping):
                continue
            init = relation.get("init")
            state = relation.get("state")
            next_value = relation.get("next")
            result = relation.get("result")
            if all(
                isinstance(value, str) and value in instances
                for value in (init, state, next_value, result)
            ):
                assert isinstance(init, str)
                assert isinstance(state, str)
                assert isinstance(next_value, str)
                assert isinstance(result, str)
                incoming[state].update((init, next_value))
                incoming[result].add(state)

    transfer_ids = {
        instance_id
        for instance_id, instance in instances.items()
        if instance.get("mechanism") == "transfer_direct"
        and str(instance.get("semantic_ref")) in accesses
    }

    def unique_pointer_root(instance_id: str) -> str | None:
        pending = [instance_id]
        visited: set[str] = set()
        roots: set[str] = set()
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            instance = instances.get(current)
            if instance is None:
                continue
            semantic_ref = instance.get("semantic_ref")
            if semantic_ref in pointer_parameters:
                roots.add(str(semantic_ref))
            pending.extend(incoming.get(current, ()))
        return next(iter(roots)) if len(roots) == 1 else None

    order_outgoing: dict[str, set[str]] = {
        instance_id: set() for instance_id in instances
    }
    for item in order_value:
        if not isinstance(item, Mapping):
            continue
        before = item.get("before")
        after = item.get("after")
        if isinstance(before, str) and isinstance(after, str):
            order_outgoing.setdefault(before, set()).add(after)

    constraints: list[
        tuple[str, str, str, str, str, str, str, str]
    ] = []
    for before in sorted(transfer_ids):
        before_region = instances[before].get("region")
        reachable: set[str] = set()
        pending = list(order_outgoing.get(before, ()))
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            current_instance = instances.get(current)
            if current_instance is None or current_instance.get(
                "region"
            ) != before_region:
                continue
            reachable.add(current)
            pending.extend(order_outgoing.get(current, ()))
        for after in sorted(reachable & transfer_ids):
            before_instance = instances[before]
            after_instance = instances[after]
            before_root = unique_pointer_root(before)
            after_root = unique_pointer_root(after)
            if before_root is None or after_root is None:
                continue
            before_access_id = str(before_instance.get("semantic_ref"))
            after_access_id = str(after_instance.get("semantic_ref"))
            before_access = accesses[before_access_id]
            after_access = accesses[after_access_id]
            before_kind = before_access.get("kind")
            after_kind = after_access.get("kind")
            if before_kind not in {"load", "store"} or after_kind not in {
                "load",
                "store",
            }:
                continue
            before_operation = f"tt.{before_kind}"
            after_operation = f"tt.{after_kind}"
            if before_access.get("predicate") is not None:
                before_operation += ".masked"
            if after_access.get("predicate") is not None:
                after_operation += ".masked"
            constraints.append(
                (
                    before,
                    after,
                    before_access_id,
                    after_access_id,
                    before_operation,
                    after_operation,
                    before_root,
                    after_root,
                )
            )
    return tuple(constraints)


def _packet_required_compiled_iteration_arities(
    packet: Mapping[str, Any],
) -> tuple[int, ...] | None:
    """Return compiler-visible loop block arities authored by J and pi.

    A structured J ``loop`` denotes one logical loop.  A
    ``pi.physical_iterations`` row denotes one additional physical loop around
    an otherwise ordinary reduce/contraction.  ``pi.iterations`` realizes its
    existing J loop and therefore is not counted a second time.  The induction
    variable contributes one block argument; only explicitly authored
    ``pi.loop_carriers`` or ``pi.physical_loop_carriers`` add carrier arguments.
    No missing loop, bound, carrier, or schedule is inferred here.
    """

    operations = _packet_intent_ops_by_path(packet)
    node_ids = packet.get("j", {}).get("node_ids", {})
    path_to_node = (
        {
            str(path).removeprefix("intent_function."): str(node)
            for node, path in node_ids.items()
        }
        if isinstance(node_ids, Mapping)
        else {}
    )
    pi = packet.get("pi")
    if not isinstance(pi, Mapping) or pi.get("schema") == NORMALIZATION_WITNESS_SCHEMA:
        # A compact normalization witness contains no physical carrier graph.
        # Assuming one induction argument per J loop would turn absence of old
        # metadata into a false exact arity claim about compiled TTIR.
        return None
    instances = {
        str(item.get("id")): str(item.get("semantic_ref"))
        for item in pi.get("instances", [])
        if isinstance(item, Mapping)
    }
    loop_carriers = [
        item for item in pi.get("loop_carriers", []) if isinstance(item, Mapping)
    ]
    physical_carriers = [
        item
        for item in pi.get("physical_loop_carriers", [])
        if isinstance(item, Mapping)
    ]

    arities: list[int] = []
    for path, operation in operations.items():
        if operation.get("op") != "loop":
            continue
        node_ref = path_to_node.get(path)
        carrier_count = sum(
            instances.get(str(item.get("loop_instance"))) == node_ref
            for item in loop_carriers
        )
        arities.append(1 + carrier_count)
    for item in pi.get("physical_iterations", []):
        if not isinstance(item, Mapping):
            continue
        region = item.get("region")
        carrier_count = sum(
            carrier.get("region") == region for carrier in physical_carriers
        )
        arities.append(1 + carrier_count)
    return tuple(sorted(arities))


def _packet_required_compiled_reduction_linearizations(
    packet: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Return authored full-axis materializations required before reductions.

    This is an exact source-blind projection of pi refinements.  It neither
    recognizes a Source API nor creates a missing representation stage.
    """

    pi = packet.get("pi")
    if not isinstance(pi, Mapping):
        return 0, ()
    refs = tuple(
        str(item.get("instance"))
        for item in pi.get("refinements", [])
        if isinstance(item, Mapping)
        and item.get("kind") == "reduction_linearization"
        and isinstance(item.get("instance"), str)
    )
    return len(refs), refs


def _packet_required_compiled_transposes(
    packet: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Return explicit transpose stages authored by J and pi."""

    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return 0, ()
    operations = _packet_intent_ops_by_path(packet)
    node_ids = j.get("node_ids")
    transpose_nodes = {
        str(node)
        for node, raw_path in node_ids.items()
        if isinstance(node_ids, Mapping)
        and isinstance(raw_path, str)
        and (
            operations.get(raw_path.removeprefix("intent_function."), {}).get("op")
            == "transpose"
        )
    } if isinstance(node_ids, Mapping) else set()
    refs = tuple(
        str(item.get("id"))
        for item in pi.get("instances", [])
        if isinstance(item, Mapping)
        and item.get("mechanism") == "realize"
        and item.get("semantic_ref") in transpose_nodes
        and isinstance(item.get("id"), str)
    )
    return len(refs), refs


def _packet_required_compiled_address_operations(
    packet: Mapping[str, Any],
) -> tuple[int | None, tuple[str, ...]]:
    """Return the pointer-address stage count closed by signature kinds and pi.

    ``address_base`` and ``address_lane`` may denote either integer address
    contributions or pointer-producing stages.  The pi grammar makes the latter
    distinction closed: a stage is a pointer operation exactly when a routed
    pointer/tensor signature carrier, or an earlier pointer-producing address
    stage, enters it.  Loop-carrier ``state`` and ``result`` instances are
    compiler block plumbing rather than address operations; their authored
    ``init -> state``, ``next -> state`` and ``state -> result`` temporal edges
    are nevertheless part of pointer reachability.  Packets whose resulting
    address topology does not expose a pointer-producing stage for every
    authored access are outside this narrow construction check; no missing
    final address operation is inferred.
    """

    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return None, ()
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    instances_value = pi.get("instances")
    routes_value = pi.get("routes")
    if not isinstance(parameters, list) or not isinstance(instances_value, list):
        return None, ()
    if not isinstance(routes_value, list):
        return None, ()

    pointer_parameters = {
        str(item.get("name"))
        for item in parameters
        if isinstance(item, Mapping)
        and item.get("kind") in {"pointer", "tensor"}
        and isinstance(item.get("name"), str)
    }
    instances = {
        str(item.get("id")): item
        for item in instances_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    address_ids = {
        instance_id
        for instance_id, item in instances.items()
        if item.get("mechanism") in {"address_base", "address_lane"}
    }
    if not address_ids:
        return None, ()

    memory_mechanisms = {
        "transfer_direct",
        "transfer_async",
        "update_atomic",
        "update_privatized",
    }
    memory_instances = [
        item
        for item in instances.values()
        if item.get("mechanism") in memory_mechanisms
    ]
    if not memory_instances or any(
        item.get("mechanism") != "transfer_direct" for item in memory_instances
    ):
        return None, ()

    incoming: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
    for route in routes_value:
        if not isinstance(route, Mapping):
            continue
        source = route.get("source")
        target = route.get("target")
        if isinstance(source, str) and isinstance(target, str) and target in incoming:
            incoming[target].add(source)

    carrier_relations = [
        item
        for key in ("loop_carriers", "physical_loop_carriers")
        for item in pi.get(key, [])
        if isinstance(item, Mapping)
    ]
    carrier_state_ids: set[str] = set()
    carrier_result_ids: set[str] = set()
    carrier_derived_ids: set[str] = set()
    for relation in carrier_relations:
        init = relation.get("init")
        state = relation.get("state")
        next_value = relation.get("next")
        result = relation.get("result")
        if not all(
            isinstance(value, str) and value in instances
            for value in (init, state, next_value, result)
        ):
            return None, ()
        assert isinstance(init, str)
        assert isinstance(state, str)
        assert isinstance(next_value, str)
        assert isinstance(result, str)
        # These are the closed temporal edges of the authored carrier.  They do
        # not invent a route or an update; they make the relation already present
        # in pi visible to this source-blind occurrence projection.
        incoming[state].update((init, next_value))
        incoming[result].add(state)
        carrier_state_ids.add(state)
        carrier_result_ids.add(result)
        carrier_derived_ids.update((state, next_value, result))

    # A pointer/tensor signature instance is already rooted in its callable ABI
    # carrier even when that same instance is an ``address_base`` receiving
    # integer offset routes.  Requiring ``mechanism=realize`` (or no incoming
    # routes) would incorrectly make such a closed pointer stage invisible and
    # silently disable the address-occurrence check.
    pointer_roots = {
        instance_id
        for instance_id, item in instances.items()
        if item.get("semantic_ref") in pointer_parameters
        and instance_id not in carrier_derived_ids
    }
    pointer_outputs = set(pointer_roots)
    pointer_passthrough_ids = address_ids | carrier_state_ids | carrier_result_ids
    changed = True
    while changed:
        changed = False
        for instance_id in pointer_passthrough_ids - pointer_outputs:
            if incoming[instance_id] & pointer_outputs:
                pointer_outputs.add(instance_id)
                changed = True

    # A carrier state is a block argument and a carrier result is a block result;
    # neither is a compiled pointer-add operation even if its typed instance uses
    # an address mechanism.  Authored init/next address stages remain counted.
    pointer_address_ids = (
        address_ids & pointer_outputs
    ) - carrier_state_ids - carrier_result_ids
    transfer_refs = {
        str(item.get("semantic_ref"))
        for item in memory_instances
        if isinstance(item.get("semantic_ref"), str)
    }
    transfer_ids = {
        str(item.get("id"))
        for item in memory_instances
        if isinstance(item.get("id"), str)
    }
    outgoing: dict[str, set[str]] = {instance_id: set() for instance_id in instances}
    for target, sources in incoming.items():
        for source in sources:
            outgoing.setdefault(source, set()).add(target)

    def reaches_any(source: str, targets: set[str]) -> bool:
        pending = [source]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in targets:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(outgoing.get(current, ()))
        return False

    if not transfer_refs or not transfer_ids:
        return None, ()
    if any(not reaches_any(source, transfer_ids) for source in pointer_address_ids):
        return None, ()
    if any(
        not any(reaches_any(source, {target}) for source in pointer_roots)
        for target in transfer_ids
    ):
        return None, ()
    return len(pointer_address_ids), tuple(sorted(transfer_refs))


def _packet_required_compiled_program_ids(
    packet: Mapping[str, Any],
) -> tuple[int | None, tuple[str, ...], tuple[int, ...] | None]:
    """Project the exact number of packet-authorized program coordinates.

    A program-region placement is only lexical scope.  Under the public Triton
    contract, a ``tl.program_id`` exists only for a J boundary domain whose
    program-region realization reaches an authored ``address_base`` stage or
    controls a physical branch region. Repeated pi instances of the same domain,
    and distinct domains bound to the same physical axis, share that coordinate;
    signature ``p*`` instances always share their already-existing callable ABI
    carrier.  This projection rejects extra compiled coordinates but never
    invents a coordinate.  In artifact v12 the authored ``pi.program_axes``
    relation supplies the exact physical axis; legacy packets retain count-only
    diagnostics and are never silently upgraded.
    """

    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return None, (), None
    boundary = j.get("boundary")
    domains_value = (
        boundary.get("domains") if isinstance(boundary, Mapping) else None
    )
    regions_value = pi.get("regions")
    instances_value = pi.get("instances")
    routes_value = pi.get("routes")
    if not isinstance(domains_value, Mapping):
        return None, (), None
    if pi.get("schema") == "intentir.pi.normalization-witness.v1":
        program_axes_value = pi.get("program_axes")
        if not isinstance(program_axes_value, list):
            return None, (), None
        domain_refs = {str(name) for name in domains_value}
        axes_by_domain = {
            str(item.get("domain_ref")): int(item["axis"])
            for item in program_axes_value
            if isinstance(item, Mapping)
            and isinstance(item.get("domain_ref"), str)
            and item.get("domain_ref") in domain_refs
            and isinstance(item.get("axis"), int)
            and not isinstance(item.get("axis"), bool)
            and 0 <= int(item["axis"]) <= 2
        }
        if len(axes_by_domain) != len(program_axes_value):
            return None, (), None
        ordered_domains = tuple(sorted(axes_by_domain))
        expected_axes = tuple(sorted(set(axes_by_domain.values())))
        return len(expected_axes), ordered_domains, expected_axes
    if not all(
        isinstance(value, list)
        for value in (regions_value, instances_value, routes_value)
    ):
        return None, (), None

    domain_refs = {str(name) for name in domains_value}
    program_regions = {
        str(item.get("id"))
        for item in regions_value
        if isinstance(item, Mapping)
        and item.get("placement") == "program"
        and isinstance(item.get("id"), str)
    }
    instances = {
        str(item.get("id")): item
        for item in instances_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    address_base_ids = {
        instance_id
        for instance_id, item in instances.items()
        if item.get("mechanism") == "address_base"
    }
    regions = {
        str(item.get("id")): item
        for item in regions_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    branch_regions = {
        region_id
        for region_id, item in regions.items()
        if item.get("placement") == "branch"
    }

    def region_is_within_branch(region_ref: str) -> bool:
        current: str | None = region_ref
        seen: set[str] = set()
        while current is not None and current not in seen:
            if current in branch_regions:
                return True
            seen.add(current)
            parent = regions.get(current, {}).get("parent")
            current = str(parent) if parent is not None else None
        return False

    outgoing: dict[str, set[str]] = {
        instance_id: set() for instance_id in instances
    }
    program_domain_branch_sources: set[str] = set()
    for route in routes_value:
        if not isinstance(route, Mapping):
            continue
        source = route.get("source")
        target = route.get("target")
        if isinstance(source, str) and isinstance(target, str):
            outgoing.setdefault(source, set()).add(target)
            source_instance = instances.get(source)
            target_instance = instances.get(target)
            if (
                route.get("slots") == []
                and source_instance is not None
                and target_instance is not None
                and source_instance.get("mechanism") == "realize"
                and source_instance.get("region") in program_regions
                and source_instance.get("semantic_ref") in domain_refs
                and region_is_within_branch(str(target_instance.get("region")))
            ):
                program_domain_branch_sources.add(source)

    def reaches_address_base(source: str) -> bool:
        pending = [source]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current in address_base_ids:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(outgoing.get(current, ()))
        return False

    authorized = {
        str(item.get("semantic_ref"))
        for instance_id, item in instances.items()
        if item.get("mechanism") == "realize"
        and item.get("region") in program_regions
        and item.get("semantic_ref") in domain_refs
        and (
            reaches_address_base(instance_id)
            or instance_id in program_domain_branch_sources
        )
    }
    ordered_domains = tuple(sorted(authorized))
    program_axes_value = pi.get("program_axes")
    if program_axes_value is None:
        return len(authorized), ordered_domains, None
    if not isinstance(program_axes_value, list):
        return None, (), None
    axes_by_domain = {
        str(item.get("domain_ref")): int(item["axis"])
        for item in program_axes_value
        if isinstance(item, Mapping)
        and isinstance(item.get("domain_ref"), str)
        and isinstance(item.get("axis"), int)
        and not isinstance(item.get("axis"), bool)
    }
    if set(axes_by_domain) != authorized:
        return len(authorized), ordered_domains, None
    expected_axes = tuple(
        sorted({axes_by_domain[domain] for domain in ordered_domains})
    )
    return len(expected_axes), ordered_domains, expected_axes


def _compiled_program_id_axes(ttir: str) -> tuple[int, ...] | None:
    """Read the literal Triton launch-axis multiset from pinned TTIR."""

    axes: list[int] = []
    axis_names = {"x": 0, "y": 1, "z": 2}
    for line in ttir.splitlines():
        if "tt.get_program_id" not in line:
            continue
        named = re.search(r"\btt\.get_program_id\s+([xyz])(?:\s|:)", line)
        if named is not None:
            axes.append(axis_names[named.group(1)])
            continue
        generic = re.search(r"\baxis\s*=\s*([0-2])(?:\s*:\s*i32)?", line)
        if generic is not None:
            axes.append(int(generic.group(1)))
            continue
        return None
    return tuple(sorted(axes))


def _packet_required_compiled_program_branches(
    packet: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Project branch regions controlled by exact empty-slot program-domain routes."""

    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return 0, ()
    boundary = j.get("boundary")
    domains = boundary.get("domains") if isinstance(boundary, Mapping) else None
    regions_value = pi.get("regions")
    instances_value = pi.get("instances")
    routes_value = pi.get("routes")
    terminal_controls_value = pi.get("terminal_controls")
    if not isinstance(domains, Mapping) or not all(
        isinstance(value, list)
        for value in (
            regions_value,
            instances_value,
            routes_value,
            terminal_controls_value,
        )
    ):
        return 0, ()
    regions = {
        str(item.get("id")): item
        for item in regions_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    instances = {
        str(item.get("id")): item
        for item in instances_value
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    program_regions = {
        region_ref
        for region_ref, item in regions.items()
        if item.get("placement") == "program"
    }

    def owning_branch(region_ref: str) -> str | None:
        current: str | None = region_ref
        seen: set[str] = set()
        while current is not None and current not in seen:
            seen.add(current)
            region = regions.get(current)
            if region is None:
                return None
            if region.get("placement") == "branch":
                return current
            parent = region.get("parent")
            current = str(parent) if parent is not None else None
        return None

    branch_by_domain: set[tuple[str, str]] = set()
    for route in routes_value:
        if not isinstance(route, Mapping) or route.get("slots") != []:
            continue
        source_ref = route.get("source")
        target_ref = route.get("target")
        if not isinstance(source_ref, str) or not isinstance(target_ref, str):
            continue
        source = instances.get(source_ref)
        target = instances.get(target_ref)
        if source is None or target is None:
            continue
        domain_ref = source.get("semantic_ref")
        branch_ref = owning_branch(str(target.get("region")))
        if (
            source.get("mechanism") == "realize"
            and source.get("region") in program_regions
            and isinstance(domain_ref, str)
            and domain_ref in domains
            and branch_ref is not None
        ):
            branch_by_domain.add((domain_ref, branch_ref))
    domain_terminals = {
        (str(item.get("domain_ref")), str(item.get("continuation_region")))
        for item in terminal_controls_value
        if isinstance(item, Mapping)
        and isinstance(item.get("domain_ref"), str)
        and isinstance(item.get("continuation_region"), str)
    }
    ordinary_branches = branch_by_domain - domain_terminals
    handles = tuple(
        f"{domain_ref}:{branch_ref}"
        for domain_ref, branch_ref in sorted(ordinary_branches)
    )
    return len(ordinary_branches), handles


def _packet_required_terminal_controls(
    packet: Mapping[str, Any],
) -> tuple[int, tuple[str, ...]]:
    """Project exact pi terminal/continuation relations without choosing syntax."""

    pi = packet.get("pi")
    if not isinstance(pi, Mapping):
        return 0, ()
    relations = pi.get("terminal_controls")
    if not isinstance(relations, list):
        return 0, ()
    handles: list[str] = []
    for item in relations:
        if not isinstance(item, Mapping):
            continue
        if isinstance(item.get("terminal_instance"), str) and isinstance(
            item.get("continuation_instance"), str
        ):
            handles.append(
                f"{item['terminal_instance']}:{item['continuation_instance']}"
            )
        elif isinstance(item.get("domain_ref"), str) and isinstance(
            item.get("continuation_region"), str
        ):
            handles.append(f"{item['domain_ref']}:{item['continuation_region']}")
    return len(handles), tuple(handles)


def _compiled_transfer_order_diagnostic(
    packet: Mapping[str, Any],
    ttir: str,
    fingerprint: TTIRFingerprint,
    *,
    path_prefix: str = "<synthetic-frontend-compile>",
) -> Mapping[str, Any] | None:
    """Reject a Replay whose own TTIR reverses an unambiguous pi order edge."""

    constraints = _packet_transfer_order_constraints(packet)
    if not constraints:
        return None
    argument_names = runtime_argument_names(ttir)
    pointer_parameters = {
        pointer
        for constraint in constraints
        for pointer in constraint[-2:]
    }
    events: list[tuple[int, str, str, tuple[str, ...]]] = []
    for index, (operation, origins, context) in enumerate(
        fingerprint.diagnostic_memory_effect_origins
    ):
        roots = {
            argument_names[position]
            for position in origins
            if position < len(argument_names)
            and argument_names[position] in pointer_parameters
        }
        if len(roots) != 1:
            continue
        events.append((index, operation, next(iter(roots)), context))

    for (
        before_instance,
        after_instance,
        before_access,
        after_access,
        before_operation,
        after_operation,
        before_pointer,
        after_pointer,
    ) in constraints:
        before_matches = [
            event
            for event in events
            if event[1] == before_operation and event[2] == before_pointer
        ]
        after_matches = [
            event
            for event in events
            if event[1] == after_operation and event[2] == after_pointer
        ]
        # Repeated accesses through one carrier cannot be assigned to individual
        # pi occurrences without inventing identity.  Leave those cases to the
        # formal observer instead of guessing.
        if len(before_matches) != 1 or len(after_matches) != 1:
            continue
        before_event = before_matches[0]
        after_event = after_matches[0]
        if before_event[3] != after_event[3]:
            continue
        if before_event[0] < after_event[0]:
            continue
        return {
            "rule_code": "compiled_transfer_order",
            "path": (
                f"{path_prefix} pi.order: "
                f"{before_access}({before_pointer}) before "
                f"{after_access}({after_pointer}); compiled lexical positions="
                f"{before_event[0]},{after_event[0]}"
            ),
            "anonymous_handles": [
                before_instance,
                after_instance,
                before_access,
                after_access,
                before_pointer,
                after_pointer,
            ],
            "line": None,
        }
    return None


def compiled_packet_obligation_diagnostics(
    packet: Mapping[str, Any],
    ttir: str,
    *,
    fingerprint: TTIRFingerprint | None = None,
    path_prefix: str = "<compiled-obligation>",
) -> tuple[Mapping[str, Any], ...]:
    """Compare closed packet obligations with one compiled TTIR observation.

    The projection is source-neutral and reject-only.  It does not choose an
    operation, address, loop, relation, or repair; callers decide whether the
    observed TTIR belongs to source-aware Lift or source-blind Replay.
    """

    observed = fingerprint if fingerprint is not None else parse_ttir(ttir)
    actual_ops = dict(observed.op_multiset)
    diagnostics: list[Mapping[str, Any]] = []

    expected_transfers, transfer_refs = _packet_transfer_direct_counts(packet)
    expected_iteration_arities = _packet_required_compiled_iteration_arities(packet)
    expected_iteration_contracts = _packet_compiled_iteration_contracts(packet)
    expected_reduction_linearizations, linearization_refs = (
        _packet_required_compiled_reduction_linearizations(packet)
    )
    expected_transposes, transpose_refs = _packet_required_compiled_transposes(packet)
    expected_address_operations, address_refs = (
        _packet_required_compiled_address_operations(packet)
    )
    expected_program_ids, program_domain_refs, expected_program_axes = (
        _packet_required_compiled_program_ids(packet)
    )
    expected_program_branches, program_branch_refs = (
        _packet_required_compiled_program_branches(packet)
    )
    expected_terminal_controls, terminal_control_refs = (
        _packet_required_terminal_controls(packet)
    )

    if expected_program_ids is not None:
        actual_program_ids = actual_ops.get("tt.get_program_id", 0)
        if expected_program_ids != actual_program_ids:
            diagnostics.append(
                {
                    "rule_code": "compiled_program_id_count",
                    "path": (
                        f"{path_prefix} tt.get_program_id: "
                        f"expected={expected_program_ids}, "
                        f"actual={actual_program_ids}"
                    ),
                    "anonymous_handles": list(program_domain_refs[:32]),
                    "line": None,
                }
            )

    if expected_program_axes is not None:
        actual_program_axes = _compiled_program_id_axes(ttir)
        if actual_program_axes is None or expected_program_axes != actual_program_axes:
            diagnostics.append(
                {
                    "rule_code": "compiled_program_id_axes",
                    "path": (
                        f"{path_prefix} tt.get_program_id axes: "
                        f"expected={list(expected_program_axes)}, "
                        f"actual={list(actual_program_axes) if actual_program_axes is not None else 'unrecognized'}"
                    ),
                    "anonymous_handles": list(program_domain_refs[:32]),
                    "line": None,
                }
            )

    actual_program_branches = actual_ops.get("control.scf.if", 0)
    if actual_program_branches < expected_program_branches:
        diagnostics.append(
            {
                "rule_code": "compiled_program_domain_branch_count",
                "path": (
                    f"{path_prefix} control.scf.if: expected-at-least="
                    f"{expected_program_branches}, actual={actual_program_branches}"
                ),
                "anonymous_handles": list(program_branch_refs[:32]),
                "line": None,
            }
        )

    actual_terminal_controls = actual_ops.get("control.conditional", 0)
    if actual_terminal_controls < expected_terminal_controls:
        diagnostics.append(
            {
                "rule_code": "compiled_terminal_control_count",
                "path": (
                    f"{path_prefix} control.conditional: expected-at-least="
                    f"{expected_terminal_controls}, actual={actual_terminal_controls}"
                ),
                "anonymous_handles": list(terminal_control_refs[:32]),
                "line": None,
            }
        )

    linearized_reduce_producers = {
        producer
        for producer, consumer, role in observed.dataflow_edges
        if producer.startswith("tt.reshape[")
        and consumer.startswith("tt.reduce[")
        and role == "operand[0]"
    }
    actual_reduction_linearizations = min(
        len(linearized_reduce_producers),
        sum(reduction.axis == 0 for reduction in observed.reductions),
    )
    if actual_reduction_linearizations < expected_reduction_linearizations:
        diagnostics.append(
            {
                "rule_code": "compiled_reduction_linearization_count",
                "path": (
                    f"{path_prefix} tt.reshape->tt.reduce(axis=0): "
                    "expected-at-least="
                    f"{expected_reduction_linearizations}, "
                    f"actual={actual_reduction_linearizations}"
                ),
                "anonymous_handles": list(linearization_refs[:32]),
                "line": None,
            }
        )

    actual_transposes = actual_ops.get("tt.trans", 0)
    if actual_transposes < expected_transposes:
        diagnostics.append(
            {
                "rule_code": "compiled_transpose_realization_count",
                "path": (
                    f"{path_prefix} tt.trans: expected-at-least="
                    f"{expected_transposes}, actual={actual_transposes}"
                ),
                "anonymous_handles": list(transpose_refs[:32]),
                "line": None,
            }
        )

    if expected_address_operations is not None:
        actual_address_operations = actual_ops.get("tt.addptr", 0)
        if expected_address_operations != actual_address_operations:
            diagnostics.append(
                {
                    "rule_code": "compiled_address_operation_count",
                    "path": (
                        f"{path_prefix} tt.addptr: expected="
                        f"{expected_address_operations}, "
                        f"actual={actual_address_operations}"
                    ),
                    "anonymous_handles": list(address_refs[:32]),
                    "line": None,
                }
            )

    if expected_iteration_arities is not None:
        actual_iteration_arities = tuple(
            sorted(loop.block_argument_count for loop in observed.loops)
        )
        if len(expected_iteration_arities) != len(actual_iteration_arities):
            diagnostics.append(
                {
                    "rule_code": "compiled_iteration_count",
                    "path": (
                        f"{path_prefix} scf.for: "
                        f"expected={len(expected_iteration_arities)}, "
                        f"actual={len(actual_iteration_arities)}"
                    ),
                    "anonymous_handles": [],
                    "line": None,
                }
            )
        elif expected_iteration_arities != actual_iteration_arities:
            diagnostics.append(
                {
                    "rule_code": "compiled_iteration_carrier_arity",
                    "path": (
                        f"{path_prefix} scf.for block-arguments: expected="
                        f"{list(expected_iteration_arities)}, "
                        f"actual={list(actual_iteration_arities)}"
                    ),
                    "anonymous_handles": [],
                    "line": None,
                }
            )

    available_iteration_bases = list(observed.diagnostic_iteration_bases)
    for contract in expected_iteration_contracts:
        compatible = [
            (index, row)
            for index, row in enumerate(available_iteration_bases)
            if row[0] == contract.block_arity
            and row[2] == contract.extent
            and row[3] == contract.tile_value
        ]
        exact = next(
            (
                (index, row)
                for index, row in compatible
                if row[1] == contract.basis
            ),
            None,
        )
        if exact is not None:
            available_iteration_bases.pop(exact[0])
            continue
        # An unclassified loop cannot prove a contradiction.  Keep the formal
        # observer responsible for it rather than guessing from a concrete
        # extent.  A classified loop over the same authored extent/tile can,
        # however, reject the wrong closed basis before a formal Gate1 round.
        if not compatible:
            continue
        actual_bases = sorted({row[1] for _, row in compatible})
        diagnostics.append(
            {
                "rule_code": "compiled_iteration_basis",
                "path": (
                    f"{path_prefix} {contract.path}.induction_basis: "
                    f"expected={contract.basis}, actual={actual_bases}"
                ),
                "anonymous_handles": [contract.extent, contract.tile],
                "line": None,
            }
        )

    if expected_transfers is not None:
        for operation in (
            "tt.load",
            "tt.load.masked",
            "tt.store",
            "tt.store.masked",
        ):
            expected = expected_transfers.get(operation, 0)
            actual = actual_ops.get(operation, 0)
            if expected != actual:
                diagnostics.append(
                    {
                        "rule_code": "compiled_transfer_direct_count",
                        "path": (
                            f"{path_prefix} {operation}: "
                            f"expected={expected}, actual={actual}"
                        ),
                        "anonymous_handles": list(transfer_refs[:32]),
                        "line": None,
                    }
                )

    transfer_order_diagnostic = _compiled_transfer_order_diagnostic(
        packet,
        ttir,
        observed,
        path_prefix=path_prefix,
    )
    if transfer_order_diagnostic is not None:
        diagnostics.append(transfer_order_diagnostic)
    return tuple(diagnostics[:20])


def source_blind_replay_compile_check(
    code: str,
    artifact: Gate1Artifact,
    function_name: str,
    *,
    compiler: TritonCompiler | None = None,
    packet: Mapping[str, Any] | None = None,
) -> ReplayIngressResult:
    """Compile Replay against the public synthetic profile, without an oracle.

    This check has no acceptance authority.  Formal Gate1 recompiles the exact
    submitted bytes against the real c0 after the agent episode ends.
    """

    selected_compiler = compiler or TritonCompiler(timeout=120.0)
    try:
        source = KernelSource.from_replay(code, function_name)
        result = selected_compiler.compile(
            source,
            build_source_blind_replay_compile_case(artifact),
        )
    except Exception as exc:
        result = CompileResult(
            False,
            diagnostic=f"{type(exc).__name__}: {exc}",
        )
    if not result.ok:
        return _source_blind_compile_failure(result)
    if result.ttir is None:
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "synthetic_replay_observer_failed",
                    "path": "<synthetic-frontend-compile>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    try:
        # This is a construction compile probe, not a semantic-membership oracle.
        # Parsing proves only that the submitted source reaches the supported raw
        # observer. Formal J/C closure remains exclusively outside the episode.
        parse_ttir(result.ttir)
    except TTIRParseError:
        return ReplayIngressResult(
            "fail",
            diagnostics=(
                {
                    "rule_code": "synthetic_replay_observer_failed",
                    "path": "<synthetic-frontend-compile>",
                    "anonymous_handles": [],
                    "line": None,
                },
            ),
        )
    return ReplayIngressResult(
        "pass",
        code=code,
        entry_function_name=function_name,
    )


ReplayCompileChecker = Callable[
    [str, Gate1Artifact, str], ReplayIngressResult
]


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
    return {
        "replay_input_protocol": str(context.get("protocol", "")),
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

    packet = build_rewriter_payload(
        artifact, frontend=frontend, masked=masked
    )
    assert_source_isolated_payload(packet)
    packet_text = json.dumps(
        packet,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    semantic_program = materialize_replay_semantics(packet["j"])
    semantic_program_text = json.dumps(
        semantic_program.to_json(),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    mask_c_text = json.dumps(
        packet["c"],
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    frontend_text = json.dumps(
        packet["u_f"],
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    instructions = """# Replay ownership contract

J is the sole semantic authority: operators, operands, constants, predicates,
domains, effects, result relations, and boundary index expressions come only from J.
C is masked and supplies only its declared anonymous slot.
`semantic-program.intent` and `semantic-program.json` are deterministic, read-only,
parameterized materializations of this episode's J. They are compiler input IR, not a
target-source skeleton or a fourth artifact. Compile the complete semantic program;
do not edit it. Source code, source
compiler IR, Harness values, Gate evidence, previous Replay, and kernel templates do
not exist in this episode.

You own the complete target program: instruction selection, address expressions,
program mapping, control, layout, memory, loops/carriers, synchronization, target API
and whole-program restructuring. Those choices may reorganize the program but may not
add, remove, or alter J-owned values, predicates, domains, accesses, effects, results,
or relations. Physical realization is not an extra semantic input. It will be compared
outside this episode after fixed normalization; no Gate observation is available here.

Before choosing physical loops or program axes, compile `boundary_ownership` as one
whole-program contract. Cross-reference shared logical axes with anonymous C
attachments, preserve every authored access/effect exactly once at the logical level,
and account for all output domains together. `domain_candidates` is a deterministic
J-only containment index; when several write domains share context axes but differ in
one output axis, co-plan one domain family and preserve each domain with a guard rather
than defaulting to independent whole-output loops. A C `tile_extent` attachment marks
a blocked logical traversal axis; do not automatically spend a launch axis on it when
the public frontend can traverse the block locally. Do not independently serialize each
output tensor into a whole-tensor loop merely because its extent is symbolic. Preserve
the authored access rank and paired coordinates unless a target restriction requires a
split; any such split must jointly realize the one authored logical access. These are
source-neutral compiler obligations derived only from J, not a target recipe.

Keep J ownership classes separate in target control. A `domain_candidates` bound is a
logical execution-domain condition, while the `predicate` inside an access row is the
only access-local mask for that access. Do not silently conjoin a missing domain bound
into every load/store mask. When a scalar program coordinate selects one member of a
write-domain family, realize each `coord < extent` domain as scalar structured control
around that effect group; keep the authored access predicate as its lane mask. Shared
outer context-domain bounds may use one outer guard or early return. Only an explicit
J access-guard relation may move a logical value guard into access-local masking.
"""
    readme = """# IntentIR source-blind Replay input

This immutable directory is the complete input allowlist for one fresh episode.

- `packet.json`: exact J + MaskC(C) + public U_f.
- `semantic-program.intent`: compact, typed, parameterized read-only S_J.
- `semantic-program.json`: exact structured form of the same S_J.
- `mask-c.json`: exact anonymous MaskC(C), with no valuation.
- `frontend.json`: public source-neutral U_f and required callable entry.
- `REPLAY_CONTRACT.md`: ownership and no-leak boundary.
- `TOOL.md`: coding, query, compile, own-IR, checkpoint, and submit commands.

Read files by path and only page sections needed for the current construction. No
Source, real C value, CompileCase/grid, Gate result, history, repository, or network
is available. S_J is derived only from this episode's J and contains no pi, C value,
target operator, API, source statement, instruction selection, or repair.
"""
    tool = """# Replay agent tools

Act like a compiler coding agent. Read `semantic-program.intent`, `frontend.json`, and
`mask-c.json` completely before drafting. Query exact J/S_J only for a named
cross-reference, and write one complete `/output/replay.source`.

```text
python .agents/intentir_agent_tool.py semantic_query overview
python .agents/intentir_agent_tool.py semantic_query ownership
python .agents/intentir_agent_tool.py semantic_query node REF
python .agents/intentir_agent_tool.py packet_query abi 0 48
python .agents/intentir_agent_tool.py packet_query graph 0 12
python .agents/intentir_agent_tool.py replay_compile_check
python .agents/intentir_agent_tool.py replay_ttir_inspect jc
python .agents/intentir_agent_tool.py submit_replay
```

Use ordinary file edits and your own helper files under `/work`. You may compile,
inspect only your current Replay IR, checkpoint and patch repeatedly within this one
episode. After construction PASS, use the bounded `jc` view; revisit your own IR when
a concrete J obligation remains, and use targeted raw/find/observer queries rather than
dumping the full fingerprint.
Remove work not authorized by S_J, and keep inactive constexpr alternatives lexically
inside their branch so the specialized compiler never materializes them.
`submit_replay` checks the exact complete
bytes but neither generates nor repairs them. Formal Gate1 runs outside this episode
and is never visible here.
"""
    return (
        {
            "README.md": readme,
            "packet.json": packet_text,
            "semantic-program.json": semantic_program_text,
            "semantic-program.intent": semantic_program.text,
            "mask-c.json": mask_c_text,
            "frontend.json": frontend_text,
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
    """Fresh bounded single-owner Replay agent; the current backend is Triton-only."""

    provider: ProviderConfig | None = None
    effort: str | None = None
    frontend: Frontend = Frontend.TRITON
    worker_config: Mapping[str, Any] = field(default_factory=dict)
    model_timeout_seconds: float | None = None
    compile_checker: ReplayCompileChecker | None = None
    stop_requested: Callable[[], int | None] | None = field(
        default=None, repr=False
    )
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
                "MiniSWE Replay v2 fails closed for non-Triton frontends"
            )
        provider = self.provider or get_provider()
        effective_effort = self.effort or provider.reasoning_effort
        timeout = self.model_timeout_seconds
        if timeout is None:
            provider_timeout = (
                XHIGH_RESPONSES_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else DEFAULT_RESPONSES_TIMEOUT_SECONDS
            )
            turn_timeout = (
                MINISWE_REPLAY_XHIGH_TURN_TIMEOUT_SECONDS
                if effective_effort.casefold() == "xhigh"
                else MINISWE_REPLAY_TURN_TIMEOUT_SECONDS
            )
            timeout = min(provider_timeout, turn_timeout)
        files, packet, instructions = _replay_input_files(
            artifact,
            frontend=self.frontend,
            masked=masked,
        )
        function_name = str(packet["u_f"]["replay_function"])
        try:
            from minisweagent.environments.intentir_replay import (
                IntentIRReplayWorkspace,
            )
            from minisweagent.run.intentir_replay import (
                IntentIRReplayRunError,
                create_intentir_replay_model,
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
        # This exact callable is used by every virtual ingress check and once
        # again by the writer after submission. It has no compiler or Source
        # dependency.
        def ingress(source_text: str) -> ReplayIngressResult:
            return replay_source_ingress(
                source_text,
                artifact=artifact,
                frontend=self.frontend,
                function_name=function_name,
                packet=packet,
            )

        probe_compiler = TritonCompiler(timeout=120.0)
        compile_checker = self.compile_checker

        def compile_check(source_text: str) -> dict[str, Any]:
            ingress_result = ingress(source_text)
            if ingress_result.status != "pass" or ingress_result.code is None:
                return ingress_result.to_check_response()
            compile_result = (
                compile_checker(
                    ingress_result.code,
                    artifact,
                    function_name,
                )
                if compile_checker is not None
                else source_blind_replay_compile_check(
                    ingress_result.code,
                    artifact,
                    function_name,
                    compiler=probe_compiler,
                    packet=packet,
                )
            )
            return compile_result.to_check_response()

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
            set_cancel_requested = getattr(model, "set_cancel_requested", None)
            if callable(set_cancel_requested):
                set_cancel_requested(self.stop_requested)
            result = run_intentir_replay(
                model=model,
                workspace=workspace,
                config_overrides=self.worker_config,
                replay_checker=lambda source_text: ingress(
                    source_text
                ).to_check_response(),
                replay_compile_checker=compile_check,
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
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                physical_provider_calls=result.physical_provider_calls,
                transport_retries=result.transport_retries,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
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
                raw_response=raw_trajectory,
                attempts=result.model_calls,
                physical_provider_calls=result.physical_provider_calls,
                transport_retries=result.transport_retries,
                elapsed_ms=result.elapsed_ms,
                trajectory=safe_trajectory,
                backend_protocol=MINISWE_REPLAY_BACKEND,
            )
            self._records.append(attempt)
            raise ReplayGenerationError(attempt)
        info = safe_trajectory.get("info", {})
        replay_check = (
            info.get("replay_check", {}) if isinstance(info, Mapping) else {}
        )
        replay_compile_check = (
            info.get("replay_compile_check", {})
            if isinstance(info, Mapping) else {}
        )
        replay_transaction = (
            info.get("replay_transaction", {})
            if isinstance(info, Mapping) else {}
        )
        context_receipt = _context_receipt({}, packet["u_f"])
        context_receipt.update(
            {
                "backend_protocol": MINISWE_REPLAY_BACKEND,
                "fresh_session": True,
                "model_calls": result.model_calls,
                "physical_provider_calls": result.physical_provider_calls,
                "transport_retries": result.transport_retries,
                "termination_status": result.exit_status,
                "replay_check": (
                    dict(replay_check)
                    if isinstance(replay_check, Mapping)
                    else {}
                ),
                "replay_compile_check": (
                    dict(replay_compile_check)
                    if isinstance(replay_compile_check, Mapping)
                    else {}
                ),
                "replay_transaction": (
                    dict(replay_transaction)
                    if isinstance(replay_transaction, Mapping)
                    else {}
                ),
                "formal_ingress_recheck": True,
                "formal_ingress_protocol": REPLAY_CHECK_PROTOCOL,
                "source_blind_synthetic_compile_probe": True,
            }
        )
        generation = ReplayGeneration(
            code=formal_ingress.code,
            function_name=function_name,
            model=provider.model,
            response_id=response_id,
            context_receipt=context_receipt,
            usage=usage,
            response_model=response_model,
            entry_function_name=formal_ingress.entry_function_name,
            response_text=result.source_text,
            raw_response=raw_trajectory,
            attempts=result.model_calls,
            physical_provider_calls=result.physical_provider_calls,
            transport_retries=result.transport_retries,
            elapsed_ms=result.elapsed_ms,
            trajectory=safe_trajectory,
            backend_protocol=MINISWE_REPLAY_BACKEND,
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
        _instructions, user = build_rewriter_prompt(
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
            context_receipt=_context_receipt({}, packet.get("u_f", {})),
            backend_protocol=STATIC_REPLAY_BACKEND,
        )


__all__ = [
    "MINISWE_REPLAY_BACKEND", "RESPONSES_REPLAY_BACKEND",
    "STATIC_REPLAY_BACKEND", "MiniSWEReplayWriter", "RecordedReplayWriter",
    "ReplayAttempt", "ReplayGeneration", "ReplayGenerationError",
    "ReplayIngressResult", "ReplayWriter", "ResponsesReplayWriter",
    "StaticReplayWriter", "build_source_blind_replay_compile_case",
    "compiled_packet_obligation_diagnostics", "replay_source_ingress",
    "source_blind_replay_compile_check",
]
