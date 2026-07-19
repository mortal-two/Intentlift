"""Replay-agent adapters for SCRC Gate1.

The replay writer is intentionally handed only ``Gate1Artifact.rewriter_view``
through :mod:`.prompt`; source text, location maps and concrete C values never
enter this API.  The deterministic compiler/comparator remain independent of
which model (or static fixture) proposes the replay code.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from intentir.extraction.providers.client import ProviderError, call_responses
from intentir.extraction.providers.provider import ProviderConfig, get_provider

from .codegen import (
    UnsafeReplayCodeError,
    extract_python_module,
    validate_triton_module,
)
from .model import (
    Gate1Artifact,
    ParameterKind,
    ReplayPlan,
    opaque_replay_abi,
)
from .prompt import build_rewriter_prompt
from .rules import RuleRegistry


@dataclass(frozen=True)
class ReplayGeneration:
    code: str
    prompt_sha256: str
    requested_model: str
    generation_id: str
    code_sha256: str
    response_model: str | None = None
    response_id: str | None = None
    masked: bool = False
    ablation_kind: str | None = None


class ReplayWriter(Protocol):
    def generate(
        self,
        artifact: Gate1Artifact,
        registry: RuleRegistry,
        *,
        masked: bool = False,
        plan_override: ReplayPlan | None = None,
        ablation_kind: str | None = None,
    ) -> ReplayGeneration: ...


_SYSTEM = """You are the source-blind replay writer in IntentIR SCRC Gate1.
Use only the supplied replay packet and selected rule cards. Never request,
infer, or reproduce source text. Return exactly one importable Triton Python
module with no Markdown fence or prose. Compiler/comparator feedback must
never be used to patch replay code. J remains the semantic authority and
concrete schedule values remain unavailable. A failed replay is handled by a
new source-aware Lift round outside this writer."""


class _RenameTargetBindings(ast.NodeTransformer):
    """Rename function-local bindings that would collide with restored ABI names."""

    def __init__(
        self, *, target_function: str, replacements: dict[str, str]
    ) -> None:
        self.target_function = target_function
        self.replacements = replacements
        self._inside_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name != self.target_function:
            return node
        previous = self._inside_target
        self._inside_target = True
        try:
            return self.generic_visit(node)
        finally:
            self._inside_target = previous

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if self._inside_target:
            node.arg = self.replacements.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self._inside_target:
            node.id = self.replacements.get(node.id, node.id)
        return node


class _RestoreOpaqueABI(ast.NodeTransformer):
    """Restore only declaration-order aliases introduced by masked replay."""

    def __init__(
        self,
        *,
        opaque_function: str,
        function_name: str,
        identifiers: dict[str, str],
    ) -> None:
        self.opaque_function = opaque_function
        self.function_name = function_name
        self.identifiers = identifiers
        self._inside_target = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name != self.opaque_function:
            return node
        node.name = self.function_name
        previous = self._inside_target
        self._inside_target = True
        try:
            return self.generic_visit(node)
        finally:
            self._inside_target = previous

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if self._inside_target:
            node.arg = self.identifiers.get(node.arg, node.arg)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self._inside_target:
            node.id = self.identifiers.get(node.id, node.id)
        return node


def _target_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"masked replay must define exactly one opaque entry {name!r}"
        )
    return matches[0]


def _is_tl_constexpr(annotation: ast.expr | None) -> bool:
    return (
        isinstance(annotation, ast.Attribute)
        and annotation.attr == "constexpr"
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "tl"
    )


def _validate_opaque_signature(
    target: ast.FunctionDef,
    artifact: Gate1Artifact,
    parameter_names: dict[str, str],
) -> None:
    actual_arguments = [
        *target.args.posonlyargs,
        *target.args.args,
        *target.args.kwonlyargs,
    ]
    expected_names = [
        parameter_names[parameter.name]
        for parameter in artifact.interface.parameters
    ]
    actual_names = [argument.arg for argument in actual_arguments]
    if (
        actual_names != expected_names
        or target.args.posonlyargs
        or target.args.kwonlyargs
        or target.args.vararg is not None
        or target.args.kwarg is not None
        or target.args.defaults
        or target.args.kw_defaults
    ):
        raise UnsafeReplayCodeError(
            "masked replay entry signature differs from the declared opaque ABI: "
            f"{actual_names!r} != {expected_names!r}"
        )
    for argument, parameter in zip(
        actual_arguments, artifact.interface.parameters, strict=True
    ):
        is_constexpr = _is_tl_constexpr(argument.annotation)
        expected_constexpr = parameter.kind is ParameterKind.CONSTEXPR
        if is_constexpr != expected_constexpr or (
            not expected_constexpr and argument.annotation is not None
        ):
            raise UnsafeReplayCodeError(
                "masked replay constexpr positions differ from the declared "
                f"opaque ABI at {argument.arg!r}"
            )


def _restore_do_not_specialize(
    target: ast.FunctionDef, identifiers: dict[str, str]
) -> None:
    """Restore only the ABI-name strings in ``@triton.jit`` metadata."""

    for decorator in target.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        function = decorator.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "jit"
            and isinstance(function.value, ast.Name)
            and function.value.id == "triton"
        ):
            continue
        for keyword in decorator.keywords:
            if keyword.arg != "do_not_specialize" or not isinstance(
                keyword.value, (ast.List, ast.Tuple)
            ):
                continue
            for element in keyword.value.elts:
                if isinstance(element, ast.Constant) and isinstance(
                    element.value, str
                ):
                    element.value = identifiers.get(element.value, element.value)


def _rename_restore_collisions(
    tree: ast.Module,
    *,
    opaque_function: str,
    opaque_parameters: set[str],
    protected_names: set[str],
) -> None:
    target = _target_function(tree, opaque_function)
    bindings = {
        node.id
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
    }
    collisions = sorted((bindings - opaque_parameters) & protected_names)
    if not collisions:
        return

    occupied = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    occupied.update(
        argument.arg
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        for argument in [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
    )
    replacements: dict[str, str] = {}
    for index, name in enumerate(collisions):
        candidate = f"scrc_local_{index}"
        suffix = 0
        while candidate in occupied:
            suffix += 1
            candidate = f"scrc_local_{index}_{suffix}"
        occupied.add(candidate)
        replacements[name] = candidate
    _RenameTargetBindings(
        target_function=opaque_function,
        replacements=replacements,
    ).visit(tree)


def restore_masked_replay_module(code: str, artifact: Gate1Artifact) -> str:
    """Restore an opaque masked-replay module to the source ABI deterministically."""

    opaque_function, parameter_names, schedule_names = opaque_replay_abi(artifact)
    validate_triton_module(code, required_function=opaque_function)
    inverse = {
        opaque: original
        for original, opaque in {**parameter_names, **schedule_names}.items()
    }
    tree = ast.parse(code, mode="exec")
    target = _target_function(tree, opaque_function)
    _validate_opaque_signature(target, artifact, parameter_names)
    _rename_restore_collisions(
        tree,
        opaque_function=opaque_function,
        opaque_parameters=set(parameter_names.values()),
        protected_names={artifact.interface.function_name, *inverse.values()},
    )
    _restore_do_not_specialize(target, inverse)
    restored = _RestoreOpaqueABI(
        opaque_function=opaque_function,
        function_name=artifact.interface.function_name,
        identifiers=inverse,
    ).visit(tree)
    ast.fix_missing_locations(restored)
    result = ast.unparse(restored).strip() + "\n"
    validate_triton_module(
        result, required_function=artifact.interface.function_name
    )
    return result


def _extract_replay_module(
    response: str,
    artifact: Gate1Artifact,
    *,
    masked: bool,
) -> str:
    function_name = artifact.interface.function_name
    if not masked:
        return extract_python_module(response, required_function=function_name)
    opaque_function, parameter_names, _ = opaque_replay_abi(artifact)
    code = extract_python_module(response, required_function=opaque_function)
    tree = ast.parse(code, mode="exec")
    _validate_opaque_signature(
        _target_function(tree, opaque_function), artifact, parameter_names
    )
    return code


class ResponsesReplayWriter:
    """OpenAI-compatible Responses transport used by the existing project."""

    def __init__(
        self,
        *,
        provider: ProviderConfig | None = None,
        model_candidates: Sequence[str] | None = None,
        effort: str = "high",
        max_retries: int = 2,
        timeout: float = 600.0,
    ) -> None:
        self.provider = provider or get_provider()
        self.model_candidates = tuple(model_candidates or (self.provider.model,))
        if not self.model_candidates:
            raise ValueError("model_candidates must not be empty")
        self.effort = effort
        self.max_retries = max_retries
        self.timeout = timeout
        self._fallback_generation = 0

    def generate(
        self,
        artifact: Gate1Artifact,
        registry: RuleRegistry,
        *,
        masked: bool = False,
        plan_override: ReplayPlan | None = None,
        ablation_kind: str | None = None,
    ) -> ReplayGeneration:
        user_input = build_rewriter_prompt(
            artifact,
            registry,
            masked=masked,
            plan_override=plan_override,
            ablation_kind=ablation_kind,
        )
        prompt_sha = hashlib.sha256(
            (_SYSTEM + "\x00" + user_input).encode("utf-8")
        ).hexdigest()
        last_error: Exception | None = None
        for model in self.model_candidates:
            provider = replace(self.provider, model=model)
            try:
                reply = call_responses(
                    _SYSTEM,
                    user_input,
                    provider=provider,
                    effort=self.effort,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
                code = _extract_replay_module(
                    reply.text, artifact, masked=masked
                )
                code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
                response_id = reply.raw.get("id")
                if response_id:
                    generation_id = f"responses:{response_id}"
                else:
                    self._fallback_generation += 1
                    generation_id = (
                        f"responses-fallback:{self._fallback_generation}:{code_sha}"
                    )
                return ReplayGeneration(
                    code=code,
                    prompt_sha256=prompt_sha,
                    requested_model=model,
                    generation_id=generation_id,
                    code_sha256=code_sha,
                    response_model=reply.raw.get("model"),
                    response_id=response_id,
                    masked=masked,
                    ablation_kind=ablation_kind,
                )
            except ProviderError as exc:
                last_error = exc
                continue
        assert last_error is not None
        raise last_error


class StaticReplayWriter:
    """Deterministic replay source for tests, cached artifacts and evaluation."""

    def __init__(
        self,
        normal: str | Sequence[str],
        *,
        masked: str | Sequence[str] | None = None,
    ) -> None:
        self._normal = list(normal) if not isinstance(normal, str) else [normal]
        masked_values: str | Sequence[str] = () if masked is None else masked
        self._masked = (
            list(masked_values)
            if not isinstance(masked_values, str)
            else [masked_values]
        )
        self._normal_index = 0
        self._masked_index = 0

    @staticmethod
    def _take(values: list[str], index: int) -> tuple[str, int]:
        if not values:
            raise RuntimeError("StaticReplayWriter has no replay source")
        return values[min(index, len(values) - 1)], index + 1

    def generate(
        self,
        artifact: Gate1Artifact,
        registry: RuleRegistry,
        *,
        masked: bool = False,
        plan_override: ReplayPlan | None = None,
        ablation_kind: str | None = None,
    ) -> ReplayGeneration:
        # Build the prompt even for static fixtures so isolation/registry errors
        # are exercised exactly as they are for a live writer.
        prompt = build_rewriter_prompt(
            artifact,
            registry,
            masked=masked,
            plan_override=plan_override,
            ablation_kind=ablation_kind,
        )
        if masked:
            invocation = self._masked_index
            code, self._masked_index = self._take(
                self._masked, self._masked_index
            )
        else:
            invocation = self._normal_index
            code, self._normal_index = self._take(
                self._normal, self._normal_index
            )
        code = _extract_replay_module(code, artifact, masked=masked)
        code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
        mode = "masked" if masked else "normal"
        return ReplayGeneration(
            code=code,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            requested_model="static",
            generation_id=f"static:{mode}:{invocation}:{code_sha}",
            code_sha256=code_sha,
            response_model="static",
            masked=masked,
            ablation_kind=ablation_kind,
        )


__all__ = [
    "ReplayGeneration",
    "ReplayWriter",
    "ResponsesReplayWriter",
    "StaticReplayWriter",
    "restore_masked_replay_module",
]
