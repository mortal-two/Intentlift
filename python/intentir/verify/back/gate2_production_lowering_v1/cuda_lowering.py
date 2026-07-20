"""Archived Gate adapter for the verification-neutral CUDA backend.

The adapter consumes only canonical IntentIR plus explicit A/B/C bindings.  It
does not accept a Gate verdict, replay status, source code, or paper receipt as
lowering authorization.  The same bundle can therefore be used by Gate2, a
lower-only workflow, or a future compatibility CLI without changing codegen.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from intentir.backend.cuda.pipeline import (
    LoweredProgram,
    execute_cuda_artifact,
    lower_to_cuda,
)
from intentir.backend.cuda.selection import LoweringTarget, select_variant
from intentir.backend.cuda.schedules import (
    ELEMENTWISE_FAST_PATH_ID,
    ROW_FAST_PATH_ID,
    ScheduleDecision,
    resolve_elementwise_schedule,
    resolve_row_schedule,
)
from intentir.ir.layer_c import LayerCResolutionError, resolve_layer_c_scalar
from intentir.frontend import emit_module
from intentir.ir.model import IntentFunction, Tensor
from intentir.verify.gate1.model import (
    AliasKind,
    Gate1Artifact,
    ParameterMode,
    SemanticArchive,
)

from .lower2c import boundary_broadcast_axes
from .model import ExecutionResult


CUDA_LOWERING_PROTOCOL = "canonical_intent_abc_cuda_cubin_v1"


def _copy_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        result[key] = item
    return result


def _resolved_shape(tensor: Tensor, dims: Mapping[str, int]) -> tuple[int, ...]:
    shape: list[int] = []
    for extent in tensor.shape:
        if isinstance(extent, bool):
            raise ValueError(f"tensor {tensor.name!r} has a boolean extent")
        if isinstance(extent, int):
            value = extent
        elif isinstance(extent, str) and extent in dims:
            value = dims[extent]
        else:
            raise ValueError(
                f"tensor {tensor.name!r} has unresolved extent {extent!r}"
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"tensor {tensor.name!r} extent {extent!r} must resolve to "
                "a non-negative integer"
            )
        shape.append(value)
    return tuple(shape)


def _validate_output_mappings(archive: SemanticArchive | None) -> None:
    if archive is None or not archive.boundary.output_mappings:
        return
    declared = {tensor.name for tensor in archive.intent.outputs}
    mapped: dict[str, str] = {}
    for mapping in archive.boundary.output_mappings:
        if mapping.output in mapped:
            raise ValueError(
                f"duplicate Boundary output mapping for {mapping.output!r}"
            )
        mapped[mapping.output] = mapping.value
    missing = sorted(declared - set(mapped))
    unknown = sorted(set(mapped) - declared)
    if missing or unknown:
        raise ValueError(
            "Boundary output mappings must cover the Intent outputs exactly; "
            f"missing={missing}, unknown={unknown}"
        )
    noncanonical = sorted(name for name, value in mapped.items() if name != value)
    if noncanonical:
        raise NotImplementedError(
            "CUDA Gate adapter requires output mappings to use the canonical "
            "Intent result names: " + ", ".join(noncanonical)
        )


def _boundary_output_aliases(
    archive: SemanticArchive | None,
) -> dict[str, str]:
    if archive is None:
        return {}
    inputs = {tensor.name for tensor in archive.intent.inputs}
    outputs = {tensor.name for tensor in archive.intent.outputs}
    aliases: dict[str, str] = {}
    for relation in archive.boundary.aliases:
        if relation.kind is not AliasKind.MUST_ALIAS:
            continue
        left_input, right_input = relation.left in inputs, relation.right in inputs
        left_output, right_output = relation.left in outputs, relation.right in outputs
        if left_output and right_input:
            output_name, input_name = relation.left, relation.right
        elif right_output and left_input:
            output_name, input_name = relation.right, relation.left
        elif (left_input and right_input) or not (left_output or right_output):
            # Input/input and internal storage identities are checked at runtime
            # or represented inside the canonical op graph; they do not bind a
            # function result buffer.
            continue
        else:
            raise NotImplementedError(
                "CUDA Gate adapter cannot materialize an output/output "
                f"must-alias relation: {relation.left!r}, {relation.right!r}"
            )
        previous = aliases.get(output_name)
        if previous not in (None, input_name):
            raise ValueError(
                f"conflicting output alias for {output_name!r}: "
                f"{previous!r} versus {input_name!r}"
            )
        aliases[output_name] = input_name
    return aliases


@dataclass(frozen=True)
class GateIntentBundle:
    """Target-neutral hand-off from a Gate artifact to a real backend."""

    intent: IntentFunction
    dims: Mapping[str, int]
    input_bindings: Mapping[str, str]
    output_bindings: Mapping[str, str]
    semantic_archive: SemanticArchive | None = None
    output_aliases: Mapping[str, str] = field(default_factory=dict)
    declared_schedule: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intent = self.intent
        if not isinstance(intent, IntentFunction):
            intent = IntentFunction.from_json(intent)
            object.__setattr__(self, "intent", intent)

        dims = _copy_mapping(self.dims, name="dims")
        for name, value in dims.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"dimension {name!r} must be a non-negative integer"
                )
        object.__setattr__(self, "dims", dims)

        inputs = _copy_mapping(self.input_bindings, name="input_bindings")
        outputs = _copy_mapping(self.output_bindings, name="output_bindings")
        input_names = {tensor.name for tensor in intent.inputs}
        output_names = {tensor.name for tensor in intent.outputs}
        missing_inputs = sorted(input_names - set(inputs))
        unknown_inputs = sorted(set(inputs) - input_names)
        missing_outputs = sorted(output_names - set(outputs))
        unknown_outputs = sorted(set(outputs) - output_names)
        if missing_inputs or unknown_inputs:
            raise ValueError(
                "Gate CUDA input bindings mismatch; "
                f"missing={missing_inputs}, unknown={unknown_inputs}"
            )
        if missing_outputs or unknown_outputs:
            raise ValueError(
                "Gate CUDA output bindings mismatch; "
                f"missing={missing_outputs}, unknown={unknown_outputs}"
            )
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("Gate CUDA output binding keys must be unique")
        if any(not isinstance(value, str) or not value for value in inputs.values()):
            raise ValueError("input binding values must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in outputs.values()):
            raise ValueError("output binding values must be non-empty strings")
        object.__setattr__(self, "input_bindings", inputs)
        object.__setattr__(self, "output_bindings", outputs)

        archive = self.semantic_archive
        if archive is not None:
            if not isinstance(archive, SemanticArchive):
                archive = SemanticArchive.from_json(archive)
                object.__setattr__(self, "semantic_archive", archive)
            if archive.intent.to_json() != intent.to_json():
                raise ValueError(
                    "semantic_archive.intent differs from the lowered IntentIR"
                )
        _validate_output_mappings(archive)
        # Validate Layer-B broadcast projection now, before compilation.
        boundary_broadcast_axes(archive)

        aliases = _copy_mapping(self.output_aliases, name="output_aliases")
        for output_name, input_name in _boundary_output_aliases(archive).items():
            previous = aliases.get(output_name)
            if previous not in (None, input_name):
                raise ValueError(
                    f"explicit and Boundary aliases conflict for {output_name!r}"
                )
            aliases[output_name] = input_name
        unknown_alias_outputs = sorted(set(aliases) - output_names)
        unknown_alias_inputs = sorted(set(aliases.values()) - input_names)
        if unknown_alias_outputs or unknown_alias_inputs:
            raise ValueError(
                "Gate CUDA output aliases reference unknown tensors; "
                f"outputs={unknown_alias_outputs}, inputs={unknown_alias_inputs}"
            )
        input_by_name = {tensor.name: tensor for tensor in intent.inputs}
        output_by_name = {tensor.name: tensor for tensor in intent.outputs}
        for output_name, input_name in aliases.items():
            output_tensor = output_by_name[output_name]
            input_tensor = input_by_name[input_name]
            if output_tensor.dtype != input_tensor.dtype or _resolved_shape(
                output_tensor, dims
            ) != _resolved_shape(input_tensor, dims):
                raise ValueError(
                    f"output alias {output_name!r}->{input_name!r} requires "
                    "identical dtype and concrete shape"
                )
        if archive is not None:
            for relation in archive.boundary.aliases:
                if relation.kind is not AliasKind.NO_ALIAS:
                    continue
                if aliases.get(relation.left) == relation.right or aliases.get(
                    relation.right
                ) == relation.left:
                    raise ValueError(
                        "output alias contradicts Boundary no_alias relation "
                        f"{relation.left!r}/{relation.right!r}"
                    )
        object.__setattr__(self, "output_aliases", aliases)
        object.__setattr__(
            self,
            "declared_schedule",
            _copy_mapping(self.declared_schedule, name="declared_schedule"),
        )

    @classmethod
    def from_artifact(
        cls,
        artifact: Gate1Artifact,
        *,
        dims: Mapping[str, int],
        input_bindings: Mapping[str, str],
        output_bindings: Mapping[str, str],
        output_aliases: Mapping[str, str] | None = None,
    ) -> "GateIntentBundle":
        """Consume validated J/B/C without source, replay, or verdict fields."""

        if not isinstance(artifact, Gate1Artifact):
            artifact = Gate1Artifact.from_json(artifact)
        return cls(
            intent=artifact.j.intent,
            dims=dims,
            input_bindings=input_bindings,
            output_bindings=output_bindings,
            semantic_archive=artifact.j,
            output_aliases=dict(output_aliases or {}),
            declared_schedule=dict(artifact.schedule.values),
        )

    @classmethod
    def from_artifact_interface(
        cls,
        artifact: Gate1Artifact,
        *,
        dims: Mapping[str, int],
        output_aliases: Mapping[str, str] | None = None,
    ) -> "GateIntentBundle":
        """Derive backend bindings from a real Lift ``KernelInterface``.

        Live Lift artifacts already bind source parameters to canonical J
        tensors through ``tensor_ref`` and classify them with ``mode``.  Make
        that persisted interface the single binding source instead of asking a
        Gate2 caller to manually restate tensor-to-argument mappings.  Missing
        or ambiguous references fail closed before MLIR emission.
        """

        if not isinstance(artifact, Gate1Artifact):
            artifact = Gate1Artifact.from_json(artifact)

        def binding_for(
            tensor_name: str,
            *,
            modes: frozenset[ParameterMode],
            role: str,
        ) -> str:
            matches = [
                parameter.name
                for parameter in artifact.interface.parameters
                if parameter.tensor_ref == tensor_name
                and parameter.mode in modes
            ]
            unique = tuple(dict.fromkeys(matches))
            if not unique:
                raise ValueError(
                    f"live Gate artifact has no {role} interface binding for "
                    f"Intent tensor {tensor_name!r}"
                )
            if len(unique) != 1:
                raise ValueError(
                    f"live Gate artifact has ambiguous {role} interface "
                    f"bindings for Intent tensor {tensor_name!r}: "
                    f"{list(unique)!r}"
                )
            return unique[0]

        input_modes = frozenset(
            {ParameterMode.INPUT, ParameterMode.INOUT, ParameterMode.VALUE}
        )
        output_modes = frozenset(
            {ParameterMode.OUTPUT, ParameterMode.INOUT}
        )
        return cls.from_artifact(
            artifact,
            dims=dims,
            input_bindings={
                tensor.name: binding_for(
                    tensor.name, modes=input_modes, role="input"
                )
                for tensor in artifact.j.intent.inputs
            },
            output_bindings={
                tensor.name: binding_for(
                    tensor.name, modes=output_modes, role="output"
                )
                for tensor in artifact.j.intent.outputs
            },
            output_aliases=output_aliases,
        )

    def semantic_contract_metadata(self) -> dict[str, Any]:
        if self.semantic_archive is None:
            return {"semantic_contract": "canonical_intent_only"}
        payload = self.semantic_archive.to_json()
        digest = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return {
            "semantic_contract": "gate_semantic_archive_abc_v1",
            "semantic_contract_sha256": digest,
            "semantic_scope_count": len(self.semantic_archive.scopes),
            "semantic_storage_count": len(self.semantic_archive.storage),
        }


@dataclass(frozen=True)
class CudaLoweringOptions:
    target: LoweringTarget | None = None
    pass_overrides: Mapping[str, str] = field(default_factory=dict)
    only_fast_path: str | None = None
    schedule_decision: ScheduleDecision | None = None
    allow_specialized: bool = True
    fold: int = 0

    def __post_init__(self) -> None:
        if self.target is not None and not isinstance(self.target, LoweringTarget):
            raise TypeError("target must be a CUDA LoweringTarget")
        overrides = _copy_mapping(self.pass_overrides, name="pass_overrides")
        if any(not isinstance(value, str) or not value for value in overrides.values()):
            raise ValueError("pass override values must be non-empty strings")
        object.__setattr__(self, "pass_overrides", overrides)
        if self.only_fast_path is not None and (
            not isinstance(self.only_fast_path, str) or not self.only_fast_path
        ):
            raise ValueError("only_fast_path must be a non-empty string")
        if self.schedule_decision is not None and not isinstance(
            self.schedule_decision, ScheduleDecision
        ):
            raise TypeError("schedule_decision must be a ScheduleDecision")
        if type(self.allow_specialized) is not bool:
            raise TypeError("allow_specialized must be boolean")
        if isinstance(self.fold, bool) or not isinstance(self.fold, int) or self.fold < 0:
            raise ValueError("fold must be a non-negative integer")


@dataclass(frozen=True)
class CudaLoweringSpec:
    bundle: GateIntentBundle
    options: CudaLoweringOptions = field(default_factory=CudaLoweringOptions)

    def __post_init__(self) -> None:
        if not isinstance(self.bundle, GateIntentBundle):
            raise TypeError("bundle must be a GateIntentBundle")
        if not isinstance(self.options, CudaLoweringOptions):
            raise TypeError("options must be CudaLoweringOptions")


@dataclass(frozen=True)
class CudaLoweringArtifact:
    program: LoweredProgram
    intent_mlir: str
    output_bindings: tuple[tuple[str, str], ...]
    output_aliases: tuple[tuple[str, str], ...]
    broadcast_consumer_count: int
    schedule_decision: ScheduleDecision | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "protocol": CUDA_LOWERING_PROTOCOL,
            "program": self.program.to_record(),
            "intent_mlir_sha256": hashlib.sha256(
                self.intent_mlir.encode("utf-8")
            ).hexdigest(),
            "output_bindings": dict(self.output_bindings),
            "output_aliases": dict(self.output_aliases),
            "broadcast_consumer_count": self.broadcast_consumer_count,
            "schedule_decision": (
                self.schedule_decision.to_dict()
                if self.schedule_decision is not None
                else None
            ),
        }


def _combined_layer_c(bundle: GateIntentBundle) -> dict[str, Any]:
    return {
        "intent_layer_c": dict(bundle.intent.layer_c),
        "gate_schedule": dict(bundle.declared_schedule),
    }


def _layer_c_schedule_decision(
    spec: CudaLoweringSpec, intent_mlir: str
) -> ScheduleDecision | None:
    options = spec.options
    if options.schedule_decision is not None:
        return options.schedule_decision
    fast_path_id = options.only_fast_path
    if fast_path_id is None and options.allow_specialized:
        selected = select_variant(
            intent_mlir,
            target=options.target,
            pass_overrides=options.pass_overrides,
            allow_specialized=True,
        )
        fast_path_id = selected.fast_path_id
    if fast_path_id not in {ROW_FAST_PATH_ID, ELEMENTWISE_FAST_PATH_ID}:
        return None
    if fast_path_id in options.pass_overrides:
        # An explicit pass override is already the caller's schedule decision.
        # Do not silently replace it with a Layer-C-derived candidate.
        return None
    layer_c = _combined_layer_c(spec.bundle)
    try:
        resolve_layer_c_scalar(layer_c, "num_warps")
    except LayerCResolutionError:
        return None
    if fast_path_id == ROW_FAST_PATH_ID:
        return resolve_row_schedule(intent_mlir, layer_c)
    return resolve_elementwise_schedule(intent_mlir, layer_c)


def _torch_dtype(torch: Any, dtype: str) -> Any:
    aliases = {
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f64": torch.float64,
        "i1": torch.bool,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
    }
    try:
        return aliases[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported CUDA Gate dtype {dtype!r}") from exc


def _raw_shares_storage(left: Any, right: Any) -> bool:
    if left is right:
        return True
    try:
        import torch

        if torch.is_tensor(left) and torch.is_tensor(right):
            return (
                left.device == right.device
                and int(left.untyped_storage().data_ptr())
                == int(right.untyped_storage().data_ptr())
            )
    except Exception:
        pass
    try:
        return bool(np.shares_memory(np.asarray(left), np.asarray(right)))
    except Exception:
        return False


def _input_alias_relations(
    bundle: GateIntentBundle,
) -> tuple[tuple[str, str, AliasKind], ...]:
    archive = bundle.semantic_archive
    if archive is None:
        return ()
    inputs = {tensor.name for tensor in bundle.intent.inputs}
    return tuple(
        (relation.left, relation.right, relation.kind)
        for relation in archive.boundary.aliases
        if relation.left in inputs and relation.right in inputs
    )


def _prepare_cuda_inputs(
    bundle: GateIntentBundle, values: Mapping[str, Any]
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    raw = {
        tensor.name: values[bundle.input_bindings[tensor.name]]
        for tensor in bundle.intent.inputs
    }
    parent = {tensor.name: tensor.name for tensor in bundle.intent.inputs}

    def find(name: str) -> str:
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, kind in _input_alias_relations(bundle):
        shares = _raw_shares_storage(raw[left], raw[right])
        if kind is AliasKind.MUST_ALIAS:
            if not shares:
                raise ValueError(
                    f"inputs {left!r} and {right!r} violate must_alias"
                )
            union(left, right)
        elif kind is AliasKind.NO_ALIAS and shares:
            raise ValueError(f"inputs {left!r} and {right!r} violate no_alias")

    tensor_by_name = {tensor.name: tensor for tensor in bundle.intent.inputs}
    prepared_by_group: dict[str, Any] = {}
    prepared: dict[str, Any] = {}
    for tensor in bundle.intent.inputs:
        group = find(tensor.name)
        shape = _resolved_shape(tensor, bundle.dims)
        dtype = _torch_dtype(torch, tensor.dtype)
        if group in prepared_by_group:
            candidate = prepared_by_group[group]
            if tuple(candidate.shape) != shape or candidate.dtype != dtype:
                raise ValueError(
                    "must-alias CUDA inputs require identical dtype and shape"
                )
            prepared[tensor.name] = candidate
            continue
        value = raw[tensor.name]
        if torch.is_tensor(value):
            candidate = value.detach().to(device="cuda", dtype=dtype).clone()
        else:
            candidate = torch.as_tensor(value, dtype=dtype, device="cuda").clone()
        if tuple(candidate.shape) != shape:
            raise ValueError(
                f"input {tensor.name!r} shape {tuple(candidate.shape)} does not "
                f"match declared {shape}"
            )
        prepared_by_group[group] = candidate
        prepared[tensor.name] = candidate

    # Binding the same external value key is an explicit request to share the
    # same logical argument object unless Layer B forbids aliasing.
    by_key: dict[str, str] = {}
    no_alias_pairs = {
        frozenset((left, right))
        for left, right, kind in _input_alias_relations(bundle)
        if kind is AliasKind.NO_ALIAS
    }
    for tensor in bundle.intent.inputs:
        key = bundle.input_bindings[tensor.name]
        previous = by_key.get(key)
        if previous is None:
            by_key[key] = tensor.name
            continue
        if frozenset((previous, tensor.name)) in no_alias_pairs:
            raise ValueError(
                f"input binding {key!r} contradicts a no_alias contract"
            )
        left = tensor_by_name[previous]
        if left.dtype != tensor.dtype or _resolved_shape(
            left, bundle.dims
        ) != _resolved_shape(tensor, bundle.dims):
            raise ValueError(
                f"shared input binding {key!r} requires identical tensor types"
            )
        prepared[tensor.name] = prepared[previous]
    return prepared


def _tensor_pointer(value: Any) -> int | None:
    try:
        return int(value.data_ptr())
    except (AttributeError, TypeError, RuntimeError):
        return None


def _validate_runtime_aliases(
    bundle: GateIntentBundle,
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
) -> None:
    output_index = {
        tensor.name: f"out{index}"
        for index, tensor in enumerate(bundle.intent.outputs)
    }
    for output_name, input_name in bundle.output_aliases.items():
        if _tensor_pointer(outputs[output_index[output_name]]) != _tensor_pointer(
            inputs[input_name]
        ):
            raise RuntimeError(
                f"CUDA lowering failed to preserve alias {output_name!r}->{input_name!r}"
            )
    archive = bundle.semantic_archive
    if archive is None:
        return
    input_names = set(inputs)
    output_names = set(output_index)

    def pointer(name: str) -> int | None:
        if name in input_names:
            return _tensor_pointer(inputs[name])
        if name in output_names:
            return _tensor_pointer(outputs[output_index[name]])
        return None

    for relation in archive.boundary.aliases:
        left, right = pointer(relation.left), pointer(relation.right)
        if left is None or right is None:
            continue
        if relation.kind is AliasKind.MUST_ALIAS and left != right:
            raise RuntimeError(
                f"CUDA result violates must_alias for {relation.left!r}/{relation.right!r}"
            )
        if relation.kind is AliasKind.NO_ALIAS and left == right:
            raise RuntimeError(
                f"CUDA result violates no_alias for {relation.left!r}/{relation.right!r}"
            )


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            copied = value.detach().cpu()
            if copied.dtype == torch.bfloat16:
                copied = copied.float()
            return copied.numpy().copy()
    except Exception:
        pass
    return np.asarray(value).copy()


class CudaLoweringRunner:
    """Lower and optionally execute a canonical Gate bundle on real CUDA."""

    def lower(self, spec: CudaLoweringSpec) -> CudaLoweringArtifact:
        bundle = spec.bundle
        broadcasts = boundary_broadcast_axes(bundle.semantic_archive)
        intent_mlir = emit_module(
            bundle.intent,
            dict(bundle.dims),
            fold=spec.options.fold,
            broadcast_axes=broadcasts,
        )
        schedule_decision = _layer_c_schedule_decision(spec, intent_mlir)
        pass_overrides = spec.options.pass_overrides
        lowering_schedule_decision = schedule_decision
        if schedule_decision is not None and spec.options.only_fast_path is None:
            # Automatic selection first discovered the structural family.  Keep
            # the second, tuned probe in the full selector order so the final
            # evidence still records rejected predecessors and can continue to
            # another compatible fast path if the Layer-C candidate stops
            # consuming the complete graph.
            pass_overrides = {
                **dict(spec.options.pass_overrides),
                schedule_decision.fast_path_id: schedule_decision.pass_argument,
            }
            lowering_schedule_decision = None
        program = lower_to_cuda(
            intent_mlir,
            target=spec.options.target,
            pass_overrides=pass_overrides,
            only_fast_path=spec.options.only_fast_path,
            schedule_decision=lowering_schedule_decision,
            allow_specialized=spec.options.allow_specialized,
        )
        output_bindings = tuple(
            (f"out{index}", bundle.output_bindings[tensor.name])
            for index, tensor in enumerate(bundle.intent.outputs)
        )
        output_aliases = tuple(
            (f"out{index}", bundle.output_aliases[tensor.name])
            for index, tensor in enumerate(bundle.intent.outputs)
            if tensor.name in bundle.output_aliases
        )
        return CudaLoweringArtifact(
            program=program,
            intent_mlir=intent_mlir,
            output_bindings=output_bindings,
            output_aliases=output_aliases,
            broadcast_consumer_count=len(broadcasts),
            schedule_decision=schedule_decision,
        )

    def execute(
        self,
        spec: CudaLoweringSpec,
        artifact: CudaLoweringArtifact,
        values: Mapping[str, Any],
    ) -> ExecutionResult:
        try:
            inputs = _prepare_cuda_inputs(spec.bundle, values)
            runtime_outputs = execute_cuda_artifact(
                artifact.program,
                artifact.intent_mlir,
                inputs,
                output_aliases=dict(artifact.output_aliases),
            )
            _validate_runtime_aliases(spec.bundle, inputs, runtime_outputs)
            outputs = {
                external_key: _to_numpy(runtime_outputs[output_key])
                for output_key, external_key in artifact.output_bindings
            }
            return ExecutionResult(
                "lowering",
                True,
                outputs,
                metadata={
                    "path": (
                        "IntentIR->intent MLIR->GPU/NVVM->PTX->cubin->CUDA driver"
                    ),
                    "protocol": CUDA_LOWERING_PROTOCOL,
                    "function": spec.bundle.intent.name,
                    "lowering": artifact.to_record(),
                    "declared_schedule": dict(spec.bundle.declared_schedule),
                    **spec.bundle.semantic_contract_metadata(),
                },
            )
        except (NotImplementedError, ValueError) as exc:
            return ExecutionResult(
                "lowering",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
            )
        except Exception as exc:
            return ExecutionResult(
                "lowering", False, error=f"{type(exc).__name__}: {exc}"
            )

    def run(
        self, spec: CudaLoweringSpec, values: Mapping[str, Any]
    ) -> ExecutionResult:
        try:
            artifact = self.lower(spec)
        except (NotImplementedError, ValueError) as exc:
            return ExecutionResult(
                "lowering",
                False,
                error=f"{type(exc).__name__}: {exc}",
                unsupported=True,
            )
        except Exception as exc:
            return ExecutionResult(
                "lowering", False, error=f"{type(exc).__name__}: {exc}"
            )
        return self.execute(spec, artifact, values)


__all__ = [
    "CUDA_LOWERING_PROTOCOL",
    "CudaLoweringArtifact",
    "CudaLoweringOptions",
    "CudaLoweringRunner",
    "CudaLoweringSpec",
    "GateIntentBundle",
]
