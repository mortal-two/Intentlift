"""Strict manifest and preflight for the formal 28-kernel LLM batch.

This module is deliberately a thin evidence boundary.  It validates frozen
source coordinates, invocation specializations, budgets, and protocols; it
does not infer semantics, compose J, select lowering rules, or authorize a Gate
verdict.  The online runner may only start after this preflight succeeds.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lift_wire import (
    LIFT_CLIENT_FILES,
    LIFT_CLIENT_FILES_V2,
    LIFT_CLIENT_FILES_V3,
    LIFT_PROTOCOL,
    LIFT_PROTOCOL_V2,
    LIFT_PROTOCOL_V3,
    LIFT_WIRE_VERSION,
    LIFT_WIRE_VERSION_V2,
    LIFT_WIRE_VERSION_V3,
)


SCHEMA_V1 = "intentir.real-llm-batch-28.v1"
SCHEMA_V2 = "intentir.real-llm-batch-28.v2"
SCHEMA_V3 = "intentir.real-llm-batch-28.v3"
SCHEMA_V4 = "intentir.real-llm-batch-28.v4"
# Historical public name retained for archived Task04 callers.
SCHEMA = SCHEMA_V1
V2_BATCH_ID = "tianchenir-formal-real-kernels-protocol-v2-2026-07-22"
V3_BATCH_ID = "tianchenir-formal-real-kernels-protocol-v3-2026-07-22"
V4_BATCH_ID = "tianchenir-formal-real-kernels-protocol-v4-2026-07-22"
V2_MANIFEST_DIGEST = (
    "486f18ce748f63afc4b7684fe2a7e5d1107da04c72a6282b7683851eb086f31e"
)
V2_MANIFEST_FILE_SHA256 = (
    "27b9a81e8230ce75105848625b85f44e9002264c91d166e57983c617e09e83b3"
)
V3_MANIFEST_DIGEST = (
    "5f0b1ee853e2821a299753165145023562df548810468c0110bdbc6895823222"
)
V3_MANIFEST_FILE_SHA256 = (
    "2528ae016445afd63f9c0083213ab2594b97756bcd7b1950edb28e1273212205"
)
CASE_COUNT = 28
STRATA = (
    "expression_fusion_reduction",
    "copy_layout_gather_indexing",
    "contraction_attention_recurrence",
    "update_atomic_dynamic_state",
)
CASES_PER_STRATUM = 7
GATE1_PROTOCOL = "llm_direct_pi_gate1_level2_v1"
GATE2_PROTOCOL = "source_replay_j_relation_v4"
PERFORMANCE_PROTOCOL = "cuda_event_interleaved_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CASE_ID = re.compile(r"[a-z][a-z0-9_]*")

_TOP_KEYS_V1 = frozenset(
    {
        "schema",
        "batch_id",
        "frozen_date",
        "source_root",
        "provider",
        "call_budget",
        "repositories",
        "protocols",
        "required_anchors",
        "cases",
    }
)
_TOP_KEYS_V2 = _TOP_KEYS_V1 | {"lift_wire"}
_TOP_KEYS_V3 = _TOP_KEYS_V2
_TOP_KEYS_V4 = _TOP_KEYS_V3
_CASE_KEYS = frozenset(
    {
        "case_id",
        "stratum",
        "mechanisms",
        "source",
        "specialization",
        "logical_contract",
        "required_capabilities",
        "benchmark",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "repository",
        "path",
        "symbol",
        "frontend",
        "file_sha256",
        "symbol_sha256",
    }
)
_SPECIALIZATION_KEYS = frozenset(
    {"arguments", "grid", "schedule", "compiler_options"}
)
_SPECIALIZATION_KEYS_V4 = frozenset(
    {"parameters", "grid", "compiler_options"}
)
_ARGUMENT_KEYS = frozenset(
    {"name", "kind", "role", "dtype", "shape", "strides", "value"}
)
_RUNTIME_PARAMETER_KEYS_V4 = _ARGUMENT_KEYS | {"binding"}
_CONSTEXPR_PARAMETER_KEYS_V4 = frozenset(
    {"name", "binding", "class", "value"}
)
_CONTRACT_KEYS = frozenset(
    {
        "inputs",
        "outputs",
        "inout",
        "shape_envelope",
        "dtype_envelope",
        "result_relation",
    }
)
_BENCHMARK_KEYS = frozenset(
    {"baseline", "warmup", "repetitions", "timer", "threshold"}
)
_BANNED_MANIFEST_KEYS = frozenset(
    {
        "j",
        "pi",
        "replay",
        "intent_mlir",
        "rule_card",
        "rule_registry",
        "composer",
        "intent_program",
        "logical_abi",
    }
)


class ManifestError(ValueError):
    """A frozen-batch fact is absent, contradictory, or no longer reproducible."""


@dataclass(frozen=True)
class BatchManifest:
    """Validated batch plus its canonical digest and resolved source root."""

    data: Mapping[str, Any]
    source_root: Path
    digest: str

    @property
    def cases(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.data["cases"])

    def case(self, case_id: str) -> Mapping[str, Any]:
        matches = [case for case in self.cases if case["case_id"] == case_id]
        if len(matches) != 1:
            raise KeyError(case_id)
        return matches[0]


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{path} must be an object")
    return value


def _array(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ManifestError(f"{path} must be an array")
    return value


def _closed(value: Any, allowed: frozenset[str], path: str) -> dict[str, Any]:
    obj = _object(value, path)
    missing = sorted(allowed - set(obj))
    unknown = sorted(set(obj) - allowed)
    if missing:
        raise ManifestError(f"{path} is missing keys: {missing}")
    if unknown:
        raise ManifestError(f"{path} has unknown keys: {unknown}")
    return obj


def _name(value: Any, path: str, pattern: re.Pattern[str] = _IDENTIFIER) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ManifestError(f"{path} has an invalid name")
    return value


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ManifestError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, path: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{path} must be a non-empty relative path")
    result = Path(value)
    if result.is_absolute() or ".." in result.parts:
        raise ManifestError(f"{path} must stay inside its repository root")
    return result


def _walk_keys(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield path, key
            yield from _walk_keys(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_keys(item, f"{path}[{index}]")


def symbol_source_bytes(source: str, symbol: str) -> bytes:
    """Return the exact decorated top-level function span for one symbol."""

    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    ]
    if len(matches) != 1:
        raise ManifestError(
            f"source symbol {symbol!r} must resolve to one top-level function; "
            f"found {len(matches)}"
        )
    node = matches[0]
    start = min([node.lineno, *(item.lineno for item in node.decorator_list)]) - 1
    lines = source.splitlines(keepends=True)
    return "".join(lines[start : node.end_lineno]).encode()


def _function_contract(source: str, symbol: str) -> tuple[list[str], set[str], str]:
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == symbol
    ]
    if len(matches) != 1:
        raise ManifestError(f"cannot resolve function contract for {symbol!r}")
    node = matches[0]
    if node.args.vararg is not None or node.args.kwarg is not None:
        raise ManifestError("formal batch kernels may not expose variadic ABIs")
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    names = [item.arg for item in arguments]
    constexprs = {
        item.arg
        for item in arguments
        if item.annotation is not None
        and ast.unparse(item.annotation).endswith("constexpr")
    }
    decorators = "\n".join(ast.unparse(item) for item in node.decorator_list)
    return names, constexprs, decorators


def _verify_source_pin(
    case: Mapping[str, Any],
    repositories: Mapping[str, Any],
    source_root: Path,
) -> Path:
    case_id = case["case_id"]
    pin = _closed(case["source"], _SOURCE_KEYS, f"cases.{case_id}.source")
    repository = _name(pin["repository"], f"cases.{case_id}.source.repository")
    if repository not in repositories:
        raise ManifestError(f"case {case_id} names unknown repository {repository!r}")
    repository_spec = _object(repositories[repository], f"repositories.{repository}")
    repository_root = _relative_path(
        repository_spec.get("root"), f"repositories.{repository}.root"
    )
    relative = _relative_path(pin["path"], f"cases.{case_id}.source.path")
    path = (source_root / repository_root / relative).resolve()
    expected_root = (source_root / repository_root).resolve()
    if expected_root not in path.parents:
        raise ManifestError(f"case {case_id} source escapes repository root")
    try:
        payload = path.read_bytes()
        source = payload.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"case {case_id} source is unreadable: {exc}") from exc
    expected_file = _digest(
        pin["file_sha256"], f"cases.{case_id}.source.file_sha256"
    )
    if hashlib.sha256(payload).hexdigest() != expected_file:
        raise ManifestError(f"case {case_id} file digest changed")
    symbol = _name(pin["symbol"], f"cases.{case_id}.source.symbol")
    symbol_payload = symbol_source_bytes(source, symbol)
    expected_symbol = _digest(
        pin["symbol_sha256"], f"cases.{case_id}.source.symbol_sha256"
    )
    if hashlib.sha256(symbol_payload).hexdigest() != expected_symbol:
        raise ManifestError(f"case {case_id} symbol digest changed")
    frontend = pin["frontend"]
    if frontend not in {"triton", "cute"}:
        raise ManifestError(f"case {case_id} has unsupported frontend {frontend!r}")
    names, constexprs, decorators = _function_contract(source, symbol)
    if frontend == "triton" and "triton.jit" not in decorators:
        raise ManifestError(f"case {case_id} is not a decorated Triton kernel")
    if frontend == "cute" and "cute.kernel" not in decorators:
        raise ManifestError(f"case {case_id} is not a decorated CuTe kernel")
    _validate_specialization(case, names, constexprs)
    return path


def _validate_argument(value: Any, path: str) -> dict[str, Any]:
    item = _closed(value, _ARGUMENT_KEYS, path)
    _name(item["name"], f"{path}.name")
    if item["kind"] not in {"tensor", "scalar", "none", "opaque"}:
        raise ManifestError(f"{path}.kind is unsupported")
    if item["role"] not in {"input", "output", "inout", "value"}:
        raise ManifestError(f"{path}.role is unsupported")
    if item["kind"] == "tensor":
        if item["role"] == "value" or not isinstance(item["dtype"], str):
            raise ManifestError(f"{path} tensor has an invalid role/dtype")
        shape = _array(item["shape"], f"{path}.shape")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape):
            raise ManifestError(f"{path}.shape must contain non-negative integers")
        strides = item["strides"]
        if strides is not None:
            strides = _array(strides, f"{path}.strides")
            if len(strides) != len(shape) or any(
                isinstance(stride, bool) or not isinstance(stride, int) or stride < 0
                for stride in strides
            ):
                raise ManifestError(f"{path}.strides must match tensor rank")
        if item["value"] is not None:
            raise ManifestError(f"{path} tensor may not carry a scalar value")
    elif item["kind"] == "scalar":
        if item["role"] != "value" or item["shape"] is not None or item["strides"] is not None:
            raise ManifestError(f"{path} scalar must be a value without shape")
        if not isinstance(item["value"], (bool, int, float)):
            raise ManifestError(f"{path} scalar requires a JSON scalar value")
    elif item["kind"] == "none":
        if item["role"] != "value" or any(
            item[key] is not None for key in ("shape", "strides", "value")
        ):
            raise ManifestError(f"{path} none argument must be an empty value")
    else:
        if (
            item["role"] != "value"
            or not isinstance(item["dtype"], str)
            or item["shape"] is not None
            or item["strides"] is not None
            or not isinstance(item["value"], str)
            or not item["value"]
        ):
            raise ManifestError(f"{path} opaque frontend argument is incomplete")
    return item


def _validate_v4_parameter(value: Any, path: str) -> dict[str, Any]:
    raw = _object(value, path)
    binding = raw.get("binding")
    if binding == "runtime":
        item = _closed(raw, _RUNTIME_PARAMETER_KEYS_V4, path)
        projected = {key: item[key] for key in _ARGUMENT_KEYS}
        _validate_argument(projected, path)
        return item
    if binding == "constexpr":
        item = _closed(raw, _CONSTEXPR_PARAMETER_KEYS_V4, path)
        _name(item["name"], f"{path}.name")
        if item["class"] not in {"binding", "schedule"}:
            raise ManifestError(
                f"{path}.class must be binding or schedule"
            )
        if not isinstance(item["value"], (bool, int, float, str)):
            raise ManifestError(f"{path}.value must be a JSON scalar")
        return item
    raise ManifestError(f"{path}.binding must be runtime or constexpr")


def _runtime_parameters(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    specialization = case["specialization"]
    if "parameters" in specialization:
        return [
            item
            for item in specialization["parameters"]
            if item["binding"] == "runtime"
        ]
    return list(specialization["arguments"])


def _constexpr_parameters(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    specialization = case["specialization"]
    if "parameters" in specialization:
        return [
            item
            for item in specialization["parameters"]
            if item["binding"] == "constexpr"
        ]
    return [
        {"name": name, "value": value, "class": "schedule"}
        for name, value in specialization["schedule"].items()
    ]


def _callable_parameter_order(case: Mapping[str, Any]) -> list[str]:
    specialization = case["specialization"]
    if "parameters" in specialization:
        return [item["name"] for item in specialization["parameters"]]
    return [
        *(item["name"] for item in specialization["arguments"]),
        *specialization["schedule"],
    ]


def _validate_specialization(
    case: Mapping[str, Any],
    function_arguments: Sequence[str],
    constexprs: set[str],
) -> None:
    case_id = case["case_id"]
    path = f"cases.{case_id}.specialization"
    raw_spec = _object(case["specialization"], path)
    if "parameters" in raw_spec:
        spec = _closed(raw_spec, _SPECIALIZATION_KEYS_V4, path)
        parameters = [
            _validate_v4_parameter(item, f"{path}.parameters[{index}]")
            for index, item in enumerate(
                _array(spec["parameters"], f"{path}.parameters")
            )
        ]
        names = [item["name"] for item in parameters]
        if len(names) != len(set(names)):
            raise ManifestError(f"case {case_id} has duplicate parameters")
        if names != list(function_arguments):
            raise ManifestError(
                f"case {case_id} parameter order differs from source ABI"
            )
        actual_constexprs = {
            item["name"]
            for item in parameters
            if item["binding"] == "constexpr"
        }
        if actual_constexprs != constexprs:
            raise ManifestError(
                f"case {case_id} constexpr classification differs from source ABI"
            )
        if any(
            item["name"] in constexprs
            for item in parameters
            if item["binding"] == "runtime"
        ):
            raise ManifestError(
                f"case {case_id} labels a constexpr parameter as runtime"
            )
        grid = _array(spec["grid"], f"{path}.grid")
        if not 1 <= len(grid) <= 3 or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in grid
        ):
            raise ManifestError(
                f"case {case_id} grid must have 1-3 positive extents"
            )
        compiler = _object(
            spec["compiler_options"], f"{path}.compiler_options"
        )
        for name, value in compiler.items():
            _name(name, f"{path}.compiler_options key", _CASE_ID)
            if not isinstance(value, (bool, int, float, str)):
                raise ManifestError(
                    f"case {case_id} compiler option is not scalar"
                )
        parameter_names = set(names)
        overlap = parameter_names & set(compiler)
        if overlap:
            raise ManifestError(
                f"case {case_id} repeats callable/compiler parameter(s): "
                f"{sorted(overlap)}"
            )
        return

    spec = _closed(raw_spec, _SPECIALIZATION_KEYS, path)
    arguments = [
        _validate_argument(item, f"{path}.arguments[{index}]")
        for index, item in enumerate(_array(spec["arguments"], f"{path}.arguments"))
    ]
    names = [item["name"] for item in arguments]
    if len(names) != len(set(names)):
        raise ManifestError(f"case {case_id} has duplicate runtime arguments")
    schedule = _object(spec["schedule"], f"{path}.schedule")
    for name, value in schedule.items():
        _name(name, f"{path}.schedule key")
        if not isinstance(value, (bool, int, float, str)):
            raise ManifestError(f"{path}.schedule.{name} is not a JSON scalar")
    expected_runtime = [name for name in function_arguments if name not in constexprs]
    expected_schedule = [name for name in function_arguments if name in constexprs]
    if names != expected_runtime or list(schedule) != expected_schedule:
        raise ManifestError(
            f"case {case_id} specialization order differs from source ABI: "
            f"runtime={names!r}, schedule={list(schedule)!r}, "
            f"source={list(function_arguments)!r}"
        )
    if set(schedule) != constexprs:
        raise ManifestError(
            f"case {case_id} schedule differs from annotated constexpr ABI"
        )
    grid = _array(spec["grid"], f"{path}.grid")
    if not 1 <= len(grid) <= 3 or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in grid
    ):
        raise ManifestError(f"case {case_id} grid must have 1-3 positive extents")
    compiler = _object(spec["compiler_options"], f"{path}.compiler_options")
    for name, value in compiler.items():
        _name(name, f"{path}.compiler_options key", _CASE_ID)
        if not isinstance(value, (bool, int, float, str)):
            raise ManifestError(f"case {case_id} compiler option is not scalar")


def _validate_contract(case: Mapping[str, Any]) -> None:
    case_id = case["case_id"]
    path = f"cases.{case_id}.logical_contract"
    contract = _closed(case["logical_contract"], _CONTRACT_KEYS, path)
    groups: dict[str, list[str]] = {}
    for key in ("inputs", "outputs", "inout"):
        values = [
            _name(item, f"{path}.{key}[{index}]")
            for index, item in enumerate(_array(contract[key], f"{path}.{key}"))
        ]
        if len(values) != len(set(values)):
            raise ManifestError(f"{path}.{key} contains duplicates")
        groups[key] = values
    all_logical = [*groups["inputs"], *groups["outputs"], *groups["inout"]]
    if len(all_logical) != len(set(all_logical)):
        raise ManifestError(f"case {case_id} logical roles overlap")
    arguments = {item["name"]: item for item in _runtime_parameters(case)}
    role_group = {"input": "inputs", "output": "outputs", "inout": "inout"}
    for name, item in arguments.items():
        if item["role"] == "value":
            continue
        expected = role_group[item["role"]]
        if name not in groups[expected]:
            raise ManifestError(f"case {case_id} logical role for {name!r} is inconsistent")
    if set(all_logical) != {
        name for name, item in arguments.items() if item["role"] != "value"
    }:
        raise ManifestError(f"case {case_id} logical contract misses tensor arguments")
    envelope = _object(contract["shape_envelope"], f"{path}.shape_envelope")
    for name, bounds in envelope.items():
        _name(name, f"{path}.shape_envelope key", _CASE_ID)
        values = _array(bounds, f"{path}.shape_envelope.{name}")
        if (
            len(values) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values)
            or values[0] > values[1]
        ):
            raise ManifestError(f"{path}.shape_envelope.{name} is not [min,max]")
    dtypes = _array(contract["dtype_envelope"], f"{path}.dtype_envelope")
    if not dtypes or not all(isinstance(item, str) and item for item in dtypes):
        raise ManifestError(f"case {case_id} dtype envelope is empty")
    relation = _object(contract["result_relation"], f"{path}.result_relation")
    kind = relation.get("kind")
    if kind not in {"exact", "tolerance", "one_of_updates"}:
        raise ManifestError(f"case {case_id} has unsupported result relation")
    if kind == "tolerance" and not all(
        isinstance(relation.get(key), (int, float)) and relation[key] >= 0
        for key in ("atol", "rtol")
    ):
        raise ManifestError(f"case {case_id} tolerance relation is incomplete")


def _validate_benchmark(case: Mapping[str, Any]) -> None:
    case_id = case["case_id"]
    path = f"cases.{case_id}.benchmark"
    benchmark = _closed(case["benchmark"], _BENCHMARK_KEYS, path)
    if benchmark["baseline"] != "frozen_upstream_symbol":
        raise ManifestError(f"case {case_id} benchmark must use the real source")
    if benchmark["timer"] != PERFORMANCE_PROTOCOL:
        raise ManifestError(f"case {case_id} benchmark timer drifted")
    for key in ("warmup", "repetitions"):
        if isinstance(benchmark[key], bool) or not isinstance(benchmark[key], int) or benchmark[key] <= 0:
            raise ManifestError(f"{path}.{key} must be positive")
    threshold = _object(benchmark["threshold"], f"{path}.threshold")
    if set(threshold) != {"metric", "maximum"}:
        raise ManifestError(f"case {case_id} threshold must freeze metric and maximum")
    if threshold["metric"] != "target_over_source_median" or not isinstance(
        threshold["maximum"], (int, float)
    ):
        raise ManifestError(f"case {case_id} threshold is malformed")
    if not 0 < float(threshold["maximum"]) <= 1.10:
        raise ManifestError(f"case {case_id} threshold is weaker than 1.10x")


def _validate_protocols(data: Mapping[str, Any]) -> None:
    protocols = _object(data["protocols"], "protocols")
    expected = {
        "gate1": GATE1_PROTOCOL,
        "gate1_observer": "normalized_compiler_topology_level2",
        "gate2": GATE2_PROTOCOL,
        "scalar": "canonical_j_scalar_cpu_mlir_v2",
        "cuda_target": "sm_120a",
        "performance": PERFORMANCE_PROTOCOL,
    }
    if data["schema"] == SCHEMA_V2:
        expected["lift"] = LIFT_PROTOCOL_V2
    elif data["schema"] == SCHEMA_V3:
        expected["lift"] = LIFT_PROTOCOL_V3
    elif data["schema"] == SCHEMA_V4:
        expected["lift"] = LIFT_PROTOCOL
    if protocols != expected:
        raise ManifestError("formal batch protocols differ from the frozen epoch")
    budget = _object(data["call_budget"], "call_budget")
    expected_budget = {
        "lift_per_case": 1,
        "normal_replay_per_case": 1,
        "masked_j_per_closed_case": 1,
        "recovery_per_case": 0,
        "maximum_total_calls": 84,
    }
    if budget != expected_budget:
        raise ManifestError("formal online call budget differs from the frozen epoch")


def _validate_lift_wire(data: Mapping[str, Any]) -> None:
    if data["schema"] == SCHEMA_V1:
        return
    wire = _closed(
        data["lift_wire"],
        frozenset({"version", "schema_sha256", "prompt_sha256", "client_files"}),
        "lift_wire",
    )
    if data["schema"] == SCHEMA_V2:
        expected_version = LIFT_WIRE_VERSION_V2
        expected_files = LIFT_CLIENT_FILES_V2
    elif data["schema"] == SCHEMA_V3:
        expected_version = LIFT_WIRE_VERSION_V3
        expected_files = LIFT_CLIENT_FILES_V3
    elif data["schema"] == SCHEMA_V4:
        expected_version = LIFT_WIRE_VERSION
        expected_files = LIFT_CLIENT_FILES
    else:  # pragma: no cover - guarded by validate_batch_manifest
        raise ManifestError("lift_wire is unsupported for this manifest epoch")
    if wire["version"] != expected_version:
        raise ManifestError("lift_wire.version differs from the manifest epoch")
    _digest(wire["schema_sha256"], "lift_wire.schema_sha256")
    _digest(wire["prompt_sha256"], "lift_wire.prompt_sha256")
    files = _array(wire["client_files"], "lift_wire.client_files")
    if len(files) != len(expected_files):
        raise ManifestError("lift_wire must bind every request-path client file")
    paths: list[str] = []
    for index, raw in enumerate(files):
        item = _closed(
            raw,
            frozenset({"path", "sha256"}),
            f"lift_wire.client_files[{index}]",
        )
        path = _relative_path(item["path"], f"lift_wire.client_files[{index}].path")
        paths.append(path.as_posix())
        _digest(item["sha256"], f"lift_wire.client_files[{index}].sha256")
    if tuple(paths) != expected_files:
        raise ManifestError("lift_wire client file set/order differs from manifest epoch")


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _verify_repository_revisions(
    repositories: Mapping[str, Any], source_root: Path
) -> None:
    for name, raw in repositories.items():
        _name(name, "repositories key")
        spec = _object(raw, f"repositories.{name}")
        if set(spec) != {"root", "revision", "version", "license"}:
            raise ManifestError(f"repositories.{name} has an open or incomplete schema")
        root = source_root / _relative_path(spec["root"], f"repositories.{name}.root")
        revision = _object(spec["revision"], f"repositories.{name}.revision")
        if set(revision) != {"kind", "value"}:
            raise ManifestError(f"repositories.{name}.revision is incomplete")
        kind, value = revision["kind"], revision["value"]
        if kind == "tree_sha256":
            _digest(value, f"repositories.{name}.revision.value")
            if _tree_digest(root) != value:
                raise ManifestError(f"repository snapshot changed: {name}")
        elif kind == "package_version":
            if not isinstance(value, str) or importlib.metadata.version(name) != value:
                raise ManifestError(f"installed package revision changed: {name}")
        elif kind == "fixture_epoch":
            if not isinstance(value, str) or not value:
                raise ManifestError(f"fixture revision is missing: {name}")
        else:
            raise ManifestError(f"repository {name} uses unsupported revision kind")
        if not isinstance(spec["version"], str) or not isinstance(spec["license"], str):
            raise ManifestError(f"repository {name} lacks version/license identity")


def validate_batch_manifest(
    value: Any,
    *,
    source_root: str | os.PathLike[str] | None = None,
    verify_sources: bool = True,
    verify_repositories: bool = False,
) -> BatchManifest:
    """Validate all frozen facts and return a digest-bound manifest."""

    raw = _object(value, "manifest")
    schema = raw.get("schema")
    if schema == SCHEMA_V1:
        data = _closed(raw, _TOP_KEYS_V1, "manifest")
    elif schema == SCHEMA_V2:
        data = _closed(raw, _TOP_KEYS_V2, "manifest")
    elif schema == SCHEMA_V3:
        data = _closed(raw, _TOP_KEYS_V3, "manifest")
    elif schema == SCHEMA_V4:
        data = _closed(raw, _TOP_KEYS_V4, "manifest")
    else:
        raise ManifestError(f"unsupported batch schema {schema!r}")
    if not isinstance(data["batch_id"], str) or not data["batch_id"]:
        raise ManifestError("batch_id must be non-empty")
    if schema == SCHEMA_V2 and data["batch_id"] != V2_BATCH_ID:
        raise ManifestError("protocol-v2 batch_id differs from the frozen epoch")
    if schema == SCHEMA_V3 and data["batch_id"] != V3_BATCH_ID:
        raise ManifestError("protocol-v3 batch_id differs from the frozen epoch")
    if schema == SCHEMA_V4 and data["batch_id"] != V4_BATCH_ID:
        raise ManifestError("protocol-v4 batch_id differs from the frozen epoch")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(data["frozen_date"])) is None:
        raise ManifestError("frozen_date must use YYYY-MM-DD")
    for path, key in _walk_keys(data):
        if key.casefold() in _BANNED_MANIFEST_KEYS:
            raise ManifestError(f"{path} contains banned semantic artifact key {key!r}")
    configured_root = Path(data["source_root"]).expanduser()
    resolved_root = Path(source_root).expanduser() if source_root is not None else configured_root
    resolved_root = resolved_root.resolve()
    if not resolved_root.is_dir():
        raise ManifestError(f"source_root does not exist: {resolved_root}")
    provider = _object(data["provider"], "provider")
    if set(provider) != {"name", "model", "wire_api", "reasoning_effort"}:
        raise ManifestError("provider identity must be closed and contain no secret")
    if provider["wire_api"] != "responses" or provider["reasoning_effort"] not in {
        "high",
        "xhigh",
    }:
        raise ManifestError("provider protocol/effort is not frozen")
    _validate_protocols(data)
    _validate_lift_wire(data)
    repositories = _object(data["repositories"], "repositories")
    if verify_repositories:
        _verify_repository_revisions(repositories, resolved_root)
    cases = _array(data["cases"], "cases")
    if len(cases) != CASE_COUNT:
        raise ManifestError(f"formal batch requires exactly {CASE_COUNT} cases")
    ids: list[str] = []
    source_coordinates: list[tuple[str, str, str]] = []
    stratum_counts = {stratum: 0 for stratum in STRATA}
    for index, raw in enumerate(cases):
        case = _closed(raw, _CASE_KEYS, f"cases[{index}]")
        case_id = _name(case["case_id"], f"cases[{index}].case_id", _CASE_ID)
        ids.append(case_id)
        stratum = case["stratum"]
        if stratum not in stratum_counts:
            raise ManifestError(f"case {case_id} has unknown stratum {stratum!r}")
        stratum_counts[stratum] += 1
        mechanisms = _array(case["mechanisms"], f"cases.{case_id}.mechanisms")
        if not mechanisms or not all(
            isinstance(item, str) and _CASE_ID.fullmatch(item) for item in mechanisms
        ):
            raise ManifestError(f"case {case_id} has invalid mechanism tags")
        capabilities = _array(
            case["required_capabilities"], f"cases.{case_id}.required_capabilities"
        )
        if not capabilities or not all(
            isinstance(item, str) and _CASE_ID.fullmatch(item) for item in capabilities
        ):
            raise ManifestError(f"case {case_id} has invalid capability declarations")
        pin = _object(case["source"], f"cases.{case_id}.source")
        source_coordinates.append((pin["repository"], pin["path"], pin["symbol"]))
        if verify_sources:
            _verify_source_pin(case, repositories, resolved_root)
        else:
            specialization = _object(
                case["specialization"], f"cases.{case_id}.specialization"
            )
            if "parameters" in specialization:
                parameter_names = [
                    str(item.get("name"))
                    for item in _array(
                        specialization.get("parameters"),
                        f"cases.{case_id}.specialization.parameters",
                    )
                    if isinstance(item, dict)
                ]
                constexpr_names = {
                    str(item.get("name"))
                    for item in specialization["parameters"]
                    if isinstance(item, dict)
                    and item.get("binding") == "constexpr"
                }
            else:
                runtime = _array(
                    specialization.get("arguments"),
                    f"cases.{case_id}.specialization.arguments",
                )
                schedule = _object(
                    specialization.get("schedule"),
                    f"cases.{case_id}.specialization.schedule",
                )
                parameter_names = [
                    str(item.get("name"))
                    for item in runtime
                    if isinstance(item, dict)
                ] + list(schedule)
                constexpr_names = set(schedule)
            _validate_specialization(
                case, parameter_names, constexpr_names
            )
        _validate_contract(case)
        _validate_benchmark(case)
    if len(ids) != len(set(ids)):
        raise ManifestError("case_id values must be unique")
    if len(source_coordinates) != len(set(source_coordinates)):
        raise ManifestError("repository/path/symbol coordinates must be unique")
    if any(count != CASES_PER_STRATUM for count in stratum_counts.values()):
        raise ManifestError(f"formal strata must contain seven cases each: {stratum_counts}")
    anchors = _array(data["required_anchors"], "required_anchors")
    if len(anchors) != len(set(anchors)) or not set(anchors) <= set(ids):
        raise ManifestError("required anchors must name unique batch case ids")
    return BatchManifest(data, resolved_root, _canonical_digest(data))


def load_batch_manifest(
    path: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] | None = None,
    verify_sources: bool = True,
    verify_repositories: bool = False,
) -> BatchManifest:
    manifest_path = Path(path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load batch manifest {manifest_path}: {exc}") from exc
    return validate_batch_manifest(
        value,
        source_root=source_root,
        verify_sources=verify_sources,
        verify_repositories=verify_repositories,
    )


def resolve_source_path(
    manifest: BatchManifest, case: Mapping[str, Any]
) -> Path:
    repository = manifest.data["repositories"][case["source"]["repository"]]
    return (
        manifest.source_root
        / repository["root"]
        / case["source"]["path"]
    ).resolve()


def scrc_compiler_view(case: Mapping[str, Any]):
    """Mechanically project one frozen invocation into the Lift harness view."""

    from intentir.extraction.lifters.scrc import (
        SCRCCompilerArgument,
        SCRCCompilerView,
    )

    specialization = case["specialization"]
    runtime_arguments = []
    for item in _runtime_parameters(case):
        runtime_arguments.append(
            SCRCCompilerArgument(
                item["name"],
                item["kind"],
                item["dtype"] or item["kind"],
                tuple(item["shape"] or ()),
                item["value"] if item["kind"] == "scalar" else None,
            )
        )
    constexprs = _constexpr_parameters(case)
    schedule = {
        item["name"]: item["value"]
        for item in constexprs
        if item["class"] == "schedule"
    }
    bound_constexprs = {
        item["name"]: item["value"]
        for item in constexprs
        if item["class"] == "binding"
    }
    compiler_options = dict(specialization["compiler_options"])
    overlap = set(schedule) & set(compiler_options)
    if overlap:
        raise ManifestError(
            f"case {case['case_id']} repeats C names across schedule/compiler: "
            f"{sorted(overlap)}"
        )
    parameter_order = _callable_parameter_order(case)
    return SCRCCompilerView(
        schedule_values={**schedule, **compiler_options},
        bound_constexpr_values=bound_constexprs,
        compiler_slots=tuple(compiler_options),
        arguments=tuple(runtime_arguments),
        grid=tuple(specialization["grid"]),
        grid_contract=tuple(str(value) for value in specialization["grid"]),
        parameter_order=tuple(parameter_order),
        constexpr_parameters=tuple(item["name"] for item in constexprs),
    )


def triton_compile_case(case: Mapping[str, Any]):
    """Project one frozen specialization into the existing Gate1 boundary."""

    from intentir.verify.gate1.compiler import CompileCase, RuntimeArgument

    arguments: dict[str, RuntimeArgument] = {}
    for item in _runtime_parameters(case):
        if item["kind"] == "tensor":
            arguments[item["name"]] = RuntimeArgument.tensor(
                tuple(item["shape"]),
                item["dtype"],
                strides=(
                    tuple(item["strides"])
                    if item["strides"] is not None
                    else None
                ),
            )
        elif item["kind"] == "scalar":
            arguments[item["name"]] = RuntimeArgument.scalar(
                item["value"], item["dtype"]
            )
        elif item["kind"] == "none":
            arguments[item["name"]] = RuntimeArgument.none()
        else:
            raise ManifestError(
                f"case {case['case_id']} has no Triton binding for "
                f"{item['kind']!r}"
            )
    constexpr_values = {
        item["name"]: item["value"] for item in _constexpr_parameters(case)
    }
    schedule = {
        **constexpr_values,
        **case["specialization"]["compiler_options"],
    }
    overlap = set(constexpr_values) & set(
        case["specialization"]["compiler_options"]
    )
    if overlap:
        raise ManifestError(
            f"case {case['case_id']} repeats C names across schedule/compiler: "
            f"{sorted(overlap)}"
        )
    return CompileCase(
        arguments,
        tuple(case["specialization"]["grid"]),
        schedule,
    )


def compiler_pythonpath(manifest: BatchManifest) -> str:
    """Frozen source import roots shared by preflight and formal Gate1."""

    python_paths = [str(Path(__file__).resolve().parents[2])]
    for repository in manifest.data["repositories"].values():
        root = (manifest.source_root / repository["root"]).resolve()
        python_paths.append(str(root))
        source_package = root / "src"
        if source_package.is_dir():
            python_paths.append(str(source_package))
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        python_paths.append(inherited)
    return os.pathsep.join(dict.fromkeys(python_paths))


def cute_compile_case(case: Mapping[str, Any]):
    """Project semantic entry values and pure C into the CuTe observer."""

    from intentir.verify.gate1.compiler import RuntimeArgument
    from intentir.verify.gate1.cute_compiler import CuteCompileCase

    specialization = case["specialization"]
    by_name = {item["name"]: item for item in _runtime_parameters(case)}
    contract = case["logical_contract"]
    entry_names = [
        *contract["inputs"],
        *contract["inout"],
        *contract["outputs"],
    ]
    arguments: list[RuntimeArgument] = []
    for name in entry_names:
        item = by_name[name]
        if item["kind"] == "tensor":
            arguments.append(
                RuntimeArgument.tensor(
                    tuple(item["shape"]),
                    item["dtype"],
                    strides=(
                        tuple(item["strides"])
                        if item["strides"] is not None
                        else None
                    ),
                )
            )
        elif item["kind"] == "scalar":
            arguments.append(RuntimeArgument.scalar(item["value"], item["dtype"]))
        elif item["kind"] == "none":
            arguments.append(RuntimeArgument.none())
        else:
            raise ManifestError(
                f"case {case['case_id']} exposes representation-only {name!r} "
                "through the semantic CuTe entry"
            )
    options = dict(specialization["compiler_options"])
    try:
        block_threads = int(options.pop("block_threads"))
        smem_bytes = int(options.pop("smem_bytes"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(
            f"case {case['case_id']} lacks frozen CuTe block/smem C"
        ) from exc
    schedule = {
        **{
            item["name"]: item["value"]
            for item in _constexpr_parameters(case)
        },
        **options,
    }
    return CuteCompileCase(
        arguments,
        tuple(specialization["grid"]),
        (block_threads, 1, 1),
        smem_bytes,
        schedule,
    )


def cute_entry_symbol(source: str, device_symbol: str) -> str:
    """Resolve the unique public ``@cute.jit`` entry that invokes the pin."""

    tree = ast.parse(source)
    matches: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = [ast.unparse(item) for item in node.decorator_list]
        if "cute.jit" not in decorators:
            continue
        invokes_device = any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == device_symbol
            for item in ast.walk(node)
        )
        if invokes_device:
            matches.append(node.name)
    if len(matches) != 1:
        raise ManifestError(
            f"CuTe symbol {device_symbol!r} must have one @cute.jit compile entry; "
            f"found {matches!r}"
        )
    return matches[0]


def probe_triton_compilation(
    manifest: BatchManifest,
    *,
    timeout: float = 300.0,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compile every frozen Triton source without launching it or calling an LLM."""

    from intentir.verify.gate1.compiler import KernelSource, TritonCompiler

    compiler = TritonCompiler(
        timeout=timeout,
        cache_dir=cache_dir,
        env={"PYTHONPATH": compiler_pythonpath(manifest)},
    )
    results: list[dict[str, Any]] = []
    for case in manifest.cases:
        if case["source"]["frontend"] != "triton":
            results.append(
                {
                    "case_id": case["case_id"],
                    "frontend": case["source"]["frontend"],
                    "status": "not_triton",
                    "ok": False,
                }
            )
            continue
        path = resolve_source_path(manifest, case)
        result = compiler.compile(
            KernelSource.from_file(
                path, case["source"]["symbol"], trusted=True
            ),
            triton_compile_case(case),
            cache_key=f"real-batch-28:{manifest.digest}:{case['case_id']}",
        )
        results.append(
            {
                "case_id": case["case_id"],
                "frontend": "triton",
                "status": "compiled" if result.ok else "compile_failed",
                "ok": result.ok,
                "elapsed_ms": result.elapsed_ms,
                "cache_hit": result.cache_hit,
                "cache_key": result.cache_key,
                "ttir_sha256": (
                    hashlib.sha256(result.ttir.encode()).hexdigest()
                    if result.ttir is not None
                    else None
                ),
                "ttgir_sha256": (
                    hashlib.sha256(result.ttgir.encode()).hexdigest()
                    if result.ttgir is not None
                    else None
                ),
                "diagnostic": result.diagnostic,
            }
        )
    triton_results = [item for item in results if item["frontend"] == "triton"]
    return {
        "schema": "intentir.real-llm-batch-compile-probe.v1",
        "manifest_digest": manifest.digest,
        "triton_total": len(triton_results),
        "triton_compiled": sum(bool(item["ok"]) for item in triton_results),
        "all_triton_compiled": all(bool(item["ok"]) for item in triton_results),
        "cases": results,
    }


def probe_cute_compilation(
    manifest: BatchManifest,
    *,
    timeout: float = 300.0,
    cache_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compile every CuTe anchor and apply the strict MLIR/PTX observer."""

    from intentir.verify.gate1.cute import fingerprint_cute
    from intentir.verify.gate1.cute_compiler import CuteCompiler, CuteKernelSource

    compiler = CuteCompiler(timeout=timeout, cache_dir=cache_dir, arch="sm_120a")
    results: list[dict[str, Any]] = []
    for case in manifest.cases:
        if case["source"]["frontend"] != "cute":
            continue
        path = resolve_source_path(manifest, case)
        source = path.read_text(encoding="utf-8")
        entry = cute_entry_symbol(source, case["source"]["symbol"])
        compile_case = cute_compile_case(case)
        result = compiler.compile(
            CuteKernelSource.from_file(
                path,
                case["source"]["symbol"],
                entry,
                trusted=True,
            ),
            compile_case,
            cache_key=f"real-batch-28:{manifest.digest}:{case['case_id']}",
        )
        fingerprint = None
        diagnostic = result.diagnostic
        if result.ok and result.mlir is not None and result.ptx is not None:
            try:
                fingerprint = fingerprint_cute(
                    result.mlir,
                    result.ptx,
                    python_interface=result.python_interface,
                    schedule={
                        **compile_case.schedule,
                        "grid": compile_case.grid,
                        "block": compile_case.block,
                        "smem_bytes": compile_case.smem_bytes,
                    },
                )
            except Exception as exc:
                diagnostic = f"{type(exc).__name__}: {exc}"
        ok = bool(result.ok and fingerprint is not None)
        results.append(
            {
                "case_id": case["case_id"],
                "frontend": "cute",
                "entry_symbol": entry,
                "status": "compiled" if ok else "compile_failed",
                "ok": ok,
                "elapsed_ms": result.elapsed_ms,
                "cache_hit": result.cache_hit,
                "cache_key": result.cache_key,
                "mlir_sha256": (
                    hashlib.sha256(result.mlir.encode()).hexdigest()
                    if result.mlir is not None
                    else None
                ),
                "ptx_sha256": (
                    hashlib.sha256(result.ptx.encode()).hexdigest()
                    if result.ptx is not None
                    else None
                ),
                "cubin_sha256": result.cubin_sha256,
                "fingerprint": fingerprint.to_dict() if fingerprint is not None else None,
                "diagnostic": diagnostic,
            }
        )
    return {
        "schema": "intentir.real-llm-batch-cute-compile-probe.v1",
        "manifest_digest": manifest.digest,
        "cute_total": len(results),
        "cute_compiled": sum(bool(item["ok"]) for item in results),
        "all_cute_compiled": all(bool(item["ok"]) for item in results),
        "cases": results,
    }


def _command_identity(command: Path, *arguments: str) -> dict[str, Any]:
    if not command.is_file() or not os.access(command, os.X_OK):
        return {"available": False, "path": str(command)}
    process = subprocess.run(
        [str(command), *arguments], capture_output=True, text=True, timeout=30
    )
    text = (process.stdout or process.stderr).strip().splitlines()
    return {
        "available": process.returncode == 0,
        "path": str(command),
        "identity": text[:3],
    }


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def probe_environment(manifest: BatchManifest) -> dict[str, Any]:
    """Return a secret-free readiness snapshot; never make an LLM call."""

    result: dict[str, Any] = {
        "schema": "intentir.real-llm-batch-environment.v1",
        "manifest_digest": manifest.digest,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "frontends": {
            name: importlib.util.find_spec(name) is not None
            for name in ("triton", "cutlass")
        },
        "packages": {
            distribution: _installed_version(distribution)
            for distribution in (
                "nvidia-cutlass-dsl",
                "nvidia-cutlass-dsl-libs-base",
                "nvidia-cutlass-dsl-libs-cu13",
            )
        },
    }
    try:
        import torch
        import triton

        cuda = bool(torch.cuda.is_available())
        result["torch"] = str(torch.__version__)
        result["triton"] = str(triton.__version__)
        result["cuda"] = {
            "available": cuda,
            "device_count": torch.cuda.device_count(),
            "device": torch.cuda.get_device_name(0) if cuda else None,
            "compute_capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
            "torch_cuda": str(torch.version.cuda),
        }
    except Exception as exc:
        result["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    repository_root = Path(__file__).resolve().parents[3]
    result["toolchain"] = {
        "intent_opt": _command_identity(
            repository_root / "build" / "tools" / "intent-opt" / "intent-opt",
            "--version",
        ),
        "mlir_opt": _command_identity(Path("/usr/lib/llvm-20/bin/mlir-opt"), "--version"),
        "ptxas": _command_identity(Path("/usr/local/cuda-12.8/bin/ptxas"), "--version"),
    }
    provider = manifest.data["provider"]
    config_path = repository_root / "llm_providers.local.json"
    configured = False
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            selected = config.get(provider["name"], {})
            configured = bool(
                isinstance(selected, dict)
                and selected.get("model") == provider["model"]
                and selected.get("api_key")
                and selected.get("base_url")
            )
        except (OSError, json.JSONDecodeError):
            configured = False
    result["provider"] = {
        "name": provider["name"],
        "model": provider["model"],
        "configured": configured,
    }
    result["ready"] = bool(
        result.get("cuda", {}).get("available")
        and result["frontends"]["triton"]
        and all(item["available"] for item in result["toolchain"].values())
        and configured
        and (
            result["frontends"]["cutlass"]
            or not any(case["source"]["frontend"] == "cute" for case in manifest.cases)
        )
    )
    return result


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--verify-repositories", action="store_true")
    parser.add_argument("--compile-triton", action="store_true")
    parser.add_argument("--compile-cute", action="store_true")
    parser.add_argument("--compile-cache", type=Path)
    parser.add_argument("--compile-timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    manifest = load_batch_manifest(
        args.manifest,
        source_root=args.source_root,
        verify_sources=True,
        verify_repositories=args.verify_repositories,
    )
    report = {"environment": probe_environment(manifest)}
    if args.compile_triton:
        report["triton_compilation"] = probe_triton_compilation(
            manifest,
            timeout=args.compile_timeout,
            cache_dir=args.compile_cache,
        )
    if args.compile_cute:
        report["cute_compilation"] = probe_cute_compilation(
            manifest,
            timeout=args.compile_timeout,
            cache_dir=args.compile_cache,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
