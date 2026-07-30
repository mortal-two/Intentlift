"""Exact compiler observation exposed only to the source-aware Lift owner.

This module deliberately stops at compilation and the existing Gate1 observer.
It never interprets the observation as J/C/pi and never constructs artifact
fields.  The same frozen anonymous Source specialization is compiled once and
stored in the branch-local read-only Lift input tree for focused queries.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

from intentir.verify.gate1.compiler import (
    CompileCase,
    KernelSource,
    RuntimeArgument,
    TritonCompiler,
)
from intentir.verify.gate1.ttir import fingerprint_ttir
from intentir.verify.gate1.replay import compiled_packet_obligation_diagnostics


SOURCE_TTIR_OBSERVATION_PROTOCOL = "intentir.lift-source-ttir.v1"
SOURCE_TTIR_RECONCILE_PROTOCOL = "intentir.lift-source-reconcile.v1"

_DTYPE_ALIASES = {
    "f16": "float16",
    "bf16": "bfloat16",
    "f32": "float32",
    "f64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
}


def _runtime_argument(value: Mapping[str, Any]) -> RuntimeArgument:
    kind = value.get("kind")
    dtype_value = value.get("dtype")
    if not isinstance(dtype_value, str) or not dtype_value:
        raise ValueError("anonymous compiler argument requires a dtype")
    dtype = _DTYPE_ALIASES.get(dtype_value, dtype_value)
    if kind == "tensor":
        shape = value.get("shape")
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in shape
        ):
            raise ValueError("anonymous tensor argument requires an integer shape")
        return RuntimeArgument.tensor(tuple(shape), dtype)
    if kind == "scalar":
        scalar = value.get("value")
        if isinstance(scalar, bool) or isinstance(scalar, (int, float)):
            return RuntimeArgument.scalar(scalar, dtype)
        raise ValueError("anonymous scalar argument requires a numeric value")
    if kind == "none":
        return RuntimeArgument.none(dtype)
    raise ValueError(
        "Source TTIR observation supports tensor/scalar/none compiler arguments"
    )


def _compile_case(view: Mapping[str, Any]) -> CompileCase:
    raw_arguments = view.get("arguments")
    if not isinstance(raw_arguments, list) or not raw_arguments:
        raise ValueError("anonymous compiler view requires runtime arguments")
    arguments: dict[str, RuntimeArgument] = {}
    for raw in raw_arguments:
        if not isinstance(raw, Mapping):
            raise TypeError("anonymous compiler arguments must be objects")
        name = raw.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("anonymous compiler argument requires an identifier")
        if name in arguments:
            raise ValueError("anonymous compiler argument names must be unique")
        arguments[name] = _runtime_argument(raw)

    raw_grid = view.get("grid")
    if not isinstance(raw_grid, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw_grid
    ):
        raise ValueError("anonymous compiler view requires an integer grid")

    schedule: dict[str, bool | int | float | str] = {}
    for field in ("bound_constexpr_values", "schedule_values"):
        values = view.get(field, {})
        if not isinstance(values, Mapping):
            raise TypeError(f"anonymous compiler view {field} must be an object")
        for name, value in values.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("anonymous constexpr names must be identifiers")
            if name in schedule:
                raise ValueError("anonymous constexpr bindings must be disjoint")
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError("anonymous constexpr values must be JSON scalars")
            schedule[name] = value
    return CompileCase(arguments, tuple(raw_grid), schedule)


def observe_anonymous_source_ttir(
    anonymous_source: str,
    compiler_view: Mapping[str, Any] | None,
    *,
    compiler: TritonCompiler | None = None,
) -> dict[str, Any]:
    """Compile one exact anonymous Source specialization and expose facts only.

    Failure is returned as a closed observation rather than being reclassified as
    a model failure.  The caller may require a passing observation for a formal
    capability run, while still retaining the exact compiler diagnostic.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_OBSERVATION_PROTOCOL,
        "authority": (
            "exact anonymous Source compilation and existing Gate1 observer only; "
            "no J/C/pi field, ownership decision, or repair is generated"
        ),
        "function": "scrc_k0",
    }
    if not isinstance(anonymous_source, str) or not anonymous_source:
        return {**base, "status": "fail", "diagnostic": "anonymous Source is absent"}
    if not isinstance(compiler_view, Mapping):
        return {
            **base,
            "status": "unavailable",
            "diagnostic": "anonymous compiler view is absent",
        }
    try:
        case = _compile_case(compiler_view)
        result = (compiler or TritonCompiler()).compile(
            KernelSource.from_code(anonymous_source, "scrc_k0", trusted=True),
            case,
        )
        if not result.ok or not isinstance(result.ttir, str) or not result.ttir:
            return {
                **base,
                "status": "fail",
                "diagnostic": result.diagnostic or "compiler returned no TTIR",
                "elapsed_ms": result.elapsed_ms,
                "compiler": dict(result.compiler),
                "target": dict(result.target),
            }
        fingerprint = fingerprint_ttir(
            result.ttir,
            python_interface=result.python_interface,
            schedule=case.schedule,
        )
        return {
            **base,
            "status": "pass",
            "ttir": result.ttir,
            "fingerprint": fingerprint.to_dict(),
            "fingerprint_digest": fingerprint.digest,
            "python_interface": dict(result.python_interface),
            "compiler": dict(result.compiler),
            "target": dict(result.target),
            "elapsed_ms": result.elapsed_ms,
            "cache_hit": result.cache_hit,
        }
    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }


def reconcile_candidate_source_ttir(
    candidate_text: str,
    observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Diagnose candidate-authored compiled obligations against Source TTIR.

    This runs only inside source-aware Lift.  It neither constructs artifact
    fields nor contributes to TypeCheck, Gate1, or certification.
    """

    base: dict[str, Any] = {
        "protocol": SOURCE_TTIR_RECONCILE_PROTOCOL,
        "authority": (
            "source-aware diagnostic comparison of model-authored closed pi "
            "obligations with the existing Source TTIR observer; reject-only, "
            "no artifact construction and no Gate or certification authority"
        ),
    }
    if not isinstance(observation, Mapping) or observation.get("status") != "pass":
        return {
            **base,
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    ttir = observation.get("ttir")
    if not isinstance(ttir, str) or not ttir:
        return {
            **base,
            "status": "pass",
            "applicable": False,
            "diagnostics": [],
        }
    try:
        packet = json.loads(candidate_text)
        if not isinstance(packet, Mapping):
            raise TypeError("candidate root is not an object")
        diagnostics = compiled_packet_obligation_diagnostics(
            packet,
            ttir,
            path_prefix="<source-ttir-obligation>",
        )
        diagnostics = (
            *diagnostics,
            *_unowned_source_runtime_carrier_diagnostics(packet, ttir),
        )
    except Exception as exc:
        return {
            **base,
            "status": "fail",
            "applicable": True,
            "diagnostics": [
                {
                    "rule_code": "source_reconcile_observer_failed",
                    "path": "<source-ttir-obligation>",
                    "anonymous_handles": [],
                    "observed_refs_or_types": [type(exc).__name__],
                    "line": None,
                }
            ],
        }
    return {
        **base,
        "status": "fail" if diagnostics else "pass",
        "applicable": True,
        "diagnostics": [dict(item) for item in diagnostics],
    }


def _unowned_source_runtime_carrier_diagnostics(
    packet: Mapping[str, Any],
    ttir: str,
) -> tuple[Mapping[str, Any], ...]:
    """Reject Source-used ABI carriers that have no authored semantic owner.

    A null ``logical_ref`` says that a callable carrier is not a J logical value;
    it does not authorize Replay to guess whether or where that carrier participates
    in Source realization.  If the exact Source specialization still uses such an
    argument, the Lift owner must expose its role through authored pi occurrences
    (or revise the J ownership).  This check observes only occurrence/non-occurrence:
    it does not choose either ownership or construct a route.
    """

    j = packet.get("j")
    pi = packet.get("pi")
    if not isinstance(j, Mapping) or not isinstance(pi, Mapping):
        return ()
    signature = j.get("signature")
    parameters = (
        signature.get("parameters") if isinstance(signature, Mapping) else None
    )
    instances = pi.get("instances")
    if not isinstance(parameters, list) or not isinstance(instances, list):
        return ()

    pi_refs = {
        item.get("semantic_ref")
        for item in instances
        if isinstance(item, Mapping) and isinstance(item.get("semantic_ref"), str)
    }
    diagnostics: list[Mapping[str, Any]] = []
    for parameter in parameters:
        if not isinstance(parameter, Mapping):
            continue
        name = parameter.get("name")
        kind = parameter.get("kind")
        logical_ref = parameter.get("logical_ref")
        if (
            not isinstance(name, str)
            or kind not in {"pointer", "scalar"}
            or logical_ref is not None
        ):
            continue
        # The function signature contributes exactly one occurrence.  A second
        # exact SSA-token occurrence means the compiled specialization consumes
        # this runtime argument in its body.  The token boundary excludes derived
        # names such as %p11_3.
        token = re.compile(rf"%{re.escape(name)}(?![A-Za-z0-9_.$])")
        source_used = len(token.findall(ttir)) > 1
        pi_owned = name in pi_refs
        if source_used and not pi_owned:
            diagnostics.append(
                {
                    "rule_code": "source_runtime_carrier_without_owner",
                    "path": (
                        "<source-ttir-obligation> j.signature.parameters: "
                        f"Source-used runtime carrier {name} has logical_ref=null "
                        "and no pi instance occurrence"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": ["source-ttir-runtime-use"],
                    "line": None,
                }
            )
        elif pi_owned and not source_used:
            diagnostics.append(
                {
                    "rule_code": "source_unused_runtime_carrier_has_occurrence",
                    "path": (
                        "<source-ttir-obligation> pi.instances: "
                        f"runtime carrier {name} has logical_ref=null and a pi "
                        "occurrence, but the selected Source TTIR has no body use"
                    ),
                    "anonymous_handles": [name],
                    "observed_refs_or_types": ["source-ttir-runtime-unused"],
                    "line": None,
                }
            )
    return tuple(diagnostics[:20])


__all__ = [
    "SOURCE_TTIR_RECONCILE_PROTOCOL",
    "SOURCE_TTIR_OBSERVATION_PROTOCOL",
    "observe_anonymous_source_ttir",
    "reconcile_candidate_source_ttir",
]
