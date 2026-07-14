"""LLM lift-agent driver for SCRC J/pi/C/interface artifacts.

Unlike the archived extraction validator, this driver does not interpret the
JSON.  It only obtains and parses the lift proposal; Gate1 and Gate2 decide
whether that proposal is faithful to the source kernel.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.model import (
    Gate1Artifact,
    ProvenanceRecord,
    ProvenanceRole,
)
from intentir.verify.gate1.rules import RuleRegistry, default_triton_rule_registry

from .prompt import SCHEMA as INTENTIR_SCHEMA


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class SCRCProvenance:
    model: str
    prompt_sha256: str
    attempts: int
    response_id: str | None = None
    repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SCRCExtraction:
    artifact: Gate1Artifact
    provenance: SCRCProvenance


def _json_object(text: str) -> dict:
    match = _FENCE.search(text)
    if match:
        text = match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("lift response contains no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("lift response root must be an object")
    return value


def _rule_catalog(registry: RuleRegistry) -> str:
    cards = [
        {
            "id": card.id,
            "applies_to": list(card.applies_to),
            "reads": list(card.reads),
        }
        for card in registry.cards
    ]
    return json.dumps(cards, ensure_ascii=False, indent=2)


def build_scrc_lift_instructions(registry: RuleRegistry) -> str:
    return f"""You are the SCRC lift agent. You may read the source kernel.
Return one JSON object for Gate1Artifact. Do not emit prose or markdown.

J.intent must follow this IntentIR vocabulary:
{INTENTIR_SCHEMA}

In addition, return exactly these top-level fields:
- j: {{kernel_id, intent, node_ids, boundary}}, where node_ids is an object
  mapping opaque ids to JSON paths, for example {{"n0":"body[0]"}}; never an array
- interface: {{kernel_id, frontend, function_name, parameters, grid,
  compiler_slots}}
- schedule: {{slots, values}}; values are the source implementation's C values
- plan: {{steps:[{{rule,node_ref?,refs?}}]}}
- location_map: {{source_name, entries:[{{span:{{start_line,end_line}},
  node_refs,access_refs}}]}}

Boundary is Layer B and may contain only domains, accesses, broadcasts,
aliases, effects, output_mappings and reductions. Every load/store access uses
exactly these field names: access_id, kind, tensor, index_expression, optional
predicate, optional fill_value, optional node_ref. Never emit a field named
"fill". Every tiled reduction records node_ref, tile_axis and exactly one of
identity or masked_contribution. Every effects entry uses exactly kind, target
and optional node_ref; access details stay in accesses, not effects.

pi/plan rules:
- every step selects one registered rule id below;
- node_ref/refs point only to identifiers declared in J;
- never copy source text, expressions or frontend API names into plan;
- semantics live in J, implementation mechanism selection lives in plan.

Registered rule ids (API spellings intentionally omitted):
{_rule_catalog(registry)}

Do not emit source or provenance fields; the driver attaches them itself.
"""


def extract_scrc_artifact(
    source: str,
    name_hint: str,
    *,
    registry: RuleRegistry | None = None,
    provider: ProviderConfig | None = None,
    max_repairs: int = 3,
    effort: str | None = None,
) -> SCRCExtraction:
    """Lift a real source kernel to the full SCRC artifact through the LLM."""

    registry = registry or default_triton_rule_registry()
    instructions = build_scrc_lift_instructions(registry)
    user = (
        f'Lift the following kernel. Use kernel_id "{name_hint}" and preserve '
        "its actual callable entry name in interface.function_name.\n\nSOURCE:\n"
        + source
    )
    prompt_sha = hashlib.sha256(
        (instructions + "\x00" + user).encode("utf-8")
    ).hexdigest()
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    repairs: list[str] = []
    conversation = user

    for attempt in range(max_repairs + 1):
        reply = call_responses(
            instructions, conversation, provider=provider, effort=effort
        )
        try:
            payload = _json_object(reply.text)
            if set(payload) == {"gate1_artifact"}:
                payload = payload["gate1_artifact"]
            if not isinstance(payload, dict):
                raise TypeError("gate1_artifact must be an object")
            payload = dict(payload)
            node_ids = payload.get("j", {}).get("node_ids")
            if isinstance(node_ids, list):
                normalized_ids: dict[str, str] = {}
                for index, item in enumerate(node_ids):
                    if not isinstance(item, dict):
                        raise TypeError("node_ids array entries must be objects")
                    node_id = item.get("id", item.get("node_id"))
                    path = item.get("path", item.get("json_path"))
                    if not isinstance(node_id, str) or not isinstance(path, str):
                        raise TypeError(
                            f"node_ids[{index}] requires string id/node_id and path/json_path"
                        )
                    normalized_ids[node_id] = path
                payload["j"]["node_ids"] = normalized_ids
            # One observed provider spelling uses the plan's prose term
            # ``fill`` for the persisted ``fill_value`` field. Normalize only
            # this enumerated wire alias before the strict artifact parser.
            boundary = payload.get("j", {}).get("boundary", {})
            for access in boundary.get("accesses", []):
                if "fill" in access and "fill_value" not in access:
                    access["fill_value"] = access.pop("fill")
            for effect in boundary.get("effects", []):
                if "target" not in effect and isinstance(effect.get("tensor"), str):
                    effect["target"] = effect["tensor"]
                for extra in (
                    "effect_id",
                    "index_expression",
                    "predicate",
                    "tensor",
                    "value",
                ):
                    effect.pop(extra, None)
            payload["source"] = source
            payload["provenance"] = [
                ProvenanceRecord(
                    ProvenanceRole.LIFTER,
                    "intentir.extraction.lifters.scrc",
                    model=reply.model,
                    source_digest=source_digest,
                    metadata={
                        "prompt_sha256": prompt_sha,
                        "response_id": reply.raw.get("id"),
                    },
                ).to_json()
            ]
            artifact = Gate1Artifact.from_json(payload)
            registry.resolve(
                artifact.plan, frontend=artifact.interface.frontend
            )
            return SCRCExtraction(
                artifact,
                SCRCProvenance(
                    reply.model,
                    prompt_sha,
                    attempt + 1,
                    response_id=reply.raw.get("id"),
                    repairs=tuple(repairs),
                ),
            )
        except Exception as exc:
            diagnostic = f"{type(exc).__name__}: {exc}"
            repairs.append(diagnostic)
            if attempt == max_repairs:
                raise RuntimeError(
                    f"SCRC lift failed after {attempt + 1} tries; last: {diagnostic}"
                ) from exc
            conversation = (
                user
                + "\n\nYour previous artifact could not be parsed:\n"
                + diagnostic
                + "\nRe-emit the complete corrected Gate1Artifact JSON only."
            )

    raise AssertionError("unreachable")


__all__ = [
    "SCRCExtraction",
    "SCRCProvenance",
    "build_scrc_lift_instructions",
    "extract_scrc_artifact",
]
