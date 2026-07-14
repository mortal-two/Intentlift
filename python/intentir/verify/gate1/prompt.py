"""Source-isolated replay prompts for normal and masked-J Gate1 runs."""

from __future__ import annotations

import json
from typing import Any

from .model import Gate1Artifact, ReplayPlan, SchemaError
from .rules import RuleRegistry


_PROMPT_PREAMBLE = """You are the replay agent in the SCRC Gate1 pipeline.
Generate one parameterized kernel for the declared frontend using only the JSON
packet below. You have no source-kernel access. Treat J as the sole authority
for semantics and boundary behavior, pi as implementation-mechanism references,
and rule_cards as the sole authority for concrete frontend API spellings.

Hard constraints:
1. Preserve function_name, parameter order/modes/dtypes/ranks, constexpr slots,
   and the declared grid contract.
2. Implement every J node and every boundary access/effect referenced by pi.
3. Keep every schedule slot symbolic. Never invent or bake in a C value. Only
   interface parameters with kind=constexpr belong in the kernel signature or
   body; schedule kind=compiler and interface.compiler_slots are launch-only
   options and must never be referenced as kernel variables.
4. Do not infer extra semantics from identifier spelling.
5. J reduction axes are global tensor axes; frontend reduction calls use the
   matching boundary.reductions.tile_axis for the local tile value.
6. A semantic broadcast may be implicit on a local tile. Do not emit an explicit
   rank change unless the selected rule and local operand ranks require it.
7. Reproduce each declared access predicate exactly; do not strengthen it with
   redundant domain bounds.
8. Return kernel source only, with no Markdown fence or explanatory prose.
"""

_MASKED_NOTE = """
ABLATION RUN: J's semantic and boundary contents were deliberately masked and
all J-local and ABI/C identifiers were consistently alpha-renamed. Preserve the
opaque function/parameter order, grid contract, and slot declarations exactly;
the harness will restore source ABI spellings deterministically before using the
same compile case. Do not reverse or guess masked names. Produce the best replay
possible from the remaining packet.
"""


def build_rewriter_payload(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    masked: bool = False,
) -> dict[str, Any]:
    """Build the only payload that may cross the lift/replay isolation boundary."""

    if not isinstance(artifact, Gate1Artifact):
        artifact = Gate1Artifact.from_json(artifact)
    if not isinstance(registry, RuleRegistry):
        registry = RuleRegistry.from_json(registry)
    view = artifact.rewriter_view(masked=masked)
    selected_plan = (
        ReplayPlan.from_json(view["plan"]) if masked else artifact.plan
    )
    retrieval = registry.rewriter_view(
        selected_plan, frontend=artifact.interface.frontend,
    )
    view["rule_selection"] = retrieval["selection"]
    view["rule_cards"] = retrieval["cards"]
    view["rule_registry_digest"] = registry.digest
    assert_source_isolated_payload(view)
    return view


def build_normal_rewriter_view(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> dict[str, Any]:
    """Return the normal source-isolated JSON view without rendering a prompt."""

    return build_rewriter_payload(artifact, registry, masked=False)


def build_masked_rewriter_view(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
) -> dict[str, Any]:
    """Return the opaque masked-J ablation view without rendering a prompt."""

    return build_rewriter_payload(artifact, registry, masked=True)


def assert_source_isolated_payload(payload: dict[str, Any]) -> None:
    """Fail closed if privileged artifact fields enter a replay packet."""

    forbidden_paths: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if key in {"source", "source_text", "location_map", "provenance"}:
                    forbidden_paths.append(child_path)
                if key == "values" and path.endswith("schedule"):
                    forbidden_paths.append(child_path)
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload, "")
    if forbidden_paths:
        raise SchemaError(
            "rewriter payload contains privileged field(s): " + ", ".join(forbidden_paths)
        )


def build_rewriter_prompt(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    masked: bool = False,
    indent: int | None = 2,
) -> str:
    payload = build_rewriter_payload(artifact, registry, masked=masked)
    preamble = _PROMPT_PREAMBLE + (_MASKED_NOTE if masked else "")
    return preamble + "\nREPLAY_PACKET_JSON:\n" + json.dumps(
        payload,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
    )


def build_normal_rewriter_prompt(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    indent: int | None = 2,
) -> str:
    return build_rewriter_prompt(artifact, registry, masked=False, indent=indent)


def build_masked_rewriter_prompt(
    artifact: Gate1Artifact,
    registry: RuleRegistry,
    *,
    indent: int | None = 2,
) -> str:
    return build_rewriter_prompt(artifact, registry, masked=True, indent=indent)


__all__ = [
    "assert_source_isolated_payload", "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload", "build_rewriter_prompt",
]
