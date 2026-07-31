"""Episode-local, agent-authored Replay source composition.

The composer owns no lowering rule.  An agent writes every source fragment and,
optionally, opts into route-bound placeholders.  Fixed code then performs only
closed name allocation, exact pi-route reference substitution, ordering checks,
and lexical rendering.  It never chooses an operator, expression, predicate, or
target spelling.
"""

from __future__ import annotations

import json
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ROUTE_COMPOSER_PROTOCOL_V1 = "intentir.replay-route-composer.v1"
ROUTE_COMPOSER_PROTOCOL = "intentir.replay-route-composer.v2"
_INSTANCE_FRAGMENT = re.compile(r"[0-9]{3,6}-(i[0-9]+)\.source")
_LOCAL_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PLACEHOLDER = re.compile(r"\{\{(out|slot:([0-9]+)|dep:([0-9]+))\}\}")
_GROUPED_PLACEHOLDER = re.compile(
    r"\{\{(out|slot|dep):(i[0-9]+)(?::([0-9]+))?\}\}"
)


@dataclass(frozen=True)
class ComposerFragment:
    name: str
    text: str
    byte_count: int


@dataclass(frozen=True)
class RouteComposerResult:
    source: str
    fragments: tuple[ComposerFragment, ...]
    bound_instances: tuple[str, ...]
    occurrence_instances: tuple[str, ...]
    consumed_routes: int
    total_routes: int
    protocol: str = ROUTE_COMPOSER_PROTOCOL_V1
    mode: str = "route_bound"

    def to_response(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "status": "pass",
            "mode": self.mode,
            "fragments": [
                {"name": item.name, "bytes": item.byte_count}
                for item in self.fragments
            ],
            "total_bytes": len(self.source.encode("utf-8")),
            "bound_instances": len(self.bound_instances),
            "occurrence_instances": len(self.occurrence_instances),
            "consumed_routes": self.consumed_routes,
            "total_routes": self.total_routes,
            "authority": (
                "agent-authored target syntax; fixed code only substitutes stable "
                "locals along explicit pi routes and checks closed references/order"
            ),
        }


def _closed_manifest(path: Path) -> tuple[str, dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("work/replay.compose.json must be a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or set(value) != {"protocol", "bindings"}:
        raise ValueError(
            "replay.compose.json must contain exactly protocol and bindings"
        )
    protocol = value.get("protocol")
    if protocol not in {ROUTE_COMPOSER_PROTOCOL_V1, ROUTE_COMPOSER_PROTOCOL}:
        raise ValueError("unsupported Replay route-composer protocol")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("replay.compose.json bindings must be an object")
    result: dict[str, str] = {}
    for instance, symbol in bindings.items():
        if (
            not isinstance(instance, str)
            or re.fullmatch(r"i[0-9]+", instance) is None
            or not isinstance(symbol, str)
            or _LOCAL_NAME.fullmatch(symbol) is None
        ):
            raise ValueError(
                "route-composer bindings must map i* to Python identifiers"
            )
        result[instance] = symbol
    return str(protocol), result


def _validate_bindings(
    bindings: Mapping[str, str], instance_rows: Mapping[str, Mapping[str, Any]]
) -> None:
    for instance, symbol in bindings.items():
        row = instance_rows.get(instance)
        if row is None:
            raise ValueError(f"binding refers to undeclared instance {instance}")
        semantic_ref = row.get("semantic_ref")
        if semantic_ref != symbol or re.fullmatch(r"[pc][0-9]+", symbol) is None:
            raise ValueError(
                f"binding {instance}->{symbol} must be the instance's exact p*/c* "
                "signature carrier"
            )


def _route_tables(
    routes: Sequence[Mapping[str, Any]],
    instance_rows: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[int, str]], dict[str, list[str]], set[str], int]:
    inbound: dict[str, dict[int, str]] = {}
    dependencies: dict[str, list[str]] = {}
    outbound: set[str] = set()
    route_count = 0
    for route in routes:
        source = route.get("source")
        target = route.get("target")
        slots = route.get("slots")
        if (
            not isinstance(source, str)
            or source not in instance_rows
            or not isinstance(target, str)
            or target not in instance_rows
            or not isinstance(slots, list)
        ):
            raise ValueError("pi.routes contains a malformed endpoint or slot list")
        if not slots:
            dependencies.setdefault(target, []).append(source)
            route_count += 1
            outbound.add(source)
            continue
        target_slots = inbound.setdefault(target, {})
        for slot in slots:
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot < 0
                or slot in target_slots
            ):
                raise ValueError(
                    f"pi route target {target} has a malformed or duplicate slot"
                )
            target_slots[slot] = source
            route_count += 1
        outbound.add(source)
    return inbound, dependencies, outbound, route_count


def _check_sharing(
    sharing: Sequence[Mapping[str, Any]],
    inbound: Mapping[str, Mapping[int, str]],
    dependencies: Mapping[str, Sequence[str]],
) -> None:
    for relation in sharing:
        source = relation.get("source")
        consumers = relation.get("consumers")
        if not isinstance(source, str) or not isinstance(consumers, list):
            raise ValueError("pi.sharing is malformed")
        for consumer in consumers:
            direct_sources = {
                *inbound.get(str(consumer), {}).values(),
                *dependencies.get(str(consumer), ()),
            }
            if not isinstance(consumer, str) or source not in direct_sources:
                raise ValueError(
                    f"pi.sharing {source} lacks a direct routed consumer {consumer}"
                )


def _render_grouped_route_composer(
    fragments: Sequence[ComposerFragment],
    *,
    bindings: Mapping[str, str],
    instances: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    order: Sequence[Mapping[str, Any]],
    sharing: Sequence[Mapping[str, Any]],
) -> RouteComposerResult:
    """Render agent-authored region/block fragments through exact opaque routes.

    Unlike the legacy one-fragment-per-occurrence surface, v2 lets one fragment
    contain any number of instance definitions and uses.  The agent still writes
    every target expression and explicitly places every placeholder; fixed code
    only allocates stable locals and verifies the already-authored pi topology.
    """

    instance_rows = {
        str(item.get("id")): item
        for item in instances
        if isinstance(item.get("id"), str)
    }
    if len(instance_rows) != len(instances):
        raise ValueError("pi.instances has duplicate or malformed ids")
    _validate_bindings(bindings, instance_rows)
    inbound, dependencies, outbound, route_count = _route_tables(
        routes, instance_rows
    )
    _check_sharing(sharing, inbound, dependencies)

    symbols = {
        instance: bindings.get(instance, f"_intentir_{instance}")
        for instance in instance_rows
    }
    seen_slots: dict[tuple[str, int], int] = {}
    seen_dependencies: dict[tuple[str, int], int] = {}
    out_counts: dict[str, int] = {}
    first_positions: dict[str, int] = {}
    rendered: list[ComposerFragment] = []
    absolute_position = 0

    for fragment in fragments:
        substitutions: list[tuple[str, str]] = []

        def substitute(token: re.Match[str]) -> str:
            kind = token.group(1)
            instance = token.group(2)
            ordinal_text = token.group(3)
            if instance not in instance_rows:
                raise ValueError(
                    f"fragment {fragment.name} references undeclared instance {instance}"
                )
            first_positions.setdefault(instance, absolute_position + token.start())
            if kind == "out":
                if ordinal_text is not None:
                    raise ValueError("grouped out placeholder cannot have an ordinal")
                if instance in bindings:
                    raise ValueError(
                        f"bound signature instance {instance} cannot define an output"
                    )
                out_counts[instance] = out_counts.get(instance, 0) + 1
                symbol = symbols[instance]
            else:
                if ordinal_text is None:
                    raise ValueError(
                        f"grouped {kind} placeholder requires an ordinal"
                    )
                ordinal = int(ordinal_text)
                if kind == "slot":
                    source = inbound.get(instance, {}).get(ordinal)
                    if source is None:
                        raise ValueError(
                            f"instance {instance} has no incoming route slot {ordinal}"
                        )
                    key = (instance, ordinal)
                    seen_slots[key] = seen_slots.get(key, 0) + 1
                else:
                    incoming = dependencies.get(instance, [])
                    if ordinal >= len(incoming):
                        raise ValueError(
                            f"instance {instance} has no empty-route dependency {ordinal}"
                        )
                    source = incoming[ordinal]
                    key = (instance, ordinal)
                    seen_dependencies[key] = seen_dependencies.get(key, 0) + 1
                symbol = symbols[source]
            marker = f"_intentir_placeholder_{len(rendered)}_{len(substitutions)}"
            if marker in fragment.text:
                raise ValueError("composer placeholder marker collides with source")
            substitutions.append((marker, symbol))
            return marker

        text = _GROUPED_PLACEHOLDER.sub(substitute, fragment.text)
        if _PLACEHOLDER.search(text) is not None:
            raise ValueError(
                "v2 grouped fragments may not mix legacy unqualified placeholders"
            )
        try:
            name_tokens = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.NAME
            ]
        except (IndentationError, tokenize.TokenError) as exc:
            raise ValueError(
                f"fragment {fragment.name} is not tokenizable Python syntax: {exc}"
            ) from exc
        hidden = [
            marker for marker, _symbol in substitutions if name_tokens.count(marker) != 1
        ]
        if hidden:
            raise ValueError(
                f"fragment {fragment.name} hides grouped route placeholders outside "
                f"Python name tokens: {hidden}"
            )
        for marker, symbol in substitutions:
            text = text.replace(marker, symbol)
        rendered.append(
            ComposerFragment(fragment.name, text, len(text.encode("utf-8")))
        )
        absolute_position += len(fragment.text)

    missing_slots = sorted(
        (target, slot)
        for target, slots in inbound.items()
        for slot in slots
        if seen_slots.get((target, slot), 0) == 0
    )
    repeated_slots = sorted(
        key for key, count in seen_slots.items() if count != 1
    )
    missing_dependencies = sorted(
        (target, ordinal)
        for target, sources in dependencies.items()
        for ordinal in range(len(sources))
        if seen_dependencies.get((target, ordinal), 0) == 0
    )
    repeated_dependencies = sorted(
        key for key, count in seen_dependencies.items() if count != 1
    )
    if missing_slots or repeated_slots or missing_dependencies or repeated_dependencies:
        raise ValueError(
            "grouped route composer must consume every routed slot/dependency exactly "
            f"once; missing_slots={missing_slots[:24]}, "
            f"repeated_slots={repeated_slots[:24]}, "
            f"missing_deps={missing_dependencies[:24]}, "
            f"repeated_deps={repeated_dependencies[:24]}"
        )
    for instance in sorted(outbound):
        if instance not in bindings and out_counts.get(instance, 0) != 1:
            raise ValueError(
                f"routed source instance {instance} must define one grouped out placeholder"
            )
    repeated_outputs = sorted(
        instance for instance, count in out_counts.items() if count != 1
    )
    if repeated_outputs:
        raise ValueError(
            f"grouped instance outputs must be unique: {repeated_outputs[:24]}"
        )
    for relation in order:
        before = relation.get("before")
        after = relation.get("after")
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("pi.order has malformed endpoints")
        if before in bindings or after in bindings:
            raise ValueError(
                "pi.order endpoint cannot be erased into a direct signature binding"
            )
        before_position = first_positions.get(before)
        after_position = first_positions.get(after)
        if before_position is None or after_position is None:
            raise ValueError("pi.order endpoint lacks a grouped composer occurrence")
        if before_position >= after_position:
            raise ValueError(
                f"grouped composer order violates pi.order {before}->{after}"
            )

    source = "".join(item.text for item in rendered)
    return RouteComposerResult(
        source=source,
        fragments=tuple(rendered),
        bound_instances=tuple(sorted(bindings)),
        occurrence_instances=tuple(sorted(first_positions)),
        consumed_routes=route_count,
        total_routes=route_count,
        protocol=ROUTE_COMPOSER_PROTOCOL,
        mode="grouped_route_bound",
    )


def _pi_rows(pi: Mapping[str, Any], name: str) -> list[Mapping[str, Any]]:
    value = pi.get(name)
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise ValueError(f"pi.{name} must be a canonical relation list")
    return list(value)


def render_route_composer(
    fragments: Sequence[ComposerFragment],
    *,
    manifest_path: Path,
    pi: Mapping[str, Any],
) -> RouteComposerResult:
    """Render exact model-authored templates through explicit pi route slots."""

    instances = _pi_rows(pi, "instances")
    routes = _pi_rows(pi, "routes")
    order = _pi_rows(pi, "order")
    sharing = _pi_rows(pi, "sharing")
    instance_rows = {
        str(item.get("id")): item
        for item in instances
        if isinstance(item.get("id"), str)
    }
    if len(instance_rows) != len(instances):
        raise ValueError("pi.instances has duplicate or malformed ids")

    protocol, bindings = _closed_manifest(manifest_path)
    if protocol == ROUTE_COMPOSER_PROTOCOL:
        return _render_grouped_route_composer(
            fragments,
            bindings=bindings,
            instances=instances,
            routes=routes,
            order=order,
            sharing=sharing,
        )
    _validate_bindings(bindings, instance_rows)

    occurrence_fragments: dict[str, ComposerFragment] = {}
    fragment_positions: dict[str, int] = {}
    for position, fragment in enumerate(fragments):
        match = _INSTANCE_FRAGMENT.fullmatch(fragment.name)
        if match is None:
            if _PLACEHOLDER.search(fragment.text) is not None:
                raise ValueError(
                    f"raw fragment {fragment.name} may not contain route placeholders"
                )
            continue
        instance = match.group(1)
        if instance not in instance_rows:
            raise ValueError(
                f"fragment {fragment.name} refers to undeclared instance {instance}"
            )
        if instance in occurrence_fragments or instance in bindings:
            raise ValueError(f"instance {instance} has more than one local definition")
        occurrence_fragments[instance] = fragment
        fragment_positions[instance] = position

    inbound: dict[str, dict[int, str]] = {}
    dependencies: dict[str, list[str]] = {}
    outbound: set[str] = set()
    route_count = 0
    for route in routes:
        source = route.get("source")
        target = route.get("target")
        slots = route.get("slots")
        if (
            not isinstance(source, str)
            or source not in instance_rows
            or not isinstance(target, str)
            or target not in instance_rows
            or not isinstance(slots, list)
        ):
            raise ValueError("pi.routes contains a malformed endpoint or slot list")
        if not slots:
            dependencies.setdefault(target, []).append(source)
            route_count += 1
            outbound.add(source)
            continue
        target_slots = inbound.setdefault(target, {})
        for slot in slots:
            if (
                isinstance(slot, bool)
                or not isinstance(slot, int)
                or slot < 0
                or slot in target_slots
            ):
                raise ValueError(
                    f"pi route target {target} has a malformed or duplicate slot"
                )
            target_slots[slot] = source
            route_count += 1
        outbound.add(source)

    missing_targets = sorted(
        target
        for target in {*inbound, *dependencies}
        if target not in occurrence_fragments and target not in bindings
    )
    if missing_targets:
        preview = ", ".join(missing_targets[:24])
        raise ValueError(f"route target occurrence fragment missing: {preview}")
    missing_sources = sorted(
        source
        for source in outbound
        if source not in occurrence_fragments and source not in bindings
    )
    if missing_sources:
        preview = ", ".join(missing_sources[:24])
        raise ValueError(f"route source local definition missing: {preview}")

    for relation in order:
        before = relation.get("before")
        after = relation.get("after")
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("pi.order has malformed endpoints")
        if before in bindings or after in bindings:
            raise ValueError(
                "pi.order endpoint cannot be erased into a direct signature binding"
            )
        before_position = fragment_positions.get(before)
        after_position = fragment_positions.get(after)
        if before_position is None or after_position is None:
            raise ValueError("pi.order endpoint lacks a composer occurrence")
        if before_position >= after_position:
            raise ValueError(
                f"composer fragment order violates pi.order {before}->{after}"
            )

    for relation in sharing:
        source = relation.get("source")
        consumers = relation.get("consumers")
        if not isinstance(source, str) or not isinstance(consumers, list):
            raise ValueError("pi.sharing is malformed")
        for consumer in consumers:
            direct_sources = {
                *inbound.get(str(consumer), {}).values(),
                *dependencies.get(str(consumer), ()),
            }
            if not isinstance(consumer, str) or source not in direct_sources:
                raise ValueError(
                    f"pi.sharing {source} lacks a direct routed consumer {consumer}"
                )

    symbols = {
        instance: bindings.get(instance, f"_intentir_{instance}")
        for instance in {*bindings, *occurrence_fragments}
    }
    rendered: list[ComposerFragment] = []
    consumed_routes = 0
    for fragment in fragments:
        match = _INSTANCE_FRAGMENT.fullmatch(fragment.name)
        if match is None:
            rendered.append(fragment)
            continue
        target = match.group(1)
        expected_slots = inbound.get(target, {})
        expected_dependencies = dependencies.get(target, [])
        seen_slots: dict[int, int] = {}
        seen_dependencies: dict[int, int] = {}
        out_count = 0
        substitutions: list[tuple[str, str]] = []

        def substitute(token: re.Match[str]) -> str:
            nonlocal out_count, consumed_routes
            kind = token.group(1)
            if kind == "out":
                out_count += 1
                symbol = symbols[target]
            elif token.group(2) is not None:
                slot = int(token.group(2))
                source = expected_slots.get(slot)
                if source is None:
                    raise ValueError(
                        f"fragment {fragment.name} references absent route slot {slot}"
                )
                seen_slots[slot] = seen_slots.get(slot, 0) + 1
                symbol = symbols[source]
                consumed_routes += 1
            else:
                dependency = int(token.group(3))
                if dependency >= len(expected_dependencies):
                    raise ValueError(
                        f"fragment {fragment.name} references absent dependency "
                        f"route {dependency}"
                    )
                source = expected_dependencies[dependency]
                seen_dependencies[dependency] = (
                    seen_dependencies.get(dependency, 0) + 1
                )
                symbol = symbols[source]
                consumed_routes += 1
            marker = f"_intentir_placeholder_{target}_{len(substitutions)}"
            if marker in fragment.text:
                raise ValueError("composer placeholder marker collides with source")
            substitutions.append((marker, symbol))
            return marker

        text = _PLACEHOLDER.sub(substitute, fragment.text)
        try:
            name_tokens = [
                token.string
                for token in tokenize.generate_tokens(io.StringIO(text).readline)
                if token.type == tokenize.NAME
            ]
        except (IndentationError, tokenize.TokenError) as exc:
            raise ValueError(
                f"fragment {fragment.name} is not tokenizable Python syntax: {exc}"
            ) from exc
        hidden = [
            marker for marker, _symbol in substitutions if name_tokens.count(marker) != 1
        ]
        if hidden:
            raise ValueError(
                f"fragment {fragment.name} hides route placeholders outside Python "
                f"name tokens: {hidden}"
            )
        for marker, symbol in substitutions:
            text = text.replace(marker, symbol)
        missing_slots = sorted(set(expected_slots) - set(seen_slots))
        repeated_slots = sorted(
            slot for slot, count in seen_slots.items() if count != 1
        )
        missing_dependencies = sorted(
            set(range(len(expected_dependencies))) - set(seen_dependencies)
        )
        repeated_dependencies = sorted(
            dependency
            for dependency, count in seen_dependencies.items()
            if count != 1
        )
        if (
            missing_slots
            or repeated_slots
            or missing_dependencies
            or repeated_dependencies
        ):
            raise ValueError(
                f"fragment {fragment.name} must consume every routed slot/dependency "
                "exactly once; "
                f"missing_slots={missing_slots}, repeated_slots={repeated_slots}, "
                f"missing_deps={missing_dependencies}, "
                f"repeated_deps={repeated_dependencies}"
            )
        if target in outbound and out_count != 1:
            raise ValueError(
                f"route source fragment {fragment.name} must define {{{{out}}}} once"
            )
        if target not in outbound and out_count > 1:
            raise ValueError(
                f"terminal fragment {fragment.name} defines {{{{out}}}} more than once"
            )
        rendered.append(
            ComposerFragment(fragment.name, text, len(text.encode("utf-8")))
        )

    if consumed_routes != route_count:
        raise ValueError(
            "route composer did not consume the exact pi route occurrence count"
        )
    source = "".join(item.text for item in rendered)
    return RouteComposerResult(
        source=source,
        fragments=tuple(rendered),
        bound_instances=tuple(sorted(bindings)),
        occurrence_instances=tuple(sorted(occurrence_fragments)),
        consumed_routes=consumed_routes,
        total_routes=route_count,
    )


__all__ = [
    "ComposerFragment",
    "ROUTE_COMPOSER_PROTOCOL",
    "ROUTE_COMPOSER_PROTOCOL_V1",
    "RouteComposerResult",
    "render_route_composer",
]
