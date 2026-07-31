"""Deterministic Triton Composer for checked J-only Target-Normal programs.

This module is intentionally a compiler component, not a Replay model helper.
It receives a checked ``F_f``, exact pi constraints, and MaskC slot declarations
after the semantic Agent episode has ended.  It never receives Source, Harness
values, Gate evidence, kernel identities, historical Replay, or concrete C
values.  Unsupported physical/semantic combinations fail closed.

The first protocol slice covers dense pointwise programs, scalar/vector program
axes, row-major accesses, constexpr selection and ordinary elementwise ops.  It
is generic over J handles and relation topology; no per-kernel recipes exist.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .target_normal_ir import CheckedTargetNormalProgram


TRITON_TARGET_COMPOSER_PROTOCOL = "intentir.triton-target-composer.v1"


class TritonComposerError(ValueError):
    def __init__(self, rule_code: str, path: str, detail: str) -> None:
        self.rule_code = rule_code
        self.path = path
        self.detail = detail
        super().__init__(f"{rule_code} at {path}: {detail}")


@dataclass(frozen=True)
class TritonComposition:
    source: str
    receipt: Mapping[str, Any]


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TritonComposerError("object_required", path, "must be an object")
    return dict(value)


def _rows(value: Any, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TritonComposerError("array_required", path, "must be an array")
    return [_object(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _ref_values(expression: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    if expression.get("op") == "ref" and isinstance(expression.get("value"), str):
        result.add(str(expression["value"]))
    args = expression.get("args", [])
    if isinstance(args, list):
        for item in args:
            if isinstance(item, Mapping):
                result.update(_ref_values(item))
    return result


def _domain_axes(
    expression: Mapping[str, Any], path: str
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Return all authored iterators and exclusive upper bounds in domain order."""

    terms = expression.get("args", []) if expression.get("op") == "and" else [expression]
    if not isinstance(terms, list):
        raise TritonComposerError("domain_form", path, "malformed domain")
    candidates: dict[str, Mapping[str, Any]] = {}

    def visit(term: Any) -> None:
        if isinstance(term, Mapping) and term.get("op") == "and":
            nested = term.get("args", [])
            if isinstance(nested, list):
                for child in nested:
                    visit(child)
            return
        if not isinstance(term, Mapping) or term.get("op") != "lt":
            return
        args = term.get("args")
        if not isinstance(args, list) or len(args) != 2:
            return
        lhs, rhs = args
        if (
            isinstance(lhs, Mapping)
            and lhs.get("op") == "ref"
            and isinstance(lhs.get("value"), str)
            and re.fullmatch(r"(?:k|v)[0-9]+", str(lhs["value"])) is not None
            and isinstance(rhs, Mapping)
        ):
            candidates[str(lhs["value"])] = dict(rhs)

    for term in terms:
        visit(term)
    ordered = tuple((name, upper) for name, upper in candidates.items())
    if not ordered:
        raise TritonComposerError(
            "domain_form", path, "requires iterator < exclusive-upper terms"
        )
    return ordered


def _expr(
    expression: Mapping[str, Any],
    names: Mapping[str, str],
    path: str,
) -> str:
    op = expression.get("op")
    if op == "ref":
        ref = expression.get("value")
        if not isinstance(ref, str) or ref not in names:
            raise TritonComposerError("unknown_expression_ref", path, str(ref))
        return names[ref]
    if op == "int":
        value = expression.get("value")
        if isinstance(value, bool) or not isinstance(value, int):
            raise TritonComposerError("integer_literal", path, repr(value))
        return str(value)
    if op == "bool":
        value = expression.get("value")
        if not isinstance(value, bool):
            raise TritonComposerError("boolean_literal", path, repr(value))
        return "True" if value else "False"
    args = expression.get("args")
    if not isinstance(args, list):
        raise TritonComposerError("expression_args", path, str(op))
    rendered = [
        _expr(_object(item, f"{path}.args[{index}]"), names, f"{path}.args[{index}]")
        for index, item in enumerate(args)
    ]
    binary = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "floordiv": "//",
        "mod": "%",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
        "eq": "==",
        "ne": "!=",
    }
    if op in binary and len(rendered) == 2:
        return f"({rendered[0]} {binary[str(op)]} {rendered[1]})"
    if op in {"and", "or"} and len(rendered) >= 2:
        token = " & " if op == "and" else " | "
        return "(" + token.join(f"({item})" for item in rendered) + ")"
    if op == "not" and len(rendered) == 1:
        return f"(~({rendered[0]}))"
    if op == "neg" and len(rendered) == 1:
        return f"(-({rendered[0]}))"
    if op == "pos" and len(rendered) == 1:
        return f"(+({rendered[0]}))"
    if op == "call" and len(rendered) == 2:
        fn = expression.get("value")
        if fn == "ceil_div":
            return f"tl.cdiv({rendered[0]}, {rendered[1]})"
        if fn in {"min", "max"}:
            return f"tl.{fn}({rendered[0]}, {rendered[1]})"
    raise TritonComposerError("unsupported_expression", path, str(op))


def _dtype(dtype: str) -> str:
    normalized = dtype.removeprefix("torch.").casefold()
    mapping = {
        "bool": "tl.int1",
        "i1": "tl.int1",
        "i8": "tl.int8",
        "i16": "tl.int16",
        "i32": "tl.int32",
        "i64": "tl.int64",
        "f16": "tl.float16",
        "bf16": "tl.bfloat16",
        "f32": "tl.float32",
        "f64": "tl.float64",
    }
    if normalized not in mapping:
        raise TritonComposerError("unsupported_dtype", "S_J.value_types", dtype)
    return mapping[normalized]


def _literal(value: Any) -> str:
    if value is True:
        return "True"
    if value is False:
        return "False"
    if value is None:
        return "None"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        lowered = value.casefold()
        if lowered in {"-inf", "-infinity"}:
            return 'float("-inf")'
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return 'float("inf")'
    raise TritonComposerError("unsupported_literal", "S_J.nodes[].attrs.value", repr(value))


def _program_axes(
    pi: Mapping[str, Any],
    domain_ids: Sequence[str],
) -> tuple[tuple[str, int], ...]:
    explicit = pi.get("program_axes", [])
    if isinstance(explicit, list) and explicit:
        rows: list[tuple[str, int]] = []
        for index, raw in enumerate(explicit):
            item = _object(raw, f"pi.program_axes[{index}]")
            domain = item.get("domain_ref")
            axis = item.get("axis")
            if domain not in domain_ids or isinstance(axis, bool) or not isinstance(axis, int):
                raise TritonComposerError(
                    "program_axis", f"pi.program_axes[{index}]", "invalid domain/axis"
                )
            rows.append((str(domain), axis))
        if len({axis for _domain, axis in rows}) != len(rows):
            raise TritonComposerError("program_axis", "pi.program_axes", "duplicate axis")
        return tuple(rows)

    # Legacy artifacts predate the explicit relation.  Their authored pi still
    # fixes the same information by the first program-region realization of
    # each logical domain.  This is a lossless projection of exact pi, not a
    # guess from tensor shapes or the Harness grid.
    regions = {
        str(item.get("id")): str(item.get("placement"))
        for item in _rows(pi.get("regions", []), "pi.regions")
        if isinstance(item.get("id"), str)
    }
    ordered: list[str] = []
    for item in _rows(pi.get("instances", []), "pi.instances"):
        semantic_ref = item.get("semantic_ref")
        if (
            item.get("mechanism") == "realize"
            and semantic_ref in domain_ids
            and regions.get(str(item.get("region"))) == "program"
            and semantic_ref not in ordered
        ):
            ordered.append(str(semantic_ref))
    if set(ordered) != set(domain_ids):
        raise TritonComposerError(
            "program_axis",
            "pi.instances",
            "legacy pi does not uniquely realize every J domain in program region",
        )
    return tuple((domain, axis) for axis, domain in enumerate(ordered))


class _Emitter:
    def __init__(
        self,
        program: CheckedTargetNormalProgram,
        pi: Mapping[str, Any],
        masked_c: Mapping[str, Any],
    ) -> None:
        self.program = program
        self.semantic = _object(program.semantic_program, "F_f.S_J")
        self.pi = _object(pi, "pi")
        self.masked_c = _object(masked_c, "MaskC")
        self.lines: list[str] = []
        self.value_names: dict[str, str] = {}
        self.coordinates: dict[str, str] = {}
        self.dimension_names: dict[str, str] = {}
        self.access_values: dict[tuple[str, int], str] = {}
        self.access_pointers: dict[str, tuple[str, str | None, Any]] = {}
        self.tensor_loads: dict[str, list[str]] = {}
        self.tensor_load_cursor: dict[str, int] = {}
        self.tensor_last_load: dict[str, str] = {}
        self.emitted_nodes: set[str] = set()
        self.line_origins: dict[str, list[int]] = {}

        function = _object(self.semantic.get("function"), "S_J.function")
        self.inputs = {
            str(item["name"]): item
            for item in _rows(function.get("inputs", []), "S_J.function.inputs")
        }
        self.outputs = {
            str(item["name"]): item
            for item in _rows(function.get("outputs", []), "S_J.function.outputs")
        }
        self.tensors = {**self.inputs, **self.outputs}
        signature = _object(self.semantic.get("signature"), "S_J.signature")
        self.parameters = _rows(signature.get("parameters", []), "S_J.signature.parameters")
        self.parameters_by_logical = {
            str(item["logical_ref"]): str(item["name"])
            for item in self.parameters
            if isinstance(item.get("logical_ref"), str)
        }
        for logical, parameter in self.parameters_by_logical.items():
            if logical.startswith("x"):
                self.dimension_names[logical] = parameter
            if logical in self.inputs and self.inputs[logical].get("shape") == []:
                self.value_names[logical] = parameter

        self.nodes = _rows(self.semantic.get("nodes", []), "S_J.nodes")
        self.nodes_by_id = {str(item["id"]): item for item in self.nodes}
        self.value_shapes = {
            str(item.get("value")): item.get("shape", [])
            for item in _rows(self.semantic.get("value_types", []), "S_J.value_types")
            if isinstance(item.get("value"), str)
        }
        self.producer: dict[str, str] = {}
        for node in self.nodes:
            result = node.get("result")
            values = [result] if isinstance(result, str) else result if isinstance(result, list) else []
            for value in values:
                if isinstance(value, str):
                    self.producer[value] = str(node["id"])
        boundary = _object(self.semantic.get("boundary"), "S_J.boundary")
        self.accesses = _rows(boundary.get("accesses", []), "S_J.boundary.accesses")
        self.accesses_by_id = {str(item["access_id"]): item for item in self.accesses}
        self.broadcasts: dict[tuple[str, str], tuple[int, ...]] = {}
        for index, item in enumerate(
            _rows(boundary.get("broadcasts", []), "S_J.boundary.broadcasts")
        ):
            node_ref = item.get("node_ref")
            operand = item.get("operand")
            axes = item.get("axes")
            if (
                isinstance(node_ref, str)
                and isinstance(operand, str)
                and isinstance(axes, list)
                and all(isinstance(axis, int) and not isinstance(axis, bool) for axis in axes)
            ):
                self.broadcasts[(node_ref, operand)] = tuple(axes)
        canonical = _object(self.semantic.get("canonical_boundary"), "S_J.canonical_boundary")
        self.canonical_domains = _object(canonical.get("domains"), "S_J.canonical_boundary.domains")
        self.canonical_accesses = {
            str(item["access_id"]): item
            for item in _rows(canonical.get("accesses", []), "S_J.canonical_boundary.accesses")
        }
        self.regions = {
            str(item["id"]): item
            for item in _rows(self.pi.get("regions", []), "pi.regions")
        }
        self.instances = _rows(self.pi.get("instances", []), "pi.instances")
        self.instances_by_id = {str(item["id"]): item for item in self.instances}
        self.access_transfer_counts: dict[str, int] = {}
        for item in self.instances:
            ref = item.get("semantic_ref")
            if item.get("mechanism") == "transfer_direct" and isinstance(ref, str):
                self.access_transfer_counts[ref] = self.access_transfer_counts.get(ref, 0) + 1
        self.routes = _rows(self.pi.get("routes", []), "pi.routes")
        self.domain_iterators = {
            domain_id: tuple(iterator for iterator, _upper in _domain_axes(
                _object(expression, f"S_J.canonical_boundary.domains.{domain_id}"),
                f"S_J.canonical_boundary.domains.{domain_id}",
            ))
            for domain_id, expression in self.canonical_domains.items()
        }
        self.domain_slot_coordinates: dict[tuple[str, int], str] = {}
        for domain_id, iterators in self.domain_iterators.items():
            for slot, iterator in enumerate(iterators):
                self.domain_slot_coordinates[(domain_id, slot)] = iterator
        self.semantic_regions: dict[str, set[str]] = {}
        for item in self.instances:
            ref = item.get("semantic_ref")
            region = item.get("region")
            if isinstance(ref, str) and isinstance(region, str):
                self.semantic_regions.setdefault(ref, set()).add(region)
        self.branch_regions = {
            region_id
            for region_id, item in self.regions.items()
            if item.get("placement") == "branch"
        }

    def add(self, text: str = "", *, origin: str | None = None, indent: int = 1) -> None:
        self.lines.append("    " * indent + text if text else "")
        if origin is not None:
            self.line_origins.setdefault(origin, []).append(len(self.lines))

    def _slot_rows(self) -> list[dict[str, Any]]:
        return _rows(self.masked_c.get("slots", []), "MaskC.slots")

    def _pi_tile_attachment(
        self,
        slot_name: str,
        domains: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        """Project a legacy tile-slot attachment from exact pi topology.

        Older C artifacts did not carry ``subject_ref``.  Their pi graph still
        authors the attachment: the slot realization is routed into an
        address-base/lane instance whose semantic reference is the tiled J
        domain.  This projection consumes only that explicit graph; it never
        consults tensor shapes, the Harness grid, C values, or kernel identity.
        Ambiguous or absent topology remains unresolved and therefore fails
        closed in ``_coordinate_plan``.
        """

        slot_instances = {
            str(item["id"])
            for item in self.instances
            if item.get("mechanism") == "realize"
            and item.get("semantic_ref") == slot_name
            and isinstance(item.get("id"), str)
        }
        if not slot_instances:
            return None
        candidates: set[tuple[str, str]] = set()
        for route in self.routes:
            if route.get("source") not in slot_instances:
                continue
            target = self.instances_by_id.get(str(route.get("target")))
            if target is None or target.get("mechanism") not in {
                "realize",
                "address_base",
                "address_lane",
            }:
                continue
            semantic_ref = target.get("semantic_ref")
            if isinstance(semantic_ref, str) and semantic_ref in domains:
                form = (
                    "local_vector"
                    if target.get("mechanism") == "realize"
                    else "blocked_program"
                )
                candidates.add((semantic_ref, form))
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _coordinate_plan(self) -> tuple[dict[str, str], list[dict[str, Any]]]:
        domains: dict[str, tuple[tuple[str, Mapping[str, Any]], ...]] = {}
        for domain_id, expression in self.canonical_domains.items():
            domains[domain_id] = _domain_axes(
                _object(expression, f"S_J.canonical_boundary.domains.{domain_id}"),
                f"S_J.canonical_boundary.domains.{domain_id}",
            )
        slots = self._slot_rows()
        tile_slots = [
            item
            for item in slots
            if item.get("kind") == "constexpr" and "tile" in str(item.get("role", ""))
        ]
        if len(tile_slots) > 1:
            raise TritonComposerError(
                "pointwise_tile",
                "MaskC.slots",
                "v1 pointwise Composer supports at most one tile slot",
            )
        tile_slot = tile_slots[0] if tile_slots else None
        tiled_domain: str | None = None
        tile_form: str | None = None
        if tile_slot is not None:
            subject = tile_slot.get("subject_ref")
            if isinstance(subject, str) and subject in domains:
                tiled_domain = subject
            else:
                attachment = self._pi_tile_attachment(str(tile_slot["name"]), domains)
                if attachment is not None:
                    tiled_domain, tile_form = attachment
            if tiled_domain is None:
                symbolic = [
                    domain_id
                    for domain_id, domain_axes in domains.items()
                    if len(domain_axes) == 1
                    and any(
                        ref.startswith("x")
                        for ref in _ref_values(domain_axes[0][1])
                    )
                ]
                if len(symbolic) != 1:
                    raise TritonComposerError(
                        "pointwise_tile",
                        "MaskC.slots",
                        "unattached tile slot requires one symbolic-extent domain",
                    )
                tiled_domain = symbolic[0]

        axes = _program_axes(self.pi, tuple(domains))
        if not self.pi.get("program_axes") and tile_form == "local_vector":
            axes = tuple(item for item in axes if item[0] != tiled_domain)

        receipt: list[dict[str, Any]] = []
        assigned_domains: set[str] = set()
        for domain_id, axis in sorted(axes, key=lambda item: item[1]):
            domain_axes = domains[domain_id]
            assigned_domains.add(domain_id)
            pid = f"tn_pid_{axis}"
            widened = f"tn_pid_{axis}_i64"
            self.add(f"{pid} = tl.program_id({axis})", origin=domain_id)
            self.add(f"{widened} = {pid}.to(tl.int64)", origin=domain_id)
            if domain_id == tiled_domain:
                assert tile_slot is not None
                if len(domain_axes) != 1:
                    raise TritonComposerError(
                        "pointwise_tile",
                        f"S_J.canonical_boundary.domains.{domain_id}",
                        "blocked tile domain must have one iterator",
                    )
                iterator, _upper = domain_axes[0]
                slot = str(tile_slot["name"])
                base = f"tn_{iterator}_base"
                lane = f"tn_{iterator}_lane"
                coordinate = f"tn_{iterator}"
                self.add(f"{base} = {widened} * {slot}", origin=domain_id)
                self.add(f"{lane} = tl.arange(0, {slot})", origin=domain_id)
                self.add(f"{coordinate} = {base} + {lane}", origin=domain_id)
                self.coordinates[iterator] = coordinate
                receipt.append(
                    {
                        "domain_ref": domain_id,
                        "program_axis": axis,
                        "iterator": iterator,
                        "form": "blocked_vector",
                        "slot": slot,
                    }
                )
            elif len(domain_axes) > 1:
                packed = widened
                decoded: list[dict[str, Any]] = []
                for reverse_index, (iterator, upper) in enumerate(
                    reversed(domain_axes)
                ):
                    coordinate = f"tn_{iterator}"
                    is_last = reverse_index == len(domain_axes) - 1
                    if is_last:
                        self.add(f"{coordinate} = {packed}", origin=domain_id)
                    else:
                        extent = _expr(
                            upper,
                            self.dimension_names,
                            f"S_J.canonical_boundary.domains.{domain_id}.{iterator}.upper",
                        )
                        self.add(
                            f"{coordinate} = {packed} % {extent}", origin=domain_id
                        )
                        next_packed = f"tn_{domain_id}_packed_{reverse_index}"
                        self.add(
                            f"{next_packed} = {packed} // {extent}",
                            origin=domain_id,
                        )
                        packed = next_packed
                    self.coordinates[iterator] = coordinate
                    decoded.append(
                        {
                            "iterator": iterator,
                            **(
                                {"extent": extent}
                                if not is_last
                                else {"extent": "quotient_remainder"}
                            ),
                        }
                    )
                receipt.append(
                    {
                        "domain_ref": domain_id,
                        "program_axis": axis,
                        "form": "row_major_mixed_radix",
                        "decoded": list(reversed(decoded)),
                    }
                )
            else:
                iterator, _upper = domain_axes[0]
                coordinate = f"tn_{iterator}"
                self.add(f"{coordinate} = {widened}", origin=domain_id)
                self.coordinates[iterator] = coordinate
                receipt.append(
                    {
                        "domain_ref": domain_id,
                        "program_axis": axis,
                        "iterator": iterator,
                        "form": "scalar_program_coordinate",
                    }
                )

        for domain_id, domain_axes in domains.items():
            if domain_id in assigned_domains:
                continue
            if len(domain_axes) != 1:
                raise TritonComposerError(
                    "local_domain",
                    f"S_J.canonical_boundary.domains.{domain_id}",
                    "unassigned local domain must have one iterator",
                )
            iterator, upper = domain_axes[0]
            local_tile = domain_id == tiled_domain and tile_slot is not None
            extent = (
                str(tile_slot["name"])
                if local_tile
                else _expr(
                    upper,
                    self.dimension_names,
                    f"S_J.canonical_boundary.domains.{domain_id}.{iterator}.upper",
                )
            )
            coordinate = f"tn_{iterator}"
            self.add(
                f"{coordinate} = tl.arange(0, {extent}).to(tl.int64)",
                origin=domain_id,
            )
            self.coordinates[iterator] = coordinate
            receipt.append(
                {
                    "domain_ref": domain_id,
                    "program_axis": None,
                    "iterator": iterator,
                    "form": "local_tiled_vector" if local_tile else "local_vector",
                    "extent": extent,
                }
            )
        self.add()
        return self.coordinates, receipt

    def _names(self) -> dict[str, str]:
        return {**self.dimension_names, **self.coordinates, **self.value_names}

    def _linear_offset(self, access_id: str, tensor: str) -> tuple[list[str], str]:
        canonical = self.canonical_accesses[access_id]
        axes = canonical.get("axes")
        if not isinstance(axes, list):
            raise TritonComposerError("access_axes", f"access.{access_id}", "missing axes")
        shape = self.tensors[tensor].get("shape")
        if not isinstance(shape, list) or len(shape) != len(axes):
            raise TritonComposerError(
                "access_rank", f"access.{access_id}", "index rank differs from tensor rank"
            )
        names = self._names()
        coordinates = [
            _expr(_object(item, f"access.{access_id}.axes[{index}]"), names, f"access.{access_id}.axes[{index}]")
            for index, item in enumerate(axes)
        ]
        if not coordinates:
            return [], "0"
        terms: list[str] = []
        for index, coordinate in enumerate(coordinates):
            suffix = shape[index + 1 :]
            factors: list[str] = []
            for dimension in suffix:
                if isinstance(dimension, int) and not isinstance(dimension, bool):
                    factors.append(str(dimension))
                elif isinstance(dimension, str) and dimension in self.dimension_names:
                    factors.append(self.dimension_names[dimension])
                else:
                    raise TritonComposerError(
                        "shape_extent", f"tensor.{tensor}.shape[{index}]", repr(dimension)
                    )
            term = coordinate
            if factors:
                term = f"({coordinate} * {' * '.join(factors)})"
            terms.append(term)
        return terms, " + ".join(terms)

    def _pi_offset_terms(self, access_id: str) -> list[str] | None:
        """Solve one current-schema access offset from the exact pi route DAG.

        A current access realization receives logical domains at ordered slots
        and optional opaque stride carriers at those same slots.  The highest
        routed slot without a carrier is the unit-stride lane; an absent carrier
        on any earlier slot denotes no address contribution.  Output accesses
        may share an input access realization while supplying their own base
        pointer.  Every choice below is therefore fixed by route endpoints and
        slots rather than tensor-shape guessing.
        """

        realization = next(
            (
                item
                for item in self.instances
                if item.get("semantic_ref") == access_id
                and item.get("mechanism") == "realize"
            ),
            None,
        )
        if realization is None:
            address_ids = {
                str(item["id"])
                for item in self.instances
                if item.get("semantic_ref") == access_id
                and item.get("mechanism") in {"address_base", "address_lane"}
            }
            source_accesses: list[Mapping[str, Any]] = []
            for route in self.routes:
                if route.get("target") not in address_ids:
                    continue
                source = self.instances_by_id.get(str(route.get("source")))
                if (
                    source is not None
                    and source.get("mechanism") == "realize"
                    and isinstance(source.get("semantic_ref"), str)
                    and str(source["semantic_ref"]).startswith("a")
                ):
                    source_accesses.append(source)
            if len(source_accesses) == 1:
                realization = source_accesses[0]
        if realization is None:
            return None

        incoming = [
            route
            for route in self.routes
            if route.get("target") == realization.get("id")
        ]
        coordinate_by_slot: dict[int, str] = {}
        stride_by_slot: dict[int, str] = {}
        for index, route in enumerate(incoming):
            source = self.instances_by_id.get(str(route.get("source")))
            if source is None:
                raise TritonComposerError(
                    "pi_route", f"pi.routes[{index}]", "unknown source instance"
                )
            slots = route.get("slots")
            if not isinstance(slots, list) or any(
                isinstance(slot, bool) or not isinstance(slot, int) or slot < 0
                for slot in slots
            ):
                raise TritonComposerError(
                    "pi_route", f"pi.routes[{index}].slots", "invalid slots"
                )
            semantic_ref = source.get("semantic_ref")
            if isinstance(semantic_ref, str) and semantic_ref in self.domain_iterators:
                iterators = self.domain_iterators[semantic_ref]
                if len(iterators) != len(slots):
                    raise TritonComposerError(
                        "pi_route",
                        f"pi.routes[{index}]",
                        "domain route slots do not match domain iterator arity",
                    )
                for slot, iterator in zip(slots, iterators, strict=True):
                    coordinate_by_slot[slot] = iterator
                continue
            if (
                isinstance(semantic_ref, str)
                and re.fullmatch(r"p[0-9]+", semantic_ref)
                and slots
            ):
                for slot in slots:
                    if slot in stride_by_slot and stride_by_slot[slot] != semantic_ref:
                        raise TritonComposerError(
                            "pi_route",
                            f"pi.routes[{index}]",
                            f"multiple stride carriers for slot {slot}",
                        )
                    stride_by_slot[slot] = semantic_ref

        if not coordinate_by_slot:
            return None
        highest = max(coordinate_by_slot)
        terms: list[str] = []
        for slot in sorted(coordinate_by_slot):
            iterator = coordinate_by_slot[slot]
            if iterator not in self.coordinates:
                raise TritonComposerError(
                    "pi_route", f"access.{access_id}", f"unmaterialized iterator {iterator}"
                )
            coordinate = self.coordinates[iterator]
            stride = stride_by_slot.get(slot)
            if stride is not None:
                terms.append(f"({coordinate} * {stride})")
            elif slot == highest:
                terms.append(coordinate)
        return terms

    def _legacy_address_pointer(self, access_id: str, base: str) -> str | None:
        """Lower explicit legacy ``address_base``/``address_lane`` stages.

        These stages are part of exact pi, so preserving them is materially
        different from reconstructing a row-major address from tensor shape.
        The supported closed form pairs ordered single-iterator domains with
        ordered scalar stride carriers, excludes the already-known tensor base,
        and emits one addptr per authored address stage.  More complicated
        legacy arities remain unsupported instead of being guessed.
        """

        stages: dict[str, Mapping[str, Any] | None] = {}
        for mechanism in ("address_base", "address_lane"):
            matches = [
                item
                for item in self.instances
                if item.get("semantic_ref") == access_id
                and item.get("mechanism") == mechanism
            ]
            if len(matches) > 1:
                raise TritonComposerError(
                    "legacy_address",
                    f"pi.instances[{access_id}]",
                    f"multiple {mechanism} instances",
                )
            stages[mechanism] = matches[0] if matches else None
        if not any(stages.values()):
            return None

        pointer = base
        prior_id: str | None = None
        for mechanism in ("address_base", "address_lane"):
            stage = stages[mechanism]
            if stage is None:
                continue
            incoming = sorted(
                (
                    route
                    for route in self.routes
                    if route.get("target") == stage.get("id")
                ),
                key=lambda route: tuple(route.get("slots", [])),
            )
            coordinates: list[str] = []
            carriers: list[str] = []
            consumes_prior = False
            for route in incoming:
                source = self.instances_by_id.get(str(route.get("source")))
                if source is None:
                    raise TritonComposerError(
                        "legacy_address", f"access.{access_id}", "unknown route source"
                    )
                if prior_id is not None and source.get("id") == prior_id:
                    consumes_prior = True
                    continue
                semantic_ref = source.get("semantic_ref")
                if isinstance(semantic_ref, str) and semantic_ref in self.domain_iterators:
                    iterators = self.domain_iterators[semantic_ref]
                    if len(iterators) != 1:
                        raise TritonComposerError(
                            "legacy_address",
                            f"access.{access_id}",
                            "legacy address stage requires single-iterator domains",
                        )
                    iterator = iterators[0]
                    if iterator not in self.coordinates:
                        raise TritonComposerError(
                            "legacy_address",
                            f"access.{access_id}",
                            f"unmaterialized iterator {iterator}",
                        )
                    coordinates.append(self.coordinates[iterator])
                    continue
                if (
                    isinstance(semantic_ref, str)
                    and re.fullmatch(r"p[0-9]+", semantic_ref)
                    and semantic_ref != base
                ):
                    carriers.append(semantic_ref)

            if prior_id is not None and not consumes_prior:
                raise TritonComposerError(
                    "legacy_address",
                    f"access.{access_id}",
                    "address_lane does not consume its address_base",
                )
            if not coordinates:
                raise TritonComposerError(
                    "legacy_address",
                    f"access.{access_id}",
                    f"{mechanism} has no logical coordinate",
                )
            if carriers and len(carriers) != len(coordinates):
                raise TritonComposerError(
                    "legacy_address",
                    f"access.{access_id}",
                    "coordinate/stride arity is ambiguous",
                )
            terms = (
                [
                    f"({coordinate} * {carrier})"
                    for coordinate, carrier in zip(coordinates, carriers, strict=True)
                ]
                if carriers
                else coordinates
            )
            suffix = "base" if mechanism == "address_base" else "lane"
            offset = f"tn_offset_{access_id}_{suffix}"
            next_pointer = f"tn_ptr_{access_id}_{suffix}"
            self.add(f"{offset} = {' + '.join(terms)}", origin=access_id)
            self.add(f"{next_pointer} = {pointer} + {offset}", origin=access_id)
            pointer = next_pointer
            prior_id = str(stage.get("id"))
        return pointer

    def _pointer(self, access_id: str) -> tuple[str, str | None, Any]:
        if access_id in self.access_pointers:
            return self.access_pointers[access_id]
        access = self.accesses_by_id[access_id]
        tensor = str(access["tensor"])
        if tensor not in self.parameters_by_logical:
            raise TritonComposerError(
                "tensor_parameter", f"access.{access_id}", f"no signature parameter for {tensor}"
            )
        base = self.parameters_by_logical[tensor]
        pointer = self._legacy_address_pointer(access_id, base)
        if pointer is None:
            pi_terms = self._pi_offset_terms(access_id)
            terms, _offset = (
                self._linear_offset(access_id, tensor)
                if pi_terms is None
                else (pi_terms, " + ".join(pi_terms))
            )
            nonzero = [term for term in terms if term != "0"]
            pointer = base
            if nonzero:
                offset = f"tn_offset_{access_id}"
                pointer = f"tn_ptr_{access_id}"
                self.add(f"{offset} = {' + '.join(nonzero)}", origin=access_id)
                self.add(f"{pointer} = {base} + {offset}", origin=access_id)
        predicate = self.canonical_accesses[access_id].get("predicate")
        predicate_text = (
            _expr(_object(predicate, f"access.{access_id}.predicate"), self._names(), f"access.{access_id}.predicate")
            if isinstance(predicate, Mapping)
            else None
        )
        result = (pointer, predicate_text, access.get("fill_value"))
        self.access_pointers[access_id] = result
        return result

    def _emit_load_access(
        self, access_id: str, occurrence: int, *, indent: int = 1
    ) -> str:
        key = (access_id, occurrence)
        if key in self.access_values:
            return self.access_values[key]
        access = self.accesses_by_id[access_id]
        tensor = str(access["tensor"])
        start = len(self.lines)
        pointer, predicate, fill = self._pointer(access_id)
        if indent != 1:
            for index in range(start, len(self.lines)):
                line = self.lines[index]
                if line.startswith("    "):
                    self.lines[index] = "    " * indent + line[4:]
        suffix = "" if occurrence == 0 else f"_{occurrence}"
        value = f"tn_load_{access_id}{suffix}"
        arguments = [pointer]
        if predicate is not None:
            arguments.append(f"mask={predicate}")
            arguments.append(f"other={_literal(fill)}")
        if self.access_transfer_counts.get(access_id, 1) > 1:
            arguments.append(
                'eviction_policy="evict_first"'
                if occurrence == 0
                else 'eviction_policy="evict_last"'
            )
        self.add(f"{value} = tl.load({', '.join(arguments)})", origin=access_id, indent=indent)
        self.access_values[key] = value
        self.tensor_loads.setdefault(tensor, []).append(value)
        return value

    def _next_tensor_load(self, tensor: str) -> str | None:
        values = self.tensor_loads.get(tensor, [])
        cursor = self.tensor_load_cursor.get(tensor, 0)
        if cursor < len(values):
            value = values[cursor]
            self.tensor_load_cursor[tensor] = cursor + 1
            self.tensor_last_load[tensor] = value
            return value
        return self.tensor_last_load.get(tensor)

    def _load_for(self, tensor: str, node_ref: str, *, indent: int = 1) -> str:
        queued = self._next_tensor_load(tensor)
        if queued is not None:
            return queued
        matches = [
            access
            for access in self.accesses
            if access.get("kind") == "load"
            and access.get("tensor") == tensor
            and (
                access.get("node_ref") == node_ref
                or access.get("node_ref") is None
            )
        ]
        if len(matches) != 1:
            raise TritonComposerError(
                "load_binding",
                f"node.{node_ref}",
                f"expected one load access for input {tensor}, found {len(matches)}",
            )
        access_id = str(matches[0]["access_id"])
        occurrence = sum(1 for key in self.access_values if key[0] == access_id)
        value = self._emit_load_access(access_id, occurrence, indent=indent)
        self.tensor_load_cursor[tensor] = len(self.tensor_loads.get(tensor, []))
        self.tensor_last_load[tensor] = value
        return value

    def _operand(self, value: str, node_ref: str, *, indent: int = 1) -> str:
        if value in self.inputs:
            parameter = self.parameters_by_logical.get(value)
            if self.inputs[value].get("shape") == [] and parameter is not None:
                self.value_names[value] = parameter
                return parameter
            return self._load_for(value, node_ref, indent=indent)
        if value in self.value_names:
            return self.value_names[value]
        raise TritonComposerError("undefined_value", f"node.{node_ref}", value)

    def _preload_ordered_inputs(self) -> None:
        """Materialize non-branch reads in exact J/pi transfer order.

        Boundary access order is J-owned and pi.order is checked by TypeCheck as
        the corresponding realization order.  Emitting these reads before pure
        arithmetic preserves that authored transfer order even when a later J
        node consumes an earlier-loaded input.  Branch-local reads stay inside
        their pi branch region.
        """

        occurrences = [
            str(item["semantic_ref"])
            for item in self.instances
            if item.get("mechanism") == "transfer_direct"
            and isinstance(item.get("semantic_ref"), str)
            and item.get("semantic_ref") in self.accesses_by_id
            and self.accesses_by_id[str(item["semantic_ref"])].get("kind") == "load"
        ]
        if not occurrences:
            occurrences = [
                access_id
                for access_id in self.program.access_order
                if self.accesses_by_id[access_id].get("kind") == "load"
            ]
        counts: dict[str, int] = {}
        for access_id in occurrences:
            if self.semantic_regions.get(access_id, set()) & self.branch_regions:
                continue
            occurrence = counts.get(access_id, 0)
            counts[access_id] = occurrence + 1
            self._emit_load_access(access_id, occurrence)
        if any(
            self.accesses_by_id[access_id].get("kind") == "load"
            for access_id in self.program.access_order
        ):
            self.add()

    def _emit_through(self, node_ref: str) -> None:
        if node_ref not in self.nodes_by_id:
            raise TritonComposerError("store_node", f"node.{node_ref}", "unknown")
        target = next(
            index for index, item in enumerate(self.nodes) if item.get("id") == node_ref
        )
        branch_nodes = {
            ref
            for ref, regions in self.semantic_regions.items()
            if regions & self.branch_regions and ref in self.nodes_by_id
        }
        for node in self.nodes[: target + 1]:
            current = str(node["id"])
            if current in self.emitted_nodes or current in branch_nodes:
                continue
            if self._branch_select(current):
                continue
            self._emit_node(current)

    def _emit_store_access(self, access_id: str) -> None:
        access = self.accesses_by_id[access_id]
        node_ref = access.get("node_ref")
        if isinstance(node_ref, str) and node_ref not in self.emitted_nodes:
            self._emit_through(node_ref)
        tensor = str(access["tensor"])
        value = self.value_names.get(tensor)
        if value is None and isinstance(node_ref, str):
            node = self.nodes_by_id[node_ref]
            result = node.get("result")
            if isinstance(result, str):
                value = self.value_names.get(result)
        if value is None:
            raise TritonComposerError("store_value", f"access.{access_id}", tensor)
        pointer, predicate, _fill = self._pointer(access_id)
        args = [pointer, value]
        if predicate is not None:
            args.append(f"mask={predicate}")
        self.add(f"tl.store({', '.join(args)})", origin=access_id)

    def _emit_transfer_schedule(self) -> bool:
        transfer_instances = [
            item
            for item in self.instances
            if item.get("mechanism") == "transfer_direct"
            and isinstance(item.get("semantic_ref"), str)
            and item.get("semantic_ref") in self.accesses_by_id
        ]
        if not transfer_instances:
            return False
        transfer_ids = {str(item["id"]) for item in transfer_instances}
        declaration = {str(item["id"]): index for index, item in enumerate(transfer_instances)}
        outgoing: dict[str, set[str]] = {item: set() for item in transfer_ids}
        incoming_count: dict[str, int] = {item: 0 for item in transfer_ids}
        for index, relation in enumerate(
            _rows(self.pi.get("order", []), "pi.order")
        ):
            before = relation.get("before")
            after = relation.get("after")
            if before not in self.instances_by_id or after not in self.instances_by_id:
                raise TritonComposerError(
                    "pi_order", f"pi.order[{index}]", "unknown instance endpoint"
                )
            if before in transfer_ids and after in transfer_ids and after not in outgoing[str(before)]:
                outgoing[str(before)].add(str(after))
                incoming_count[str(after)] += 1
        ready = sorted(
            (item for item, count in incoming_count.items() if count == 0),
            key=declaration.__getitem__,
        )
        ordered_ids: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered_ids.append(current)
            for target in sorted(outgoing[current], key=declaration.__getitem__):
                incoming_count[target] -= 1
                if incoming_count[target] == 0:
                    ready.append(target)
                    ready.sort(key=declaration.__getitem__)
        if len(ordered_ids) != len(transfer_ids):
            raise TritonComposerError("pi_order", "pi.order", "transfer cycle")
        by_id = {str(item["id"]): item for item in transfer_instances}
        events = [str(by_id[item]["semantic_ref"]) for item in ordered_ids]
        counts: dict[str, int] = {}
        emitted_stores: set[str] = set()
        for access_id in events:
            access = self.accesses_by_id[access_id]
            kind = access.get("kind")
            if kind == "load":
                if self.semantic_regions.get(access_id, set()) & self.branch_regions:
                    continue
                occurrence = counts.get(access_id, 0)
                counts[access_id] = occurrence + 1
                self._emit_load_access(access_id, occurrence)
            elif kind == "store":
                if access_id in emitted_stores:
                    raise TritonComposerError(
                        "store_occurrence",
                        f"pi.instances.{access_id}",
                        "pointwise/reduction v1 supports one store occurrence per J access",
                    )
                self._emit_store_access(access_id)
                emitted_stores.add(access_id)
            else:
                raise TritonComposerError(
                    "transfer_kind", f"access.{access_id}", str(kind)
                )
        for access_id in self.program.access_order:
            access = self.accesses_by_id[access_id]
            if access.get("kind") == "store" and access_id not in emitted_stores:
                self._emit_store_access(access_id)
        return True

    def _result_name(self, value: str) -> str:
        return "tn_value_" + re.sub(r"[^A-Za-z0-9_]", "_", value)

    def _emit_node(self, node_ref: str, *, indent: int = 1) -> None:
        if node_ref in self.emitted_nodes:
            return
        node = self.nodes_by_id[node_ref]
        op = str(node.get("op"))
        operands = node.get("operands", [])
        if not isinstance(operands, list):
            raise TritonComposerError("node_operands", f"node.{node_ref}", "must be an array")
        resolved: list[str] = []
        node_type = node.get("type") if isinstance(node.get("type"), Mapping) else {}
        target_shape = node_type.get("shape", []) if isinstance(node_type, Mapping) else []
        for value in operands:
            value_ref = str(value)
            rendered = self._operand(value_ref, node_ref, indent=indent)
            axes = self.broadcasts.get((node_ref, value_ref))
            operand_shape = self.value_shapes.get(value_ref, [])
            if (
                axes
                and isinstance(operand_shape, list)
                and isinstance(target_shape, list)
                and len(operand_shape) > 1
                and len(target_shape) > len(operand_shape)
            ):
                expanded = f"tn_expand_{node_ref}_{len(resolved)}"
                self.add(
                    f"{expanded} = tl.expand_dims({rendered}, 0)",
                    origin=node_ref,
                    indent=indent,
                )
                rendered = expanded
            resolved.append(rendered)
        raw_result = node.get("result")
        if not isinstance(raw_result, str):
            raise TritonComposerError(
                "node_result", f"node.{node_ref}", "pointwise v1 requires one result"
            )
        result = self._result_name(raw_result)
        attrs = node.get("attrs", {})
        attrs = dict(attrs) if isinstance(attrs, Mapping) else {}

        if op == "const":
            value = attrs.get("value")
            dtype = attrs.get("dtype")
            if isinstance(dtype, str):
                literal = "-1e999" if isinstance(value, str) and value.casefold() == "-inf" else _literal(value)
                expression = f"tl.full((), {literal}, {_dtype(dtype)})"
            else:
                expression = _literal(value)
        elif op == "map_binary" and len(resolved) == 2:
            fn = attrs.get("fn")
            binary = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "floordiv": "//",
                "mod": "%",
                "lt": "<",
                "le": "<=",
                "gt": ">",
                "ge": ">=",
                "eq": "==",
                "ne": "!=",
                "and": "&",
                "or": "|",
            }
            if fn in binary:
                expression = f"({resolved[0]} {binary[str(fn)]} {resolved[1]})"
            elif fn in {"max", "min"}:
                expression = f"tl.maximum({resolved[0]}, {resolved[1]})" if fn == "max" else f"tl.minimum({resolved[0]}, {resolved[1]})"
            else:
                raise TritonComposerError("map_binary", f"node.{node_ref}.attrs.fn", str(fn))
        elif op == "map_unary" and len(resolved) == 1:
            fn = attrs.get("fn")
            unary = {
                "neg": f"(-{resolved[0]})",
                "abs": f"tl.abs({resolved[0]})",
                "exp": f"tl.exp({resolved[0]})",
                "log": f"tl.log({resolved[0]})",
                "sqrt": f"tl.sqrt({resolved[0]})",
                "rsqrt": f"tl.rsqrt({resolved[0]})",
                "tanh": f"libdevice.tanh({resolved[0]})",
                "sigmoid": f"tl.sigmoid({resolved[0]})",
            }
            if fn not in unary:
                raise TritonComposerError("map_unary", f"node.{node_ref}.attrs.fn", str(fn))
            expression = unary[str(fn)]
        elif op == "cast" and len(resolved) == 1:
            to_dtype = attrs.get("to_dtype")
            if not isinstance(to_dtype, str):
                raise TritonComposerError("cast", f"node.{node_ref}.attrs", "missing to_dtype")
            expression = f"{resolved[0]}.to({_dtype(to_dtype)})"
        elif op == "reduce" and len(resolved) == 1:
            fn = attrs.get("fn")
            axes = attrs.get("axes")
            if not isinstance(axes, list) or len(axes) != 1 or not isinstance(axes[0], int):
                raise TritonComposerError(
                    "reduce", f"node.{node_ref}.attrs.axes", "requires one axis"
                )
            operand_ref = str(operands[0])
            producer_ref = self.producer.get(operand_ref)
            linearized = False
            if producer_ref is not None:
                producer_instances = {
                    str(item["id"])
                    for item in self.instances
                    if item.get("semantic_ref") == producer_ref
                }
                linearized = any(
                    item.get("kind") == "reduction_linearization"
                    and item.get("instance") in producer_instances
                    for item in _rows(self.pi.get("refinements", []), "pi.refinements")
                )
            reduction_input = resolved[0]
            if linearized:
                operand_shape = self.value_shapes.get(operand_ref, [])
                axis = int(axes[0])
                if not isinstance(operand_shape, list) or not 0 <= axis < len(operand_shape):
                    raise TritonComposerError(
                        "reduce", f"node.{node_ref}", "axis is outside operand shape"
                    )
                extent = operand_shape[axis]
                if isinstance(extent, str) and extent in self.dimension_names:
                    extent_text = self.dimension_names[extent]
                elif isinstance(extent, int) and not isinstance(extent, bool):
                    extent_text = str(extent)
                else:
                    raise TritonComposerError(
                        "reduce", f"node.{node_ref}", "unbound reduction extent"
                    )
                reshaped = f"tn_reduce_input_{node_ref}"
                self.add(
                    f"{reshaped} = tl.reshape({reduction_input}, ({extent_text},), can_reorder=True)",
                    origin=node_ref,
                    indent=indent,
                )
                reduction_input = reshaped
            if fn == "sum":
                expression = f"tl.sum({reduction_input}, axis=0)"
            elif fn == "max":
                expression = f"tl.max({reduction_input}, axis=0)"
            elif fn == "min":
                expression = f"tl.min({reduction_input}, axis=0)"
            else:
                raise TritonComposerError(
                    "reduce", f"node.{node_ref}.attrs.fn", str(fn)
                )
        elif op == "select" and len(resolved) == 3:
            expression = f"tl.where({resolved[0]}, {resolved[1]}, {resolved[2]})"
        else:
            raise TritonComposerError(
                "unsupported_semantic_op", f"node.{node_ref}", op
            )
        self.add(f"{result} = {expression}", origin=node_ref, indent=indent)
        self.value_names[raw_result] = result
        self.emitted_nodes.add(node_ref)

    def _branch_select(self, select_ref: str) -> bool:
        node = self.nodes_by_id[select_ref]
        if node.get("op") != "select":
            return False
        operands = node.get("operands")
        if not isinstance(operands, list) or len(operands) != 3:
            return False
        condition, positive, negative = (str(item) for item in operands)
        positive_node = self.producer.get(positive)
        negative_node = self.producer.get(negative)
        positive_branch = bool(
            positive_node
            and self.semantic_regions.get(positive_node, set()) & self.branch_regions
        )
        negative_branch = bool(
            negative_node
            and self.semantic_regions.get(negative_node, set()) & self.branch_regions
        )
        if positive_branch == negative_branch:
            return False
        condition_value = self._operand(condition, select_ref)
        branch_node = positive_node if positive_branch else negative_node
        assert branch_node is not None
        self.add(f"if {condition_value}:", origin=select_ref)
        self._emit_node(branch_node, indent=2)
        branch_value = self.value_names[positive if positive_branch else negative]
        outside_value = self._operand(negative if positive_branch else positive, select_ref)
        result_value = node.get("result")
        if not isinstance(result_value, str):
            raise TritonComposerError("select_result", f"node.{select_ref}", "missing result")
        result = self._result_name(result_value)
        if positive_branch:
            self.add(
                f"{result} = {branch_value} if {condition_value} else {outside_value}",
                origin=select_ref,
            )
        else:
            self.add(
                f"{result} = {outside_value} if {condition_value} else {branch_value}",
                origin=select_ref,
            )
        self.value_names[result_value] = result
        self.emitted_nodes.add(select_ref)
        return True

    def _stores(self) -> None:
        for access_id in self.program.access_order:
            access = self.accesses_by_id[access_id]
            if access.get("kind") != "store":
                continue
            self._emit_store_access(access_id)

    def emit(self) -> TritonComposition:
        if self.program.frontend != "triton":
            raise TritonComposerError("frontend", "F_f.frontend", self.program.frontend)
        unsupported_pi = {
            name: self.pi.get(name)
            for name in (
                "iterations",
                "loop_carriers",
                "physical_iterations",
                "physical_loop_carriers",
                "terminal_controls",
                "program_linearizations",
            )
            if self.pi.get(name)
        }
        if unsupported_pi:
            raise TritonComposerError(
                "unsupported_pi_profile",
                "pi",
                "pointwise v1 does not yet consume " + ", ".join(sorted(unsupported_pi)),
            )
        slot_rows = self._slot_rows()
        compiler_slots = [item for item in slot_rows if item.get("kind") == "compiler"]
        if compiler_slots:
            raise TritonComposerError(
                "compiler_slot", "MaskC.slots", "pointwise v1 does not consume compiler slots"
            )

        parameters: list[str] = []
        for item in self.parameters:
            name = str(item["name"])
            parameters.append(
                name + (": tl.constexpr" if item.get("kind") == "constexpr" else "")
            )
        for item in slot_rows:
            if item.get("kind") == "constexpr":
                parameters.append(str(item["name"]) + ": tl.constexpr")

        header = [
            "import triton",
            "import triton.language as tl",
            "from triton.language.extra.cuda import libdevice",
            "",
            "",
            "@triton.jit",
            f"def {self.program.entry}(",
        ]
        header.extend(f"    {parameter}," for parameter in parameters)
        header.append("):")
        self.lines = header
        _coordinates, coordinate_receipt = self._coordinate_plan()
        scheduled = self._emit_transfer_schedule()
        for node in self.nodes:
            node_ref = str(node["id"])
            if node_ref in self.emitted_nodes:
                continue
            if self.semantic_regions.get(node_ref, set()) & self.branch_regions:
                continue
            if self._branch_select(node_ref):
                continue
            self._emit_node(node_ref)
        if not scheduled:
            self.add()
            self._stores()
        source = "\n".join(self.lines).rstrip() + "\n"

        used_pi_fields = ["regions", "instances"]
        if self.pi.get("program_axes"):
            used_pi_fields.append("program_axes")
        if self.pi.get("routes"):
            used_pi_fields.append("routes")
        if self.pi.get("order"):
            used_pi_fields.append("order")
        if self.pi.get("sharing"):
            used_pi_fields.append("sharing")
        if self.pi.get("refinements"):
            used_pi_fields.append("refinements")
        receipt = {
            "protocol": TRITON_TARGET_COMPOSER_PROTOCOL,
            "status": "pass",
            "frontend": "triton",
            "profile": "generic_map_reduce_access_v1",
            "target_normal": self.program.to_evidence(),
            "program_coordinates": coordinate_receipt,
            "consumed_pi_fields": used_pi_fields,
            "consumed_c_slots": [str(item["name"]) for item in slot_rows],
            "source_map": copy.deepcopy(self.line_origins),
            "source_bytes": len(source.encode("utf-8")),
            "authority": (
                "deterministic F_f/pi/MaskC composition only; no Source, Harness "
                "valuation, Gate evidence, historical Replay, or concrete C value"
            ),
        }
        return TritonComposition(source=source, receipt=receipt)


def compose_triton_target_normal(
    program: CheckedTargetNormalProgram,
    pi: Mapping[str, Any],
    masked_c: Mapping[str, Any],
) -> TritonComposition:
    if program.semantic_program.get("status") == "masked":
        semantic = _object(program.semantic_program, "F_f.S_J")
        signature = _object(semantic.get("signature"), "S_J.signature")
        parameters = _rows(signature.get("parameters", []), "S_J.signature.parameters")
        slots = _rows(_object(masked_c, "MaskC").get("slots", []), "MaskC.slots")
        rendered = [
            str(item["name"])
            + (": tl.constexpr" if item.get("kind") == "constexpr" else "")
            for item in parameters
        ]
        rendered.extend(
            str(item["name"]) + ": tl.constexpr"
            for item in slots
            if item.get("kind") == "constexpr"
        )
        lines = [
            "import triton",
            "import triton.language as tl",
            "",
            "",
            "@triton.jit",
            f"def {program.entry}(",
            *(f"    {item}," for item in rendered),
            "):",
            "    tn_unknown_pid = tl.program_id(0)",
            "    tn_unknown_value = tn_unknown_pid + 1",
        ]
        source = "\n".join(lines) + "\n"
        return TritonComposition(
            source=source,
            receipt={
                "protocol": TRITON_TARGET_COMPOSER_PROTOCOL,
                "status": "pass",
                "frontend": "triton",
                "profile": "masked_semantic_bottom_v1",
                "target_normal": program.to_evidence(),
                "consumed_pi_fields": [],
                "consumed_c_slots": [str(item["name"]) for item in slots],
                "source_map": {},
                "source_bytes": len(source.encode("utf-8")),
                "authority": (
                    "MaskJ negative-control bottom: no hidden semantic or physical "
                    "fact is reconstructed; the result is compileable but intentionally "
                    "cannot implement the masked program"
                ),
            },
        )
    return _Emitter(program, pi, masked_c).emit()


__all__ = [
    "TRITON_TARGET_COMPOSER_PROTOCOL",
    "TritonComposerError",
    "TritonComposition",
    "compose_triton_target_normal",
]
