"""Deterministic semantic fingerprints for Triton 3.6 TTIR.

Gate1 compares two compilations produced by the *same* pinned Triton frontend.
It therefore does not need a general MLIR parser, but it must avoid comparing
unstable presentation details such as SSA names, debug locations, or the text
order of independent pure operations.  This module implements that deliberately
small boundary:

* parse the public ``tt.func`` runtime ABI and infer pointer access modes;
* build an SSA producer/consumer graph, including typed ``tt.reduce``/
  ``tt.scan`` regions and ``scf.for`` loop-carried values;
* canonicalize commutative operands and bounded comparison rewrites;
* record reductions, masked memory effects and fills, dots, loops, and effects;
* emit an immutable, JSON-serializable fingerprint.

The parser is intentionally strict.  An unresolved SSA value is a parse error,
not an invitation to silently produce a weaker fingerprint.  It targets the
custom assembly emitted by Triton 3.6 TTIR and also accepts quoted generic op
names (for example ``"tt.reduce"``).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

__all__ = [
    "DotFingerprint",
    "InterfaceArgument",
    "LoopFingerprint",
    "MemoryEffect",
    "ReductionFingerprint",
    "RuntimeInterface",
    "TerminalForkArmFingerprint",
    "TerminalForkFingerprint",
    "TTIRFingerprint",
    "TTIR_FINGERPRINT_SCHEMA",
    "TTIRParseError",
    "TTIRParser",
    "extract_ttir_fingerprint",
    "fingerprint_ttir",
    "parse_ttir",
]


TTIR_FINGERPRINT_SCHEMA = "triton-ttir-3.6-gate1-v7"
_SCHEMA_VERSION = TTIR_FINGERPRINT_SCHEMA
_SSA_RE = re.compile(r"%[A-Za-z0-9_.$-]+(?:#[0-9]+)?")
_LOC_START_RE = re.compile(r"loc\s*\(")
_OP_RE = re.compile(
    r'(?P<quoted>"[A-Za-z_][A-Za-z0-9_.-]*")|'
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_.-]*)"
)
_COMMUTATIVE_OPS = {
    "arith.addf",
    "arith.addi",
    "arith.andi",
    "arith.maxf",
    "arith.maxnumf",
    "arith.maximumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minf",
    "arith.minnumf",
    "arith.minimumf",
    "arith.minsi",
    "arith.minui",
    "arith.mulf",
    "arith.muli",
    "arith.ori",
    "arith.xori",
}
_REGION_OWNERS = {"tt.reduce", "tt.scan", "scf.for"}
_REGION_TERMINATORS = {"tt.reduce.return", "tt.scan.return", "scf.yield"}
_EXPECTED_REGION_TERMINATOR = {
    "tt.reduce": "tt.reduce.return",
    "tt.scan": "tt.scan.return",
    "scf.for": "scf.yield",
}
_TERMINATORS = {"tt.return", *_REGION_TERMINATORS}
_KNOWN_PURE_OPS = {
    # Pinned pure Arith operations observed in the corpus or emitted by the
    # supported bounded normalizations.  Dialect prefixes are intentionally
    # not trusted: a future ``arith.*`` spelling is unsupported until it is
    # classified here.
    "arith.addf",
    "arith.addi",
    "arith.andi",
    "arith.bitcast",
    "arith.cmpf",
    "arith.cmpi",
    "arith.constant",
    "arith.divf",
    "arith.divsi",
    "arith.divui",
    "arith.extf",
    "arith.extsi",
    "arith.extui",
    "arith.floordivsi",
    "arith.fptosi",
    "arith.fptoui",
    "arith.index_cast",
    "arith.index_castui",
    "arith.maximumf",
    "arith.maxf",
    "arith.maxnumf",
    "arith.maxsi",
    "arith.maxui",
    "arith.minimumf",
    "arith.minf",
    "arith.minnumf",
    "arith.minsi",
    "arith.minui",
    "arith.mulf",
    "arith.muli",
    "arith.negf",
    "arith.ori",
    "arith.remsi",
    "arith.remui",
    "arith.select",
    "arith.shli",
    "arith.shrsi",
    "arith.shrui",
    "arith.sitofp",
    "arith.subf",
    "arith.subi",
    "arith.truncf",
    "arith.trunci",
    "arith.uitofp",
    "arith.xori",
    # Pinned pure Math operations.  Keep this explicit for the same reason.
    "math.absf",
    "math.absi",
    "math.acos",
    "math.asin",
    "math.atan",
    "math.atan2",
    "math.cbrt",
    "math.ceil",
    "math.copysign",
    "math.cos",
    "math.erf",
    "math.exp",
    "math.exp2",
    "math.expm1",
    "math.floor",
    "math.fma",
    "math.log",
    "math.log10",
    "math.log1p",
    "math.log2",
    "math.powf",
    "math.round",
    "math.roundeven",
    "math.rsqrt",
    "math.sin",
    "math.sqrt",
    "math.tan",
    "math.tanh",
    "math.trunc",
    # Structured owners are pure only when every operation in their nested
    # region is accepted by this same closed check.
    "scf.for",
    "tt.reduce",
    "tt.scan",
    "tt.return",
    *_REGION_TERMINATORS,
    # Triton value/address constructors.  Loads, stores, atomics,
    # descriptors, barriers, assertions, and other observable operations are
    # deliberately absent.
    "tt.addptr",
    "tt.advance",
    "tt.bitcast",
    "tt.broadcast",
    "tt.cat",
    "tt.clamp",
    "tt.dot",
    "tt.expand_dims",
    "tt.fp_to_fp",
    "tt.get_num_programs",
    "tt.get_program_id",
    "tt.join",
    "tt.make_range",
    "tt.make_tensor_descriptor",
    "tt.precise_divf",
    "tt.precise_sqrt",
    "tt.reshape",
    "tt.split",
    "tt.splat",
    "tt.trans",
}
_KNOWN_EFFECT_OPS = {
    "gpu.barrier",
    "gpu.wait",
    "tt.assert",
    "tt.async_wait",
    "tt.atomic_cas",
    "tt.atomic_rmw",
    "tt.barrier",
    "tt.debug_barrier",
    "tt.descriptor_gather",
    "tt.descriptor_load",
    "tt.descriptor_reduce",
    "tt.descriptor_scatter",
    "tt.descriptor_store",
    "tt.device_assert",
    "tt.load",
    "tt.print",
    "tt.store",
}
_CMP_FLIPS = {
    "sgt": "slt",
    "sge": "sle",
    "ugt": "ult",
    "uge": "ule",
    "ogt": "olt",
    "oge": "ole",
}
_SYMMETRIC_CMP = {"eq", "ne", "oeq", "one", "ueq", "une"}


class TTIRParseError(ValueError):
    """The TTIR is outside the supported deterministic Gate1 subset."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*([,(){}\[\]<>])\s*", r"\1", text)
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"\s*=\s*", "=", text)
    return text


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split while respecting MLIR's nested delimiters and quoted strings."""

    pieces: list[str] = []
    start = 0
    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == delimiter and not stack:
            pieces.append(text[start:index].strip())
            start = index + 1
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int:
    if start >= len(text) or text[start] != opening:
        raise TTIRParseError(f"expected {opening!r} at offset {start}")
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
    raise TTIRParseError(f"unterminated {opening!r} starting at offset {start}")


def _strip_locations(text: str) -> str:
    """Remove location definitions and balanced ``loc(...)`` annotations."""

    if not isinstance(text, str) or not text.strip():
        raise TTIRParseError("TTIR text is empty")
    text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#loc")
    )
    output: list[str] = []
    index = 0
    while index < len(text):
        match = _LOC_START_RE.match(text, index)
        boundary = index == 0 or not (text[index - 1].isalnum() or text[index - 1] in "_.$")
        if match and boundary:
            opening = match.end() - 1
            index = _balanced_end(text, opening, "(", ")") + 1
            continue
        output.append(text[index])
        index += 1
    return "".join(output)


def _canonical_attrs(text: str) -> tuple[tuple[str, str], ...]:
    text = text.strip()
    if not text:
        return ()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    attrs: list[tuple[str, str]] = []
    for item in _split_top_level(text):
        if "=" in item:
            name, value = item.split("=", 1)
            attrs.append((_normalize_space(name), _normalize_space(value)))
        else:
            attrs.append((_normalize_space(item), ""))
    return tuple(sorted(attrs))


def _validate_attribute_dictionary(
    text: str, *, context: str
) -> tuple[tuple[str, str], ...]:
    """Parse one pinned bare-name attr dictionary and reject duplicates."""

    attrs = _canonical_attrs(text)
    seen: set[str] = set()
    for name, _value in attrs:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", name) is None:
            raise TTIRParseError(
                f"{context} has malformed attribute name {name!r}"
            )
        if name in seen:
            raise TTIRParseError(
                f"{context} repeats attribute {name!r}"
            )
        seen.add(name)
    return attrs


def _pointer_dtype(type_text: str) -> str | None:
    match = re.search(r"!tt\.ptr<([^,>]+)", type_text)
    return _normalize_space(match.group(1)) if match else None


@dataclass(frozen=True)
class InterfaceArgument:
    """One positional runtime argument; SSA spelling is deliberately omitted."""

    position: int
    type: str
    kind: str
    dtype: str | None
    rank: int | None
    access: str
    attributes: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "type": self.type,
            "kind": self.kind,
            "dtype": self.dtype,
            "rank": self.rank,
            "access": self.access,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class RuntimeInterface:
    arguments: tuple[InterfaceArgument, ...]
    results: tuple[str, ...] = ()
    python: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arguments": [argument.to_dict() for argument in self.arguments],
            "results": list(self.results),
            "python": {key: json.loads(value) for key, value in self.python},
        }


@dataclass(frozen=True)
class ReductionFingerprint:
    axis: int
    combiners: tuple[str, ...]
    input_count: int
    result_count: int

    @property
    def combiner(self) -> str | None:
        return self.combiners[0] if len(self.combiners) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "combiners": list(self.combiners),
            "input_count": self.input_count,
            "result_count": self.result_count,
        }


@dataclass(frozen=True)
class MemoryEffect:
    kind: str
    masked: bool
    predicate: str | None = None
    other: str | None = None
    attributes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "masked": self.masked,
            "predicate": self.predicate,
            "other": self.other,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class DotFingerprint:
    operation: str
    operand_count: int
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "operand_count": self.operand_count,
            "signature": self.signature,
        }


@dataclass(frozen=True)
class LoopFingerprint:
    lower: str
    upper: str
    step: str
    init_values: tuple[str, ...]
    yielded_values: tuple[str, ...]
    body_effects: tuple[tuple[str, int], ...]
    block_argument_count: int
    induction_use_count: int
    carried_argument_use_counts: tuple[int, ...]
    captured_values: tuple[str, ...]
    body_op_multiset: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "step": self.step,
            "init_values": list(self.init_values),
            "yielded_values": list(self.yielded_values),
            "body_effects": dict(self.body_effects),
            "block_argument_count": self.block_argument_count,
            "induction_use_count": self.induction_use_count,
            "carried_argument_use_counts": list(
                self.carried_argument_use_counts
            ),
            "captured_values": list(self.captured_values),
            "body_op_multiset": dict(self.body_op_multiset),
        }


@dataclass(frozen=True)
class TerminalForkArmFingerprint:
    """One role-preserving arm of the closed terminal binary-fork profile."""

    role: str
    members: tuple[str, ...]
    effect_order: tuple[str, ...]
    structured_regions: tuple[str, ...]
    terminal: str = "tt.return:void"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "members": list(self.members),
            "effect_order": list(self.effect_order),
            "structured_regions": list(self.structured_regions),
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class TerminalForkFingerprint:
    """Hard evidence for one non-merging scalar conditional terminal fork."""

    predicate: str
    region_context: tuple[str, ...]
    true_arm: TerminalForkArmFingerprint
    false_arm: TerminalForkArmFingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "region_context": list(self.region_context),
            "true_arm": self.true_arm.to_dict(),
            "false_arm": self.false_arm.to_dict(),
        }


@dataclass(frozen=True)
class TTIRFingerprint:
    """Immutable Gate1 fingerprint.  All sequence fields are canonically sorted."""

    interface: RuntimeInterface
    op_multiset: tuple[tuple[str, int], ...]
    dataflow_edges: tuple[tuple[str, str, str], ...]
    reductions: tuple[ReductionFingerprint, ...]
    memory_effects: tuple[MemoryEffect, ...]
    mask_predicates: tuple[str, ...]
    effect_multiset: tuple[tuple[str, int], ...]
    dots: tuple[DotFingerprint, ...]
    loops: tuple[LoopFingerprint, ...]
    terminal_forks: tuple[TerminalForkFingerprint, ...] = ()
    schedule_view: tuple[tuple[str, str], ...] = ()
    schema_version: str = _SCHEMA_VERSION

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": self.interface.to_dict(),
            "op_multiset": dict(self.op_multiset),
            "dataflow_edges": [list(edge) for edge in self.dataflow_edges],
            "reductions": [item.to_dict() for item in self.reductions],
            "memory_effects": [item.to_dict() for item in self.memory_effects],
            "mask_predicates": list(self.mask_predicates),
            "effect_multiset": dict(self.effect_multiset),
            "dots": [item.to_dict() for item in self.dots],
            "loops": [item.to_dict() for item in self.loops],
            "terminal_forks": [
                item.to_dict() for item in self.terminal_forks
            ],
            "schedule_view": {key: json.loads(value) for key, value in self.schedule_view},
        }


@dataclass
class _Operation:
    node_id: int
    name: str
    result_names: tuple[str, ...]
    operands: list[str]
    local_signature: str
    context: tuple[int, ...]
    axis: int | None = None
    reverse: bool | None = None
    cmp_predicate: str | None = None
    cmp_flipped: bool = False
    header_operand_count: int = 0
    region_results: list[str] = field(default_factory=list)
    region_block_args: list[str] = field(default_factory=list)
    constant_value: float | None = None
    callee: str | None = None
    call_operand_types: tuple[str, ...] = ()
    call_result_types: tuple[str, ...] = ()
    terminator_types: tuple[str, ...] = ()
    fork_id: int | None = None
    fork_arm: str | None = None


@dataclass
class _TerminalFork:
    fork_id: int
    predicate: str
    context: tuple[int, ...]
    sequence_index: int


@dataclass(frozen=True)
class _FunctionParts:
    name: str
    visibility: str
    arguments: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]
    results: tuple[str, ...]
    body: str
    attributes: tuple[tuple[str, str], ...] = ()


_FUNCTION_RE = re.compile(
    r"^\s*tt\.func\s+(?:(?P<visibility>public|private)\s+)?@"
    r'(?P<name>"(?:\\.|[^"\\])*"|[A-Za-z0-9_.$-]+)\s*\(',
    flags=re.MULTILINE,
)
_SYMBOL_RE = r'"(?:\\.|[^"\\])*"|[A-Za-z0-9_.$-]+'


def _symbol_value(token: str) -> str:
    if not token.startswith('"'):
        return token
    try:
        value = json.loads(token)
    except json.JSONDecodeError as error:
        raise TTIRParseError(f"invalid quoted MLIR symbol {token!r}") from error
    if not isinstance(value, str):
        raise TTIRParseError(f"invalid quoted MLIR symbol {token!r}")
    return value


def _parse_function_definition(text: str, match: re.Match[str]) -> _FunctionParts:
    args_open = match.end() - 1
    args_close = _balanced_end(text, args_open, "(", ")")
    args_text = text[args_open + 1 : args_close]

    arguments: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for item in _split_top_level(args_text):
        argument_match = re.match(
            r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<rest>.*)",
            item,
            re.S,
        )
        if argument_match is None:
            raise TTIRParseError(f"cannot parse function argument: {item!r}")
        rest = argument_match.group("rest").strip()
        attr_start = _find_top_level_char(rest, "{")
        if attr_start is None:
            type_text, attrs = rest, ()
        else:
            attr_end = _balanced_end(rest, attr_start, "{", "}")
            type_text = rest[:attr_start]
            attrs = _canonical_attrs(rest[attr_start : attr_end + 1])
        arguments.append(
            (
                argument_match.group("name"),
                _validate_closed_type(
                    type_text,
                    context=f"tt.func argument {argument_match.group('name')}",
                ),
                attrs,
            )
        )

    # The first brace after the argument list can be the function attribute
    # dictionary.  ``_next_body_brace`` skips it and returns the actual body.
    body_open = _next_body_brace(text, args_close + 1)
    between = text[args_close + 1 : body_open]
    attribute_marker = re.search(r"\battributes\s*\{", between)
    function_attributes: tuple[tuple[str, str], ...] = ()
    if attribute_marker is not None:
        attribute_open = between.find("{", attribute_marker.start())
        attribute_close = _balanced_end(
            between, attribute_open, "{", "}"
        )
        if between[attribute_close + 1 :].strip():
            raise TTIRParseError(
                "unexpected syntax after tt.func attribute dictionary"
            )
        function_attributes = _canonical_attrs(
            between[attribute_open : attribute_close + 1]
        )
        between = between[: attribute_marker.start()]
    body_close = _balanced_end(text, body_open, "{", "}")

    result_text = between.strip()
    if result_text.startswith("->"):
        result_text = result_text[2:].strip()
        if result_text.startswith("(") and result_text.endswith(")"):
            results = tuple(
                _validate_closed_type(
                    item,
                    context=f"tt.func @{_symbol_value(match.group('name'))} result",
                )
                for item in _split_top_level(result_text[1:-1])
            )
        else:
            results = (
                _validate_closed_type(
                    result_text,
                    context=f"tt.func @{_symbol_value(match.group('name'))} result",
                ),
            )
    else:
        results = ()
    return _FunctionParts(
        name=_symbol_value(match.group("name")),
        visibility=match.group("visibility") or "private",
        arguments=tuple(arguments),
        results=results,
        body=text[body_open + 1 : body_close],
        attributes=function_attributes,
    )


def _parse_functions(text: str) -> dict[str, _FunctionParts]:
    functions: dict[str, _FunctionParts] = {}
    for match in _FUNCTION_RE.finditer(text):
        function = _parse_function_definition(text, match)
        if function.name in functions:
            raise TTIRParseError(f"duplicate tt.func symbol @{function.name}")
        functions[function.name] = function
    public = [
        function
        for function in functions.values()
        if function.visibility == "public"
    ]
    if len(public) != 1:
        raise TTIRParseError(
            f"expected exactly one public tt.func, found {len(public)}"
        )
    return functions


def _parse_function(text: str) -> _FunctionParts:
    functions = _parse_functions(text)
    return next(
        function for function in functions.values() if function.visibility == "public"
    )


def _validate_function_metadata_boundary(function: _FunctionParts) -> None:
    """Validate the closed function-metadata boundary for TTIR v7."""

    if function.visibility != "public":
        for position, (_, _, attributes) in enumerate(function.arguments):
            if attributes:
                raise TTIRParseError(
                    f"reachable private tt.func @{function.name} argument "
                    f"{position} has unsupported attributes"
                )
    seen: set[str] = set()
    for name, value in function.attributes:
        if name in seen:
            raise TTIRParseError(
                f"reachable tt.func @{function.name} repeats "
                f"attribute {name!r}"
            )
        seen.add(name)
        if name != "noinline" or value not in {"false", "true"}:
            raise TTIRParseError(
                f"reachable tt.func @{function.name} has unsupported "
                f"function attribute {name!r}={value!r}"
            )


def _find_top_level_char(text: str, wanted: str) -> int | None:
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "<": ">"}
    quote: str | None = None
    for index, char in enumerate(text):
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = None
            continue
        if char in {'"', "'"}:
            quote = char
        elif char == wanted and not stack:
            return index
        elif char in pairs:
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
    return None


def _next_body_brace(text: str, start: int) -> int:
    cursor = start
    while cursor < len(text):
        if text.startswith("attributes", cursor):
            attr_open = text.find("{", cursor)
            cursor = _balanced_end(text, attr_open, "{", "}") + 1
            continue
        if text[cursor] == "{":
            return cursor
        cursor += 1
    raise TTIRParseError("public tt.func has no body")


def _parse_results(lhs: str) -> tuple[str, ...]:
    results: list[str] = []
    for item in _split_top_level(lhs):
        item = item.strip()
        group = re.fullmatch(r"(?P<base>%[A-Za-z0-9_.$-]+):(?P<count>[0-9]+)", item)
        if group:
            base = group.group("base")
            count = int(group.group("count"))
            results.extend(f"{base}#{index}" for index in range(count))
        elif _SSA_RE.fullmatch(item):
            results.append(item)
        else:
            raise TTIRParseError(f"cannot parse operation result: {item!r}")
    return tuple(results)


def _operation_head(
    line: str,
) -> tuple[tuple[str, ...], str, str, bool] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(("^", "}", ")", "module", "tt.func")):
        return None
    lhs: tuple[str, ...] = ()
    tail = stripped
    assignment = re.match(r"(?P<lhs>%[^=]+?)\s*=\s*(?P<tail>.*)$", stripped)
    if assignment:
        lhs = _parse_results(assignment.group("lhs"))
        tail = assignment.group("tail").strip()
    match = _OP_RE.match(tail)
    if match is None:
        return None
    token = match.group("quoted") or match.group("plain")
    assert token is not None
    name = token.strip('"')
    if "." not in name:
        return None
    return lhs, name, tail[match.end() :].strip(), token.startswith('"')


def _validate_quoted_generic_shell(
    name: str,
    tail: str,
    *,
    result_count: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Validate the closed, region-free MLIR generic-operation shell.

    Gate1 validates syntax before applying its closed operation/effect
    classification.  Merely recognizing ``"dialect.op"`` is insufficient:
    accepting arbitrary trailing prose would make malformed TTIR
    indistinguishable from a well-formed but observer-unsupported operation.

    Region owners are parsed by their dedicated structured parsers, so this
    helper intentionally accepts only the region-free spelling
    ``(%operands) [<{properties}>] [{attributes}] : inputs -> results``.
    """

    stripped = tail.strip()
    if not stripped.startswith("("):
        raise TTIRParseError(
            f'quoted generic operation "{name}" is missing its operand list'
        )
    operands_close = _balanced_end(stripped, 0, "(", ")")
    operand_items = _split_top_level(stripped[1:operands_close])
    for index, item in enumerate(operand_items):
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f'quoted generic operation "{name}" operand {index} is not '
                f"one SSA value: {item!r}"
            )

    remainder = stripped[operands_close + 1 :].strip()
    colon = _find_top_level_token(remainder, ":", last=True)
    if colon is None:
        raise TTIRParseError(
            f'quoted generic operation "{name}" is missing its explicit '
            "function type"
        )
    prefix = remainder[:colon].strip()
    signature = remainder[colon + 1 :].strip()

    # Properties and attributes are both optional, but their order and
    # delimiters are part of the pinned generic syntax.  No unparsed token is
    # allowed between the operand list and the type contract.
    if prefix.startswith("<"):
        properties_close = _balanced_end(prefix, 0, "<", ">")
        properties = prefix[: properties_close + 1]
        if not (properties.startswith("<{") and properties.endswith("}>")):
            raise TTIRParseError(
                f'quoted generic operation "{name}" has malformed properties'
            )
        _validate_attribute_dictionary(
            properties[1:-1],
            context=f'quoted generic operation "{name}" properties',
        )
        prefix = prefix[properties_close + 1 :].strip()
    if prefix.startswith("{"):
        attributes_close = _balanced_end(prefix, 0, "{", "}")
        _validate_attribute_dictionary(
            prefix[: attributes_close + 1],
            context=f'quoted generic operation "{name}" attributes',
        )
        prefix = prefix[attributes_close + 1 :].strip()
    if prefix:
        raise TTIRParseError(
            f'quoted generic operation "{name}" has unsupported or malformed '
            f"trailing syntax before its type: {prefix!r}"
        )

    input_types, result_types = _functional_types(
        signature, context=f'quoted generic operation "{name}"'
    )
    if len(input_types) != len(operand_items):
        raise TTIRParseError(
            f'quoted generic operation "{name}" operand/type arity mismatch'
        )
    if len(result_types) != result_count:
        raise TTIRParseError(
            f'quoted generic operation "{name}" result/type arity mismatch'
        )
    return tuple(operand_items), input_types, result_types


def _parse_call_contract(
    tail: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Parse a Triton ``tt.call`` target and its explicit function type.

    Triton 3.6 normally emits custom assembly, ``tt.call @callee(...) :
    (...) -> (...)``.  The quoted generic spelling is accepted only when it
    carries the same explicit ``callee = @...`` attribute.  Gate1 does not
    infer missing call contracts from a callee: that would turn malformed or
    version-skewed TTIR into weaker evidence.
    """

    custom = re.match(rf"@(?P<name>{_SYMBOL_RE})\s*\(", tail)
    if custom is not None:
        callee = _symbol_value(custom.group("name"))
        args_open = custom.end() - 1
    else:
        generic = re.search(rf"\bcallee\s*=\s*@(?P<name>{_SYMBOL_RE})", tail)
        if generic is None:
            raise TTIRParseError("tt.call is missing a statically known callee")
        callee = _symbol_value(generic.group("name"))
        stripped = tail.lstrip()
        if not stripped.startswith("("):
            raise TTIRParseError("generic tt.call is missing its operand list")
        args_open = len(tail) - len(stripped)

    args_close = _balanced_end(tail, args_open, "(", ")")
    signature_colon = tail.find(":", args_close + 1)
    if signature_colon < 0:
        raise TTIRParseError(f"tt.call @{callee} is missing its explicit type")
    signature = tail[signature_colon + 1 :].strip()
    if not signature.startswith("("):
        raise TTIRParseError(f"tt.call @{callee} has an invalid operand type list")
    input_close = _balanced_end(signature, 0, "(", ")")
    operand_types = tuple(
        _normalize_space(item)
        for item in _split_top_level(signature[1:input_close])
    )
    remainder = signature[input_close + 1 :].strip()
    if not remainder.startswith("->"):
        raise TTIRParseError(f"tt.call @{callee} is missing its result type")
    result_text = remainder[2:].strip()
    if result_text == "()":
        result_types: tuple[str, ...] = ()
    elif result_text.startswith("("):
        result_close = _balanced_end(result_text, 0, "(", ")")
        if result_text[result_close + 1 :].strip():
            raise TTIRParseError(f"tt.call @{callee} has trailing type syntax")
        result_types = tuple(
            _normalize_space(item)
            for item in _split_top_level(result_text[1:result_close])
        )
    else:
        result_types = (_normalize_space(result_text),)
    return callee, operand_types, result_types


def _find_top_level_token(text: str, token: str, *, last: bool = False) -> int | None:
    """Find a token outside MLIR delimiters and quoted strings."""

    stack: list[str] = []
    quote: str | None = None
    escaped = False
    pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    found: int | None = None
    index = 0
    while index <= len(text) - len(token):
        char = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        if char in pairs:
            stack.append(pairs[char])
            index += 1
            continue
        if stack and char == stack[-1]:
            stack.pop()
            index += 1
            continue
        if not stack and text.startswith(token, index):
            found = index
            if not last:
                return found
            index += len(token)
            continue
        index += 1
    return found


def _validate_closed_type(type_text: str, *, context: str) -> str:
    """Validate the encodingless, alias-free type boundary observed by Gate1.

    TTIR v7 deliberately does not interpret module-level type/encoding aliases.
    Accepting only their use-site spelling would let two different alias
    definitions hard-pass, so every proof-relevant type must be self-contained.
    Pinned TTIR is pre-layout and therefore needs no ranked-tensor encoding.
    """

    normalized = _normalize_space(type_text)
    if not normalized:
        raise TTIRParseError(f"{context} has an empty type")
    if re.search(r"#[A-Za-z_][A-Za-z0-9_.$-]*", normalized) is not None:
        raise TTIRParseError(
            f"{context} uses a module-level type/encoding alias; tensor "
            "encodings are unsupported and Gate1 v7 requires self-contained "
            "encodingless TTIR types"
        )
    if re.search(r"\s", normalized) is not None:
        raise TTIRParseError(
            f"{context} has unsupported trailing or compound type syntax: "
            f"{normalized!r}"
        )
    cursor = 0
    while True:
        marker = normalized.find("tensor<", cursor)
        if marker < 0:
            break
        opening = marker + len("tensor")
        closing = _balanced_end(normalized, opening, "<", ">")
        pieces = _split_top_level(normalized[opening + 1 : closing])
        if len(pieces) != 1:
            raise TTIRParseError(
                f"{context} uses a ranked-tensor encoding; Gate1 v7 "
                "requires encodingless TTIR types"
            )
        cursor = closing + 1
    return normalized


def _declared_types(text: str) -> tuple[str, ...]:
    text = text.strip()
    if not text or text == "()":
        return ()
    if text.startswith("("):
        closing = _balanced_end(text, 0, "(", ")")
        if text[closing + 1 :].strip():
            raise TTIRParseError(f"trailing syntax after type list {text!r}")
        text = text[1:closing]
    return tuple(
        _validate_closed_type(item, context="explicit operation type")
        for item in _split_top_level(text)
    )


def _functional_types(
    text: str, *, context: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse one explicit ``inputs -> results`` MLIR type contract."""

    signature = text.strip()
    arrow = _find_top_level_token(signature, "->")
    if arrow is None:
        raise TTIRParseError(f"{context} is missing its explicit function type")
    if _find_top_level_token(signature[arrow + 2 :], "->") is not None:
        raise TTIRParseError(f"{context} has more than one result arrow")
    input_text = signature[:arrow].strip()
    if not input_text.startswith("("):
        raise TTIRParseError(
            f"{context} function type inputs must use an explicit tuple"
        )
    input_close = _balanced_end(input_text, 0, "(", ")")
    if input_text[input_close + 1 :].strip():
        raise TTIRParseError(
            f"{context} has trailing syntax after its input type tuple"
        )
    inputs = _declared_types(input_text)
    results = _declared_types(signature[arrow + 2 :])
    return inputs, results


def _ranked_tensor_contract(
    type_text: str | None,
) -> tuple[tuple[str, ...], str, tuple[str, ...]] | None:
    """Return ``(shape, element, encoding)`` for a ranked tensor type.

    This deliberately recognizes only the closed ranked-tensor spelling used
    by pinned Triton TTIR.  It is not a general MLIR type parser.
    """

    if type_text is None:
        return None
    normalized = _normalize_space(type_text)
    if not normalized.startswith("tensor<"):
        return None
    opening = normalized.find("<")
    try:
        closing = _balanced_end(normalized, opening, "<", ">")
    except TTIRParseError:
        return None
    if normalized[closing + 1 :].strip():
        return None
    pieces = _split_top_level(normalized[opening + 1 : closing])
    if not pieces:
        return None
    if len(pieces) != 1:
        # Gate1 v7 does not parse module-level encoding aliases or prove their
        # denotation.  Comparing only alias spellings would let two different
        # layouts hard-pass, so the pinned scan subset is explicitly
        # encodingless.
        raise TTIRParseError(
            "tt.scan tensor encodings are unsupported by Gate1 v7; inputs "
            "and results must be encodingless"
        )
    shaped = _split_top_level(pieces[0], delimiter="x")
    if not shaped:
        return None
    dimensions = tuple(_normalize_space(item) for item in shaped[:-1])
    if any(re.fullmatch(r"(?:\?|[0-9]+)", item) is None for item in dimensions):
        return None
    element = _normalize_space(shaped[-1])
    if not element:
        return None
    return dimensions, element, ()


def _parse_scan_header(
    tail: str,
) -> tuple[list[str], int, bool]:
    """Parse Triton 3.6's quoted-generic ``tt.scan`` opening line.

    ``tt.scan`` has no custom assembly format in Triton 3.6.  The pinned
    compiler emits the operand list, a closed property dictionary containing
    exactly ``axis`` and ``reverse``, and then the single combine region.
    """

    stripped = tail.strip()
    if not stripped.startswith("("):
        raise TTIRParseError("generic tt.scan is missing its operand list")
    operands_close = _balanced_end(stripped, 0, "(", ")")
    operand_items = _split_top_level(stripped[1:operands_close])
    if not operand_items:
        raise TTIRParseError("tt.scan must have at least one input")
    operands: list[str] = []
    for index, item in enumerate(operand_items):
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f"tt.scan operand {index} is not one SSA value: {item!r}"
            )
        operands.append(item)

    remainder = stripped[operands_close + 1 :].strip()
    if not remainder.startswith("<{"):
        raise TTIRParseError(
            "tt.scan requires the Triton 3.6 <{axis, reverse}> property dictionary"
        )
    properties_close = _balanced_end(remainder, 0, "<", ">")
    properties_text = remainder[1:properties_close]
    properties = _canonical_attrs(properties_text)
    if len(properties) != 2 or {name for name, _ in properties} != {
        "axis",
        "reverse",
    }:
        raise TTIRParseError(
            "tt.scan properties must contain exactly one axis and one reverse"
        )
    property_map = dict(properties)
    axis_match = re.fullmatch(r"(-?[0-9]+):i32", property_map["axis"])
    if axis_match is None:
        raise TTIRParseError("tt.scan axis must be an explicit i32 integer")
    reverse_text = property_map["reverse"]
    if reverse_text not in {"false", "true"}:
        raise TTIRParseError("tt.scan reverse must be an explicit boolean")

    region_open = _normalize_space(remainder[properties_close + 1 :])
    if region_open != "({":
        raise TTIRParseError(
            "tt.scan must have exactly one quoted-generic combine region"
        )
    return operands, int(axis_match.group(1)), reverse_text == "true"


def _parse_reduce_header(tail: str) -> tuple[list[str], int]:
    """Parse Triton 3.6's quoted-generic ``tt.reduce`` opening line."""

    stripped = tail.strip()
    if not stripped.startswith("("):
        raise TTIRParseError("generic tt.reduce is missing its operand list")
    operands_close = _balanced_end(stripped, 0, "(", ")")
    operand_items = _split_top_level(stripped[1:operands_close])
    if not operand_items:
        raise TTIRParseError("tt.reduce must have at least one input")
    operands: list[str] = []
    for index, item in enumerate(operand_items):
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f"tt.reduce operand {index} is not one SSA value: {item!r}"
            )
        operands.append(item)

    remainder = stripped[operands_close + 1 :].strip()
    if not remainder.startswith("<{"):
        raise TTIRParseError(
            "tt.reduce requires the Triton 3.6 <{axis}> property dictionary"
        )
    properties_close = _balanced_end(remainder, 0, "<", ">")
    properties = _validate_attribute_dictionary(
        remainder[1:properties_close],
        context="tt.reduce properties",
    )
    if len(properties) != 1 or properties[0][0] != "axis":
        raise TTIRParseError("tt.reduce properties must contain exactly one axis")
    axis_match = re.fullmatch(r"(-?[0-9]+):i32", properties[0][1])
    if axis_match is None:
        raise TTIRParseError("tt.reduce axis must be an explicit i32 integer")
    if _normalize_space(remainder[properties_close + 1 :]) != "({":
        raise TTIRParseError(
            "tt.reduce must have exactly one quoted-generic combine region"
        )
    return operands, int(axis_match.group(1))


def _parse_typed_terminator(
    name: str,
    tail: str,
    *,
    allow_empty: bool,
) -> tuple[list[str], tuple[str, ...]]:
    """Parse one exact pinned custom terminator shell."""

    stripped = tail.strip()
    if not stripped:
        if allow_empty:
            return [], ()
        raise TTIRParseError(f"{name} requires at least one returned SSA value")
    if "{" in stripped or "}" in stripped:
        raise TTIRParseError(
            f"{name} attributes are unsupported by Gate1's pinned terminator shell"
        )
    colon = _find_top_level_token(stripped, ":", last=True)
    if colon is None:
        raise TTIRParseError(f"{name} is missing its explicit result types")
    operand_text = stripped[:colon].strip()
    type_text = stripped[colon + 1 :].strip()
    if not type_text:
        raise TTIRParseError(f"{name} is missing its explicit result types")
    if type_text.startswith("("):
        raise TTIRParseError(
            f"{name} custom result types must be an unparenthesized list"
        )

    if (
        not operand_text
        or operand_text.startswith(",")
        or operand_text.endswith(",")
        or re.search(r",\s*,", operand_text) is not None
    ):
        raise TTIRParseError(
            f"{name} requires a non-empty comma-separated operand list"
        )
    operand_items = _split_top_level(operand_text)
    if not operand_items or any(not item.strip() for item in operand_items):
        raise TTIRParseError(
            f"{name} requires a non-empty comma-separated operand list"
        )
    operands: list[str] = []
    for index, item in enumerate(operand_items):
        item = item.strip()
        if _SSA_RE.fullmatch(item) is None:
            raise TTIRParseError(
                f"{name} operand {index} is not one SSA value: {item!r}"
            )
        operands.append(item)

    declared = _declared_types(type_text)
    if len(declared) != len(operands):
        raise TTIRParseError(f"{name} SSA/type arity mismatch")
    return operands, declared


def _parse_scan_return(
    tail: str,
) -> tuple[list[str], tuple[str, ...]]:
    """Parse the exact Triton 3.6 custom ``tt.scan.return`` spelling."""

    return _parse_typed_terminator(
        "tt.scan.return", tail, allow_empty=False
    )


def _trailing_type_signature(tail: str) -> str | None:
    colon = _find_top_level_token(tail, ":", last=True)
    if colon is None:
        return None
    signature = tail[colon + 1 :].strip()
    return signature or None


def _loaded_value_type(pointer_type: str | None) -> str | None:
    if pointer_type is None:
        return None
    pointer_type = _normalize_space(pointer_type)
    if pointer_type.startswith("!tt.ptr<"):
        opening = pointer_type.find("<")
        closing = _balanced_end(pointer_type, opening, "<", ">")
        if pointer_type[closing + 1 :].strip():
            return None
        parts = _split_top_level(pointer_type[opening + 1 : closing])
        return _normalize_space(parts[0]) if parts else None
    marker = pointer_type.find("!tt.ptr<")
    if marker < 0 or not pointer_type.startswith("tensor<"):
        return None
    opening = pointer_type.find("<", marker)
    closing = _balanced_end(pointer_type, opening, "<", ">")
    parts = _split_top_level(pointer_type[opening + 1 : closing])
    if not parts:
        return None
    return _normalize_space(
        pointer_type[:marker] + parts[0] + pointer_type[closing + 1 :]
    )


def _comparison_result_type(operand_type: str | None) -> str | None:
    if operand_type is None:
        return None
    operand_type = _normalize_space(operand_type)
    if not operand_type.startswith("tensor<"):
        return "i1"
    body = operand_type[len("tensor<") : -1] if operand_type.endswith(">") else ""
    if not body:
        return None
    pieces = _split_top_level(body)
    shape_and_element = pieces[0]
    separator = shape_and_element.rfind("x")
    if separator < 0:
        return None
    pieces[0] = shape_and_element[: separator + 1] + "i1"
    return _normalize_space(f"tensor<{','.join(pieces)}>")


def _operation_result_types(
    name: str,
    tail: str,
    result_count: int,
    operand_types: Sequence[str | None],
    *,
    call_result_types: tuple[str, ...] = (),
) -> tuple[str | None, ...]:
    """Infer only result types exposed explicitly by Triton custom assembly.

    This is intentionally not a general MLIR type checker.  Unknown result
    forms remain ``None`` and are rejected only if they cross a ``tt.call`` or
    ``tt.return`` boundary where Gate1 must prove the actual SSA type.
    """

    if result_count == 0:
        return ()
    if name == "tt.call":
        if len(call_result_types) != result_count:
            raise TTIRParseError("tt.call result SSA/type arity mismatch")
        return tuple(call_result_types)
    if name == "scf.for":
        arrow = _find_top_level_token(tail, "->", last=True)
        if arrow is None:
            return (None,) * result_count
        result_text = tail[arrow + 2 :].lstrip()
        if not result_text.startswith("("):
            return (None,) * result_count
        closing = _balanced_end(result_text, 0, "(", ")")
        inferred = _declared_types(result_text[: closing + 1])
        return (
            tuple(inferred)
            if len(inferred) == result_count
            else (None,) * result_count
        )
    if name == "tt.addptr":
        inferred = operand_types[0] if operand_types else None
        return (inferred,) if result_count == 1 else (None,) * result_count
    if name == "tt.load" or name.startswith("tt.atomic"):
        inferred = _loaded_value_type(operand_types[0] if operand_types else None)
        return (inferred,) if result_count == 1 else (None,) * result_count
    if name in {"arith.cmpi", "arith.cmpf"}:
        inferred = _comparison_result_type(
            operand_types[0] if operand_types else None
        )
        return (inferred,) if result_count == 1 else (None,) * result_count
    if name == "arith.select":
        inferred = operand_types[1] if len(operand_types) >= 3 else None
        if (
            inferred is None
            or operand_types[2] != inferred
            or result_count != 1
        ):
            return (None,) * result_count
        signature = _trailing_type_signature(tail)
        if signature is None:
            return (None,)
        declared = _declared_types(signature)
        expected = (operand_types[0], inferred)
        if None in expected or declared != expected:
            return (None,)
        return (inferred,)

    signature = _trailing_type_signature(tail)
    if signature is None:
        return (None,) * result_count
    arrow = _find_top_level_token(signature, "->", last=True)
    if arrow is not None:
        inferred = _declared_types(signature[arrow + 2 :])
    else:
        conversion = _find_top_level_token(signature, " to ", last=True)
        inferred = (
            _declared_types(signature[conversion + 4 :])
            if conversion is not None
            else _declared_types(signature)
        )
    if len(inferred) != result_count:
        return (None,) * result_count
    return tuple(inferred)


def _return_contract_types(tail: str, operand_count: int) -> tuple[str, ...]:
    if operand_count == 0:
        if _trailing_type_signature(tail) is not None:
            raise TTIRParseError("void tt.return unexpectedly carries a type")
        return ()
    signature = _trailing_type_signature(tail)
    if signature is None:
        raise TTIRParseError("value tt.return is missing its explicit type")
    result = _declared_types(signature)
    if len(result) != operand_count:
        raise TTIRParseError("tt.return SSA/type arity mismatch")
    return result


def _scalar_constant(tail: str) -> float | None:
    """Return a simple scalar or dense-splat arith.constant value."""

    value = tail.split(":", 1)[0].strip()
    if value.startswith("dense<"):
        value = value[len("dense<") :]
        if value.endswith(">"):
            value = value[:-1]
        if any(token in value for token in ("[", "]", ",", "0x")):
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _bounded_normalize_operations(
    operations: list[_Operation],
    result_to_node: dict[str, int],
    terminal_forks: list[_TerminalFork],
) -> tuple[list[_Operation], dict[str, int], list[_TerminalFork]]:
    """Apply the registered bounded rewrites after lexical SSA resolution.

    Gate1 deliberately keeps these false-reject normalizations narrow: neutral
    ``x + 0``, ``1 / sqrt(x)`` versus ``rsqrt(x)``, and Triton's exact
    libdevice-rsqrt spelling versus ``math.rsqrt``.  Applying them to the
    parsed graph is essential: MLIR SSA spellings are region-scoped, so a
    textual whole-function substitution would otherwise depend on whether
    sibling or nested regions happened to reuse the same local name.
    """

    node_by_id = {operation.node_id: operation for operation in operations}
    original_uses: Counter[str] = Counter()
    for operation in operations:
        original_uses.update(operation.operands)
        original_uses.update(operation.region_results)

    def direct_constant(name: str) -> float | None:
        node_id = result_to_node.get(name)
        if node_id is None:
            return None
        producer = node_by_id[node_id]
        if producer.name != "arith.constant":
            return None
        return producer.constant_value

    substitutions: dict[str, str] = {}
    removed_nodes: set[int] = set()
    normalization_constant_nodes: set[int] = set()

    # Rule N0: Triton 3.6 lowers ``tl.extra.libdevice.rsqrt`` to the exact
    # ``tt.extern_elementwise`` symbols below, while replaying the semantic
    # ``rsqrt`` rule produces ``math.rsqrt``.  Only the one-input/one-output
    # signatures whose symbol and result type agree are normalized; all other
    # extern calls remain distinct hard evidence.
    extern_rsqrt_types = {
        "__nv_rsqrtf": "f32",
        "__nv_rsqrt": "f64",
    }
    for operation in operations:
        if (
            operation.name != "tt.extern_elementwise"
            or len(operation.result_names) != 1
            or len(operation.operands) != 1
        ):
            continue
        symbol_match = re.search(
            r'\bsymbol\s*=\s*"(?P<symbol>[^"]+)"',
            operation.local_signature,
        )
        signature_match = re.search(
            r":\((?P<input>[^,(){}]+)\)->(?P<result>[^,(){}]+)$",
            operation.local_signature,
        )
        if symbol_match is None or signature_match is None:
            continue
        expected_type = extern_rsqrt_types.get(symbol_match.group("symbol"))
        input_type = _normalize_space(signature_match.group("input"))
        result_type = _normalize_space(signature_match.group("result"))
        if (
            expected_type is None
            or input_type != expected_type
            or result_type != expected_type
        ):
            continue
        operation.name = "math.rsqrt"
        operation.local_signature = _normalize_space(
            f"math.rsqrt % : {result_type}"
        )
        operation.header_operand_count = 1

    # Rule N1: scalar/tensor floating or integer neutral addition.
    for operation in operations:
        if (
            operation.name not in {"arith.addf", "arith.addi"}
            or len(operation.result_names) != 1
            or len(operation.operands) != 2
        ):
            continue
        left, right = operation.operands
        neutral: str | None = None
        replacement: str | None = None
        if direct_constant(left) == 0.0:
            neutral, replacement = left, right
        elif direct_constant(right) == 0.0:
            neutral, replacement = right, left
        if neutral is None or replacement is None:
            continue
        substitutions[operation.result_names[0]] = replacement
        removed_nodes.add(operation.node_id)
        normalization_constant_nodes.add(result_to_node[neutral])

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in substitutions and name not in seen:
            seen.add(name)
            name = substitutions[name]
        return name

    for operation in operations:
        operation.operands = [resolve(name) for name in operation.operands]
        operation.region_results = [
            resolve(name) for name in operation.region_results
        ]
    for terminal_fork in terminal_forks:
        terminal_fork.predicate = resolve(terminal_fork.predicate)

    # Rule N2: direct 1 / sqrt(x) is represented as the same semantic node as
    # direct rsqrt(x).  This stays deliberately narrower than algebraic
    # simplification and does not chase arbitrary producer chains.
    for operation in operations:
        if (
            operation.node_id in removed_nodes
            or operation.name != "arith.divf"
            or len(operation.result_names) != 1
            or len(operation.operands) != 2
        ):
            continue
        numerator, denominator = operation.operands
        sqrt_node_id = result_to_node.get(denominator)
        if sqrt_node_id is None or direct_constant(numerator) != 1.0:
            continue
        sqrt_operation = node_by_id[sqrt_node_id]
        if (
            sqrt_operation.name != "math.sqrt"
            or len(sqrt_operation.result_names) != 1
            or len(sqrt_operation.operands) != 1
        ):
            continue
        operation.name = "math.rsqrt"
        operation.operands = [sqrt_operation.operands[0]]
        result_type = operation.local_signature.rsplit(":", 1)[-1]
        operation.local_signature = _normalize_space(
            f"math.rsqrt % : {result_type}"
        )
        operation.header_operand_count = 1
        normalization_constant_nodes.add(result_to_node[numerator])
        if original_uses[denominator] == 1:
            removed_nodes.add(sqrt_node_id)

    remaining_uses: Counter[str] = Counter()
    for operation in operations:
        if operation.node_id in removed_nodes:
            continue
        remaining_uses.update(operation.operands)
        remaining_uses.update(operation.region_results)
    for node_id in normalization_constant_nodes:
        operation = node_by_id[node_id]
        if all(remaining_uses[result] == 0 for result in operation.result_names):
            removed_nodes.add(node_id)

    normalized = [
        operation
        for operation in operations
        if operation.node_id not in removed_nodes
    ]
    normalized_result_to_node = {
        result: operation.node_id
        for operation in normalized
        for result in operation.result_names
    }
    return normalized, normalized_result_to_node, terminal_forks


def _cmp_info(name: str, tail: str) -> tuple[str | None, bool]:
    if name not in {"arith.cmpi", "arith.cmpf"}:
        return None, False
    match = re.match(r"(?P<predicate>[A-Za-z]+)\s*,", tail)
    if match is None:
        return None, False
    predicate = match.group("predicate").lower()
    return _CMP_FLIPS.get(predicate, predicate), predicate in _CMP_FLIPS


def _local_signature(name: str, tail: str) -> str:
    erased = _SSA_RE.sub("%", tail)
    cache_modifiers = {
        "tt.load": "ca|cg|cv",
        "tt.store": "wb|cg|cs|wt",
    }.get(name)
    if cache_modifiers is not None:
        # Triton cache modifiers are target cache-policy hints.  They do not
        # change the memory effect, predicate, fill, address/dataflow topology,
        # or J denotation, and the closed J/C/pi artifact does not carry a
        # frontend cache-policy spelling.  Keeping them in the hard fingerprint
        # would therefore make Replay closure mathematically underdetermined.
        # Normalize only Triton 3.6's closed cache-hint values; eviction,
        # masks, fills, types, and all other operation attributes remain
        # observed.
        erased = re.sub(
            rf"\bcacheModifier\s*=\s*(?:{cache_modifiers})(?=\s|:|,|$)",
            "",
            erased,
        )
    predicate, _ = _cmp_info(name, tail)
    if predicate is not None:
        erased = re.sub(r"^[A-Za-z]+(?=\s*,)", predicate, erased)
    # Region opening punctuation does not describe the operation itself.
    erased = re.sub(r"\(\{\s*$", "", erased).strip()
    return _normalize_space(f"{name} {erased}")


def _parse_for_header(
    tail: str,
) -> tuple[list[str], list[str], str, tuple[str, ...]]:
    match = re.match(
        r"(?P<iv>%[A-Za-z0-9_.$-]+)\s*=\s*(?P<lower>%[A-Za-z0-9_.$-]+)"
        r"\s+to\s+(?P<upper>%[A-Za-z0-9_.$-]+)\s+step\s+"
        r"(?P<step>%[A-Za-z0-9_.$-]+)(?P<rest>.*)",
        tail,
    )
    if match is None:
        raise TTIRParseError(f"cannot parse scf.for header: {tail!r}")
    operands = [match.group("lower"), match.group("upper"), match.group("step")]
    block_args = [match.group("iv")]
    rest = match.group("rest").strip()
    if not rest.endswith("{"):
        raise TTIRParseError("scf.for header must end with exactly one body opener")
    rest = rest[:-1].strip()
    colon = _find_top_level_token(rest, ":", last=True)
    if colon is None:
        raise TTIRParseError("scf.for header is missing its induction type")
    prefix = rest[:colon].strip()
    induction_type = _validate_closed_type(
        rest[colon + 1 :], context="scf.for induction type"
    )
    result_types: tuple[str, ...] = ()
    if prefix:
        marker = re.match(r"iter_args\s*\(", prefix)
        if marker is None:
            raise TTIRParseError(
                f"scf.for has unsupported trailing header syntax {prefix!r}"
            )
        opening = prefix.find("(", marker.start())
        closing = _balanced_end(rest, opening, "(", ")")
        items = _split_top_level(prefix[opening + 1 : closing])
        if not items:
            raise TTIRParseError("scf.for iter_args must not be empty")
        for item in items:
            pair = re.fullmatch(
                r"(?P<arg>%[A-Za-z0-9_.$-]+)\s*=\s*(?P<init>%[A-Za-z0-9_.$-]+)",
                item,
            )
            if pair is None:
                raise TTIRParseError(f"cannot parse scf.for iter_arg: {item!r}")
            block_args.append(pair.group("arg"))
            operands.append(pair.group("init"))
        suffix = prefix[closing + 1 :].strip()
        if not suffix.startswith("->"):
            raise TTIRParseError("scf.for iter_args require explicit result types")
        result_text = suffix[2:].strip()
        if not result_text.startswith("("):
            raise TTIRParseError("scf.for result types must use an explicit tuple")
        result_close = _balanced_end(result_text, 0, "(", ")")
        if result_text[result_close + 1 :].strip():
            raise TTIRParseError("scf.for has trailing syntax after result types")
        result_types = _declared_types(result_text[: result_close + 1])
    return operands, block_args, induction_type, result_types


_BLOCK_HEADER_RE = re.compile(
    r"\s*\^(?P<label>[A-Za-z0-9_.$-]+)"
    r"(?:\s*\((?P<args>.*)\))?\s*:\s*(?://.*)?$"
)
_TERMINAL_COND_BR_RE = re.compile(
    r"cf\.cond_br\s+(?P<predicate>%[A-Za-z0-9_.$-]+)\s*,\s*"
    r"\^(?P<true>[A-Za-z0-9_.$-]+)\s*,\s*"
    r"\^(?P<false>[A-Za-z0-9_.$-]+)\s*$"
)


def _parse_block_header(line: str) -> tuple[str, tuple[tuple[str, str], ...]] | None:
    match = _BLOCK_HEADER_RE.fullmatch(line)
    if match is None:
        return None
    arguments: list[tuple[str, str]] = []
    for item in _split_top_level(match.group("args") or ""):
        argument = re.fullmatch(
            r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<type>.+)",
            item,
        )
        if argument is None:
            raise TTIRParseError(f"cannot parse region block argument: {item!r}")
        arguments.append(
            (
                argument.group("name"),
                _validate_closed_type(
                    argument.group("type"),
                    context=f"region block argument {argument.group('name')}",
                ),
            )
        )
    return match.group("label"), tuple(arguments)


def _parse_block_args(line: str) -> list[str]:
    header = _parse_block_header(line)
    if header is None:
        return []
    return [name for name, _ in header[1]]


def _operation_has_effect(operation: _Operation) -> bool:
    """Classify one operation under Gate1's closed effect contract.

    Returning ``False`` is a positive purity claim, not the absence of a match
    in an effect denylist.  Unknown operations are observer-unsupported at
    every nesting level because otherwise reordering two future effectful ops
    could hard-pass before Gate1 learns their semantics.
    """

    if operation.name in {
        "tt.extern_elementwise",
        "tt.elementwise_inline_asm",
    }:
        pure = re.search(r"\bpure=(true|false)\b", operation.local_signature)
        # Missing/unknown conditional-purity metadata is outside the closed
        # observer boundary and is conservatively effectful.
        return pure is None or pure.group(1) != "true"
    if operation.name in _KNOWN_EFFECT_OPS:
        return True
    if operation.name in _KNOWN_PURE_OPS:
        return False
    raise TTIRParseError(
        f"unclassified operation {operation.name} is outside Gate1's closed "
        "operation/effect profile"
    )


def _canonical_python_interface(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not metadata:
        return ()
    # Only callable-signature facts participate in the hard interface. Triton's
    # do_not_specialize metadata controls JIT specialization policy, not the
    # source/replay semantics tested by SCRC.
    metadata = {
        key: value
        for key, value in metadata.items()
        if key
        in {
            "arg_names",
            "constexprs",
            "constexpr_indices",
            "function_name",
            "launch_grid",
            "grid",
            "grid_contract",
        }
    }
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if (
            key == "constexprs"
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
        ):
            normalized[str(key)] = sorted(value, key=str)
        elif isinstance(value, tuple):
            normalized[str(key)] = list(value)
        else:
            normalized[str(key)] = value
    return tuple((key, _stable_json(value)) for key, value in sorted(normalized.items()))


def _canonical_schedule(
    schedule: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...]:
    if not schedule:
        return ()
    return tuple((str(key), _stable_json(value)) for key, value in sorted(schedule.items()))


class TTIRParser:
    """Parse one Triton 3.6 TTIR module into a Gate1 fingerprint."""

    def parse(
        self,
        text: str,
        *,
        python_interface: Mapping[str, Any] | None = None,
        schedule: Mapping[str, Any] | None = None,
    ) -> TTIRFingerprint:
        cleaned = _strip_locations(text)
        # Module type aliases are definitions, while proof-relevant use sites
        # carry only the alias spelling.  Ignoring the definition would let a
        # source/replay pair keep ``!alias`` unchanged while changing its
        # denotation.  Gate1 v7 does not elaborate MLIR aliases, so reject the
        # entire closed profile before parsing any interface or operation type.
        if re.search(
            r"(?m)^\s*![A-Za-z_][A-Za-z0-9_.$-]*\s*=\s*type(?:\s|$)",
            cleaned,
        ) is not None:
            raise TTIRParseError(
                "module-level type aliases are unsupported by Gate1 v7; "
                "proof-relevant types must be self-contained"
            )
        functions = _parse_functions(cleaned)
        function = next(
            item for item in functions.values() if item.visibility == "public"
        )
        operations, result_to_node, leaf_labels, terminal_forks = (
            self._expand_reachable_calls(function, functions)
        )
        operations, result_to_node, terminal_forks = (
            _bounded_normalize_operations(
                operations, result_to_node, terminal_forks
            )
        )
        return self._fingerprint(
            function,
            operations,
            result_to_node,
            leaf_labels,
            terminal_forks,
            python_interface=python_interface,
            schedule=schedule,
        )

    def _parse_operations(
        self, function: _FunctionParts
    ) -> tuple[
        list[_Operation],
        dict[str, int],
        dict[str, str],
        _TerminalFork | None,
    ]:
        operations: list[_Operation] = []
        result_to_node: dict[str, int] = {}
        leaf_labels = {
            name: f"arg[{index}]:{type_text}"
            for index, (name, type_text, _) in enumerate(function.arguments)
        }
        value_types: dict[str, str | None] = {
            name: type_text for name, type_text, _ in function.arguments
        }
        value_fork_arms: dict[str, str | None] = {
            name: None for name, _, _ in function.arguments
        }
        region_stack: list[tuple[int, int]] = []
        region_block_counts: dict[int, int] = {}
        scan_return_types: dict[int, tuple[str, ...]] = {}
        region_terminator_types: dict[int, tuple[str, ...]] = {}
        # One lexical binding map for the function and one for every active
        # operation region.  MLIR SSA names need only be unique within their
        # region; sibling scf.for/tt.reduce/tt.scan bodies may legally reuse
        # spellings such as ``%n_offset``.  Internal qualified keys keep the
        # existing fingerprint graph globally addressable without exposing SSA
        # spelling in the final fingerprint.
        scope_bindings: list[dict[str, str]] = [
            {name: name for name, _, _ in function.arguments}
        ]
        has_cf_operation = re.search(
            r'(?m)^\s*(?:"cf\.[A-Za-z0-9_.-]+"|cf\.[A-Za-z0-9_.-]+)\b',
            function.body,
        ) is not None
        if has_cf_operation and function.results:
            raise TTIRParseError(
                "closed terminal binary fork requires a void tt.func"
            )
        if has_cf_operation and re.search(
            r'(?m)^\s*(?:%[^=]+?\s*=\s*)?(?:"tt\.call"|tt\.call)\b',
            function.body,
        ) is not None:
            raise TTIRParseError(
                "a terminal-fork tt.func cannot contain tt.call; CFG helper "
                "interprocedural expansion is unsupported"
            )
        terminal_fork: _TerminalFork | None = None
        fork_targets: dict[str, str] = {}
        active_fork_arm: str | None = None
        seen_fork_arms: set[str] = set()
        terminated_fork_arms: set[str] = set()
        entry_terminated = False

        def qualified_name(name: str, owner_node: int | None) -> str:
            if owner_node is None:
                return name
            if "#" in name:
                base, suffix = name.rsplit("#", 1)
                return f"{base}$r{owner_node}#{suffix}"
            return f"{name}$r{owner_node}"

        def define(
            name: str,
            *,
            line_number: int,
            bindings: dict[str, str] | None = None,
            owner_node: int | None = None,
        ) -> str:
            selected = scope_bindings[-1] if bindings is None else bindings
            if name in selected:
                raise TTIRParseError(
                    f"line {line_number}: duplicate SSA definition {name}"
                )
            if owner_node is None and bindings is None and region_stack:
                owner_node = region_stack[-1][0]
            qualified = qualified_name(name, owner_node)
            selected[name] = qualified
            value_fork_arms[qualified] = active_fork_arm
            return qualified

        def resolve(name: str) -> str:
            for bindings in reversed(scope_bindings):
                qualified = bindings.get(name)
                if qualified is not None:
                    return qualified
            # Preserve the original spelling so the existing final unresolved
            # check emits a deterministic parse error with useful context.
            return name

        def close_scan_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            operation = operations[owner_id]
            if not closing_text.startswith("})"):
                raise TTIRParseError(
                    f"line {line_number}: tt.scan must close its quoted-generic region"
                )
            suffix = closing_text[2:].strip()
            if not suffix.startswith(":"):
                raise TTIRParseError(
                    f"line {line_number}: tt.scan is missing its explicit function type"
                )
            declared_inputs, declared_results = _functional_types(
                suffix[1:], context="tt.scan"
            )
            input_count = len(operation.operands)
            if input_count < 1:
                raise TTIRParseError("tt.scan must have at least one input")
            if len(operation.result_names) != input_count:
                raise TTIRParseError(
                    "tt.scan must have the same number of SSA results as inputs"
                )
            if len(declared_inputs) != input_count:
                raise TTIRParseError(
                    "tt.scan explicit input type arity does not match its operands"
                )
            if len(declared_results) != input_count:
                raise TTIRParseError(
                    "tt.scan explicit result type arity does not match its inputs"
                )

            actual_inputs = tuple(value_types.get(name) for name in operation.operands)
            if any(type_text is None for type_text in actual_inputs):
                raise TTIRParseError("tt.scan input SSA type cannot be established")
            normalized_actual_inputs = tuple(
                _normalize_space(type_text)
                for type_text in actual_inputs
                if type_text is not None
            )
            if declared_inputs != normalized_actual_inputs:
                raise TTIRParseError(
                    "tt.scan explicit input types do not match actual SSA types"
                )
            if declared_results != declared_inputs:
                raise TTIRParseError(
                    "tt.scan result type must match the corresponding input type"
                )

            tensor_contracts = tuple(
                _ranked_tensor_contract(type_text)
                for type_text in declared_inputs
            )
            if any(contract is None for contract in tensor_contracts):
                raise TTIRParseError("tt.scan inputs and results must be ranked tensors")
            checked_contracts = tuple(
                contract for contract in tensor_contracts if contract is not None
            )
            first_shape, _, first_encoding = checked_contracts[0]
            if any(
                shape != first_shape or encoding != first_encoding
                for shape, _, encoding in checked_contracts[1:]
            ):
                raise TTIRParseError(
                    "tt.scan inputs and results must have one common shape and encoding"
                )
            if operation.axis is None:
                raise TTIRParseError("tt.scan is missing its axis property")
            if not 0 <= operation.axis < len(first_shape):
                raise TTIRParseError(
                    f"tt.scan axis {operation.axis} is outside rank {len(first_shape)}"
                )
            if operation.reverse is None:
                raise TTIRParseError("tt.scan is missing its reverse property")

            if region_block_counts.get(owner_id, 0) != 1:
                raise TTIRParseError("tt.scan must contain exactly one combine block")
            expected_block_types = tuple(
                [contract[1] for contract in checked_contracts] * 2
            )
            actual_block_types = tuple(
                value_types.get(name) for name in operation.region_block_args
            )
            if len(actual_block_types) != 2 * input_count:
                raise TTIRParseError(
                    "tt.scan combine block must take exactly 2N ordered arguments"
                )
            if actual_block_types != expected_block_types:
                raise TTIRParseError(
                    "tt.scan combine block argument types must be ordered lhs[0:N], "
                    "rhs[0:N]"
                )

            declared_returns = scan_return_types.get(owner_id)
            if declared_returns is None:
                raise TTIRParseError(
                    "tt.scan combine block must end with exactly one tt.scan.return"
                )
            if len(operation.region_results) != input_count:
                raise TTIRParseError(
                    "tt.scan.return must return exactly one value per scan input"
                )
            actual_return_types = tuple(
                value_types.get(name) for name in operation.region_results
            )
            expected_return_types = tuple(
                contract[1] for contract in checked_contracts
            )
            if any(type_text is None for type_text in actual_return_types):
                raise TTIRParseError(
                    "tt.scan.return operand SSA type cannot be established"
                )
            if declared_returns != expected_return_types:
                raise TTIRParseError(
                    "tt.scan.return explicit types do not match scan element types"
                )
            if actual_return_types != expected_return_types:
                raise TTIRParseError(
                    "tt.scan.return actual SSA types do not match scan element types"
                )

            for result, type_text in zip(
                operation.result_names, declared_results, strict=True
            ):
                value_types[result] = type_text
            operation.local_signature = _normalize_space(
                "tt.scan "
                f"axis={operation.axis}:i32 "
                f"reverse={'true' if operation.reverse else 'false'} "
                f":({','.join(declared_inputs)})->"
                f"({','.join(declared_results)})"
            )

        def close_reduce_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            operation = operations[owner_id]
            if not closing_text.startswith("})"):
                raise TTIRParseError(
                    f"line {line_number}: tt.reduce must close its "
                    "quoted-generic region"
                )
            suffix = closing_text[2:].strip()
            if not suffix.startswith(":"):
                raise TTIRParseError(
                    f"line {line_number}: tt.reduce is missing its explicit "
                    "function type"
                )
            declared_inputs, declared_results = _functional_types(
                suffix[1:], context="tt.reduce"
            )
            input_count = len(operation.operands)
            if input_count < 1 or len(operation.result_names) != input_count:
                raise TTIRParseError(
                    "tt.reduce must have one SSA result per input"
                )
            if len(declared_inputs) != input_count:
                raise TTIRParseError(
                    "tt.reduce explicit input type arity does not match its operands"
                )
            if len(declared_results) != input_count:
                raise TTIRParseError(
                    "tt.reduce explicit result type arity does not match its inputs"
                )
            actual_inputs = tuple(value_types.get(name) for name in operation.operands)
            if any(type_text is None for type_text in actual_inputs):
                raise TTIRParseError("tt.reduce input SSA type cannot be established")
            if declared_inputs != tuple(actual_inputs):
                raise TTIRParseError(
                    "tt.reduce explicit input types do not match actual SSA types"
                )
            tensor_contracts = tuple(
                _ranked_tensor_contract(type_text) for type_text in declared_inputs
            )
            if any(contract is None for contract in tensor_contracts):
                raise TTIRParseError("tt.reduce inputs must be ranked tensors")
            contracts = tuple(
                contract for contract in tensor_contracts if contract is not None
            )
            first_shape, _, first_encoding = contracts[0]
            if any(
                shape != first_shape or encoding != first_encoding
                for shape, _, encoding in contracts[1:]
            ):
                raise TTIRParseError(
                    "tt.reduce inputs must have one common shape and encoding"
                )
            if operation.axis is None or not 0 <= operation.axis < len(first_shape):
                raise TTIRParseError(
                    f"tt.reduce axis {operation.axis} is outside rank {len(first_shape)}"
                )
            element_types = tuple(contract[1] for contract in contracts)
            expected_result_types = tuple(
                (
                    contract[1]
                    if len(contract[0]) == 1
                    else "tensor<"
                    + "x".join(
                        [
                            *(
                                extent
                                for index, extent in enumerate(contract[0])
                                if index != operation.axis
                            ),
                            contract[1],
                        ]
                    )
                    + ">"
                )
                for contract in contracts
            )
            if declared_results != expected_result_types:
                raise TTIRParseError(
                    "tt.reduce result types must remove exactly its reduction axis"
                )
            if region_block_counts.get(owner_id, 0) != 1:
                raise TTIRParseError("tt.reduce must contain exactly one combine block")
            block_types = tuple(
                value_types.get(name) for name in operation.region_block_args
            )
            if block_types != tuple([*element_types, *element_types]):
                raise TTIRParseError(
                    "tt.reduce combine block arguments must be ordered "
                    "lhs[0:N], rhs[0:N]"
                )
            terminator_types = region_terminator_types.get(owner_id)
            if terminator_types is None:
                raise TTIRParseError(
                    "tt.reduce combine block must end with exactly one "
                    "tt.reduce.return"
                )
            if len(operation.region_results) != input_count:
                raise TTIRParseError(
                    "tt.reduce.return must return exactly one value per input"
                )
            if terminator_types != element_types:
                raise TTIRParseError(
                    "tt.reduce.return explicit types do not match reduce "
                    "element types"
                )
            actual_return_types = tuple(
                value_types.get(name) for name in operation.region_results
            )
            if actual_return_types != element_types:
                raise TTIRParseError(
                    "tt.reduce.return actual SSA types do not match reduce "
                    "element types"
                )
            for result, type_text in zip(
                operation.result_names, declared_results, strict=True
            ):
                value_types[result] = type_text
            operation.local_signature = _normalize_space(
                "tt.reduce "
                f"axis={operation.axis}:i32 "
                f":({','.join(declared_inputs)})->"
                f"({','.join(declared_results)})"
            )

        def close_for_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            closing = closing_text.strip()
            if not closing.startswith("}"):
                raise TTIRParseError(
                    f"line {line_number}: scf.for has malformed closing syntax"
                )
            operation = operations[owner_id]
            suffix = closing[1:].strip()
            if suffix:
                if not suffix.startswith("{"):
                    raise TTIRParseError(
                        f"line {line_number}: scf.for has malformed closing syntax"
                    )
                attr_close = _balanced_end(suffix, 0, "{", "}")
                if suffix[attr_close + 1 :].strip():
                    raise TTIRParseError(
                        f"line {line_number}: scf.for has trailing closing syntax"
                    )
                close_attrs = _validate_attribute_dictionary(
                    suffix[: attr_close + 1],
                    context="scf.for closing attributes",
                )
                if any(
                    name != "tt.num_stages"
                    or re.fullmatch(r"[1-9][0-9]*:i32", value) is None
                    for name, value in close_attrs
                ):
                    raise TTIRParseError(
                        "scf.for has unsupported closing attributes"
                    )
                operation.local_signature = _normalize_space(
                    operation.local_signature
                    + " close_attrs={"
                    + ",".join(f"{name}={value}" for name, value in close_attrs)
                    + "}"
                )
            if region_block_counts.get(owner_id, 0) != 1:
                raise TTIRParseError("scf.for must contain exactly one body block")
            init_count = max(0, len(operation.operands) - 3)
            if owner_id not in region_terminator_types and init_count:
                raise TTIRParseError(
                    "scf.for body must end with exactly one scf.yield"
                )
            init_types = tuple(
                value_types.get(name) for name in operation.operands[3:]
            )
            if any(type_text is None for type_text in init_types):
                raise TTIRParseError(
                    "scf.for iter_arg input SSA type cannot be established"
                )
            if len(operation.region_block_args) != 1 + len(init_types):
                raise TTIRParseError(
                    "scf.for block arguments must contain one induction value "
                    "and one value per iter_arg"
                )
            block_types = tuple(
                value_types.get(name) for name in operation.region_block_args[1:]
            )
            result_types = tuple(
                value_types.get(name) for name in operation.result_names
            )
            yield_types = region_terminator_types.get(owner_id, ())
            actual_yield_types = tuple(
                value_types.get(name) for name in operation.region_results
            )
            if not (
                block_types == init_types
                and result_types == init_types
                and yield_types == init_types
                and actual_yield_types == init_types
                and len(operation.region_results) == len(init_types)
            ):
                raise TTIRParseError(
                    "scf.for iter_args, results, block arguments, and yield "
                    "types must close positionally"
                )

        def close_region(
            owner_id: int, closing_text: str, *, line_number: int
        ) -> None:
            owner = operations[owner_id]
            if owner.name == "tt.scan":
                close_scan_region(
                    owner_id, closing_text, line_number=line_number
                )
            elif owner.name == "tt.reduce":
                close_reduce_region(
                    owner_id, closing_text, line_number=line_number
                )
            elif owner.name == "scf.for":
                close_for_region(
                    owner_id, closing_text, line_number=line_number
                )
            else:  # pragma: no cover - region owner set is closed above
                raise TTIRParseError(f"unsupported region owner {owner.name}")

        for line_number, line in enumerate(function.body.splitlines(), 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if not stripped or stripped.startswith("//"):
                continue
            cond_br = _TERMINAL_COND_BR_RE.fullmatch(stripped)
            if cond_br is not None:
                if not has_cf_operation or terminal_fork is not None:
                    raise TTIRParseError(
                        f"line {line_number}: terminal binary fork must have "
                        "exactly one cf.cond_br"
                    )
                if region_stack or active_fork_arm is not None:
                    raise TTIRParseError(
                        f"line {line_number}: cf.cond_br must terminate the "
                        "function entry block"
                    )
                true_label = cond_br.group("true")
                false_label = cond_br.group("false")
                if true_label == false_label:
                    raise TTIRParseError(
                        "terminal binary fork requires two distinct successor arms"
                    )
                predicate = resolve(cond_br.group("predicate"))
                if value_types.get(predicate) != "i1":
                    raise TTIRParseError(
                        "terminal binary fork predicate must be a scalar i1 SSA value"
                    )
                terminal_fork = _TerminalFork(
                    0, predicate, (), len(operations)
                )
                fork_targets = {
                    true_label: "true",
                    false_label: "false",
                }
                entry_terminated = True
                continue
            if stripped.startswith(('cf.', '"cf.')):
                raise TTIRParseError(
                    f"line {line_number}: only the exact operand-free "
                    "cf.cond_br terminal binary-fork shell is supported"
                )
            if stripped.startswith(("})", "}")):
                if not region_stack:
                    raise TTIRParseError(
                        f"line {line_number}: unexpected structured-region close"
                    )
                first_closed = True
                closed_any = False
                while region_stack and indent <= region_stack[-1][1]:
                    owner_id = region_stack[-1][0]
                    close_region(
                        owner_id,
                        stripped if first_closed else "}",
                        line_number=line_number,
                    )
                    region_stack.pop()
                    scope_bindings.pop()
                    first_closed = False
                    closed_any = True
                if not closed_any:
                    raise TTIRParseError(
                        f"line {line_number}: region close indentation does not "
                        "match an active owner"
                    )
                continue

            block_header = _parse_block_header(line)
            if block_header is not None:
                if not region_stack:
                    label, block_arguments = block_header
                    if terminal_fork is None or not entry_terminated:
                        raise TTIRParseError(
                            f"line {line_number}: top-level block label "
                            f"^{label} is unsupported; Gate1 TTIR v7 only "
                            "accepts the closed terminal binary-fork profile"
                        )
                    role = fork_targets.get(label)
                    if role is None:
                        raise TTIRParseError(
                            f"line {line_number}: unexpected terminal-fork "
                            f"block ^{label}"
                        )
                    if block_arguments:
                        raise TTIRParseError(
                            "terminal-fork successor blocks cannot take block "
                            "arguments"
                        )
                    if role in seen_fork_arms:
                        raise TTIRParseError(
                            f"terminal-fork {role} arm is declared more than once"
                        )
                    if (
                        active_fork_arm is not None
                        and active_fork_arm not in terminated_fork_arms
                    ):
                        raise TTIRParseError(
                            f"terminal-fork {active_fork_arm} arm does not end "
                            "in void tt.return"
                        )
                    active_fork_arm = role
                    seen_fork_arms.add(role)
                    continue
                block_args = [name for name, _ in block_header[1]]
                block_arg_types = [type_text for _, type_text in block_header[1]]
                parent_id = region_stack[-1][0]
                parent = operations[parent_id]
                if region_block_counts.get(parent_id, 0) >= 1:
                    raise TTIRParseError(
                        f"line {line_number}: structured region "
                        f"{parent.name} has more than one block; Gate1 TTIR "
                        "v7 supports only single-block structured regions"
                    )
                region_block_counts[parent_id] = 1
                qualified_args = [
                    define(name, line_number=line_number)
                    for name in block_args
                ]
                for name, type_text in zip(
                    qualified_args, block_arg_types, strict=True
                ):
                    value_types[name] = type_text
                parent.region_block_args.extend(qualified_args)
                path = "/".join(operations[node].name for node, _ in region_stack)
                for index, name in enumerate(qualified_args):
                    leaf_labels[name] = f"region[{path}].arg[{index}]"
                continue

            if entry_terminated and active_fork_arm is None:
                raise TTIRParseError(
                    f"line {line_number}: only the two cf.cond_br target "
                    "labels may follow the entry terminator"
                )
            if active_fork_arm in terminated_fork_arms:
                raise TTIRParseError(
                    f"line {line_number}: operation after terminal-fork "
                    f"{active_fork_arm} arm tt.return"
                )

            if region_stack and region_stack[-1][0] in region_terminator_types:
                owner = operations[region_stack[-1][0]]
                raise TTIRParseError(
                    f"line {line_number}: operation after {owner.name} terminator"
                )

            head = _operation_head(line)
            if head is None:
                raise TTIRParseError(
                    f"line {line_number}: cannot parse non-empty TTIR body "
                    f"line: {stripped!r}"
                )
            results, name, tail, quoted = head
            if has_cf_operation and name == "tt.call":
                raise TTIRParseError(
                    "a terminal-fork tt.func cannot contain tt.call; CFG "
                    "helper interprocedural expansion is unsupported"
                )
            if has_cf_operation and name == "tt.return":
                if (
                    terminal_fork is None
                    or region_stack
                    or active_fork_arm is None
                ):
                    raise TTIRParseError(
                        "terminal binary fork permits tt.return only as the "
                        "top-level terminator of an arm"
                    )
            if (
                name not in _REGION_OWNERS
                and stripped.endswith("{")
            ):
                raise TTIRParseError(
                    f"line {line_number}: unsupported region-bearing TTIR "
                    f"operation {name!r}; Gate1 refuses to flatten an unknown "
                    "SSA scope"
                )
            if quoted and name in _TERMINATORS:
                raise TTIRParseError(
                    f"{name} must use Triton 3.6 custom terminator syntax"
                )
            generic_contract: tuple[
                tuple[str, ...], tuple[str, ...], tuple[str, ...]
            ] | None = None
            if quoted and name not in _REGION_OWNERS:
                generic_contract = _validate_quoted_generic_shell(
                    name,
                    tail,
                    result_count=len(results),
                )
            raw_operands = (
                list(generic_contract[0])
                if generic_contract is not None
                else _SSA_RE.findall(tail)
            )
            raw_block_args: list[str] = []
            scan_axis: int | None = None
            scan_reverse: bool | None = None
            for_induction_type: str | None = None
            for_result_types: tuple[str, ...] = ()
            parsed_terminator_types: tuple[str, ...] = ()
            if name == "scf.for":
                (
                    raw_operands,
                    raw_block_args,
                    for_induction_type,
                    for_result_types,
                ) = _parse_for_header(tail)
            elif name == "tt.reduce":
                if not quoted:
                    raise TTIRParseError(
                        "tt.reduce outer operation must use Triton 3.6 "
                        "quoted-generic syntax"
                    )
                raw_operands, scan_axis = _parse_reduce_header(tail)
                if len(results) != len(raw_operands):
                    raise TTIRParseError(
                        "tt.reduce must have the same number of SSA results as inputs"
                    )
            elif name == "tt.scan":
                if not quoted:
                    raise TTIRParseError(
                        "tt.scan outer operation must use Triton 3.6 quoted-generic syntax"
                    )
                raw_operands, scan_axis, scan_reverse = _parse_scan_header(tail)
                if len(results) != len(raw_operands):
                    raise TTIRParseError(
                        "tt.scan must have the same number of SSA results as inputs"
                    )
            elif name == "tt.scan.return":
                raw_operands, parsed_terminator_types = _parse_scan_return(tail)
            elif name == "tt.reduce.return":
                raw_operands, parsed_terminator_types = _parse_typed_terminator(
                    name, tail, allow_empty=False
                )
            elif name == "scf.yield":
                raw_operands, parsed_terminator_types = _parse_typed_terminator(
                    name, tail, allow_empty=True
                )
            elif name == "tt.return":
                raw_operands, parsed_terminator_types = _parse_typed_terminator(
                    name, tail, allow_empty=True
                )
            callee: str | None = None
            call_operand_types: tuple[str, ...] = ()
            call_result_types: tuple[str, ...] = ()
            if name == "tt.call":
                callee, call_operand_types, call_result_types = (
                    _parse_call_contract(tail)
                )
            operands = [resolve(operand) for operand in raw_operands]
            if active_fork_arm is not None:
                for operand in operands:
                    if operand not in value_fork_arms:
                        raise TTIRParseError(
                            f"line {line_number}: SSA value {operand} does "
                            "not dominate its terminal-fork arm use"
                        )
                    defining_arm = value_fork_arms.get(operand)
                    if defining_arm not in {None, active_fork_arm}:
                        raise TTIRParseError(
                            f"line {line_number}: terminal-fork "
                            f"{active_fork_arm} arm uses SSA value from its "
                            f"{defining_arm} sibling"
                        )
            operand_types = tuple(value_types.get(operand) for operand in operands)
            if generic_contract is not None:
                declared_generic_inputs = generic_contract[1]
                if len(operand_types) != len(declared_generic_inputs):
                    raise TTIRParseError(
                        f'quoted generic operation "{name}" operand/type arity mismatch'
                    )
                for index, (actual, declared) in enumerate(
                    zip(operand_types, declared_generic_inputs, strict=True)
                ):
                    if actual is None:
                        raise TTIRParseError(
                            f'quoted generic operation "{name}" operand {index} '
                            "actual SSA type cannot be established"
                        )
                    if _normalize_space(actual) != declared:
                        raise TTIRParseError(
                            f'quoted generic operation "{name}" operand {index} '
                            f"actual SSA type {actual} does not match {declared}"
                        )
            if name == "tt.call":
                if len(operand_types) != len(call_operand_types):
                    raise TTIRParseError("tt.call operand SSA/type arity mismatch")
                for index, (actual, declared) in enumerate(
                    zip(operand_types, call_operand_types, strict=True)
                ):
                    if actual is None:
                        raise TTIRParseError(
                            f"tt.call operand {index} actual SSA type cannot "
                            "be established"
                        )
                    if actual != declared:
                        raise TTIRParseError(
                            f"tt.call operand {index} actual SSA type {actual} "
                            f"does not match its explicit call type {declared}"
                        )
            if name == "scf.for":
                assert for_induction_type is not None
                if any(
                    actual is None or _normalize_space(actual) != for_induction_type
                    for actual in operand_types[:3]
                ):
                    raise TTIRParseError(
                        "scf.for lower/upper/step SSA types must match its "
                        "explicit induction type"
                    )
                init_types = tuple(operand_types[3:])
                if any(type_text is None for type_text in init_types):
                    raise TTIRParseError(
                        "scf.for iter_arg input SSA type cannot be established"
                    )
                if tuple(init_types) != for_result_types:
                    raise TTIRParseError(
                        "scf.for iter_arg input types must match its explicit "
                        "result types"
                    )
                if len(results) != len(for_result_types):
                    raise TTIRParseError(
                        "scf.for SSA result arity must match its iter_args"
                    )
            if name == "tt.return":
                declared_returns = parsed_terminator_types
                if declared_returns != function.results:
                    raise TTIRParseError(
                        f"tt.return explicit types {declared_returns} do not "
                        f"match tt.func @{function.name} results "
                        f"{function.results}"
                    )
                for index, (actual, declared) in enumerate(
                    zip(operand_types, declared_returns, strict=True)
                ):
                    if actual is None:
                        raise TTIRParseError(
                            f"tt.return operand {index} actual SSA type cannot "
                            "be established"
                        )
                    if actual != declared:
                        raise TTIRParseError(
                            f"tt.return operand {index} actual SSA type "
                            f"{actual} does not match its explicit return "
                            f"type {declared}"
                        )
            qualified_results = tuple(
                define(result, line_number=line_number) for result in results
            )
            inferred_result_types = _operation_result_types(
                name,
                tail,
                len(qualified_results),
                operand_types,
                call_result_types=call_result_types,
            )
            for result, type_text in zip(
                qualified_results, inferred_result_types, strict=True
            ):
                value_types[result] = type_text
            axis_match = re.search(r"\baxis\s*=\s*(-?[0-9]+)", tail)
            operation = _Operation(
                node_id=len(operations),
                name=name,
                result_names=qualified_results,
                operands=list(operands),
                local_signature=_local_signature(name, tail),
                context=tuple(node for node, _ in region_stack),
                axis=(
                    scan_axis
                    if name == "tt.scan"
                    else int(axis_match.group(1)) if axis_match else None
                ),
                reverse=scan_reverse,
                cmp_predicate=_cmp_info(name, tail)[0],
                cmp_flipped=_cmp_info(name, tail)[1],
                header_operand_count=len(operands),
                region_block_args=[],
                constant_value=(
                    _scalar_constant(tail)
                    if name == "arith.constant"
                    else None
                ),
                callee=callee,
                call_operand_types=call_operand_types,
                call_result_types=call_result_types,
                terminator_types=parsed_terminator_types,
                fork_id=(
                    terminal_fork.fork_id
                    if active_fork_arm is not None
                    else None
                ),
                fork_arm=active_fork_arm,
            )
            operations.append(operation)
            for result in qualified_results:
                if result in result_to_node or result in leaf_labels:
                    raise TTIRParseError(
                        f"line {line_number}: duplicate internal SSA key {result}"
                    )
                result_to_node[result] = operation.node_id

            if name in _REGION_TERMINATORS:
                if not region_stack:
                    raise TTIRParseError(
                        f"line {line_number}: {name} appears outside a structured region"
                    )
                parent_id = region_stack[-1][0]
                parent = operations[parent_id]
                expected = _EXPECTED_REGION_TERMINATOR[parent.name]
                if name != expected:
                    raise TTIRParseError(
                        f"line {line_number}: {parent.name} requires {expected}, got {name}"
                    )
                if results:
                    raise TTIRParseError(
                        f"line {line_number}: region terminator {name} cannot define SSA results"
                    )
                if parent_id in region_terminator_types:
                    raise TTIRParseError(
                        f"line {line_number}: {parent.name} has more than one "
                        f"{expected}"
                    )
                region_terminator_types[parent_id] = parsed_terminator_types
                if parent.name == "tt.scan":
                    scan_return_types[parent_id] = parsed_terminator_types
                parent.region_results.extend(operands)

            if name == "tt.return" and terminal_fork is not None:
                assert active_fork_arm is not None
                if results or operands or parsed_terminator_types:
                    raise TTIRParseError(
                        "terminal-fork arms must end in an empty void tt.return"
                    )
                terminated_fork_arms.add(active_fork_arm)

            if name in _REGION_OWNERS:
                path = "/".join(
                    [*(operations[node].name for node, _ in region_stack), name]
                )
                child_bindings: dict[str, str] = {}
                qualified_block_args = [
                    define(
                        block_name,
                        line_number=line_number,
                        bindings=child_bindings,
                        owner_node=operation.node_id,
                    )
                    for block_name in raw_block_args
                ]
                raw_block_arg_types = (
                    [
                        operand_types[0] if operand_types else None,
                        *operand_types[3:],
                    ]
                    if name == "scf.for"
                    else []
                )
                if len(raw_block_arg_types) != len(qualified_block_args):
                    raise TTIRParseError(
                        "scf.for block argument/type arity mismatch"
                    )
                for block_name, type_text in zip(
                    qualified_block_args, raw_block_arg_types, strict=True
                ):
                    value_types[block_name] = type_text
                operation.region_block_args.extend(qualified_block_args)
                for index, block_name in enumerate(qualified_block_args):
                    leaf_labels[block_name] = f"region[{path}].arg[{index}]"
                region_stack.append((operation.node_id, indent))
                scope_bindings.append(child_bindings)
                # Custom ``scf.for`` assembly declares its one body block via
                # the induction variable/iter_args in the operation header.
                # ``tt.reduce``/``tt.scan`` declare their block on the
                # following ^bb line.
                region_block_counts[operation.node_id] = (
                    1 if raw_block_args else 0
                )

        if region_stack:
            raise TTIRParseError("unterminated operation region")
        if has_cf_operation:
            if terminal_fork is None:
                raise TTIRParseError(
                    "reachable tt.func uses control flow outside the closed "
                    "terminal binary-fork profile"
                )
            if seen_fork_arms != {"true", "false"}:
                raise TTIRParseError(
                    "terminal binary fork must declare exactly its true and "
                    "false successor blocks"
                )
            if terminated_fork_arms != {"true", "false"}:
                raise TTIRParseError(
                    "each terminal-fork arm must end in an empty void tt.return"
                )
        for operation in operations:
            for operand in [*operation.operands, *operation.region_results]:
                if operand not in result_to_node and operand not in leaf_labels:
                    raise TTIRParseError(
                        f"unresolved SSA value {operand!r} used by {operation.name}"
                    )
        return operations, result_to_node, leaf_labels, terminal_fork

    def _expand_reachable_calls(
        self,
        public: _FunctionParts,
        functions: Mapping[str, _FunctionParts],
    ) -> tuple[
        list[_Operation],
        dict[str, int],
        dict[str, str],
        list[_TerminalFork],
    ]:
        """Inline the reachable acyclic ``tt.call`` graph into one SSA graph.

        This is a fingerprint-layer canonicalization, not a compiler rewrite:
        both Source and Replay pass through it and the public function ABI is
        kept verbatim.  Private function names, private argument attributes,
        and the frontend's choice to outline or inline are therefore erased,
        while every reachable operation and memory effect remains observable.

        The current TTIR observer models structured
        ``scf.for``/``tt.reduce``/``tt.scan`` regions and one closed terminal
        binary-fork shape.  The latter is expanded as a synthetic conditional
        control member whose true/false roles remain distinct; its arms are
        never flattened into an unconditional sequence.  Arbitrary CFGs,
        joins, phi/block arguments, backedges, and calls from a CFG helper
        remain unsupported.  Unreachable helpers are deliberately never
        parsed and cannot perturb the fingerprint.
        """

        parsed: dict[
            str,
            tuple[
                list[_Operation],
                dict[str, int],
                dict[str, str],
                _TerminalFork | None,
            ],
        ] = {}
        expanded: list[_Operation] = []
        expanded_results: dict[str, int] = {}
        expanded_terminal_forks: list[_TerminalFork] = []
        leaf_labels = {
            name: f"arg[{index}]:{type_text}"
            for index, (name, type_text, _) in enumerate(public.arguments)
        }
        value_serial = 0
        fork_serial = 0

        def fresh_value() -> str:
            nonlocal value_serial
            value = f"%__gate1_ip{value_serial}"
            value_serial += 1
            return value

        def parse_reachable(
            function: _FunctionParts,
        ) -> tuple[
            list[_Operation],
            dict[str, int],
            dict[str, str],
            _TerminalFork | None,
        ]:
            cached = parsed.get(function.name)
            if cached is not None:
                return cached
            _validate_function_metadata_boundary(function)
            cached = self._parse_operations(function)
            parsed[function.name] = cached
            return cached

        def mapped_value(value_map: Mapping[str, str], value: str) -> str:
            try:
                return value_map[value]
            except KeyError as error:
                raise TTIRParseError(
                    f"interprocedural expansion lost SSA value {value!r}"
                ) from error

        def validate_call(
            operation: _Operation,
            caller: _FunctionParts,
        ) -> _FunctionParts:
            if operation.callee is None:
                raise TTIRParseError(
                    f"tt.call in @{caller.name} has no statically known callee"
                )
            callee = functions.get(operation.callee)
            if callee is None:
                raise TTIRParseError(
                    f"tt.call in @{caller.name} references unknown callee "
                    f"@{operation.callee}"
                )
            declared_operands = tuple(
                type_text for _, type_text, _ in callee.arguments
            )
            if len(operation.operands) != len(operation.call_operand_types):
                raise TTIRParseError(
                    f"tt.call @{callee.name} operand SSA/type arity mismatch: "
                    f"{len(operation.operands)} versus "
                    f"{len(operation.call_operand_types)}"
                )
            if len(operation.call_operand_types) != len(declared_operands):
                raise TTIRParseError(
                    f"tt.call @{callee.name} argument arity does not match its "
                    "tt.func signature"
                )
            if operation.call_operand_types != declared_operands:
                raise TTIRParseError(
                    f"tt.call @{callee.name} operand types do not match its "
                    "tt.func signature"
                )
            if len(operation.result_names) != len(operation.call_result_types):
                raise TTIRParseError(
                    f"tt.call @{callee.name} result SSA/type arity mismatch: "
                    f"{len(operation.result_names)} versus "
                    f"{len(operation.call_result_types)}"
                )
            if len(operation.call_result_types) != len(callee.results):
                raise TTIRParseError(
                    f"tt.call @{callee.name} result arity does not match its "
                    "tt.func signature"
                )
            if operation.call_result_types != callee.results:
                raise TTIRParseError(
                    f"tt.call @{callee.name} result types do not match its "
                    "tt.func signature"
                )
            return callee

        def emit_function(
            function: _FunctionParts,
            actual_arguments: tuple[str, ...],
            context_prefix: tuple[int, ...],
            stack: tuple[str, ...],
            *,
            preserve_return: bool,
        ) -> tuple[str, ...]:
            nonlocal fork_serial
            if function.name in stack:
                cycle = " -> ".join((*stack, function.name))
                raise TTIRParseError(
                    f"recursive tt.call graph is unsupported: {cycle}"
                )
            if len(actual_arguments) != len(function.arguments):
                raise TTIRParseError(
                    f"tt.func @{function.name} argument arity mismatch: "
                    f"expected {len(function.arguments)}, got "
                    f"{len(actual_arguments)}"
                )

            local_operations, _, _, local_terminal_fork = parse_reachable(
                function
            )
            expanded_fork_id: int | None = None
            if local_terminal_fork is not None:
                expanded_fork_id = fork_serial
                fork_serial += 1
            value_map = {
                name: actual
                for (name, _, _), actual in zip(
                    function.arguments, actual_arguments, strict=True
                )
            }
            context_map: dict[int, int] = {}
            deferred_region_results: list[tuple[_Operation, tuple[str, ...]]] = []
            returned_values: list[tuple[str, ...]] = []
            saw_top_level_return = False
            expanded_fork_sequence_index: int | None = None

            for operation in local_operations:
                if (
                    operation.fork_id is not None
                    and expanded_fork_sequence_index is None
                ):
                    expanded_fork_sequence_index = len(expanded)
                if saw_top_level_return and local_terminal_fork is None:
                    if operation.name == "tt.return":
                        raise TTIRParseError(
                            f"tt.func @{function.name} has more than one "
                            "top-level tt.return"
                        )
                    raise TTIRParseError(
                        f"tt.func @{function.name} has an operation after its "
                        "top-level tt.return"
                    )
                try:
                    mapped_context = context_prefix + tuple(
                        context_map[parent] for parent in operation.context
                    )
                except KeyError as error:
                    raise TTIRParseError(
                        f"tt.func @{function.name} has a region whose owner "
                        "does not dominate it"
                    ) from error

                if operation.name == "tt.call":
                    callee = validate_call(operation, function)
                    actuals = tuple(
                        mapped_value(value_map, operand)
                        for operand in operation.operands
                    )
                    returned = emit_function(
                        callee,
                        actuals,
                        mapped_context,
                        (*stack, function.name),
                        preserve_return=False,
                    )
                    if len(returned) != len(operation.result_names):
                        raise TTIRParseError(
                            f"tt.call @{callee.name} returned {len(returned)} "
                            f"values for {len(operation.result_names)} results"
                        )
                    value_map.update(
                        zip(operation.result_names, returned, strict=True)
                    )
                    continue

                mapped_operands = [
                    mapped_value(value_map, operand)
                    for operand in operation.operands
                ]
                if operation.name == "tt.return":
                    if operation.context:
                        raise TTIRParseError(
                            f"tt.return in @{function.name} appears inside a "
                            "structured operation region"
                        )
                    if len(mapped_operands) != len(function.results):
                        raise TTIRParseError(
                            f"tt.return in @{function.name} has "
                            f"{len(mapped_operands)} values, expected "
                            f"{len(function.results)}"
                        )
                    returned_values.append(tuple(mapped_operands))
                    if local_terminal_fork is None:
                        saw_top_level_return = True
                    if not preserve_return:
                        continue

                new_results = tuple(fresh_value() for _ in operation.result_names)
                new_block_args = [
                    fresh_value() for _ in operation.region_block_args
                ]
                clone = _Operation(
                    node_id=len(expanded),
                    name=operation.name,
                    result_names=new_results,
                    operands=mapped_operands,
                    local_signature=operation.local_signature,
                    context=mapped_context,
                    axis=operation.axis,
                    reverse=operation.reverse,
                    cmp_predicate=operation.cmp_predicate,
                    cmp_flipped=operation.cmp_flipped,
                    header_operand_count=operation.header_operand_count,
                    region_results=[],
                    region_block_args=new_block_args,
                    constant_value=operation.constant_value,
                    callee=None,
                    terminator_types=operation.terminator_types,
                    fork_id=(
                        expanded_fork_id
                        if operation.fork_id is not None
                        else None
                    ),
                    fork_arm=operation.fork_arm,
                )
                expanded.append(clone)
                context_map[operation.node_id] = clone.node_id
                for old, new in zip(
                    operation.result_names, new_results, strict=True
                ):
                    value_map[old] = new
                    expanded_results[new] = clone.node_id
                for old, new in zip(
                    operation.region_block_args, new_block_args, strict=True
                ):
                    value_map[old] = new
                deferred_region_results.append(
                    (clone, tuple(operation.region_results))
                )

            for clone, local_results in deferred_region_results:
                clone.region_results.extend(
                    mapped_value(value_map, value) for value in local_results
                )

            if local_terminal_fork is not None:
                assert expanded_fork_id is not None
                try:
                    mapped_predicate = value_map[
                        local_terminal_fork.predicate
                    ]
                    mapped_fork_context = context_prefix + tuple(
                        context_map[parent]
                        for parent in local_terminal_fork.context
                    )
                except KeyError as error:
                    raise TTIRParseError(
                        f"tt.func @{function.name} terminal fork lost its "
                        "predicate or structured region context"
                    ) from error
                expanded_terminal_forks.append(
                    _TerminalFork(
                        expanded_fork_id,
                        mapped_predicate,
                        mapped_fork_context,
                        (
                            len(expanded)
                            if expanded_fork_sequence_index is None
                            else expanded_fork_sequence_index
                        ),
                    )
                )

            if not returned_values:
                raise TTIRParseError(f"tt.func @{function.name} has no tt.return")
            if function.results:
                first = returned_values[0]
                if any(values != first for values in returned_values[1:]):
                    raise TTIRParseError(
                        f"tt.func @{function.name} has multiple non-identical "
                        "SSA returns; CFG result merging is unsupported"
                    )
                return first
            return ()

        public_arguments = tuple(name for name, _, _ in public.arguments)
        emit_function(
            public,
            public_arguments,
            (),
            (),
            preserve_return=True,
        )

        # Reconstruct region leaves from the expanded graph so an outlined
        # helper called inside a supported structured region gets exactly the
        # same structural path as the equivalent inline operations.
        for operation in expanded:
            path = "/".join(
                [
                    *(expanded[parent].name for parent in operation.context),
                    operation.name,
                ]
            )
            for index, name in enumerate(operation.region_block_args):
                leaf_labels[name] = f"region[{path}].arg[{index}]"

        for operation in expanded:
            for operand in [*operation.operands, *operation.region_results]:
                if operand not in expanded_results and operand not in leaf_labels:
                    raise TTIRParseError(
                        f"unresolved expanded SSA value {operand!r} used by "
                        f"{operation.name}"
                    )
        return (
            expanded,
            expanded_results,
            leaf_labels,
            expanded_terminal_forks,
        )

    def _fingerprint(
        self,
        function: _FunctionParts,
        operations: list[_Operation],
        result_to_node: dict[str, int],
        leaf_labels: dict[str, str],
        terminal_forks: list[_TerminalFork],
        *,
        python_interface: Mapping[str, Any] | None,
        schedule: Mapping[str, Any] | None,
    ) -> TTIRFingerprint:
        node_by_id = {operation.node_id: operation for operation in operations}
        structural_cache: dict[int, str] = {}
        semantic_cache: dict[int, str] = {}

        def operand_ref(name: str, *, semantic: bool) -> str:
            if name in leaf_labels:
                return leaf_labels[name]
            node_id = result_to_node[name]
            producer = node_by_id[node_id]
            base = node_ref(node_id, semantic=semantic)
            if len(producer.result_names) <= 1:
                return base
            try:
                slot = producer.result_names.index(name)
            except ValueError as error:  # pragma: no cover - map is constructed together
                raise TTIRParseError(
                    f"SSA value {name!r} is absent from its multi-result producer"
                ) from error
            return f"{base}#result[{slot}]"

        def normalized_operand_refs(
            operation: _Operation, *, semantic: bool
        ) -> list[str]:
            names = [*operation.operands, *operation.region_results]
            refs = [operand_ref(name, semantic=semantic) for name in names]
            if operation.cmp_flipped and len(refs) >= 2:
                refs[0], refs[1] = refs[1], refs[0]
            if (
                operation.name in _COMMUTATIVE_OPS
                or operation.cmp_predicate in _SYMMETRIC_CMP
            ):
                header = refs[: operation.header_operand_count]
                header.sort()
                refs = header + refs[operation.header_operand_count :]
            return refs

        def node_ref(node_id: int, *, semantic: bool) -> str:
            cache = semantic_cache if semantic else structural_cache
            if node_id in cache:
                return cache[node_id]
            operation = node_by_id[node_id]
            refs = normalized_operand_refs(operation, semantic=semantic)
            if semantic:
                local = operation.local_signature
            else:
                # Producer/consumer topology deliberately uses operation kinds,
                # while predicates, fill values, dot attributes and reductions
                # are compared in their dedicated hard components.
                local = _semantic_op_name(operation, node_by_id, result_to_node)
            label = f"{operation.name}[{_digest([local, refs])[:20]}]"
            cache[node_id] = label
            return label

        op_counts = Counter(
            _semantic_op_name(operation, node_by_id, result_to_node)
            for operation in operations
            if operation.name not in _TERMINATORS
        )
        if terminal_forks:
            op_counts["control.conditional"] += len(terminal_forks)

        edges: list[tuple[str, str, str]] = []
        for operation in operations:
            if operation.name in _TERMINATORS:
                continue
            # Hard dataflow labels include normalized operation attributes and
            # constants, not just graph topology.  Otherwise an unchanged op
            # multiset wired through ``0.5`` instead of ``2.0`` could close.
            # Dedicated reduction/predicate fields may report the same mutation
            # more specifically; exactness is intentionally preferred here.
            refs = normalized_operand_refs(operation, semantic=True)
            consumer = node_ref(operation.node_id, semantic=True)
            for index, producer in enumerate(refs):
                if index < operation.header_operand_count:
                    role = f"operand[{index}]"
                else:
                    role = f"region_result[{index - operation.header_operand_count}]"
                edges.append((producer, consumer, role))

        # The function result mapping is semantic dataflow, not terminator
        # presentation.  Without these positional edges, returning a different
        # already-computed SSA value (or swapping equal-typed multi-results)
        # would leave the operation graph unchanged and falsely hard-pass.
        public_returns = [
            operation
            for operation in operations
            if operation.name == "tt.return" and not operation.context
        ]
        fork_returns = [
            operation
            for operation in public_returns
            if operation.fork_id is not None
        ]
        if fork_returns:
            if (
                len(public_returns) != 2
                or {operation.fork_arm for operation in fork_returns}
                != {"true", "false"}
                or any(operation.operands for operation in fork_returns)
            ):
                raise TTIRParseError(
                    "expanded public terminal fork must have exactly one void "
                    "tt.return in each role-preserving arm"
                )
        else:
            if len(public_returns) != 1:
                raise TTIRParseError(
                    "expanded public tt.func must contain exactly one "
                    "top-level tt.return"
                )
            for index, value in enumerate(public_returns[0].operands):
                edges.append(
                    (
                        operand_ref(value, semantic=True),
                        "region[root]",
                        f"function.return[{index}]",
                    )
                )

        # Region ownership is hard evidence.  A global operation multiset plus
        # descendant counts cannot distinguish non-isomorphic loop trees, and
        # sorting memory effects loses the order of writes to the same address.
        # Build a recursive, content-sensitive region identity without using
        # SSA names, source function names, or numeric node ids.  Direct member
        # edges bind every operation/effect to its parent; the ordered effect
        # trace retains lexical happens-before for reads, writes, atomics,
        # barriers, and assertions.  Outlined calls have already been expanded,
        # so outlined and equivalent inline programs traverse this same graph.
        region_ids = {
            operation.node_id
            for operation in operations
            if operation.name in _REGION_OWNERS
        }
        terminal_fork_by_id = {
            item.fork_id: item for item in terminal_forks
        }
        if len(terminal_fork_by_id) != len(terminal_forks):
            raise TTIRParseError("expanded terminal-fork identities are not unique")
        direct_terminal_forks: dict[int | None, list[_TerminalFork]] = {}
        for terminal_fork in terminal_forks:
            parent = (
                terminal_fork.context[-1]
                if terminal_fork.context
                else None
            )
            if parent is not None and parent not in region_ids:
                raise TTIRParseError(
                    "terminal fork has a non-structured caller context"
                )
            direct_terminal_forks.setdefault(parent, []).append(terminal_fork)
        direct_children: dict[int | None, list[int]] = {}
        direct_members: dict[int | None, list[_Operation]] = {}
        direct_sequence: dict[int | None, list[_Operation]] = {}
        for operation in operations:
            parent = operation.context[-1] if operation.context else None
            if parent is not None and parent not in region_ids:
                raise TTIRParseError(
                    f"operation {operation.name} has a non-region parent"
                )
            terminal_fork = (
                terminal_fork_by_id.get(operation.fork_id)
                if operation.fork_id is not None
                else None
            )
            if operation.fork_id is not None and terminal_fork is None:
                raise TTIRParseError(
                    "operation refers to an unknown expanded terminal fork"
                )
            if operation.fork_arm not in {None, "true", "false"}:
                raise TTIRParseError("operation has an invalid terminal-fork role")
            is_direct_fork_member = (
                terminal_fork is not None
                and operation.context == terminal_fork.context
            )
            # The two arms are mutually exclusive.  Their direct members are
            # attached to the synthetic control node below, never flattened
            # into the caller's ordinary region sequence.  Nested structured
            # regions still retain their own internal membership graph.
            if not is_direct_fork_member:
                direct_sequence.setdefault(parent, []).append(operation)
                if operation.node_id in region_ids:
                    direct_children.setdefault(parent, []).append(operation.node_id)
                elif operation.name not in _TERMINATORS:
                    direct_members.setdefault(parent, []).append(operation)

            restricted_ancestor = next(
                (
                    node_by_id[ancestor].name
                    for ancestor in reversed(operation.context)
                    if node_by_id[ancestor].name in {"tt.reduce", "tt.scan"}
                ),
                None,
            )
            # Classification is total over the pinned observer profile.  Do
            # this even at function root: unknown operations may be effectful,
            # so treating them as pure until a future release understands them
            # would make lexical reordering invisible.
            operation_has_effect = _operation_has_effect(operation)
            if operation_has_effect and restricted_ancestor is not None:
                raise TTIRParseError(
                    f"side-effecting operation {operation.name} inside "
                    f"{restricted_ancestor} is unsupported"
                )

        effectful_cache: dict[int, bool] = {}

        def terminal_fork_has_effect(terminal_fork: _TerminalFork) -> bool:
            return any(
                operation.name not in _TERMINATORS
                and _operation_has_effect(operation)
                for operation in operations
                if operation.fork_id == terminal_fork.fork_id
            )

        def region_has_effect(region_id: int) -> bool:
            cached = effectful_cache.get(region_id)
            if cached is not None:
                return cached
            value = any(
                _operation_has_effect(operation)
                for operation in direct_members.get(region_id, ())
            ) or any(
                region_has_effect(child)
                for child in direct_children.get(region_id, ())
            ) or any(
                terminal_fork_has_effect(terminal_fork)
                for terminal_fork in direct_terminal_forks.get(region_id, ())
            )
            effectful_cache[region_id] = value
            return value

        region_ref_cache: dict[int, str] = {}
        terminal_fork_fingerprint_cache: dict[
            int, TerminalForkFingerprint
        ] = {}
        terminal_fork_ref_cache: dict[int, str] = {}

        def terminal_fork_arm(
            terminal_fork: _TerminalFork, role: str
        ) -> TerminalForkArmFingerprint:
            direct = [
                operation
                for operation in operations
                if operation.fork_id == terminal_fork.fork_id
                and operation.fork_arm == role
                and operation.context == terminal_fork.context
            ]
            members = sorted(
                node_ref(operation.node_id, semantic=True)
                for operation in direct
                if operation.node_id not in region_ids
                and operation.name not in _TERMINATORS
            )
            structured_regions = sorted(
                region_ref(operation.node_id)
                for operation in direct
                if operation.node_id in region_ids
            )
            effect_order: list[str] = []
            for operation in direct:
                if operation.node_id in region_ids:
                    if region_has_effect(operation.node_id):
                        effect_order.append(
                            f"region:{region_ref(operation.node_id)}"
                        )
                elif (
                    operation.name not in _TERMINATORS
                    and _operation_has_effect(operation)
                ):
                    effect_order.append(
                        f"{operation.name}:"
                        f"{node_ref(operation.node_id, semantic=True)}"
                    )
            return TerminalForkArmFingerprint(
                role=role,
                members=tuple(members),
                effect_order=tuple(effect_order),
                structured_regions=tuple(structured_regions),
            )

        def terminal_fork_fingerprint(
            terminal_fork: _TerminalFork,
        ) -> TerminalForkFingerprint:
            cached = terminal_fork_fingerprint_cache.get(
                terminal_fork.fork_id
            )
            if cached is not None:
                return cached
            cached = TerminalForkFingerprint(
                predicate=operand_ref(
                    terminal_fork.predicate, semantic=True
                ),
                region_context=tuple(
                    node_ref(owner, semantic=True)
                    for owner in terminal_fork.context
                ),
                true_arm=terminal_fork_arm(terminal_fork, "true"),
                false_arm=terminal_fork_arm(terminal_fork, "false"),
            )
            terminal_fork_fingerprint_cache[terminal_fork.fork_id] = cached
            return cached

        def terminal_fork_ref(terminal_fork: _TerminalFork) -> str:
            cached = terminal_fork_ref_cache.get(terminal_fork.fork_id)
            if cached is not None:
                return cached
            fingerprint = terminal_fork_fingerprint(terminal_fork)
            cached = (
                "control.conditional["
                f"{_digest(fingerprint.to_dict())[:20]}]"
            )
            terminal_fork_ref_cache[terminal_fork.fork_id] = cached
            return cached

        def terminal_arm_ref(
            arm: TerminalForkArmFingerprint,
        ) -> str:
            return (
                f"control.arm[{arm.role}:"
                f"{_digest(arm.to_dict())[:20]}]"
            )

        def direct_effect_events(parent: int | None) -> list[str]:
            events: list[str] = []
            sequence: list[tuple[int, int, _Operation | _TerminalFork]] = [
                (operation.node_id, 1, operation)
                for operation in direct_sequence.get(parent, ())
            ]
            sequence.extend(
                (terminal_fork.sequence_index, 0, terminal_fork)
                for terminal_fork in direct_terminal_forks.get(parent, ())
            )
            for _, _, item in sorted(
                sequence, key=lambda entry: (entry[0], entry[1])
            ):
                if isinstance(item, _TerminalFork):
                    if terminal_fork_has_effect(item):
                        events.append(f"control:{terminal_fork_ref(item)}")
                elif item.node_id in region_ids:
                    if region_has_effect(item.node_id):
                        events.append(f"region:{region_ref(item.node_id)}")
                elif _operation_has_effect(item):
                    events.append(
                        f"{item.name}:"
                        f"{node_ref(item.node_id, semantic=True)}"
                    )
            return events

        def region_ref(region_id: int) -> str:
            cached = region_ref_cache.get(region_id)
            if cached is not None:
                return cached
            operation = node_by_id[region_id]
            child_refs = sorted(
                region_ref(child)
                for child in direct_children.get(region_id, ())
            )
            member_refs = sorted(
                node_ref(member.node_id, semantic=True)
                for member in direct_members.get(region_id, ())
            )
            member_refs.extend(
                sorted(
                    terminal_fork_ref(terminal_fork)
                    for terminal_fork in direct_terminal_forks.get(
                        region_id, ()
                    )
                )
            )
            payload = {
                "owner": node_ref(region_id, semantic=True),
                "children": child_refs,
                "members": member_refs,
                "effect_order": direct_effect_events(region_id),
            }
            cached = f"region[{operation.name}:{_digest(payload)[:20]}]"
            region_ref_cache[region_id] = cached
            return cached

        for parent in [None, *sorted(region_ids)]:
            parent_ref = "region[root]" if parent is None else region_ref(parent)
            for child in direct_children.get(parent, ()):
                edges.append((parent_ref, region_ref(child), "region.child"))
            for member in direct_members.get(parent, ()):
                edges.append(
                    (
                        parent_ref,
                        node_ref(member.node_id, semantic=True),
                        "region.member",
                    )
                )
            for terminal_fork in direct_terminal_forks.get(parent, ()):
                fingerprint = terminal_fork_fingerprint(terminal_fork)
                control = terminal_fork_ref(terminal_fork)
                edges.extend(
                    [
                        (
                            fingerprint.predicate,
                            control,
                            "condition",
                        ),
                        (
                            terminal_arm_ref(fingerprint.true_arm),
                            control,
                            "true",
                        ),
                        (
                            terminal_arm_ref(fingerprint.false_arm),
                            control,
                            "false",
                        ),
                        (
                            parent_ref,
                            control,
                            "region.member",
                        ),
                    ]
                )
            effect_events = direct_effect_events(parent)
            if effect_events:
                edges.append(
                    (
                        parent_ref,
                        f"effect.trace[{_digest(effect_events)[:20]}]",
                        "effect.order",
                    )
                )

        reductions: list[ReductionFingerprint] = []
        for operation in operations:
            if operation.name != "tt.reduce":
                continue
            if operation.axis is None:
                raise TTIRParseError("tt.reduce is missing its axis attribute")
            combiners: list[str] = []
            for result_name in operation.region_results:
                producer = node_by_id[result_to_node[result_name]]
                combiners.append(producer.name)
            if not combiners:
                raise TTIRParseError("tt.reduce region has no returned combiner")
            reductions.append(
                ReductionFingerprint(
                    axis=operation.axis,
                    combiners=tuple(combiners),
                    input_count=operation.header_operand_count,
                    result_count=max(1, len(operation.result_names)),
                )
            )

        memory_effects: list[MemoryEffect] = []
        effect_counts: Counter[str] = Counter()

        def value_ref(value: str) -> str:
            """Expose typed constants while keeping other SSA values opaque."""

            node_id = result_to_node.get(value)
            if node_id is not None:
                producer = node_by_id[node_id]
                if producer.name == "arith.constant":
                    return producer.local_signature
            return operand_ref(value, semantic=True)

        def direct_constant_ref(value: str) -> str | None:
            """Return an exact typed constant only for a direct SSA producer."""

            node_id = result_to_node.get(value)
            if node_id is None:
                return None
            producer = node_by_id[node_id]
            return (
                producer.local_signature
                if producer.name == "arith.constant"
                else None
            )

        for operation in operations:
            if operation.name == "tt.load":
                masked = operation.header_operand_count >= 2
                predicate = (
                    operand_ref(operation.operands[1], semantic=True) if masked else None
                )
                other = (
                    value_ref(operation.operands[2])
                    if operation.header_operand_count >= 3
                    else None
                )
                effect = MemoryEffect(
                    "load", masked, predicate, other, operation.local_signature
                )
                memory_effects.append(effect)
                effect_counts["read.masked" if masked else "read"] += 1
            elif operation.name == "tt.store":
                masked = operation.header_operand_count >= 3
                predicate = (
                    operand_ref(operation.operands[2], semantic=True) if masked else None
                )
                effect = MemoryEffect(
                    "store",
                    masked,
                    predicate,
                    direct_constant_ref(operation.operands[1]),
                    operation.local_signature,
                )
                memory_effects.append(effect)
                effect_counts["write.masked" if masked else "write"] += 1
            elif operation.name.startswith("tt.atomic"):
                effect_counts[operation.name] += 1
            elif operation.name in {"tt.assert", "tt.device_assert"}:
                effect_counts[operation.name] += 1
            elif "barrier" in operation.name or operation.name in {
                "gpu.wait",
                "tt.async_wait",
            }:
                effect_counts[operation.name] += 1
            elif operation.name == "scf.for":
                effect_counts["control.scf.for"] += 1

        dots = [
            DotFingerprint(
                operation.name,
                operation.header_operand_count,
                operation.local_signature,
            )
            for operation in operations
            if operation.name in {"tt.dot", "tt.dot_scaled"}
        ]

        loops: list[LoopFingerprint] = []
        for operation in operations:
            if operation.name != "scf.for":
                continue
            header = [operand_ref(name, semantic=True) for name in operation.operands]
            if len(header) < 3:
                raise TTIRParseError("scf.for has fewer than lower/upper/step operands")
            body_effects = Counter()
            body_ops: Counter[str] = Counter()
            local_region_arguments = set(operation.region_block_args)
            for nested_region in operations:
                if operation.node_id in nested_region.context:
                    local_region_arguments.update(
                        nested_region.region_block_args
                    )
            captured_values: set[str] = set()
            region_argument_use_counts: Counter[str] = Counter()
            directly_nested_forks = [
                terminal_fork
                for terminal_fork in terminal_forks
                if terminal_fork.context
                and terminal_fork.context[-1] == operation.node_id
            ]
            directly_nested_fork_ids = {
                terminal_fork.fork_id
                for terminal_fork in directly_nested_forks
            }
            for nested in operations:
                if operation.node_id not in nested.context:
                    continue
                if nested.fork_id in directly_nested_fork_ids:
                    # The mutually exclusive arms are summarized by one
                    # role-preserving synthetic control member below.  Do not
                    # turn them into an unconditional flat loop body.
                    continue
                if nested.name not in _TERMINATORS:
                    body_ops[
                        _semantic_op_name(
                            nested, node_by_id, result_to_node
                        )
                    ] += 1
                for operand in [*nested.operands, *nested.region_results]:
                    if operand in local_region_arguments:
                        region_argument_use_counts[operand] += 1
                        continue
                    producer_id = result_to_node.get(operand)
                    if producer_id is not None:
                        producer = node_by_id[producer_id]
                        if operation.node_id in producer.context:
                            continue
                    captured_values.add(operand_ref(operand, semantic=True))
                if nested.name == "tt.load":
                    body_effects[
                        "read.masked" if nested.header_operand_count >= 2 else "read"
                    ] += 1
                elif nested.name == "tt.store":
                    body_effects[
                        "write.masked" if nested.header_operand_count >= 3 else "write"
                    ] += 1
                elif nested.name.startswith("tt.atomic"):
                    body_effects[nested.name] += 1
            for terminal_fork in directly_nested_forks:
                body_ops["control.conditional"] += 1
                if terminal_fork_has_effect(terminal_fork):
                    body_effects["control.conditional"] += 1
                predicate = terminal_fork.predicate
                producer_id = result_to_node.get(predicate)
                if producer_id is None or operation.node_id not in node_by_id[
                    producer_id
                ].context:
                    captured_values.add(
                        operand_ref(predicate, semantic=True)
                    )
            loops.append(
                LoopFingerprint(
                    lower=header[0],
                    upper=header[1],
                    step=header[2],
                    init_values=tuple(header[3:]),
                    yielded_values=tuple(
                        operand_ref(name, semantic=True)
                        for name in operation.region_results
                    ),
                    body_effects=tuple(sorted(body_effects.items())),
                    block_argument_count=len(operation.region_block_args),
                    induction_use_count=(
                        0
                        if not operation.region_block_args
                        else region_argument_use_counts[
                            operation.region_block_args[0]
                        ]
                    ),
                    carried_argument_use_counts=tuple(
                        region_argument_use_counts[name]
                        for name in operation.region_block_args[1:]
                    ),
                    captured_values=tuple(sorted(captured_values)),
                    body_op_multiset=tuple(sorted(body_ops.items())),
                )
            )

        access = self._infer_access(
            function, operations, result_to_node, leaf_labels, node_by_id
        )
        interface_arguments: list[InterfaceArgument] = []
        for position, (_, type_text, attrs) in enumerate(function.arguments):
            dtype = _pointer_dtype(type_text)
            if dtype is not None:
                kind = "pointer"
                rank = None
                argument_access = access.get(position, "unused")
            else:
                tensor = re.fullmatch(r"tensor<(?P<body>.*)>", type_text)
                if tensor:
                    kind = "tensor"
                    pieces = tensor.group("body").split("x")
                    rank = max(0, len(pieces) - 1)
                    dtype = pieces[-1]
                else:
                    kind = "scalar"
                    rank = 0
                    dtype = type_text
                argument_access = "value"
            interface_arguments.append(
                InterfaceArgument(
                    position,
                    type_text,
                    kind,
                    dtype,
                    rank,
                    argument_access,
                    attrs,
                )
            )

        memory_effects.sort(key=lambda item: _stable_json(item.to_dict()))
        reductions.sort(key=lambda item: _stable_json(item.to_dict()))
        dots.sort(key=lambda item: _stable_json(item.to_dict()))
        loops.sort(key=lambda item: _stable_json(item.to_dict()))
        terminal_fork_fingerprints = sorted(
            (
                terminal_fork_fingerprint(item)
                for item in terminal_forks
            ),
            key=lambda item: _stable_json(item.to_dict()),
        )
        mask_predicates = sorted(
            effect.predicate
            for effect in memory_effects
            if effect.predicate is not None
        )
        return TTIRFingerprint(
            interface=RuntimeInterface(
                tuple(interface_arguments),
                function.results,
                _canonical_python_interface(python_interface),
            ),
            op_multiset=tuple(sorted(op_counts.items())),
            dataflow_edges=tuple(sorted(edges)),
            reductions=tuple(reductions),
            memory_effects=tuple(memory_effects),
            mask_predicates=tuple(mask_predicates),
            effect_multiset=tuple(sorted(effect_counts.items())),
            dots=tuple(dots),
            loops=tuple(loops),
            terminal_forks=tuple(terminal_fork_fingerprints),
            schedule_view=_canonical_schedule(schedule),
        )

    def _infer_access(
        self,
        function: _FunctionParts,
        operations: list[_Operation],
        result_to_node: dict[str, int],
        leaf_labels: dict[str, str],
        node_by_id: dict[int, _Operation],
    ) -> dict[int, str]:
        arg_position = {
            name: index for index, (name, _, _) in enumerate(function.arguments)
        }
        value_sources: dict[str, tuple[str, ...]] = {}

        for operation in operations:
            if operation.name == "tt.scan":
                input_count = len(operation.operands)
                if (
                    input_count < 1
                    or len(operation.result_names) != input_count
                    or len(operation.region_results) != input_count
                    or len(operation.region_block_args) != 2 * input_count
                ):
                    raise TTIRParseError(
                        "tt.scan recurrence arities changed after call expansion"
                    )
                for index, block_arg in enumerate(operation.region_block_args):
                    value_sources[block_arg] = (
                        operation.operands[index % input_count],
                    )
                for result_name, initial, returned in zip(
                    operation.result_names,
                    operation.operands,
                    operation.region_results,
                    strict=True,
                ):
                    value_sources[result_name] = (initial, returned)
                continue
            if operation.name != "scf.for":
                sources = tuple(operation.operands)
                for result_name in operation.result_names:
                    value_sources[result_name] = sources
                continue

            # scf.for has three control operands followed by one init value per
            # iter_arg.  Its first region block argument is the induction
            # variable; the remaining block arguments, operation results, and
            # scf.yield operands correspond positionally.  A carried value may
            # depend both on its initial value and on values captured by the
            # yielded update.  Keep those equations per slot: joining all
            # iter_args would incorrectly turn unrelated input/output pointers
            # into inout arguments.
            carried_block_args = tuple(operation.region_block_args[1:])
            init_values = tuple(operation.operands[3:])
            yielded_values = tuple(operation.region_results)
            result_names = tuple(operation.result_names)
            arities = {
                len(carried_block_args),
                len(init_values),
                len(yielded_values),
                len(result_names),
            }
            if len(arities) != 1:
                raise TTIRParseError(
                    "scf.for iter_args, results, block arguments, and yield "
                    "values must have identical arity"
                )
            if not operation.region_block_args:
                raise TTIRParseError(
                    "scf.for is missing its induction block argument"
                )

            value_sources[operation.region_block_args[0]] = tuple(
                operation.operands[:3]
            )
            for block_arg, result_name, initial, yielded in zip(
                carried_block_args,
                result_names,
                init_values,
                yielded_values,
                strict=True,
            ):
                sources = (initial, yielded)
                value_sources[block_arg] = sources
                value_sources[result_name] = sources

        # Region-carried SSA contains intentional cycles: a yielded pointer is
        # commonly computed from the corresponding iter_arg.  Recursive DFS
        # either drops that dependency at a region leaf or rejects the valid
        # cycle.  Compute the least fixed point of function-argument provenance
        # instead.  The lattice is finite (sets of argument positions), so this
        # terminates deterministically and also captures a different pointer
        # introduced by a yielded update.
        dependencies: dict[str, set[int]] = {
            name: {position} for name, position in arg_position.items()
        }
        for name in leaf_labels:
            dependencies.setdefault(name, set())
        for name in result_to_node:
            dependencies.setdefault(name, set())

        changed = True
        while changed:
            changed = False
            for name, sources in value_sources.items():
                inferred = (
                    set().union(
                        *(dependencies[source] for source in sources)
                    )
                    if sources
                    else set()
                )
                if not inferred.issubset(dependencies[name]):
                    dependencies[name].update(inferred)
                    changed = True

        # Preserve the pre-existing rejection of an ordinary cyclic SSA
        # producer graph.  Valid loop-carried cycles pass through region block
        # arguments (leaves in this structural walk) and are handled only by
        # the fixed-point equations above.
        acyclic: set[str] = set()

        def validate_acyclic(
            name: str,
            active: frozenset[int] = frozenset(),
        ) -> None:
            if name in acyclic or name in leaf_labels:
                return
            node_id = result_to_node[name]
            if node_id in active:
                raise TTIRParseError(
                    "cyclic SSA dependency while inferring interface"
                )
            for operand in node_by_id[node_id].operands:
                validate_acyclic(operand, active | {node_id})
            acyclic.add(name)

        modes: dict[int, set[str]] = {}
        for operation in operations:
            if not operation.operands:
                continue
            if operation.name == "tt.load":
                mode = "read"
            elif operation.name == "tt.store":
                mode = "write"
            elif operation.name.startswith("tt.atomic"):
                mode = "readwrite"
            else:
                continue
            validate_acyclic(operation.operands[0])
            for position in dependencies[operation.operands[0]]:
                modes.setdefault(position, set()).add(mode)
        access: dict[int, str] = {}
        for position, seen in modes.items():
            read = bool(seen & {"read", "readwrite"})
            write = bool(seen & {"write", "readwrite"})
            access[position] = "inout" if read and write else "input" if read else "output"
        return access


def _semantic_op_name(
    operation: _Operation,
    node_by_id: Mapping[int, _Operation],
    result_to_node: Mapping[str, int],
) -> str:
    if operation.name == "tt.load":
        return "tt.load.masked" if operation.header_operand_count >= 2 else "tt.load"
    if operation.name == "tt.store":
        return "tt.store.masked" if operation.header_operand_count >= 3 else "tt.store"
    return operation.name


def parse_ttir(
    text: str,
    *,
    python_interface: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> TTIRFingerprint:
    """Parse TTIR and return its normalized Gate1 fingerprint."""

    return TTIRParser().parse(
        text, python_interface=python_interface, schedule=schedule
    )


def runtime_argument_names(text: str) -> tuple[str, ...]:
    """Return surviving TTIR runtime argument names in ABI order.

    Triton may specialize ordinary scalar Python parameters and remove them
    from the TTIR function signature.  The remaining SSA names retain their
    Python parameter spelling and let Gate1 validate the surviving subsequence
    without pretending that specialization changed the public Python ABI.
    """

    cleaned = _strip_locations(text)
    function = _parse_function(cleaned)
    return tuple(name.removeprefix("%") for name, _, _ in function.arguments)


def fingerprint_ttir(
    text: str,
    *,
    python_interface: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> TTIRFingerprint:
    return parse_ttir(text, python_interface=python_interface, schedule=schedule)


extract_ttir_fingerprint = fingerprint_ttir
