"""Deterministic semantic fingerprints for Triton 3.6 TTIR.

Gate1 compares two compilations produced by the *same* pinned Triton frontend.
It therefore does not need a general MLIR parser, but it must avoid comparing
unstable presentation details such as SSA names, debug locations, or the text
order of independent pure operations.  This module implements that deliberately
small boundary:

* parse the public ``tt.func`` runtime ABI and infer pointer access modes;
* build an SSA producer/consumer graph, including ``tt.reduce`` regions and
  ``scf.for`` loop-carried values;
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
    "TTIRFingerprint",
    "TTIR_FINGERPRINT_SCHEMA",
    "TTIRParseError",
    "TTIRParser",
    "extract_ttir_fingerprint",
    "fingerprint_ttir",
    "parse_ttir",
]


TTIR_FINGERPRINT_SCHEMA = "triton-ttir-3.6-gate1-v4"
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
_TERMINATORS = {"tt.return", "tt.reduce.return", "scf.yield"}
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
    cmp_predicate: str | None = None
    cmp_flipped: bool = False
    header_operand_count: int = 0
    region_results: list[str] = field(default_factory=list)
    region_block_args: list[str] = field(default_factory=list)
    constant_value: float | None = None


@dataclass(frozen=True)
class _FunctionParts:
    arguments: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]
    results: tuple[str, ...]
    body: str


def _parse_function(text: str) -> _FunctionParts:
    matches = list(re.finditer(r"\btt\.func\s+public\s+@[A-Za-z0-9_.$-]+\s*\(", text))
    if len(matches) != 1:
        raise TTIRParseError(
            f"expected exactly one public tt.func, found {len(matches)}"
        )
    start = matches[0].start()
    args_open = text.find("(", start)
    args_close = _balanced_end(text, args_open, "(", ")")
    args_text = text[args_open + 1 : args_close]

    arguments: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    for item in _split_top_level(args_text):
        match = re.match(r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:\s*(?P<rest>.*)", item, re.S)
        if match is None:
            raise TTIRParseError(f"cannot parse function argument: {item!r}")
        rest = match.group("rest").strip()
        attr_start = _find_top_level_char(rest, "{")
        if attr_start is None:
            type_text, attrs = rest, ()
        else:
            attr_end = _balanced_end(rest, attr_start, "{", "}")
            type_text = rest[:attr_start]
            attrs = _canonical_attrs(rest[attr_start : attr_end + 1])
        arguments.append((match.group("name"), _normalize_space(type_text), attrs))

    cursor = args_close + 1
    remainder = text[cursor:]
    attr_match = re.match(r"\s*attributes\s*", remainder)
    if attr_match:
        attr_open = cursor + attr_match.end()
        while attr_open < len(text) and text[attr_open].isspace():
            attr_open += 1
        attr_close = _balanced_end(text, attr_open, "{", "}")
        between = text[args_close + 1 : cursor + attr_match.start()]
        cursor = attr_close + 1
    else:
        between = text[args_close + 1 : cursor]

    body_open = cursor
    while body_open < len(text) and text[body_open].isspace():
        body_open += 1
    if body_open >= len(text) or text[body_open] != "{":
        # A result type can appear before attributes/body.  Locate the next
        # top-level body brace after the argument list.
        body_open = _next_body_brace(text, args_close + 1)
        between = text[args_close + 1 : body_open]
        between = re.sub(r"\battributes\s*\{.*?\}\s*$", "", between, flags=re.S)
    body_close = _balanced_end(text, body_open, "{", "}")

    result_text = between.strip()
    if result_text.startswith("->"):
        result_text = result_text[2:].strip()
        if result_text.startswith("(") and result_text.endswith(")"):
            results = tuple(
                _normalize_space(item)
                for item in _split_top_level(result_text[1:-1])
            )
        else:
            results = (_normalize_space(result_text),)
    else:
        results = ()
    return _FunctionParts(tuple(arguments), results, text[body_open + 1 : body_close])


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


def _operation_head(line: str) -> tuple[tuple[str, ...], str, str] | None:
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
    return lhs, name, tail[match.end() :].strip()


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
) -> tuple[list[_Operation], dict[str, int]]:
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
    return normalized, normalized_result_to_node


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


def _parse_for_header(tail: str) -> tuple[list[str], list[str]]:
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
    rest = match.group("rest")
    marker = re.search(r"\biter_args\s*\(", rest)
    if marker:
        opening = rest.find("(", marker.start())
        closing = _balanced_end(rest, opening, "(", ")")
        for item in _split_top_level(rest[opening + 1 : closing]):
            pair = re.fullmatch(
                r"(?P<arg>%[A-Za-z0-9_.$-]+)\s*=\s*(?P<init>%[A-Za-z0-9_.$-]+)",
                item,
            )
            if pair is None:
                raise TTIRParseError(f"cannot parse scf.for iter_arg: {item!r}")
            block_args.append(pair.group("arg"))
            operands.append(pair.group("init"))
    return operands, block_args


def _parse_block_args(line: str) -> list[str]:
    match = re.match(r"\s*\^[A-Za-z0-9_.$-]+\s*\((?P<args>.*)\)\s*:\s*$", line)
    if match is None:
        return []
    names: list[str] = []
    for item in _split_top_level(match.group("args")):
        arg = re.match(r"(?P<name>%[A-Za-z0-9_.$-]+)\s*:", item)
        if arg is None:
            raise TTIRParseError(f"cannot parse region block argument: {item!r}")
        names.append(arg.group("name"))
    return names


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
        function = _parse_function(cleaned)
        operations, result_to_node, leaf_labels = self._parse_operations(function)
        operations, result_to_node = _bounded_normalize_operations(
            operations, result_to_node
        )
        return self._fingerprint(
            function,
            operations,
            result_to_node,
            leaf_labels,
            python_interface=python_interface,
            schedule=schedule,
        )

    def _parse_operations(
        self, function: _FunctionParts
    ) -> tuple[list[_Operation], dict[str, int], dict[str, str]]:
        operations: list[_Operation] = []
        result_to_node: dict[str, int] = {}
        leaf_labels = {
            name: f"arg[{index}]:{type_text}"
            for index, (name, type_text, _) in enumerate(function.arguments)
        }
        region_stack: list[tuple[int, int]] = []
        # One lexical binding map for the function and one for every active
        # operation region.  MLIR SSA names need only be unique within their
        # region; sibling scf.for/tt.reduce bodies may legally reuse spellings
        # such as ``%n_offset``.  Internal qualified keys keep the existing
        # fingerprint graph globally addressable without exposing SSA spelling
        # in the final fingerprint.
        scope_bindings: list[dict[str, str]] = [
            {name: name for name, _, _ in function.arguments}
        ]

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
            return qualified

        def resolve(name: str) -> str:
            for bindings in reversed(scope_bindings):
                qualified = bindings.get(name)
                if qualified is not None:
                    return qualified
            # Preserve the original spelling so the existing final unresolved
            # check emits a deterministic parse error with useful context.
            return name

        for line_number, line in enumerate(function.body.splitlines(), 1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            if not stripped or stripped.startswith("//"):
                continue
            if stripped.startswith(("})", "}")):
                while region_stack and indent <= region_stack[-1][1]:
                    region_stack.pop()
                    scope_bindings.pop()
                continue

            block_args = _parse_block_args(line)
            if block_args:
                if not region_stack:
                    raise TTIRParseError(
                        f"line {line_number}: block arguments outside a region"
                    )
                parent = operations[region_stack[-1][0]]
                qualified_args = [
                    define(name, line_number=line_number)
                    for name in block_args
                ]
                parent.region_block_args.extend(qualified_args)
                path = "/".join(operations[node].name for node, _ in region_stack)
                for index, name in enumerate(qualified_args):
                    leaf_labels[name] = f"region[{path}].arg[{index}]"
                continue

            head = _operation_head(line)
            if head is None:
                continue
            results, name, tail = head
            if (
                name not in {"tt.reduce", "scf.for"}
                and stripped.endswith("{")
            ):
                raise TTIRParseError(
                    f"line {line_number}: unsupported region-bearing TTIR "
                    f"operation {name!r}; Gate1 refuses to flatten an unknown "
                    "SSA scope"
                )
            raw_operands = _SSA_RE.findall(tail)
            raw_block_args: list[str] = []
            if name == "scf.for":
                raw_operands, raw_block_args = _parse_for_header(tail)
            operands = [resolve(operand) for operand in raw_operands]
            qualified_results = tuple(
                define(result, line_number=line_number) for result in results
            )
            axis_match = re.search(r"\baxis\s*=\s*(-?[0-9]+)", tail)
            operation = _Operation(
                node_id=len(operations),
                name=name,
                result_names=qualified_results,
                operands=list(operands),
                local_signature=_local_signature(name, tail),
                context=tuple(node for node, _ in region_stack),
                axis=int(axis_match.group(1)) if axis_match else None,
                cmp_predicate=_cmp_info(name, tail)[0],
                cmp_flipped=_cmp_info(name, tail)[1],
                header_operand_count=len(operands),
                region_block_args=[],
                constant_value=(
                    _scalar_constant(tail)
                    if name == "arith.constant"
                    else None
                ),
            )
            operations.append(operation)
            for result in qualified_results:
                if result in result_to_node or result in leaf_labels:
                    raise TTIRParseError(
                        f"line {line_number}: duplicate internal SSA key {result}"
                    )
                result_to_node[result] = operation.node_id

            if name in {"tt.reduce.return", "scf.yield"} and region_stack:
                parent = operations[region_stack[-1][0]]
                parent.region_results.extend(operands)

            if name in {"tt.reduce", "scf.for"}:
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
                operation.region_block_args.extend(qualified_block_args)
                for index, block_name in enumerate(qualified_block_args):
                    leaf_labels[block_name] = f"region[{path}].arg[{index}]"
                region_stack.append((operation.node_id, indent))
                scope_bindings.append(child_bindings)

        if region_stack:
            raise TTIRParseError("unterminated operation region")
        for operation in operations:
            for operand in [*operation.operands, *operation.region_results]:
                if operand not in result_to_node and operand not in leaf_labels:
                    raise TTIRParseError(
                        f"unresolved SSA value {operand!r} used by {operation.name}"
                    )
        return operations, result_to_node, leaf_labels

    def _fingerprint(
        self,
        function: _FunctionParts,
        operations: list[_Operation],
        result_to_node: dict[str, int],
        leaf_labels: dict[str, str],
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
            return node_ref(node_id, semantic=semantic)

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
            for nested in operations:
                if operation.node_id not in nested.context:
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
