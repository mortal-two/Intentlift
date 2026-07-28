"""Semantic, alpha-normalized PTX observer shared by CUDA and TileLang Gate1.

The observer intentionally ignores presentation details (entry names, virtual
register spelling, labels, debug directives, compiler banners, cache hints and
equivalent address-algebra spelling) while retaining the ABI, value-operation
topology, address spaces, predication, memory effects, collectives,
synchronization and thread/block coordinate usage. Exact normalized
instructions remain diagnostic evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping


PTX_FINGERPRINT_SCHEMA = "cuda-ptx-semantic-effects-v2"


class PTXObservationError(ValueError):
    """PTX cannot support a strict compiler-observation comparison."""


_ENTRY_RE = re.compile(
    r"(?:\.visible\s+)?\.entry\s+(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*)\s*\(",
    re.MULTILINE,
)
_PARAM_RE = re.compile(
    r"\.param\s+(?:\.align\s+(?P<align>[0-9]+)\s+)?"
    r"(?P<type>\.[A-Za-z][A-Za-z0-9]*)\s+"
    r"(?P<name>[A-Za-z_.$][A-Za-z0-9_.$]*)(?:\[(?P<extent>[0-9]+)\])?"
)
_REGISTER_RE = re.compile(r"%[A-Za-z_][A-Za-z0-9_.$]*(?:\.[A-Za-z0-9_]+)?")
_REGISTER_DECL_RE = re.compile(
    r"\.reg\s+(?P<type>\.[A-Za-z][A-Za-z0-9]*)\s+"
    r"(?P<base>%[A-Za-z_][A-Za-z0-9_.$]*)(?:<(?P<count>[0-9]+)>)?\s*;"
)
_OPCODE_RE = re.compile(
    r"^(?:(?P<predicate>@!?%[A-Za-z_][A-Za-z0-9_.$]*)\s+)?"
    r"(?P<opcode>[A-Za-z][A-Za-z0-9_.]*)\s*(?P<operands>.*);$"
)
_SPECIAL_REGISTERS = (
    "%clock",
    "%clock64",
    "%ctaid",
    "%envreg",
    "%globaltimer",
    "%gridid",
    "%laneid",
    "%lanemask",
    "%nctaid",
    "%nsmid",
    "%ntid",
    "%pm",
    "%smid",
    "%tid",
    "%warpid",
)
_CACHE_HINTS = frozenset({"ca", "cg", "cs", "lu", "cv", "nc", "wb", "wt"})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _balanced(text: str, opening: int, left: str, right: str) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == left:
            depth += 1
        elif text[index] == right:
            depth -= 1
            if depth == 0:
                return index
    raise PTXObservationError(f"unterminated PTX {left}{right} region")


def _strip_comments(ptx: str) -> str:
    ptx = re.sub(r"/\*.*?\*/", "", ptx, flags=re.DOTALL)
    return "\n".join(line.split("//", 1)[0] for line in ptx.splitlines())


def _entry_region(ptx: str, entry: str | None) -> tuple[str, str, str]:
    matches = list(_ENTRY_RE.finditer(ptx))
    if entry is not None:
        matches = [item for item in matches if item.group("name") == entry]
    if len(matches) != 1:
        detail = "requested entry" if entry else "unique entry"
        raise PTXObservationError(f"PTX must contain one {detail}; found {len(matches)}")
    match = matches[0]
    params_end = _balanced(ptx, match.end() - 1, "(", ")")
    body_start = ptx.find("{", params_end)
    if body_start < 0:
        raise PTXObservationError("PTX entry has no body")
    body_end = _balanced(ptx, body_start, "{", "}")
    return (
        match.group("name"),
        ptx[match.end() : params_end],
        ptx[body_start + 1 : body_end],
    )


def _register_types(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in _REGISTER_DECL_RE.finditer(body):
        base = match.group("base")
        dtype = match.group("type").removeprefix(".")
        count = int(match.group("count") or 1)
        if count == 1:
            result[base] = dtype
            continue
        for index in range(count):
            result[f"{base}{index}"] = dtype
    return result


def _statements(body: str) -> list[str]:
    lines = [line.strip() for line in body.splitlines()]
    output: list[str] = []
    current = ""
    for line in lines:
        if not line:
            continue
        if line.endswith(":"):
            if current:
                raise PTXObservationError("label occurs inside PTX instruction")
            output.append(line)
            continue
        if line.startswith((".loc", ".file", ".pragma", ".reg")):
            continue
        if line.startswith((".shared", ".local")) and line.endswith(";"):
            output.append(re.sub(r"\s+", " ", line))
            continue
        current = (current + " " + line).strip()
        while ";" in current:
            statement, current = current.split(";", 1)
            if statement.strip():
                output.append(statement.strip() + ";")
            current = current.strip()
    if current and current not in {"{", "}"}:
        raise PTXObservationError("unterminated PTX instruction")
    return output


def _is_special_register(token: str) -> bool:
    return any(token == base or token.startswith(base + ".") for base in _SPECIAL_REGISTERS)


def _canonical_instructions(
    body: str,
    *,
    parameters: Mapping[str, str],
) -> tuple[str, ...]:
    register_types = _register_types(body)
    registers: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    functions: dict[str, str] = {}

    def register(match: re.Match[str]) -> str:
        token = match.group(0)
        if _is_special_register(token):
            return token
        if token not in registers:
            dtype = register_types.get(token, "v")
            index = type_counts[dtype]
            type_counts[dtype] += 1
            registers[token] = f"%{dtype}{index}"
        return registers[token]

    def parameter_names(text: str) -> str:
        for source, canonical in sorted(parameters.items(), key=lambda item: -len(item[0])):
            text = re.sub(rf"(?<![A-Za-z0-9_.$]){re.escape(source)}(?![A-Za-z0-9_.$])", canonical, text)
        return text

    def label_name(source: str) -> str:
        if source not in labels:
            labels[source] = f"L{len(labels)}"
        return labels[source]

    output: list[str] = []
    for raw in _statements(body):
        if raw.endswith(":"):
            output.append(label_name(raw[:-1].strip()) + ":")
            continue
        if raw.startswith((".shared", ".local")):
            declaration = parameter_names(raw)
            declaration = _REGISTER_RE.sub(register, declaration)
            declaration = re.sub(
                r"(?P<prefix>\.(?:shared|local)(?:\s+\.align\s+[0-9]+)?\s+\.[A-Za-z0-9]+\s+)"
                r"[A-Za-z_.$][A-Za-z0-9_.$]*",
                lambda match: match.group("prefix") + f"mem{len(output)}",
                declaration,
            )
            output.append(re.sub(r"\s+", " ", declaration.strip()))
            continue
        match = _OPCODE_RE.match(raw)
        if match is None:
            raise PTXObservationError(f"cannot parse PTX instruction: {raw!r}")
        opcode = match.group("opcode")
        predicate = match.group("predicate") or ""
        operands = parameter_names(match.group("operands"))
        predicate = _REGISTER_RE.sub(register, predicate)
        operands = _REGISTER_RE.sub(register, operands)
        if opcode.startswith(("bra", "brx")):
            operands = re.sub(
                r"(?<![A-Za-z0-9_.$%])([A-Za-z_$][A-Za-z0-9_.$]*)(?![A-Za-z0-9_.$])",
                lambda item: label_name(item.group(1)),
                operands,
                count=1,
            )
        if opcode.startswith("call"):
            def normalize_function(item: re.Match[str]) -> str:
                name = item.groupdict().get("name") or item.group(0)
                if name.startswith("__nv_"):
                    return name
                if name not in functions:
                    functions[name] = f"fn{len(functions)}"
                return functions[name]

            # PTX calls spell the callee between an optional result tuple and
            # the argument tuple.  Normalize only that symbol; register names
            # and type tokens inside the tuples remain proof-relevant.
            operands = re.sub(
                r"(?P<prefix>(?:\)\s*,\s*|^))"
                r"(?P<name>(?:_Z)?[A-Za-z_][A-Za-z0-9_.$]*)"
                r"(?=\s*,\s*\()",
                lambda item: item.group("prefix")
                + normalize_function(item),
                operands,
                count=1,
            )
        operands = re.sub(r"\s+", " ", operands.strip())
        operands = re.sub(r"\s*,\s*", ",", operands)
        operands = re.sub(r"\[\s*", "[", operands)
        operands = re.sub(r"\s*\]", "]", operands)
        output.append(f"{predicate + ' ' if predicate else ''}{opcode}{(' ' + operands) if operands else ''};")
    if not output:
        raise PTXObservationError("PTX entry instruction sequence is empty")
    return tuple(output)


def _opcode(instruction: str) -> str:
    value = instruction
    if value.endswith(":") or value.startswith("."):
        return value.split(" ", 1)[0]
    if value.startswith("@"):
        value = value.split(" ", 1)[1]
    return value.split(" ", 1)[0]


def _category(instructions: tuple[str, ...], prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(item for item in instructions if _opcode(item).startswith(prefixes))


def _split_operands(operands: str) -> tuple[str, ...]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, character in enumerate(operands):
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if stack and stack[-1] == pairs[character]:
                stack.pop()
        elif character == "," and not stack:
            result.append(operands[start:index].strip())
            start = index + 1
    tail = operands[start:].strip()
    if tail:
        result.append(tail)
    return tuple(result)


def _instruction_parts(
    instruction: str,
) -> tuple[str, str, tuple[str, ...]] | None:
    if instruction.endswith(":") or instruction.startswith("."):
        return None
    match = _OPCODE_RE.match(instruction)
    if match is None:
        return None
    return (
        match.group("predicate") or "",
        match.group("opcode"),
        _split_operands(match.group("operands")),
    )


def _semantic_memory_opcode(opcode: str) -> str:
    parts = opcode.split(".")
    return ".".join(
        part for index, part in enumerate(parts) if index == 0 or part not in _CACHE_HINTS
    )


def _semantic_memory_effects(
    instructions: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    effects: Counter[str] = Counter()
    for instruction in instructions:
        parts = _instruction_parts(instruction)
        if parts is None:
            continue
        predicate, opcode, _operands = parts
        if not opcode.startswith(("ld.", "st.", "atom.", "red.", "suld.", "sust.")):
            continue
        polarity = "not_predicated:" if predicate.startswith("@!") else "predicated:" if predicate else ""
        effects[polarity + _semantic_memory_opcode(opcode)] += 1
    return tuple(sorted(effects.items()))


def _semantic_value_opcodes(
    instructions: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    """Retain operations that produce observable stored/control values.

    Loads are leaves of the value graph, so pointer arithmetic feeding their
    addresses is deliberately excluded.  Integer arithmetic still remains hard
    whenever it feeds a stored value or a memory predicate.
    """

    records = [_instruction_parts(item) for item in instructions]
    definitions: dict[str, int] = {}
    no_destination = (
        "st.", "sust.", "red.", "bra", "brx", "ret", "exit", "bar.",
        "barrier.", "mbarrier.", "fence.", "membar.",
    )
    for index, record in enumerate(records):
        if record is None:
            continue
        _predicate, opcode, operands = record
        if not operands or opcode.startswith(no_destination):
            continue
        destination = operands[0]
        if destination.startswith("["):
            continue
        registers = _REGISTER_RE.findall(destination)
        if len(registers) == 1:
            definitions[registers[0]] = index

    roots: set[str] = set()
    for record in records:
        if record is None:
            continue
        predicate, opcode, operands = record
        roots.update(_REGISTER_RE.findall(predicate))
        if opcode.startswith(("st.", "sust.")) and operands:
            roots.update(_REGISTER_RE.findall(operands[-1]))
        elif opcode.startswith(("atom.", "red.")):
            for operand in operands:
                if not operand.startswith("["):
                    roots.update(_REGISTER_RE.findall(operand))
        elif opcode.startswith(("bra", "brx")):
            roots.update(_REGISTER_RE.findall(predicate))

    selected: set[int] = set()

    def select(register: str) -> None:
        index = definitions.get(register)
        if index is None or index in selected:
            return
        selected.add(index)
        record = records[index]
        assert record is not None
        predicate, opcode, operands = record
        for dependency in _REGISTER_RE.findall(predicate):
            select(dependency)
        # A load's address is not value dataflow. Its space/type/predicate is
        # already retained by the semantic memory-effect projection.
        if opcode.startswith(("ld.", "suld.")):
            return
        for operand in operands[1:]:
            for dependency in _REGISTER_RE.findall(operand):
                select(dependency)

    for root in roots:
        select(root)

    opcodes: Counter[str] = Counter()
    for index in sorted(selected):
        record = records[index]
        assert record is not None
        _predicate, opcode, _operands = record
        opcodes[_semantic_memory_opcode(opcode)] += 1
    return tuple(sorted(opcodes.items()))


@dataclass(frozen=True)
class PTXFingerprint:
    schema_version: str
    interface: tuple[tuple[str, str, int | None], ...]
    instructions: tuple[str, ...]
    opcode_multiset: tuple[tuple[str, int], ...]
    memory_effects: tuple[str, ...]
    semantic_value_opcodes: tuple[tuple[str, int], ...]
    semantic_memory_effects: tuple[tuple[str, int], ...]
    collectives: tuple[str, ...]
    control_flow: tuple[str, ...]
    coordinate_uses: tuple[str, ...]
    target_view: tuple[tuple[str, str], ...]
    schedule_view: tuple[tuple[str, str], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "interface": [list(item) for item in self.interface],
            "instructions": list(self.instructions),
            "opcode_multiset": [list(item) for item in self.opcode_multiset],
            "memory_effects": list(self.memory_effects),
            "semantic_value_opcodes": [
                list(item) for item in self.semantic_value_opcodes
            ],
            "semantic_memory_effects": [
                list(item) for item in self.semantic_memory_effects
            ],
            "collectives": list(self.collectives),
            "control_flow": list(self.control_flow),
            "coordinate_uses": list(self.coordinate_uses),
            "target_view": [list(item) for item in self.target_view],
            "schedule_view": [list(item) for item in self.schedule_view],
            "digest": self.digest,
        }


def fingerprint_ptx(
    ptx: str,
    *,
    entry: str | None = None,
    interface: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
) -> PTXFingerprint:
    if not isinstance(ptx, str) or not ptx.strip():
        raise PTXObservationError("PTX is empty")
    cleaned = _strip_comments(ptx)
    _name, params, body = _entry_region(cleaned, entry)
    abi: list[tuple[str, str, int | None]] = []
    param_names: dict[str, str] = {}
    for index, match in enumerate(_PARAM_RE.finditer(params)):
        dtype = match.group("type").removeprefix(".")
        extent = int(match.group("extent")) if match.group("extent") else None
        align = match.group("align") or ""
        abi.append((align, dtype, extent))
        param_names[match.group("name")] = f"param{index}"
    if params.strip() and not abi:
        raise PTXObservationError("PTX entry parameter ABI cannot be parsed")
    semantic_interface = dict(interface or {})
    semantic_interface["ptx_abi"] = abi
    interface_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in semantic_interface.items()
        )
    )
    # Store the complete semantic interface as string-valued rows while keeping
    # the public dataclass shape uniform with the raw PTX ABI rows.
    full_interface: tuple[tuple[str, str, int | None], ...] = tuple(
        (key, value, None) for key, value in interface_view
    )
    instructions = _canonical_instructions(body, parameters=param_names)
    opcodes = tuple(_opcode(item) for item in instructions if not item.endswith(":"))
    memory = _category(instructions, ("ld.", "st.", "atom.", "red.", "suld.", "sust."))
    semantic_value_opcodes = _semantic_value_opcodes(instructions)
    semantic_memory_effects = _semantic_memory_effects(instructions)
    collectives = _category(
        instructions,
        ("bar.", "barrier.", "mbarrier.", "cp.async", "fence.", "shfl.", "vote.", "match.", "mma.", "wgmma.", "elect."),
    )
    control = tuple(
        item
        for item in instructions
        if item.endswith(":") or _opcode(item).startswith(("bra", "brx", "call", "ret", "exit"))
    )
    coordinates = tuple(
        token
        for item in instructions
        for token in _REGISTER_RE.findall(item)
        if _is_special_register(token)
    )
    target_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (target or {}).items()
        )
    )
    schedule_view = tuple(
        sorted(
            (str(key), json.dumps(value, sort_keys=True, separators=(",", ":")))
            for key, value in (schedule or {}).items()
        )
    )
    payload = {
        "schema": PTX_FINGERPRINT_SCHEMA,
        "interface": full_interface,
        "instructions": instructions,
        "memory": memory,
        "semantic_value_opcodes": semantic_value_opcodes,
        "semantic_memory_effects": semantic_memory_effects,
        "collectives": collectives,
        "control": control,
        "coordinates": coordinates,
        "target": target_view,
    }
    return PTXFingerprint(
        PTX_FINGERPRINT_SCHEMA,
        full_interface,
        instructions,
        tuple(sorted(Counter(opcodes).items())),
        memory,
        semantic_value_opcodes,
        semantic_memory_effects,
        collectives,
        control,
        coordinates,
        target_view,
        schedule_view,
        _digest(payload),
    )


@dataclass(frozen=True)
class PTXDifference:
    category: str
    source: Any
    replay: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "source": self.source,
            "replay": self.replay,
            "message": self.message,
        }


@dataclass(frozen=True)
class PTXComparison:
    hard_pass: bool
    hard_passes: Mapping[str, bool]
    diagnostic_passes: Mapping[str, bool]
    differences: tuple[PTXDifference, ...]
    source_digest: str
    replay_digest: str

    @property
    def primary_difference_category(self) -> str | None:
        return next(
            (
                item.category
                for item in self.differences
                if item.category in self.hard_passes
                and not self.hard_passes[item.category]
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_pass": self.hard_pass,
            "hard_passes": dict(self.hard_passes),
            "diagnostic_passes": dict(self.diagnostic_passes),
            "primary_difference_category": self.primary_difference_category,
            "differences": [item.to_dict() for item in self.differences],
            "source_digest": self.source_digest,
            "replay_digest": self.replay_digest,
        }


def compare_ptx_fingerprints(
    source: PTXFingerprint,
    replay: PTXFingerprint,
) -> PTXComparison:
    fields = (
        ("schema", source.schema_version, replay.schema_version),
        ("interface", source.interface, replay.interface),
        (
            "value_operations",
            source.semantic_value_opcodes,
            replay.semantic_value_opcodes,
        ),
        (
            "memory_effects",
            source.semantic_memory_effects,
            replay.semantic_memory_effects,
        ),
        ("collectives", source.collectives, replay.collectives),
        ("control_flow", source.control_flow, replay.control_flow),
        ("coordinate_uses", source.coordinate_uses, replay.coordinate_uses),
        ("target", source.target_view, replay.target_view),
    )
    passes = {category: expected == actual for category, expected, actual in fields}
    diagnostic_fields = (
        ("raw_instruction_topology", source.instructions, replay.instructions),
        ("raw_opcode_multiset", source.opcode_multiset, replay.opcode_multiset),
        ("raw_memory_effects", source.memory_effects, replay.memory_effects),
    )
    diagnostic_passes = {
        category: expected == actual
        for category, expected, actual in diagnostic_fields
    }
    differences = tuple(
        PTXDifference(
            category,
            expected,
            actual,
            f"PTX {category} differs between Source and Replay",
        )
        for category, expected, actual in (*fields, *diagnostic_fields)
        if expected != actual
    )
    return PTXComparison(
        all(passes.values()),
        passes,
        diagnostic_passes,
        differences,
        source.digest,
        replay.digest,
    )


class PTXComparator:
    def compare(self, source: PTXFingerprint, replay: PTXFingerprint) -> PTXComparison:
        return compare_ptx_fingerprints(source, replay)


__all__ = [
    "PTXComparator",
    "PTXComparison",
    "PTXDifference",
    "PTXFingerprint",
    "PTXObservationError",
    "PTX_FINGERPRINT_SCHEMA",
    "compare_ptx_fingerprints",
    "fingerprint_ptx",
]
