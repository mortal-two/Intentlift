#!/usr/bin/env python3
"""Reproduce Task 33's line-level frontend semantic-SLoC ledger.

The audit is intentionally repository-local and deterministic.  It records
every line event in the two frozen history windows, every line in the current
production frontend footprint, and every line in the four Task 28 authoring
modules.  Classification is by path and Python symbol, never by a measured
outcome.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import subprocess
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = TASK_DIR / "semantic_loc_audit"

D307_BASE = "01d42b287c02146c64a6f2d398c909df04dc0121"
D307_HEAD = "d30753e7a1c7aafbd6d28632bf9e46e119249efc"
FINAL_FIVE_BASE = "b4f1b6dabb1f071d093c010ee6c18d2a251e69d9"
FINAL_FIVE_HEAD = "df95382c57ffb406d933e11ef93c0ff170c97ecb"
TASK28_HEAD = "9c437c907a98a600152f95fe35f3b62dfaca37cd"
CURRENT_FOOTPRINT_HEAD = "1ce53f68391c7da57c112926c46dc331f649c9e1"

BUCKETS = (
    "frontend_semantic",
    "frontend_execution_adapter",
    "observer_semantic",
    "public_context",
    "shared_contract",
    "target_consumer",
    "per_kernel_persistent",
    "declarative_data",
    "artifact_output",
    "test_evidence_harness",
    "docs_generated_vendor",
)
CORE_BUCKETS = frozenset(
    {"frontend_semantic", "observer_semantic", "per_kernel_persistent"}
)

# Gate artifacts and agent trajectories are already immutable evidence.  Their
# full raw size is reported from git numstat, but expanding tens of thousands
# of JSON rows into a semantic ownership ledger would duplicate artifacts and
# falsely suggest that generated output is code under audit.
LINE_LEDGER_EXCLUDED_MARKERS = ("/runs/",)

PRODUCTION_FRONTEND_PATHS = (
    "python/intentir/verify/adapters.py",
    "python/intentir/verify/frontend_dataset.py",
    "python/intentir/verify/frontend_workloads.py",
    "python/intentir/verify/frontend_tensor_map_sidecars.json",
    "python/intentir/verify/workload_registry.py",
    "python/intentir/verify/gate1/_tilelang_worker.py",
    "python/intentir/verify/gate1/cuda_codegen.py",
    "python/intentir/verify/gate1/cuda_compiler.py",
    "python/intentir/verify/gate1/frontend.py",
    "python/intentir/verify/gate1/native_orchestrate.py",
    "python/intentir/verify/gate1/native_semantic.py",
    "python/intentir/verify/gate1/profile.py",
    "python/intentir/verify/gate1/prompt.py",
    "python/intentir/verify/gate1/ptx.py",
    "python/intentir/verify/gate1/replay.py",
    "python/intentir/verify/gate1/tilelang.py",
    "python/intentir/verify/gate1/tilelang_codegen.py",
    "python/intentir/verify/gate1/tilelang_compiler.py",
    "python/intentir/verify/gate2/native.py",
)

TASK28_AUTHORING_PATHS = (
    ".trellis/tasks/08-01-j-only-replay-multifrontend-batch-28/manual_g1_artifacts.py",
    ".trellis/tasks/08-01-j-only-replay-multifrontend-batch-28/manual_g2_artifacts.py",
    ".trellis/tasks/08-01-j-only-replay-multifrontend-batch-28/manual_descriptor_artifacts.py",
    ".trellis/tasks/08-01-j-only-replay-multifrontend-batch-28/manual_g3_g7_artifacts.py",
)

# These are semantic J constructors, not ABI/signature glue or receipts.  A
# constructor shared by several corpus members is charged at its first use in
# the frozen Task 33 order; the note column makes that reuse explicit.
KERNEL_CONSTRUCTORS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "add2d": ((TASK28_AUTHORING_PATHS[0], "_add2d_j"),),
    "floor2d": ((TASK28_AUTHORING_PATHS[0], "_unary_2d_j"),),
    "row_sum": ((TASK28_AUTHORING_PATHS[1], "_row_reduce_j"),),
    "layernorm": ((TASK28_AUTHORING_PATHS[1], "_layernorm_j"),),
    "masked_softmax2d": ((TASK28_AUTHORING_PATHS[1], "_masked_softmax_j"),),
    "gemm": ((TASK28_AUTHORING_PATHS[2], "_matmul_relu_j"),),
    "matmul_fused_epilogue2d": ((TASK28_AUTHORING_PATHS[2], "_fused_epilogue_j"),),
    "gather2d": (
        (TASK28_AUTHORING_PATHS[3], "_gather_tilelang_j"),
        (TASK28_AUTHORING_PATHS[3], "_gather_cuda_j"),
    ),
    "transpose2d": ((TASK28_AUTHORING_PATHS[3], "_transpose_j"),),
    "flash_attention2d": ((TASK28_AUTHORING_PATHS[3], "_attention_2d_j"),),
    "any_kernel_dim": ((TASK28_AUTHORING_PATHS[0], "_any_kernel_dim_j"),),
    "rowmask_where2d": ((TASK28_AUTHORING_PATHS[0], "_rowmask_j"),),
}

SHARED_CONSTRUCTOR_NOTES = {
    "_unary_2d_j": "generic unary constructor; first charged at floor2d",
    "_row_reduce_j": "generic row-reduction constructor; first charged at row_sum",
    "_layernorm_j": "parameterized width/residual constructor",
    "_matmul_relu_j": "parameterized IO-dtype matmul/ReLU constructor",
    "_attention_2d_j": "parameterized attention-size constructor",
}

TASK28_SEMANTIC_SYMBOLS = frozenset(
    symbol for rows in KERNEL_CONSTRUCTORS.values() for _path, symbol in rows
)

FRONTEND_WORKLOAD_SEMANTIC_SYMBOLS = frozenset(
    {
        "_logical_abi_parameters",
        "_source_shape_contract",
        "_parameter_alias",
        "_metadata_rows_in_source_order",
        "_metadata_parameters",
        "_shape_coordinate_mapping",
        "_grid_shape_coordinates",
        "_normalize_dataset_dtype",
        "_cuda_pointer_mode",
        "_handwritten_runtime_metadata",
        "_runtime_metadata",
    }
)


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    first: int
    last: int
    depth: int


@dataclass(frozen=True)
class SourceFacts:
    lines: tuple[str, ...]
    code_lines: frozenset[int]
    symbols: tuple[SymbolSpan, ...]

    def symbol_at(self, line: int) -> str:
        candidates = [item for item in self.symbols if item.first <= line <= item.last]
        if not candidates:
            return "<module>"
        winner = max(candidates, key=lambda item: (item.depth, -(item.last - item.first)))
        return winner.name


@dataclass(frozen=True)
class LedgerRow:
    window: str
    base_commit: str
    head_commit: str
    change_kind: str
    path: str
    line_number: int
    symbol: str
    frontend: str
    bucket: str
    capability: str
    is_nonblank_noncomment: bool
    included_semantic_core: bool
    rationale: str
    content_sha256: str


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def _blob(revision: str, path: str) -> str:
    return _git("show", f"{revision}:{path}")


def _docstring_lines(tree: ast.AST) -> set[int]:
    result: set[int] = set()
    owners = [tree, *(node for node in ast.walk(tree) if isinstance(
        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ))]
    for owner in owners:
        body = getattr(owner, "body", ())
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            result.update(range(first.lineno, first.end_lineno + 1))
    return result


def _python_facts(text: str) -> SourceFacts:
    lines = tuple(text.splitlines())
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _generic_facts(text)

    meaningful: set[int] = set()
    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        ignored = {
            tokenize.COMMENT,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
        }
        for token in tokens:
            if token.type in ignored:
                continue
            for line in range(token.start[0], token.end[0] + 1):
                meaningful.add(line)
    except (IndentationError, tokenize.TokenError):
        return _generic_facts(text)
    meaningful.difference_update(_docstring_lines(tree))

    spans: list[SymbolSpan] = []

    def visit(node: ast.AST, prefix: str, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{child.name}" if prefix else child.name
                spans.append(SymbolSpan(name, child.lineno, child.end_lineno, depth))
                visit(child, name, depth + 1)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                names = [target.id for target in targets if isinstance(target, ast.Name)]
                # Module constants need an addressable symbol for declarative
                # tables.  Local assignments must retain their owning function;
                # otherwise a constructor body would be fragmented into dozens
                # of meaningless local-variable "owners".
                if names and not prefix:
                    name = names[0]
                    spans.append(SymbolSpan(name, child.lineno, child.end_lineno, depth))
                visit(child, prefix, depth)
            else:
                visit(child, prefix, depth)

    visit(tree, "", 1)
    return SourceFacts(lines, frozenset(meaningful), tuple(spans))


def _generic_facts(text: str) -> SourceFacts:
    lines = tuple(text.splitlines())
    code: set[int] = set()
    in_block_comment = False
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            in_block_comment = "*/" not in stripped
            continue
        if stripped.startswith(("#", "//", "*", "*/")):
            continue
        code.add(number)
    return SourceFacts(lines, frozenset(code), ())


def _source_facts(path: str, text: str) -> SourceFacts:
    return _python_facts(text) if path.endswith(".py") else _generic_facts(text)


def _frontend(path: str, symbol: str, content: str) -> str:
    normalized = path.lower()
    symbol_lower = symbol.lower()
    if normalized.endswith("/gate1/ptx.py"):
        return "cuda"
    if normalized.endswith("/gate1/tilelang.py"):
        return "tilelang"
    if path in TASK28_AUTHORING_PATHS:
        if "cuda" in symbol_lower:
            return "cuda"
        if "tilelang" in symbol_lower:
            return "tilelang"
        return "paired"
    filename = Path(normalized).name
    if "cuda" in filename and "tilelang" not in filename:
        return "cuda"
    if "tilelang" in filename and "cuda" not in filename:
        return "tilelang"
    haystack = f"{symbol} {content}".lower()
    has_tilelang = "tilelang" in haystack or "tir" in haystack
    has_cuda = "cuda" in haystack or "ptx" in haystack or "tensor_map" in haystack
    if has_tilelang and not has_cuda:
        return "tilelang"
    if has_cuda and not has_tilelang:
        return "cuda"
    if has_cuda and has_tilelang:
        return "paired"
    return "shared"


def _task28_bucket(symbol: str) -> tuple[str, str, str]:
    leaf = symbol.split(".")[-1]
    if leaf in TASK28_SEMANTIC_SYMBOLS:
        return (
            "per_kernel_persistent",
            "manual_j_semantic_constructor",
            "Task 28 source-aware J constructor retained as per-kernel authoring code",
        )
    if leaf == "_signature":
        return (
            "frontend_semantic",
            "manual_frontend_abi_mapping",
            "frontend-specific logical signature/ABI ownership",
        )
    if leaf == "authoring_receipt":
        return (
            "artifact_output",
            "authoring_receipt",
            "receipt generation is evidence output, not semantic mechanism",
        )
    if leaf in {"CASE_IDS", "CASE_BUILDERS", "CASE_RECEIPTS", "_J_BUILDERS", "_BLOCK_X"}:
        return (
            "declarative_data",
            "manual_case_registration",
            "case registry/launch values are declarative experiment data",
        )
    return (
        "shared_contract",
        "manual_authoring_infrastructure",
        "shared J/C/pi construction infrastructure, counted once outside per-kernel cost",
    )


def _classify(path: str, symbol: str, content: str) -> tuple[str, str, str]:
    normalized = path.lower()
    leaf = symbol.split(".")[-1]
    if path in TASK28_AUTHORING_PATHS:
        return _task28_bucket(symbol)
    if (
        normalized.startswith("doc/")
        or normalized.startswith("docs/")
        or "/archive/" in normalized
        or normalized.endswith((".md", ".rst", ".pdf"))
        or "generated" in normalized and not normalized.endswith("frontend_tensor_map_sidecars.json")
        or "/vendor/" in normalized
    ):
        return (
            "docs_generated_vendor",
            "excluded_documentation_or_generated",
            "documentation/generated/vendor lines are outside semantic engineering cost",
        )
    if (
        normalized.startswith("python/tests/")
        or "/tests/" in normalized
        or normalized.startswith(".trellis/tasks/")
        or normalized.startswith("tools/")
        or normalized.startswith("scripts/")
    ):
        return (
            "test_evidence_harness",
            "excluded_test_or_experiment_harness",
            "test/CLI/runner/evidence code is reported but excluded from semantic core",
        )
    if normalized.endswith((".json", ".yaml", ".yml", ".toml")):
        return (
            "declarative_data",
            "frontend_declarative_metadata",
            "declarative ABI/launch/sidecar data is not executable semantic mechanism",
        )
    if normalized.endswith("/gate1/ptx.py"):
        return (
            "observer_semantic",
            "cuda_ptx_observer",
            "PTX parsing and semantic fingerprinting define the CUDA observation edge",
        )
    if normalized.endswith("/gate1/tilelang.py"):
        return (
            "observer_semantic",
            "tilelang_tir_observer",
            "TIR normalization and fingerprinting define the TileLang observation edge",
        )
    if normalized.endswith("/gate1/native_semantic.py"):
        return (
            "observer_semantic",
            "native_observation_semantics",
            "frontend-native opcode/effect normalization belongs to observation semantics",
        )
    if normalized.endswith("/frontend_workloads.py") and leaf in FRONTEND_WORKLOAD_SEMANTIC_SYMBOLS:
        return (
            "frontend_semantic",
            "typed_abi_and_shape_projection",
            "source-grounded logical ABI/effect/shape projection owns frontend meaning",
        )
    if normalized.endswith("/gate1/prompt.py"):
        return (
            "public_context",
            "frontend_public_context",
            "public frontend/API context assists an agent but is not executable semantics",
        )
    if normalized.endswith("/frontend_dataset.py"):
        if leaf in {"_tilelang_kernels", "_cuda_kernels"}:
            return (
                "declarative_data",
                "dataset_catalog",
                "dataset membership and paths are declarative corpus data",
            )
        return (
            "shared_contract",
            "frontend_dataset_protocol",
            "frontend-neutral dataset catalog and validation contract",
        )
    if normalized.endswith("/frontend_workloads.py"):
        if leaf in {"_tensor_map_sidecars", "FRONTEND_DATASET_HARNESSES"}:
            return (
                "declarative_data",
                "frontend_invocation_sidecar",
                "finite invocation and TensorMap rows are explicit declarative data",
            )
        return (
            "frontend_execution_adapter",
            "dataset_execution_projection",
            "source materialization, invocation, and Gate-2 binding are execution adapters",
        )
    if normalized.endswith("/workload_registry.py"):
        return (
            "test_evidence_harness",
            "workload_registration",
            "experimental workload registration is evidence harness code",
        )
    adapter_suffixes = (
        "/gate1/_tilelang_worker.py",
        "/gate1/cuda_codegen.py",
        "/gate1/cuda_compiler.py",
        "/gate1/native_orchestrate.py",
        "/gate1/tilelang_codegen.py",
        "/gate1/tilelang_compiler.py",
        "/gate2/native.py",
        "/adapters.py",
    )
    if normalized.endswith(adapter_suffixes):
        return (
            "frontend_execution_adapter",
            "native_compile_execute_adapter",
            "frontend compiler/runner/ABI bridge executes observations without deriving J semantics",
        )
    if normalized.endswith("/gate1/profile.py"):
        return (
            "shared_contract",
            "observer_profile_contract",
            "observer profile identifiers and policies are shared protocol metadata",
        )
    if normalized.endswith("/gate1/frontend.py"):
        return (
            "shared_contract",
            "frontend_capability_registry",
            "frontend capability registration is shared protocol plumbing",
        )
    if normalized.endswith("/gate1/replay.py"):
        return (
            "shared_contract",
            "replay_frontend_dispatch",
            "Replay validation/extraction dispatch is shared gate contract, not a kernel rule",
        )
    if normalized.startswith("python/intentir/"):
        return (
            "shared_contract",
            "intentir_shared_infrastructure",
            "production IntentIR code outside the audited frontend-specific semantic symbols",
        )
    return (
        "docs_generated_vendor",
        "outside_frontend_audit_scope",
        "line is retained for raw-window completeness but lies outside production frontend scope",
    )


def _line_row(
    *,
    window: str,
    base: str,
    head: str,
    kind: str,
    path: str,
    number: int,
    content: str,
    facts: SourceFacts,
) -> LedgerRow:
    symbol = facts.symbol_at(number)
    bucket, capability, rationale = _classify(path, symbol, content)
    if bucket not in BUCKETS:
        raise AssertionError(f"unknown bucket {bucket}")
    code = number in facts.code_lines
    return LedgerRow(
        window,
        base,
        head,
        kind,
        path,
        number,
        symbol,
        _frontend(path, symbol, content),
        bucket,
        capability,
        code,
        code and bucket in CORE_BUCKETS,
        rationale,
        hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _diff_rows(window: str, base: str, head: str) -> list[LedgerRow]:
    patch = _git(
        "diff",
        "--no-ext-diff",
        "--no-renames",
        "--unified=0",
        "--no-color",
        base,
        head,
    )
    blob_cache: dict[tuple[str, str], SourceFacts] = {}

    def facts(revision: str, path: str) -> SourceFacts:
        key = (revision, path)
        if key not in blob_cache:
            blob_cache[key] = _source_facts(path, _blob(revision, path))
        return blob_cache[key]

    old_path = ""
    new_path = ""
    old_line = 0
    new_line = 0
    in_hunk = False
    result: list[LedgerRow] = []
    hunk = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            in_hunk = False
            continue
        if not in_hunk and line.startswith("--- "):
            raw = line[4:]
            old_path = "" if raw == "/dev/null" else raw.removeprefix("a/")
            continue
        if not in_hunk and line.startswith("+++ "):
            raw = line[4:]
            new_path = "" if raw == "/dev/null" else raw.removeprefix("b/")
            continue
        match = hunk.match(line)
        if match:
            old_line = int(match.group(1))
            new_line = int(match.group(2))
            in_hunk = True
            continue
        if not in_hunk or line.startswith("\\ No newline"):
            continue
        prefix, content = line[:1], line[1:]
        if prefix == "-":
            if not old_path:
                raise AssertionError("deleted line without old path")
            if not any(marker in f"/{old_path}" for marker in LINE_LEDGER_EXCLUDED_MARKERS):
                result.append(
                    _line_row(
                        window=window,
                        base=base,
                        head=head,
                        kind="deleted",
                        path=old_path,
                        number=old_line,
                        content=content,
                        facts=facts(base, old_path),
                    )
                )
            old_line += 1
        elif prefix == "+":
            if not new_path:
                raise AssertionError("added line without new path")
            if not any(marker in f"/{new_path}" for marker in LINE_LEDGER_EXCLUDED_MARKERS):
                result.append(
                    _line_row(
                        window=window,
                        base=base,
                        head=head,
                        kind="added",
                        path=new_path,
                        number=new_line,
                        content=content,
                        facts=facts(head, new_path),
                    )
                )
            new_line += 1
        elif prefix == " ":
            old_line += 1
            new_line += 1
    return result


def _snapshot_rows(
    window: str, revision: str, paths: Sequence[str]
) -> list[LedgerRow]:
    result: list[LedgerRow] = []
    for path in paths:
        text = _blob(revision, path)
        facts = _source_facts(path, text)
        for number, content in enumerate(facts.lines, 1):
            result.append(
                _line_row(
                    window=window,
                    base=revision,
                    head=revision,
                    kind="present",
                    path=path,
                    number=number,
                    content=content,
                    facts=facts,
                )
            )
    return result


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ledger_dict(row: LedgerRow) -> dict[str, object]:
    return {
        "window": row.window,
        "base_commit": row.base_commit,
        "head_commit": row.head_commit,
        "change_kind": row.change_kind,
        "path": row.path,
        "line_number": row.line_number,
        "symbol": row.symbol,
        "frontend": row.frontend,
        "bucket": row.bucket,
        "capability": row.capability,
        "is_nonblank_noncomment": int(row.is_nonblank_noncomment),
        "included_semantic_core": int(row.included_semantic_core),
        "rationale": row.rationale,
        "content_sha256": row.content_sha256,
    }


def _summary(rows: Sequence[LedgerRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[LedgerRow]] = defaultdict(list)
    for row in rows:
        if row.window == "task28_authoring_footprint":
            continue
        grouped[
            (
                row.window,
                row.base_commit,
                row.head_commit,
                row.frontend,
                row.bucket,
                row.capability,
            )
        ].append(row)
    result: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        window, base, head, frontend, bucket, capability = key
        result.append(
            {
                "window": window,
                "base_commit": base,
                "head_commit": head,
                "frontend": frontend,
                "bucket": bucket,
                "capability": capability,
                "raw_added": sum(item.change_kind == "added" for item in values),
                "raw_deleted": sum(item.change_kind == "deleted" for item in values),
                "raw_present": sum(item.change_kind == "present" for item in values),
                "nonblank_noncomment_added": sum(
                    item.change_kind == "added" and item.is_nonblank_noncomment
                    for item in values
                ),
                "nonblank_noncomment_deleted": sum(
                    item.change_kind == "deleted" and item.is_nonblank_noncomment
                    for item in values
                ),
                "nonblank_noncomment_present": sum(
                    item.change_kind == "present" and item.is_nonblank_noncomment
                    for item in values
                ),
                "semantic_core_added": sum(
                    item.change_kind == "added" and item.included_semantic_core
                    for item in values
                ),
                "semantic_core_deleted": sum(
                    item.change_kind == "deleted" and item.included_semantic_core
                    for item in values
                ),
                "semantic_core_present": sum(
                    item.change_kind == "present" and item.included_semantic_core
                    for item in values
                ),
                "files": len({item.path for item in values}),
            }
        )
    return result


def _symbol_span(path: str, symbol: str) -> tuple[SourceFacts, SymbolSpan]:
    facts = _source_facts(path, _blob(TASK28_HEAD, path))
    matches = [item for item in facts.symbols if item.name == symbol]
    if len(matches) != 1:
        raise AssertionError(f"expected one {path}:{symbol}, got {len(matches)}")
    return facts, matches[0]


def _kernel_summary() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    cumulative = 0
    for order, (kernel, constructors) in enumerate(KERNEL_CONSTRUCTORS.items(), 1):
        raw = 0
        code = 0
        locations: list[str] = []
        notes: list[str] = []
        frontends: set[str] = set()
        for path, symbol in constructors:
            facts, span = _symbol_span(path, symbol)
            raw += span.last - span.first + 1
            code += sum(line in facts.code_lines for line in range(span.first, span.last + 1))
            locations.append(f"{path}:{span.first}-{span.last}:{symbol}")
            notes.append(SHARED_CONSTRUCTOR_NOTES.get(symbol, "kernel semantic constructor"))
            if symbol.endswith("_tilelang_j"):
                frontends.add("tilelang")
            elif symbol.endswith("_cuda_j"):
                frontends.add("cuda")
            else:
                frontends.add("paired")
        cumulative += code
        result.append(
            {
                "frozen_order": order,
                "kernel": kernel,
                "frontend_coverage": "+".join(sorted(frontends)),
                "constructor_count": len(constructors),
                "raw_constructor_lines": raw,
                "nonblank_noncomment_constructor_sloc": code,
                "cumulative_constructor_sloc": cumulative,
                "locations": ";".join(locations),
                "reuse_note": ";".join(notes),
                "measurement_scope": (
                    "J semantic constructor only; ABI/signature, shared helpers, receipts, "
                    "runner, and generated artifacts are separately classified in ledger"
                ),
            }
        )
    return result


def _numstat(base: str, head: str) -> tuple[int, int]:
    added = 0
    deleted = 0
    for line in _git("diff", "--numstat", base, head).splitlines():
        raw_added, raw_deleted, _path = line.split("\t", 2)
        if raw_added == "-" or raw_deleted == "-":
            continue
        added += int(raw_added)
        deleted += int(raw_deleted)
    return added, deleted


def _window_totals(rows: Sequence[LedgerRow]) -> list[dict[str, object]]:
    specifications = {
        "d307_joint_onboarding": (D307_BASE, D307_HEAD),
        "final_five_increment": (FINAL_FIVE_BASE, FINAL_FIVE_HEAD),
    }
    result: list[dict[str, object]] = []
    for window in (
        "d307_joint_onboarding",
        "current_frontend_footprint",
        "final_five_increment",
        "task28_authoring_footprint",
    ):
        selected = [row for row in rows if row.window == window]
        audited_added = sum(row.change_kind == "added" for row in selected)
        audited_deleted = sum(row.change_kind == "deleted" for row in selected)
        audited_present = sum(row.change_kind == "present" for row in selected)
        if window in specifications:
            full_added, full_deleted = _numstat(*specifications[window])
        else:
            full_added, full_deleted = 0, 0
        result.append(
            {
                "window": window,
                "full_git_raw_added": full_added,
                "full_git_raw_deleted": full_deleted,
                "line_audited_raw_added": audited_added,
                "line_audited_raw_deleted": audited_deleted,
                "snapshot_raw_present": audited_present,
                "generated_run_artifact_lines_excluded_from_line_ledger": (
                    max(0, full_added - audited_added)
                    + max(0, full_deleted - audited_deleted)
                ),
                "nonblank_noncomment_audited": sum(
                    row.is_nonblank_noncomment for row in selected
                ),
                "semantic_core_audited": sum(
                    row.included_semantic_core for row in selected
                ),
            }
        )
    return result


def _method_markdown(
    rows: Sequence[LedgerRow], totals: Sequence[Mapping[str, object]]
) -> str:
    table_rows = []
    for item in totals:
        table_rows.append(
            "| " + " | ".join(
                [
                    str(item["window"]),
                    str(item["full_git_raw_added"]),
                    str(item["full_git_raw_deleted"]),
                    str(item["line_audited_raw_added"]),
                    str(item["line_audited_raw_deleted"]),
                    str(item["snapshot_raw_present"]),
                    str(item["generated_run_artifact_lines_excluded_from_line_ledger"]),
                    str(item["nonblank_noncomment_audited"]),
                    str(item["semantic_core_audited"]),
                ]
            ) + " |"
        )
    return f"""# Task 33 line-level semantic-SLoC audit method

This directory is generated by `audit_semantic_loc.py`.  Re-run from the
repository root with:

```bash
python .trellis/tasks/08-02-frontend-extensibility-matched-baseline-33/audit_semantic_loc.py
```

## Frozen revisions

- Initial joint onboarding: `{D307_BASE}..{D307_HEAD}`.
- Final-five increment: `{FINAL_FIVE_BASE}..{FINAL_FIVE_HEAD}`.
- Task 28 authoring snapshot: `{TASK28_HEAD}`.
- Current production footprint: `{CURRENT_FOOTPRINT_HEAD}` (the committed
  Task 33 input state, frozen before this audit implementation).

## Counting rule

`raw_added`, `raw_deleted`, and `raw_present` count physical line events.
`nonblank_noncomment` uses Python tokenization, removes AST docstrings, and for
non-Python files removes blank/full-line comment rows.  Closing delimiters and
other executable syntax remain code.  Every row has exactly one ownership
bucket and a content hash; source text is not duplicated into the CSV.

The two history windows additionally retain whole-commit `git --numstat`
totals.  Immutable files below `runs/` are aggregated in
`window_raw_totals.csv` and are not duplicated line-by-line.  They are always
excluded generated evidence, never semantic code.  All non-run source,
test/CLI, task, documentation, and production lines remain line-audited.

The paper's **audited semantic core** is the union of:

- `frontend_semantic`: source-grounded ABI/effect/shape-to-logical ownership;
- `observer_semantic`: PTX/TIR/native observation semantics;
- `per_kernel_persistent`: manually retained J semantic constructors.

Execution adapters, public prompt context, shared contracts, target consumers,
declarative sidecars, receipts, tests/runners, docs, generated files, and vendor
code are reported separately and are not silently counted as semantic core.
This is a semantic ownership metric, not total implementation size.

## Window sanity totals

| window | full git + | full git - | audited + | audited - | snapshot present | run-artifact lines excluded | audited nonblank/noncomment | audited semantic core |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Kernel marginal table boundary

`kernel_authoring_summary.csv` counts only the Task 28 J semantic constructor
body charged at first use in the frozen 12-pair order.  It deliberately does
not claim that signatures, C/pi assembly, shared graph helpers, gate execution,
or receipts are free: those lines remain individually visible in
`semantic_loc_ledger.csv`.  This split prevents the historical 2,223 raw-line
authoring modules from being misreported as either all-semantic or all-generic.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = [
        *_diff_rows("d307_joint_onboarding", D307_BASE, D307_HEAD),
        *_snapshot_rows(
            "current_frontend_footprint",
            CURRENT_FOOTPRINT_HEAD,
            PRODUCTION_FRONTEND_PATHS,
        ),
        *_diff_rows("final_five_increment", FINAL_FIVE_BASE, FINAL_FIVE_HEAD),
        *_snapshot_rows("task28_authoring_footprint", TASK28_HEAD, TASK28_AUTHORING_PATHS),
    ]
    rows.sort(
        key=lambda item: (
            item.window,
            item.path,
            item.change_kind,
            item.line_number,
        )
    )
    ledger_fields = tuple(_ledger_dict(rows[0]))
    _write_csv(
        output / "semantic_loc_ledger.csv",
        ledger_fields,
        (_ledger_dict(row) for row in rows),
    )

    summary = _summary(rows)
    summary_fields = tuple(summary[0])
    _write_csv(output / "frontend_onboarding_summary.csv", summary_fields, summary)

    kernels = _kernel_summary()
    kernel_fields = tuple(kernels[0])
    _write_csv(output / "kernel_authoring_summary.csv", kernel_fields, kernels)

    totals = _window_totals(rows)
    total_fields = tuple(totals[0])
    _write_csv(output / "window_raw_totals.csv", total_fields, totals)

    (output / "audit_method.md").write_text(
        _method_markdown(rows, totals), encoding="utf-8"
    )
    manifest = {
        "protocol": "intentir.experiment.frontend-semantic-sloc-ledger.v1",
        "generated_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
        "row_count": len(rows),
        "semantic_core_row_count": sum(row.included_semantic_core for row in rows),
        "buckets": list(BUCKETS),
        "core_buckets": sorted(CORE_BUCKETS),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
