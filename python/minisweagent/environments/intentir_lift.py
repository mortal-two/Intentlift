"""Fail-closed filesystem sandbox for one IntentIR Lift invocation."""

from __future__ import annotations

import copy
import json
import hashlib
import os
import re
import site
import stat
import shutil
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel

from intentir.ir.canonical_construction import semantic_operation_wire_rows
from intentir.ir.intent_op_contract import (
    STRUCTURED_INTENT_OPS,
    _broadcast_projection,
    operation_attr_contract,
    operation_operands_match_arity,
)
from minisweagent.exceptions import (
    CandidateRejected,
    LiftPreparationFailed,
    Submitted,
)
from intentir.extraction.lifters.scrc_schema import (
    canonical_semantic_mcp_write_protocol,
    unique_access_guard_projection_axes,
)
from minisweagent.mcp_lift import (
    IntentIRLiftMCPServer,
    MCP_CONTEXT_ARGUMENTS,
    MCP_CONTEXT_VIEWS,
    MCP_FIXED_SUMMARY_VIEWS,
    MCP_SEMANTIC_WRITE_TOOL_NAMES,
    MCP_SCHEMA_LOOKUP_ITEM_CHARS,
    MCP_SELECTOR_CHARS,
    MAX_MCP_SCHEMA_LOOKUP_ITEMS,
    MAX_MCP_PATH_CHARS,
    MAX_MCP_QUERY_CHARS,
    MCPRequestError,
    mcp_resource_uris_for_views,
    mcp_resource_count_upper,
)
INTENTIR_LIFT_SUBMIT = "INTENTIR_LIFT_SUBMIT"
CANDIDATE_CHECK_PROTOCOL = "intentir.scrc.candidate-check.v1"
MULTI_ACTION_PROTOCOL = "intentir.agent-multi-action.v1"
MCP_SEMANTIC_FINALIZE_PROTOCOL = "intentir.mcp-semantic-finalize"
_LIFT_READ_ONLY_CONTEXT_COMMANDS = MCP_CONTEXT_VIEWS
_SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS = frozenset(
    {
        # The first brief remains intentionally small. A fresh owner may read
        # one exact fact family only when the source brief cannot settle its
        # current construction decision; this avoids re-embedding Source,
        # ABI, TTIR and transfer facts on every Lift.
        "source",
        "source_obligations",
        "semantic_transfer",
        "harness",
        "harness_abi",
        "source_ttir",
        "source_ttir_outline",
        "source_ttir_def_use",
    }
)
_SEMANTIC_ENTRY_CONTEXT_VIEWS = (
    "source_brief",
    *_SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS,
)
_SOURCE_OBLIGATION_SECTIONS = frozenset(
    {
        "scopes",
        "program_axes",
        "index_construction",
        "mask_uses",
        "fills",
        "casts",
        "memory_accesses",
        "control",
        "helper_calls",
        "expression_facts",
    }
)
MAX_CANDIDATE_BYTES = 4 * 1024 * 1024
MAX_ACTION_OUTPUT_BYTES = 1024 * 1024
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024
MAX_WORKSPACE_ENTRIES = 2048
MAX_WORKSPACE_DEPTH = 64
MAX_CANDIDATE_CHECK_FIELD_CHARS = 4096
MAX_CANDIDATE_CHECK_OBSERVED_ITEMS = 20
MAX_CANDIDATE_CHECK_RESULT_BYTES = 128 * 1024
MAX_STARTUP_CONTEXT_BYTES = 64 * 1024
_SANDBOX_START_MARKER = b"__INTENTIR_LIFT_SANDBOX_STARTED__\n"
# A fresh semantic graph normally needs one declaration.  With the fixed
# 16-turn budget, Source brief + transfer + one bounded lookup + begin take
# four turns and access/output/finalize take three more.  Permit five changed
# graph corrections (the sixth failure terminates) so a provider that receives
# one precise receipt at a time can repair independent operand, shape, and
# attribute mistakes without consuming the lifecycle's required final steps.
# This remains a finite, observable preparation boundary; it is not a second
# candidate, a repair base, or an increase to the model step limit.
_MAX_OPERATION_GRAPH_DECLARATION_FAILURES = 6
# Access declarations have the same bounded pre-candidate lifecycle as an
# operation graph.  Without this boundary a provider can keep resubmitting an
# empty or prefix array after the begin receipt has already supplied every
# Source-order slot, exhausting the physical-call cap without ever reaching
# finalize.  This is an authoring-preparation limit only: it neither accepts a
# partial access array nor changes CandidateCheck.
_MAX_ACCESS_DECLARATION_FAILURES = 4
_PROVIDER_CORRECTABLE_ACCESS_DECLARATION_DIAGNOSTICS = frozenset(
    {
        "accesses_must_be_json_array",
        "access_declaration_must_cover_each_source_occurrence",
        "access_declaration_rows_must_be_json_objects",
        "access_declaration_must_follow_source_ordinal",
        "access_declaration_is_missing_required_source_slot_field",
        "access_declaration_must_not_use_id_alias",
        "access_declaration_provider_predicate_must_be_local_label",
        "access_declaration_provider_predicate_must_select_compatible_boolean_result",
        "access_declaration_provider_node_ref_must_be_local_label",
        "access_declaration_provider_node_ref_must_select_direct_consumer",
        "access_declaration_must_match_source_wire_template",
    }
)
# Output selection is provider-owned and can be corrected only by choosing a
# different already-authored, function-visible value.  Bound that preparation
# loop exactly like graph/access authoring: a bad mapping never becomes a
# candidate or repair base, and repeated writes must not consume the global
# owner step limit.
_MAX_OUTPUT_DECLARATION_FAILURES = 4
_PROVIDER_CORRECTABLE_OUTPUT_DECLARATION_DIAGNOSTICS = frozenset(
    {
        "output_mappings_must_be_json_array",
        "output_declaration_must_cover_each_source_output",
        "output_declaration_rows_must_be_json_objects",
        "output_declaration_is_missing_required_source_slot_field",
        "output_declaration_must_contain_only_value",
        "output_declaration_requires_declared_operation_graph",
        "output_declaration_value_must_reference_source_input_or_local_result",
        "output_declaration_value_must_be_function_visible",
        "output_declaration_value_type_must_match_source_output",
    }
)
# Boundary collections are the last semantic write before finalization.  A
# missing graph-required broadcast used to consume the one CandidateCheck
# attempt, while an unchanged empty object could still consume all remaining
# turns.  Keep this pre-candidate correction finite and receipt-only.
_MAX_BOUNDARY_FIELDS_DECLARATION_FAILURES = 4
# A dynamic MCP phase exposes only one legal lifecycle edge, but an
# application-level error still returns a syntactically valid tool result. It
# is therefore not a ``FormatError``, and changing irrelevant request fields
# also keeps it from tripping exact action/output stagnation. Bound that
# pre-candidate loop by the phase edge and a digest of the safe error rule.
# Two occurrences of the same rule prove that another request at the same edge
# cannot reveal new construction facts; four distinct rules are the absolute
# per-edge preparation ceiling. Neither boundary creates or accepts a draft.
_MAX_REPEATED_MCP_PHASE_RULE_FAILURES = 2
_MAX_MCP_PHASE_APPLICATION_FAILURES = 4
_PROVIDER_CORRECTABLE_OPERATION_GRAPH_DIAGNOSTICS = frozenset(
    {
        "operation_graph_must_be_valid_json",
        "operation_graph_must_not_contain_duplicate_json_keys",
        "operation_graph_must_use_finite_json_numbers",
        "operation_graph_must_be_nonempty_json_array",
        "operation_graph_rows_must_be_json_objects",
        "operation_graph_lexical_wire_is_malformed",
        "operation_row_must_include_canonical_op_field",
        "operation_must_use_only_canonical_leaf_fields",
        "operation_must_include_required_leaf_fields",
        "operation_op_must_be_canonical_j_name",
        "operation_id_must_be_provider_local_label",
        "operation_result_must_be_provider_local_label",
        "operation_operands_must_be_json_array",
        "operation_operands_must_match_canonical_leaf_arity",
        "operation_operand_must_reference_source_tensor_or_prior_local_result",
        "operation_result_must_be_new_provider_local_label",
        "operation_attrs_must_be_json_object",
        "operation_attrs_must_use_canonical_leaf_fields",
        "operation_attrs_must_include_required_leaf_fields",
        "operation_attr_value_must_match_canonical_leaf_contract",
        "operation_result_must_have_complete_value_type",
        "operation_type_shape_must_use_valid_provider_extent",
        "operation_type_shape_must_use_provider_dimension_alias",
        "operation_result_type_shape_must_match_operands",
        "operation_iota_result_type_must_match_attrs",
        "operation_nontrailing_broadcast_requires_local_id",
        "operation_graph_must_expose_each_source_output_type",
        "operation_graph_must_expose_compatible_access_predicate_type",
        "operation_graph_must_expose_access_consumer_id",
        "operation_graph_must_cover_source_operation_families",
        "operation_graph_must_cover_source_operation_attributes",
    }
)


def _operation_attr_value_matches_schema(value: Any, schema: Any) -> bool:
    """Check the bounded JSON-Schema leaf forms used by operation attrs.

    This is intentionally narrower than full artifact-schema validation.  It
    rejects only a value which directly contradicts the one canonical leaf
    contract already used by schema lookup and TypeCheck; cross-operation
    semantics remain with canonical construction and CandidateCheck.
    """

    if not isinstance(schema, Mapping):
        return False
    if "const" in schema:
        return value == schema["const"]
    enum = schema.get("enum")
    if isinstance(enum, list):
        return value in enum
    alternatives = schema.get("anyOf")
    if isinstance(alternatives, list):
        return any(_operation_attr_value_matches_schema(value, item) for item in alternatives)
    expected_type = schema.get("type")
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        return (
            not isinstance(item_schema, Mapping)
            or all(_operation_attr_value_matches_schema(item, item_schema) for item in value)
        )
    if expected_type == "object":
        return isinstance(value, Mapping)
    # A schema leaf without an immediately checkable container restriction is
    # left to the canonical validator instead of guessed at ingress.
    return True


def _reject_non_json_constant(value: str) -> None:
    """Reject JavaScript-only numeric constants at the artifact boundary."""

    raise ValueError(f"invalid JSON constant: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Keep candidate wire parsing injective instead of folding duplicate keys."""

    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _parse_candidate_object(text: str) -> dict[str, Any]:
    """Validate one semantic candidate document without inspecting its fields.

    This is deliberately only a wire-level check. CandidateCheck remains the
    authority for schema and semantic validity. Keeping this check before an
    atomic replace prevents a malformed or concatenated model response from
    replacing the current attempt's last complete candidate bytes.
    """

    value = json.loads(
        text,
        parse_constant=_reject_non_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(value, dict):
        raise TypeError("candidate.json must contain one JSON object")
    return value


def _json_container_kind(value: Any) -> str:
    """Return the JSON primitive/container kind used by the mounted schema."""

    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _exception_candidate_check_diagnostics(
    exc: BaseException,
) -> list[dict[str, Any]] | None:
    """Project only a checked, bounded diagnostic receipt from a callback.

    Constructor errors may know an exact rule/path, but their message can also
    contain provider or Source-derived text.  The environment must never turn
    that message into relift context.  It accepts only the closed diagnostic
    tuple exported by the canonical ingress boundary.
    """

    raw = getattr(exc, "candidate_check_diagnostics", None)
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 20:
        return None
    projected: list[dict[str, Any]] = []
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"rule_code", "path", "observed_refs_or_types"}
            or not isinstance(item.get("rule_code"), str)
            or not item["rule_code"]
            or len(item["rule_code"]) > 128
            or not isinstance(item.get("path"), str)
            or not item["path"]
            or len(item["path"]) > MAX_CANDIDATE_CHECK_FIELD_CHARS
            or not isinstance(item.get("observed_refs_or_types"), (list, tuple))
            or len(item["observed_refs_or_types"]) > MAX_CANDIDATE_CHECK_OBSERVED_ITEMS
            or any(
                not isinstance(value, str)
                or not value
                or len(value) > 128
                for value in item["observed_refs_or_types"]
            )
        ):
            return None
        projected.append(
            {
                "rule_code": item["rule_code"],
                "path": item["path"],
                "observed_refs_or_types": list(item["observed_refs_or_types"]),
            }
        )
    return projected


def _remove_path_iterative(path: Path) -> None:
    """Remove one exact path tree without Python recursion."""

    try:
        root_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(root_stat.st_mode) or path.is_symlink():
        path.unlink(missing_ok=True)
        return
    stack: list[tuple[Path, bool]] = [(path, False)]
    while stack:
        current, visited = stack.pop()
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(current_stat.st_mode) or current.is_symlink():
            current.unlink(missing_ok=True)
            continue
        os.chmod(current, stat.S_IRWXU)
        if visited:
            current.rmdir()
            continue
        stack.append((current, True))
        with os.scandir(current) as iterator:
            for entry in iterator:
                child = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append((child, False))
                else:
                    child.unlink(missing_ok=True)


@dataclass
class IntentIRLiftWorkspace:
    """Private host directories projected into exactly one agent sandbox."""

    root: Path
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, files: Mapping[str, str | bytes]) -> "IntentIRLiftWorkspace":
        root = Path(tempfile.mkdtemp(prefix="intentir-lift-"))
        workspace = cls(
            root=root,
            input_dir=root / "input",
            output_dir=root / "output",
            scratch_dir=root / "work",
            tmp_dir=root / "tmp",
        )
        workspace.input_dir.mkdir()
        workspace.output_dir.mkdir()
        workspace.scratch_dir.mkdir()
        workspace.tmp_dir.mkdir()
        for relative, content in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                workspace.cleanup()
                raise ValueError(f"unsafe Lift input path: {relative!r}")
            destination = workspace.input_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                destination.write_bytes(content)
            else:
                destination.write_text(content, encoding="utf-8")
        return workspace

    @property
    def candidate_path(self) -> Path:
        return self.output_dir / "candidate.json"

    def read_candidate(self) -> str:
        entries = tuple(self.output_dir.iterdir())
        if entries != (self.candidate_path,):
            names = sorted(path.name for path in entries)
            raise ValueError(
                "Lift output must contain exactly candidate.json; "
                f"found {names}"
            )
        path = self.candidate_path
        if path.is_symlink() or not path.is_file():
            raise ValueError("candidate.json must be a regular file")
        size = path.stat().st_size
        if not 1 <= size <= MAX_CANDIDATE_BYTES:
            raise ValueError(
                f"candidate.json size must be in [1, {MAX_CANDIDATE_BYTES}]"
            )
        raw = path.read_bytes()
        if len(raw) != size:
            raise ValueError("candidate.json changed while being read")
        text = raw.decode("utf-8")
        _parse_candidate_object(text)
        return text

    def cleanup(self) -> None:
        _remove_path_iterative(self.root)


class IntentIRLiftEnvironmentConfig(BaseModel):
    input_dir: Path
    output_dir: Path
    scratch_dir: Path
    tmp_dir: Path | None = None
    timeout: float | None = None
    candidate_check_limit: int = 0
    candidate_submission_check_limit: int = 0
    executable: str = os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
    env: dict[str, str] = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
    }


class IntentIRLiftEnvironment:
    """Execute agent shell actions with only /input, /output, and /work visible."""

    def __init__(
        self,
        *,
        config_class: type[IntentIRLiftEnvironmentConfig] = IntentIRLiftEnvironmentConfig,
        candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
        semantic_candidate_checker: Callable[[str], Mapping[str, Any]] | None = None,
        candidate_constructor: Callable[
            [str], tuple[str, tuple[Mapping[str, Any], ...]]
        ] | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config_class(**kwargs)
        self._semantic_candidate_skeleton, self._semantic_operation_namespace = (
            self._mounted_semantic_authoring_contract()
        )
        self._semantic_write_contract = self._mounted_semantic_write_contract()
        # The generic schema seed deliberately has no Source-specific rows.
        # Read the already-mounted construction requirements once so the MCP
        # lifecycle can expose the exact *slots* it must complete without
        # publishing a second Source/ABI/C projection or inventing an
        # environment-local schema.  The full semantic content of a slot
        # remains available through the bounded transfer-ledger resource.
        self._semantic_construction_identity = (
            self._mounted_semantic_construction_identity()
        )
        self._semantic_construction_slots = (
            self._mounted_semantic_construction_slots()
        )
        self._semantic_source_tensor_handles = (
            self._mounted_semantic_source_tensor_handles()
        )
        self._semantic_source_tensor_value_types = (
            self._mounted_semantic_source_tensor_value_types()
        )
        self._semantic_source_input_tensor_handles = (
            self._mounted_semantic_input_tensor_handles()
        )
        self._semantic_source_output_tensor_value_types = (
            self._mounted_semantic_output_tensor_value_types()
        )
        if tuple(self._semantic_write_contract["steps"]) != MCP_SEMANTIC_WRITE_TOOL_NAMES:
            raise ValueError(
                "mounted MCP semantic write steps drifted from server tool surface"
            )
        if self._semantic_write_contract["protocol"] != canonical_semantic_mcp_write_protocol():
            raise ValueError(
                "mounted MCP semantic write authority drifted from canonical schema"
            )
        if self.config.candidate_check_limit < 0:
            raise ValueError("candidate_check_limit must be non-negative")
        if self.config.candidate_submission_check_limit < 0:
            raise ValueError(
                "candidate_submission_check_limit must be non-negative"
            )
        self._candidate_checker = candidate_checker
        self._semantic_candidate_checker = semantic_candidate_checker
        # This callback is the one host-owned Source/Harness construction
        # transaction reached by MCP finalize.  It is deliberately not a
        # provider patch hook: it receives one complete semantic surface and
        # either returns the complete canonical artifact or rejects it.
        self._candidate_constructor = candidate_constructor
        self._candidate_check_calls = 0
        self._candidate_check_evaluations = 0
        self._candidate_check_calls_by_trigger = {
            "explicit": 0,
            "submission": 0,
        }
        self._candidate_check_evaluations_by_trigger = {
            "explicit": 0,
            "submission": 0,
        }
        self._candidate_check_shared_evaluations = 0
        self._candidate_check_submission_reserved_evaluations = 0
        self._candidate_check_audit: list[dict[str, Any]] = []
        self._candidate_constructor_audit: list[dict[str, Any]] = []
        self._candidate_check_passed_bytes: bytes | None = None
        self._candidate_finalize_audit: list[dict[str, Any]] = []
        self._candidate_submission_attempted = False
        # The MCP writer owns only this semantic J/pi buffer.  It is initialized
        # from the schema-derived empty semantic template. C, canonical handles,
        # effects and the remaining pi proof are never mutable MCP state.
        self._semantic_construction_buffer: dict[str, Any] | None = None
        self._semantic_construction_audit: list[dict[str, Any]] = []
        self._semantic_construction_completed_steps: set[str] = set()
        # This is intentionally receipt-only telemetry.  It contains a
        # normalized declaration digest and small structural facts, never the
        # provider declaration or a Source/candidate fragment.
        self._semantic_operation_graph_failures: list[dict[str, Any]] = []
        self._semantic_access_declaration_failures: list[dict[str, Any]] = []
        self._semantic_output_declaration_failures: list[dict[str, Any]] = []
        self._semantic_boundary_field_failures: list[dict[str, Any]] = []
        self._multi_action_rejections = 0
        self._structured_context_queries: list[str] = []
        self._structured_schema_queries: list[list[str]] = []
        self._mcp_requests: list[dict[str, Any]] = []
        self._mcp_phase_application_failures: list[dict[str, Any]] = []
        self._preparation_failure_receipts: list[dict[str, Any]] = []
        self._semantic_entry_read = False
        self._semantic_entry_source_read = False
        self._semantic_entry_transfer_ledger_read = False
        # A pre-construction read establishes the initial Source authority.
        # Access authoring additionally requires one refresh after the graph
        # has been accepted, bound to this same in-memory transaction.
        self._semantic_transfer_read_generation = 0
        self._semantic_operation_graph_transfer_generation: int | None = None
        self._semantic_access_transfer_generation: int | None = None
        self._semantic_transaction_identity: str | None = None
        self._semantic_entry_supplemental_read: str | None = None
        self._semantic_entry_schema_lookup_calls = 0
        # Filled only from the mounted source/brief semantic spine.  A fresh
        # owner may request an exact operation leaf only when that compact
        # Source fact explicitly says the brief lacks it.
        self._semantic_entry_missing_operation_leaves: frozenset[str] = frozenset()
        # Counts only summarize the already-mounted Source semantic spine.
        # They let the begin receipt distinguish a complete calculation from
        # an index/mask-only setup graph without exposing Source text or
        # creating a second provider-authored operation template.
        self._semantic_entry_operation_fact_count = 0
        self._semantic_entry_non_boundary_operation_fact_count = 0
        self._semantic_entry_required_operation_families: dict[str, int] = {}
        self._semantic_entry_required_operation_attributes: tuple[
            tuple[str, str, str, int], ...
        ] = ()
        # Once ingress has exposed begin_candidate, stale/static clients must
        # not spend the remaining Lift budget attempting reads or obsolete
        # tools. Two rejected attempts are enough to establish that the
        # construction phase was not entered; no partial buffer is retained.
        self._semantic_preconstruction_phase_failures = 0
        self._stage_action_calls = 0
        self._stage_action_failures = 0
        self._stage_action_elapsed_ms = 0.0
        # ``/input`` contains the complete host construction seed solely for
        # the bounded MCP resource implementation.  An owner shell must not
        # bypass source-brief/read budgets by opening those files directly.
        # Host-issued MCP readers enter through the small scoped helper below.
        self._host_mcp_read_depth = 0
        self._mcp_server = IntentIRLiftMCPServer(
            self._mcp_dispatch,
            active_resource_uris=self._active_mcp_resource_uris,
        )
        self._validate_paths()
        self._startup_context, self._startup_context_receipt = (
            self._build_startup_context()
        )
        self._probe()

    def _mounted_semantic_authoring_contract(self) -> tuple[str, str]:
        """Read the host-published authoring contract; never recreate it here."""

        path = self.config.input_dir / "schema" / "canonical_wire_bundle.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical semantic authoring bundle is unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        profile = bundle.get("profile") if isinstance(bundle, Mapping) else None
        semantic = (
            profile.get("semantic_candidate")
            if isinstance(profile, Mapping)
            else None
        )
        template = semantic.get("template") if isinstance(semantic, Mapping) else None
        namespace = (
            semantic.get("operation_namespace")
            if isinstance(semantic, Mapping)
            else None
        )
        if not isinstance(template, Mapping) or not isinstance(namespace, Mapping):
            raise ValueError("canonical semantic authoring contract is malformed")
        root_fields = semantic.get("root_fields") if isinstance(semantic, Mapping) else None
        j_fields = semantic.get("j_fields") if isinstance(semantic, Mapping) else None
        j_field_types = semantic.get("j_field_types") if isinstance(semantic, Mapping) else None
        template_j = template.get("j")
        template_pi = template.get("pi")
        if (
            not isinstance(root_fields, list)
            or not isinstance(j_fields, list)
            or not isinstance(j_field_types, Mapping)
            or set(template) != set(root_fields)
            or not isinstance(template_j, Mapping)
            or not isinstance(template_pi, Mapping)
            or set(template_j) != set(j_fields)
            or template_pi
            or not isinstance(template_j.get("intent"), Mapping)
            or not isinstance(template_j["intent"].get("intent_function"), Mapping)
            or not isinstance(template_j["intent"]["intent_function"].get("body"), list)
            or any(
                j_field_types.get(field) == "object"
                and not isinstance(template_j.get(field), Mapping)
                for field in j_fields
            )
            or any(
                j_field_types.get(field) == "array"
                and not isinstance(template_j.get(field), list)
                for field in j_fields
            )
        ):
            raise ValueError("mounted semantic candidate skeleton drifted")
        return (
            json.dumps(template, ensure_ascii=False, separators=(",", ":")),
            json.dumps(namespace, ensure_ascii=False, separators=(",", ":")),
        )

    def _mounted_semantic_write_contract(self) -> dict[str, Any]:
        """Read the MCP write projection from the same mounted wire bundle.

        This is intentionally a reader, not another declaration of J fields.
        The schema module emits the profile; the environment only verifies the
        bounded destinations before accepting a provider declaration.
        """

        path = self.config.input_dir / "schema" / "canonical_wire_bundle.json"
        bundle = json.loads(path.read_text(encoding="utf-8"))
        profile = bundle.get("profile") if isinstance(bundle, Mapping) else None
        semantic = (
            profile.get("semantic_candidate")
            if isinstance(profile, Mapping)
            else None
        )
        write = (
            profile.get("semantic_mcp_write")
            if isinstance(profile, Mapping)
            else None
        )
        if not isinstance(semantic, Mapping) or not isinstance(write, Mapping):
            raise ValueError("canonical MCP semantic write contract is unavailable")
        writes = write.get("writes")
        steps = write.get("steps")
        required_steps = write.get("required_steps")
        initial_template = write.get("initial_template")
        tool_arguments = write.get("tool_arguments")
        boundary_types = semantic.get("boundary_field_types")
        j_types = semantic.get("j_field_types")
        if (
            not isinstance(writes, Mapping)
            or not isinstance(steps, list)
            or not steps
            or any(not isinstance(step, str) or not step for step in steps)
            or len(steps) != len(set(steps))
            or not isinstance(required_steps, list)
            or not required_steps
            or any(
                not isinstance(step, str) or step not in steps
                for step in required_steps
            )
            or len(required_steps) != len(set(required_steps))
            or initial_template != semantic.get("template")
            or not isinstance(tool_arguments, Mapping)
            or not isinstance(boundary_types, Mapping)
            or not isinstance(j_types, Mapping)
            or not isinstance(writes.get("boundary_field"), list)
            or not isinstance(writes.get("j_field"), list)
            or set(tool_arguments) != set(steps)
            or any(
                not isinstance(item, str)
                for item in (
                    *writes["boundary_field"],
                    *writes["j_field"],
                )
            )
            or any(
                row is not None
                and (
                    not isinstance(row, Mapping)
                    or set(row) != {"field", "json_container"}
                    or not isinstance(row.get("field"), str)
                    or row.get("json_container") not in {"array", "object"}
                )
                for row in tool_arguments.values()
            )
        ):
            raise ValueError("canonical MCP semantic write fields are malformed")
        return {
            "boundary_fields": tuple(writes["boundary_field"]),
            "j_fields": tuple(writes["j_field"]),
            "boundary_field_types": dict(boundary_types),
            "j_field_types": dict(j_types),
            "tool_arguments": copy.deepcopy(dict(tool_arguments)),
            "steps": tuple(steps),
            "required_steps": tuple(required_steps),
            "protocol": copy.deepcopy(dict(write)),
        }

    @staticmethod
    def _stable_host_digest(value: Mapping[str, Any]) -> str:
        """Return an opaque digest for a mounted host-owned record."""

        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def _mounted_semantic_construction_identity(self) -> dict[str, str]:
        """Load the identity shared by source/brief and transfer-ledger.

        The provider does not author any of these values.  The lifecycle uses
        them solely to reject a stale mounted brief or ledger before it can be
        associated with a fresh semantic transaction.
        """

        construction_path = self.config.input_dir / "canonical_construction.json"
        source_wire_path = (
            self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        )
        if (
            not construction_path.is_file()
            or construction_path.is_symlink()
            or not source_wire_path.is_file()
            or source_wire_path.is_symlink()
        ):
            raise ValueError("canonical construction identity is unavailable")
        construction = json.loads(construction_path.read_text(encoding="utf-8"))
        source_wire = json.loads(source_wire_path.read_text(encoding="utf-8"))
        source_facts = (
            construction.get("source_facts")
            if isinstance(construction, Mapping)
            else None
        )
        transfer_ledger = (
            construction.get("transfer_ledger")
            if isinstance(construction, Mapping)
            else None
        )
        allocator = (
            construction.get("allocator")
            if isinstance(construction, Mapping)
            else None
        )
        construction_digest = (
            construction.get("authority_digest")
            if isinstance(construction, Mapping)
            else None
        )
        source_facts_digest = (
            self._stable_host_digest(source_facts)
            if isinstance(source_facts, Mapping)
            else None
        )
        if (
            not isinstance(construction_digest, str)
            or not construction_digest
            or not isinstance(source_facts_digest, str)
            or not source_facts_digest
            or not isinstance(transfer_ledger, Mapping)
            or transfer_ledger.get("source_facts_digest") != source_facts_digest
            or not isinstance(allocator, Mapping)
            or not isinstance(source_wire, Mapping)
            or source_wire.get("construction_digest") != construction_digest
            or source_wire.get("source_transfer_ledger") != transfer_ledger
        ):
            raise ValueError("canonical construction identity is malformed")
        payload = {
            key: value
            for key, value in construction.items()
            if key != "authority_digest"
        }
        if self._stable_host_digest(payload) != construction_digest:
            raise ValueError("canonical construction identity digest is stale")
        return {
            "construction_digest": construction_digest,
            "source_facts_digest": source_facts_digest,
            "transfer_ledger_digest": self._stable_host_digest(transfer_ledger),
            "allocator_snapshot_digest": self._stable_host_digest(allocator),
        }

    def _mounted_semantic_construction_slots(self) -> dict[str, tuple[dict[str, Any], ...]]:
        """Read Source-sized semantic completion slots from host authority.

        This is intentionally a narrow projection of
        ``semantic_construction_requirements`` rather than a second candidate
        grammar.  The schema/constructor remains responsible for the actual
        access and output row shapes.  The environment needs only enough
        immutable information to tell a fresh MCP owner which rows cannot be
        omitted before it spends a finalization attempt.
        """

        path = (
            self.config.input_dir
            / "schema"
            / "canonical_source_wire_bundle.json"
        )
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source construction requirements are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        requirements = (
            bundle.get("semantic_construction_requirements")
            if isinstance(bundle, Mapping)
            else None
        )
        source_transfer = (
            bundle.get("source_transfer_ledger")
            if isinstance(bundle, Mapping)
            else None
        )
        transfer_projection = (
            bundle.get("semantic_transfer_projection")
            if isinstance(bundle, Mapping)
            else None
        )
        if not isinstance(source_transfer, Mapping):
            raise ValueError("canonical Source transfer ledger is unavailable")
        if source_transfer.get("status") != "pass":
            raise ValueError(
                "canonical Source transfer ledger is not closed: "
                f"{source_transfer.get('reason')}"
            )
        semantic_required = (
            requirements.get("semantic_required")
            if isinstance(requirements, Mapping)
            else None
        )
        accesses = (
            semantic_required.get("accesses")
            if isinstance(semantic_required, Mapping)
            else None
        )
        outputs = (
            semantic_required.get("outputs")
            if isinstance(semantic_required, Mapping)
            else None
        )
        access_rows = accesses.get("rows") if isinstance(accesses, Mapping) else None
        output_rows = outputs.get("rows") if isinstance(outputs, Mapping) else None
        projected_access_rows = (
            transfer_projection.get("occurrences")
            if isinstance(transfer_projection, Mapping)
            else None
        )
        if (
            not isinstance(access_rows, list)
            or not isinstance(output_rows, list)
            or not isinstance(projected_access_rows, list)
            or len(access_rows) != len(projected_access_rows)
            or any(not isinstance(row, Mapping) for row in access_rows)
            or any(not isinstance(row, Mapping) for row in projected_access_rows)
            or any(not isinstance(row, Mapping) for row in output_rows)
        ):
            raise ValueError("canonical Source semantic completion slots are malformed")

        access_slots: list[dict[str, Any]] = []
        for ordinal, (requirement_row, wire_row) in enumerate(
            zip(access_rows, projected_access_rows, strict=True)
        ):
            label = wire_row.get("access_label")
            required = wire_row.get("required_fields")
            template = wire_row.get("wire_template")
            provider_owned_fields = wire_row.get("provider_owned_fields", [])
            if (
                requirement_row.get("ordinal") != ordinal
                or requirement_row.get("access_label") != label
                or wire_row.get("ordinal") != ordinal
                or label != f"access{ordinal}"
                or not isinstance(required, list)
                or any(not isinstance(field, str) or not field for field in required)
                or not isinstance(template, Mapping)
                or template.get("ordinal") != ordinal
                or template.get("access_label") != label
                or "id" in template
                or not isinstance(provider_owned_fields, list)
                or any(
                    not isinstance(field, str)
                    or field not in required
                    or field not in template
                    for field in provider_owned_fields
                )
            ):
                raise ValueError("canonical Source access completion slots are malformed")
            access_slots.append(
                {
                    "ordinal": ordinal,
                    "access_label": label,
                    "required_fields": list(required),
                    "wire_template": copy.deepcopy(dict(template)),
                    "provider_owned_fields": list(provider_owned_fields),
                }
            )

        output_slots: list[dict[str, Any]] = []
        for ordinal, row in enumerate(output_rows):
            required = row.get("required_fields")
            template = row.get("wire_template")
            if (
                not isinstance(row.get("output"), str)
                or not isinstance(required, list)
                or required != ["value"]
                or not isinstance(template, Mapping)
                or set(template) != {"value"}
            ):
                raise ValueError("canonical Source output completion slots are malformed")
            # Output names/types remain in the transfer-ledger view.  An ABI
            # ordinal plus the schema-owned required selection is all that is
            # needed to prevent an omitted mapping while preserving a compact
            # first construction response.
            output_slots.append(
                {
                    "ordinal": ordinal,
                    "required_fields": ["value"],
                    "wire_template": copy.deepcopy(dict(template)),
                }
            )
        return {
            "accesses": tuple(access_slots),
            "outputs": tuple(output_slots),
        }

    def _mounted_semantic_source_tensor_declarations(
        self,
    ) -> dict[str, tuple[str, tuple[str | int, ...]]]:
        """Read the canonical Source tensor declarations once, consistently.

        ``intent_function.inputs`` and ``intent_function.outputs`` are two ABI
        collections, not two namespaces.  A legal inout therefore repeats its
        input ``t*`` in outputs.  Repetition within either collection is still
        ambiguous, and a cross-collection repetition is legal only when its
        complete declared type is identical.  Keeping this projection shared
        prevents graph-handle and graph-type ingress from disagreeing about a
        Source inout.
        """

        path = (
            self.config.input_dir
            / "schema"
            / "canonical_source_wire_bundle.json"
        )
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source tensor declarations are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        seed = bundle.get("artifact_seed") if isinstance(bundle, Mapping) else None
        j = seed.get("j") if isinstance(seed, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        if not isinstance(function, Mapping):
            raise ValueError("canonical Source tensor declarations are malformed")
        result: dict[str, tuple[str, tuple[str | int, ...]]] = {}
        for field in ("inputs", "outputs"):
            rows = function.get(field)
            if not isinstance(rows, list):
                raise ValueError("canonical Source tensor declarations are malformed")
            collection_names: set[str] = set()
            for row in rows:
                name = row.get("name") if isinstance(row, Mapping) else None
                dtype = row.get("dtype") if isinstance(row, Mapping) else None
                shape = row.get("shape") if isinstance(row, Mapping) else None
                if (
                    not isinstance(name, str)
                    or re.fullmatch(r"t[0-9]+", name) is None
                    or not isinstance(dtype, str)
                    or not dtype
                    or not isinstance(shape, list)
                ):
                    raise ValueError("canonical Source tensor declarations are malformed")
                if name in collection_names:
                    raise ValueError("canonical Source tensor declarations are ambiguous")
                collection_names.add(name)
                provider_shape: list[str | int] = []
                for extent in shape:
                    if isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0:
                        provider_shape.append(extent)
                        continue
                    if isinstance(extent, str):
                        match = re.fullmatch(r"x([0-9]+)", extent)
                        if match is not None:
                            provider_shape.append(f"dim{match.group(1)}")
                            continue
                    raise ValueError("canonical Source tensor declarations are malformed")
                declaration = (dtype, tuple(provider_shape))
                previous = result.get(name)
                if previous is not None and previous != declaration:
                    raise ValueError("canonical Source tensor declarations are ambiguous")
                result[name] = declaration
        return result

    def _mounted_semantic_source_tensor_handles(self) -> frozenset[str]:
        """Read the closed Source ``t*`` set used by graph ingress checks."""

        return frozenset(self._mounted_semantic_source_tensor_declarations())

    def _mounted_semantic_source_tensor_value_types(
        self,
    ) -> dict[str, tuple[str, tuple[str | int, ...]]]:
        """Read Source tensor types for bounded graph shape consistency.

        This is the same mounted seed used to validate source ``t*``
        references.  Its canonical ``x*`` dimensions are projected to the
        provider's pre-allocation ``dim*`` names solely for comparison with a
        provider-owned operation result type; no type is authored, repaired,
        or exposed through a second MCP resource.
        """

        try:
            result = self._mounted_semantic_source_tensor_declarations()
        except ValueError as exc:
            detail = str(exc)
            if detail == "canonical Source tensor declarations are unavailable":
                raise ValueError("canonical Source tensor types are unavailable") from exc
            if detail == "canonical Source tensor declarations are ambiguous":
                raise ValueError("canonical Source tensor types are ambiguous") from exc
            raise ValueError("canonical Source tensor types are malformed") from exc
        if frozenset(result) != self._semantic_source_tensor_handles:
            raise ValueError("canonical Source tensor type/handle projections disagree")
        return result

    def _mounted_semantic_input_tensor_handles(self) -> frozenset[str]:
        """Return only Source values legal as direct semantic output choices.

        A provider can return an existing logical input for an alias/inout
        output, but it cannot select an output declaration itself as though it
        were an SSA producer.  This is a narrow Source-seed projection, not a
        second output-record constructor.
        """

        path = self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source input declarations are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        seed = bundle.get("artifact_seed") if isinstance(bundle, Mapping) else None
        j = seed.get("j") if isinstance(seed, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        rows = function.get("inputs") if isinstance(function, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("canonical Source input declarations are malformed")
        names = {
            row.get("name")
            for row in rows
            if isinstance(row, Mapping)
            and isinstance(row.get("name"), str)
            and re.fullmatch(r"t[0-9]+", row["name"]) is not None
        }
        if len(names) != len(rows):
            raise ValueError("canonical Source input declarations are malformed")
        return frozenset(names)

    def _mounted_semantic_output_tensor_value_types(
        self,
    ) -> tuple[tuple[str, tuple[str | int, ...]], ...]:
        """Return Source output types in the sole ABI output order.

        The provider still chooses one existing semantic value for each output.
        This ordered type projection only lets the MCP write boundary reject a
        choice which the canonical output constructor must reject later; it
        does not choose, rename, or materialize an output value.
        """

        path = self.config.input_dir / "schema" / "canonical_source_wire_bundle.json"
        if not path.is_file() or path.is_symlink():
            raise ValueError("canonical Source output declarations are unavailable")
        bundle = json.loads(path.read_text(encoding="utf-8"))
        seed = bundle.get("artifact_seed") if isinstance(bundle, Mapping) else None
        j = seed.get("j") if isinstance(seed, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        rows = function.get("outputs") if isinstance(function, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError("canonical Source output declarations are malformed")
        result: list[tuple[str, tuple[str | int, ...]]] = []
        for row in rows:
            name = row.get("name") if isinstance(row, Mapping) else None
            value_type = (
                self._semantic_source_tensor_value_types.get(name)
                if isinstance(name, str)
                else None
            )
            if value_type is None:
                raise ValueError("canonical Source output declarations are malformed")
            result.append(value_type)
        if len(result) != len(self._semantic_construction_slots["outputs"]):
            raise ValueError("canonical Source output declarations are malformed")
        return tuple(result)

    def _semantic_slot_receipt(self) -> dict[str, Any]:
        """Return the bounded required-slot view for ``begin_candidate``."""

        return {
            "accesses": [dict(row) for row in self._semantic_construction_slots["accesses"]],
            "outputs": [dict(row) for row in self._semantic_construction_slots["outputs"]],
            "operation_graph": {
                "source_operation_facts": self._semantic_entry_operation_fact_count,
                "non_boundary_source_operation_facts": (
                    self._semantic_entry_non_boundary_operation_fact_count
                ),
                "required_operation_families": dict(
                    self._semantic_entry_required_operation_families
                ),
                "required_operation_attributes": [
                    {
                        "op": operation,
                        "attribute": attribute,
                        "value": value,
                        "count": count,
                    }
                    for operation, attribute, value, count in (
                        self._semantic_entry_required_operation_attributes
                    )
                ],
                "requirement": (
                    "represent the complete Source calculation spine; index and "
                    "mask setup alone are not a complete operation graph"
                ),
            },
        }

    def _semantic_access_slot_diagnostic(
        self, accesses: Sequence[Any]
    ) -> tuple[str, dict[str, Any]] | None:
        """Reject an incomplete/wrong-order access declaration immediately.

        This is a slot-completion check only.  The canonical semantic
        CandidateCheck still owns row grammar, source fact compatibility, and
        all cross-field acceptance at ``finalize_candidate``.
        """

        expected = self._semantic_construction_slots["accesses"]
        if len(accesses) != len(expected):
            return (
                "access_declaration_must_cover_each_source_occurrence",
                {
                    "expected_row_count": len(expected),
                    "actual_row_count": len(accesses),
                },
            )
        predicate_choices: tuple[frozenset[str], ...] | None = None
        consumer_choices: tuple[frozenset[str], ...] | None = None
        for index, (slot, row) in enumerate(zip(expected, accesses, strict=True)):
            if not isinstance(row, Mapping):
                return (
                    "access_declaration_rows_must_be_json_objects",
                    {"row_index": index},
                )
            if row.get("access_label") != slot["access_label"]:
                return (
                    "access_declaration_must_follow_source_ordinal",
                    {
                        "row_index": index,
                        "expected_access_label": slot["access_label"],
                        "actual_access_label": row.get("access_label"),
                    },
                )
            if row.get("ordinal") != slot["ordinal"]:
                return (
                    "access_declaration_must_follow_source_ordinal",
                    {
                        "row_index": index,
                        "expected_ordinal": slot["ordinal"],
                        "actual_ordinal": row.get("ordinal"),
                    },
                )
            if "id" in row:
                return (
                    "access_declaration_must_not_use_id_alias",
                    {"row_index": index, "field": "id"},
                )
            missing_fields = [
                field for field in slot["required_fields"] if field not in row
            ]
            if missing_fields:
                return (
                    "access_declaration_is_missing_required_source_slot_field",
                    {
                        "row_index": index,
                        "required_fields": list(slot["required_fields"]),
                        # Field names and the row ordinal are structural
                        # receipts, not a retained rejected declaration.  The
                        # provider already owns the declaration and receives
                        # the canonical Source wire template separately.
                        "missing_fields": missing_fields,
                    },
                )
            template = slot.get("wire_template")
            if not isinstance(template, Mapping):
                return (
                    "canonical_source_access_wire_template_is_malformed",
                    {"row_index": index},
                )
            provider_owned = slot.get("provider_owned_fields", ())
            if not isinstance(provider_owned, (list, tuple)) or any(
                not isinstance(field, str) for field in provider_owned
            ):
                return (
                    "canonical_source_access_wire_template_is_malformed",
                    {"row_index": index},
                )
            # A transfer row's kind/tensor/index/predicate/fill is Source
            # authority, except an explicitly unresolved masked predicate:
            # Source owns its presence but the provider owns the local J label
            # selected for the boolean result.  Reject all other drift at the
            # MCP declaration that introduced it instead of waiting for the
            # ledger join in CandidateCheck.  Host-only absent fields (for
            # example a store's null fill) remain absent from the narrow
            # provider wire, hence the required-fields intersection.
            for field in (
                "kind",
                "tensor",
                "index_expression",
                "predicate",
                "fill_value",
                "node_ref",
                "effect_kind",
                "effect_target",
                "effect_node_ref",
            ):
                if field in provider_owned:
                    value = row.get(field)
                    if (
                        not isinstance(value, str)
                        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None
                        or value.startswith("<")
                    ):
                        return (
                            (
                                "access_declaration_provider_node_ref_must_be_local_label"
                                if field in {"node_ref", "effect_node_ref"}
                                else "access_declaration_provider_predicate_must_be_local_label"
                            ),
                            {
                                "row_index": index,
                                "field": field,
                                "expected": (
                                    "a provider-local operation id"
                                    if field in {"node_ref", "effect_node_ref"}
                                    else "a provider-local boolean result label"
                                ),
                            },
                        )
                    if field == "predicate":
                        if predicate_choices is None:
                            buffer = self._semantic_construction_buffer
                            j = buffer.get("j") if isinstance(buffer, Mapping) else None
                            intent = j.get("intent") if isinstance(j, Mapping) else None
                            function = (
                                intent.get("intent_function")
                                if isinstance(intent, Mapping)
                                else None
                            )
                            body = (
                                function.get("body")
                                if isinstance(function, Mapping)
                                else None
                            )
                            predicate_choices = (
                                self._semantic_access_predicate_choices(body)
                                if isinstance(body, list)
                                else None
                            )
                        if (
                            predicate_choices is None
                            or len(predicate_choices) != len(expected)
                        ):
                            return (
                                "canonical_source_access_wire_template_is_malformed",
                                {"row_index": index, "field": field},
                            )
                        if value not in predicate_choices[index]:
                            return (
                                "access_declaration_provider_predicate_must_select_compatible_boolean_result",
                                {"row_index": index, "field": field},
                            )
                    if field in {"node_ref", "effect_node_ref"}:
                        if consumer_choices is None:
                            buffer = self._semantic_construction_buffer
                            j = buffer.get("j") if isinstance(buffer, Mapping) else None
                            intent = j.get("intent") if isinstance(j, Mapping) else None
                            function = (
                                intent.get("intent_function")
                                if isinstance(intent, Mapping)
                                else None
                            )
                            body = (
                                function.get("body")
                                if isinstance(function, Mapping)
                                else None
                            )
                            consumer_choices = (
                                self._semantic_access_consumer_choices(body)
                                if isinstance(body, list)
                                else None
                            )
                        if (
                            consumer_choices is None
                            or len(consumer_choices) != len(expected)
                        ):
                            return (
                                "canonical_source_access_wire_template_is_malformed",
                                {"row_index": index, "field": field},
                            )
                        if value not in consumer_choices[index]:
                            return (
                                "access_declaration_provider_node_ref_must_select_direct_consumer",
                                {
                                    "row_index": index,
                                    "field": field,
                                    "provider_local_operation_labels": sorted(
                                        consumer_choices[index]
                                    )[:64],
                                },
                            )
                    continue
                if field in slot["required_fields"] and row.get(field) != template.get(field):
                    return (
                        "access_declaration_must_match_source_wire_template",
                        {
                            "row_index": index,
                            "field": field,
                            "category": "source_wire_value_mismatch",
                        },
                    )
            if (
                "node_ref" in provider_owned
                and "effect_node_ref" in provider_owned
                and row.get("node_ref") != row.get("effect_node_ref")
            ):
                return (
                    "access_declaration_effect_node_owner_must_match_access_owner",
                    {"row_index": index, "field": "effect_node_ref"},
                )
        return None

    def _semantic_access_failure_template(self, accesses: Sequence[Any]) -> dict[str, Any] | None:
        """Return a value-free shape for the first failed access row.

        The initial authoring brief already carries the exact Source wire row.
        A failed declaration must not echo either that row's literal values or
        the rejected declaration back into provider feedback.
        """

        def safe_template(slot: Mapping[str, Any]) -> dict[str, Any]:
            required = slot.get("required_fields")
            provider_owned = slot.get("provider_owned_fields", ())
            if not isinstance(required, list):
                return {}
            owned = set(provider_owned) if isinstance(provider_owned, (list, tuple)) else set()
            result: dict[str, Any] = {}
            for field in required:
                if field == "access_label":
                    result[field] = slot.get("access_label")
                elif field == "ordinal":
                    result[field] = slot.get("ordinal")
                elif field in owned:
                    result[field] = "<provider_local_value>"
                else:
                    result[field] = "<copy_from_initial_source_wire>"
            return result

        expected = self._semantic_construction_slots["accesses"]
        for index, slot in enumerate(expected):
            row = accesses[index] if index < len(accesses) else None
            if (
                not isinstance(row, Mapping)
                or row.get("access_label") != slot["access_label"]
                or row.get("ordinal") != slot["ordinal"]
                or "id" in row
                or any(field not in row for field in slot["required_fields"])
            ):
                return safe_template(slot)
            template = slot.get("wire_template")
            if isinstance(template, Mapping):
                provider_owned = slot.get("provider_owned_fields", ())
                if not isinstance(provider_owned, (list, tuple)):
                    return safe_template(slot)
                for field in (
                    "kind",
                    "tensor",
                    "index_expression",
                    "predicate",
                    "fill_value",
                    "node_ref",
                    "effect_kind",
                    "effect_target",
                    "effect_node_ref",
                ):
                    if field in provider_owned:
                        value = row.get(field)
                        if (
                            not isinstance(value, str)
                            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]*", value) is None
                            or value.startswith("<")
                        ):
                            return safe_template(slot)
                    elif field in slot["required_fields"] and row.get(field) != template.get(field):
                        return safe_template(slot)
                if (
                    "node_ref" in provider_owned
                    and "effect_node_ref" in provider_owned
                    and row.get("node_ref") != row.get("effect_node_ref")
                ):
                    return safe_template(slot)
        return None

    def _semantic_output_value_types(
        self,
        function: Mapping[str, Any],
    ) -> tuple[set[str], dict[str, tuple[str, tuple[str | int, ...]]]] | None:
        """Project provider labels which can legally escape as outputs.

        All local results remain useful for distinguishing an unknown label
        from a region-local one.  The typed map is deliberately narrower: it
        contains direct Source inputs and top-level operation results whose
        complete type is already authored in this same graph.  This mirrors
        the visibility/type preconditions of ``canonical_output_records``
        without constructing an output mapping or selecting a value.
        """

        rows = semantic_operation_wire_rows(function)
        if rows is None:
            return None
        all_local_results: set[str] = set()
        visible_types = {
            name: value_type
            for name, value_type in self._semantic_source_tensor_value_types.items()
            if name in self._semantic_source_input_tensor_handles
        }
        for path, operation in rows:
            raw_result = operation.get("result")
            results = raw_result if isinstance(raw_result, list) else (raw_result,)
            all_local_results.update(
                result for result in results if isinstance(result, str)
            )
            if not isinstance(path, str) or re.fullmatch(
                r"body\[[0-9]+\]", path
            ) is None:
                continue
            operation_type = operation.get("type")
            if (
                isinstance(raw_result, str)
                and isinstance(operation_type, Mapping)
                and isinstance(operation_type.get("dtype"), str)
                and isinstance(operation_type.get("shape"), list)
            ):
                visible_types[raw_result] = (
                    operation_type["dtype"],
                    tuple(operation_type["shape"]),
                )
                continue
            if operation.get("op") != "loop" or not isinstance(raw_result, list):
                continue
            carries = operation.get("carry")
            if not isinstance(carries, list) or len(carries) != len(raw_result):
                continue
            for result, carry in zip(raw_result, carries, strict=True):
                carry_type = carry.get("type") if isinstance(carry, Mapping) else None
                if (
                    isinstance(result, str)
                    and isinstance(carry_type, Mapping)
                    and isinstance(carry_type.get("dtype"), str)
                    and isinstance(carry_type.get("shape"), list)
                ):
                    visible_types[result] = (
                        carry_type["dtype"],
                        tuple(carry_type["shape"]),
                    )
        return all_local_results, visible_types

    def _semantic_graph_missing_output_type(
        self,
        body: Sequence[Any],
    ) -> tuple[int, tuple[str, tuple[str | int, ...]]] | None:
        """Return the first Source output type absent from graph-visible values.

        This is an existence proof only.  Source inputs and top-level typed
        results are the same function-visible values accepted later by output
        selection.  The host neither chooses one of those values nor creates a
        producer; a missing type sends the complete graph back to its provider
        while graph authoring is still the active transaction phase.
        """

        projection = self._semantic_output_value_types({"body": list(body)})
        if projection is None:
            return None
        _local_results, visible_types = projection
        available_types = frozenset(visible_types.values())
        for ordinal, required_type in enumerate(
            self._semantic_source_output_tensor_value_types
        ):
            if required_type not in available_types:
                return ordinal, required_type
        return None

    def _semantic_access_predicate_choices(
        self,
        body: Sequence[Any],
    ) -> tuple[frozenset[str], ...] | None:
        """Return compatible provider-local booleans for each access slot.

        The graph owns result labels/types and the Source wire owns each
        access target.  This projection only asks whether an already authored
        boolean has the same unique guard projection later consumed by the
        canonical constructor.  It never selects a label or creates a guard.
        """

        projection = self._semantic_output_value_types({"body": list(body)})
        if projection is None:
            return None
        local_results, visible_types = projection
        choices: list[frozenset[str]] = []
        for slot in self._semantic_construction_slots["accesses"]:
            if not isinstance(slot, Mapping):
                return None
            provider_owned = slot.get("provider_owned_fields", ())
            template = slot.get("wire_template")
            if (
                not isinstance(provider_owned, (list, tuple))
                or not isinstance(template, Mapping)
            ):
                return None
            if "predicate" not in provider_owned:
                choices.append(frozenset())
                continue
            tensor = template.get("tensor")
            kind = template.get("kind")
            target_type = (
                self._semantic_source_tensor_value_types.get(tensor)
                if isinstance(tensor, str)
                else None
            )
            if target_type is None or kind not in {"load", "store", "update"}:
                return None
            compatible: set[str] = set()
            for label in local_results:
                predicate_type = visible_types.get(label)
                if (
                    predicate_type is None
                    or predicate_type[0].casefold() not in {"i1", "bool"}
                ):
                    continue
                if kind != "load" or unique_access_guard_projection_axes(
                    predicate_type[1], target_type[1]
                ) is not None:
                    compatible.add(label)
            choices.append(frozenset(compatible))
        return tuple(choices)

    def _semantic_access_consumer_choices(
        self,
        body: Sequence[Any],
    ) -> tuple[frozenset[str], ...] | None:
        """Return provider-local direct consumers for guarded load slots."""

        rows = semantic_operation_wire_rows({"body": list(body)})
        if rows is None:
            return None
        choices: list[frozenset[str]] = []
        for slot in self._semantic_construction_slots["accesses"]:
            if not isinstance(slot, Mapping):
                return None
            provider_owned = slot.get("provider_owned_fields", ())
            template = slot.get("wire_template")
            if (
                not isinstance(provider_owned, (list, tuple))
                or not isinstance(template, Mapping)
            ):
                return None
            if "node_ref" not in provider_owned:
                choices.append(frozenset())
                continue
            tensor = template.get("tensor")
            if not isinstance(tensor, str) or template.get("kind") != "load":
                return None
            compatible: set[str] = set()
            for _path, operation in rows:
                operation_id = operation.get("id")
                operands = operation.get("operands")
                result = operation.get("result")
                if (
                    isinstance(operation_id, str)
                    and isinstance(operands, list)
                    and isinstance(result, str)
                    and operands.count(tensor) == 1
                ):
                    compatible.add(operation_id)
            choices.append(frozenset(compatible))
        return tuple(choices)

    def _semantic_graph_missing_access_predicate_type(
        self,
        body: Sequence[Any],
    ) -> tuple[int, int] | None:
        """Return the first masked access without a compatible graph value."""

        choices = self._semantic_access_predicate_choices(body)
        if choices is None:
            return None
        for ordinal, (slot, labels) in enumerate(
            zip(
                self._semantic_construction_slots["accesses"],
                choices,
                strict=True,
            )
        ):
            provider_owned = slot.get("provider_owned_fields", ())
            if "predicate" in provider_owned and not labels:
                return ordinal, 0
        return None

    def _semantic_graph_missing_access_consumer_id(
        self,
        body: Sequence[Any],
    ) -> tuple[int, int] | None:
        """Return the first guarded load without a named direct consumer."""

        choices = self._semantic_access_consumer_choices(body)
        if choices is None:
            return None
        for ordinal, (slot, labels) in enumerate(
            zip(
                self._semantic_construction_slots["accesses"],
                choices,
                strict=True,
            )
        ):
            provider_owned = slot.get("provider_owned_fields", ())
            if "node_ref" in provider_owned and not labels:
                return ordinal, 0
        return None

    def _semantic_output_slot_diagnostic(self, mappings: Sequence[Any]) -> str | None:
        """Reject an incomplete output selection before finalization."""

        expected = self._semantic_construction_slots["outputs"]
        if len(mappings) != len(expected):
            return "output_declaration_must_cover_each_source_output"
        buffer = self._semantic_construction_buffer
        j = buffer.get("j") if isinstance(buffer, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        value_projection = (
            self._semantic_output_value_types(function)
            if isinstance(function, Mapping)
            else None
        )
        if value_projection is None:
            return "output_declaration_requires_declared_operation_graph"
        local_results, visible_types = value_projection
        for index, row in enumerate(mappings):
            if not isinstance(row, Mapping):
                return "output_declaration_rows_must_be_json_objects"
            # The semantic write projection is deliberately narrower than the
            # final construction record.  ``output`` and ``node_ref`` belong
            # to the host allocator; accepting them here creates a second
            # output wire and postpones a deterministic error until
            # CandidateCheck.
            if set(row) != {"value"}:
                return "output_declaration_must_contain_only_value"
            if "value" not in row:
                return "output_declaration_is_missing_required_source_slot_field"
            value = row.get("value")
            if not isinstance(value, str) or (
                value not in local_results
                and value not in self._semantic_source_input_tensor_handles
            ):
                return "output_declaration_value_must_reference_source_input_or_local_result"
            if value not in visible_types:
                return "output_declaration_value_must_be_function_visible"
            if visible_types[value] != self._semantic_source_output_tensor_value_types[index]:
                return "output_declaration_value_type_must_match_source_output"
        return None

    def _semantic_output_failure_template(self, mappings: Sequence[Any]) -> dict[str, Any] | None:
        """Return the sole legal provider row for the first bad output slot."""

        expected = self._semantic_construction_slots["outputs"]
        for index, slot in enumerate(expected):
            row = mappings[index] if index < len(mappings) else None
            if not isinstance(row, Mapping) or set(row) != {"value"}:
                template = slot.get("wire_template")
                return copy.deepcopy(dict(template)) if isinstance(template, Mapping) else None
        return None

    def _semantic_output_failure_diagnostic(
        self,
        mappings: Sequence[Any],
        detail: str,
    ) -> dict[str, Any] | None:
        """Return safe provider-local output choices for a bad value selection.

        A Source output tensor is a boundary declaration, not an SSA producer.
        When that distinction rejects an output mapping, the only useful safe
        correction hint is the local results already declared by this same
        provider graph.  Do not expose candidate bytes, Source text, canonical
        identities, or host-created output records.
        """

        if detail not in {
            "output_declaration_value_must_reference_source_input_or_local_result",
            "output_declaration_value_must_be_function_visible",
            "output_declaration_value_type_must_match_source_output",
        }:
            return None
        buffer = self._semantic_construction_buffer
        j = buffer.get("j") if isinstance(buffer, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        value_projection = (
            self._semantic_output_value_types(function)
            if isinstance(function, Mapping)
            else None
        )
        if value_projection is None:
            return None
        local_results, visible_types = value_projection
        row_index = next(
            (
                index
                for index, row in enumerate(mappings)
                if not isinstance(row, Mapping)
                or not isinstance(row.get("value"), str)
                or row.get("value") not in local_results
                and row.get("value") not in self._semantic_source_input_tensor_handles
                or row.get("value") not in visible_types
                or index >= len(self._semantic_source_output_tensor_value_types)
                or visible_types.get(row.get("value"))
                != self._semantic_source_output_tensor_value_types[index]
            ),
            None,
        )
        if row_index is None or row_index >= len(self._semantic_source_output_tensor_value_types):
            return None
        expected_type = self._semantic_source_output_tensor_value_types[row_index]
        results = sorted(
            value
            for value, value_type in visible_types.items()
            if value in local_results and value_type == expected_type
        )
        return {
            "row_index": row_index,
            "field": "value",
            "provider_local_result_labels": results[:64],
        }

    def _normalize_host_owned_output_ordinals(
        self,
        mappings: Sequence[Any],
    ) -> tuple[list[Any], bool]:
        """Remove only redundant ABI ordinals from semantic output selections.

        Semantic output rows are ordered by their array position and the host
        owns the final ABI ordinal.  A provider sometimes echoes that ordinal
        from the transfer ledger, which is mechanically redundant but used to
        force another model turn.  Repair it only when every ordinal exactly
        matches the already-fixed Source order; any other extra field remains
        a fail-closed semantic-wire error.
        """

        repaired: list[Any] = []
        changed = False
        for ordinal, row in enumerate(mappings):
            if (
                isinstance(row, Mapping)
                and set(row) == {"ordinal", "value"}
                and row.get("ordinal") == ordinal
            ):
                repaired.append({"value": copy.deepcopy(row.get("value"))})
                changed = True
            else:
                repaired.append(row)
        return repaired, changed

    def _semantic_operation_graph_diagnostic(
        self,
        body: Sequence[Any],
    ) -> str | None:
        """Validate the provider operation envelope at the MCP write boundary.

        The canonical schema has two distinct names in an operation row:
        ``op`` is the closed J operation name, while ``id`` and ``result``
        are provider-local labels consumed by the host allocator.  Keeping
        this check here prevents a provider-local ``op0`` or a host-owned
        ``v*`` from surviving several MCP calls before CandidateCheck reports
        the same deterministic contract error.
        """

        try:
            namespace = json.loads(self._semantic_operation_namespace)
        except (TypeError, ValueError, json.JSONDecodeError):
            return "canonical_operation_namespace_is_malformed"
        operation_catalog = namespace.get("operation_catalog")
        if not isinstance(operation_catalog, list) or any(
            not isinstance(name, str) for name in operation_catalog
        ):
            return "canonical_operation_namespace_is_malformed"
        catalog = frozenset(operation_catalog)
        canonical_prefixes = frozenset({"n", "v", "x", "p", "c", "a", "e"})
        local_label = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")

        def valid_local_label(value: Any) -> bool:
            if not isinstance(value, str) or not local_label.fullmatch(value):
                return False
            return not any(
                re.fullmatch(rf"{prefix}[0-9]+", value)
                for prefix in canonical_prefixes
            )

        def visit(value: Any) -> str | None:
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, Mapping):
                        return "operation_graph_rows_must_be_json_objects"
                    diagnostic = visit(item)
                    if diagnostic is not None:
                        return diagnostic
                return None
            if not isinstance(value, Mapping):
                return "operation_graph_rows_must_be_json_objects"
            if "op" not in value:
                return "operation_row_must_include_canonical_op_field"
            operation = value.get("op")
            if operation not in catalog:
                return "operation_op_must_be_canonical_j_name"
            if operation not in STRUCTURED_INTENT_OPS:
                allowed = {"id", "op", "result", "operands", "attrs", "type"}
                if set(value) - allowed:
                    return "operation_must_use_only_canonical_leaf_fields"
                if "attrs" not in value or "type" not in value:
                    return "operation_must_include_required_leaf_fields"
                attrs = value.get("attrs")
                if not isinstance(attrs, Mapping):
                    return "operation_attrs_must_be_json_object"
                try:
                    attr_contract = operation_attr_contract(str(operation))
                except (KeyError, TypeError, ValueError):
                    # The operation catalog has already been checked above.
                    # A contract lookup failure is therefore host-only state,
                    # not something safe or useful to disclose to a provider.
                    return "canonical_operation_namespace_is_malformed"
                allowed_attrs = attr_contract.get("allowed")
                required_attrs = attr_contract.get("required")
                if not isinstance(allowed_attrs, frozenset) or not isinstance(
                    required_attrs, frozenset
                ):
                    return "canonical_operation_namespace_is_malformed"
                if set(attrs) - allowed_attrs:
                    return "operation_attrs_must_use_canonical_leaf_fields"
                if required_attrs - set(attrs):
                    return "operation_attrs_must_include_required_leaf_fields"
                properties = attr_contract.get("properties")
                if not isinstance(properties, Mapping):
                    return "canonical_operation_namespace_is_malformed"
                if any(
                    not _operation_attr_value_matches_schema(
                        attr_value,
                        properties.get(attr_name),
                    )
                    for attr_name, attr_value in attrs.items()
                ):
                    return "operation_attr_value_must_match_canonical_leaf_contract"
                if "operands" in value and not isinstance(value.get("operands"), list):
                    return "operation_operands_must_be_json_array"
            if "id" in value and value.get("id") is not None and not valid_local_label(value.get("id")):
                return "operation_id_must_be_provider_local_label"
            result = value.get("result")
            result_values = result if isinstance(result, list) else [result]
            if result is not None and any(
                not valid_local_label(item) for item in result_values
            ):
                return "operation_result_must_be_provider_local_label"
            # Only structured operation regions contain operation rows.  Do
            # not recurse into ordinary ``attrs`` or ``type`` mappings: doing
            # that would mistake a valid leaf object for an operation which
            # lacks ``op`` and reproduce the historical live/replay split.
            if operation == "loop":
                child = value.get("body")
                if child is not None:
                    diagnostic = visit(child)
                    if diagnostic is not None:
                        return diagnostic
            elif operation in {"effect_if", "value_if"}:
                for arm_name in ("positive", "negative"):
                    arm = value.get(arm_name)
                    if not isinstance(arm, Mapping):
                        continue
                    child = arm.get("body")
                    if child is not None:
                        diagnostic = visit(child)
                        if diagnostic is not None:
                            return diagnostic
            return None

        envelope_diagnostic = visit(list(body))
        if envelope_diagnostic is not None:
            return envelope_diagnostic

        # This is deliberately an ingress-only def-use check.  It rejects a
        # reference which cannot possibly be resolved by the host allocator,
        # but does not infer a missing producer, rewrite an ``arg*`` spelling,
        # or attempt structured SSA validation.  The canonical constructor
        # remains responsible for complete loop/branch/scope semantics.
        rows = semantic_operation_wire_rows({"body": list(body)})
        if rows is None:
            return "operation_graph_lexical_wire_is_malformed"
        visible = set(self._semantic_source_tensor_handles)
        value_types = dict(self._semantic_source_tensor_value_types)

        def tensor_shape(operation: Mapping[str, Any]) -> tuple[str | int, ...] | None:
            operation_type = operation.get("type")
            if (
                not isinstance(operation_type, Mapping)
                or not isinstance(operation_type.get("dtype"), str)
                or not isinstance(operation_type.get("shape"), list)
            ):
                return None
            shape = operation_type["shape"]
            if any(
                not (
                    isinstance(extent, str)
                    and bool(extent)
                    or isinstance(extent, int)
                    and not isinstance(extent, bool)
                    and extent >= 0
                )
                for extent in shape
            ):
                return None
            return tuple(shape)

        def result_shape_matches_known_operands(
            operation: Mapping[str, Any],
            operands: Sequence[Any],
        ) -> bool:
            """Reject only rank/shape loss with a unique local proof.

            Full broadcasting semantics remain the canonical constructor's
            responsibility. The graph ingress merely blocks a result shape
            that contradicts one known operand exactly, or the sole maximal
            non-scalar shape among otherwise scalar/equal operands. This
            prevents a provider from shrinking a projected Source tensor to a
            vector and reaching CandidateCheck as a missing typed-operation
            bundle, without choosing an interpretation for general broadcast.
            """

            result = tensor_shape(operation)
            known = [
                value_types[operand][1]
                for operand in operands
                if isinstance(operand, str) and operand in value_types
            ]
            if result is None or not known:
                return True
            op = operation.get("op")
            if op in {"cast", "map_unary"} and len(known) == 1:
                return result == known[0]
            if op != "map_binary" or len(known) != 2:
                return True
            maximum_rank = max(len(shape) for shape in known)
            maximal = {shape for shape in known if len(shape) == maximum_rank}
            if len(maximal) != 1:
                return True
            expected = next(iter(maximal))
            if all(shape == expected or not shape for shape in known):
                return result == expected
            return True

        for _path, operation in rows:
            operands = operation.get("operands", ())
            if not isinstance(operands, list):
                return "operation_operands_must_be_json_array"
            for operand in operands:
                if not isinstance(operand, str):
                    return "operation_operand_must_reference_source_tensor_or_prior_local_result"
                if re.fullmatch(r"t[0-9]+", operand) is not None:
                    if operand not in self._semantic_source_tensor_handles:
                        return (
                            "operation_operand_must_reference_source_tensor_"
                            "or_prior_local_result"
                        )
                elif operand not in visible:
                    return (
                        "operation_operand_must_reference_source_tensor_"
                        "or_prior_local_result"
                    )
            # Resolve each supplied operand before judging the operation's
            # total arity.  A dangling provider label is independently
            # correctable even when it also makes a zero-input operation
            # over-arity; returning its precise def-use receipt preserves the
            # same declaration classification in MCP and direct replay.
            attrs = operation.get("attrs")
            if not operation_operands_match_arity(
                str(operation.get("op")),
                operands,
                attrs if isinstance(attrs, Mapping) else None,
            ):
                return "operation_operands_must_match_canonical_leaf_arity"
            result = operation.get("result")
            result_values = result if isinstance(result, list) else (result,)
            for value in result_values:
                if value is None:
                    continue
                if not isinstance(value, str) or value in visible:
                    return "operation_result_must_be_new_provider_local_label"
                # Host canonicalization can allocate n*/v* identities, but
                # it cannot infer the provider-owned logical result type.
                # The canonical operation table consumes a value-type object,
                # not a bare dtype spelling: ordinary tensors use
                # {"dtype": "f32", "shape": ["dim0", ...]}.  Letting a
                # missing or string-only type through this write boundary used
                # to make an otherwise complete Source-ledger submission fail
                # later as canonical_construction_bundle_operation_incomplete.
                if operation.get("op") not in STRUCTURED_INTENT_OPS:
                    operation_type = operation.get("type")
                    tensor_type = (
                        isinstance(operation_type, Mapping)
                        and isinstance(operation_type.get("dtype"), str)
                        and bool(operation_type.get("dtype"))
                        and isinstance(operation_type.get("shape"), list)
                    )
                    structured_type = (
                        isinstance(operation_type, Mapping)
                        and operation_type.get("kind") in {"record", "ragged"}
                    )
                    if not tensor_type and not structured_type:
                        return "operation_result_must_have_complete_value_type"
                    if tensor_type and any(
                        not (
                            isinstance(extent, str)
                            and bool(extent)
                            or isinstance(extent, int)
                            and not isinstance(extent, bool)
                            and extent >= 0
                        )
                        for extent in operation_type["shape"]
                    ):
                        return "operation_type_shape_must_use_valid_provider_extent"
                    if tensor_type and any(
                        isinstance(extent, str)
                        and re.fullmatch(r"x[0-9]+", extent) is not None
                        for extent in operation_type["shape"]
                    ):
                        # ``x*`` is allocated only after this semantic write.
                        # Reject it at the declaration which contains it so a
                        # provider can correct the public ``dim*`` spelling
                        # instead of consuming its one CandidateCheck attempt.
                        return "operation_type_shape_must_use_provider_dimension_alias"
                    if tensor_type and operation.get("op") == "iota":
                        # Iota has no operands from which the generic
                        # rank-preservation check can recover its type.  Its
                        # semantic result is nevertheless fully fixed by its
                        # own public attrs: one dimension ``n`` and the
                        # declared integer dtype.  Enforce that local,
                        # schema-derived relation at graph declaration so a
                        # rank-two iota cannot spend the single finalization
                        # attempt only to fail Decoder type inference.
                        iota_extent = attrs.get("n") if isinstance(attrs, Mapping) else None
                        iota_dtype = attrs.get("dtype") if isinstance(attrs, Mapping) else None
                        if (
                            tensor_shape(operation) != (iota_extent,)
                            or (
                                isinstance(iota_dtype, str)
                                and operation_type.get("dtype") != iota_dtype
                            )
                        ):
                            return "operation_iota_result_type_must_match_attrs"
                    if tensor_type and operation.get("op") in {"map_binary", "select"}:
                        result_shape = tensor_shape(operation)
                        if result_shape is not None:
                            requires_boundary_broadcast = any(
                                isinstance(operand, str)
                                and operand in value_types
                                and self._unique_nontrailing_broadcast_axes(
                                    value_types[operand][1], result_shape
                                )
                                is not None
                                for operand in operands
                            )
                            if requires_boundary_broadcast and not isinstance(
                                operation.get("id"), str
                            ):
                                # The provider owns the boundary relation, but
                                # its node_ref must name the same local graph
                                # node.  Reject before commit when an implicit
                                # id would make that relation unwriteable; do
                                # not allocate or infer an id here.
                                return "operation_nontrailing_broadcast_requires_local_id"
                    if not result_shape_matches_known_operands(operation, operands):
                        return "operation_result_type_shape_must_match_operands"
                visible.add(value)
                if tensor_type:
                    shape = tensor_shape(operation)
                    dtype = operation_type.get("dtype")
                    if shape is not None and isinstance(dtype, str):
                        value_types[value] = (dtype, shape)
        if self._semantic_graph_missing_output_type(body) is not None:
            return "operation_graph_must_expose_each_source_output_type"
        if self._semantic_graph_missing_access_predicate_type(body) is not None:
            return "operation_graph_must_expose_compatible_access_predicate_type"
        if self._semantic_graph_missing_access_consumer_id(body) is not None:
            return "operation_graph_must_expose_access_consumer_id"
        operation_counts: dict[str, int] = {}
        for _path, operation in rows:
            name = operation.get("op")
            if isinstance(name, str):
                operation_counts[name] = operation_counts.get(name, 0) + 1
        if any(
            operation_counts.get(name, 0) < required
            for name, required in self._semantic_entry_required_operation_families.items()
        ):
            return "operation_graph_must_cover_source_operation_families"
        for operation, attribute, value, required in (
            self._semantic_entry_required_operation_attributes
        ):
            actual = sum(
                1
                for _path, row in rows
                if row.get("op") == operation
                and isinstance(row.get("attrs"), Mapping)
                and row["attrs"].get(attribute) == value
            )
            if actual < required:
                return "operation_graph_must_cover_source_operation_attributes"
        return None

    def _semantic_graph_requires_transfer_reread(self) -> bool:
        """Return whether access authoring lacks its graph-bound ledger read."""

        return (
            self._semantic_construction_buffer is not None
            and "declare_operation_graph" in self._semantic_construction_completed_steps
            and "declare_accesses" not in self._semantic_construction_completed_steps
            and self._semantic_operation_graph_transfer_generation is not None
            and self._semantic_transfer_read_generation
            <= self._semantic_operation_graph_transfer_generation
        )

    def _semantic_transfer_identity_is_current(
        self, result: Mapping[str, Any]
    ) -> bool:
        """Check a transfer receipt without retaining its Source-owned rows."""

        if result.get("returncode") != 0:
            return False
        try:
            payload = json.loads(str(result.get("output", "")))
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        return (
            isinstance(payload, Mapping)
            and payload.get("kind") == "canonical_semantic_transfer_ledger"
            and payload.get("construction_identity")
            == self._semantic_construction_identity
        )

    def _mcp_phase_surface(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return the only MCP resource/tool choices valid in this phase.

        This is the one lifecycle authority used both to constrain the
        provider-facing Responses function schema and to reject a forged or
        stale action at dispatch.  The resource names here are view names;
        URI conversion remains owned by the MCP catalogue.
        """

        if self._semantic_construction_buffer is not None:
            next_step = self._semantic_next_required_action()
            if next_step is None:
                return (), ()
            if self._semantic_graph_requires_transfer_reread():
                # Do not let an operation graph displace the exact Source-order
                # occurrence table from the provider context.  This is a read
                # barrier, not a new construction step or a relaxation of the
                # complete-array access contract.
                return ("semantic_transfer",), ()
            return (), (next_step,)

        if not self._semantic_entry_read:
            return ("source_brief",), ()
        lookup_allowed = (
            bool(self._semantic_entry_missing_operation_leaves)
            and self._semantic_entry_schema_lookup_calls == 0
        )
        if not self._semantic_entry_transfer_ledger_read:
            # The transfer ledger is the source-order authority and must be
            # read before selecting a missing operation leaf. Keeping this edge
            # singular prevents a provider from looking up operations before it
            # has seen the transfer/output slots it must connect.
            return ("semantic_transfer",), ()
        if lookup_allowed:
            # Every Source-declared missing leaf is fetched atomically before
            # authoring; begin_candidate cannot bypass this prerequisite.
            return (), ("schema_lookup",)
        return (), ("begin_candidate",)

    def response_tools(self) -> list[dict[str, Any]]:
        """Project the current phase authority into one Responses tool schema."""

        from minisweagent.models.utils.actions_toolcall_response import (
            mcp_call_tool_response_api_for_surface,
        )

        views, tools = self._mcp_phase_surface()
        schema_lookup_items = (
            tuple(
                f"op:{leaf}"
                for leaf in sorted(self._semantic_entry_missing_operation_leaves)
            )
            if tools == ("schema_lookup",)
            else None
        )
        return [
            mcp_call_tool_response_api_for_surface(
                resource_uris=mcp_resource_uris_for_views(views),
                tool_names=tools,
                schema_lookup_items=schema_lookup_items,
            )
        ]

    def _mcp_action_is_in_phase_surface(self, action: Mapping[str, Any]) -> bool:
        """Return whether a parsed gateway action is currently exposed."""

        views, tools = self._mcp_phase_surface()
        if action.get("kind") == "resource":
            return action.get("view") in views
        if action.get("kind") == "tool":
            return action.get("name") in tools
        return False

    def _reject_mcp_action_outside_phase_surface(
        self,
        action: Mapping[str, Any],
    ) -> None:
        """Fail closed when a stale/static client bypasses the dynamic schema."""

        raise MCPRequestError("MCP action is not available in the current transaction phase")

    def _mcp_dispatch(self, action: Mapping[str, Any]) -> Mapping[str, Any]:
        """Dispatch one validated MCP resource/tool request to closed operations."""

        if not self._mcp_action_is_in_phase_surface(action):
            self._reject_mcp_action_outside_phase_surface(action)
        kind = action.get("kind")
        if kind == "resource":
            view = action.get("view")
            post_graph_transfer_reread = (
                view == "semantic_transfer"
                and self._semantic_graph_requires_transfer_reread()
            )
            if (
                self._semantic_construction_buffer is not None
                and not post_graph_transfer_reread
            ):
                self._raise_lift_preparation_failure(
                    detail="semantic_construction_forbids_resource_reads_after_begin",
                    attempted_step=f"resource:{view}",
                )
            if not post_graph_transfer_reread:
                if not self._semantic_entry_read and view != "source_brief":
                    raise MCPRequestError(
                        "read intentir://lift/source/brief before any other authoring resource"
                    )
                if view == "source_brief" and self._semantic_entry_read:
                    raise MCPRequestError(
                        "semantic-entry authoring permits one source/brief read"
                    )
                if self._semantic_entry_read and view not in _SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS:
                    raise MCPRequestError(
                        "semantic-entry authoring permits one exact supplemental fact read before construction"
                    )
                # The semantic transfer ledger is the sole Source-order
                # authority for access, fill, owner and output slots.
                if (
                    self._semantic_entry_read
                    and not self._semantic_entry_transfer_ledger_read
                    and view != "semantic_transfer"
                ):
                    raise MCPRequestError(
                        "read intentir://lift/source/transfer-ledger before any other supplemental fact"
                    )
                if (
                    view in _SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS
                    and self._semantic_entry_supplemental_read is not None
                ):
                    raise MCPRequestError(
                        "semantic-entry authoring permits one total supplemental fact read"
                    )
            resource_action: dict[str, Any] = {}
            resource_action.update(
                {
                    key: value
                    for key, value in action.items()
                    if key not in {"kind", "view", "uri"}
                }
            )
            resource_action["view"] = view
            result = self._read_context_view(resource_action)
            if view == "source_brief" and result.get("returncode") == 0:
                try:
                    brief = json.loads(str(result.get("output", "")))
                    entry = brief.get("semantic_authoring_entry")
                    spine = entry.get("source_semantic_spine") if isinstance(entry, Mapping) else None
                    missing = spine.get("missing_operation_contracts") if isinstance(spine, Mapping) else None
                    operations = spine.get("operations") if isinstance(spine, Mapping) else None
                    required_families = (
                        spine.get("required_operation_families")
                        if isinstance(spine, Mapping)
                        else None
                    )
                    required_attributes = (
                        spine.get("required_operation_attributes")
                        if isinstance(spine, Mapping)
                        else None
                    )
                    if (
                        not isinstance(spine, Mapping)
                        or spine.get("status") != "pass"
                    ):
                        reason = (
                            spine.get("reason")
                            if isinstance(spine, Mapping)
                            else None
                        )
                        detail = "semantic_source_operation_spine_unavailable"
                        if (
                            isinstance(reason, str)
                            and re.fullmatch(r"[a-z0-9_]+", reason) is not None
                        ):
                            detail += f":{reason}"
                        self._raise_lift_preparation_failure(
                            detail=detail,
                            attempted_step="read_source_brief",
                        )
                    if (
                        not isinstance(missing, list)
                        or not isinstance(operations, list)
                        or not isinstance(required_families, Mapping)
                        or not isinstance(required_attributes, list)
                        or any(not isinstance(item, Mapping) for item in operations)
                        or any(
                            not isinstance(name, str)
                            or not name
                            or isinstance(count, bool)
                            or not isinstance(count, int)
                            or count < 0
                            for name, count in required_families.items()
                        )
                        or any(
                            not isinstance(row, Mapping)
                            or set(row) != {"op", "attribute", "value", "count"}
                            or not all(
                                isinstance(row.get(name), str) and row.get(name)
                                for name in ("op", "attribute", "value")
                            )
                            or isinstance(row.get("count"), bool)
                            or not isinstance(row.get("count"), int)
                            or row["count"] <= 0
                            for row in required_attributes
                        )
                        or any(
                            not isinstance(item, str)
                            or re.fullmatch(r"[A-Za-z0-9_]+", item) is None
                            for item in missing
                        )
                        or entry.get("construction_identity")
                        != self._semantic_construction_identity
                    ):
                        raise ValueError("source brief semantic identity or spine is malformed")
                    self._semantic_entry_missing_operation_leaves = frozenset(missing)
                    self._semantic_entry_operation_fact_count = len(operations)
                    self._semantic_entry_non_boundary_operation_fact_count = sum(
                        1
                        for item in operations
                        if item.get("kind") != "boundary_transfer"
                    )
                    self._semantic_entry_required_operation_families = {
                        str(name): int(count)
                        for name, count in required_families.items()
                    }
                    self._semantic_entry_required_operation_attributes = tuple(
                        (
                            str(row["op"]),
                            str(row["attribute"]),
                            str(row["value"]),
                            int(row["count"]),
                        )
                        for row in required_attributes
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    self._semantic_entry_read = False
                    return self._structured_tool_error(
                        "source brief lacks a valid semantic-spine lookup policy"
                    )
            if (
                view == "semantic_transfer"
                and result.get("returncode") == 0
                and not self._semantic_transfer_identity_is_current(result)
            ):
                return self._structured_tool_error(
                    "source transfer ledger identity differs from current construction"
                )
            if (
                view in _SEMANTIC_ENTRY_SUPPLEMENTAL_CONTEXT_VIEWS
                and result.get("returncode") == 0
                and not post_graph_transfer_reread
            ):
                self._semantic_entry_supplemental_read = str(view)
                self._semantic_entry_source_read = view == "source"
                self._semantic_entry_transfer_ledger_read = (
                    view == "semantic_transfer"
                )
                if view == "semantic_transfer":
                    self._semantic_transfer_read_generation += 1
            if (
                post_graph_transfer_reread
                and result.get("returncode") == 0
            ):
                self._semantic_transfer_read_generation += 1
                self._semantic_access_transfer_generation = (
                    self._semantic_transfer_read_generation
                )
                self._semantic_construction_audit.append(
                    {
                        "step": "read_transfer_ledger_after_operation_graph",
                        "status": "pass",
                        "transaction": self._semantic_transaction_identity,
                        "transfer_read_generation": (
                            self._semantic_transfer_read_generation
                        ),
                    }
                )
            return result
        if kind == "tool":
            name = action.get("name")
            arguments = action.get("arguments")
            if not isinstance(name, str) or not isinstance(arguments, Mapping):
                raise MCPRequestError("invalid MCP tool dispatch")
            if name == "schema_lookup":
                if self._semantic_construction_buffer is not None:
                    self._raise_lift_preparation_failure(
                        detail="semantic_construction_forbids_schema_lookup_after_begin",
                        attempted_step="schema_lookup",
                    )
                if not self._semantic_entry_read:
                    raise MCPRequestError(
                        "read intentir://lift/source/brief before schema lookup"
                    )
                if self._semantic_entry_schema_lookup_calls >= 1:
                    raise MCPRequestError(
                        "semantic-entry authoring permits one schema lookup"
                    )
                requested = arguments.get("items")
                requested_leaves = (
                    tuple(item[3:] for item in requested)
                    if isinstance(requested, list)
                    and 1 <= len(requested) <= MAX_MCP_SCHEMA_LOOKUP_ITEMS
                    and len(requested) == len(set(requested))
                    and all(
                        isinstance(item, str)
                        and re.fullmatch(r"op:[A-Za-z0-9_]+", item) is not None
                        for item in requested
                    )
                    else ()
                )
                expected_leaves = tuple(
                    sorted(self._semantic_entry_missing_operation_leaves)
                )
                if requested_leaves != expected_leaves:
                    raise MCPRequestError(
                        "schema_lookup must request exactly the Source-declared "
                        "missing operation contracts in canonical order"
                    )
                result = self._read_schema_bundle(dict(arguments))
                if result.get("returncode") == 0:
                    self._semantic_entry_schema_lookup_calls += 1
                return result
            if name in MCP_SEMANTIC_WRITE_TOOL_NAMES:
                return self._mcp_semantic_construction_write(
                    name, dict(arguments)
                )
            raise MCPRequestError(f"unsupported MCP tool: {name}")
        raise MCPRequestError("unsupported MCP dispatch kind")

    def _active_mcp_resource_uris(self) -> tuple[str, ...]:
        """Expose the fresh semantic-entry read surface."""

        return mcp_resource_uris_for_views(_SEMANTIC_ENTRY_CONTEXT_VIEWS)

    @staticmethod
    def _mcp_result(
        result: Mapping[str, Any], *, method: str, is_error: bool = False
    ) -> dict[str, Any]:
        """Wrap an environment result in a bounded MCP text result."""

        payload = dict(result)
        text = payload.get("output", "")
        if not isinstance(text, str):
            text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return {
            "mcp": "intentir-lift",
            "method": method,
            "isError": is_error or payload.get("returncode") not in {0, None},
            "content": [{"type": "text", "text": text}],
            "returncode": payload.get("returncode", 0),
            "exception_info": payload.get("exception_info", ""),
        }

    def _mcp_phase_failure_key(self) -> tuple[str, str]:
        """Return a receipt-only identity for the current lifecycle edge."""

        views, tools = self._mcp_phase_surface()
        if len(views) + len(tools) != 1:
            # Active Lift phases are singular. Retain a deterministic key for
            # malformed or future non-singular surfaces; this remains
            # telemetry, not action-admission authority.
            phase = ",".join(
                [
                    *(f"resource:{view}" for view in views),
                    *(f"tool:{tool}" for tool in tools),
                ]
            ) or "terminal"
        elif views:
            phase = f"resource:{views[0]}"
        else:
            phase = f"tool:{tools[0]}"
        state = {
            "mode": "semantic_entry",
            "phase": phase,
            "construction_started": self._semantic_construction_buffer is not None,
            "completed_steps": sorted(self._semantic_construction_completed_steps),
            "transfer_read_generation": self._semantic_transfer_read_generation,
        }
        phase_sha256 = hashlib.sha256(
            json.dumps(
                state,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return phase, phase_sha256

    @staticmethod
    def _mcp_application_error_text(response: Mapping[str, Any]) -> str:
        """Extract an error rule for hashing without retaining response bytes."""

        content = response.get("content")
        if isinstance(content, list) and len(content) == 1:
            item = content[0]
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                text = item["text"]
                try:
                    payload = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return text
                if isinstance(payload, Mapping):
                    diagnostic = payload.get("diagnostic")
                    if isinstance(diagnostic, str):
                        return diagnostic
                return text
        return str(response.get("returncode", "application_error"))

    def _record_mcp_phase_application_failure(
        self,
        *,
        phase: str,
        phase_sha256: str,
        category: str,
        rule_text: str,
    ) -> None:
        """Fail closed on repeated errors at one candidate lifecycle edge.

        Only hashes and fixed state-machine labels enter evidence. Raw MCP
        output, provider arguments, Source content, and rejected declarations
        are deliberately excluded, so this receipt can never become a repair
        base or a semantic hint.
        """

        rule_sha256 = hashlib.sha256(
            f"{category}\0{rule_text}".encode("utf-8", errors="replace")
        ).hexdigest()
        phase_rows = [
            row
            for row in self._mcp_phase_application_failures
            if row.get("phase_sha256") == phase_sha256
        ]
        repeated_count = 1 + sum(
            row.get("category") == category
            and row.get("rule_sha256") == rule_sha256
            for row in phase_rows
        )
        row = {
            "phase": phase,
            "phase_sha256": phase_sha256,
            "category": category,
            "rule_sha256": rule_sha256,
            "attempt": len(phase_rows) + 1,
            "same_rule_count": repeated_count,
        }
        self._mcp_phase_application_failures.append(row)
        if repeated_count >= _MAX_REPEATED_MCP_PHASE_RULE_FAILURES:
            self._raise_lift_preparation_failure(
                detail="mcp_phase_repeated_application_error",
                attempted_step=phase,
            )
        if len(phase_rows) + 1 >= _MAX_MCP_PHASE_APPLICATION_FAILURES:
            self._raise_lift_preparation_failure(
                detail="mcp_phase_application_error_limit_exhausted",
                attempted_step=phase,
            )

    def _execute_mcp(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Execute an MCP request while retaining the existing sandbox audit."""

        started = time.monotonic()
        method = str(action.get("method", ""))
        params = action.get("params", {})
        phase, phase_sha256 = self._mcp_phase_failure_key()
        error_category: str | None = None
        try:
            if method in {"initialize", "resources/list", "tools/list"}:
                result = self._mcp_server.handle(method, params)
                response = {
                    "mcp": "intentir-lift",
                    "method": method,
                    "isError": False,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                result, sort_keys=True, ensure_ascii=False
                            ),
                        }
                    ],
                    "returncode": 0,
                    "exception_info": "",
                }
            else:
                result = self._mcp_server.dispatch(method, params)
                response = self._mcp_result(result, method=method)
        except (CandidateRejected, LiftPreparationFailed, Submitted):
            # A passing MCP candidate submission is the environment's normal
            # agent-termination signal.  A rejected submission is also a
            # terminal signal for this logical Lift: its bounded receipt is
            # consumed by the host relift lifecycle, not returned as a normal
            # MCP observation for the same agent to reinterpret.
            raise
        except MCPRequestError as exc:
            _views, tools = self._mcp_phase_surface()
            if (
                self._semantic_construction_buffer is None
                and tools == ("begin_candidate",)
            ):
                self._semantic_preconstruction_phase_failures += 1
                if self._semantic_preconstruction_phase_failures >= 2:
                    self._raise_lift_preparation_failure(
                        detail="semantic_construction_not_started_after_begin_exposed",
                        attempted_step="begin_candidate",
                    )
            error_category = "mcp_request_rejected"
            response = {
                "mcp": "intentir-lift",
                "method": method,
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "fail",
                                "diagnostic": f"MCPRequestError: {exc}",
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "returncode": 1,
                "exception_info": "",
            }
        except Exception as exc:
            error_category = f"internal_{type(exc).__name__}"
            response = {
                "mcp": "intentir-lift",
                "method": method,
                "isError": True,
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "fail",
                                "diagnostic": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                    }
                ],
                "returncode": 1,
                "exception_info": "",
            }
        if response["isError"] and (
            error_category is not None
            or phase.startswith("resource:")
            or self._semantic_construction_buffer is None
        ):
            self._record_mcp_phase_application_failure(
                phase=phase,
                phase_sha256=phase_sha256,
                category=error_category or "application_error",
                rule_text=self._mcp_application_error_text(response),
            )
        response["elapsed_ms"] = (time.monotonic() - started) * 1000.0
        self._mcp_requests.append(
            {
                "method": method,
                "status": "error" if response["isError"] else "pass",
                "elapsed_ms": response["elapsed_ms"],
            }
        )
        return {
            "output": json.dumps(response, sort_keys=True) + "\n",
            "returncode": response["returncode"],
            "exception_info": response["exception_info"],
        }

    def _build_startup_context(self) -> tuple[str, dict[str, Any]]:
        """Publish receipts and MCP entry points without embedding Source text."""

        receipts: dict[str, dict[str, Any]] = {}
        total_bytes = 0
        startup_files = (
            "source.py",
            "harness_view.json",
            "canonical_construction.json",
            "feedback_frontier.json",
            "feedback.json",
            "schema/artifact.schema.json",
            "schema/canonical_skeleton.json",
            "schema/canonical_wire_bundle.json",
            "schema/canonical_source_wire_bundle.json",
            "knowledge/lift_contract.md",
            "knowledge/preflight.md",
        )
        for relative in startup_files:
            path = self.config.input_dir / relative
            if path.is_symlink():
                raise ValueError(f"Lift startup input must not be a symlink: {relative}")
            if not path.is_file():
                receipts[relative] = {"missing": True}
                continue
            raw = path.read_bytes()
            total_bytes += len(raw)
            receipts[relative] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        candidate_path = self.config.output_dir / "candidate.json"
        # The output directory is disposable owner state. A stale or complete
        # candidate from any prior path is never an authoring entry.
        try:
            candidate_path.unlink(missing_ok=True)
        except OSError as exc:
            raise ValueError(
                "candidate.json cannot be cleared before fresh semantic authoring"
            ) from exc
        candidate_present = False
        self._semantic_entry_read = False
        self._semantic_entry_source_read = False
        self._semantic_entry_transfer_ledger_read = False
        self._semantic_transfer_read_generation = 0
        self._semantic_operation_graph_transfer_generation = None
        self._semantic_access_transfer_generation = None
        self._semantic_transaction_identity = None
        self._semantic_entry_supplemental_read = None
        self._semantic_entry_schema_lookup_calls = 0
        self._semantic_entry_missing_operation_leaves = frozenset()
        self._semantic_entry_operation_fact_count = 0
        self._semantic_entry_non_boundary_operation_fact_count = 0
        self._semantic_entry_required_operation_families = {}
        self._semantic_entry_required_operation_attributes = ()
        active_resources = [
            "intentir://lift/source/brief",
            "intentir://lift/source/transfer-ledger before construction",
            "an op:NAME schema lookup only when exposed",
            "the currently exposed construction action",
        ]
        payload: dict[str, Any] = {
            "mcp": "intentir-lift",
            "candidate_present": candidate_present,
            "authoring_mode": "semantic_entry",
            "authoring_resource": "intentir://lift/source/brief",
            "semantic_entry_required": True,
            "input_file_count": len(receipts),
            "input_total_bytes": total_bytes,
            "resources": active_resources,
            "authority": (
                "exact Source, Harness, compiler observation, feedback, and schema "
                "facts are available only through bounded MCP reads; no candidate "
                "is mounted, and MCP declarations have no canonical construction "
                "or acceptance authority"
            ),
        }
        receipt = {
            "mcp": payload["mcp"],
            "total_bytes": total_bytes,
            "candidate_present": candidate_present,
            "authoring_mode": "semantic_entry",
            "authoring_resource": payload["authoring_resource"],
            "semantic_entry_required": not candidate_present,
            "files": receipts,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False), receipt

    @staticmethod
    def _is_batched_read_only_command(command: Any) -> bool:
        if not isinstance(command, str) or any(
            token in command
            for token in (";", "|", "&", ">", "<", "\n", "\r", "`", "$(")
        ):
            return False
        try:
            arguments = shlex.split(command)
        except ValueError:
            return False
        if len(arguments) < 3 or arguments[0] != "python":
            return False
        if arguments[1] == "/input/tools/schema_query.py":
            return True
        return (
            arguments[1] == "/input/tools/context_query.py"
            and arguments[2] in _LIFT_READ_ONLY_CONTEXT_COMMANDS
        )

    def _semantic_write_sequence_error(self, names: Sequence[str]) -> str | None:
        """Return an exact protocol error for an invalid construction sequence.

        The canonical profile retains the complete write catalogue for
        extension compatibility, but active Lift construction has exactly one
        mandatory transaction chain.  Optional catalogue entries must not
        become alternate phase transitions merely because an older static
        client can name them.
        """

        required = tuple(self._semantic_write_contract["required_steps"])
        completed = set(self._semantic_construction_completed_steps)
        for name in names:
            if name not in self._semantic_write_contract["steps"]:
                return "semantic_construction_unknown_step"
            if name not in required:
                return "semantic_construction_optional_step_not_permitted"
            if name in completed:
                return "semantic_construction_duplicate_step"
            expected = next((step for step in required if step not in completed), None)
            if expected is None:
                return "semantic_construction_is_already_complete"
            if name != expected:
                return "semantic_construction_only_next_required_step_permitted_" + expected
            completed.add(name)
        return None

    def _semantic_next_required_action(self) -> str | None:
        """Return the sole remaining mandatory write in this transaction."""

        completed = self._semantic_construction_completed_steps
        for step in self._semantic_write_contract["required_steps"]:
            if step not in completed:
                return str(step)
        return None

    def _semantic_next_legal_action(self) -> str | None:
        """Return the next phase edge, including a mandatory read barrier.

        The write sequence alone names ``declare_accesses`` after an operation
        graph. That declaration is nevertheless unavailable until the graph
        has been followed by a fresh transfer-ledger read. Receipts must
        advertise the same phase edge as the dynamic MCP tool surface; otherwise
        a provider is directed to an action that the host immediately rejects.
        """

        if self._semantic_graph_requires_transfer_reread():
            return "read_transfer_ledger_after_operation_graph"
        return self._semantic_next_required_action()

    def _last_operation_graph_failure_receipt(self) -> dict[str, Any] | None:
        """Project the last graph failure without retaining declaration bytes.

        A terminal graph-preparation exit must preserve the same public
        row/type/attribute signal that the immediately preceding MCP receipt
        exposed.  The next owner is nevertheless fresh: this is bounded
        diagnostic metadata, not a graph, candidate, draft, or repair base.
        """

        if not self._semantic_operation_graph_failures:
            return None
        last = self._semantic_operation_graph_failures[-1]
        if not isinstance(last, Mapping):
            return None
        receipt: dict[str, Any] = {
            "failure_count": len(self._semantic_operation_graph_failures),
            "last_rule_code": str(last.get("detail", ""))[:128],
        }
        for key in (
            "attempt",
            "operation_rows",
            "row_index",
            "result_rank",
            "expected_rank",
            "output_ordinal",
            "access_ordinal",
            "function_visible_matching_value_count",
            "function_visible_compatible_boolean_result_count",
            "direct_single_result_consumer_id_count",
            "required_operation_count",
            "actual_operation_count",
        ):
            value = last.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                receipt[key] = value
        for key in (
            "json_container",
            "declaration_sha256",
            "operation",
            "field",
            "attribute",
            "required_attribute_value",
        ):
            value = last.get(key)
            if isinstance(value, str) and value:
                receipt[key] = value[:256]
        for key in ("allowed_fields", "required_fields"):
            value = last.get(key)
            if isinstance(value, list) and all(
                isinstance(item, str) and item for item in value
            ):
                receipt[key] = list(value[:32])
        required_type = last.get("required_output_type")
        if isinstance(required_type, Mapping):
            dtype = required_type.get("dtype")
            shape = required_type.get("shape")
            if (
                isinstance(dtype, str)
                and isinstance(shape, list)
                and all(
                    isinstance(extent, (str, int)) and not isinstance(extent, bool)
                    for extent in shape
                )
            ):
                receipt["required_output_type"] = {
                    "dtype": dtype[:64],
                    "shape": list(shape[:16]),
                }
        return receipt

    def _raise_lift_preparation_failure(
        self,
        *,
        detail: str,
        attempted_step: str | None = None,
    ) -> None:
        """Terminate an unrecoverable pre-candidate construction transaction.

        After a semantic buffer starts, reopening Source ingress, starting a
        second buffer, or skipping a required lifecycle edge cannot repair the
        current transaction.  Continuing used to consume the whole model
        budget while no candidate could exist.  Keep the exit receipt bounded:
        it names only state-machine facts and never carries candidate bytes or
        Source content.
        """

        receipt: dict[str, Any] = {
            "failure_class": "lift_agent_preparation",
            "diagnostic": detail,
            "completed_steps": sorted(self._semantic_construction_completed_steps),
        }
        if attempted_step is not None:
            receipt["attempted_step"] = attempted_step
        if attempted_step == "declare_operation_graph":
            graph_failure = self._last_operation_graph_failure_receipt()
            if graph_failure is not None:
                receipt["operation_graph_failure"] = graph_failure
        next_action = self._semantic_next_legal_action()
        if next_action is not None:
            receipt["next_legal_action"] = next_action
        self._preparation_failure_receipts.append(copy.deepcopy(receipt))
        raise LiftPreparationFailed(
            {
                "role": "exit",
                "content": "LiftPreparationFailed",
                "extra": {
                    "exit_status": "LiftPreparationFailed",
                    "submission": "",
                    "preparation_receipt": receipt,
                },
            }
        )

    def validate_actions(self, actions: list[dict[str, Any]]) -> str | None:
        """Admit independent reads or one ordered semantic construction batch."""

        def is_read_only_mcp_action(action: Any) -> bool:
            if not isinstance(action, Mapping):
                return False
            tool_name = action.get("tool_name")
            if tool_name != "mcp_call_tool":
                return False
            method = action.get("method")
            params = action.get("params")
            if method in {"resources/list", "tools/list"}:
                return isinstance(params, Mapping) and not params
            if method == "tools/call" and isinstance(params, Mapping):
                # schema_lookup is a bounded read-only MCP tool.  It may be
                # batched with resource reads; candidate writes/checks remain
                # single-action below.
                return (
                    set(params) == {"name", "arguments"}
                    and params.get("name") == "schema_lookup"
                    and isinstance(params.get("arguments"), Mapping)
                )
            if method != "resources/read" or not isinstance(params, Mapping):
                return False
            # The gateway is shared with write/check calls.  Permit only the
            # bounded resource URI read shape in a batch; tools/call is always
            # single-action so a candidate/check/patch cannot be coalesced.
            return (
                set(params) <= {
                    "uri", "query", "path", "start", "count", "context",
                    "op", "section",
                }
                and isinstance(params.get("uri"), str)
                and params["uri"].startswith("intentir://lift/")
            )

        def is_ordered_semantic_construction_batch() -> bool:
            """Allow one short, deterministic begin→declare→finalize batch.

            These writes are not independent. They are safe to coalesce only
            because the order is checked before any action executes and each
            step addresses the one in-memory semantic buffer. This keeps the
            agent to a few provider turns without reopening arbitrary patches
            or parallel candidate writes.
            """

            names: list[str] = []
            for action in actions:
                if not isinstance(action, Mapping) or action.get("tool_name") != "mcp_call_tool":
                    return False
                if action.get("method") != "tools/call":
                    return False
                params = action.get("params")
                if not isinstance(params, Mapping) or set(params) != {"name", "arguments"}:
                    return False
                name = params.get("name")
                if name not in MCP_SEMANTIC_WRITE_TOOL_NAMES or not isinstance(
                    params.get("arguments"), Mapping
                ):
                    return False
                names.append(str(name))
            if len(names) != len(set(names)):
                return False
            return self._semantic_write_sequence_error(names) is None

        if len(actions) <= 1:
            return None
        if len(actions) > 4:
            self._multi_action_rejections += 1
            return (
                f"{MULTI_ACTION_PROTOCOL}: at most four read-only or ordered "
                "semantic-construction calls are allowed; no action in this "
                "response was executed"
            )
        if all(
            (
                self._is_batched_read_only_command(action.get("command"))
                if isinstance(action, Mapping)
                and "command" in action
                else is_read_only_mcp_action(action)
            )
            for action in actions
        ):
            return None
        if is_ordered_semantic_construction_batch():
            return None
        self._multi_action_rejections += 1
        return (
            f"{MULTI_ACTION_PROTOCOL}: multiple calls are allowed only for "
            "independent read-only calls or one ordered semantic construction batch; "
            "no action in this response was executed"
        )

    @staticmethod
    def _structured_tool_error(diagnostic: str) -> dict[str, Any]:
        return {
            "output": json.dumps(
                {
                    "mcp": "intentir-lift",
                    "status": "fail",
                    "diagnostic": diagnostic,
                },
                sort_keys=True,
            )
            + "\n",
            "returncode": 2,
            "exception_info": "",
        }

    def _read_context_view(
        self, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        allowed = MCP_CONTEXT_ARGUMENTS
        if not set(action).issubset(allowed):
            return self._structured_tool_error("unknown context-inspection argument")
        view = action.get("view")
        if view not in MCP_CONTEXT_VIEWS:
            return self._structured_tool_error("unsupported context-inspection view")

        def bounded_int(name: str, default: int, lower: int, upper: int) -> int:
            value = action.get(name, default)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
            return value

        def bounded_text(name: str, *, required: bool = False) -> str | None:
            value = action.get(name)
            if value is None and not required:
                return None
            max_chars = (
                MAX_MCP_QUERY_CHARS
                if name == "query"
                else MAX_MCP_PATH_CHARS
                if name == "path"
                else MCP_SELECTOR_CHARS
            )
            if (
                not isinstance(value, str)
                or (required and not value)
                or len(value) > max_chars
            ):
                raise ValueError(f"{name} must be a bounded string")
            return value

        try:
            start = bounded_int("start", 1, 1, 1_000_000)
            fixed_summary_views = MCP_FIXED_SUMMARY_VIEWS
            count_upper = (
                min(mcp_resource_count_upper(str(view)), 500)
                if view in fixed_summary_views
                else 80
                if view in {
                    "harness_abi", "source_obligations", "semantic_transfer",
                }
                else 160
                if view in {"source_ttir"}
                else 200
                if view in {"source_ttir_outline"}
                else 160
            )
            count = bounded_int(
                "count",
                24 if view == "harness_abi" else 80 if view in {"source_ttir", "source_ttir_outline"} else 32 if view in {"source_obligations", "semantic_transfer"} else 40,
                1,
                count_upper,
            )
            context = bounded_int("context", 3, 0, 16)
            query = bounded_text(
                "query",
                required=view
                in {
                    "source_find", "source_ttir_find", "source_ttir_def_use",
                    "feedback_category", "knowledge",
                },
            )
            path = bounded_text("path")
            section = bounded_text("section")
        except ValueError as exc:
            return self._structured_tool_error(str(exc))

        if view == "source_obligations":
            if section in {"", "overview"}:
                section = None
            elif section is not None and section not in _SOURCE_OBLIGATION_SECTIONS:
                return self._structured_tool_error(
                    "source_obligations section must be one of "
                    + ", ".join(sorted(_SOURCE_OBLIGATION_SECTIONS))
                )
        command = ["python", "/input/tools/context_query.py"]
        if view == "source":
            command += ["source", "--start", str(start), "--lines", str(count)]
        elif view == "source_ttir":
            command += ["source-ttir", "--start", str(start), "--lines", str(count)]
        elif view == "source_ttir_outline":
            command += ["source-ttir-outline", "--start", str(start), "--count", str(count)]
        elif view == "source_ttir_find":
            command += ["source-ttir-find", str(query), "--context", str(context)]
        elif view == "source_ttir_def_use":
            command += ["source-ttir-def-use", str(query), "--context", str(context)]
        elif view == "source_observer":
            command.append("source-observer")
            if path is not None:
                command += ["--path", path]
        elif view == "realization_ledger":
            command.append("realization-ledger")
        elif view == "source_outline":
            command += ["source-outline", "--start", str(start), "--count", str(count)]
        elif view == "source_find":
            command += ["source-find", str(query), "--context", str(context)]
        elif view == "source_brief":
            command.append("source-brief")
        elif view == "source_obligations":
            command.append("source-obligations")
            if section is not None:
                command += [
                    "--section",
                    section,
                    "--start",
                    str(start),
                    "--count",
                    str(count),
                ]
        elif view == "semantic_transfer":
            command += [
                "semantic-transfer",
                "--start",
                str(start),
                "--count",
                str(count),
            ]
        elif view == "feedback_frontier":
            command.append("feedback-frontier")
        elif view in {"harness", "feedback"}:
            command.append(view)
            if path is not None:
                command += ["--path", path]
        elif view == "feedback_category":
            command += ["feedback-category", str(query)]
        elif view == "harness_abi":
            command += [
                "harness-abi",
                "--start",
                str(start),
                "--count",
                str(count),
            ]
        elif view == "harness_c_contract":
            command.append("harness-c-contract")
        else:
            command += ["knowledge", str(query), "--context", str(context)]
        self._structured_context_queries.append(str(view))
        result = self._execute_host_mcp_reader(
            " ".join(shlex.quote(item) for item in command)
        )
        if view == "source_brief" and result.get("returncode") == 0:
            self._semantic_entry_read = True
        return result

    def _read_schema_bundle(
        self, action: Mapping[str, Any]
    ) -> dict[str, Any]:
        # schema_lookup and all context resources share the same mounted-wire
        # receipt validation.  Do this before dispatching schema_query.py so a
        # stale bundle cannot expose a schema leaf while j-framework would
        # reject the same workspace.
        receipt = self._execute_host_mcp_reader(
            "python /input/tools/context_query.py canonical-wire-check"
        )
        if receipt.get("returncode") != 0:
            return self._structured_tool_error(
                "canonical wire bundle validation failed before schema_lookup"
            )
        if set(action) - {"tool_name", "tool_call_id", "items"}:
            return self._structured_tool_error("unknown schema-inspection argument")
        items = action.get("items")
        if (
            not isinstance(items, list)
            or not 1 <= len(items) <= MAX_MCP_SCHEMA_LOOKUP_ITEMS
            or len(items) != len(set(items))
            or any(
                not isinstance(item, str)
                or len(item) > MCP_SCHEMA_LOOKUP_ITEM_CHARS
                or re.fullmatch(r"op:[A-Za-z0-9_]+", item) is None
                for item in items
            )
        ):
            return self._structured_tool_error(
                "schema items must use a bounded distinct semantic operation selector set"
            )
        self._structured_schema_queries.append(list(items))
        command = ["python", "/input/tools/schema_query.py", "bundle", *items]
        return self._execute_host_mcp_reader(
            " ".join(shlex.quote(item) for item in command)
        )

    def _execute_host_mcp_reader(self, command: str) -> dict[str, Any]:
        """Execute one host-selected bounded MCP reader.

        The command is constructed by the dispatcher from validated MCP
        arguments.  The depth flag is deliberately scoped rather than a
        provider-visible capability: ordinary owner shell actions cannot open
        `/input` and therefore cannot bypass the resource lifecycle.
        """

        self._host_mcp_read_depth += 1
        try:
            return self.execute({"command": command})
        finally:
            self._host_mcp_read_depth -= 1

    @staticmethod
    def _parse_semantic_declaration(text: str) -> Any:
        """Parse one MCP declaration without accepting non-JSON wire forms."""

        return json.loads(
            text,
            parse_constant=_reject_non_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )

    @staticmethod
    def _operation_graph_exception_diagnostic(exc: BaseException) -> str | None:
        """Classify safe, provider-correctable graph wire failures.

        JSON decoder wording and internal parser exceptions are deliberately
        not exposed verbatim.  The returned codes describe only the public
        declaration contract, and the caller keeps all other failures behind
        the generic schema-derived receipt.
        """

        if isinstance(exc, json.JSONDecodeError):
            return "operation_graph_must_be_valid_json"
        detail = str(exc)
        if detail.startswith("duplicate JSON key "):
            return "operation_graph_must_not_contain_duplicate_json_keys"
        if detail.startswith("invalid JSON constant:"):
            return "operation_graph_must_use_finite_json_numbers"
        if detail in _PROVIDER_CORRECTABLE_OPERATION_GRAPH_DIAGNOSTICS:
            return detail
        return None

    def _operation_graph_receipt_diagnostic(
        self,
        declaration_text: str | None,
        body: Any,
        detail: str | None = None,
    ) -> dict[str, Any]:
        """Return non-sensitive structure for a rejected graph declaration.

        The stable error code is the semantic authority.  For an attribute
        envelope error, also name the declaration row and its public contract
        field set.  A leaf-value mismatch additionally carries that one
        schema-derived public leaf contract.  This avoids making a provider
        guess which operation in a large Source graph needs correction,
        without returning Source text, candidate bytes, or a canonical handle.
        """

        normalized: bytes
        container = "invalid"
        row_count: int | None = None
        if isinstance(body, list):
            container = "array"
            row_count = len(body)
            try:
                normalized = json.dumps(
                    body,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError, UnicodeError):
                normalized = b"<unserializable-json-array>"
        elif body is not None:
            container = _json_container_kind(body)
            normalized = f"<{container}>".encode("ascii")
        else:
            # Invalid JSON has no normalized value.  A digest of its exact
            # bytes is still safe to compare without writing those bytes to an
            # evidence file or provider receipt.
            normalized = (declaration_text or "").encode("utf-8", "replace")
        result: dict[str, Any] = {
            "json_container": container,
            "declaration_sha256": hashlib.sha256(normalized).hexdigest(),
        }
        if row_count is not None:
            result["operation_rows"] = row_count

        def declaration_rows(value: Any) -> list[tuple[str, Mapping[str, Any]]]:
            """Return structural row paths without accepting the graph."""

            found: list[tuple[str, Mapping[str, Any]]] = []

            def visit_rows(rows: Any, prefix: str) -> None:
                if not isinstance(rows, list):
                    return
                for index, row in enumerate(rows):
                    if not isinstance(row, Mapping):
                        continue
                    path = f"{prefix}[{index}]"
                    found.append((path, row))
                    operation = row.get("op")
                    if operation == "loop":
                        visit_rows(row.get("body"), path + ".body")
                    elif operation in {"effect_if", "value_if"}:
                        for arm_name in ("positive", "negative"):
                            arm = row.get(arm_name)
                            if isinstance(arm, Mapping):
                                visit_rows(
                                    arm.get("body"), path + f".{arm_name}.body"
                                )

            visit_rows(value, "body")
            return found

        structural_rows = declaration_rows(body)
        try:
            namespace = json.loads(self._semantic_operation_namespace)
            operation_catalog = namespace.get("operation_catalog")
            diagnostic_catalog = (
                frozenset(operation_catalog)
                if isinstance(operation_catalog, list)
                and all(isinstance(item, str) for item in operation_catalog)
                else frozenset()
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            diagnostic_catalog = frozenset()
        local_label = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")

        def provider_local_label(value: Any) -> bool:
            return (
                isinstance(value, str)
                and local_label.fullmatch(value) is not None
                and re.fullmatch(r"[nvxpcae][0-9]+", value) is None
            )

        field_details = {
            "operation_row_must_include_canonical_op_field": "op",
            "operation_op_must_be_canonical_j_name": "op",
            "operation_must_use_only_canonical_leaf_fields": "row",
            "operation_must_include_required_leaf_fields": "row",
            "operation_id_must_be_provider_local_label": "id",
            "operation_result_must_be_provider_local_label": "result",
            "operation_operands_must_be_json_array": "operands",
            "operation_operands_must_match_canonical_leaf_arity": "operands",
            "operation_operand_must_reference_source_tensor_or_prior_local_result": "operands",
            "operation_result_must_be_new_provider_local_label": "result",
            "operation_attrs_must_be_json_object": "attrs",
            "operation_result_must_have_complete_value_type": "type",
            "operation_type_shape_must_use_valid_provider_extent": "type.shape",
            "operation_type_shape_must_use_provider_dimension_alias": "type.shape",
            "operation_iota_result_type_must_match_attrs": "type",
            "operation_nontrailing_broadcast_requires_local_id": "id",
        }
        target_field = field_details.get(str(detail))
        if target_field is not None and structural_rows:
            selected_path: str | None = None
            selected_row: Mapping[str, Any] | None = None
            visible_labels = set(self._semantic_source_tensor_handles)
            for path, row in structural_rows:
                operation = row.get("op")
                operands = row.get("operands")
                operation_type = row.get("type")
                raw_result = row.get("result")
                result_labels = (
                    raw_result if isinstance(raw_result, list) else [raw_result]
                )
                bad = (
                    detail == "operation_row_must_include_canonical_op_field"
                    and "op" not in row
                    or detail == "operation_op_must_be_canonical_j_name"
                    and operation not in diagnostic_catalog
                    or detail == "operation_must_use_only_canonical_leaf_fields"
                    and operation not in STRUCTURED_INTENT_OPS
                    and bool(
                        set(row)
                        - {"id", "op", "result", "operands", "attrs", "type"}
                    )
                    or detail == "operation_must_include_required_leaf_fields"
                    and operation not in STRUCTURED_INTENT_OPS
                    and ("attrs" not in row or "type" not in row)
                    or detail == "operation_id_must_be_provider_local_label"
                    and row.get("id") is not None
                    and not provider_local_label(row.get("id"))
                    or detail == "operation_result_must_be_provider_local_label"
                    and raw_result is not None
                    and any(not provider_local_label(value) for value in result_labels)
                    or detail == "operation_operands_must_be_json_array"
                    and not isinstance(operands, list)
                    or detail == "operation_operands_must_match_canonical_leaf_arity"
                    and isinstance(operation, str)
                    and isinstance(operands, list)
                    and not operation_operands_match_arity(
                        operation,
                        operands,
                        row.get("attrs") if isinstance(row.get("attrs"), Mapping) else None,
                    )
                    or detail
                    == "operation_operand_must_reference_source_tensor_or_prior_local_result"
                    and isinstance(operands, list)
                    and any(
                        not isinstance(operand, str) or operand not in visible_labels
                        for operand in operands
                    )
                    or detail == "operation_result_must_be_new_provider_local_label"
                    and raw_result is not None
                    and any(value in visible_labels for value in result_labels)
                    or detail == "operation_attrs_must_be_json_object"
                    and not isinstance(row.get("attrs"), Mapping)
                    or detail == "operation_result_must_have_complete_value_type"
                    and "result" in row
                    and not isinstance(operation_type, Mapping)
                    or detail.startswith("operation_type_shape_")
                    and isinstance(operation_type, Mapping)
                    or detail == "operation_iota_result_type_must_match_attrs"
                    and operation == "iota"
                    or detail == "operation_nontrailing_broadcast_requires_local_id"
                    and operation in {"map_binary", "select"}
                    and not isinstance(row.get("id"), str)
                )
                if bad:
                    selected_path, selected_row = path, row
                    break
                visible_labels.update(
                    value
                    for value in result_labels
                    if isinstance(value, str)
                )
            if selected_row is None:
                selected_path, selected_row = structural_rows[0]
            top_level = re.match(r"body\[([0-9]+)\]", selected_path or "")
            if top_level is not None:
                result["row_index"] = int(top_level.group(1))
            operation = selected_row.get("op")
            if isinstance(operation, str):
                result["operation"] = operation[:128]
            result["field"] = target_field
        if (
            detail
            in {
                "operation_attrs_must_use_canonical_leaf_fields",
                "operation_attrs_must_include_required_leaf_fields",
                "operation_attr_value_must_match_canonical_leaf_contract",
            }
            and isinstance(body, list)
        ):
            for index, row in enumerate(body):
                if not isinstance(row, Mapping) or not isinstance(row.get("op"), str):
                    continue
                try:
                    contract = operation_attr_contract(row["op"])
                except (KeyError, TypeError, ValueError):
                    continue
                allowed = contract.get("allowed")
                required = contract.get("required")
                attrs = row.get("attrs")
                if not isinstance(allowed, frozenset) or not isinstance(
                    required, frozenset
                ):
                    continue
                has_extra = isinstance(attrs, Mapping) and bool(set(attrs) - allowed)
                missing_required = not isinstance(attrs, Mapping) or bool(
                    required - set(attrs)
                )
                properties = contract.get("properties")
                invalid_value_name = (
                    next(
                        (
                            name
                            for name, attr_value in attrs.items()
                            if isinstance(properties, Mapping)
                            and not _operation_attr_value_matches_schema(
                                attr_value,
                                properties.get(name),
                            )
                        ),
                        None,
                    )
                    if isinstance(attrs, Mapping)
                    else None
                )
                if (
                    detail == "operation_attrs_must_use_canonical_leaf_fields"
                    and has_extra
                    or detail
                    == "operation_attrs_must_include_required_leaf_fields"
                    and missing_required
                    or detail == "operation_attr_value_must_match_canonical_leaf_contract"
                    and invalid_value_name is not None
                ):
                    result["row_index"] = index
                    result["operation"] = row["op"]
                    result["field"] = "attrs"
                    result["allowed_fields"] = sorted(allowed)
                    result["required_fields"] = sorted(required)
                    if invalid_value_name is not None:
                        result["attribute"] = invalid_value_name
                        property_schema = properties.get(invalid_value_name)
                        if isinstance(property_schema, Mapping):
                            # ``operation_attr_contract`` is the same
                            # canonical leaf authority already exposed by a
                            # permitted schema lookup and consumed by
                            # TypeCheck.  It contains no Source-specific
                            # value, candidate fragment, or host handle.
                            result["attribute_contract"] = copy.deepcopy(
                                dict(property_schema)
                            )
                    break
        if (
            detail == "operation_result_type_shape_must_match_operands"
            and isinstance(body, list)
        ):
            # This is the same deliberately narrow rank-preservation proof
            # used by graph ingress.  Return only the public row index/op and
            # ranks, never a provider label, operand value, Source fragment,
            # or candidate row, so a provider can correct the one type that
            # triggered the bounded declaration failure.
            value_shapes = {
                value: shape
                for value, (_dtype, shape) in self._semantic_source_tensor_value_types.items()
            }
            rows = semantic_operation_wire_rows({"body": body})
            if rows is not None:
                for path, row in rows:
                    operands = row.get("operands")
                    operation_type = row.get("type")
                    result = row.get("result")
                    shape = (
                        tuple(operation_type.get("shape"))
                        if isinstance(operation_type, Mapping)
                        and isinstance(operation_type.get("shape"), list)
                        else None
                    )
                    if row.get("op") == "map_binary" and isinstance(operands, list):
                        known = [
                            value_shapes[operand]
                            for operand in operands
                            if isinstance(operand, str) and operand in value_shapes
                        ]
                        if shape is not None and len(known) == 2:
                            maximum_rank = max(len(item) for item in known)
                            maximal = {
                                item for item in known if len(item) == maximum_rank
                            }
                            expected = next(iter(maximal)) if len(maximal) == 1 else None
                            if (
                                expected is not None
                                and not all(item == expected or not item for item in known)
                                and shape != expected
                            ):
                                match = re.fullmatch(r"body\\[([0-9]+)\\]", path)
                                if match is not None:
                                    result.update(
                                        {
                                            "row_index": int(match.group(1)),
                                            "operation": row.get("op"),
                                            "field": "type.shape",
                                            "result_rank": len(shape),
                                            "expected_rank": len(expected),
                                        }
                                    )
                                break
                    if isinstance(result, str) and shape is not None:
                        value_shapes[result] = shape
        if (
            detail == "operation_graph_must_expose_each_source_output_type"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_output_type(body)
            if missing is not None:
                output_ordinal, (dtype, shape) = missing
                result.update(
                    {
                        "output_ordinal": output_ordinal,
                        "field": "type",
                        "required_output_type": {
                            "dtype": dtype,
                            "shape": list(shape),
                        },
                        "function_visible_matching_value_count": 0,
                    }
                )
        if (
            detail == "operation_graph_must_cover_source_operation_families"
            and isinstance(body, list)
        ):
            rows = semantic_operation_wire_rows({"body": body})
            if rows is not None:
                counts: dict[str, int] = {}
                for _path, row in rows:
                    operation = row.get("op")
                    if isinstance(operation, str):
                        counts[operation] = counts.get(operation, 0) + 1
                missing_family = next(
                    (
                        (name, required, counts.get(name, 0))
                        for name, required in sorted(
                            self._semantic_entry_required_operation_families.items()
                        )
                        if counts.get(name, 0) < required
                    ),
                    None,
                )
                if missing_family is not None:
                    name, required, actual = missing_family
                    result.update(
                        {
                            "operation": name,
                            "field": "op",
                            "required_operation_count": required,
                            "actual_operation_count": actual,
                        }
                    )
        if (
            detail == "operation_graph_must_cover_source_operation_attributes"
            and isinstance(body, list)
        ):
            rows = semantic_operation_wire_rows({"body": body})
            if rows is not None:
                for operation, attribute, value, required in (
                    self._semantic_entry_required_operation_attributes
                ):
                    actual = sum(
                        1
                        for _path, row in rows
                        if row.get("op") == operation
                        and isinstance(row.get("attrs"), Mapping)
                        and row["attrs"].get(attribute) == value
                    )
                    if actual >= required:
                        continue
                    result.update(
                        {
                            "operation": operation,
                            "field": f"attrs.{attribute}",
                            "attribute": attribute,
                            "required_attribute_value": value,
                            "required_operation_count": required,
                            "actual_operation_count": actual,
                        }
                    )
                    break
        if (
            detail
            == "operation_graph_must_expose_compatible_access_predicate_type"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_access_predicate_type(body)
            if missing is not None:
                access_ordinal, matching_count = missing
                result.update(
                    {
                        "access_ordinal": access_ordinal,
                        "field": "predicate",
                        "function_visible_compatible_boolean_result_count": (
                            matching_count
                        ),
                    }
                )
        if (
            detail == "operation_graph_must_expose_access_consumer_id"
            and isinstance(body, list)
        ):
            missing = self._semantic_graph_missing_access_consumer_id(body)
            if missing is not None:
                access_ordinal, matching_count = missing
                result.update(
                    {
                        "access_ordinal": access_ordinal,
                        "field": "node_ref",
                        "direct_single_result_consumer_id_count": matching_count,
                    }
                )
        return result

    def _record_operation_graph_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        body: Any,
    ) -> None:
        """Bound a failed graph phase without retaining provider bytes."""

        diagnostic = self._operation_graph_receipt_diagnostic(
            declaration_text,
            body,
            detail,
        )
        repeated_unchanged = any(
            row.get("declaration_sha256") == diagnostic["declaration_sha256"]
            for row in self._semantic_operation_graph_failures
        )
        row = {
            "step": "declare_operation_graph",
            "status": "fail",
            "detail": detail,
            **diagnostic,
            "attempt": len(self._semantic_operation_graph_failures) + 1,
        }
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_operation_graph_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="semantic_operation_graph_repeated_unchanged_declaration",
                attempted_step="declare_operation_graph",
            )
        if len(self._semantic_operation_graph_failures) >= (
            _MAX_OPERATION_GRAPH_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_operation_graph_failure_limit_exhausted",
                attempted_step="declare_operation_graph",
            )

    def _record_semantic_access_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        accesses: Any,
        diagnostic: Mapping[str, Any] | None,
    ) -> None:
        """Bound failed complete-access declarations without retaining bytes.

        The begin receipt already contains every Source-order access slot.  A
        failed write may receive a bounded first-row template and diagnostic,
        but it cannot turn into an unbounded stream of empty/prefix arrays.
        Keep only a digest and row-count telemetry; rejected declaration bytes
        must not become a candidate, diagnostic template, or repair base.
        """

        serialized = (
            declaration_text.encode("utf-8")
            if isinstance(declaration_text, str)
            else b""
        )
        declaration_sha256 = hashlib.sha256(serialized).hexdigest()
        repeated_unchanged = any(
            row.get("declaration_sha256") == declaration_sha256
            and row.get("detail") == detail
            for row in self._semantic_access_declaration_failures
        )
        safe_diagnostic: dict[str, Any] = {}
        if isinstance(diagnostic, Mapping):
            # Retain only Source-slot coordinates and field names.  In
            # particular, do not copy an actual provider value from a rejected
            # declaration into the audit/evidence lineage.
            for field in (
                "row_index",
                "expected_row_count",
                "actual_row_count",
                "required_fields",
                "missing_fields",
                "field",
                "provider_local_operation_labels",
            ):
                value = diagnostic.get(field)
                if field in {
                    "required_fields",
                    "missing_fields",
                    "provider_local_operation_labels",
                }:
                    if isinstance(value, list) and all(
                        isinstance(item, str) for item in value
                    ):
                        safe_diagnostic[field] = list(value)
                elif field in {"row_index", "expected_row_count", "actual_row_count"}:
                    if isinstance(value, int) and not isinstance(value, bool):
                        safe_diagnostic[field] = value
                elif isinstance(value, str):
                    safe_diagnostic[field] = value
        row = {
            "step": "declare_accesses",
            "status": "fail",
            "detail": detail,
            "declaration_sha256": declaration_sha256,
            "row_count": len(accesses) if isinstance(accesses, list) else None,
            "attempt": len(self._semantic_access_declaration_failures) + 1,
        }
        if safe_diagnostic:
            row["diagnostic"] = safe_diagnostic
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_access_declaration_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="semantic_access_declaration_repeated_unchanged",
                attempted_step="declare_accesses",
            )
        if len(self._semantic_access_declaration_failures) >= (
            _MAX_ACCESS_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_access_declaration_failure_limit_exhausted",
                attempted_step="declare_accesses",
            )

    def _record_semantic_output_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        mappings: Any,
        diagnostic: Mapping[str, Any] | None,
    ) -> None:
        """Bound failed output selections without retaining provider bytes.

        The row count, failing field, and labels already authored in the same
        provider graph are safe structural receipts.  The rejected mapping
        itself is represented only by a digest and can never become a repair
        base or a host-selected output value.
        """

        serialized = (
            declaration_text.encode("utf-8")
            if isinstance(declaration_text, str)
            else b""
        )
        declaration_sha256 = hashlib.sha256(serialized).hexdigest()
        repeated_unchanged = any(
            row.get("declaration_sha256") == declaration_sha256
            and row.get("detail") == detail
            for row in self._semantic_output_declaration_failures
        )
        safe_diagnostic: dict[str, Any] = {}
        if isinstance(diagnostic, Mapping):
            row_index = diagnostic.get("row_index")
            field = diagnostic.get("field")
            labels = diagnostic.get("provider_local_result_labels")
            if isinstance(row_index, int) and not isinstance(row_index, bool):
                safe_diagnostic["row_index"] = row_index
            if isinstance(field, str):
                safe_diagnostic["field"] = field
            if (
                isinstance(labels, list)
                and len(labels) <= 64
                and all(isinstance(label, str) for label in labels)
            ):
                safe_diagnostic["provider_local_result_labels"] = list(labels)
        row = {
            "step": "declare_output_mappings",
            "status": "fail",
            "detail": detail,
            "declaration_sha256": declaration_sha256,
            "row_count": len(mappings) if isinstance(mappings, list) else None,
            "attempt": len(self._semantic_output_declaration_failures) + 1,
        }
        if safe_diagnostic:
            row["diagnostic"] = safe_diagnostic
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_output_declaration_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="semantic_output_declaration_repeated_unchanged",
                attempted_step="declare_output_mappings",
            )
        if len(self._semantic_output_declaration_failures) >= (
            _MAX_OUTPUT_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_output_declaration_failure_limit_exhausted",
                attempted_step="declare_output_mappings",
            )

    @staticmethod
    def _unique_nontrailing_broadcast_axes(
        source_shape: tuple[str | int, ...],
        result_shape: tuple[str | int, ...],
    ) -> tuple[int, ...] | None:
        """Return a uniquely required rank insertion, never a generated row.

        This invokes the canonical broadcast projection used by J typing.  It
        only recognizes a relation when an unannotated operand cannot align,
        the rank difference fixes the number of inserted axes, and exactly one
        explicit axis set can align that operand with the already declared
        result type.  Ambiguous/value-dependent cases remain Decoder-owned.
        """

        if len(source_shape) >= len(result_shape):
            return None
        # The canonical projection accepts an empty axes list for a lower-rank
        # operand and leaves ordinary trailing NumPy broadcasting to its
        # consumer.  Determine that ordinary case directly before asking for
        # a rank insertion; otherwise [dim0] -> [dim0, dim1] would be
        # mistaken for a trailing broadcast merely because the projection
        # deferred alignment.
        trailing_compatible = all(
            source_extent == result_extent
            or source_extent == 1
            or result_extent == 1
            for source_extent, result_extent in zip(
                reversed(source_shape), reversed(result_shape)
            )
        )
        if trailing_compatible:
            return None
        missing_rank = len(result_shape) - len(source_shape)
        matches: list[tuple[int, ...]] = []
        for axes in combinations(range(len(result_shape)), missing_rank):
            try:
                _projected, inserted = _broadcast_projection(
                    list(source_shape), axes, list(result_shape)
                )
            except (NotImplementedError, ValueError):
                continue
            if tuple(inserted) == axes:
                matches.append(axes)
        return matches[0] if len(matches) == 1 else None

    def _semantic_boundary_broadcast_diagnostic(
        self, fields: Mapping[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        """Require graph-implied non-trailing broadcast rows before finalize.

        The provider remains the sole author of ``boundary.broadcasts``.  This
        check only compares the provider's already declared graph types using
        the canonical projection and returns the exact provider-local relation
        still required by that graph.  It neither creates a row nor changes a
        value/result type, and intentionally leaves ambiguous relations to the
        normal Decoder/CandidateCheck authority.
        """

        buffer = self._semantic_construction_buffer
        j = buffer.get("j") if isinstance(buffer, Mapping) else None
        intent = j.get("intent") if isinstance(j, Mapping) else None
        function = intent.get("intent_function") if isinstance(intent, Mapping) else None
        body = function.get("body") if isinstance(function, Mapping) else None
        rows = semantic_operation_wire_rows({"body": body}) if isinstance(body, list) else None
        if rows is None:
            return None
        value_shapes = {
            value: shape
            for value, (_dtype, shape) in self._semantic_source_tensor_value_types.items()
        }
        required: list[dict[str, Any]] = []
        for _path, operation in rows:
            operands = operation.get("operands")
            operation_type = operation.get("type")
            result = operation.get("result")
            shape = (
                tuple(operation_type.get("shape"))
                if isinstance(operation_type, Mapping)
                and isinstance(operation_type.get("shape"), list)
                else None
            )
            if (
                operation.get("op") in {"map_binary", "select"}
                and isinstance(operands, list)
                and shape is not None
            ):
                node_ref = operation.get("id")
                for operand in operands:
                    source_shape = value_shapes.get(operand)
                    if not isinstance(operand, str) or source_shape is None:
                        continue
                    axes = self._unique_nontrailing_broadcast_axes(
                        source_shape, shape
                    )
                    if axes is None:
                        continue
                    if not isinstance(node_ref, str) or not node_ref:
                        return (
                            "boundary_broadcast_requires_provider_operation_id",
                            {"operation": operation.get("op")},
                        )
                    required.append(
                        {
                            "node_ref": node_ref,
                            "operand": operand,
                            "axes": list(axes),
                        }
                    )
            if isinstance(result, str) and shape is not None:
                value_shapes[result] = shape
        if not required:
            return None
        broadcasts = fields.get("broadcasts", [])
        if not isinstance(broadcasts, list):
            return (
                "boundary_broadcasts_must_be_json_array",
                {"required_broadcasts": required},
            )
        declared = {
            (row.get("node_ref"), row.get("operand")): row.get("axes")
            for row in broadcasts
            if isinstance(row, Mapping)
        }
        missing = [
            row
            for row in required
            if declared.get((row["node_ref"], row["operand"])) != row["axes"]
        ]
        if missing:
            return (
                "boundary_broadcasts_must_cover_nontrailing_operations",
                {
                    "required_broadcasts": missing,
                    "declared_broadcast_count": len(broadcasts),
                },
            )
        return None

    def _record_semantic_boundary_fields_failure(
        self,
        *,
        detail: str,
        declaration_text: str | None,
        fields: Any,
    ) -> None:
        """Bound a repeated final semantic-field failure without retaining it."""

        serialized = (
            declaration_text.encode("utf-8")
            if isinstance(declaration_text, str)
            else b""
        )
        declaration_sha256 = hashlib.sha256(serialized).hexdigest()
        repeated_unchanged = any(
            row.get("declaration_sha256") == declaration_sha256
            and row.get("detail") == detail
            for row in self._semantic_boundary_field_failures
        )
        row = {
            "step": "declare_boundary_fields",
            "status": "fail",
            "detail": detail,
            "declaration_sha256": declaration_sha256,
            "field_names": (
                sorted(fields) if isinstance(fields, Mapping) else None
            ),
            "attempt": len(self._semantic_boundary_field_failures) + 1,
        }
        if repeated_unchanged:
            row["repeated_unchanged"] = True
        self._semantic_boundary_field_failures.append(dict(row))
        self._semantic_construction_audit.append(row)
        if repeated_unchanged:
            self._raise_lift_preparation_failure(
                detail="semantic_boundary_fields_repeated_unchanged",
                attempted_step="declare_boundary_fields",
            )
        if len(self._semantic_boundary_field_failures) >= (
            _MAX_BOUNDARY_FIELDS_DECLARATION_FAILURES
        ):
            self._raise_lift_preparation_failure(
                detail="semantic_boundary_fields_failure_limit_exhausted",
                attempted_step="declare_boundary_fields",
            )

    @staticmethod
    def _semantic_write_receipt(
        *,
        status: str,
        step: str,
        detail: str,
        returncode: int,
        required_slots: Mapping[str, Any] | None = None,
        next_legal_action: str | None = None,
        canonical_row_template: Mapping[str, Any] | None = None,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "step": step,
            "detail": detail,
        }
        if required_slots is not None:
            payload["required_slots"] = copy.deepcopy(dict(required_slots))
        if next_legal_action is not None:
            payload["next_legal_action"] = next_legal_action
        if canonical_row_template is not None:
            payload["canonical_row_template"] = copy.deepcopy(
                dict(canonical_row_template)
            )
        if diagnostic is not None:
            payload["diagnostic"] = copy.deepcopy(dict(diagnostic))
        return {
            "output": json.dumps(payload, sort_keys=True) + "\n",
            "returncode": returncode,
            "exception_info": "",
        }

    def _semantic_write_ready(self, step: str) -> dict[str, Any] | None:
        """Enforce the fresh semantic-entry read boundary for MCP writes."""

        if not self._semantic_entry_read:
            return self._semantic_write_receipt(
                status="fail",
                step=step,
                detail="read_source_brief_before_semantic_construction",
                returncode=1,
            )
        if not self._semantic_entry_transfer_ledger_read:
            return self._semantic_write_receipt(
                status="fail",
                step=step,
                detail="read_source_transfer_ledger_before_semantic_construction",
                returncode=1,
            )
        if step == "declare_accesses" and self._semantic_graph_requires_transfer_reread():
            return self._semantic_write_receipt(
                status="fail",
                step=step,
                detail="reread_source_transfer_ledger_after_operation_graph_before_accesses",
                returncode=1,
                next_legal_action="read_transfer_ledger_after_operation_graph",
            )
        return None

    def _semantic_construction_seed(self) -> dict[str, Any]:
        """Return the sole template available to this fresh MCP owner."""

        return _parse_candidate_object(self._semantic_candidate_skeleton)

    def _mcp_semantic_construction_write(
        self,
        name: str,
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Accumulate one bounded J semantic write and finalize once.

        The provider never writes a complete artifact here.  It may fill only
        source-visible J semantics and normalization classes in an isolated
        buffer. ``finalize_candidate`` hands that one object to the existing
        semantic preflight and canonical construction transaction, which is
        still the only path that creates C, canonical handles, effects and π
        bindings.
        """

        ready = self._semantic_write_ready(name)
        if ready is not None:
            return ready
        sequence_error = self._semantic_write_sequence_error((name,))
        if sequence_error is not None:
            if self._semantic_construction_buffer is not None and (
                sequence_error
                in {
                    "semantic_construction_duplicate_step",
                    "semantic_construction_optional_step_not_permitted",
                    "semantic_construction_is_already_complete",
                }
                or sequence_error.startswith(
                    "semantic_construction_only_next_required_step_permitted_"
                )
            ):
                self._raise_lift_preparation_failure(
                    detail=sequence_error,
                    attempted_step=name,
                )
            next_step = None
            template = None
            marker = "semantic_construction_required_step_missing_"
            if sequence_error.startswith(marker):
                next_step = sequence_error.removeprefix(marker)
                if next_step == "declare_accesses":
                    template = self._semantic_access_failure_template(())
                elif next_step == "declare_output_mappings":
                    template = self._semantic_output_failure_template(())
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail=sequence_error,
                returncode=1,
                next_legal_action=next_step,
                canonical_row_template=template,
            )
        if name == "begin_candidate":
            if self._semantic_construction_buffer is not None:
                return self._semantic_write_receipt(
                    status="fail",
                    step=name,
                    detail="semantic_construction_already_started",
                    returncode=1,
                )
            try:
                self._semantic_construction_buffer = copy.deepcopy(
                    self._semantic_construction_seed()
                )
            except Exception:
                return self._semantic_write_receipt(
                    status="fail",
                    step=name,
                    detail="semantic_construction_seed_unavailable",
                    returncode=1,
                )
            # This nonce is receipt-only state.  It identifies this fresh
            # in-memory transaction without asking the provider to echo it or
            # coupling separate fresh roots which happen to share a kernel.
            self._semantic_transaction_identity = "semantic-entry-" + os.urandom(16).hex()
            self._semantic_construction_audit.append(
                {
                    "step": name,
                    "status": "pass",
                    "mode": "semantic_entry",
                    "transaction": self._semantic_transaction_identity,
                }
            )
            self._semantic_construction_completed_steps.add(name)
            return self._semantic_write_receipt(
                status="pass",
                step=name,
                detail=(
                    "semantic_candidate_buffer_started; complete the listed "
                    "Source-order slots before finalize_candidate"
                ),
                returncode=0,
                required_slots=self._semantic_slot_receipt(),
            )

        buffer = self._semantic_construction_buffer
        if not isinstance(buffer, dict):
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail="begin_candidate_required",
                returncode=1,
            )
        j = buffer.get("j")
        pi = buffer.get("pi")
        if not isinstance(j, dict) or not isinstance(pi, dict):
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail="semantic_candidate_buffer_malformed",
                returncode=1,
            )
        boundary = j.get("boundary")
        intent = j.get("intent")
        function = intent.get("intent_function") if isinstance(intent, dict) else None
        if not isinstance(boundary, dict) or not isinstance(function, dict):
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail="canonical_semantic_template_malformed",
                returncode=1,
            )

        def declaration_text(step: str) -> str:
            row = self._semantic_write_contract["tool_arguments"].get(step)
            if (
                not isinstance(row, Mapping)
                or not isinstance(row.get("field"), str)
                or row["field"] not in action
            ):
                raise ValueError("canonical MCP declaration argument is unavailable")
            return str(action[row["field"]])

        graph_declaration_text: str | None = None
        access_declaration_text: str | None = None
        output_declaration_text: str | None = None
        boundary_declaration_text: str | None = None
        graph_body: Any = None
        try:
            deterministic_repair: str | None = None
            write_diagnostic: Mapping[str, Any] | None = None
            if name == "declare_operation_graph":
                graph_declaration_text = declaration_text(name)
                body = self._parse_semantic_declaration(graph_declaration_text)
                graph_body = body
                if not isinstance(body, list):
                    raise ValueError("operation_graph_must_be_nonempty_json_array")
                if not body:
                    raise ValueError("operation_graph_must_be_nonempty_json_array")
                if any(not isinstance(row, Mapping) for row in body):
                    raise ValueError("operation_graph_rows_must_be_json_objects")
                operation_diagnostic = self._semantic_operation_graph_diagnostic(body)
                if operation_diagnostic is not None:
                    raise ValueError(operation_diagnostic)
                function["body"] = copy.deepcopy(body)
            elif name == "declare_accesses":
                access_declaration_text = declaration_text(name)
                accesses = self._parse_semantic_declaration(access_declaration_text)
                if not isinstance(accesses, list) or any(
                    not isinstance(row, Mapping) for row in accesses
                ):
                    raise ValueError("accesses_must_be_json_array")
                slot_diagnostic = self._semantic_access_slot_diagnostic(accesses)
                if slot_diagnostic is not None:
                    detail, write_diagnostic = slot_diagnostic
                    raise ValueError(detail)
                boundary["accesses"] = copy.deepcopy(accesses)
            elif name == "declare_output_mappings":
                output_declaration_text = declaration_text(name)
                mappings = self._parse_semantic_declaration(output_declaration_text)
                if not isinstance(mappings, list) or any(
                    not isinstance(row, Mapping) for row in mappings
                ):
                    raise ValueError("output_mappings_must_be_json_array")
                mappings, ordinal_repaired = self._normalize_host_owned_output_ordinals(
                    mappings
                )
                if ordinal_repaired:
                    deterministic_repair = "output_ordinal_host_derived"
                slot_diagnostic = self._semantic_output_slot_diagnostic(mappings)
                if slot_diagnostic is not None:
                    write_diagnostic = self._semantic_output_failure_diagnostic(
                        mappings,
                        slot_diagnostic,
                    )
                    raise ValueError(slot_diagnostic)
                boundary["output_mappings"] = copy.deepcopy(mappings)
            elif name == "declare_boundary_fields":
                boundary_declaration_text = declaration_text(name)
                fields = self._parse_semantic_declaration(boundary_declaration_text)
                allowed = set(self._semantic_write_contract["boundary_fields"])
                types = self._semantic_write_contract["boundary_field_types"]
                if not isinstance(fields, Mapping) or not set(fields) <= allowed:
                    raise ValueError("boundary_fields_must_use_canonical_semantic_names")
                for field, value in fields.items():
                    expected = types.get(field)
                    actual = _json_container_kind(value)
                    if isinstance(expected, str) and actual != expected:
                        raise ValueError(f"boundary_field_{field}_has_wrong_container")
                    boundary[field] = copy.deepcopy(value)
                broadcast_diagnostic = self._semantic_boundary_broadcast_diagnostic(fields)
                if broadcast_diagnostic is not None:
                    detail, write_diagnostic = broadcast_diagnostic
                    raise ValueError(detail)
            elif name == "declare_j_fields":
                fields = self._parse_semantic_declaration(declaration_text(name))
                allowed = set(self._semantic_write_contract["j_fields"])
                types = self._semantic_write_contract["j_field_types"]
                if not isinstance(fields, Mapping) or not set(fields) <= allowed:
                    raise ValueError("j_fields_must_use_canonical_semantic_names")
                for field, value in fields.items():
                    expected = types.get(field)
                    actual = _json_container_kind(value)
                    if isinstance(expected, str) and actual != expected:
                        raise ValueError(f"j_field_{field}_has_wrong_container")
                    j[field] = copy.deepcopy(value)
            elif name == "declare_normalization_classes":
                classes = self._parse_semantic_declaration(declaration_text(name))
                if not isinstance(classes, list) or any(
                    not isinstance(row, Mapping) for row in classes
                ):
                    raise ValueError("normalization_classes_must_be_json_array")
                if classes:
                    pi["normalization_classes"] = copy.deepcopy(classes)
                else:
                    pi.pop("normalization_classes", None)
            elif name == "finalize_candidate":
                candidate_text = json.dumps(
                    buffer,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                self._semantic_construction_audit.append(
                    {"step": name, "status": "attempted"}
                )
                self._semantic_construction_completed_steps.add(name)
                return self._finalize_semantic_candidate(candidate_text)
            else:
                raise ValueError("unknown_semantic_construction_step")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # Source-sized slot failures are intentionally actionable at the
            # declaration which caused them.  Keep all other parsing/schema
            # failures compact and generic so this MCP response never becomes
            # a second full validation authority.
            detail = str(exc)
            graph_diagnostic = (
                self._operation_graph_exception_diagnostic(exc)
                if name == "declare_operation_graph"
                else None
            )
            if graph_diagnostic is not None:
                detail = graph_diagnostic
                graph_receipt_diagnostic = self._operation_graph_receipt_diagnostic(
                    graph_declaration_text,
                    graph_body,
                    detail,
                )
                self._record_operation_graph_failure(
                    detail=detail,
                    declaration_text=graph_declaration_text,
                    body=graph_body,
                )
                write_diagnostic = graph_receipt_diagnostic
            elif name == "declare_operation_graph":
                # A graph failure outside the public declaration taxonomy is
                # host state, not a provider repair task.  Do not disguise it
                # as a malformed provider graph and consume more model turns.
                self._raise_lift_preparation_failure(
                    detail="semantic_operation_graph_host_validation_failure",
                    attempted_step="declare_operation_graph",
                )
            elif name == "declare_accesses":
                if detail not in _PROVIDER_CORRECTABLE_ACCESS_DECLARATION_DIAGNOSTICS:
                    self._raise_lift_preparation_failure(
                        detail="semantic_access_declaration_host_validation_failure",
                        attempted_step="declare_accesses",
                    )
                self._record_semantic_access_failure(
                    detail=detail,
                    declaration_text=access_declaration_text,
                    accesses=locals().get("accesses"),
                    diagnostic=(
                        write_diagnostic
                        if isinstance(write_diagnostic, Mapping)
                        else None
                    ),
                )
                if isinstance(write_diagnostic, Mapping):
                    write_diagnostic = {
                        key: value
                        for key, value in write_diagnostic.items()
                        if key
                        in {
                            "row_index",
                            "expected_row_count",
                            "actual_row_count",
                            "required_fields",
                            "missing_fields",
                            "field",
                            "provider_local_operation_labels",
                        }
                    }
            elif name == "declare_output_mappings":
                if (
                    detail
                    not in _PROVIDER_CORRECTABLE_OUTPUT_DECLARATION_DIAGNOSTICS
                ):
                    self._raise_lift_preparation_failure(
                        detail="semantic_output_declaration_host_validation_failure",
                        attempted_step="declare_output_mappings",
                    )
                self._record_semantic_output_failure(
                    detail=detail,
                    declaration_text=output_declaration_text,
                    mappings=locals().get("mappings"),
                    diagnostic=(
                        write_diagnostic
                        if isinstance(write_diagnostic, Mapping)
                        else None
                    ),
                )
                if isinstance(write_diagnostic, Mapping):
                    write_diagnostic = {
                        key: value
                        for key, value in write_diagnostic.items()
                        if key
                        in {
                            "row_index",
                            "field",
                            "provider_local_result_labels",
                        }
                    }
            elif name == "declare_boundary_fields":
                if detail not in {
                    "boundary_fields_must_use_canonical_semantic_names",
                    "boundary_broadcasts_must_be_json_array",
                    "boundary_broadcasts_must_cover_nontrailing_operations",
                    "boundary_broadcast_requires_provider_operation_id",
                } and not detail.startswith("boundary_field_"):
                    self._raise_lift_preparation_failure(
                        detail="semantic_boundary_fields_host_validation_failure",
                        attempted_step="declare_boundary_fields",
                    )
                self._record_semantic_boundary_fields_failure(
                    detail=detail,
                    declaration_text=boundary_declaration_text,
                    fields=locals().get("fields"),
                )
            elif detail not in {
                # These are provider-correctable operation-graph errors.  Do
                # not collapse them to the generic schema message: doing so
                # makes a provider that used a host ``x*`` result label guess
                # at the JSON envelope instead of correcting that label.
                "operation_op_must_be_canonical_j_name",
                "operation_id_must_be_provider_local_label",
                "operation_result_must_be_provider_local_label",
                "operation_operand_must_reference_source_tensor_or_prior_local_result",
                "operation_result_must_be_new_provider_local_label",
                "operation_result_must_have_complete_value_type",
                "operation_type_shape_must_use_provider_dimension_alias",
                "operation_result_type_shape_must_match_operands",
                "operation_graph_lexical_wire_is_malformed",
                "access_declaration_must_cover_each_source_occurrence",
                "access_declaration_rows_must_be_json_objects",
                "access_declaration_must_follow_source_ordinal",
                "access_declaration_is_missing_required_source_slot_field",
                "access_declaration_must_not_use_id_alias",
                "access_declaration_provider_predicate_must_be_local_label",
                "access_declaration_provider_predicate_must_select_compatible_boolean_result",
                "access_declaration_provider_node_ref_must_be_local_label",
                "access_declaration_provider_node_ref_must_select_direct_consumer",
                "access_declaration_must_match_source_wire_template",
                "output_declaration_must_cover_each_source_output",
                "output_declaration_rows_must_be_json_objects",
                "output_mappings_must_be_json_array",
                "output_declaration_is_missing_required_source_slot_field",
                "output_declaration_must_contain_only_value",
                "output_declaration_requires_declared_operation_graph",
                "output_declaration_value_must_reference_source_input_or_local_result",
                "output_declaration_value_must_be_function_visible",
                "output_declaration_value_type_must_match_source_output",
            }:
                detail = "invalid_schema_derived_semantic_declaration"
            access_template = (
                self._semantic_access_failure_template(
                    accesses if isinstance(locals().get("accesses"), list) else ()
                )
                if name == "declare_accesses"
                else None
            )
            output_template = (
                self._semantic_output_failure_template(
                    mappings if isinstance(locals().get("mappings"), list) else ()
                )
                if name == "declare_output_mappings"
                else None
            )
            access_slots = (
                {
                    "accesses": [
                        dict(row)
                        for row in self._semantic_construction_slots["accesses"]
                    ],
                    "rule": (
                        "resubmit one complete Source-ordinal access array; "
                        "a prefix or partial prior declaration is not a base"
                    ),
                }
                if name == "declare_accesses"
                else None
            )
            return self._semantic_write_receipt(
                status="fail",
                step=name,
                detail=detail,
                returncode=1,
                required_slots=access_slots,
                next_legal_action=name,
                canonical_row_template=access_template or output_template,
                diagnostic=locals().get("write_diagnostic"),
            )
        audit_row: dict[str, Any] = {"step": name, "status": "pass"}
        if deterministic_repair is not None:
            audit_row["deterministic_repair"] = deterministic_repair
        if name == "declare_operation_graph":
            self._semantic_operation_graph_transfer_generation = (
                self._semantic_transfer_read_generation
            )
            self._semantic_access_transfer_generation = None
            audit_row["transfer_read_generation"] = (
                self._semantic_operation_graph_transfer_generation
            )
        if self._semantic_transaction_identity is not None:
            audit_row["transaction"] = self._semantic_transaction_identity
        self._semantic_construction_audit.append(audit_row)
        self._semantic_construction_completed_steps.add(name)
        next_step = self._semantic_next_legal_action()
        return self._semantic_write_receipt(
            status="pass",
            step=name,
            detail=(
                "semantic_required_slots_complete_finalize_now"
                if next_step == "finalize_candidate"
                else "semantic_declaration_recorded"
            ),
            returncode=0,
            next_legal_action=next_step,
        )

    def _finalize_semantic_candidate(self, candidate_text: str) -> dict[str, Any]:
        """Finalize the MCP-owned semantic buffer through the sole ingress."""

        def reject(result: Mapping[str, Any]) -> None:
            raise CandidateRejected(
                {
                    "role": "exit",
                    "content": "CandidateRejected",
                    "extra": {
                        "exit_status": "CandidateRejected",
                        "submission": "",
                        "diagnostics": dict(result),
                    },
                }
            )

        if self._candidate_submission_attempted:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": (
                    "one complete semantic candidate submission is permitted "
                    "per logical Lift; use the next fresh relift with its "
                    "structured guidance"
                ),
            }
            reject(result)
        self._candidate_submission_attempted = True
        if not isinstance(candidate_text, str) or not candidate_text:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": "finalize requires a non-empty semantic construction buffer",
            }
            reject(result)
        candidate_bytes = candidate_text.encode("utf-8")
        if not 1 <= len(candidate_bytes) <= MAX_CANDIDATE_BYTES:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": "candidate_text size is outside the allowed range",
            }
            reject(result)
        try:
            _parse_candidate_object(candidate_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            result = {
                "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                "status": "fail",
                "diagnostic": (
                    "semantic construction buffer must serialize to one JSON candidate; "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
            reject(result)
        if self._semantic_candidate_checker is not None:
            try:
                surface = self._semantic_candidate_checker(candidate_text)
            except Exception as exc:
                diagnostics = _exception_candidate_check_diagnostics(exc)
                surface = {
                    "protocol": CANDIDATE_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": diagnostics
                    if diagnostics is not None
                    else [
                        {
                            "rule_code": "semantic_surface_check_failed",
                            "path": "<candidate>",
                            "observed_refs_or_types": [type(exc).__name__],
                        }
                    ],
                }
            if (
                not isinstance(surface, Mapping)
                or set(surface) != {"protocol", "status", "diagnostics"}
                or surface.get("protocol") != CANDIDATE_CHECK_PROTOCOL
                or surface.get("status") not in {"pass", "fail"}
                or not isinstance(surface.get("diagnostics"), list)
            ):
                raise RuntimeError("semantic candidate surface checker returned invalid result")
            if surface["status"] == "fail":
                # Semantic-surface admission happens before the candidate is
                # written and before host construction.  It is nevertheless
                # the only actionable reason for this fresh relift to end, so
                # retain the same bounded diagnostic shape as CandidateCheck.
                # Without this receipt the lifecycle incorrectly reduces a
                # precise semantic-wire rejection to "worker terminated
                # without a candidate" and the next fresh agent cannot fix
                # the actual field/handle violation.
                diagnostics = [
                    dict(item)
                    for item in surface["diagnostics"]
                    if isinstance(item, Mapping)
                ]
                self._candidate_check_audit.append(
                    {
                        "protocol": CANDIDATE_CHECK_PROTOCOL,
                        "status": "fail",
                        "trigger": "semantic_preflight",
                        "evaluated": True,
                        "budget_pool": "semantic_preflight",
                        "rule_codes": [
                            item.get("rule_code")
                            for item in diagnostics
                            if isinstance(item.get("rule_code"), str)
                        ],
                        "rule_paths": [
                            {
                                "rule_code": item["rule_code"],
                                "path": item["path"],
                            }
                            for item in diagnostics
                            if isinstance(item.get("rule_code"), str)
                            and isinstance(item.get("path"), str)
                        ],
                        "rule_diagnostics": diagnostics,
                    }
                )
                result = {
                    "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                    "status": "fail",
                    "diagnostics": surface["diagnostics"],
                    "authority": (
                        "schema-derived semantic surface preflight; no host "
                        "construction, repair, or acceptance occurred"
                    ),
                }
                reject(result)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".candidate-direct-submit-",
            dir=self.config.output_dir,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output_file:
                output_file.write(candidate_bytes)
                output_file.flush()
                os.fsync(output_file.fileno())
            os.replace(temporary, self.config.output_dir / "candidate.json")
        finally:
            temporary.unlink(missing_ok=True)
        self._candidate_check_passed_bytes = None
        receipt = {
            "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
            "status": "written",
            "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "candidate_bytes": len(candidate_bytes),
            "authority": (
                "MCP-assembled semantic J bytes; finalize performs the required "
                "single-object JSON wire check before existing fail-closed "
                "preflight and canonical construction"
            ),
        }
        self._candidate_finalize_audit.append(dict(receipt))
        output = {
            "output": (
                INTENTIR_LIFT_SUBMIT
                + "\n"
                + json.dumps(receipt, sort_keys=True)
                + "\n"
            ),
            "returncode": 0,
            "exception_info": "",
        }
        self._check_submission(output)
        # _check_submission mutates the output with the bounded CandidateCheck
        # receipt before rejecting it.  End this logical Lift immediately so a
        # model cannot spend additional turns rereading the same J and trying
        # a second submission after the single-submit boundary.
        if output.get("returncode") != 0:
            reject(
                {
                    "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                    "status": "fail",
                    "diagnostic": "candidate submission failed host admission",
                    "output": output.get("output", ""),
                }
            )
        return output

    def stagnation_state(self) -> dict[str, Any]:
        """Return only the exact current candidate byte identity."""

        path = self.config.output_dir / "candidate.json"
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("candidate absent")
            raw = path.read_bytes()
        except (OSError, ValueError):
            return {"protocol": "intentir.lift-stagnation-state.v1", "candidate": None}
        return {
            "protocol": "intentir.lift-stagnation-state.v1",
            "candidate": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
        }

    def material_progress_state(self) -> dict[str, Any]:
        """Return exact identities of model-writable artifact and work bytes."""

        files: list[dict[str, Any]] = []
        for label, root in (
            ("output", self.config.output_dir),
            ("work", self.config.scratch_dir),
        ):
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    files.append(
                        {"root": label, "path": relative, "kind": "symlink"}
                    )
                    continue
                if not path.is_file():
                    continue
                raw = path.read_bytes()
                files.append(
                    {
                        "root": label,
                        "path": relative,
                        "kind": "file",
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
        return {
            "protocol": "intentir.lift-material-progress.v1",
            "files": files,
        }

    def _execute_candidate_check(
        self,
        candidate_path_name: str,
        *,
        trigger: str = "explicit",
    ) -> dict[str, Any]:
        if trigger not in {"explicit", "submission"}:
            raise ValueError("unsupported CandidateCheck trigger")
        self._candidate_check_calls += 1
        self._candidate_check_calls_by_trigger[trigger] += 1
        candidate_bytes: bytes | None = None
        budget_pool: str | None = None
        checker_evaluated = False
        if candidate_path_name not in {
            "/work/candidate.json", "/output/candidate.json",
        }:
            result: Mapping[str, Any] = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "invalid_candidate_path",
                        "path": "<candidate>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
        elif self._candidate_checker is None:
            result = {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "fail",
                "diagnostics": [
                    {
                        "rule_code": "candidate_check_unavailable",
                        "path": "<candidate>",
                        "observed_refs_or_types": [],
                    }
                ],
            }
        else:
            candidate_path = (
                self.config.scratch_dir / "candidate.json"
                if candidate_path_name.startswith("/work/")
                else self.config.output_dir / "candidate.json"
            )
            try:
                if candidate_path.is_symlink() or not candidate_path.is_file():
                    raise ValueError("candidate must be a regular file")
                size = candidate_path.stat().st_size
                if not 1 <= size <= MAX_CANDIDATE_BYTES:
                    raise ValueError("candidate size is outside the allowed range")
                candidate_bytes = candidate_path.read_bytes()
                if len(candidate_bytes) != size:
                    raise ValueError("candidate changed while being read")
                candidate_text = candidate_bytes.decode("utf-8")
            except Exception as exc:
                result = {
                    "protocol": CANDIDATE_CHECK_PROTOCOL,
                    "status": "fail",
                    "diagnostics": [
                        {
                            "rule_code": "candidate_check_failed",
                            "path": "<candidate>",
                            "observed_refs_or_types": [type(exc).__name__],
                        }
                    ],
                }
            else:
                if (
                    self.config.candidate_check_limit == 0
                    or self._candidate_check_shared_evaluations
                    < self.config.candidate_check_limit
                ):
                    self._candidate_check_shared_evaluations += 1
                    budget_pool = "shared"
                elif (
                    trigger == "submission"
                    and self._candidate_check_submission_reserved_evaluations
                    < self.config.candidate_submission_check_limit
                ):
                    self._candidate_check_submission_reserved_evaluations += 1
                    budget_pool = "submission_reserve"
                if budget_pool is None:
                    result = {
                        "protocol": CANDIDATE_CHECK_PROTOCOL,
                        "status": "fail",
                        "diagnostics": [
                            {
                                "rule_code": "check_budget_exhausted",
                                "path": "<candidate>",
                                "observed_refs_or_types": [],
                            }
                        ],
                    }
                else:
                    checker_evaluated = True
                    self._candidate_check_evaluations += 1
                    self._candidate_check_evaluations_by_trigger[trigger] += 1
                    try:
                        if self._candidate_constructor is not None:
                            constructed_text, construction_receipts = self._candidate_constructor(
                                candidate_text
                            )
                            if not isinstance(constructed_text, str) or not constructed_text:
                                raise TypeError(
                                    "canonical candidate constructor must return non-empty text"
                                )
                            if not isinstance(construction_receipts, tuple) or any(
                                not isinstance(item, Mapping) for item in construction_receipts
                            ):
                                raise TypeError(
                                    "canonical constructor receipts must be a tuple of mappings"
                                )
                            constructed_bytes = constructed_text.encode("utf-8")
                            if not 1 <= len(constructed_bytes) <= MAX_CANDIDATE_BYTES:
                                raise ValueError(
                                    "constructed candidate size is outside the allowed range"
                                )
                            before_sha256 = hashlib.sha256(
                                candidate_bytes
                            ).hexdigest()
                            after_sha256 = hashlib.sha256(constructed_bytes).hexdigest()
                            changed = constructed_bytes != candidate_bytes
                            if changed:
                                descriptor, temporary_name = tempfile.mkstemp(
                                    prefix=".candidate-construction-",
                                    dir=candidate_path.parent,
                                )
                                temporary = Path(temporary_name)
                                try:
                                    with os.fdopen(descriptor, "wb") as output:
                                        output.write(constructed_bytes)
                                        output.flush()
                                        os.fsync(output.fileno())
                                    os.replace(temporary, candidate_path)
                                finally:
                                    temporary.unlink(missing_ok=True)
                                if candidate_path.read_bytes() != constructed_bytes:
                                    raise RuntimeError(
                                        "candidate changed while canonical construction "
                                        "was being committed"
                                    )
                                candidate_bytes = constructed_bytes
                                candidate_text = constructed_text
                            self._candidate_constructor_audit.append(
                                {
                                    "status": "pass",
                                    "trigger": trigger,
                                    "changed": changed,
                                    "before_sha256": before_sha256,
                                    "after_sha256": after_sha256,
                                    "construction_receipt_count": len(construction_receipts),
                                    "rule_codes": [
                                        item.get("rule_code")
                                        for item in construction_receipts
                                        if isinstance(item.get("rule_code"), str)
                                    ],
                                }
                            )
                        result = self._candidate_checker(candidate_text)
                    except Exception as exc:
                        diagnostics = _exception_candidate_check_diagnostics(exc)
                        result = {
                            "protocol": CANDIDATE_CHECK_PROTOCOL,
                            "status": "fail",
                            "diagnostics": diagnostics
                            if diagnostics is not None
                            else [
                                {
                                    "rule_code": "candidate_check_failed",
                                    "path": "<candidate>",
                                    "observed_refs_or_types": [
                                        type(exc).__name__
                                    ],
                                }
                            ],
                        }
        protocol = result.get("protocol")
        status = result.get("status")
        diagnostics = result.get("diagnostics")
        if (
            set(result) != {"protocol", "status", "diagnostics"}
            or protocol != CANDIDATE_CHECK_PROTOCOL
            or status not in {"pass", "fail"}
            or not isinstance(diagnostics, list)
            or len(diagnostics) > 20
            or (status == "pass" and bool(diagnostics))
            or (status == "fail" and not diagnostics)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {
                    "rule_code",
                    "path",
                    "observed_refs_or_types",
                }
                or not isinstance(item.get("rule_code"), str)
                or not isinstance(item.get("path"), str)
                or not isinstance(item.get("observed_refs_or_types"), list)
                or len(item.get("rule_code", "")) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                or len(item.get("path", "")) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                or len(item.get("observed_refs_or_types", []))
                > MAX_CANDIDATE_CHECK_OBSERVED_ITEMS
                or any(
                    not isinstance(observed, str)
                    or len(observed) > MAX_CANDIDATE_CHECK_FIELD_CHARS
                    for observed in item.get("observed_refs_or_types", [])
                )
                for item in diagnostics
            )
            or len(
                json.dumps(result, sort_keys=True, ensure_ascii=False).encode(
                    "utf-8"
                )
            )
            > MAX_CANDIDATE_CHECK_RESULT_BYTES
        ):
            raise RuntimeError("candidate checker returned an invalid closed result")
        self._candidate_check_audit.append(
            {
                "protocol": protocol,
                "status": status,
                "trigger": trigger,
                "budget_pool": budget_pool,
                "evaluated": checker_evaluated,
                "rule_codes": [
                    item.get("rule_code")
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                ],
                "rule_paths": [
                    {
                        "rule_code": item.get("rule_code"),
                        "path": item.get("path"),
                    }
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                    and isinstance(item.get("path"), str)
                ],
                "rule_diagnostics": [
                    {
                        "rule_code": item.get("rule_code"),
                        "path": item.get("path"),
                        "observed_refs_or_types": list(
                            item.get("observed_refs_or_types", [])
                        ),
                    }
                    for item in diagnostics
                    if isinstance(item, Mapping)
                    and isinstance(item.get("rule_code"), str)
                    and isinstance(item.get("path"), str)
                    and isinstance(item.get("observed_refs_or_types"), list)
                ],
            }
        )
        if status == "pass" and checker_evaluated and candidate_bytes is not None:
            self._candidate_check_passed_bytes = candidate_bytes
        elif (
            checker_evaluated
            and candidate_bytes == self._candidate_check_passed_bytes
        ):
            self._candidate_check_passed_bytes = None
        return {
            "output": json.dumps(result, sort_keys=True) + "\n",
            "returncode": 0 if status == "pass" else 1,
            "exception_info": "",
        }

    def _validate_paths(self) -> None:
        if self.config.tmp_dir is None:
            self.config.tmp_dir = self.config.scratch_dir.parent / "tmp"
        paths = (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        )
        resolved = tuple(path.resolve(strict=True) for path in paths)
        if len(set(resolved)) != len(resolved):
            raise ValueError("Lift input, output, and work directories must differ")
        if any(not path.is_dir() or path.is_symlink() for path in paths):
            raise ValueError("Lift sandbox paths must be real directories")
        (
            self.config.input_dir,
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ) = resolved

    def _base_command(self) -> list[str]:
        command = [
            self.config.executable,
            "--unshare-all",
            "--unshare-user-try",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
        ]
        for system_path in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            command.extend(["--ro-bind-try", system_path, system_path])
        for package_root in site.getsitepackages():
            intentir_package = (Path(package_root) / "intentir").resolve()
            if intentir_package.is_dir() and str(intentir_package).startswith(
                ("/usr/", "/lib/", "/lib64/")
            ):
                command.extend(
                    [
                        "--tmpfs",
                        str(intentir_package),
                        "--remount-ro",
                        str(intentir_package),
                    ]
                )
        command.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--ro-bind",
                str(self.config.input_dir),
                "/input",
                "--bind",
                str(self.config.output_dir),
                "/output",
                "--bind",
                str(self.config.scratch_dir),
                "/work",
                "--bind",
                str(self.config.tmp_dir),
                "/tmp",
                "--chdir",
                "/work",
                "--remount-ro",
                "/dev",
                "--remount-ro",
                "/",
            ]
        )
        for key, value in self.config.env.items():
            command.extend(["--setenv", key, value])
        return command

    def _probe(self) -> None:
        executable = shutil.which(self.config.executable)
        if executable is None:
            raise RuntimeError("IntentIR Lift requires bubblewrap (bwrap)")
        command = self._base_command() + [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            (
                "python -c 'import importlib.util,sys; "
                "sys.exit(importlib.util.find_spec(\"intentir\") is not None)'"
            ),
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except Exception as exc:
            raise RuntimeError(f"bubblewrap capability probe failed: {exc}") from exc
        if result.returncode != 0:
            diagnostic = result.stdout.strip()[:500]
            raise RuntimeError(
                "bubblewrap isolation probe failed"
                + (f": {diagnostic}" if diagnostic else "")
            )

    def _writable_usage(self) -> tuple[int, int, int]:
        entries = 0
        size = 0
        max_depth = 0
        stack = [
            (self.config.output_dir, 0),
            (self.config.scratch_dir, 0),
            (self.config.tmp_dir, 0),
        ]
        while stack:
            directory, depth = stack.pop()
            max_depth = max(max_depth, depth)
            if max_depth > MAX_WORKSPACE_DEPTH:
                return entries, size, max_depth
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries += 1
                    if entries > MAX_WORKSPACE_ENTRIES:
                        return entries, size, max_depth
                    item_stat = entry.stat(follow_symlinks=False)
                    size += max(item_stat.st_size, item_stat.st_blocks * 512)
                    if size > MAX_WORKSPACE_BYTES:
                        return entries, size, max_depth
                    if entry.is_dir(follow_symlinks=False):
                        stack.append((Path(entry.path), depth + 1))
        return entries, size, max_depth

    def _reset_writable_dirs(self) -> None:
        for directory in (
            self.config.output_dir,
            self.config.scratch_dir,
            self.config.tmp_dir,
        ):
            os.chmod(directory, stat.S_IRWXU)
            for child in directory.iterdir():
                _remove_path_iterative(child)

    def reset_stage_execution_metrics(self) -> None:
        """Start a fresh per-agent-stage action timer without changing state."""

        self._stage_action_calls = 0
        self._stage_action_failures = 0
        self._stage_action_elapsed_ms = 0.0

    @property
    def stage_execution_metrics(self) -> dict[str, Any]:
        return {
            "elapsed_ms": self._stage_action_elapsed_ms,
            "count": self._stage_action_calls,
            "failures": self._stage_action_failures,
        }

    def execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Time one agent action while preserving the closed executor path."""

        started = time.monotonic()
        self._stage_action_calls += 1
        try:
            return self._execute(action, cwd, timeout=timeout)
        except Exception:
            self._stage_action_failures += 1
            raise
        finally:
            self._stage_action_elapsed_ms += (
                time.monotonic() - started
            ) * 1000.0

    def _execute(
        self,
        action: dict,
        cwd: str = "",
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if cwd not in {"", "/work"}:
            raise ValueError("IntentIR Lift cwd is fixed at /work")
        command = action.get("command", "")
        if action.get("tool_name") == "mcp_call_tool":
            return self._execute_mcp(
                {
                    "method": action.get("method"),
                    "params": action.get("params"),
                }
            )
        if not isinstance(command, str):
            raise TypeError("Lift shell command must be a string")
        if "/input" in command and self._host_mcp_read_depth == 0:
            raise ValueError(
                "Lift owner input is available only through bounded MCP resources"
            )
        bounded_shell = (
            f"printf '{_SANDBOX_START_MARKER.decode()}'; "
            # Keep non-time resource protections, but do not impose a hidden
            # CPU-time deadline on an otherwise unbounded Lift action.
            "ulimit -f 8192 -n 64 -u 128 -v 524288; "
            "set -o pipefail; "
            f"/bin/bash --noprofile --norc -c \"$1\" 2>&1 | "
            f"/usr/bin/head -c {MAX_ACTION_OUTPUT_BYTES}"
        )
        invocation = self._base_command() + [
            "/bin/bash",
            "--noprofile",
            "--norc",
            "-c",
            bounded_shell,
            "intentir-lift-action",
            command,
        ]
        action_timeout = timeout if timeout is not None else self.config.timeout
        process: subprocess.Popen[bytes] | None = None
        limit_error = ""
        try:
            with tempfile.TemporaryFile() as output_file:
                process = subprocess.Popen(
                    invocation,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                )
                started = time.monotonic()
                while process.poll() is None:
                    if (
                        action_timeout is not None
                        and time.monotonic() - started >= action_timeout
                    ):
                        limit_error = (
                            f"action exceeded {action_timeout} second timeout"
                        )
                        process.kill()
                        break
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = (
                            "workspace became unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        process.kill()
                        break
                    if (
                        entries > MAX_WORKSPACE_ENTRIES
                        or size > MAX_WORKSPACE_BYTES
                        or depth > MAX_WORKSPACE_DEPTH
                    ):
                        limit_error = (
                            "workspace exceeded limit: "
                            f"entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                            f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                            f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                        )
                        process.kill()
                        break
                    time.sleep(0.05)
                returncode = process.wait(timeout=5)
                if not limit_error:
                    try:
                        entries, size, depth = self._writable_usage()
                    except Exception as exc:
                        limit_error = (
                            "workspace became unreadable: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    else:
                        if (
                            entries > MAX_WORKSPACE_ENTRIES
                            or size > MAX_WORKSPACE_BYTES
                            or depth > MAX_WORKSPACE_DEPTH
                        ):
                            limit_error = (
                                "workspace exceeded limit: "
                                f"entries={entries}/{MAX_WORKSPACE_ENTRIES}, "
                                f"bytes={size}/{MAX_WORKSPACE_BYTES}, "
                                f"depth={depth}/{MAX_WORKSPACE_DEPTH}"
                            )
                output_file.seek(0)
                raw_output = output_file.read(
                    len(_SANDBOX_START_MARKER) + MAX_ACTION_OUTPUT_BYTES + 1
                )
            marker_at = raw_output.find(_SANDBOX_START_MARKER)
            if marker_at < 0:
                diagnostic = raw_output[:500].decode("utf-8", errors="replace")
                raise RuntimeError(
                    "bubblewrap action failed before sandbox start"
                    + (f": {diagnostic.strip()}" if diagnostic.strip() else "")
                )
            action_output = raw_output[marker_at + len(_SANDBOX_START_MARKER) :]
            output_text = action_output[:MAX_ACTION_OUTPUT_BYTES].decode(
                "utf-8", errors="replace"
            )
            output: dict[str, Any] = {
                "output": output_text,
                "returncode": returncode,
                "exception_info": limit_error,
            }
            if len(action_output) >= MAX_ACTION_OUTPUT_BYTES:
                output["returncode"] = returncode or 2
                output["exception_info"] = (
                    f"action output reached {MAX_ACTION_OUTPUT_BYTES} byte limit"
                )
            if limit_error:
                output["returncode"] = -1
                self._reset_writable_dirs()
        except Exception as exc:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            raise RuntimeError(
                f"IntentIR Lift sandbox action failed: {type(exc).__name__}: {exc}"
            ) from exc
        return output

    def _check_submission(self, output: dict[str, Any]) -> None:
        """Commit only the internally emitted MCP finalize marker.

        Ordinary shell output no longer calls this method.  The marker is an
        implementation detail of ``finalize_candidate`` rather than an
        alternate agent submission protocol, so a model cannot write an
        arbitrary ``candidate.json`` and enter construction by printing it.
        """

        lines = str(output.get("output", "")).lstrip().splitlines()
        marker = INTENTIR_LIFT_SUBMIT
        if (
            not lines
            or lines[0].strip() != marker
            or output.get("returncode") != 0
        ):
            return
        if not self._semantic_entry_read:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected: read the semantic authoring entry through "
                "intentir://lift/source/brief before submitting the first candidate.\n"
            )
            return
        workspace = IntentIRLiftWorkspace(
            root=self.config.input_dir.parent,
            input_dir=self.config.input_dir,
            output_dir=self.config.output_dir,
            scratch_dir=self.config.scratch_dir,
            tmp_dir=self.config.tmp_dir,
        )
        try:
            candidate_text = workspace.read_candidate()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected by the local syntax check: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return
        candidate_bytes = candidate_text.encode("utf-8")
        if self._candidate_checker is not None:
            if candidate_bytes != self._candidate_check_passed_bytes:
                preflight = self._execute_candidate_check(
                    "/output/candidate.json", trigger="submission"
                )
                if preflight["returncode"] != 0:
                    output["returncode"] = 2
                    output["output"] += (
                        "\nSubmission preflight rejected the current candidate. "
                        "Do not resubmit in this logical Lift; the next fresh "
                        "relift receives the bounded structured diagnostic.\n"
                        + str(preflight["output"])
                    )
                    return
        try:
            final_candidate_text = workspace.read_candidate()
        except Exception as exc:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected while confirming the checked bytes: "
                f"{type(exc).__name__}: {exc}\n"
            )
            return
        checked_bytes = self._candidate_check_passed_bytes
        if checked_bytes is None or final_candidate_text.encode("utf-8") != checked_bytes:
            output["returncode"] = 2
            output["output"] += (
                "\nSubmission rejected because canonical candidate bytes changed "
                "after CandidateCheck. Start a new logical Lift rather than "
                "resubmitting this workspace.\n"
            )
            return
        raise Submitted(
            {
                "role": "exit",
                "content": "candidate.json submitted",
                "extra": {
                    "exit_status": "Submitted",
                    "submission": "candidate.json",
                },
            }
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "system": "Linux",
            "startup_context": self._startup_context,
            "semantic_candidate_skeleton": self._semantic_candidate_skeleton,
            "semantic_operation_namespace": self._semantic_operation_namespace,
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {
            "info": {
                "config": {
                    "environment": {
                        "sandbox": "intentir-bubblewrap-v1",
                        "network": "unshared",
                        "input": "read-only",
                        "output": "candidate-only",
                        "command_timeout_seconds": self.config.timeout,
                        "candidate_check_shared_limit": (
                            self.config.candidate_check_limit
                        ),
                        "candidate_check_submission_reserve": (
                            self.config.candidate_submission_check_limit
                        ),
                    },
                    "environment_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                },
                "startup_context": dict(self._startup_context_receipt),
                "authoring": {
                    "mode": "semantic_entry",
                    "resource": "intentir://lift/source/brief",
                    "semantic_entry_required": True,
                    "semantic_entry_read": self._semantic_entry_read,
                    "semantic_entry_source_read": self._semantic_entry_source_read,
                    "semantic_entry_transfer_ledger_read": (
                        self._semantic_entry_transfer_ledger_read
                    ),
                    "semantic_transfer_read_generation": (
                        self._semantic_transfer_read_generation
                    ),
                    "operation_graph_transfer_generation": (
                        self._semantic_operation_graph_transfer_generation
                    ),
                    "access_phase_transfer_generation": (
                        self._semantic_access_transfer_generation
                    ),
                    "access_phase_transfer_reread_required": (
                        self._semantic_graph_requires_transfer_reread()
                    ),
                    "semantic_entry_supplemental_read": (
                        self._semantic_entry_supplemental_read
                    ),
                    "semantic_entry_schema_lookup_calls": (
                        self._semantic_entry_schema_lookup_calls
                    ),
                    "semantic_preconstruction_phase_failures": (
                        self._semantic_preconstruction_phase_failures
                    ),
                    "mcp_phase_application_failures": list(
                        self._mcp_phase_application_failures
                    ),
                    "preparation_failures": list(
                        self._preparation_failure_receipts
                    ),
                },
                "candidate_check": {
                    "calls": self._candidate_check_calls,
                    "attempts": self._candidate_check_calls,
                    "evaluations": self._candidate_check_evaluations,
                    "limit": (
                        0
                        if self.config.candidate_check_limit == 0
                        else self.config.candidate_check_limit
                        + self.config.candidate_submission_check_limit
                    ),
                    "time_governed": self.config.candidate_check_limit == 0,
                    "explicit": {
                        "calls": self._candidate_check_calls_by_trigger[
                            "explicit"
                        ],
                        "evaluations": (
                            self._candidate_check_evaluations_by_trigger[
                                "explicit"
                            ]
                        ),
                    },
                    "submission": {
                        "calls": self._candidate_check_calls_by_trigger[
                            "submission"
                        ],
                        "evaluations": (
                            self._candidate_check_evaluations_by_trigger[
                                "submission"
                            ]
                        ),
                    },
                    "shared_budget": {
                        "used": self._candidate_check_shared_evaluations,
                        "limit": self.config.candidate_check_limit,
                    },
                    "submission_reserve": {
                        "used": (
                            self._candidate_check_submission_reserved_evaluations
                        ),
                        "limit": self.config.candidate_submission_check_limit,
                    },
                    "results": list(self._candidate_check_audit),
                },
                "canonical_constructor": {
                    "authority": (
                        "one bounded deterministic Source/Harness construction before "
                        "CandidateCheck; construction telemetry is not part of its result"
                    ),
                    "calls": len(self._candidate_constructor_audit),
                    "results": list(self._candidate_constructor_audit),
                },
                "mcp_semantic_construction": {
                    "protocol": MCP_SEMANTIC_FINALIZE_PROTOCOL,
                    "authority": (
                        "MCP fills only schema-derived J semantics and optional "
                        "normalization classes; finalize is the sole path to "
                        "canonical Source/Harness construction"
                    ),
                    "steps": list(self._semantic_construction_audit),
                    "finalizations": list(self._candidate_finalize_audit),
                },
                "multi_action_policy": {
                    "protocol": MULTI_ACTION_PROTOCOL,
                    "max_actions": 4,
                    "authority": "whole-response fail-before-execute",
                    "allowed": (
                        "independent read-only context/schema calls or one "
                        "ordered MCP semantic construction batch"
                    ),
                    "rejections": self._multi_action_rejections,
                },
                "structured_inspection": {
                    "protocol": "intentir.agent-structured-inspection.v1",
                    "authority": (
                        "bounded exact read-only views; no semantic classification, "
                        "candidate edit, or acceptance authority"
                    ),
                    "context_views": list(self._structured_context_queries),
                    "schema_bundles": [
                        list(items) for items in self._structured_schema_queries
                    ],
                },
                "mcp": {
                    "authority": (
                        "local bounded resources, the mechanical canonical-J "
                        "framework projection, and exact candidate tools only; "
                        "MCP does not invent unresolved semantics, repair, "
                        "select acceptance, or certify J/C/pi"
                    ),
                    "requests": list(self._mcp_requests),
                },
            }
        }


__all__ = [
    "CANDIDATE_CHECK_PROTOCOL",
    "MCP_SEMANTIC_FINALIZE_PROTOCOL",
    "INTENTIR_LIFT_SUBMIT",
    "MULTI_ACTION_PROTOCOL",
    "IntentIRLiftEnvironment",
    "IntentIRLiftEnvironmentConfig",
    "IntentIRLiftWorkspace",
    "MAX_ACTION_OUTPUT_BYTES",
    "MAX_CANDIDATE_BYTES",
    "MAX_STARTUP_CONTEXT_BYTES",
    "MAX_WORKSPACE_BYTES",
    "MAX_WORKSPACE_DEPTH",
    "MAX_WORKSPACE_ENTRIES",
]
