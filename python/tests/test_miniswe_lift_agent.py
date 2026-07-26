from __future__ import annotations

import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.model import IntentOp
from intentir.extraction.lifters.miniswe import (
    MINISWE_LIFT_KNOWLEDGE,
    MiniSWELiftAdapter,
    _input_files,
)
from intentir.extraction.lifters.scrc import (
    CANDIDATE_CHECK_PROTOCOL as SCRC_CANDIDATE_CHECK_PROTOCOL,
    SCRCCandidateCheckContext,
    SCRCLiftAttempt,
    SCRCReliftRecord,
    check_scrc_lift_candidate,
    prepare_scrc_lift,
    scrc_candidate_check_context,
)
from intentir.extraction.providers.provider import ProviderConfig
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.typecheck import type_check
from intentir.verify.scrc_runner import (
    CandidateLiftError,
    FeedbackKind,
    HarnessBindingInstance,
    LiftFeedback,
    SCRCRunRequest,
    SourceProgram,
)
from minisweagent.config.intentir_context_query import (
    _bootstrap,
    _candidate,
    _candidate_abi,
    _candidate_find,
    _candidate_node,
    _candidate_nodes,
    _feedback,
    _feedback_category,
    _source_find,
)
from minisweagent.environments.intentir_lift import (
    CANDIDATE_CHECK_PROTOCOL,
    INTENTIR_CANDIDATE_CHECK,
    INTENTIR_LIFT_SUBMIT,
    IntentIRLiftEnvironment,
    IntentIRLiftWorkspace,
    MAX_ACTION_OUTPUT_BYTES,
    MAX_CANDIDATE_BYTES,
)
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall_response import (
    parse_toolcall_actions_response,
)
from minisweagent.exceptions import FormatError
from minisweagent.run.intentir_lift import (
    IntentIRLiftRunError,
    IntentIRLiftRunResult,
    create_intentir_lift_model,
    load_intentir_lift_config,
    run_intentir_lift,
)


class _ScriptedModel:
    def __init__(self, commands: list[str]):
        self.commands = list(commands)
        self.query_inputs: list[list[dict]] = []

    def query(self, messages, **kwargs):
        del kwargs
        self.query_inputs.append(list(messages))
        command = self.commands.pop(0)
        return {
            "role": "assistant",
            "content": "",
            "extra": {"actions": [{"command": command}], "cost": 0.0},
        }

    def format_message(self, **kwargs):
        return dict(kwargs)

    def format_observation_messages(self, message, outputs, template_vars=None):
        del message, template_vars
        return [
            {
                "role": "user",
                "content": output["output"],
                "extra": {"returncode": output["returncode"]},
            }
            for output in outputs
        ]

    def get_template_vars(self, **kwargs):
        return dict(kwargs)

    def serialize(self):
        return {"info": {"config": {"model": "scripted"}}}


def _workspace() -> IntentIRLiftWorkspace:
    return IntentIRLiftWorkspace.create(
        {
            "README.md": "allowed input",
            "schema/artifact.schema.json": "{}",
        }
    )


def _real_lift_input_files() -> dict[str, str]:
    prepared = prepare_scrc_lift(
        """@triton.jit
def source_copy(a, b, n, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    value = tl.load(a + offsets, mask=mask)
    tl.store(b + offsets, value, mask=mask)
"""
    )
    return _input_files(prepared)


def test_real_input_bundle_has_compact_source_neutral_knowledge() -> None:
    files = _real_lift_input_files()

    assert CANDIDATE_CHECK_PROTOCOL == SCRC_CANDIDATE_CHECK_PROTOCOL
    assert MINISWE_LIFT_KNOWLEDGE == "intentir.lift-knowledge.v39"
    assert "knowledge/minimal_example.json" in files
    assert "knowledge/intent_ops.md" in files
    assert "tools/schema_query.py" in files
    assert "tools/context_query.py" in files
    assert "map_binary" in files["knowledge/intent_ops.md"]
    assert "contraction: binary einsum" in files["knowledge/intent_ops.md"]
    assert '"lhs,rhs->out" label string (no ellipsis)' in (
        files["knowledge/intent_ops.md"]
    )
    assert "Do not page through the raw schema" in files["README.md"]
    assert "context_query.py bootstrap" in files["README.md"]
    assert "source-find" in files["README.md"]
    assert "candidate-nodes" in files["README.md"]
    assert "read-only discovery" in files["README.md"]
    assert "integral rank-0 scalar explicitly equated" in files["README.md"]
    assert "J-owned" in files["README.md"]
    assert "value_dimension_relations" in files["README.md"]
    assert "explicit sum-then-div stages" in files["README.md"]
    assert "intentir_candidate_check /output/candidate.json" in (
        files["README.md"]
    )
    readme = " ".join(files["README.md"].split())
    assert "eleven-evaluation ordinary allowance" in readme
    assert "separately reserved submission preflight" in readme
    assert "bounded source-free CandidateCheck diagnostic summary" in readme
    assert "must be the entire shell action exactly as shown" in readme
    assert "Do not prepend `set`" in readme
    lift_contract = " ".join(files["knowledge/lift_contract.md"].split())
    assert "never names a p* parameter, source/local, or J-body v*" in lift_contract
    assert "Harness-owned, non-C JIT bindings" in lift_contract
    assert "constexprness alone never makes one a graph selector" in lift_contract
    assert "every semantic use is exclusively compile-time control" in lift_contract
    assert "fixed code checks only the authored wire/type contract" in lift_contract
    preflight = " ".join(files["knowledge/preflight.md"].split())
    assert "Inspect every Source use of that parameter" in preflight
    assert "Any such active use requires t*/x* even when constant-folded" in preflight
    assert "Every semantic `reduce` and inclusive `scan`" in preflight
    assert "one `consumer_extensions` relation for each affected" in preflight
    assert "A reduction identity or load fill is not a substitute" in preflight
    assert "Never replace the program path with constant zero" in preflight
    assert "route `p*` directly into `address_base`" in preflight
    assert "pointer `p*` bypasses it" in preflight
    assert "Never decide from the base-to-lane edge alone" in preflight
    assert "same program-axis identity, not evidence for another axis" in preflight
    assert "defined once and reused by several accesses" in preflight
    assert "one producer with routed consumers" in preflight
    assert "syntactic uses do not create repeated producer instances" in preflight
    assert "uses only canonical `attrs.subscripts`" in preflight
    assert "never emit `lhs_contract_axes`/`rhs_contract_axes`" in preflight
    assert "direct Source contraction primitive" in preflight
    assert "one contraction-anchored `realize` instance" in preflight
    assert "separately compiler-visible outer-product/multiply/reduction" in preflight
    assert "compiler-visible semantic recurrence" in preflight
    assert "existing structured J `loop`" in preflight
    assert "every carry and initial value" in preflight
    assert "matching next/yield/result values" in preflight
    assert "result-axis orientation" in preflight
    assert "expect pi or Replay to recover it" in preflight
    assert "cannot carry or infer an initialization" in preflight
    assert "one Layer-A `effect_if`, not a value `select`" in preflight
    assert "positive and negative arms are both explicit" in preflight
    assert "matching `EffectSpec.access_ref` are the only membership" in preflight
    assert "one-arm early return" in preflight
    assert "entire lexical continuation" in preflight
    assert "one Layer-A `value_if`, not a `select` or `effect_if`" in preflight
    assert "`{scope_id,body,yield_value}` arms" in preflight
    assert "no access/effect, unknown op, or nested" in preflight
    assert "`rounded_extent_step`" in preflight
    assert "same tensor declared with identical type in both J inputs" in preflight
    assert "`boundary.access_guard_relations`" in preflight
    assert "guard shaped `[M]` used by an access shaped `[M,N]`" in preflight
    assert "`select(guard,candidate,current)`" in preflight
    assert "never inserts select, changes output to inout" in preflight
    assert "Classify across all uses of one scalar" in preflight
    assert "reuse that same `t*`" in preflight
    assert "Never rewrite it to `mean`" in preflight
    assert "generated `sum(ones)` count" in preflight
    assert "never copy it to sibling parameters" in preflight
    assert "strictly increasing width" in preflight
    assert "local tile axis after program/grid axes" in preflight


def test_context_bootstrap_keeps_feedback_visible_and_copies_previous(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    previous = {"j": {"large": "x" * 30_000}, "c": {}, "pi": {}}
    feedback = {
        "failure_kind": "gate1",
        "relift_round": 2,
        "diagnostics": {
            "observer_diff": {
                "categories": [
                    {
                        "category": "op_multiset",
                        "difference_count": 1,
                        "examples": [
                            {
                                "kind": "missing",
                                "item": "scf.for",
                                "expected_count": 1,
                                "actual_count": 0,
                            }
                        ],
                    }
                ]
            }
        },
        "previous_artifact": previous,
    }
    (input_root / "feedback.json").write_text(json.dumps(feedback))
    (input_root / "source.py").write_text("def source():\n    pass\n")
    (input_root / "harness_view.json").write_text("{}\n")

    _bootstrap(input_root, output_root)

    observation = capsys.readouterr().out
    assert len(observation) < 12_000
    assert '"category": "op_multiset"' in observation
    assert '"candidate_copied_to_output": true' in observation
    assert '"feedback_kind": "gate1"' in observation
    assert json.loads((output_root / "candidate.json").read_text()) == previous


def test_dataflow_region_feedback_routes_agent_to_lifetime_knowledge(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    feedback = {
        "failure_kind": "gate1",
        "diagnostics": {
            "observer_diff": {
                "categories": [
                    {
                        "category": "dataflow",
                        "difference_count": 2,
                        "examples": [
                            {
                                "kind": "missing",
                                "item": [
                                    "edge",
                                    "region[root]",
                                    "tt.make_range[node]",
                                    "region.member",
                                ],
                            },
                            {
                                "kind": "unexpected",
                                "item": [
                                    "edge",
                                    "region[scf.for:node]",
                                    "tt.make_range[node]",
                                    "region.member",
                                ],
                            },
                        ],
                    }
                ]
            }
        },
        "previous_artifact": None,
    }
    (input_root / "feedback.json").write_text(json.dumps(feedback))

    _feedback_category(input_root, "dataflow")

    result = json.loads(capsys.readouterr().out)
    assert "cross root/scf.for" in result["diagnostic_hint"]
    assert "knowledge 'lexical lifetime'" in result["knowledge_query"]
    assert result["examples"] == feedback["diagnostics"]["observer_diff"][
        "categories"
    ][0]["examples"]


def test_dataflow_feedback_keeps_effect_order_when_node_witnesses_are_large(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_root = tmp_path / "input"
    input_root.mkdir()
    effect_witnesses = [
        {
            "side": "source",
            "kind": "missing",
            "region": "region[root]",
            "trace": "effect.trace[source_node_0]",
            "ordered_events": ["region:region[scf.for:source_node_1]", "tt.store"],
        },
        {
            "side": "replay",
            "kind": "unexpected",
            "region": "region[root]",
            "trace": "effect.trace[replay_node_0]",
            "ordered_events": ["tt.store", "region:region[scf.for:replay_node_1]"],
        },
    ]
    feedback = {
        "failure_kind": "gate1",
        "diagnostics": {
            "observer_diff": {
                "categories": [
                    {
                        "category": "dataflow",
                        "difference_count": 200,
                        "examples": [],
                        "effect_order_witnesses": effect_witnesses,
                        "node_witnesses": [
                            {"node": f"node_{index}", "detail": "x" * 800}
                            for index in range(20)
                        ],
                    }
                ]
            }
        },
        "previous_artifact": None,
    }
    (input_root / "feedback.json").write_text(json.dumps(feedback))

    _feedback_category(input_root, "dataflow")

    result = json.loads(capsys.readouterr().out)
    assert result["effect_order_witnesses"] == effect_witnesses
    assert result["node_witness_count"] == 20
    assert "effect_order_witnesses" in result["effect_order_query"]
    assert "bulk node witnesses omitted" in result["bounded_view"]


def test_candidate_abi_query_reports_only_public_mechanical_differences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    harness = {
        "parameter_order": ["p0", "p1", "p2", "p3", "c0"],
        "arguments": [
            {
                "name": "p0",
                "kind": "tensor",
                "mode": "input",
                "dtype": "f32",
                "shape": [2, 3],
            },
            {
                "name": "p1",
                "kind": "scalar",
                "mode": "value",
                "dtype": "i64",
                "value": 3,
            },
            {
                "name": "p2",
                "kind": "none",
                "mode": "value",
                "dtype": "none",
            },
        ],
        "bound_constexpr_values": {"p3": 4},
        "schedule_values": {"c0": 128},
        "compiler_slots": [],
        "compiler_roles": {},
        "constexpr_parameters": ["p3", "c0"],
        "aliases": [],
    }
    candidate = {
        "j": {
            "signature": {
                "parameters": [
                    {
                        "name": "p0",
                        "kind": "pointer",
                        "mode": "input",
                        "dtype": "f32",
                        "rank": 2,
                        "logical_ref": "t0",
                    },
                    {
                        "name": "p1",
                        "kind": "scalar",
                        "mode": "value",
                        "dtype": "i64",
                        "rank": 0,
                        "logical_ref": "x0",
                    },
                    {
                        "name": "p2",
                        "kind": "opaque",
                        "mode": "value",
                        "dtype": "none",
                        "rank": None,
                        "logical_ref": None,
                    },
                    {
                        "name": "p3",
                        "kind": "constexpr",
                        "mode": "value",
                        "dtype": "i32",
                        "rank": 0,
                        "logical_ref": "t1",
                    },
                ]
            },
            "boundary": {"aliases": []},
        },
        "c": {
            "slots": [
                {
                    "name": "c0",
                    "kind": "constexpr",
                    "role": "tile_extent",
                }
            ],
            "values": {"c0": 128},
        },
        "pi": {},
    }
    (input_root / "harness_view.json").write_text(json.dumps(harness))
    candidate_path = output_root / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    before = candidate_path.read_bytes()

    _candidate_abi(input_root, output_root)

    report = json.loads(capsys.readouterr().out)
    assert report["matches_public_mechanical_abi"] is True
    assert report["issue_count"] == 0
    assert "logical_ref" in " ".join(report["agent_owned_not_inferred"])
    assert candidate_path.read_bytes() == before

    candidate["j"]["signature"]["parameters"][1]["dtype"] = "f32"
    candidate["c"]["values"]["c0"] = 64
    candidate_path.write_text(json.dumps(candidate))
    _candidate_abi(input_root, output_root)

    report = json.loads(capsys.readouterr().out)
    assert report["matches_public_mechanical_abi"] is False
    assert {item["code"] for item in report["issues"]} == {
        "parameter_field_mismatch",
        "c_values_mismatch",
    }
    assert {item["path"] for item in report["issues"]} == {
        "j.signature.parameters[p1].dtype",
        "c.values",
    }


def test_context_query_reads_exact_feedback_and_candidate_paths_without_editing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    feedback = {
        "failure_kind": "gate1",
        "diagnostics": {
            "unresolved_gate1": {
                "diagnostic": "missing scf.for",
                "round": 2,
            },
            "candidate_check_feedback": {
                "rule_code": "scope_invalid",
            },
        },
        "previous_artifact": {"omitted": True},
    }
    candidate = {
        "j": {
            "signature": {"parameters": [{"name": "p0"}]},
            "scopes": [
                {
                    "scope_id": "q0",
                    "definitions": ["t0", "v0"],
                    "captures": [],
                }
            ],
        },
        "c": {"slots": [], "values": {}},
        "pi": {"instances": [{"id": "i0", "semantic_ref": "v0"}]},
    }
    (input_root / "feedback.json").write_text(json.dumps(feedback))
    candidate_path = output_root / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    before = candidate_path.read_bytes()

    _feedback(input_root, "diagnostics.unresolved_gate1")
    assert json.loads(capsys.readouterr().out) == {
        "diagnostic": "missing scf.for",
        "round": 2,
    }

    _candidate(output_root, "j.scopes.0")
    assert json.loads(capsys.readouterr().out)["scope_id"] == "q0"

    _candidate_find(output_root, "v0")
    report = json.loads(capsys.readouterr().out)
    assert report["paths"] == [
        "j.scopes.0.definitions.1",
        "pi.instances.0.semantic_ref",
    ]
    assert report["authority"].startswith("read-only")
    assert candidate_path.read_bytes() == before


def test_context_query_searches_source_and_pages_authored_nodes_without_editing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    (input_root / "source.py").write_text(
        "def kernel(x):\n"
        "    value = load(x)\n"
        "    for offset in range(4):\n"
        "        value = value + offset\n"
        "    store(x, value)\n"
    )
    candidate = {
        "j": {
            "intent": {
                "intent_function": {
                    "body": [
                        {
                            "id": "n0",
                            "op": "map_unary",
                            "operands": ["t0"],
                            "result": "v0",
                            "attrs": {"fn": "neg"},
                        },
                        {
                            "id": "n1",
                            "op": "loop",
                            "body": [
                                {
                                    "id": "n2",
                                    "op": "map_binary",
                                    "operands": ["v0", "t1"],
                                    "result": "v1",
                                    "attrs": {"fn": "add"},
                                }
                            ],
                        },
                    ]
                }
            }
        },
        "c": {"slots": [], "values": {}},
        "pi": {},
    }
    candidate_path = output_root / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    before = candidate_path.read_bytes()

    _source_find(input_root, "value", 1)
    source_report = capsys.readouterr().out
    assert "2 |     value = load(x)" in source_report
    assert "4 |         value = value + offset" in source_report

    _candidate_nodes(output_root, 1, 2)
    page = json.loads(capsys.readouterr().out)
    assert page["total"] == 3
    assert [node["id"] for node in page["nodes"]] == ["n0", "n1"]
    assert page["next_command"].endswith("--start 3 --count 2")

    _candidate_nodes(output_root, 1, 10, "map_binary")
    filtered = json.loads(capsys.readouterr().out)
    assert filtered["total"] == 1
    assert filtered["nodes"][0]["id"] == "n2"

    _candidate_node(output_root, "n2")
    node = json.loads(capsys.readouterr().out)
    assert node["path"].endswith("body.1.body.0")
    assert node["node"]["operands"] == ["v0", "t1"]
    assert candidate_path.read_bytes() == before


def test_context_discovery_queries_work_in_real_bubblewrap() -> None:
    workspace = IntentIRLiftWorkspace.create(_real_lift_input_files())
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        copied = environment.execute(
            {
                "command": (
                    "cp /input/knowledge/minimal_example.json "
                    "/output/candidate.json"
                )
            }
        )
        assert copied["returncode"] == 0, copied
        before = workspace.candidate_path.read_bytes()
        result = environment.execute(
            {
                "command": (
                    "python /input/tools/context_query.py source-find mask "
                    "--context 1; "
                    "python /input/tools/context_query.py candidate-nodes "
                    "--start 1 --count 2; "
                    "python /input/tools/context_query.py candidate-node n0"
                )
            }
        )
        after = workspace.candidate_path.read_bytes()
    finally:
        workspace.cleanup()

    assert result["returncode"] == 0, result
    assert "--- source.py:" in result["output"]
    assert ">" in result["output"] and " < " in result["output"]
    assert result["output"].count(
        '"protocol": "intentir.lift-candidate-node-query.v1"'
    ) == 2
    assert '"id": "n0"' in result["output"]
    assert after == before


def test_compact_schema_queries_work_in_real_bubblewrap() -> None:
    workspace = IntentIRLiftWorkspace.create(_real_lift_input_files())
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json top; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json catalog; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json op map_binary; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json op effect_if; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json op value_if; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section parameter; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section access; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section output_mapping; "
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json section pi_instance"
                )
            }
        )
        loop_carrier_result = environment.execute(
            {
                "command": (
                    "python /input/tools/schema_query.py "
                    "/input/schema/artifact.schema.json "
                    "section pi_loop_carrier"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 0, result
    output = result["output"]
    assert "root: required=['schema', 'j', 'c', 'pi']" in output
    assert "ordinary operations: map_binary, map_unary" in output
    assert 'fn in' in output
    assert '"effect_if"' in output
    assert '"value_if"' in output
    assert '"yield_value"' in output
    assert '"positive"' in output and '"negative"' in output
    assert '"logical_ref"' in output
    assert "rank-0 J input" in output
    assert "never names a v* SSA value" in output
    assert '"access_id"' in output
    assert "deterministically elaborated" in output
    assert "fill_value is legal only on a load" in output
    assert "leave output_mappings empty" in output
    assert '"semantic_ref"' in output
    assert len(output.splitlines()) < 450
    assert loop_carrier_result["returncode"] == 0
    assert '"loop_instance"' in loop_carrier_result["output"]
    assert '"state"' in loop_carrier_result["output"]
    assert len(loop_carrier_result["output"].splitlines()) < 80


def test_minimal_example_passes_static_schema_and_canonical_parser() -> None:
    files = _real_lift_input_files()
    payload = json.loads(files["knowledge/minimal_example.json"])
    artifact = Gate1Artifact.from_json(payload)
    assert artifact.to_json()["j"]["scopes"][0]["scope_id"] == "q0"
    assert type_check(artifact.j, artifact.c, artifact.pi).passed
    assert payload["j"]["signature"]["parameters"][2]["logical_ref"] == "t1"
    assert payload["j"]["signature"]["parameters"][2]["name"] == "p2"
    assert len(payload["j"]["signature"]["parameters"]) == 3
    assert payload["j"]["value_dimension_relations"] == [
        {"value_ref": "t1", "dimension_ref": "x0", "relation": "eq"}
    ]
    preflight = " ".join(files["knowledge/preflight.md"].split())
    assert "exactly one `logical_ref`" in preflight
    assert "never duplicate or" in preflight
    assert "author one `J.value_dimension_relations` item" in preflight
    assert "fixed code, Replay, C, and pi never add or repair" in preflight
    assert "t1" in payload["j"]["boundary"]["accesses"][0]["predicate"]
    assert payload["j"]["intent"]["intent_function"]["body"][0]["attrs"][
        "fn"
    ] == "sum"
    assert payload["j"]["intent"]["intent_function"]["body"][2]["attrs"][
        "fn"
    ] == "div"
    assert payload["j"]["boundary"]["reductions"][0]["tile_axis"] == 0

    workspace = IntentIRLiftWorkspace.create(files)
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "cp /input/knowledge/minimal_example.json "
                    "/output/candidate.json && "
                    "python /input/tools/validate_candidate.py "
                    "/input/schema/artifact.schema.json "
                    "/output/candidate.json"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 0, result
    assert "static schema: PASS" in result["output"]


def test_value_dimension_equality_is_explicitly_authored_not_inferred(
) -> None:
    files = _real_lift_input_files()
    payload = json.loads(files["knowledge/minimal_example.json"])
    artifact = Gate1Artifact.from_json(payload)
    contract = " ".join(files["knowledge/lift_contract.md"].split())
    preflight = " ".join(files["knowledge/preflight.md"].split())

    # The physical scalar still binds only t1; the explicit J relation, rather
    # than a duplicated p* or Harness coincidence, connects it to x0.
    assert type_check(artifact.j, artifact.c, artifact.pi).passed
    assert payload["j"]["signature"]["parameters"][2]["logical_ref"] == "t1"
    assert payload["j"]["value_dimension_relations"] == [
        {"value_ref": "t1", "dimension_ref": "x0", "relation": "eq"}
    ]
    assert "J.value_dimension_relations" in contract
    assert "Author it only from exact Source identity" in contract
    assert "author one `J.value_dimension_relations` item" in preflight
    assert "Do not infer equality from a current Harness value" in preflight


def test_intentir_observation_exposes_remaining_turn_budget() -> None:
    config = load_intentir_lift_config()
    assert config["agent"]["step_limit"] == 80
    assert config["environment"]["candidate_check_limit"] == 15
    model = create_intentir_lift_model(
        model_name="test",
        api_base="http://provider.invalid/v1",
        api_key="not-used",
        reasoning_effort="xhigh",
        timeout_seconds=1,
    )
    message = {
        "extra": {
            "actions": [
                {"command": "true", "tool_call_id": "call-budget"}
            ]
        }
    }
    observation = model.format_observation_messages(
        message,
        [{"returncode": 0, "output": "ok", "exception_info": ""}],
        {"n_model_calls": 60, "step_limit": 80},
    )[0]
    rendered = json.loads(observation["output"])

    assert rendered["turn_budget"] == {
        "used": 60,
        "limit": 80,
        "remaining": 20,
    }
    assert rendered["directive"].startswith("Stop browsing")


def test_real_default_agent_and_bubblewrap_deliver_one_candidate() -> None:
    workspace = _workspace()
    candidate = '{"j": {}, "c": {}, "pi": {}}'
    model = _ScriptedModel(
        [
            f"printf '%s\\n' '{candidate}' > /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
        ]
    )
    try:
        result = run_intentir_lift(model=model, workspace=workspace)
    finally:
        workspace.cleanup()

    assert json.loads(result.candidate_text or "") == json.loads(candidate)
    assert result.exit_status == "Submitted"
    assert result.model_calls == 2
    assert len(model.query_inputs) == 2
    assert len(model.query_inputs[0]) == 2


def test_candidate_check_is_bounded_closed_and_diagnostics_only() -> None:
    workspace = _workspace()
    workspace.candidate_path.write_text('{"candidate": 1}\n', encoding="utf-8")
    seen: list[str] = []

    def checker(candidate_text: str) -> dict:
        seen.append(candidate_text)
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        candidate_checker=checker,
        candidate_check_limit=2,
    )
    original = workspace.candidate_path.read_bytes()
    try:
        first = environment.execute(
            {"command": f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json"}
        )
        second = environment.execute(
            {"command": f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json"}
        )
        exhausted = environment.execute(
            {"command": f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json"}
        )
        evidence = environment.serialize()["info"]["candidate_check"]
    finally:
        workspace.cleanup()

    assert first["returncode"] == second["returncode"] == 0
    assert exhausted["returncode"] == 1
    assert json.loads(exhausted["output"])["diagnostics"][0][
        "rule_code"
    ] == "check_budget_exhausted"
    assert seen == ['{"candidate": 1}\n', '{"candidate": 1}\n']
    assert original == b'{"candidate": 1}\n'
    assert evidence["calls"] == 3
    assert evidence["attempts"] == 3
    assert evidence["evaluations"] == 2
    assert evidence["limit"] == 3
    assert evidence["shared_budget"] == {"used": 2, "limit": 2}
    assert evidence["submission_reserve"] == {"used": 0, "limit": 1}
    assert evidence["explicit"] == {"calls": 3, "evaluations": 2}
    assert evidence["submission"] == {"calls": 0, "evaluations": 0}
    assert evidence["results"][-1]["budget_pool"] is None
    assert evidence["results"][-1]["evaluated"] is False
    assert [item["status"] for item in evidence["results"]] == [
        "pass", "pass", "fail"
    ]
    assert [item["trigger"] for item in evidence["results"]] == [
        "explicit", "explicit", "explicit"
    ]


def test_agent_can_use_candidate_check_before_independent_submission() -> None:
    workspace = _workspace()
    candidate = '{"j": {}, "c": {}, "pi": {}}'
    model = _ScriptedModel(
        [
            f"printf '%s\\n' '{candidate}' > /output/candidate.json",
            f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
        ]
    )

    def checker(candidate_text: str) -> dict:
        assert json.loads(candidate_text) == json.loads(candidate)
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    try:
        result = run_intentir_lift(
            model=model,
            workspace=workspace,
            candidate_checker=checker,
        )
    finally:
        workspace.cleanup()

    assert result.exit_status == "Submitted"
    assert result.model_calls == 3
    check_evidence = result.trajectory["info"]["candidate_check"]
    assert check_evidence["calls"] == 1
    assert check_evidence["attempts"] == 1
    assert check_evidence["evaluations"] == 1
    assert check_evidence["limit"] == 16
    assert check_evidence["shared_budget"] == {"used": 1, "limit": 15}
    assert check_evidence["submission_reserve"] == {
        "used": 0,
        "limit": 1,
    }
    assert check_evidence["results"][0]["status"] == "pass"
    assert check_evidence["results"][0]["trigger"] == "explicit"


def test_submission_preflight_returns_diagnostics_then_accepts_repair() -> None:
    workspace = _workspace()
    rejected = '{"valid": false}'
    accepted = '{"valid": true}'
    model = _ScriptedModel(
        [
            f"printf '%s\\n' '{rejected}' > /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
            f"printf '%s\\n' '{accepted}' > /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
        ]
    )

    def checker(candidate_text: str) -> dict:
        if json.loads(candidate_text).get("valid") is True:
            return {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "status": "pass",
                "diagnostics": [],
            }
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "fail",
            "diagnostics": [
                {
                    "rule_code": "candidate_invalid",
                    "path": "<candidate>",
                    "observed_refs_or_types": [],
                }
            ],
        }

    try:
        result = run_intentir_lift(
            model=model,
            workspace=workspace,
            candidate_checker=checker,
        )
    finally:
        workspace.cleanup()

    assert json.loads(result.candidate_text or "") == {"valid": True}
    assert result.model_calls == 4
    assert "Submission preflight rejected" in json.dumps(result.trajectory)
    evidence = result.trajectory["info"]["candidate_check"]
    assert evidence["calls"] == 2
    assert [item["status"] for item in evidence["results"]] == ["fail", "pass"]
    assert [item["trigger"] for item in evidence["results"]] == [
        "submission", "submission"
    ]
    assert evidence["results"][0]["rule_diagnostics"] == [
        {
            "rule_code": "candidate_invalid",
            "path": "<candidate>",
            "observed_refs_or_types": [],
        }
    ]


def test_submission_has_one_check_after_ordinary_budget_is_exhausted() -> None:
    workspace = _workspace()
    rejected = '{"valid": false}'
    accepted = '{"valid": true}'
    model = _ScriptedModel(
        [
            f"printf '%s\\n' '{rejected}' > /output/candidate.json",
            f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json",
            f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json",
            f"printf '%s\\n' '{accepted}' > /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
        ]
    )

    def checker(candidate_text: str) -> dict:
        valid = json.loads(candidate_text).get("valid") is True
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "pass" if valid else "fail",
            "diagnostics": [] if valid else [
                {
                    "rule_code": "candidate_invalid",
                    "path": "j.intent_function",
                    "observed_refs_or_types": [],
                }
            ],
        }

    try:
        result = run_intentir_lift(
            model=model,
            workspace=workspace,
            candidate_checker=checker,
            config_overrides={
                "environment": {
                    "candidate_check_limit": 2,
                    "candidate_submission_check_limit": 1,
                }
            },
        )
    finally:
        workspace.cleanup()

    assert json.loads(result.candidate_text or "") == {"valid": True}
    evidence = result.trajectory["info"]["candidate_check"]
    assert evidence["evaluations"] == 3
    assert evidence["shared_budget"] == {"used": 2, "limit": 2}
    assert evidence["submission_reserve"] == {"used": 1, "limit": 1}
    assert [item["budget_pool"] for item in evidence["results"]] == [
        "shared",
        "shared",
        "submission_reserve",
    ]
    assert [item["trigger"] for item in evidence["results"]] == [
        "explicit",
        "explicit",
        "submission",
    ]


def test_submission_preflight_matches_exact_bytes_not_json_equivalence() -> None:
    workspace = _workspace()
    compact = '{"j":{},"c":{},"pi":{}}'
    pretty = '{"j": {}, "c": {}, "pi": {}}'
    model = _ScriptedModel(
        [
            f"printf '%s\\n' '{compact}' > /output/candidate.json",
            f"{INTENTIR_CANDIDATE_CHECK} /output/candidate.json",
            f"printf '%s\\n' '{pretty}' > /output/candidate.json",
            f"printf '{INTENTIR_LIFT_SUBMIT}\\n'",
        ]
    )
    checked: list[str] = []

    def checker(candidate_text: str) -> dict:
        checked.append(candidate_text)
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    try:
        result = run_intentir_lift(
            model=model,
            workspace=workspace,
            candidate_checker=checker,
        )
    finally:
        workspace.cleanup()

    assert result.exit_status == "Submitted"
    assert checked == [compact + "\n", pretty + "\n"]
    evidence = result.trajectory["info"]["candidate_check"]
    assert [item["trigger"] for item in evidence["results"]] == [
        "explicit", "submission"
    ]


def test_submission_rejects_checker_side_effect_after_pass() -> None:
    workspace = _workspace()
    checked = '{"which": "checked"}\n'
    changed = '{"which": "changed-after-check"}\n'
    workspace.candidate_path.write_text(checked, encoding="utf-8")

    def checker(candidate_text: str) -> dict:
        assert candidate_text == checked
        workspace.candidate_path.write_text(changed, encoding="utf-8")
        return {
            "protocol": CANDIDATE_CHECK_PROTOCOL,
            "status": "pass",
            "diagnostics": [],
        }

    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
        candidate_checker=checker,
    )
    try:
        result = environment.execute(
            {"command": f"printf '{INTENTIR_LIFT_SUBMIT}\\n'"}
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 2
    assert "changed after the exact bytes were checked" in result["output"]


def test_bubblewrap_hides_host_env_repo_and_network_and_protects_input(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INTENTIR_TEST_CPA_KEY", "must-not-enter-sandbox")
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    host_net_namespace = os.readlink("/proc/self/ns/net")
    try:
        checks = environment.execute(
            {
                "command": (
                    "test -r /input/README.md && "
                    "test ! -w /input/README.md && "
                    "test ! -e /home/kingdom && "
                    "! touch /rootfile && "
                    "! touch /dev/unmonitored && "
                    "if [ -d /dev/shm ]; then ! touch /dev/shm/unmonitored; fi && "
                    "touch /tmp/monitored && "
                    "test -z \"${INTENTIR_TEST_CPA_KEY+x}\" && "
                    "python -c 'import importlib.util; "
                    "assert importlib.util.find_spec(\"intentir\") is None' && "
                    "readlink /proc/self/ns/net"
                )
            }
        )
        monitored_tmp_exists = (workspace.tmp_dir / "monitored").exists()
    finally:
        workspace.cleanup()

    assert checks["returncode"] == 0, checks
    assert checks["output"].splitlines()[-1] != host_net_namespace
    assert monitored_tmp_exists


def test_submission_rejects_extra_output_file_without_ending_session() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "printf '{}\\n' > /output/candidate.json; "
                    "printf 'draft\\n' > /output/extra.txt; "
                    f"printf '{INTENTIR_LIFT_SUBMIT}\\n'"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] == 2
    assert "exactly candidate.json" in result["output"]


def test_candidate_size_is_rejected_before_unbounded_read() -> None:
    workspace = _workspace()
    try:
        with workspace.candidate_path.open("wb") as stream:
            stream.truncate(MAX_CANDIDATE_BYTES + 1)
        with pytest.raises(ValueError, match="candidate.json size"):
            workspace.read_candidate()
    finally:
        workspace.cleanup()


def test_shell_stdout_is_hard_bounded() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "head -c 2097152 /dev/zero | tr '\\0' x"
                )
            }
        )
    finally:
        workspace.cleanup()

    assert result["returncode"] != 0
    assert len(result["output"].encode()) <= MAX_ACTION_OUTPUT_BYTES
    assert "output reached" in result["exception_info"]


def test_writable_workspace_total_size_is_bounded_and_reset() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "for i in $(seq 1 9); do "
                    "truncate -s 8388608 /work/file-$i; done"
                )
            }
        )
        remaining = tuple(workspace.scratch_dir.iterdir())
    finally:
        workspace.cleanup()

    assert result["returncode"] != 0
    assert "workspace exceeded limit" in result["exception_info"]
    assert remaining == ()


def test_workspace_cleanup_handles_depth_beyond_python_recursion_limit() -> None:
    workspace = _workspace()
    root = workspace.root
    current = workspace.scratch_dir
    for _index in range(1100):
        current = current / "d"
        current.mkdir()

    workspace.cleanup()

    assert not root.exists()


def test_agent_created_excessive_directory_depth_is_reset() -> None:
    workspace = _workspace()
    environment = IntentIRLiftEnvironment(
        input_dir=workspace.input_dir,
        output_dir=workspace.output_dir,
        scratch_dir=workspace.scratch_dir,
    )
    try:
        result = environment.execute(
            {
                "command": (
                    "for i in $(seq 1 80); do mkdir d; cd d; done"
                )
            }
        )
        remaining = tuple(workspace.scratch_dir.iterdir())
    finally:
        workspace.cleanup()

    assert result["returncode"] != 0
    assert "workspace exceeded limit" in result["exception_info"]
    assert remaining == ()


def test_five_candidate_workspaces_are_concurrently_isolated() -> None:
    barrier = threading.Barrier(5, timeout=5)

    def run(branch: int) -> tuple[Path, str]:
        workspace = IntentIRLiftWorkspace.create(
            {"branch.txt": f"branch-{branch}"}
        )
        environment = IntentIRLiftEnvironment(
            input_dir=workspace.input_dir,
            output_dir=workspace.output_dir,
            scratch_dir=workspace.scratch_dir,
        )
        root = workspace.root
        try:
            barrier.wait()
            result = environment.execute({"command": "cat /input/branch.txt"})
            assert result["returncode"] == 0
            return root, result["output"].strip()
        finally:
            workspace.cleanup()

    with ThreadPoolExecutor(max_workers=5) as executor:
        outcomes = list(executor.map(run, range(1, 6)))

    assert len({root for root, _value in outcomes}) == 5
    assert {value for _root, value in outcomes} == {
        f"branch-{index}" for index in range(1, 6)
    }
    assert all(not root.exists() for root, _value in outcomes)


def test_step_limit_is_one_candidate_failure_not_an_internal_relift() -> None:
    workspace = _workspace()
    model = _ScriptedModel(["true"])
    try:
        with pytest.raises(IntentIRLiftRunError) as captured:
            run_intentir_lift(
                model=model,
                workspace=workspace,
                config_overrides={"agent": {"step_limit": 1}},
            )
    finally:
        workspace.cleanup()

    assert captured.value.cause is None
    assert captured.value.result.exit_status == "LimitsExceeded"
    assert captured.value.result.model_calls == 1


def test_litellm_model_has_one_explicit_attempt(monkeypatch) -> None:
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        del kwargs
        calls += 1
        raise RuntimeError("transport failed")

    monkeypatch.setattr("litellm.responses", fail_once)
    model = LitellmResponseModel(
        model_name="openai/test",
        retry_attempts=1,
        cost_tracking="ignore_errors",
    )
    with pytest.raises(RuntimeError, match="transport failed"):
        model.query([])
    assert calls == 1
    assert model.physical_provider_calls == 1
    assert model.transport_retries == 0


def test_litellm_model_retries_one_agent_turn_without_changing_messages(
    monkeypatch,
) -> None:
    calls: list[list[dict]] = []

    def fail_twice(messages, **kwargs):
        del kwargs
        calls.append(list(messages))
        if len(calls) < 3:
            raise RuntimeError("transient transport failure")
        return {"output": []}

    model = LitellmResponseModel(
        model_name="openai/test",
        retry_attempts=3,
        cost_tracking="ignore_errors",
    )
    monkeypatch.setattr(model, "_query", fail_twice)
    monkeypatch.setattr(model, "_calculate_cost", lambda _response: {"cost": 0.0})
    monkeypatch.setattr(model, "_parse_actions", lambda _response: [])

    result = model.query([{"role": "user", "content": "same packet"}])

    assert result["extra"]["actions"] == []
    assert calls == [
        [{"role": "user", "content": "same packet"}],
        [{"role": "user", "content": "same packet"}],
        [{"role": "user", "content": "same packet"}],
    ]
    assert model.physical_provider_calls == 3
    assert model.transport_retries == 2


def test_non_string_shell_action_is_a_model_format_error() -> None:
    with pytest.raises(FormatError) as captured:
        parse_toolcall_actions_response(
            [
                {
                    "type": "function_call",
                    "name": "bash",
                    "call_id": "call-invalid",
                    "arguments": json.dumps({"command": 123}),
                }
            ],
            format_error_template="{{ error }}",
        )
    assert "non-string" in captured.value.messages[0]["content"][0]["text"]


def test_multiple_shell_actions_are_one_turn_format_error() -> None:
    calls = [
        {
            "type": "function_call",
            "name": "bash",
            "call_id": f"call-{index}",
            "arguments": json.dumps({"command": "true"}),
        }
        for index in range(2)
    ]
    with pytest.raises(FormatError) as captured:
        parse_toolcall_actions_response(
            calls,
            format_error_template="{{ error }}",
        )
    assert "exactly one" in captured.value.messages[0]["content"][0]["text"]


def test_model_serialization_redacts_provider_key() -> None:
    model = create_intentir_lift_model(
        model_name="test",
        api_base="http://provider.invalid/v1",
        api_key="not-for-evidence",
        reasoning_effort="xhigh",
        timeout_seconds=1,
    )
    serialized = json.dumps(model.serialize())
    assert "not-for-evidence" not in serialized
    assert "<redacted>" in serialized
    assert model.config.retry_attempts == 3
    assert model.config.use_global_model_stats is False
    assert model.config.model_kwargs["num_retries"] == 0


def test_existing_litellm_responses_model_sends_the_expected_cpa_packet() -> None:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append(
                {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": json.loads(self.rfile.read(length)),
                }
            )
            payload = {
                "id": "response-local",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "test",
                "output": [
                    {
                        "id": "fc-local",
                        "type": "function_call",
                        "call_id": "call-local",
                        "name": "bash",
                        "arguments": json.dumps({"command": "true"}),
                        "status": "completed",
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {
                    "input_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 1,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 3,
                },
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = create_intentir_lift_model(
            model_name="test",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-key",
            reasoning_effort="xhigh",
            timeout_seconds=5,
        )
        global_calls_before = GLOBAL_MODEL_STATS.n_calls
        message = model.query(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "task"},
            ]
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert message["extra"]["actions"] == [
        {"command": "true", "tool_call_id": "call-local"}
    ]
    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/v1/responses"
    assert request["authorization"] == "Bearer local-test-key"
    assert request["body"]["model"] == "test"
    assert request["body"]["reasoning"] == {"effort": "xhigh"}
    assert request["body"]["parallel_tool_calls"] is False
    assert request["body"]["tools"][0]["name"] == "bash"
    assert GLOBAL_MODEL_STATS.n_calls == global_calls_before


def test_cpa_transient_http_failure_uses_bounded_transport_retries() -> None:
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            nonlocal request_count
            request_count += 1
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            payload = json.dumps(
                {"error": {"message": "single failure", "type": "server_error"}}
            ).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        model = create_intentir_lift_model(
            model_name="test",
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            api_key="local-test-key",
            reasoning_effort="xhigh",
            timeout_seconds=5,
        )
        with pytest.raises(Exception, match="single failure"):
            model.query([{"role": "user", "content": "task"}])
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert request_count == 3
    assert model.physical_provider_calls == 3
    assert model.transport_retries == 2


def _adapter_request() -> SCRCRunRequest:
    source = """def source_copy(a, b, n, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n
    value = tl.load(a + offsets, mask=mask)
    tl.store(b + offsets, value, mask=mask)
"""
    return SCRCRunRequest(
        source=SourceProgram(source, "gate-source"),
        binding=HarnessBindingInstance.create(
            {"c0": 1024}, gate1_case="same-c0"
        ),
        frontend="triton",
    )


def test_authoritative_candidate_check_uses_source_free_versioned_context() -> None:
    request = _adapter_request()
    prepared = prepare_scrc_lift(request.source.text)
    context = scrc_candidate_check_context(prepared)
    assert context.expected_parameters == ("p0", "p1", "p2")
    assert context.typecheck_protocol == "intentir.scrc.typecheck.v17"
    assert "source_copy" not in repr(context)
    assert not hasattr(context, "user_input")
    assert not hasattr(context, "harness_binding")

    artifact = make_copy_artifact()
    passed = check_scrc_lift_candidate(
        json.dumps(artifact.to_json()),
        context,
    )
    assert passed == {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "pass",
        "diagnostics": [],
    }

    representation_only = artifact.to_json()
    representation_intent = representation_only["j"]["intent"][
        "intent_function"
    ]
    representation_intent["inputs"][0]["shape"] = [4]
    representation_intent["outputs"][0]["shape"] = [4]
    representation_parameter = representation_only["j"]["signature"][
        "parameters"
    ][2]
    representation_parameter["rank"] = 0
    representation_parameter.pop("logical_ref")
    representation_only["pi"]["instances"].append(
        {
            "id": "i1",
            "semantic_ref": "p2",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    assert check_scrc_lift_candidate(
        json.dumps(representation_only), context
    ) == {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "pass",
        "diagnostics": [],
    }

    unused_scalar = artifact.to_json()
    intent = unused_scalar["j"]["intent"]["intent_function"]
    intent["inputs"][0]["shape"] = [4]
    intent["inputs"].append({"name": "t1", "dtype": "i32", "shape": []})
    intent["outputs"][0]["name"] = "t2"
    intent["outputs"][0]["shape"] = [4]
    intent["body"][0]["result"] = "t2"
    parameters = unused_scalar["j"]["signature"]["parameters"]
    parameters[1]["logical_ref"] = "t2"
    parameters[1]["rank"] = 1
    parameters[2]["logical_ref"] = "t1"
    parameters[2]["rank"] = 0
    unused_rejection = check_scrc_lift_candidate(
        json.dumps(unused_scalar), context
    )
    assert unused_rejection == {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "fail",
        "diagnostics": [
            {
                "rule_code": "logical_input_unused",
                "path": "j.signature.parameters[2].logical_ref",
                "observed_refs_or_types": [
                    "t1",
                    "detail: rank-0 logical input does not contribute to any J "
                    "output dependency or explicit boundary/shape semantic "
                    "relation; a representation-only ABI parameter must use "
                    "logical_ref=null and may be realized only through pi",
                ],
            }
        ],
    }

    malformed_reduce = make_reduce_artifact()
    malformed_reduce.j.intent.body[1].attrs = {
        "kind": "sum",
        "axes": [1],
        "keepdims": False,
    }
    shared_core_rejection = check_scrc_lift_candidate(
        json.dumps(malformed_reduce.to_json()), context
    )
    assert shared_core_rejection["status"] == "fail"
    assert {
        item["rule_code"] for item in shared_core_rejection["diagnostics"]
    } == {"known_op_contract_invalid"}
    assert {
        item["path"] for item in shared_core_rejection["diagnostics"]
    } == {
        "j.intent.intent_function.body[1].attrs.assoc",
        "j.intent.intent_function.body[1].attrs.fn",
        "j.intent.intent_function.body[1].attrs.kind",
    }

    malformed_contraction = make_copy_artifact()
    contraction = malformed_contraction.j.intent.body[0]
    contraction.op = "contraction"
    contraction.operands = ["t0", "t0"]
    contraction.attrs = {
        "lhs_contract_axes": [0],
        "rhs_contract_axes": [0],
    }
    contraction.type = {"dtype": "f32", "shape": ["x0"]}
    contraction_rejection = check_scrc_lift_candidate(
        json.dumps(malformed_contraction.to_json()), context
    )
    assert contraction_rejection["status"] == "fail"
    assert {
        item["rule_code"] for item in contraction_rejection["diagnostics"]
    } == {"known_op_contract_invalid"}
    assert {
        item["path"] for item in contraction_rejection["diagnostics"]
    } == {
        "j.intent.intent_function.body[0].attrs.lhs_contract_axes",
        "j.intent.intent_function.body[0].attrs.rhs_contract_axes",
        "j.intent.intent_function.body[0].attrs.subscripts",
    }

    incomplete_boundary = artifact.to_json()
    intent = incomplete_boundary["j"]["intent"]["intent_function"]
    intent["inputs"][0]["shape"] = [2, "x0"]
    intent["outputs"][0]["shape"] = [2, "x0"]
    incomplete_boundary["j"]["signature"]["parameters"][0]["rank"] = 2
    incomplete_boundary["j"]["signature"]["parameters"][1]["rank"] = 2
    incomplete_boundary["j"]["boundary"]["domains"] = {
        "d0": "0 <= k0 and k0 < 2"
    }
    incomplete_boundary["j"]["boundary"]["accesses"] = [
        {
            "access_id": "a0",
            "kind": "load",
            "tensor": "t0",
            "predicate": "k1 < x0",
            "fill_value": 0.0,
            "node_ref": "n0",
            "index_expression": "[k0, k1]",
        },
        {
            "access_id": "a1",
            "kind": "store",
            "tensor": "t1",
            "predicate": "k1 < x0",
            "node_ref": "n0",
            "index_expression": "[k0, k1]",
        },
    ]
    incomplete_boundary["j"]["boundary"]["effects"] = [
        {
            "effect_id": "e0",
            "kind": "read",
            "target": "t0",
            "node_ref": "n0",
            "access_ref": "a0",
        },
        {
            "effect_id": "e1",
            "kind": "write",
            "target": "t1",
            "node_ref": "n0",
            "access_ref": "a1",
        },
    ]
    boundary_rejection = check_scrc_lift_candidate(
        json.dumps(incomplete_boundary), context
    )
    assert boundary_rejection["status"] == "fail"
    assert [
        item["rule_code"] for item in boundary_rejection["diagnostics"]
    ] == ["functional_boundary_incomplete", "functional_boundary_incomplete"]
    assert [item["path"] for item in boundary_rejection["diagnostics"]] == [
        "j.boundary.accesses[0].predicate",
        "j.boundary.accesses[1].predicate",
    ]

    forward_guard = artifact.to_json()
    intent = forward_guard["j"]["intent"]["intent_function"]
    intent["body"][0]["type"] = {"dtype": "f32", "shape": ["x0"]}
    intent["body"].append(
        {
            "id": "n1",
            "op": "const",
            "result": "v0",
            "attrs": {"value": True, "dtype": "i1", "shape": []},
            "type": {"dtype": "i1", "shape": []},
        }
    )
    forward_guard["j"]["node_ids"]["n1"] = "body[1]"
    forward_guard["j"]["boundary"]["accesses"] = [
        {
            "access_id": "a0",
            "kind": "store",
            "tensor": "t1",
            "predicate": "v0",
            "node_ref": "n0",
            "index_expression": "[k0]",
        }
    ]
    forward_guard["j"]["boundary"]["effects"] = [
        {
            "effect_id": "e0",
            "kind": "write",
            "target": "t1",
            "node_ref": "n0",
            "access_ref": "a0",
        }
    ]
    forward_rejection = check_scrc_lift_candidate(
        json.dumps(forward_guard), context
    )
    assert forward_rejection == {
        "protocol": CANDIDATE_CHECK_PROTOCOL,
        "status": "fail",
        "diagnostics": [
            {
                "rule_code": "boundary_value_not_dominating",
                "path": "j.boundary.accesses[0].predicate",
                "observed_refs_or_types": [
                    "a0",
                    "detail: Layer-A value 'v0' does not dominate the access "
                    "node 'n0'",
                ],
            }
        ],
    }

    stale = SCRCCandidateCheckContext(
        context.expected_parameters,
        context.anonymous_compiler_view,
        context.schema_version,
        context.decode_protocol,
        "intentir.scrc.typecheck.v4",
    )
    mismatch = check_scrc_lift_candidate(
        json.dumps(artifact.to_json()), stale
    )
    assert mismatch["status"] == "fail"
    assert (
        mismatch["diagnostics"][0]["rule_code"]
        == "candidate_check_context_mismatch"
    )

    untyped = make_copy_artifact()
    untyped.j.intent.body.append(
        IntentOp.from_json(
            {
                "id": "n1",
                "op": "opaque_untyped",
                "result": "v0",
                "operands": [],
                "attrs": {},
            }
        ),
    )
    untyped.j.node_ids = {"n0": "body[0]", "n1": "body[1]"}
    shared_core_rejection = check_scrc_lift_candidate(
        json.dumps(untyped.to_json()), context
    )
    assert shared_core_rejection["status"] == "fail"
    assert (
        shared_core_rejection["diagnostics"][0]["rule_code"]
        == "j_value_type_incomplete"
    )

    invalid = artifact.to_json()
    invalid["c"]["slots"][0]["subject_ref"] = "n999"
    rejected = check_scrc_lift_candidate(json.dumps(invalid), context)
    assert rejected["status"] == "fail"
    assert rejected["diagnostics"][0]["rule_code"] == "c_subject_unknown"

    oversized = artifact.to_json()
    oversized["j"]["x" * 10000] = True
    bounded = check_scrc_lift_candidate(json.dumps(oversized), context)
    observed = bounded["diagnostics"][0]["observed_refs_or_types"][0]
    assert bounded["status"] == "fail"
    assert bounded["diagnostics"][0]["rule_code"] == "decode_rejected"
    assert len(observed) <= 2048
    assert observed.endswith("...<truncated>")


def test_candidate_check_rejects_semantics_on_absent_none_carrier() -> None:
    context = SCRCCandidateCheckContext(("p0", "p1", "p2", "p3"))
    candidate = make_copy_artifact().to_json()
    candidate["j"]["signature"]["parameters"].append(
        {"name": "p3", "kind": "opaque", "mode": "value", "dtype": "none"}
    )
    candidate["j"]["boundary"]["realization_authorizations"] = [
        {
            "auth_id": "u0",
            "kind": "reassociation",
            "subject_ref": "p3",
        }
    ]
    candidate["pi"]["instances"].append(
        {
            "id": "i1",
            "semantic_ref": "p3",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    candidate["pi"]["refinements"] = [
        {"instance": "i1", "kind": "reassociation", "auth_ref": "u0"}
    ]

    rejected = check_scrc_lift_candidate(json.dumps(candidate), context)

    assert rejected["status"] == "fail"
    assert [item["rule_code"] for item in rejected["diagnostics"]] == [
        "absent_carrier_semantic_use_forbidden",
        "absent_carrier_semantic_use_forbidden",
        "absent_carrier_semantic_use_forbidden",
    ]
    assert [item["path"] for item in rejected["diagnostics"]] == [
        "j.boundary.realization_authorizations[0].subject_ref",
        "pi.instances[1].semantic_ref",
        "pi.refinements[0].instance",
    ]


def test_candidate_check_aggregates_independent_signature_decode_errors() -> None:
    context = scrc_candidate_check_context(
        prepare_scrc_lift(_adapter_request().source.text)
    )
    candidate = make_copy_artifact().to_json()
    parameters = candidate["j"]["signature"]["parameters"]
    del parameters[0]["rank"]
    del parameters[0]["logical_ref"]
    del parameters[1]["dtype"]
    original = json.dumps(candidate, sort_keys=True)

    rejected = check_scrc_lift_candidate(json.dumps(candidate), context)

    assert json.dumps(candidate, sort_keys=True) == original
    assert rejected["protocol"] == CANDIDATE_CHECK_PROTOCOL
    assert rejected["status"] == "fail"
    assert [item["rule_code"] for item in rejected["diagnostics"]] == [
        "decode_rejected",
        "decode_rejected",
        "decode_rejected",
    ]
    assert [item["path"] for item in rejected["diagnostics"]] == [
        "j.signature.parameters[0].rank",
        "j.signature.parameters[0].logical_ref",
        "j.signature.parameters[1].dtype",
    ]
    assert all(
        item["observed_refs_or_types"]
        == [
            "SchemaError: tensor/pointer signature parameters require dtype, "
            "rank, and a J logical_ref"
        ]
        for item in rejected["diagnostics"]
    )


def test_candidate_check_aggregates_dependent_signature_errors_stably() -> None:
    context = scrc_candidate_check_context(
        prepare_scrc_lift(_adapter_request().source.text)
    )
    candidate = make_copy_artifact().to_json()
    parameters = candidate["j"]["signature"]["parameters"]
    parameters[0]["dtype"] = "f16"
    parameters[0]["rank"] = 0
    parameters[1]["logical_ref"] = "t0"

    rejected = check_scrc_lift_candidate(json.dumps(candidate), context)

    assert rejected["status"] == "fail"
    assert [item["path"] for item in rejected["diagnostics"]] == [
        "j.signature.parameters[0].dtype",
        "j.signature.parameters[0].rank",
        "j.signature.parameters[1].mode",
        "j.signature.parameters",
    ]
    assert all(
        item["rule_code"] == "decode_rejected"
        for item in rejected["diagnostics"]
    )
    assert "does not bind logical tensor(s): t1" in rejected["diagnostics"][-1][
        "observed_refs_or_types"
    ][0]


def test_candidate_check_decode_aggregation_is_bounded_and_non_speculative() -> None:
    context = scrc_candidate_check_context(
        prepare_scrc_lift(_adapter_request().source.text)
    )
    candidate = make_copy_artifact().to_json()
    candidate["j"]["signature"]["parameters"] = [
        {
            "name": f"p{index}",
            "kind": "scalar",
            "mode": "value",
            "dtype": "i32",
        }
        for index in range(25)
    ]

    bounded = check_scrc_lift_candidate(json.dumps(candidate), context)

    assert len(bounded["diagnostics"]) == 20
    assert [item["path"] for item in bounded["diagnostics"]] == [
        f"j.signature.parameters[{index}].logical_ref" for index in range(20)
    ]

    blocked = make_copy_artifact().to_json()
    del blocked["j"]["signature"]["parameters"][0]["logical_ref"]
    blocked_result = check_scrc_lift_candidate(json.dumps(blocked), context)
    assert [item["path"] for item in blocked_result["diagnostics"]] == [
        "j.signature.parameters[0].logical_ref"
    ]
    assert all(
        "does not bind logical tensor" not in observed
        for item in blocked_result["diagnostics"]
        for observed in item["observed_refs_or_types"]
    )


def test_candidate_check_reports_independent_surface_errors_after_decode_rejects() -> None:
    candidate = make_copy_artifact().to_json()
    parameters = candidate["j"]["signature"]["parameters"]
    original_mode = parameters[0]["mode"]
    context = SCRCCandidateCheckContext(
        tuple(item["name"] for item in parameters),
        anonymous_compiler_view={
            "arguments": [
                {
                    "name": parameters[0]["name"],
                    "kind": "tensor",
                    "dtype": parameters[0]["dtype"],
                    "shape": [4],
                    "mode": original_mode,
                }
            ],
            "schedule_values": candidate["c"]["values"],
            "aliases": candidate["j"]["boundary"]["aliases"],
        },
    )
    del candidate["j"]["signature"]["parameters"][0]["logical_ref"]
    body = candidate["j"]["intent"]["intent_function"]["body"]
    body[0]["id"] = "n7"
    candidate["j"]["node_ids"] = {"n7": "body[0]"}
    candidate["j"]["signature"]["parameters"][0]["mode"] = (
        "output" if original_mode != "output" else "input"
    )

    rejected = check_scrc_lift_candidate(json.dumps(candidate), context)

    assert rejected["status"] == "fail"
    assert rejected["diagnostics"][0]["path"] == (
        "j.signature.parameters[0].logical_ref"
    )
    diagnostics = {
        (item["rule_code"], item["path"]): item
        for item in rejected["diagnostics"]
    }
    assert (
        "construction_handle_sequence",
        "j.intent.intent_function.body recursive node ids",
    ) in diagnostics
    assert (
        "construction_abi_mismatch",
        "j.signature.parameters[p0].mode",
    ) in diagnostics
    assert body[0]["id"] == "n7"


def test_decode_ignores_json_object_key_order_for_complete_node_map() -> None:
    candidate = make_copy_artifact().to_json()
    body = candidate["j"]["intent"]["intent_function"]["body"]
    for index in range(1, 11):
        body.append(
            {
                "id": f"n{index}",
                "op": "map_unary",
                "result": f"v{index}",
                "operands": ["t0"],
                "attrs": {"fn": "identity"},
                "type": {"dtype": "f32", "shape": ["x0"]},
            }
        )
    node_map = {
        f"n{index}": f"body[{index}]" for index in range(len(body))
    }
    candidate["j"]["node_ids"] = dict(sorted(node_map.items()))

    decoded = Gate1Artifact.from_json(candidate)

    assert list(candidate["j"]["node_ids"])[:4] == ["n0", "n1", "n10", "n2"]
    assert list(decoded.j.node_ids) == [f"n{index}" for index in range(11)]
    assert decoded.j.node_ids["n10"] == "body[10]"


def test_decode_ignores_json_object_key_order_for_domain_map() -> None:
    candidate = make_copy_artifact().to_json()
    domains = {
        f"d{index}": f"0 <= k{index}" for index in range(11)
    }
    candidate["j"]["boundary"]["domains"] = dict(sorted(domains.items()))

    decoded = Gate1Artifact.from_json(candidate)

    assert list(candidate["j"]["boundary"]["domains"])[:4] == [
        "d0",
        "d1",
        "d10",
        "d2",
    ]
    assert list(decoded.j.boundary.domains) == [
        f"d{index}" for index in range(11)
    ]


def test_adapter_uses_fresh_workspaces_and_canonical_decoder(monkeypatch) -> None:
    artifact = make_copy_artifact()
    captured: list[tuple[Path, dict]] = []

    def fake_model(**kwargs):
        assert kwargs["reasoning_effort"] == "xhigh"
        return object()

    def fake_run(*, model, workspace, config_overrides, candidate_checker):
        del model, config_overrides
        feedback = json.loads(
            (workspace.input_dir / "feedback.json").read_text()
        )
        captured.append((workspace.root, feedback))
        assert candidate_checker(json.dumps(artifact.to_json()))["status"] == "pass"
        response = {
            "object": "response",
            "id": f"response-{len(captured)}",
            "model": "test-model-version",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }
        return IntentIRLiftRunResult(
            json.dumps(artifact.to_json()),
            {
                "messages": [response],
                "info": {"config": {"authorization": "secret"}},
            },
            "Submitted",
            3,
            25.0,
        )

    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.create_intentir_lift_model", fake_model
    )
    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.run_intentir_lift", fake_run
    )
    adapter = MiniSWELiftAdapter(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "secret", "test-model",
            reasoning_effort="xhigh",
        )
    )
    request = _adapter_request()
    first = adapter.lift(request, None)
    feedback = LiftFeedback(
        FeedbackKind.GATE1,
        {"observer_diff": {"category": "dataflow"}},
        1,
        artifact,
    )
    second = adapter.lift(request, feedback)

    assert first.artifact.to_json() == second.artifact.to_json()
    assert first.physical_calls == second.physical_calls == 3
    assert first.metadata["fresh_session"] is True
    assert first.metadata["artifact_schema"] == "intentir.scrc.artifact.v7"
    assert first.metadata["decode_protocol"] == "intentir.scrc.decode.v11"
    assert first.metadata["typecheck_protocol"] == "intentir.scrc.typecheck.v17"
    assert first.metadata["candidate_check"] == {}
    assert first.metadata["physical_provider_calls"] == 3
    assert first.metadata["transport_retries"] == 0
    assert captured[0][0] != captured[1][0]
    assert captured[0][1] is None
    assert captured[1][1]["failure_kind"] == "gate1"
    assert captured[1][1]["previous_artifact"] == artifact.to_json()
    assert all(not path.exists() for path, _feedback in captured)
    assert len(adapter.records) == 2
    assert adapter.records[0].extraction is not None
    assert adapter.records[0].extraction.provenance.trajectory is not None
    assert "secret" not in json.dumps(
        adapter.records[0].extraction.provenance.trajectory
    )


def test_worker_limit_carries_last_bounded_diagnostics_to_next_fresh_lift(
    monkeypatch,
) -> None:
    artifact = make_copy_artifact()
    captured_feedback: list[object] = []
    runs = 0

    def fake_model(**kwargs):
        del kwargs
        return object()

    def fake_run(*, model, workspace, config_overrides, candidate_checker):
        nonlocal runs
        del model, config_overrides, candidate_checker
        runs += 1
        captured_feedback.append(
            json.loads((workspace.input_dir / "feedback.json").read_text())
        )
        if runs == 1:
            trajectory = {
                "messages": [],
                "info": {
                    "candidate_check": {
                        "calls": 8,
                        "evaluations": 7,
                        "limit": 12,
                        "results": [
                            {
                                "protocol": CANDIDATE_CHECK_PROTOCOL,
                                "status": "fail",
                                "trigger": "explicit",
                                "budget_pool": "shared",
                                "evaluated": True,
                                "rule_codes": ["dependent_type_rejected"],
                                "rule_paths": [
                                    {
                                        "rule_code": "dependent_type_rejected",
                                        "path": "pi.instances[4].semantic_ref",
                                    }
                                ],
                                "rule_diagnostics": [
                                    {
                                        "rule_code": "dependent_type_rejected",
                                        "path": "pi.instances[4].semantic_ref",
                                        "observed_refs_or_types": [
                                            "i4",
                                            "detail: loop_carrier route must bind "
                                            "the declared carry result",
                                        ],
                                    }
                                ],
                            },
                            {
                                "protocol": CANDIDATE_CHECK_PROTOCOL,
                                "status": "fail",
                                "trigger": "explicit",
                                "budget_pool": None,
                                "evaluated": False,
                                "rule_codes": ["check_budget_exhausted"],
                                "rule_paths": [
                                    {
                                        "rule_code": "check_budget_exhausted",
                                        "path": "<candidate>",
                                    }
                                ],
                            },
                        ],
                    }
                },
            }
            result = IntentIRLiftRunResult(
                None,
                trajectory,
                "LimitsExceeded",
                24,
                20.0,
            )
            raise IntentIRLiftRunError(
                "worker exhausted its bounded episode",
                result=result,
            )
        return IntentIRLiftRunResult(
            json.dumps(artifact.to_json()),
            {"messages": []},
            "Submitted",
            2,
            10.0,
        )

    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.create_intentir_lift_model", fake_model
    )
    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.run_intentir_lift", fake_run
    )
    adapter = MiniSWELiftAdapter(
        provider=ProviderConfig(
            "test",
            "http://provider.invalid/v1",
            "secret",
            "test-model",
            reasoning_effort="xhigh",
        )
    )
    request = _adapter_request()

    with pytest.raises(CandidateLiftError) as rejected:
        adapter.lift(request, None)
    diagnostics = rejected.value.diagnostics
    repaired = adapter.lift(
        request,
        LiftFeedback(FeedbackKind.LIFT, diagnostics),
    )

    assert repaired.artifact.to_json() == canonicalize_gate1_artifact(
        artifact
    ).to_json()
    assert captured_feedback[0] is None
    assert captured_feedback[1] == {
        "relift_round": 1,
        "failure_kind": "lift",
        "diagnostics": {
            "diagnostic": "worker exhausted its bounded episode",
            "failure_class": "lift_worker_no_candidate",
            "candidate_check_feedback": {
                "protocol": CANDIDATE_CHECK_PROTOCOL,
                "diagnostics": [
                    {
                        "rule_code": "dependent_type_rejected",
                        "path": "pi.instances[4].semantic_ref",
                        "observed_refs_or_types": [
                            "i4",
                            "detail: loop_carrier route must bind the declared "
                            "carry result",
                        ],
                    }
                ],
            },
        },
        "previous_artifact": None,
    }


def test_worker_limit_continues_adjacent_complete_llm_draft(
    monkeypatch,
) -> None:
    artifact = make_copy_artifact()
    draft = artifact.to_json()
    draft["pi"]["instances"][0]["semantic_ref"] = "n99"
    captured_feedback: list[object] = []
    roots: list[Path] = []

    def fake_model(**kwargs):
        del kwargs
        return object()

    def fake_run(*, model, workspace, config_overrides, candidate_checker):
        del model, config_overrides, candidate_checker
        roots.append(workspace.root)
        captured_feedback.append(
            json.loads((workspace.input_dir / "feedback.json").read_text())
        )
        if len(captured_feedback) == 1:
            workspace.candidate_path.write_text(json.dumps(draft))
            result = IntentIRLiftRunResult(
                None,
                {"messages": [], "info": {"candidate_check": {}}},
                "LimitsExceeded",
                32,
                20.0,
            )
            raise IntentIRLiftRunError(
                "worker exhausted after writing a complete draft",
                result=result,
            )
        return IntentIRLiftRunResult(
            json.dumps(artifact.to_json()),
            {"messages": []},
            "Submitted",
            2,
            10.0,
        )

    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.create_intentir_lift_model", fake_model
    )
    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.run_intentir_lift", fake_run
    )
    adapter = MiniSWELiftAdapter(
        provider=ProviderConfig(
            "test",
            "http://provider.invalid/v1",
            "secret",
            "test-model",
            reasoning_effort="xhigh",
        )
    )
    request = _adapter_request()

    with pytest.raises(CandidateLiftError) as rejected:
        adapter.lift(request, None)
    repaired = adapter.lift(
        request,
        LiftFeedback(
            FeedbackKind.LIFT,
            rejected.value.diagnostics,
            previous_artifact=artifact,
        ),
    )

    assert repaired.artifact.to_json() == canonicalize_gate1_artifact(
        artifact
    ).to_json()
    assert captured_feedback[1]["previous_artifact"] == draft
    assert captured_feedback[1]["diagnostics"]["continuation_draft"][
        "sha256"
    ]
    assert adapter.records[0].continuation_candidate == draft
    assert all(not root.exists() for root in roots)


def test_parser_rejection_enters_only_next_fresh_worker_with_candidate(
    monkeypatch,
) -> None:
    artifact = make_copy_artifact()
    invalid = json.loads(
        _real_lift_input_files()["knowledge/minimal_example.json"]
    )
    invalid["j"]["signature"]["parameters"][2]["logical_ref"] = "v0"
    captured: list[tuple[Path, object]] = []

    def fake_model(**kwargs):
        del kwargs
        return object()

    def fake_run(*, model, workspace, config_overrides, candidate_checker):
        del model, config_overrides, candidate_checker
        feedback = json.loads(
            (workspace.input_dir / "feedback.json").read_text()
        )
        captured.append((workspace.root, feedback))
        candidate = invalid if len(captured) == 1 else artifact.to_json()
        return IntentIRLiftRunResult(
            json.dumps(candidate),
            {"messages": []},
            "Submitted",
            2,
            10.0,
        )

    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.create_intentir_lift_model", fake_model
    )
    monkeypatch.setattr(
        "minisweagent.run.intentir_lift.run_intentir_lift", fake_run
    )
    adapter = MiniSWELiftAdapter(
        provider=ProviderConfig(
            "test", "http://provider.invalid/v1", "secret", "test-model",
            reasoning_effort="xhigh",
        )
    )
    request = _adapter_request()

    with pytest.raises(CandidateLiftError) as rejected:
        adapter.lift(request, None)
    feedback = LiftFeedback(
        FeedbackKind.LIFT,
        rejected.value.diagnostics,
        None,
        artifact,
    )
    repaired = adapter.lift(request, feedback)

    assert repaired.artifact.to_json() == canonicalize_gate1_artifact(
        artifact
    ).to_json()
    assert captured[0][1] is None
    assert captured[1][1]["failure_kind"] == "lift"
    assert captured[1][1]["previous_artifact"] == invalid
    assert captured[0][0] != captured[1][0]
    assert all(not root.exists() for root, _feedback in captured)


@pytest.mark.parametrize(
    "response_text",
    ["[]", "{not-json", '{"value": NaN}'],
)
def test_non_object_rejection_is_not_reused_as_candidate(
    response_text: str,
) -> None:
    diagnostic = "SchemaError: rejected"
    adapter = MiniSWELiftAdapter()
    adapter._records.append(
        SCRCReliftRecord(
            1,
            diagnostic=diagnostic,
            rejected_attempts=(
                SCRCLiftAttempt("test", diagnostic, response_text),
            ),
        )
    )
    previous = make_copy_artifact()
    context = adapter._relift_context(
        LiftFeedback(
            FeedbackKind.LIFT,
            {
                "failure_class": "lift_response",
                "diagnostic": diagnostic,
            },
            None,
            previous,
        )
    )

    assert context is not None
    assert context.previous_artifact == previous.to_json()


def test_rejected_candidate_state_is_adapter_local_and_adjacent() -> None:
    diagnostic = "SchemaError: rejected"
    rejected = SCRCLiftAttempt("test", diagnostic, '{"candidate": 1}')
    first = MiniSWELiftAdapter()
    first._records.append(
        SCRCReliftRecord(
            1,
            diagnostic=diagnostic,
            rejected_attempts=(rejected,),
        )
    )
    first._records.append(
        SCRCReliftRecord(2, diagnostic="worker produced no candidate")
    )
    feedback = LiftFeedback(
        FeedbackKind.LIFT,
        {
            "failure_class": "lift_response",
            "diagnostic": diagnostic,
        },
    )

    adjacent = first._relift_context(feedback)
    separate = MiniSWELiftAdapter()._relift_context(feedback)

    assert adjacent is not None and adjacent.previous_artifact is None
    assert separate is not None and separate.previous_artifact is None
