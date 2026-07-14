"""Run the first real held-out SCRC batch without the legacy corpus verifier.

Every ready case points at an installed, unmodified FlagGems Triton source
function and is executed through the shared live Lift -> Gate1 -> conditional
Gate2 path.  The larger manifest is the reviewed 20--40-kernel queue; cases are
promoted to ``ready`` only after their concrete compiler/execution contract is
recorded here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from intentir.extraction.lifters.scrc import (
    SCRCCompilerArgument,
    scrc_lift_prompt_sha256,
)
from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate2 import Tolerance

from scrc_live_e2e import (
    LiveKernelCase,
    _compile_case,
    _write_report,
    run_case,
)


@dataclass(frozen=True)
class HeldoutCandidate:
    name: str
    module: str
    function: str
    family: str
    readiness: str = "queued"


CANDIDATES = (
    HeldoutCandidate(
        "fg_dot", "flag_gems.ops.dot", "dot_kernel", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_addr",
        "flag_gems.ops.addr",
        "addr_kernel",
        "broadcast",
        "ready",
    ),
    HeldoutCandidate("fg_ones", "flag_gems.ops.ones", "ones_kernel", "fill"),
    HeldoutCandidate("fg_arange", "flag_gems.ops.arange", "arange_func", "iota"),
    HeldoutCandidate("fg_eye", "flag_gems.ops.eye_m", "eye_kernel", "select"),
    HeldoutCandidate(
        "fg_trace", "flag_gems.ops.trace", "trace_kernel", "gather_reduce"
    ),
    HeldoutCandidate(
        "fg_embedding",
        "flag_gems.ops.embedding",
        "embedding_kernel",
        "gather",
    ),
    HeldoutCandidate(
        "fg_count_nonzero",
        "flag_gems.ops.count_nonzero",
        "count_nonzero_kernel_1",
        "atomic_reduce",
    ),
    HeldoutCandidate("fg_amax", "flag_gems.ops.amax", "amax_kernel_1", "reduce"),
    HeldoutCandidate("fg_sum", "flag_gems.ops.sum", "sum_kernel_1", "reduce"),
    HeldoutCandidate("fg_mean", "flag_gems.ops.mean", "mean_kernel_1", "reduce"),
    HeldoutCandidate("fg_max", "flag_gems.ops.max", "max_kernel_1", "reduce"),
    HeldoutCandidate("fg_min", "flag_gems.ops.min", "min_kernel_1", "reduce"),
    HeldoutCandidate("fg_prod", "flag_gems.ops.prod", "prod_kernel_mid", "reduce"),
    HeldoutCandidate(
        "fg_vector_l2",
        "flag_gems.ops.vector_norm",
        "l2_norm_kernel_1",
        "reduce",
    ),
    HeldoutCandidate(
        "fg_vector_l1",
        "flag_gems.ops.vector_norm",
        "l1_norm_kernel_1",
        "reduce",
    ),
    HeldoutCandidate("fg_sort", "flag_gems.ops.sort", "sort_kernel", "sort"),
    HeldoutCandidate(
        "fg_log_softmax",
        "flag_gems.ops.log_softmax",
        "log_softmax_kernel",
        "structured_loop",
    ),
    HeldoutCandidate(
        "fg_fused_add_rms_norm",
        "flag_gems.fused.fused_add_rms_norm",
        "fused_add_rms_norm_kernel",
        "norm",
    ),
    HeldoutCandidate(
        "fg_skip_layer_norm",
        "flag_gems.fused.skip_layernorm",
        "skip_layer_norm_kernel",
        "norm",
    ),
    HeldoutCandidate(
        "fg_masked_select",
        "flag_gems.ops.masked_select",
        "masked_select_single_pass_kernel",
        "scan",
    ),
    HeldoutCandidate(
        "fg_topk_stage1",
        "flag_gems.ops.topk",
        "topk_stage1_kernel",
        "topk",
    ),
    HeldoutCandidate(
        "fg_rotary_embedding",
        "flag_gems.fused.rotary_embedding",
        "apply_rotary_pos_emb_kernel",
        "indexed_elementwise",
    ),
    HeldoutCandidate(
        "fg_cumsum_part",
        "flag_gems.ops.cumsum",
        "scan_part_sum_kernel",
        "scan",
    ),
)

_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}
READY_CASES = tuple(
    candidate.name for candidate in CANDIDATES if candidate.readiness == "ready"
)


def _function(candidate: HeldoutCandidate) -> Any:
    module = importlib.import_module(candidate.module)
    return getattr(module, candidate.function)


def build_heldout_case(name: str) -> LiveKernelCase:
    candidate = _BY_NAME[name]
    if candidate.readiness != "ready":
        raise ValueError(f"held-out case {name!r} has no reviewed compile contract")

    if name == "fg_dot":
        n = 1003
        block = 1024
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument("x_ptr", "tensor", "f32", (n,)),
                SCRCCompilerArgument("y_ptr", "tensor", "f32", (n,)),
                SCRCCompilerArgument("out_ptr", "tensor", "f32", ()),
                SCRCCompilerArgument("N", "scalar", "i32", value=n),
            ),
            (1,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1",),
            ("BLOCK_SIZE",),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )

    if name == "fg_addr":
        m, n = 17, 13
        block_m = block_n = 32
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument("input_ptr", "tensor", "f32", (m, n)),
                SCRCCompilerArgument("vec1_ptr", "tensor", "f32", (m,)),
                SCRCCompilerArgument("vec2_ptr", "tensor", "f32", (n,)),
                SCRCCompilerArgument("output_ptr", "tensor", "f32", (m, n)),
                SCRCCompilerArgument("beta", "scalar", "f32", value=0.75),
                SCRCCompilerArgument("alpha", "scalar", "f32", value=-1.25),
                SCRCCompilerArgument("M", "scalar", "i32", value=m),
                SCRCCompilerArgument("N", "scalar", "i32", value=n),
                SCRCCompilerArgument("stride_input_m", "scalar", "i32", value=n),
                SCRCCompilerArgument("stride_input_n", "scalar", "i32", value=1),
                SCRCCompilerArgument("stride_vec1", "scalar", "i32", value=1),
                SCRCCompilerArgument("stride_vec2", "scalar", "i32", value=1),
                SCRCCompilerArgument("stride_output_m", "scalar", "i32", value=n),
                SCRCCompilerArgument("stride_output_n", "scalar", "i32", value=1),
            ),
            (1, 1),
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            (
                "ceil_div(M, BLOCK_SIZE_M)",
                "ceil_div(N, BLOCK_SIZE_N)",
            ),
            ("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )

    raise AssertionError(name)


def _case_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        effort=args.effort,
        seed=args.seed,
        lift_repairs=args.lift_repairs,
        rewrite_repairs=args.rewrite_repairs,
        diagnose_rejected=args.diagnose_rejected,
    )


def _batch_identity(
    case: LiveKernelCase, args: argparse.Namespace
) -> dict[str, Any]:
    registry = default_triton_rule_registry()
    source_sha = hashlib.sha256(case.source_text.encode("utf-8")).hexdigest()
    return {
        "schema": 1,
        "model": args.model,
        "effort": args.effort,
        "seed": args.seed,
        "lift_repairs": args.lift_repairs,
        "rewrite_repairs": args.rewrite_repairs,
        "diagnose_rejected": bool(args.diagnose_rejected),
        "source_sha256": source_sha,
        "compiler_view": case.compiler_view().to_json(),
        "rule_registry_digest": registry.digest,
        "lift_prompt_sha256": scrc_lift_prompt_sha256(
            case.source_text,
            case.function_name,
            registry=registry,
            compiler_view=case.compiler_view(),
        ),
    }


def _resume_row(
    saved: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any] | None:
    batch = saved.get("batch")
    if not isinstance(batch, dict) or not batch.get("terminal"):
        return None
    if batch.get("identity") != identity:
        return None
    row = batch.get("row")
    return row if isinstance(row, dict) else None


def _result_status(report: dict[str, Any]) -> str:
    if report.get("preflight", {}).get("ok") is False:
        return "source_compile_failed"
    if not report.get("lift", {}).get("ok"):
        return "lift_failed"
    gate1 = report.get("gate1", {}).get("status")
    if gate1 != "replay_closed":
        return gate1 or "gate1_missing"
    return report.get("gate2", {}).get("status") or "gate2_missing"


def _response_ids(report: dict[str, Any]) -> tuple[list[str], str | None]:
    gate1 = report.get("gate1")
    if not isinstance(gate1, dict):
        return [], None
    normal: list[str] = []
    for item in gate1.get("candidates", []):
        generation = item.get("generation") if isinstance(item, dict) else None
        if isinstance(generation, dict) and generation.get("response_id"):
            normal.append(str(generation["response_id"]))
    masked = gate1.get("masked_generation")
    masked_id = (
        str(masked["response_id"])
        if isinstance(masked, dict) and masked.get("response_id")
        else None
    )
    return normal, masked_id


def _load_cached_lift(
    path: Path,
    case: LiveKernelCase,
    args: argparse.Namespace,
) -> tuple[Gate1Artifact, dict[str, Any]]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(case.source_text.encode("utf-8")).hexdigest()
    registry = default_triton_rule_registry()
    if saved.get("model") != args.model or saved.get("effort") != args.effort:
        raise ValueError("cached lift model/effort differs from this batch")
    if saved.get("source", {}).get("sha256") != source_sha:
        raise ValueError("cached lift source hash differs from current source")
    if saved.get("compiler_view") != case.compiler_view().to_json():
        raise ValueError("cached lift compiler view differs from current case")
    lift = saved.get("lift")
    if not isinstance(lift, dict) or not lift.get("ok"):
        raise ValueError("cached report does not contain a successful lift")
    if lift.get("rule_registry_digest") != registry.digest:
        raise ValueError("cached lift rule registry differs from current registry")
    expected_prompt_sha = scrc_lift_prompt_sha256(
        case.source_text,
        case.function_name,
        registry=registry,
        compiler_view=case.compiler_view(),
    )
    if lift.get("prompt_sha256") != expected_prompt_sha:
        raise ValueError("cached lift prompt differs from current Lift prompt")
    artifact_payload = dict(lift["artifact"])
    artifact_payload.pop("source_sha256", None)
    artifact_payload.pop("source_bytes", None)
    artifact_payload["source"] = case.source_text
    return Gate1Artifact.from_json(artifact_payload), lift


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=READY_CASES)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument("--rewrite-repairs", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/scrc_heldout_batch"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lift-cache-dir", type=Path)
    parser.add_argument("--diagnose-rejected", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.list:
        print(json.dumps([asdict(item) for item in CANDIDATES], indent=2))
        return 0

    selected = tuple(args.case or READY_CASES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    infrastructure_failed = False
    for name in selected:
        path = args.out_dir / f"{name}.json"
        print(json.dumps({"case": name, "stage": "start"}), flush=True)
        try:
            case = build_heldout_case(name)
            identity = _batch_identity(case, args)
            if args.resume and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                resumed = _resume_row(saved, identity)
                if resumed is not None:
                    rows.append(resumed)
                    print(json.dumps({"case": name, "stage": "resumed"}))
                    continue
                print(
                    json.dumps({"case": name, "stage": "resume_stale"}),
                    flush=True,
                )
            compile_case = replace(
                _compile_case(case), schedule=case.schedule_values
            )
            preflight = TritonCompiler().compile(
                case.kernel_source, compile_case
            )
            if not preflight.ok:
                code = 1
                report = {
                    "case": name,
                    "model": args.model,
                    "effort": args.effort,
                    "source": {
                        "path": str(case.source_path),
                        "function": case.function_name,
                    },
                    "preflight": {
                        "ok": False,
                        "diagnostic": preflight.diagnostic,
                    },
                }
            else:
                cached_artifact = None
                cached_lift = None
                cached_origin = None
                if args.lift_cache_dir is not None:
                    cached_path = args.lift_cache_dir / f"{name}.json"
                    if cached_path.exists():
                        cached_artifact, cached_lift = _load_cached_lift(
                            cached_path, case, args
                        )
                        cached_origin = str(cached_path)
                code, report = run_case(
                    case,
                    _case_args(args),
                    artifact=cached_artifact,
                    lift_evidence=cached_lift,
                    lift_origin=cached_origin,
                )
                report["preflight"] = {
                    "ok": True,
                    "cache_hit": preflight.cache_hit,
                    "elapsed_ms": preflight.elapsed_ms,
                }
            status = _result_status(report)
            normal_ids, masked_id = _response_ids(report)
            row = {
                "case": name,
                "status": status,
                "individual_exit_code": code,
                "report": str(path),
                "lift_response_id": report.get("lift", {}).get("response_id"),
                "normal_replay_response_ids": normal_ids,
                "masked_response_id": masked_id,
            }
            report["batch"] = {
                "terminal": True,
                "identity": identity,
                "row": row,
            }
        except Exception as exc:  # batch records infrastructure failure and continues
            infrastructure_failed = True
            row = {
                "case": name,
                "status": "harness_error",
                "individual_exit_code": 5,
                "report": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            report = {
                "case": name,
                "model": args.model,
                "effort": args.effort,
                "batch": {"terminal": True, "row": row},
            }
        _write_report(str(path), report)
        rows.append(row)
        print(
            json.dumps(
                {"case": name, "stage": "complete", "status": row["status"]}
            ),
            flush=True,
        )

    counts = Counter(row["status"] for row in rows)
    summary = {
        "model": args.model,
        "effort": args.effort,
        "selected": list(selected),
        "counts": dict(sorted(counts.items())),
        "results": rows,
    }
    _write_report(str(args.out_dir / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if infrastructure_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
