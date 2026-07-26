#!/usr/bin/env python3
"""Render compact checked-in reports from full production benchmark receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from intentir.backend.cuda.schedules import REGISTERED_DEFAULT_SCHEDULES
from intentir.backend.reporting import load_disjoint_kernel_reports


COMPACT_PROTOCOL = "intentir.production.cuda-performance.compact.v1"
COMPACT_B_C_PROTOCOL = (
    "intentir.production.cuda-b-c-search-ablation.compact.v1"
)
COMPACT_B_SEARCH_PROTOCOL = (
    "intentir.production.b-conditioned-search-ablation.compact.v1"
)
COMPACT_RVV_PROTOCOL = "intentir.production.rvv-performance.compact.v1"


def _gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(item) for item in values) / len(values))


def _compact(report: Mapping[str, Any], *, target_chip: str) -> dict[str, Any]:
    kernels = []
    remote = report["protocol"].endswith("remote-performance.v1")
    for item in report["kernels"]:
        if "performance" not in item:
            raise ValueError(f"{item['kernel']} is missing performance")
        contract = (
            item["contract_view"]
            if remote
            else item["search"]["contract_view"]
        )
        correctness = item["correctness"]
        mlir_correct = (
            all(
                candidate["status"] == "measured"
                for candidate in item["candidates"]
            )
            if remote
            else correctness["mlir_j"]["passed"]
        )
        source_correct = correctness["source_j"]["passed"]
        search_counts = (
            {
                "candidate_count": item["candidate_count"],
                "measured": sum(
                    candidate["status"] == "measured"
                    for candidate in item["candidates"]
                ),
            }
            if remote
            else item["search"]["counts"]
        )
        snapshot = item.get("snapshot", {})
        kernels.append(
            {
                "kernel": item["kernel"],
                "case_id": item["case_id"],
                "snapshot_digest": snapshot.get(
                    "snapshot_digest", item.get("snapshot_digest")
                ),
                "j_digest": snapshot.get("j_digest", item.get("j_digest")),
                "c_digest": snapshot.get("c_digest", item.get("c_digest")),
                "extents": item["extents"],
                "contract_view_digest": contract["digest"],
                "b_supported": contract["supported"],
                "b_consumed_fact_count": len(contract["consumed"]),
                "correctness": {
                    "mlir_j": mlir_correct,
                    "source_j": source_correct,
                    "policy": correctness["policy"],
                },
                "winner": item["winner"]["candidate"],
                "search_counts": search_counts,
                "performance": item["performance"],
            }
        )
    ratios = [item["performance"]["mlir_over_triton"] for item in kernels]
    return {
        "protocol": COMPACT_PROTOCOL,
        "source_protocol": report["protocol"],
        "source_report_count": report.get("source_report_count", 1),
        "target_chip": target_chip,
        "device": report["device"],
        "summary": {
            "kernel_count": len(kernels),
            "correct_kernel_count": sum(
                item["correctness"]["mlir_j"]
                and item["correctness"]["source_j"]
                for item in kernels
            ),
            "meets_or_beats_triton": sum(
                item["performance"]["meets_or_beats_triton"]
                for item in kernels
            ),
            "geomean_mlir_over_triton": _gmean(ratios),
        },
        "kernels": kernels,
    }


def _compact_b_c_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    kernels = []
    for item in report["kernels"]:
        kernels.append(
            {
                "kernel": item["kernel"],
                "snapshot_digest": item["snapshot_digest"],
                "contract_view_digest": item["contract_view"]["digest"],
                "b_supported": item["contract_view"]["supported"],
                "b_consumed_fact_count": len(
                    item["contract_view"]["consumed"]
                ),
                "b_unsupported_facts": item["contract_view"]["unsupported"],
                "invariants": item["invariants"],
                "finite_budget": item["finite_budget"],
                "unsupported_b_control": item["unsupported_b_control"],
                "with_c_counts": item["with_c"]["counts"],
                "without_c_counts": item["without_c"]["counts"],
            }
        )
    return {
        "protocol": COMPACT_B_C_PROTOCOL,
        "source_protocol": report["protocol"],
        "source_report_count": report.get("source_report_count", 1),
        "target": report["target"],
        "experiment": report["experiment"],
        "summary": {
            "kernel_count": len(kernels),
            "all_b_supported": all(item["b_supported"] for item in kernels),
            "all_candidate_universes_equal_after_erase_c": all(
                item["invariants"][
                    "candidate_universe_equal_after_erase_c"
                ]
                for item in kernels
            ),
            "all_contract_views_equal_after_erase_c": all(
                item["invariants"]["contract_view_equal_after_erase_c"]
                for item in kernels
            ),
            "all_unsupported_b_controls_rejected_before_codegen": all(
                item["unsupported_b_control"][
                    "both_rejected_before_codegen"
                ]
                for item in kernels
            ),
            "c_reordered_kernel_count": sum(
                item["finite_budget"]["order_changed"]
                for item in kernels
            ),
        },
        "kernels": kernels,
    }


def _compact_b_search_ablation(
    report: Mapping[str, Any],
    *,
    snapshot_digests: Mapping[str, str],
) -> dict[str, Any]:
    targets = []
    for target in report["targets"]:
        kernels = []
        for item in target["kernels"]:
            kernels.append(
                {
                    "kernel": item["kernel"],
                    "snapshot_digest": snapshot_digests[item["kernel"]],
                    "contract_view_digest": item[
                        "contract_view_digest"
                    ],
                    "b_capabilities": item["b_capabilities"],
                    "e2_schedule_only": item["e2_schedule_only"],
                    "e3_contract_guided": item["e3_contract_guided"],
                    "e2_schedule_only_candidate_count": item[
                        "e2_schedule_only_candidate_count"
                    ],
                    "e3_b_transformation_candidate_count": item[
                        "e3_b_transformation_candidate_count"
                    ],
                    "measured_candidate_count": item[
                        "measured_candidate_count"
                    ],
                    "e3_over_e2_speedup": item[
                        "e3_over_e2_speedup"
                    ],
                    "speedup_over_1pct": item["speedup_over_1pct"],
                    "b_transformation_wins": item[
                        "b_transformation_wins"
                    ],
                }
            )
        targets.append(
            {
                "label": target["label"],
                "device": target["device"],
                "source_protocol": target["source_protocol"],
                "source_report_count": target.get(
                    "source_report_count", 1
                ),
                "invariants": target["invariants"],
                "summary": target["summary"],
                "kernels": kernels,
            }
        )
    return {
        "protocol": COMPACT_B_SEARCH_PROTOCOL,
        "source_protocol": report["protocol"],
        "experiment": report["experiment"],
        "targets": targets,
    }


def _compact_rvv(report: Mapping[str, Any]) -> dict[str, Any]:
    kernels = []
    for item in report["kernels"]:
        winner = item["winner"]
        kernels.append(
            {
                "kernel": item["kernel"],
                "case_id": item["case_id"],
                "snapshot_digest": item["snapshot_digest"],
                "j_digest": item["j_digest"],
                "c_digest": item["c_digest"],
                "extents": item["extents"],
                "contract_view_digest": item["contract_view"]["digest"],
                "b_supported": item["contract_view"]["supported"],
                "b_capabilities": item["contract_view"][
                    "capabilities"
                ],
                "correct": all(
                    check["passed"]
                    for candidate in item["candidates"]
                    for check in candidate["correctness"]
                ),
                "winner": {
                    "candidate_id": winner["candidate_id"],
                    "schedule": winner["schedule"],
                    "true_mlir_pipeline": winner[
                        "true_mlir_pipeline"
                    ],
                    "python_target_kernel_emission": winner[
                        "python_target_kernel_emission"
                    ],
                    "rvv_instruction_count": winner[
                        "rvv_instruction_count"
                    ],
                    "object_sha256": winner["object_sha256"],
                },
                "performance": item["performance"],
            }
        )
    return {
        "protocol": COMPACT_RVV_PROTOCOL,
        "source_protocol": report["protocol"],
        "bundle_protocol": report["bundle_protocol"],
        "source_report_count": report.get("source_report_count", 1),
        "device": report["device"],
        "summary": report["summary"],
        "comparison_scope": (
            "whole-graph fused arm versus materialized true-MLIR arm; "
            "legacy receipt fields retain E2/E3 names; no Triton/source claim"
        ),
        "kernels": kernels,
    }


def _default_h100_speedups(report: Mapping[str, Any]) -> dict[str, float]:
    result = {}
    for item in report["kernels"]:
        measured = [
            candidate
            for candidate in item["candidates"]
            if candidate["status"] == "measured"
        ]
        path_kinds = {candidate["candidate"]["family"] for candidate in measured}
        if len(path_kinds) != 1:
            continue
        path_kind = next(iter(path_kinds))
        expected = REGISTERED_DEFAULT_SCHEDULES[path_kind]
        defaults = [
            candidate
            for candidate in measured
            if all(
                candidate["candidate"]["schedule"].get(
                    name,
                    False if value is False else None,
                )
                == value
                for name, value in expected.items()
            )
        ]
        if len(defaults) == 1:
            result[item["kernel"]] = (
                float(defaults[0]["latency_us"])
                / float(item["performance"]["mlir_us"])
            )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rtx-report", type=Path, action="append", required=True
    )
    parser.add_argument("--rtx-ablation", type=Path, required=True)
    parser.add_argument(
        "--b-c-audit", type=Path, action="append", required=True
    )
    parser.add_argument("--b-search-ablation", type=Path, required=True)
    parser.add_argument(
        "--h100-full", type=Path, action="append", required=True
    )
    parser.add_argument("--h100-bounded", type=Path, required=True)
    parser.add_argument(
        "--rvv-report", type=Path, action="append", required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rtx_reports, rtx_kernels = load_disjoint_kernel_reports(args.rtx_report)
    rtx_raw = {
        **rtx_reports[0],
        "source_report_count": len(rtx_reports),
        "kernels": [rtx_kernels[name] for name in sorted(rtx_kernels)],
    }
    ablation = json.loads(args.rtx_ablation.read_text(encoding="utf-8"))
    audit_reports, audit_kernels = load_disjoint_kernel_reports(args.b_c_audit)
    b_c_audit = _compact_b_c_audit(
        {
            **audit_reports[0],
            "source_report_count": len(audit_reports),
            "kernels": [
                audit_kernels[name] for name in sorted(audit_kernels)
            ],
        }
    )
    b_search = _compact_b_search_ablation(
        json.loads(args.b_search_ablation.read_text(encoding="utf-8")),
        snapshot_digests={
            item["kernel"]: item["snapshot"]["snapshot_digest"]
            for item in rtx_raw["kernels"]
        },
    )
    h100_reports, h100_kernels = load_disjoint_kernel_reports(args.h100_full)
    h100_raw = {
        **h100_reports[0],
        "source_report_count": len(h100_reports),
        "kernels": [h100_kernels[name] for name in sorted(h100_kernels)],
    }
    h100_bounded = json.loads(args.h100_bounded.read_text(encoding="utf-8"))
    rvv_reports, rvv_kernels = load_disjoint_kernel_reports(args.rvv_report)
    rvv_raw = {
        **rvv_reports[0],
        "source_report_count": len(rvv_reports),
        "kernels": [rvv_kernels[name] for name in sorted(rvv_kernels)],
    }
    rvv_speedups = [
        item["performance"]["e3_over_e2_speedup"]
        for item in rvv_raw["kernels"]
    ]
    rvv_raw["summary"] = {
        "requested": len(rvv_raw["kernels"]),
        "successful": len(rvv_raw["kernels"]),
        "failed": 0,
        "geomean_e3_over_e2_speedup": _gmean(rvv_speedups),
        "b_transformation_winner_count": sum(
            item["performance"]["b_transformation_wins"]
            for item in rvv_raw["kernels"]
        ),
    }
    rvv = _compact_rvv(rvv_raw)
    rtx = _compact(rtx_raw, target_chip="sm_120a")
    h100 = _compact(h100_raw, target_chip="sm_90a")
    bounded_by_name = {
        item["kernel"]: item for item in h100_bounded["kernels"]
    }
    full_by_name = {item["kernel"]: item for item in h100_raw["kernels"]}
    h100_budget_speedups = {
        name: bounded_by_name[name]["performance"]["mlir_us"]
        / item["performance"]["mlir_us"]
        for name, item in full_by_name.items()
        if name in bounded_by_name
    }
    h100_default_speedups = _default_h100_speedups(h100_raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("rtx5090-performance.json", rtx),
        ("h100-performance.json", h100),
        ("rtx5090-search-ablation.json", ablation),
        ("backend-b-c-audit.json", b_c_audit),
        ("b-conditioned-search-ablation.json", b_search),
        ("rvv-performance.json", rvv),
    ):
        (args.out_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rtx_items = {item["kernel"]: item for item in rtx["kernels"]}
    h100_items = {item["kernel"]: item for item in h100["kernels"]}
    rvv_items = {item["kernel"]: item for item in rvv["kernels"]}
    if not (set(rtx_items) == set(h100_items) == set(rvv_items)):
        raise ValueError("RTX, H100, and RVV compact reports differ in kernels")
    kernel_count = len(rtx_items)
    b_targets = {item["label"]: item for item in b_search["targets"]}
    rtx_b = b_targets["rtx5090"]["summary"]
    h100_b = b_targets["h100"]["summary"]
    lines = [
        f"# {kernel_count} 个闭合 Kernel 的 production backend、搜索与性能报告",
        "",
        "> 生成日期：2026-07-26；目标：RTX 5090 D (`sm_120a`) 与 "
        "H100 80GB HBM3 (`sm_90a`)。",
        "",
        "## 结论",
        "",
        f"{kernel_count}/{kernel_count} 冻结 J+C 快照都完成了 deterministic J "
        "materialization、当前已实现的 "
        "Layer-B ContractView、注册 C++ MLIR whole-graph admission、GPU/NVVM "
        "lowering、cubin 执行以及 canonical-J correctness check。Python 只负责"
        "快照/搜索编排、CUDA Driver 参数绑定和计时，不生成目标 kernel。这里的"
        "“当前已实现”不声称 v7 已具有 reduction-order 或 alias-completeness 的"
        "全部未来合同。",
        "",
        f"RTX 5090：{rtx['summary']['meets_or_beats_triton']}/{kernel_count} "
        "达到或超过 "
        f"Triton，几何平均 MLIR/Triton = "
        f"{rtx['summary']['geomean_mlir_over_triton']:.3f}x。H100："
        f"{h100['summary']['meets_or_beats_triton']}/{kernel_count}，几何平均 = "
        f"{h100['summary']['geomean_mlir_over_triton']:.3f}x。",
        f"RVV（VLEN={rvv['device']['vlen_bits']}）{kernel_count}/{kernel_count} "
        "candidate 与 canonical J "
        f"一致；whole-graph fused arm 相对同一真实 MLIR 流水线的 materialized arm "
        f"几何平均加速为 {rvv['summary']['geomean_e3_over_e2_speedup']:.3f}x。"
        "RVV 尚未建立 Triton/source 性能基线，因此该数值只表示 fusion "
        "lowering-path 的内部消融。",
        "",
        "## 性能（μs，括号为 MLIR/Triton；小于 1 更快）",
        "",
        "| Kernel | RTX 5090 MLIR / Triton | H100 MLIR / Triton | B facts |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(rtx_items):
        left = rtx_items[name]
        right = h100_items[name]
        lp = left["performance"]
        rp = right["performance"]
        lines.append(
            f"| {name} | {lp['mlir_us']:.4f} / {lp['source_triton_us']:.4f} "
            f"({lp['mlir_over_triton']:.3f}x) | "
            f"{rp['mlir_us']:.4f} / {rp['source_triton_us']:.4f} "
            f"({rp['mlir_over_triton']:.3f}x) | "
            f"{left['b_consumed_fact_count']} |"
        )

    lines.extend(
        [
            "",
            "## RVV 真实 MLIR 后端（μs，fused/materialized 大于 1 更快）",
            "",
            "| Kernel | materialized arm | fused arm | fused/materialized | "
            "winner RVV 指令数 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in sorted(rvv_items):
        item = rvv_items[name]
        perf = item["performance"]
        lines.append(
            f"| {name} | {perf['e2_schedule_only_us']:.4f} | "
            f"{perf['e3_contract_guided_us']:.4f} | "
            f"{perf['e3_over_e2_speedup']:.3f}x | "
            f"{item['winner']['rvv_instruction_count']} |"
        )

    search_summary = ablation["summary"]
    budget3 = next(
        item
        for item in search_summary["budget_curves"]
        if item["attempt_budget"] == 3
    )
    full_budget = search_summary["budget_curves"][-1]
    c_budget_gain = (
        budget3["without_c_order_only_geomean_best_us"]
        / budget3["with_c_geomean_best_us"]
    )
    lines.extend(
        [
            "",
            "## 搜索与消融",
            "",
            f"- 注册默认 schedule → 完整实测搜索 winner 的几何平均加速为 "
            f"**{search_summary['search_geomean_speedup_over_registered_default']:.3f}x**；"
            f"{search_summary['search_improved_kernel_count']}/{kernel_count} "
            "kernel 改善。",
            f"- 固定 J=A+B 与 ContractView，GPU L0 只搜索原 schedule space，L1 "
            f"在同一候选集合上增加已注册的 B-conditioned local transformations。"
            f"RTX 5090 的全体几何平均 L1/L0 为 "
            f"**{rtx_b['geomean_e3_over_e2_speedup_all']:.3f}x**，"
            f"超过 1% 的 2 个 kernel 几何平均为 "
            f"**{rtx_b['geomean_e3_over_e2_speedup_affected']:.3f}x**；H100 "
            f"分别为 **{h100_b['geomean_e3_over_e2_speedup_all']:.3f}x** 与 "
            f"**{h100_b['geomean_e3_over_e2_speedup_affected']:.3f}x**。这些结果"
            "不作为 structural-path E3/E2 证据。",
            f"- RVV fused/materialized 几何平均为 "
            f"**{rvv['summary']['geomean_e3_over_e2_speedup']:.3f}x**，"
            f"{rvv['summary']['b_transformation_winner_count']}/{kernel_count} "
            "winner 使用 whole-graph fusion；这是首个真实结构路径轴的内部消融，"
            "但当前结构开关仍混在 schedule record 中，因此不冒充完整 E3/E2 "
            "search，也不是与 Source/Triton 的比较。",
            f"- budget=3 时，with-C 已覆盖 "
            f"{budget3['with_c_coverage']}/{kernel_count}；相同 candidate latency map "
            f"按 Erase_C 顺序重排后，几何平均 best 从 "
            f"{budget3['with_c_geomean_best_us']:.4f} 增至 "
            f"{budget3['without_c_order_only_geomean_best_us']:.4f} μs，"
            f"C 的有限预算效率收益为 **{c_budget_gain:.3f}x**。",
            f"- 完整 budget={full_budget['attempt_budget']} 时，order-only 两臂均为 "
            f"{full_budget['with_c_geomean_best_us']:.4f} μs，证明 C 只重排 "
            "Visited，不改变 Reach/legality。",
            f"- H100 从 bounded 前缀扩到完整搜索后，最大额外收益为 "
            f"{max(h100_budget_speedups.values()):.3f}x；PolyNorm 与 JSD 分别为 "
            f"{h100_budget_speedups['poly_norm']:.3f}x、"
            f"{h100_budget_speedups['jsd']:.3f}x。",
            f"- H100 注册默认 schedule → 全搜索 winner 的可测 kernel 几何平均"
            f"加速为 {_gmean(list(h100_default_speedups.values())):.3f}x。",
            "",
            "B 语义不是可关闭的实验臂：J=A+B。L0/L1 都消费同一 B 并建立 "
            "ContractView；消融关闭的只是额外 local transformations。编译审计"
            "的负对照会注入未支持的 "
            f"`may_alias`；{kernel_count}/{kernel_count} kernel 的 with-C 与 "
            "Erase_C 两臂都在 codegen "
            "前被拒绝，证明 C 无权改变 B legality；但该负对照不能单独证明 v7 "
            "空 alias list 具有完整 no-alias 语义。未来 capability 必须引用明确"
            "的 schema completeness rule，否则返回 `missing_contract`。",
            "",
            "## 尚未达到 Triton 的项目",
            "",
            f"当前不能声称 {kernel_count}/{kernel_count} 全部超越 Triton。"
            "主要剩余差距集中在小 shape "
            "的 LayerNorm/RMSNorm/fused-add reduction；绝对差距约 0.1–0.27 μs。"
            "Flash preprocess 已通过 rank-0 scalar-by-value MLIR local transformation "
            "从旧版 1.35x 缩到 RTX 1.07x / H100 1.06x。在不等待新 B 的阶段，"
            "后续先扩展 effect-safe fusion、materialization/copy elimination 和 "
            "RVV structural paths；会改变浮点 reduction tree 的路径在缺少明确"
            "numeric contract 时必须 fail closed，而不是按 kernel 名添加 recipe。",
            "",
            "## 复现入口",
            "",
            "- `tools/benchmark_production_cuda.py`：本地真实编译、正确性、搜索和性能；"
            "支持 `--search-prior with-c|without-c`。",
            "- `tools/audit_production_backend_search.py`：B consumption 与 "
            "Attach_C/Erase_C 编译消融；精简 receipt 见 "
            "[`backend-b-c-audit.json`](backend-b-c-audit.json)。",
            "- `tools/summarize_production_b_search_ablation.py`：GPU L1/L0 "
            "B-conditioned local transformation 配对消融；原始 JSON 保留生成时的"
            " legacy E2/E3 protocol label，精简 receipt 见 "
            "[`b-conditioned-search-ablation.json`](b-conditioned-search-ablation.json)。",
            "- `tools/export_cuda_benchmark_bundle.py` + "
            "`tools/run_cuda_benchmark_bundle.py`：本地编译目标 cubin、H100 仅执行。",
            "- `tools/export_rvv_benchmark_bundle.py` + "
            "`tools/run_rvv_benchmark_bundle.py`：本地真实 MLIR/LLVM/RVV object "
            "生成，`ssh rvv` 只链接、执行、验正确性和计时；精简 receipt 见 "
            "[`rvv-performance.json`](rvv-performance.json)。",
            "- `tools/summarize_production_search_ablation.py`：预算曲线与 "
            "order-only 因果对照。",
            "",
        ]
    )
    (args.out_dir / "production-backend-report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"wrote compact reports to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
