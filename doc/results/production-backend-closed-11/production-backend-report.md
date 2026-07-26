# 11 个闭合 Kernel 的 production backend、搜索与性能报告

> 生成日期：2026-07-26；目标：RTX 5090 D (`sm_120a`) 与 H100 80GB HBM3 (`sm_90a`)。

## 结论

11/11 冻结 J+C 快照都完成了 deterministic J materialization、完整 Layer-B ContractView、注册 C++ MLIR whole-graph admission、GPU/NVVM lowering、cubin 执行以及 canonical-J correctness check。Python 只负责快照/搜索编排、CUDA Driver 参数绑定和计时，不生成目标 kernel。

RTX 5090：5/11 达到或超过 Triton，几何平均 MLIR/Triton = 1.024x。H100：5/11，几何平均 = 1.026x。
RVV（VLEN=128）11/11 candidate 与 canonical J 一致；B 条件化 E3 相对同一真实 MLIR 流水线的 E2 schedule-only 几何平均加速为 1.159x。RVV 尚未建立 Triton/source 性能基线，因此该数值只表示机制内消融。

## 性能（μs，括号为 MLIR/Triton；小于 1 更快）

| Kernel | RTX 5090 MLIR / Triton | H100 MLIR / Triton | B facts |
|---|---:|---:|---:|
| dyt | 0.8997 / 0.7886 (1.141x) | 1.2781 / 1.1756 (1.087x) | 15 |
| flash_bwd_preprocess | 0.9460 / 0.8892 (1.064x) | 1.3725 / 1.3021 (1.054x) | 19 |
| fused_add_rms_norm | 0.9992 / 0.8940 (1.118x) | 1.4807 / 1.3526 (1.095x) | 30 |
| geglu | 0.7755 / 0.7703 (1.007x) | 1.1799 / 1.1876 (0.993x) | 11 |
| jsd | 0.8321 / 1.0141 (0.821x) | 1.2927 / 1.4844 (0.871x) | 28 |
| layer_norm | 1.1637 / 0.9766 (1.192x) | 1.7546 / 1.4849 (1.182x) | 29 |
| liger_rms_norm | 0.9802 / 0.8623 (1.137x) | 1.4810 / 1.3110 (1.130x) | 29 |
| poly_norm | 0.9068 / 1.0173 (0.891x) | 1.4127 / 1.5185 (0.930x) | 52 |
| softmax | 0.9905 / 0.9907 (1.000x) | 1.5077 / 1.4728 (1.024x) | 11 |
| swiglu | 0.7589 / 0.7866 (0.965x) | 1.1379 / 1.1645 (0.977x) | 11 |
| xformers_rms_norm | 0.9607 / 0.9701 (0.990x) | 1.4491 / 1.4773 (0.981x) | 18 |

## RVV 真实 MLIR 后端（μs，E3/E2 大于 1 更快）

| Kernel | E2 schedule-only | E3 B-conditioned | E3/E2 | winner RVV 指令数 |
|---|---:|---:|---:|---:|
| dyt | 22.2574 | 20.9886 | 1.060x | 0 |
| flash_bwd_preprocess | 13.4004 | 8.9612 | 1.495x | 10 |
| fused_add_rms_norm | 12.1761 | 11.1988 | 1.087x | 43 |
| geglu | 23.5362 | 21.4118 | 1.099x | 148 |
| jsd | 22.5263 | 19.2994 | 1.167x | 210 |
| layer_norm | 12.3635 | 10.9065 | 1.134x | 54 |
| liger_rms_norm | 10.9646 | 9.0030 | 1.218x | 29 |
| poly_norm | 15.6011 | 10.8003 | 1.445x | 81 |
| softmax | 11.9795 | 11.9107 | 1.006x | 68 |
| swiglu | 13.2708 | 13.2708 | 1.000x | 86 |
| xformers_rms_norm | 6.9920 | 6.1254 | 1.141x | 36 |

## 搜索与消融

- 注册默认 schedule → 完整实测搜索 winner 的几何平均加速为 **1.255x**；10/11 kernel 改善。
- 固定 J=A+B 与 ContractView，E2 仅搜索 schedule，E3 在同一 E2 候选集合上增加已注册的 B 条件化 MLIR 变换。RTX 5090 的全体几何平均 E3/E2 为 **1.033x**，超过 1% 的 2 个 kernel 几何平均为 **1.194x**；H100 分别为 **1.021x** 与 **1.120x**。
- RVV 的同口径 E3/E2 几何平均为 **1.159x**，10/11 winner 使用 B 条件化 whole-graph fusion；这是内部消融，不是与 Source/Triton 的比较。
- budget=3 时，with-C 已覆盖 11/11；相同 candidate latency map 按 Erase_C 顺序重排后，几何平均 best 从 0.9729 增至 1.0466 μs，C 的有限预算效率收益为 **1.076x**。
- 完整 budget=83 时，order-only 两臂均为 0.9224 μs，证明 C 只重排 Visited，不改变 Reach/legality。
- H100 从 bounded 前缀扩到完整搜索后，最大额外收益为 1.159x；PolyNorm 与 JSD 分别为 1.159x、1.069x。
- H100 注册默认 schedule → 全搜索 winner 的可测 kernel 几何平均加速为 1.155x。

B 语义不是可关闭的实验臂：J=A+B。E2/E3 都消费完整 B 并建立 ContractView；未消费的 alias/effect/predicate/order/numeric relation 会在编译前 fail-closed；消融关闭的只是由 B 事实授权的额外 MLIR 变换。编译审计的负对照还会注入未支持的 `may_alias`；11/11 kernel 的 with-C 与 Erase_C 两臂都在 codegen 前被拒绝，C 无权改变 B legality。

## 尚未达到 Triton 的项目

当前不能声称 11/11 全部超越 Triton。主要剩余差距集中在小 shape 的 LayerNorm/RMSNorm/fused-add reduction；绝对差距约 0.1–0.27 μs。Flash preprocess 已通过 rank-0 scalar-by-value MLIR schedule 从旧版 1.35x 缩到 RTX 1.07x / H100 1.06x。后续优化应继续扩展通用 reduction implementation family，而不是按 kernel 名添加 recipe。

## 复现入口

- `tools/benchmark_production_cuda.py`：本地真实编译、正确性、搜索和性能；支持 `--search-prior with-c|without-c`。
- `tools/audit_production_backend_search.py`：B consumption 与 Attach_C/Erase_C 编译消融；精简 receipt 见 [`backend-b-c-audit.json`](backend-b-c-audit.json)。
- `tools/summarize_production_b_search_ablation.py`：GPU E2/E3 B 条件化搜索配对消融；精简 receipt 见 [`b-conditioned-search-ablation.json`](b-conditioned-search-ablation.json)。
- `tools/export_cuda_benchmark_bundle.py` + `tools/run_cuda_benchmark_bundle.py`：本地编译目标 cubin、H100 仅执行。
- `tools/export_rvv_benchmark_bundle.py` + `tools/run_rvv_benchmark_bundle.py`：本地真实 MLIR/LLVM/RVV object 生成，`ssh rvv` 只链接、执行、验正确性和计时；精简 receipt 见 [`rvv-performance.json`](rvv-performance.json)。
- `tools/summarize_production_search_ablation.py`：预算曲线与 order-only 因果对照。
