# SCRC 历史实验归档

本目录保存已退出当前验证主线的手工 probe、strict-v5 能力实验和旧 live 报告。

这些文件仍可用于复现历史实验或提供单元测试 fixture，但不得计入当前
`j_complete_deterministic_pi_relift_v8` 的正式 Gate1/Gate2 证据。当前真实入口是：

- `test/scrc_live_e2e.py`；
- `test/scrc_heldout_batch.py`；
- `python/intentir/verify/gate1/`；
- `python/intentir/verify/gate2/`。

归档内容：

- `scrc_gate1_capability_v5.py`：旧五核手工 mutant/两方动态代理实验；
- `scrc_probe_compile_v0.py`、`scrc_probe_compare_v0.py`：最初人工 TTIR probe；
- `scrc_probe_inputs_v0.json`、`scrc_probe_replays_v0.py`：旧手工输入与 GPT-5.5 replay；
- `scrc_gate2_three_way_tests_v1.py`、`scrc_gate2_cuda_lowering_tests_v1.py`：
  旧 source/replay/production-lowering 三方 Gate2 与 CUDA adapter 测试；
- `high_performance_live_gate_cuda_benchmark_v1.py` 及对应 tests：旧 CUDA
  production-lowering benchmark adapter。完整 CLI 依赖 2026-07-20 前的 live
  helper contract，应从 commit `c59bd69` 复现；当前文件只作源码审计快照；
- `*_report_*.md`：对应历史结果快照。

归档实现不得被 production Gate1/Gate2 作为备用裁决器。若当前单元测试使用其中的
fixture，必须显式引用本目录，避免把它误认为 fresh LLM 提取证据。
