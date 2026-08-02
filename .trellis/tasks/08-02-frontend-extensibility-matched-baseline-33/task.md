# Task 33：前端扩展性 semantic-SLoC 与 matched baseline 最终实验

> 状态：`READY`（2026-08-02）  
> 论文设计：`/home/kingdom/phdworks/papers/intentir-ppopp27/experiments/FRONTEND_EXTENSIBILITY_MATCHED_BASELINE_PLAN_20260802.md`  
> 已冻结历史证据：paper campaign `20260802_claim1_tilelang_cuda_reuse_jonly_9c437c90`

## 1. 唯一目标

用有限资源补齐 Claim 1 当前唯一关键缺口：

> 在同一 TileLang/CUDA same-origin corpus、同一 J schema 与同一 gates 下，量化 IntentIR 相对于显式
> frontend converter/lifter 的持久 semantic engineering cost；同时分开报告首次 frontend onboarding 与
> 新 kernel 的边际成本。

本 Task 不继续做精度、多 Q、性能或扩大相似 kernel 数量。Task 28--32 只作为冻结输入，不改写历史结果。

## 2. Baselines

1. `B0 IntentIR`：source-aware Codex 作者化 `(J,C,pi)`，fresh J-only Replay，现有真实 gates 与 backend；
2. `B1 Explicit converter`：TileLang/CUDA 各自用 deterministic converter rules 产生同一 exact J，再走
   同一 TypeCheck、Gate 2 与 CUDA/RVV backend；这是主定量 baseline；
3. `B2 Direct-TV`：只验证 Source-to-target observation/refinement edge，是较弱能力下界；资源不足时只做
   预注册 6-case subset，不能静默缩分母或冒充 reusable-J baseline。

禁止用第三方系统总 LOC 做跨项目比例。Alive2/Halide/Tenspiler/TVM 只用于定义 baseline 职责和计量单位。

## 3. 冻结 12-pair 顺序

在任何 B1 implementation 前固定：

1. `add2d`
2. `floor2d`
3. `row_sum`
4. `layernorm`
5. `masked_softmax2d`
6. `gemm`
7. `matmul_fused_epilogue2d`
8. `gather2d`
9. `transpose2d`
10. `flash_attention2d`
11. `any_kernel_dim`
12. `rowmask_where2d`

该顺序覆盖主要语义族以及 Task 29 全部五个 semantic-core non-exact pair。不得读取 baseline 结果后换
case。B1 先做 12 对；通用机制稳定后扩到 Task 28 全 31 对。

## 4. P1：line-level ownership ledger

逐行审计当前 frontend onboarding 与关键增量窗口。每一行只能属于：

- `frontend_semantic`
- `frontend_execution_adapter`
- `observer_semantic`
- `public_context`
- `shared_contract`
- `target_consumer`
- `per_kernel_persistent`
- `declarative_data`
- `artifact_output`
- `test_evidence_harness`
- `docs_generated_vendor`

至少覆盖：

- `d30753e7^..d30753e7` 的联合初次 onboarding；
- 当前 production frontend files 的最终 footprint；
- `b4f1b6da..df95382c` 的 final-five 增量；
- Task 28 四个 manual artifact authoring modules。

输出：

- `semantic_loc_ledger.csv`
- `frontend_onboarding_summary.csv`
- `kernel_authoring_summary.csv`
- `audit_method.md`

raw added/deleted、nonblank/noncomment 与 audited semantic SLoC 必须同时报告；tests/docs/CLI/runner/run
artifacts 不能进入核心 semantic SLoC。

## 5. P2：B1 explicit converter

在独立 clean worktree 中实现通用 source-observation-to-J converter：

- rules 按 source op / typed boundary relation 泛化；
- 可以有共享 shape/type/SSA helper；
- 禁止 kernel id、source path、benchmark family matcher；
- 禁止读取 Task 28 手工 artifact、Replay、Gate diff 或 sibling baseline implementation；
- unsupported/timeout/failure 保留；
- 每新增一个冻结 pair 后提交 checkpoint。

每个 checkpoint 记录：

- cumulative closed unique pairs；
- persistent semantic SLoC added/deleted；
- converter rule count 与新增 capability；
- files touched；
- active authoring time、wall time、Codex turns、人工干预；
- exact J TypeCheck、Gate 2、CUDA/RVV lower；
- kernel-specific branch audit。

B1 的 PASS 必须是 deterministic converter 真实生成 exact J 后的真实 closure，不能复制 B0 J 或用 mock。

## 6. P3：B0 matched re-accounting

对同一 12-pair 顺序重算 B0：

- episode artifact authoring SLoC/bytes；
- persistent production semantic delta；
- generic capability delta；
- agent turns / active time / wall time；
- TypeCheck/Gate1/Gate2/backend closure。

一次性 artifact 与 production mechanism 分列。历史四个 authoring modules 的 2223 raw LoC 只作 upper
bound；正式曲线认 checkpoint 与 ledger。

## 7. P4：B2（可选但边界严格）

冻结前六对，复用相同 source/target observers，显式实现 write coverage、read definedness、value/effect
equality refinement mapping。记录 semantic SLoC/rules/time/coverage。表中必须标注：

- `no reusable J`
- `no RVV reuse`
- weaker capability baseline

若未完成，不影响 B0/B1 主实验；论文以一手文献作定性背景，不编造数值。

## 8. 结果输出

在 paper repository 新建一个新的 fresh campaign，不覆盖历史 campaign。至少包含：

- `normalized/claim1-onboarding.csv`
- `normalized/per_kernel_marginal_cost.csv`
- `normalized/baseline_capability_matrix.csv`
- `normalized/semantic_loc_ledger.csv`
- `derived/paper_table_onboarding.csv`
- `derived/paper_table_kernel_cost.csv`
- `derived/coverage_vs_persistent_sloc.csv`
- raw checkpoints 与 SHA256 receipts。

论文主单位为 31 pair / 62 artifacts。378 J nodes、901 boundary field instances、372 stage events、401
cubin records 只作 hierarchical evidence depth；310 Q 留 appendix。

## 9. 完成条件

- [ ] 12-pair baseline selection 在 implementation 前提交；
- [ ] line ledger 完整且每行 ownership 唯一；
- [ ] B1 12/12 或所有 failure/unsupported 完整保留；
- [ ] B0/B1 同一 J schema、TypeCheck、Gate 2 与 backend；
- [ ] 每个 kernel checkpoint 可由 commit 与 receipt 复核；
- [ ] frontend onboarding 与 new-kernel cost 分开；
- [ ] persistent mechanism 与 episode artifact 分开；
- [ ] tests/evidence/docs/generated data 全部排除或单列；
- [ ] per-kernel branch audit 完成；
- [ ] paper fresh campaign 完整；
- [ ] 不声称 Task 28 是 automatic Lift；
- [ ] source 与 paper worktree 在各自提交后无本 Task 未提交改动。
