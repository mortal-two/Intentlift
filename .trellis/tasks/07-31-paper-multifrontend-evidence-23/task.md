# Task 23：论文多前端接入与可复用 Lowering 最小实证

> 状态：执行中  
> 日期：2026-07-31  
> 分支：`codex/frontend-extension-20260726`  
> frontend 基线：`e77fe1d4`  
> backend 合并：`0599a252`（输入提交 `928502d3`）

## 1. 唯一目标

用最小时间与模型预算，为论文提供真实、可追溯、可写入正文的证据，支持以下主张：

> IntentIR 将新前端的碎片化成本局部化到 Source→J 接入，并让既有 J feature 的 backend consumer 跨前端复用；新 J feature 的成本按 feature 支付一次，而不是按 frontend×backend×kernel 重复支付。

本任务服务论文数据，不以继续完善整个项目或全量闭合所有多前端 Kernel 为目标。

## 2. 真实性规则

- 论文结果只能来自可重放命令、Git 对象、保存的 provider receipt 或确定性程序输出。
- 估计、预期和示意数字必须显式标为 `estimated` / `illustrative`，不得进入结果列。
- registry/prepare/compile 不得写成 Gate1/Gate2/SCRC closure。
- 公开 Source-vs-J probe 不得写成 certificate。
- TileLang/CUDA 配对不要求 full-J equality；两边必须独立闭合，A/B 比较只做事后描述与差异归因。
- 负结果、unsupported 和中断运行保留在分母与 failure taxonomy 中。

## 3. 最小 baseline 决策

### 3.1 主定量 baseline

只使用同代码库、可重放的工程增量：

1. Triton real-workload checkpoint：`0c6e4704`；
2. TileLang/CUDA frontend capability：`d30753e7`；
3. Dataset closure follow-up：`eb58f3ae`；
4. frontend canary integration：`a21f90f2`。

这些 commit 按文件 ownership 分类，不把混入的 tests、docs、generated artifacts 或无关 shared-core 改动计入 headline LoC。不能把 `0c6e4704..d30753e7` 整段历史直接称为 frontend-only delta。

### 3.2 不执行的 baseline

- 不临时构造一个故意弱的 rule-based lifter；当前仓库没有可运行的 PACT/CanonicalEvidence baseline。
- 不做 8–12 个完整 manual recovery；时间不足，且人工时间无法历史复原。
- 若能在一次完整留痕会话内完成，可补 1 个 manual artifact 作为描述性规模参照；否则删除该行，不编造人工时间。
- LLMLift、Tenspiler 与传统 verified lifting 只作为 related-work 工程单位参照，不与本实验 LoC 做倍数比较。

## 4. 正文只保留的证据

### Table A：Frontend onboarding and coverage

只填以下真实字段：

- raw catalog cases；
- runnable projection / unavailable；
- frontend-specific production additions/deletions/files；
- shared-core additions/deletions/files；
- backend delta；
- per-kernel branches/rules；
- compile probe pass/fail。

### Table B：Paired closure, layered differences, and reuse

预注册 cohort：

1. `add2d`：elementwise；
2. `row_sum`：reduction；
3. `gather2d`：irregular access。

每组 TileLang/CUDA 各一个 fresh Codex episode，最多 6 个。记录 CandidateCheck、TypeCheck、normal-J Gate1、Gate2、Layer-A correspondence、Layer-B/ABI differences、Source-grounded attribution 与 backend lower 状态。

## 5. 执行阶段

### P0：无 LLM 基础数据

- [ ] 保存 `frontend_dataset.py --validate` 原始 JSON；
- [ ] 保存 `scrc_frontend.py --list --json` 原始 JSON；
- [ ] 生成 catalog/runnable/unavailable/pair counts 与 failure taxonomy；
- [ ] 生成 onboarding commit-level LoC ownership 表；
- [ ] 保存命令、环境、commit 与 SHA-256 receipt。

### P1：真实 Source compile probe

- [ ] TileLang catalog compile probe；
- [ ] CUDA catalog compile probe；
- [ ] 报告 pass/fail/unsupported，不把 compile 写成 closure。

### P2：最多六次 fresh closure

- [ ] 冻结 provider/model/effort/timeout；
- [ ] 每个 frontend/case 只运行一个正式 normal-J episode；
- [ ] 共享机制错误最多修一次机制后重跑原 case；
- [ ] 禁止按 kernel 名、路径或 case-id 添加 production 分支。

### P3：source-independent J consumer

- [ ] 对闭合 J 运行 shared J→Intent MLIR materialization；
- [ ] 验证 backend 不读取 frontend、Source、π 或 case id；
- [ ] 对至少 elementwise 与 reduction 各保存一个合法 lower receipt；
- [ ] 新 J feature 只在真实出现且能跨 artifact 复用时选择一个，不为凑表发明 feature。

### P4：论文交付

- [ ] 填 Table A/B；
- [ ] 写一段 failure taxonomy；
- [ ] 将复现命令与完整结果放入 artifact/doc，正文只保留 4–6 个 headline 数字；
- [ ] 对所有数字执行来源审计。

## 6. 预算与停止条件

- 无 LLM 审计优先完成；其结果本身必须足以形成一张 onboarding 表。
- LLM 物理调用硬上限：6 个 fresh episode；未经明确决策不扩展 corpus。
- 不建设新 baseline 系统，不跑完整模型 sweep，不重复 masked-J authority 实验。
- 若 6 个 episode 中闭合很少，如实报告并把主结果收缩到工程 locality、compile coverage 与 backend interface smoke。
- 正文最多两张表，不增加 cumulative-person-hours 图，除非存在真实完整计时记录。

## 7. 当前已知但仍需落盘复验的事实

- raw native catalog：TileLang 31、CUDA 35；
- runnable projection：Triton 28、TileLang 26、CUDA 30，共 84；
- native unavailable：10；
- 可运行 TileLang↔generated-CUDA pair：26；
- 当前 `add2d` artifacts 尚无完整 Gate1+Gate2 certificate，不计为闭合。

上述数字在 P0 生成新的机器可读 receipt 后才能进入论文结果表。

