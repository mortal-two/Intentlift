# Task 20：J-only Replay、typed π-relative Gate1 Observer 与最终闭合

> 状态：`DESIGN_ACTIVE`（2026-07-31）
> 接管已归档 Task 19；Task 11 继续是不停止的 28-Kernel 总闭合主线。

## 0. 唯一目标

保持 Figure 3 的外部 SCRC 流程不变，将内部接口改为：

```text
Lift(Source) -> (J,C,π)
ReplayAgent(J, MaskC(C); U_f) -> R
Gate1: CanonPair_pi(Obs(Source), Obs(R); c0,f,p,nu) has empty residual
Gate2: Source–Replay / Source–J / Replay–J
```

π不进入 Replay；不建设 Target-Normal IR 或确定性 Composer；π只由版本化 frontend observer
canonicalizer消费。

设计依据：

- `doc/method/IntentIR_J-only_Replay与pi参数化Observer_Gate1_2026-07-31.md`；
- 四份 2026-07-22 SCRC 权威文档；
- 论文 Figure 3 的既有外部流程。

## 1. 起点与已知事实

- Task 19 的 Composer/Target-Normal 方法与任务已归档；其 runtime prototype 暂不删除，等待本任务明确迁移；
- 当前 Replay 仍有直接消费 π 的旧路径；
- 当前 Triton Gate1 diff 是 generic counter delta，没有 J anchor 或 π-relative normalization；
- CuTe/TileLang/CUDA/Triton 使用不同 observer/diff vocabulary；
- 当前 π schema 已封闭、无 wildcard/free text，但 relation 尚未获得 Gate1 normalization semantics；
- 28-Kernel 严格基线与 evidence 必须在迁移前重新核对，不继承 Task 19 未完成 claim。

## 2. P0：归档与规范边界

- [x] 归档 Task 19 与活动 Composer 方法文档；
- [x] 更新 tasks/doc/spec 索引，Task 20 成为唯一活动 Replay 方法；
- [x] Composer prototype code 暂保留为显式 opt-in 历史原型，不作为默认路径、不提供 Task 20 evidence；runtime 迁移时再归档或删除；
- [x] 不新增/恢复 `python/tests`，只使用真实 closure workflow。

## 3. P1：数学与 π-normalization contract

- [ ] 冻结 `R=Replay(J,MaskC(C);U_f)`，从所有 Replay packet/workspace/tool删除 π；
- [x] 定义 `CanonPair_pi(O_S,O_R;c0,f,p,nu)`、两侧 `Phi_pi` 与 Gate1 equality；
- [x] 定义二侧 `CanonPair_pi`、`Diff subset Denote(pi)` coverage receipt 与 normalized residual 的关系；
- [x] 定义 deterministic/symmetric/idempotent/hard-preserving/fail-closed/atomic 性质；
- [ ] 为每个可 normalize relation 实现固定的 typed anchor/applicability recognizer；不匹配时保留 residual；
- [x] 明确本协议只给 observer/test-domain closure，不 overclaim 全输入形式等价或 Source→J/π 完备验证。
- [x] 明确 compiler/observer/rule-table/CanonPair/comparator 属于 Gate1 TCB；错误 rule 是 profile 缺陷，不是 π 权限。

## 4. P2：π schema / TypeCheck vNext

- [ ] 建立 hard-by-default capability table；所有 relation 默认 `witness_only`，只对已有完整 profile rule spec 的 exact kind 窄升；
- [ ] 固定 relation 对 observer diff 的 denotation、erasable atom allowlist、anchor/premise/scope/composition，禁止 wildcard、free text、Kernel rule；
- [ ] 保证 π 无法覆盖 J-owned op/reducer/predicate/fill/domain/effect/output diff；
- [ ] sensitive numeric/state/control relation必须由 J/B authorization约束，默认不 normalize；
- [ ] TypeCheck 同步版本化，但不读取 Source observation或自动推导 π；Gate1 另做二侧 applicability receipt；
- [ ] 记录 proposed/used/unused π 与 exact coverage receipt；证书只认证 used clauses。
- [x] 第一阶段不改 π wire；normalization capability、recognizer 与 coverage receipt 留在 profile/runtime metadata，只有真实 anchor 缺口才版本化最小 generic 扩展。

## 5. P3：Frontend-local π-relative Observer

- [ ] 定义 profile-scoped `normalize_with_pi(observation_pair,J,C,π)` 接口；
- [ ] Triton先完成窄 relation vertical slice；
- [ ] CuTe/TileLang/CUDA分别声明支持子集，不强求共同 raw diff schema；共同 wire syntax 不冒充共同 denotation；
- [ ] 证书记录 frontend/compiler stage/observer/canonicalizer/pi-schema versions；
- [ ] 歧义、缺少anchor、unsupported relation保持hard或报observer_unsupported；
- [ ] 禁止按Kernel增加normalization rule。

## 6. P4：J-only Replay 与反馈

- [ ] 默认 MiniSWE/Codex Replay workspace只挂载 J/S_J、MaskC schema 与 U_f；
- [ ] 删除 π.intent、π查询工具、π ledger 与 prompt 文案；
- [ ] 保留 persistent source-blind coding loop、compile工具与完整Replay源码所有权；
- [ ] Gate1反馈仅向 source-aware Lift owner交付typed diff/coverage，不进入Replay；
- [ ] J/C修改触发fresh Replay；π-only修改允许 exact-R Gate1 recheck；
- [ ] normal/MaskJ各自fresh派生，记录J necessity。

## 7. P5：真实纵切与28项闭合

- [ ] 先用一个已闭合Kernel验证迁移前后Gate2不退化；
- [ ] 用一个program-linearization Kernel验证 π-relative normalization；
- [ ] 用一个loop/carrier/control Kernel验证unsupported/fail-closed边界；
- [ ] 横向推进剩余未闭合Kernel，至少两个同时运行、same-case多branch；
- [ ] 稳定后fresh Replay全部28项；Gate1已闭合但Gate2失败先区分J与RefEval/Replay问题；
- [ ] 不以protocol mock、hash/banner或归档tests替代真实closure。

## 8. P6：文档、论文与实验

- [ ] 同步四份权威SCRC文档与 `.trellis/spec`；
- [ ] Figure 3 拓扑不变，只更新 Replay input 与 π-relative observer文字/符号；
- [ ] 论文明确 π 是 typed realization quotient witness，不是Replay输入或语义权威；
- [ ] 报告 first-pass/relift closure、π proposed/used/unused、MaskJ、observer profile与test-domain边界；
- [ ] reviewer审计：候选不能扩张 `Σπ`，不同frontend证书强度分别报告。
- [ ] reviewer审计：证书只认证 used π clauses；自适应 relift 是确定性 witness search，不伪装独立一次预测。

## 9. 完成条件

- [ ] Replay运行时、spec、论文均明确不读取 π；
- [ ] Gate1正式谓词为在固定 `f,p,nu,c0` 下 typed π-relative normalized equality，或等价的空 residual；
- [ ] π relation algebra封闭，J-owned hard projection不可被normalize；
- [ ] 至少Triton真实纵切通过Gate1/Gate2，其他frontend准确报告capability；
- [ ] 28/28 current-version SCRC结果完成或对剩余项给出真实capability边界；
- [ ] production backend仍只消费J+C；
- [ ] 提交后工作区干净。

最终裁决：

> **Replay只从J构造；π只参数化固定Gate1 observer quotient。Gate1检查π-relative compiler closure，
> Gate2检查声明执行域中的三边行为；任何未被封闭π algebra覆盖的差异保持hard。**
