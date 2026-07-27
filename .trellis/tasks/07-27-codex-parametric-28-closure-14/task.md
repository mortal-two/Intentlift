# Task 14：Codex Agent 的 28-Kernel Fresh 参数化 SCRC 全闭合

> 状态：待执行（Task 13 工程提交后启动）
>
> 类型：Task 11 的 fresh corpus 执行波；不是 production backend、性能搜索、论文实验或历史证书重判。
>
> 历史基线：18/28 只说明旧版本各自曾取得证书；本任务对当前 28 项全部 fresh 运行，不把旧证书
> 当作当前参数化/runtime 版本的替代品。

## 目标

使用 Task 13 收口后的 Codex single-owner agent runtime，为 active registry 的 28 个真实 Kernel
重新形成各自独立的：

```text
Source + anonymous Harness/compiler view
  -> complete LLM-authored (J,C,pi)
  -> fresh source-blind Replay
  -> Gate1 closure
  -> direct three-leg Gate2
  -> current-version SCRC certificate and immutable evidence
```

旧 artifact、Replay、Gate verdict 与 digest 保持不可变，仅作历史对照。每个当前证书必须来自当前
Source、adapter、Lift knowledge、Replay backend、observer、RefEval 与 Harness valuation 的 fresh
lifecycle。

## 当前执行栈

- artifact/decode/typecheck：`artifact.v8 / decode.v12 / typecheck.v20`；
- Lift：`intentir.lift-knowledge.v56`；
- Replay：`openai.codex-cli.intentir-replay.v6`；
- Harness：Triton workload adapter v5，区分 concrete valuation 与 Harness-only `h*` identity；
- Gate1：当前 observer profile 与 candidate-fanout v4 first-observed winner；
- Gate2：Source–Replay、Source–J、Replay–J 三条直接比较，具体阈值只来自外部 Q。

协议升级后不得追认旧 verdict。任何后续通用能力修复都只对受影响 case 建立新的 fresh evidence。

## 不可改变的数学与信息边界

1. 每个 branch 只有一个 source-aware Codex Lift owner，贯穿该 branch 的所有 formal Gate1 relift；
   feedback 只回到这个 owner，且每轮重新提交完整 `(J,C,pi)`。
2. 每个 formal round 创建全新的 source-blind Replay episode。Replay 只见
   `J + pi + MaskC(C) + U_f`，不见 Source、Harness value、Gate diff、旧 Replay、sibling 或其他 case。
3. LLM 完整作者化 artifact 与 Replay。固定代码只搬运 exact bytes、执行 closed checks、拒绝、诊断、
   隔离、调度和记录 receipt；不得补 relation、猜 axis、修 J 或生成 Kernel recipe。
4. Harness `h*` 不进入 artifact，也不自动生成 J `x*`。Lift 必须显式作者化 logical dimension，必要时
   作者化 rank-0 value/dimension equality；固定代码只检查 source-grounded contract 是否满足。
5. π 在 SCRC 后停止；本任务不实现 production backend，也不让 backend 读取 Source/π。
6. compile-time variant 是不同程序/证书。本任务只认证 adapter 选定 variant，不从相等 valuation
   猜测产品、capacity、rounded extent 或未声明 variant family。

## Agent 与预算

Codex 是主执行路径；MiniSWE 只用于隔离 runtime/provider 问题时的同合同交叉核验，不形成不同
artifact 或不同 Gate。

- 主模型：Terra high；Sol high/xhigh 只作为明确记录的独立 branch/backend 对照；
- 同一 Kernel 默认 `B=3`，三个 branch 独立，不共享 J、feedback、notes 或 Replay；
- formal Gate1 上限默认 `K=5`。它是上限而非目标；首个 Gate1 closure 立即成为 winner，sibling
  cooperative stop；
- agent 内部 tool/model step 不设“三次”能力上限，由 active wall-time、transport safety 和 exact
  stagnation 管理；
- 若同一 branch 的 artifact digest、Replay digest 与 closed Gate1 frontier 连续不变，只能归类为
  orchestration/semantic stagnation，不能靠无限 K 掩盖；
- 外层按多个语义族并行，满足 `W*B<=G`。不得因一个慢 case 让其他 case 空闲等待。

TypeCheck、CandidateCheck、ReplayCheck 和 synthetic compile 只记录构造阶段，不作为能力进展指标。
能力指标只有 Gate1 frontier、Gate1 closure、Gate2 三边和最终 SCRC verdict。

## 参数化执行政策

### 可以直接要求 fresh 参数化的部分

- row-wise rows/cols；
- Flash/Linear/GroupNorm 中由 tensor shape、runtime scalar 或显式 bound constexpr 唯一绑定的 batch、
  sequence、head、head-dim、rounded capacity；
- Qwen2VL MRoPE、Liger RoPE 中唯一对应 tensor axes 的 batch/sequence/head/head-dim；
- Llama4 RoPE 中 batch/sequence/head-count/head-dim-half。

### 不得擅自提升的部分

- Llama4 `full_head_dim = 2 * head_dim_half` 在当前 closed relation vocabulary 没有乘积关系时保持
  selected-variant literal；
- SplitK 的产品、vararg、derived capacity、split count 与 compile-time family 保持 adapter 选定
  variant，直到出现 production 与 Gate 都共同消费的 closed relation；
- inactive/dummy ABI carrier、static polynomial coefficient axis 与 stride/layout metadata不因数值类似
  logical extent而提升。

若 Lift 把已声明 `h*` 的 source-grounded identity 固化成 concrete literal，candidate 必须在进入正式
Replay前被拒绝；fixed code不得替 Lift 自动修成 `x*`。

## 横向执行流程

### Wave 0：runtime acceptance

1. 一个历史简单闭合 case fresh 跑通参数化 Gate1/Gate2；
2. 一个 Flash/Linear 复杂 case验证 `physical_iterations/physical_loop_carriers`、TMP inout、完整 π
   realization view 与 fresh Replay；
3. 只接受真实 Gate frontier，不把 provider、marker、路径、mailbox 或工具发现失败记成语义失败。

### Wave 1：28 项第一遍横向覆盖

- 按 registry ordinal 与语义族分波并发；
- 每项保存完整 branch/round/evidence，首个 winner 后立即停止 sibling；
- Gate1 PASS 后固定 artifact 与 Replay再进入 Gate2；
- Gate2-only consumer/Q 缺口不触发 relift。

### Wave 2：通用能力修复与受影响 case fresh rerun

失败按下列互斥主因聚类：

```text
Lift artifact construction
J/B semantic omission
pi realization/time/control omission
Replay packet consumption
Gate1 observer unsupported
Gate2 RefEval/Q/consumer gap
provider/runtime infrastructure
```

只修通用能力，不加入 case/source symbol 专用 prompt、模板 J、hard-coded Replay 或跨 case semantic
cache。修复后只 fresh rerun 确受影响的 case，同时继续横向推进其他类别。

### Wave 3：最终证书与消融

- 形成当前版本 28/28 `SCRC_CERTIFIED` corpus；
- 对 winner 运行既有 Masked-J / necessity 消融；
- 汇总 formal rounds、Gate1 frontier演化、Gate2 三边误差、wall-time、model/tool telemetry 与失败分类；
- 不把 TypeCheck pass rate 写成 semantic capability。

## Gate1 / Gate2 复用规则

- Gate1 FAIL：只回到同一 branch-local Lift owner做完整 relift，下一轮使用 fresh Replay；
- Gate1 PASS、Gate2 FAIL：先判断 exact J 与 Replay 是否正确；若只是通用 RefEval/Q/frontend consumer
  缺口，修 consumer 后零模型复用 exact Gate1-closed artifact/Replay做 Gate2-only recheck；
- 只有 J/B relation 本身错误才启动新的完整 Lift；
- 任何 hidden Gate2输入或结果都不得回流 Lift、Replay或 sibling selection。

## 完成条件

1. active registry 28 项均有当前版本、fresh、case-local 的 accepted `(J,C,pi)`、Replay、Gate1 与
   三边 Gate2 evidence；
2. 每个应参数化的 logical extent 都由 J 作者化为 `x*`/合法 equality，而不是 concrete Harness
   valuation；明确保留的 selected-variant literal 有 source-grounded解释；
3. 28 项均为 `SCRC_CERTIFIED`，历史 18/28 只保留为旧版本对照；
4. evidence/report按 registry ordinal稳定汇总，first-winner、sibling stop、formal K、fresh Replay 与
   provider/runtime receipts完整；
5. 代码、spec、Task11状态与最终 corpus事实一致，测试通过，工作区干净。

## 非目标

- 不实现或评估 production backend搜索；
- 不从 Forge/旧 artifact复制当前证书或实验结果；
- 不增加 FMA/TF32/reassociation 等静态 permission 字段来替代执行域闭合；
- 不恢复 planner/builder/reviewer 多作者流水线；
- 不用 TypeCheck、更多 token、更多 K 或特化 prompt冒充 Gate1/Gate2闭合能力。
