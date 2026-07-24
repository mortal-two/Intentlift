# Task 10：J Semantic Readiness 与 SCRC vNext 横向闭合

> 状态：执行中
>
> 类型：全量 28-Kernel SCRC 前的语义基线与跨层协议实现；不是 backend 性能实现、论文实验或
> 单一 Kernel 修补任务。

## 目标

在不预先设计“终极 A/B/C”的前提下，建立可供 SCRC 与未来 production backend 共同消费的稳定
J 语义基线，并实现下一协议版本的横向机械合同。Task 10 完成后，Task 03 使用同一版本顺序推进
28/28 SCRC；后端 B0/B1 只在稳定 J corpus 形成后独立展开。

```text
planned generic transformations
          |
          v
J readiness audit: A denotation / B permission / C prior / compiler contract
          |
          v
SCRC vNext shared schema + Decode + TypeCheck + CandidateCheck
          |
          v
six semantic-family bring-up cases
          |
          v
Task 03: sequential 28/28 fresh closure
```

## 第一性原理

1. `J=A+B` 是唯一 denotational object；C 是可忽略 prior，不属于 J。
2. 只有影响 denotation/allowed behavior 的事实进入 A/B；target capability、current MLIR state 与
   pass implementation precondition 留在 compiler contract。
3. LLM 继续完整生成 `(J,C,pi)`；固定代码只执行 closed parsing、dependent checking、拒绝和诊断。
4. 缺少 B permission 时禁用 transformation，不把旧 J 判成非法，也不从测试输出补 permission。
5. 一个真实失败先归入通用语义族；只有跨样本可复用的证据才扩展 schema/consumer/prompt。
6. 横向覆盖优先：每次 checkpoint 同时覆盖 schema、parser、TypeCheck、worker diagnostics 与多个
   semantic family，不围绕一个 Kernel 无限追加专用规则。

## Phase R0：J readiness matrix

建立 source-neutral transformation-demand matrix，至少覆盖：

| 语义族/变换 | A 必须表达 | B 才能授权 | 不进入 J |
|---|---|---|---|
| elementwise/transcendental | typed op/dataflow | rounding/approx relation（若允许） | target intrinsic availability |
| reduction | operator/axes/input/result type | order/reassociation/accumulator contract | warp/block/shared-memory family |
| mask/padding | logical domain/access/fill/effect | observable padding/result obligation | physical launch padding |
| update/state | state transition/index/combiner | duplicate/order/conflict/atomic permission | target atomic capability |
| recurrence/attention | carry/scan/trip/dataflow | numeric/order equivalence permission | fused schedule/pipeline |
| contraction | typed contraction/axes/epilogue | contraction/FMA/precision permission | tensor-core/layout capability |

每一行只允许三种结论：当前 J 已表达、属于 compiler contract、确有最小 closed A/B 缺口。不得把
“backend 以后可能有用”当作新增字段的理由。

## Phase R1：SCRC vNext 横向合同

### R1.1 C attachment

- slot 保留 `name/kind/role/value_type/required`；
- 新增可选 `subject_ref/axis_ref`；
- `subject_ref` 只允许 closed `CSubject_v1(J)`；
- `axis_ref` 必须满足 closed `AxisRefs_v1(J,subject_ref)`；
- attachment 由 Lift 显式生成，`MaskC(C)` 保留；缺失/含混时 backend 安全忽略；
- 不增加 `legal_domain/strength/retunable/fixed`。

### R1.2 dependent TypeCheck

- 修正 `numeric_cast`：source dtype 对 operand，target dtype 对 result；
- 每个 pi instance 至多一个 refinement；
- 实现 `RequiresTolerance_v1`，只拒绝不相容 `exact`，不替 Lift 选择 relation；
- schema/decode/typecheck/receipt version 同步升级，历史 artifact 不重判。

### R1.3 shared CandidateCheck

- agent 只对自己已写出的完整 candidate 调用；
- 与正式 Runner 调用同一 parser/Decode/TypeCheck entrypoint/ruleset；
- 输入只有 candidate bytes 与显式匿名 compiler view/version context；
- 只返回 closed diagnostics，不修改/补全 candidate，不读取 Replay/Gate/RefEval；
- 正式提交后 Runner 重新执行权威检查。

## Phase R2：横向 bring-up

先使用六类真实 Source，不按单 Kernel 增长 ontology：

1. DYT：elementwise、rank-0 scalar、mask；
2. SwiGLU：transcendental 与 result relation；
3. Attention `_bwd_preprocess_do_o_dot`：cast、mul、floating reduction；
4. softmax：max/sum/broadcast/div；
5. RMSNorm：square/mean/epsilon/rsqrt/affine；
6. contraction：dot/accumulator/epilogue。

每次 fresh run 结束后按 failure class 横向修复；同一改动至少有正例、反例和无关族回归。六项只是
协议 bring-up，不替代 Task 03 的 28/28 individual evidence。

## 验收

1. readiness matrix 对计划中的首批 generic transformation families 完成 ownership 分类；
2. C attachment 在 schema/parser/model/MaskC/TypeCheck/opaque remap 中一致，非法 ref/axis fail closed；
3. numeric-cast、refinement uniqueness 与 result-relation compatibility 有聚焦正反例；
4. CandidateCheck 与正式 Runner 共用实际 callable core，而非复制规则；
5. 六类 bring-up 每项形成新的 terminal evidence；失败准确定位，不以 per-kernel patch 掩盖；
6. 全量回归通过，protocol/receipt 版本一致，工作区无 secret、临时 artifact 或未提交修改；
7. Task 03 获得明确的 vNext handoff，随后顺序推进 28/28 closure。

## 非目标

- 不实现 backend search、cost model、multi-lowering 或性能实验；
- 不先把所有 target pass precondition 塞入 B；
- 不增加第四 artifact、Composer、HarnessAgent、RuleCard、kernel allowlist 或 hash/冻结门；
- 不用 Source/π/Gate output 自动生成或修补 J/C/π；
- 不因为一个 Kernel 卡住而暂停其他语义族的 schema/consumer/test 覆盖。

## Checkpoint 记录

本节随实现追加实际代码、测试与 fresh run evidence；设计目标和历史 verdict 不反向改写。

### Checkpoint 10-A：vNext 机械合同横向贯通（2026-07-24）

状态：R0 ownership matrix 与 R1 实现完成；R2 六类 fresh bring-up 尚未开始，Task 10 保持执行中。

- 激活 `artifact.v2 / decode.v2 / typecheck.v3 / candidate-check.v1`；新 Lift response 只接受 v2，
  v1 仍可按原 wire 读取但不能携带 C attachment，历史 receipt 不被新规则静默重判。
- C slot 已在 schema/parser/model/opaque `MaskC` view/TypeCheck 全链增加由 Lift 明示的可选
  `subject_ref/axis_ref`；`CSubject_v1` 与 `AxisRefs_v1` 非法引用 fail closed，缺失 attachment 不补造。
- numeric-cast authorization 改为检查显式 cast operand/result dtype；π refinement 仍由 LLM 完整
  声明，parser 对重复 refinement 拒绝，固定代码没有派生 authorization。
- `RequiresTolerance_v1` 已沿 output producer/return/mapping 的 A dependency closure 检查 floating
  reduction、closed transcendental 与 reassociation authorization；只拒绝 default/explicit exact，
  不选择 relation，也不持有 Q 的数值阈值。
- MiniSWE worker 可对自己写出的完整 candidate 最多调用三次只读 CandidateCheck。host service 只收
  candidate bytes、匿名 `p*`/compiler view 与版本上下文，复用正式 source-free Decode/TypeCheck core，
  仅返回 closed diagnostics；正式 Runner 提交后独立重跑。Source 文本、原始名字、Replay/Gate/RefEval
  均不进入该服务。
- Lift receipt 记录 artifact/decode/typecheck 版本、CandidateCheck 调用次数与 rule codes；调用不计
  formal Gate round。
- 聚焦回归：123 passed，覆盖 v1/v2 迁移、C attachment 正反例、numeric cast、浮点/整数 relation、
  CandidateCheck 隔离/预算/worker 接线及 Runner receipt；全量 `python/tests`：865 passed。

哲学审查：本 checkpoint 没有引入 typed skeleton、deterministic artifact construction、字段自动修复、
第四 artifact 或 backend membership oracle；LLM 仍 first 并完整生成 `(J,C,π)`，固定代码只做解析、
依赖检查、拒绝与诊断。
