# Task 10：J Semantic Readiness 与 SCRC vNext 横向闭合

> 状态：已完成（2026-07-24；最终 checkpoint）
>
> 类型：全量 28-Kernel SCRC 前的语义基线与跨层协议实现；不是 backend 性能实现、论文实验或
> 单一 Kernel 修补任务。

## 目标

在不预先设计“终极 A/B/C”的前提下，建立可供 SCRC 与未来 production backend 共同消费的稳定
J 语义基线，并实现下一协议版本的横向机械合同。Task 10 完成后，新建 Task 11 使用同一版本按
隔离的有界 case-level 并行推进
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
Task 11: bounded isolated 28/28 fresh closure
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
| reduction | operator/axes/input/result/accumulator dtype | order/reassociation/rounding/approx permission | warp/block/shared-memory family |
| mask/padding | logical domain/access/fill/effect | observable padding/result obligation | physical launch padding |
| update/state | state transition/index/combiner | duplicate/order/conflict/atomic permission | target atomic capability |
| recurrence/attention | carry/scan/trip/dataflow | numeric/order equivalence permission | fused schedule/pipeline |
| contraction | typed contraction/axes/result与accumulator dtype/cast/epilogue | FMA/reassociation/lower-precision permission | tensor-core/layout capability |

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
- 共享正式 Runner 的同一 source-free parser/Decode/anonymous-view/TypeCheck callable core；正式
  Runner 另执行 source-identifier privacy、receipt/provenance，并在提交后重跑该 core；
- 输入只有 candidate bytes 与显式匿名 compiler view/version context；
- 只返回 closed diagnostics，不修改/补全 candidate，不读取 Replay/Gate/RefEval；
- 正式提交后 Runner 重新执行权威检查；
- 激活的 vNext SCRC adapter 必须安装 checker；generic environment 无 checker 只可用于非认证测试，
  不属于 vNext SCRC 认证路径。

## Phase R2：横向 bring-up

先使用六类真实 Source，不按单 Kernel 增长 ontology：

1. DYT：elementwise、rank-0 scalar、mask；
2. SwiGLU：transcendental 与 result relation；
3. Attention `_bwd_preprocess_do_o_dot`：cast、mul、floating reduction；
4. softmax：max/sum/broadcast/div；
5. RMSNorm：square/mean/epsilon/rsqrt/affine；
6. contraction：dot/accumulator/epilogue。

每次 fresh run 结束后按 failure class 横向修复；同一改动至少有正例、反例和无关族回归。六项只是
协议 bring-up，不替代后续 Task 11 的 28/28 individual evidence。

## 验收

1. readiness matrix 对计划中的首批 generic transformation families 完成 ownership 分类；
2. C attachment 在 schema/parser/model/MaskC/TypeCheck/opaque remap 中一致，非法 ref/axis fail closed；
3. numeric-cast、refinement uniqueness 与 result-relation compatibility 有聚焦正反例；
4. CandidateCheck 与正式 Runner 共用实际 callable core，而非复制规则；
5. 六类 bring-up 每项形成新的 terminal evidence；失败准确定位，不以 per-kernel patch 掩盖；
6. 全量回归通过，protocol/receipt 版本一致，工作区无 secret、临时 artifact 或未提交修改；
7. 后续新 Task 11 获得明确的 vNext handoff，随后以独立 fresh lifecycle 做有界 case-level 并行，
   按 registry ordinal 稳定汇总 28/28 closure；同一 case 的 pre-Gate2 wave barrier 后按固定
   `(branch_id, formal_round)` 选择 winner，wall-clock 完成先后不参与选择。

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
  candidate bytes、匿名 `p*`/compiler view 与版本上下文，复用正式 Runner 的同一 source-free
  parser/Decode/anonymous-view/TypeCheck callable core，仅返回 closed diagnostics；正式 Runner
  另执行 source-identifier privacy、receipt/provenance，并在提交后独立重跑该 core。Source 文本、
  原始名字、Replay/Gate/RefEval 均不进入该服务。
- Lift receipt 记录 artifact/decode/typecheck 版本、CandidateCheck 调用次数与 rule codes；调用不计
  formal Gate round。
- 聚焦回归：123 passed，覆盖 v1/v2 迁移、C attachment 正反例、numeric cast、浮点/整数 relation、
  CandidateCheck 隔离/预算/worker 接线及 Runner receipt；全量 `python/tests`：865 passed。

哲学审查：本 checkpoint 没有引入 typed skeleton、deterministic artifact construction、字段自动修复、
第四 artifact 或 backend membership oracle；LLM 仍 first 并完整生成 `(J,C,π)`，固定代码只做解析、
依赖检查、拒绝与诊断。

### Checkpoint 10-B：六类 fresh bring-up 与通用诊断边界（2026-07-24）

状态：六类均已形成 fresh terminal evidence；DYT 与 SwiGLU 已完整认证，其余四类继续做
协议内的通用收敛，Task 10 保持执行中。本轮使用相互独立的 case lifecycle 并行执行，并未
共享 candidate、conversation、Replay、Gate diff、winner、C valuation 或可变 Harness 状态。

- SwiGLU `run-001` 为 `SCRC_CERTIFIED`：3 次 TypeCheck、2 次正式 Gate1、1 次 Gate2。Gate2
  三边全过；Source–Replay 误差为 0，Source–J 与 Replay–J 最大误差均为
  `1.4901161193847656e-08`，J 的 `tolerance` relation 与 external Q 按合同正常工作，旧
  `exact` false rejection 已消失。
- DYT knowledge-v9 `run-003` 为 `SCRC_CERTIFIED`：2 次 Lift/TypeCheck/正式 Gate1，第二个
  candidate 闭合并进入 Gate2；三边全过，Source–Replay 误差为 0，两条 J 边最大误差
  为 `2.9802322387695312e-08`，J-necessity 为 `proved`。Replay policy 对直接 `tl.tanh`
  fail closed，要求现行 Triton 3.6 合同中的 `libdevice.tanh`；未重写 Replay。
- Attention reduction knowledge-v8 `run-003` 有 3 份 TypeCheck-valid J 并实际执行 3 次
  Gate1；两份 Replay 因 block value 写入 scalar pointer 编译失败，一份编译成功后在 address
  topology、reduction axis 与 tensor rank 上 hard mismatch。TypeCheck-valid 只说明三元组在当前 dependent
  ruleset 下 well-typed，不能单独推出 J/π 已完整或错误只在 Replay。当前证据只将剩余缺口定位到
  typed artifact–blind Replay realization interface：可能是 J/π 尚未充分表达实现结构，也可能是
  Replay 未正确消费；二者尚未在 `K=3` 内区分并收敛。
- softmax knowledge-v9 `run-003` 有 3 份 TypeCheck-valid J 并实际执行 3 次 Gate1。最优
  candidate 的 operator、address、reduction、mask/fill 与数据流已结构闭合，唯一差异是 Source
  中 `.ca/.cs` cache hint。cache hint 不影响 J denotation、memory effect 或 transfer topology，又
  不属于 closed J/C/π 可表达事实；依据 `Derive(J) | Derive(C) | Derive(π) | NormalizeAway`
  完备性合同，Triton observer 全局只归一该命名 hint，profile 升级为
  `triton-ttir-3.6-gate1-v4`。mask、fill、type、address、reduction、effect 及其他 attributes
  仍为 hard evidence。用上一轮原始 TTIR 只读复算已通过，但该复算不改写历史 verdict，
  仍需要 v4 fresh run。
- RMSNorm 与 contraction knowledge-v9 均在 3 个 fresh worker 中稳定收到 CandidateCheck
  的 closed diagnostics，模型也实际修正了 returns ABI order、portable scope definitions、symbolic
  dimensions、authorization 与 route slots；但 14 轮与 3 次 checker evaluation 在最后一次修正/
  诊断后耗尽。因此有界预算最小调整为 16 model turns 与 4 次 checker evaluation，并升级
  `intentir.lift-knowledge.v10`；没有增加 schema 字段或自动修补。
- 提交边界现在对尚无精确 bytes-level `PASS` 的完整 candidate 自动执行同一只读
  CandidateCheck；FAIL 只返回 closed diagnostics，PASS 仍不是证书，正式 Runner 仍独立重跑
  Decode/TypeCheck。它不产生 typed skeleton、canonical artifact 或 patch。
- v10 首次四 case 并行与后续两 case 重试均在首个 provider request 收到
  `server_is_overloaded`；报告正确记为零候选/零 Gate 的 `INFRASTRUCTURE_FAILED`，不计入上述
  语义终态。后续使用单 case canary 探测 provider 恢复，再恢复明示上限的有界并行。

当前聚焦回归为 155 passed；全量 `python/tests` 为 868 passed。cache-hint 闭合集合
进一步收窄后的 TTIR 聚焦正反例为 26 passed。

### Checkpoint 10-C：xhigh fixed-consumer 与 TypeCheck 版本审计（2026-07-24）

状态：Task 10 的协议实现、六类横向 bring-up、版本审计与回归均完成。最终四类重跑统一使用
`gpt-5.6-sol/xhigh`、独立 case lifecycle 与 `B=1` 并行执行；此前被人工中断的 high run 及
`server_is_overloaded` 零候选报告不进入语义判断。`B=1,K=3` 只用于 bring-up/归因，不冒充
Task 11 的标准 `B=5` closure。

- Softmax：xhigh `run-008` 以 1 份 candidate 越过 TypeCheck、Replay 与 Gate1，随后暴露 v7
  RefEval 只允许 direct-identity node-less load 的过窄边界；`j_scalar_cpu_mlir_v8` 现允许唯一且
  真实进入 Layer-A dataflow 的 logical-input boundary load，重复、混合或未消费声明继续 fail closed。
  独立 non-identity 正反例闭合了这一能力。但 run-008 的 `max/sub` 与 `sum/div` 均缺少 J-owned
  non-trailing broadcast；`typecheck.v4` 会在 Gate1 前拒绝，故不能把 v8 修正追认为历史 PASS。
  补跑 `run-009` 形成 1 次 Lift rejection、3 次 TypeCheck/Gate1 hard mismatch 的 `NO_CLOSURE`，
  留作 v4 fresh relift 的输入，不继续用单分支抽样赌博。
- RMSNorm：xhigh `run-008` 有 2 份 TypeCheck candidate，第 2 份 Replay/Gate1 闭合并到达 Gate2，
  旧 v7 报 `REFERENCE_UNSUPPORTED`。深入检查表明它不是单纯 fixed consumer 缺口：旧 elaborator
  丢失 rank-0 empty shape，旧 TypeCheck 又未要求 total typing；同时 `n6: [2,x0] * [2]` 不满足
  NumPy broadcast 且 `J.boundary.broadcasts=[]`。v4 现在准确拒绝而不猜 row axis。只有 fresh J
  显式给出 broadcast 后，Task 11 才能继续判断 embedded mean/reduce shape 与 rsqrt CPU lowering。
- Attention reduction：xhigh `run-006` 的 3 份候选均越过旧 TypeCheck；第 3 轮已使 interface 与
  reduction hard comparison 通过，Gate1 score 提升至 `0.84941`。剩余 hard diff 来自 Replay 对
  无 predicate 的 output store 自行增加 mask，以及把 `address_base -> address_lane` 错做成额外
  pointer addition。公共 `U_f` 已明确：无 predicate 的 access 必须 unconditional；base/lane 的
  整数 contribution 在对应 pointer/address op 前合并，route 不授权连续 pointer add。没有增加
  Attention 专用规则或放宽 observer。
- Contraction：xhigh `run-007` 的 3 份旧 v3 candidate 都到达 Gate1，但均 `NO_CLOSURE`；最好一份
  score 为 `0.40283`。三份都遗漏 `contraction` 与 bias-add 的 result type，随后又用 identity 将
  f32 路径声明成 f16；Replay 只是因 Triton dot 默认行为偶然生成 f32 accumulate。v4 会以
  `j_value_type_incomplete` 在 Replay 前拒绝。现有通用 `op.type={dtype,shape}` 已足够表达 f32
  contraction result，不新增 accumulator schema 字段，也不允许 `tolerance` 代替 typed A/cast。
  loop-carried pointer topology、C attachment/π 地址依赖与 Gate1 region-aware liveness 留给 Task 11
  在 v4 fresh artifact 上横向处理。

最终活动版本与通用实现如下：

- `artifact.v2 / decode.v2 / typecheck.v4 / candidate-check.v1`；v4 修复 rank-0 closed type，要求
  Layer-A value typing total，并只消费 Lift 显式声明的 non-trailing broadcast relation；
- `j_scalar_cpu_mlir_v8`；能力扩展单独版本化，不改写 v7 `REFERENCE_UNSUPPORTED`；
- `triton-ttir-3.6-gate1-v4`；只 NormalizeAway closed load/store cache-policy hint，eviction、volatile、
  wrong-direction/unknown policy 与其他 memory attributes 仍是 hard evidence；
- pinned Triton 3.6 frontend matrix：不可用的 `tl.tanh/tan/atan/atan2/pow/isnan` 在编译前拒绝，
  使用已真实 CUDA 编译验证的 libdevice/public operator spelling；
- MiniSWE 使用 16 model turns、最多 4 次 CandidateCheck evaluation；显式与 submission preflight
  共用预算。提交只复用精确相同的已 PASS bytes，并在 PASS 后再次读取确认；checker 不改对象，
  Runner 仍独立重跑 source privacy、Decode 与 TypeCheck。这是 TOCTOU/诊断边界，不是 freeze/hash gate；
- same-case fan-out 升为 `candidate-fanout.v3`：等待 pre-Gate2 wave barrier，再按固定
  `(branch_id, formal_round)` 选唯一 winner；Gate2 只在选择后执行一次，wall-clock 次序不参与。

验证：合并后的四组聚焦与相邻回归为 `273 passed`；固定 Triton 3.6/CUDA libdevice 真实编译
`4 passed` 已包含其中；全量 `python/tests` 为 `895 passed`。Task 01–09 已整体迁入
`.trellis/tasks/archive/2026-07/`，活动目录只保留本 Task；历史正文中的旧状态不再重新激活任务。

哲学审查：LLM 仍完整生成 `(J,C,π)`。v4 只裁决“已声明图是否 total/self-consistent”，不会从
Source、kernel family、Gate diff 或 shape 猜 accumulator、broadcast axis、cast、authorization 或 π；
v8 只消费合法 fixed J；U_f 只收紧对已有 J/π 的忠实消费。没有 typed skeleton、deterministic
artifact construction、第四 artifact、backend membership oracle、per-kernel prompt 或历史 verdict
追认。Task 10 到此结束并归档；Task 11 从 v4 fresh relift 与真正的 bounded case-level parallel 开始。
