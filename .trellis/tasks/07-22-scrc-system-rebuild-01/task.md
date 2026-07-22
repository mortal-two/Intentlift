# Task 01：SCRC Gate Core 整体门禁能力重构

> 状态：执行中
>
> 类型：代码系统新增/重构；不是规模实验、论文实验或性能任务。
>
> 规则：本文件只借用 `.trellis/tasks` 的目录布局，不使用 Trellis CLI、`task.json`、自动
> session 或 workflow。

## 任务依据

本任务直接实现以下上位设计，设计本身不在本任务中重新辩论：

1. [C：所有权、绑定、消费与三方验证](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)
2. [π：依赖类型实现见证与 Replay 安全](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)
3. [Lift、Replay、Gate1、Gate2 执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)
4. [SCRC 最终数学、执行流程与论文表述](../../../doc/IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)

工程约束以 [活动 spec](../../spec/index.md) 为准。历史任务、`spec/archive` 和
`doc/archive` 只用于追溯，不参与活动 import、schema 或运行路径。

## 目标

把当前分散的 Gate1/Gate2 组件重构成一条真实、可导入、可测试的门禁核心：

```text
Lift_H(Source) -> (J,C,pi)
                  |
          TypeCheck(J,C,pi)
                  |
 Replay(J,pi,MaskC(C);U_f) -> R
                  |
 Bind_H(C)=c0 -> Gate1(Source,R,c0,p)
                  |
       Gate2(Source,R,RefEval(J),c0,Q)
                  |
        SCRC_CERTIFIED / terminal failure
```

完成本任务只表示门禁能力和系统边界已经真实落地；不在本任务内调用外部 LLM、不恢复 28-kernel
批量、不做性能优化，也不声称任何全域语义定理。后续任务再用真实 Source/Lift/Replay 和
实际性能数据检验这条能力。

## 不可改变的实现边界

- proof-relevant artifact 只有 `(J,C,pi)`。
- J 是唯一语义对象，拥有完整 logical signature（inputs、outputs、runtime scalars、
  destinations、dtype/rank/shape、alias/inout/effect 和结果合同）。
- C 只有匿名 implementation-selection slots 及其 Harness valuation；`MaskC(C)` 擦除具体值。
- π 是依赖 `(J,C)` 的 typed relational witness；没有 `Ids(interface)`。
- 需要调用接口时只计算 `InterfaceView(J)`，不独立序列化 `interface`/`ABI`/`BindingMap`。
- Replay 的实际输入严格是 `J + pi + MaskC(C) + U_f`；`U_f` 是固定、source-neutral 的公共
  前端上下文。
- Harness 只绑定已有坐标和同一个 `c0`，不构造语义或 Replay AST；RefEval 只读取 J 和逻辑输入。
- Gate1 使用声明的 frontend/observer profile；失败 diff 只给 source-aware Lift，下一轮输出
  完整新三元组和全新盲 Replay。
- Gate2 使用外部 Q 的确定性、对称 `Close_Q`，直接检查 Source–Replay、Source–J、Replay–J
  三条边；当前主路径只接收 deterministic/contract-deterministic 合同。
- `main_status`、`typecheck_status`、`j_necessity_status` 分离；masked-J 不直接否决主认证。
- 不新增 hash/digest/commitment/epoch/freeze 门，不恢复 RuleCard、composer、Replay patch 或
  任何旧控制面旁路。普通日志标识若已有，只能用于追溯，不能参与合法性或认证结论。

## 现状基线（开始实现前记录）

当前代码仍存在以下过渡形态，均在本任务内直接迁移或退役，不建立新旧双轨：

- `Gate1Artifact` 根对象仍包含独立 `KernelInterface`、`ScheduleSpec`、`LocationMap`；
- `RealizationWitness` 只做自闭合语法/图检查，尚未有统一 `TypeCheck(J,C,pi)`；
- `build_rewriter_payload` 仍把 `harness` 作为 Replay packet 字段；
- `Gate1Runner` 与 `CuteGate1Runner` 分开维护候选生命周期，尚无统一 relift runner；
- `Gate2Runner` 当前只有 Source–Replay 与 Source–scalar 两条比较边，tolerance 由 J 直接携带，
  没有外部 Q 和 Replay–J 结果；
- 状态、报告和 Gate2 metrics 仍按旧的 Gate1/Gate2 局部模型组织。

第一步必须把这些事实映射到新边界，再修改生产代码；不得为了让旧测试继续通过而添加兼容旁路。

## 执行记录

- 重构前工程基线：`669 passed, 4 skipped`。
- 阶段 1 已完成：`SemanticArchive.signature` 成为 J-owned logical signature；proof artifact wire
  直接断代为 `schema + j + c + pi`；独立 `KernelInterface/KernelParameter/ScheduleSpec` 已从活动
  imports/exports 移除；Source、Harness binding 与 provider provenance 留在 extraction sidecar。
- Replay request 根 allowlist 已同步收敛为 `j + c + pi + u_f`，其中 c 是 `MaskC(C)`；旧
  `harness/interface/location/source/provenance` 字段直接拒绝，不提供兼容转换。
- 阶段 1 定向回归：`112 passed`；全量回归：`669 passed, 4 skipped`。
- 阶段 2 已完成：新增唯一生产入口 `type_check(j,c,pi)`；机械检查 J/C/π 引用、C 实际
  valuation、route slot/type 与图、observable J anchor、sharing、敏感 refinement 的 J/B
  authorization（含 kind/subject/source dtype）以及固定 `GlobalSafeRefinement`。TypeCheck 失败时
  Replay builder 直接拒绝，不调用生成器。新增 11 个最小正反例；阶段 2 后全量回归：
  `678 passed, 4 skipped`。
- 阶段 3 的 Replay 输入边界已完成：新增 typed `ReplayPacket` 和唯一 canonical builder；根字段固定
  为 `j + c + pi + u_f`，其中 c 只有 `MaskC(C).slots`，J/C/π/U_f 均使用闭合字段集合。
  生产 API 不再接受调用方任意注入 frontend context；Source、Harness、BindingMap、interface、diff、
  prior Replay/conversation 等嵌套旁路均 fail closed。相关定向回归：`77 passed`。

## 实施顺序

### 1. Canonical artifact 与 J.signature

- 将 logical signature 建模为 J 的确定性部分，补齐 handles、类型、rank/shape、alias/inout/effect。
- 把 C 收敛为一个 canonical 对象和 `MaskC(C)` 投影。
- 将 `InterfaceView(J)` 做成纯投影；活动 wire/schema/API 拒绝独立 interface namespace、
  `Ids(interface)` 和未知字段。
- π anchor 只解析到 J/C；旧格式直接失败，不写自动转换器。
- 保留真正通用的 compiler adapter 与 Harness 机械绑定逻辑，删除其语义 sidecar 依赖。

### 2. Dependent TypeCheck

新增生产入口 `TypeCheck(J,C,pi)`（命名可按代码风格调整，但必须只有一个活动入口），机械检查：

1. J/C/signature ref 存在；
2. route slot、arity、carrier type、region/order/sharing 图合法且无环；
3. 产生 value/store/effect 的 observable instance 锚定 J-owned node/access/effect；
4. π/C 没有 operator、predicate、fill、formula、semantic constant、Source/API、C value 或
   可执行自由文本；
5. cast、integer width、accumulation dtype、reassociation、atomic 等 sensitive refinement
   引用 J/B authorization，或命中系统封闭的 `GlobalSafeRefinement`；
6. unknown field/relation/free-text 扩展 fail closed。

`Parse(pi)` 只负责语法，不能代替 dependent TypeCheck。错误必须结构化、可返回 Lift；TypeCheck
不做 SMT、range proof 或一般语义等价证明，正式 claim 只到显式语义所有权隔离。

### 3. Source-isolated Replay 与 Harness

- 建立唯一 Replay packet builder，allowlist 只有 `J`、`pi`、`MaskC(C)`、版本化 `U_f`；
- 单独测试并拒绝 Source/Harness 内容、C values、source name/position、BindingMap、interface
  sidecar、compiler fingerprint/diff、旧 Replay、旧 conversation、per-kernel RuleCard；
- Replay 每次生成完整、可独立构建的源码，固定代码只解析、编译和绑定，不从 π 施工源码；
- Harness 在生成后把同一个 `c0` 绑定给 Source/Replay；RefEval 的类型和实现不接收 C/π/Q。

### 4. Gate1、observer 与 relift

- 为 Triton、CuTe 等前端统一 adapter 合同：compile、observe、canonicalize、run；记录
  frontend、observer profile/version、dump stage、canonicalization version。
- Gate1 直接比较同一 c0 下 Source/Replay 的 observer output；diff 只描述结构差异，不替 Lift
  决定修复策略。
- 建立统一的完整 relift 生命周期：每一轮都是完整 `(J,C,pi)`、TypeCheck、全新盲 Replay；
  最多三轮正式候选，失败为 `NO_CLOSURE`。Gate2 失败不回流 relift。
- masked-J 只写入 `j_necessity_status`，不映射为旧的 `LEAKAGE_REJECTED` 主状态。

### 5. Q 与 Gate2 三边

- 引入外部、版本化 Q policy；J/B 只能声明封闭 comparison class 和必要合同，不能输出任意
  comparator/tolerance。
- `Close_Q` 显式对称，处理 exact、数值容差、NaN/Inf 和 unsupported 类别，非法参数 fail closed。
- 一次执行 Source、Replay、RefEval(J)，直接保存并判断三条 pairwise 边；加入近似关系非传递的
  负例，证明缺少 Replay–J 边会被拒绝。
- 对真正集合值语义暂时明确 unsupported，不把有限 membership 偷换成当前三边等价。

### 6. 统一门禁 runner 与回归

- 建立一个活动的 `SCRCGateRunner`（或等价唯一入口），统一 Lift artifact、TypeCheck、Replay、
  Harness、Gate1、Gate2 和 terminal 状态；旧局部 runner 只能作为 adapter 被调用，不能各自维护
  候选生命周期。
- 统一记录实际 artifact、Replay request/output、observer、Q、c0、测试输入、三边结果、失败分类
  和 LLM 调用计数；这些记录不成为 proof artifact 或认证门。
- 对每一条新边界补正例、最小负例和跨模块集成测试；全量现有回归通过后，用本地 deterministic
  test provider 做 end-to-end smoke。该 smoke 不是后续真实 LLM 实验的替代。

## 验收条件

1. 活动 proof artifact 的序列化/解析只保留 `(J,C,pi)`；独立 interface、未知字段和旧 wire 失败。
2. `J.signature` 被 TypeCheck、Harness、Replay binding、RefEval、Gate2 实际消费，而非只存在
   schema。
3. 任意未通过 TypeCheck 的 π 不会进入 Replay；所有 sensitive refinement 都有授权或 fail closed。
4. 生成的 Replay packet 可由 allowlist 精确枚举，且每个候选都使用同一个 c0，不读 source sidecar。
5. Gate1 具备 observer profile、结构化 diff、完整 relift、三轮终止和 masked-J 独立状态。
6. Gate2 对同一组三个输出执行三条直接、对称的 `Close_Q`；Q 不受候选控制，Gate2 失败为 terminal。
7. 一个统一 runner 能输出 `SCRC_CERTIFIED` 或明确 terminal failure，并分离三个状态维度。
8. 生产代码、exports、测试和文档中不再有活动旧 hash/冻结门、RuleCard/composer、Replay patch、
   独立 interface 旁路或静默 fallback。
9. 全量回归、门禁核心负例、跨模块 smoke 和 `git diff --check` 通过；工作区提交后干净。

## 本任务完成后的下一步

Task 01 完成后，依据实际缺口建立 Task 02：先用少量真实 Source + source-aware Lift + 全新
source-blind Replay 验证 exact/masked/reduction 三类能力，再决定如何扩展到 28 个高性能 Kernel 和
性能测量。真实 LLM、规模统计和系统能力改进属于后续任务，不在本任务内伪装成已完成。
