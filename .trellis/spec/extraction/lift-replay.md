# Lift 与 Replay 合同

> 状态：当前派生规范。上位设计见 [π](../../../doc/IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md) 与 [执行流程](../../../doc/IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md)。

## Lift

```text
Lift_H(Source) -> (J,C,pi)
```

Lift 直接完成 whole-tensor semantic lifting、C 分离与 source realization π 的提出。proof-relevant
输出只有 J、C、π。diagnostic、location、provider receipt 等可以作为 proof-irrelevant sidecar
保存，但不得进入 Replay、Gate verdict 或 production lowering。

J 必须包含 logical signature；不允许 root sibling `interface` 成为独立对象。实现若需要接口，
只能按需计算 `InterfaceView(J)`。H 可以向 Lift 提供匿名、未分类的 bindable coordinates，但
不得预先替 Lift 做 J/C/π 语义分类。

匿名 handle 是机械 wire contract：J tensor 必须按 `intent.inputs` 后接 `intent.outputs` 独立、
无间隙地编号为 `t0,t1,...`，不得把 `p*` 参数位置直接复制为 `t*` 后缀。例如 input pointer `p0`、
output pointer `p1`、runtime extent scalar `p2` 对应 input `t0`、output `t1` 与 dimension `x0`。
`harness_compiler_view` 中 runtime scalar 的当前 value 只是本次 Harness binding；若 Source 用它控制
logical extent、index、predicate 或 result，J 必须保留 symbolic `x*` 并在相关 tensor shape 中引用，
不得把当前 value 特化成 J 常量。`x*` 是 signature/type handle，不是 SSA value，不能进入 portable
scope definitions/captures 或 storage。

`J.signature.logical_ref` 只允许指向声明的 J input/output tensor 或 symbolic dimension，不能指向
body 内部 `v*` SSA value。logical runtime scalar 或 non-C semantic constexpr 若作为 predicate、
operator 或 result 的逻辑数据，必须声明 dtype 相容、shape `[]` 的 rank-0 J input，并让该输入实际
进入 J dataflow；只有 integral extent-only handle 才绑定 `x*`。value 参数不能为了复用现成 ID 而
绑定方向或 dtype 不相容的 output tensor。

对 ordinary dense logical access，Lift 只需使用 closed ContractExpr surface grammar。`k0`/`[k0]`
与 `k1,k0`/`[k1,k0]` 分别由固定 elaborator 归一为相同 structured axes；这类标点差异不触发
relift。predicated pure functional boundary 的 observable store producer 类型若能由 closed op
signature 与 J graph 唯一确定，可以省略；显式类型必须一致。direct same-handle identity mapping 可
省略或显式写出，canonicalizer 统一为省略形式；非恒等 `returns`/mapping 仍必须明确且一致。
`fill_value` 只允许出现在带非空 predicate 的 load 上；无条件 access 的
predicate/fill 均为空。static schema 可以提前报错，TypeCheck 仍必须权威拒绝 orphan fill，消费者
不得静默删除。这些是全局 J wire/consumer 合同，不是 per-kernel Replay hint。

provider 已返回、但因 schema/匿名合同不合法而被拒绝的响应不进入 TypeCheck，也不计正式 Gate1
round；Runner 对这类 pre-Replay rejection 使用有限调用上限并如实报告。拒绝响应、receipt、耗时
和 token 作为 proof-irrelevant sidecar 保存，不能只留下一个错误字符串，也不得进入 Replay 或
认证谓词。上限耗尽可以结束为 `NO_CLOSURE`，但报告必须显示正式 Gate1 round 为 0，而不是伪造
一次 Gate1 失败。worker 连 candidate 都没有提交时同样属于 pre-TypeCheck Lift rejection；报告必须
显式显示 submitted candidate、canonical decode、TypeCheck、Replay 与 Gate 执行数均为 0，不能
把 aggregate `NO_CLOSURE` 解读成一个已经进入 Gate1 的 artifact 未闭合。

### 有界 agent 实现

Lift 可以由仓库内的有界 mini-swe agent 实现，但其 agent loop 只属于一次生成动作：

```text
一次 LiftAdapter 调用
  = 一个全新 agent session
  = 一个独立受限 workspace
  = 最多提交一个完整 (J,C,pi)
```

该 session 内可以多次调用模型，按需读取匿名 Source、最小 Harness compiler view、闭合 JSON
Schema、source-neutral Lift 合同与短检查表，在私有草稿目录中修改 candidate，并运行 JSON
语法/静态 Schema 检查。这是普通文件上的轻量 retrieval；不需要 vector database，也不能按
当前 Kernel、历史失败或 sibling candidate 动态选择知识库。

session 内不得调用 canonical artifact parser 的依赖检查、TypeCheck、Replay、compiler observer、
Gate1、Gate2 或 RefEval。提交 `candidate.json` 后 session 结束；任何 parser/TypeCheck/Gate1 反馈
只能由 Runner 显式放入下一次全新 Lift 调用，不能恢复旧 conversation 或旧可写 workspace。
因此 agent 内部多步不增加正式 Gate round，也不削弱门禁；报告必须分别记录外部 Lift invocation、
agent model turns 与实际 provider calls。

若完整 JSON object 已提交但被外部 canonical parser 拒绝，Runner 可以把该完整 rejected candidate
和与它精确对应的机械 diagnostic 交给同 branch 的紧邻下一次 fresh Lift invocation。新 worker 应
在完整对象上做聚焦修正并重新提交完整 `(J,C,pi)`；它不能恢复旧 trajectory，也不能只交 patch。
rejected candidate 不是第四 artifact，不进入 Replay 或 certificate。没有提交对象、对象不是有限
JSON、feedback 不匹配或中间隔着其他 invocation 时不得复用更早候选。

agent shell 必须 fail closed 地隔离：`/input` 只读、`/output` 只允许唯一 candidate、私有 `/work`
可写、网络 namespace 隔离、宿主环境清空，且不挂载仓库、历史 evidence、Gate executable 或
provider secret。隔离设施不可用时报告 infrastructure failure，不得降级到宿主 LocalEnvironment。

> 实现状态（2026-07-23，不是认证证据）：第一次 DYT canary 执行了 3 个 fresh MiniSWE Lift
> invocation，每个达到 8 model turn，共 24 次物理 provider 调用；三次均在浏览 schema 时耗尽预算，
> 未提交 candidate，因此 canonical decode、TypeCheck、Replay、Gate1 和 Gate2 均未执行。该记录
> 只用于改进 Lift worker，不改变任何认证谓词；详细数据见 Task04 的 run-001。
>
> 第二次 DYT canary（run-002）使用 knowledge-v2：3 个 fresh worker 均在第 9 turn 提交完整
> candidate，证明“只浏览、不生成”已解决；但三份 candidate 都在 canonical signature binding
> 检查被拒，TypeCheck/Replay/Gate 仍为 0。该结果定位了 schema 中允许 `logical_ref` 指向任意
> “value”的旧文字与 canonical parser 只接受 tensor/dimension 的冲突，详见 Task04。
>
> 第三次 DYT canary（run-003）使用 knowledge-v3：3 份 candidate 均通过 canonical decode 与
> TypeCheck，第 3 个 formal round 获得 normal Replay closure，masked-J mismatch 证明 J necessity。
> Gate2 的 Source--Replay 腿通过且误差为 0，但当时 RefEval 因二维 dense access 使用未括号化
> `k1,k0` 而 `REFERENCE_UNSUPPORTED`。Task06 随后把它确认为 response decoding/elaboration 缺口：
> 原 artifact 不经 relift即可规范为 `[k1,k0]`、物化唯一 result type、消去冗余 identity mapping，
> 并通过同一 RefEval boundary。该后续系统修改不回写 run-003 的历史 verdict。

## Replay 前检查

```text
response JSON -> Decode/Elaborate -> (J,C,pi)
TypeCheck(J,C,pi)
```

Decode/Elaborate 是 Replay 前固定代码的一部分，不是 Agent/Gate，也不产生第二份 J。它不读
Source/C concrete values/Gate output，只处理唯一可判定的
surface normalization、handle/type elaboration 和恒等表示消除；无唯一解时 fail closed。

最小 TypeCheck 必须机械检查：

1. J access 满足 `fill != null => load and predicate != null`；
2. 所有 J/C 引用存在；
3. route operand slot 与类型兼容；
4. observable instance 锚定 J-owned node/access/effect；
5. π 不含语义原子、Source/API、C value 或可执行自由文本；
6. sensitive refinement 引用 J/B authorization，或属于系统级安全白名单；
7. 未知字段、未知 relation 与自由文本扩展 fail closed。

TypeCheck 不做 SMT、range proof 或一般语义等价证明。检查失败时当前 artifact fail closed，
不进入 Replay。它只支撑 `FreshLogicalSemDecl(C,pi | J)=empty` 的显式所有权结论，不证明 π
没有统计语义线索，也不证明 realization 行为正确。

## Replay

```text
ReplayLLM(J,pi,MaskC(C);U_f) -> complete source R_f
```

`U_f` 是版本化、source-neutral 的公共前端文档与工具上下文，不按当前 Kernel/op/pattern/失败
样本选择。Replay 看不到 Source、Harness、C values、source parameter name/position、真实
BindingMap、独立 interface sidecar、source/compiler fingerprint、旧候选、旧 Replay、Gate1 diff、
旧模型 conversation context 或 per-kernel RuleCard。`MaskC(C)` 只暴露 C 的匿名 schema/slot
视图，不暴露 valuation 与私有 provenance。

Replay 必须生成完整、可独立构建的当前前端源码；固定代码只做绑定和检查，不根据 π 或 compiler
diff 施工。

π region placement 不是隐式代码生成指令：`placement=program` 不等于存在 `program_id`。只有 π
中另有一个表示 compiler-visible program/grid coordinate 的独立 `realize` instance 与 routes 时，
Replay 才能生成对应 stage；不得按常见 kernel 模板从 placement、grid 惯例或 C 猜出额外 grid
offset、整数乘加或 affine address。

现有关系的机械解释是：锚定 J domain、位于 program region 且 scalar dependency path 进入
`address_base` 的 `realize` stage 是 program coordinate；进入 `address_lane` 的 domain/C path 是
lane coordinate。Triton `U_f` 按声明顺序把前者映射为 `tl.program_id(axis)`，后者才使用
`tl.arange` 等 lane realization；placement 单独出现时两者都不能推断。Triton 的 `tl.arange` 等
compile-time-only API 必须直接读取仍带 `tl.constexpr` 注解的匿名 `c*` 参数，不得先复制到普通
Python local 再传入。单个锚定 `c*` 的 `realize` instance 由参数 binding 和直接 routed use 即可
满足，不要求恒等 alias。`IntentFunction.reduce.axes` 是 logical axes；匹配的
`J.boundary.reductions.tile_axis` 才是 Triton reduction call 使用的物理 tile axis。这里都是固定、
source-neutral 的前端合同，不是 per-kernel hint。

base/lane 是 address contribution 角色而不是 pointer-add 计数。Source 若形成可分离的
program-uniform integer base 与 lane offset，并在一次 pointer add 前合并，π 仍保留 program path
进入 `address_base`、再 route 到 `address_lane` 的拓扑；Replay 保留两个 integer stage、一个
pointer add，且即使当前 grid extent 为 1 也不能用常量替换 program coordinate。只有 Source 没有
可分离 program-uniform contribution 时才允许 lone `address_lane`。

## Gate1 失败与 relift

Gate1 的结构化 diff 只返回 source-aware Lift。下一轮 Lift 可以读取 Source、上一轮完整
`(J,C,π)` 与累计 Gate1 diff，但必须重新输出完整新三元组。随后使用新三元组发起全新的盲
Replay。

runner 必须完整保存 observer comparison 作为审计 evidence；返回 Lift 的可以是该 comparison
的确定性、有限、可行动诊断投影，例如 hard category、op-count 差异、关键 topology 差异与每类
少量代表边。该投影不得改变 Gate1 verdict、隐藏失败类别或替 Lift 修改 artifact；低层节点
fingerprint、重复 components、score/rank 等非行动信息不应占满 relift 上下文。完整 evidence 与
relift diagnostic view 必须明确区分。

禁止把 diff 交给 Replay、patch 旧 Replay、只输出 artifact 增量，或跨候选复用 Replay。

当前同一 case 使用 `B=5` 个独立并行 branch；每 branch 最多 `K=3` 个 TypeCheck-valid 正式
Gate1 round，`K` 包含首次候选。每个 branch 的 previous artifact、Gate1 diff、Lift conversation、
Replay writer 与调用预算完全隔离；sibling candidate 不能进入 Lift/Replay 输入，也不能拼接为一个
artifact。首个正式 Gate1 closure exactly-once 成为 winner，并立即生成隐藏 Gate2 输入和执行 Gate2，
不等待 fanout barrier。其他 branch 已开始的 provider call自然返回，在紧邻阶段边界记录为
`SUPERSEDED` 并停止，不再启动新的 Lift/Replay/relift；它们不能冒充 `NO_CLOSURE` 或 infrastructure
failure。只有没有 winner 且所有未被外部停止的完整 branch 都未闭合时才报告
`NO_CLOSURE_(5,3)`；infrastructure-incomplete branch 必须单独报告。
