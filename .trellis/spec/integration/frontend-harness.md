# Frontend Harness 合同

> 状态：当前派生规范。上位设计见 [C 文档](../../../doc/IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)。

Harness 是预先存在的运行与调用环境，不是 IR、模型产物或第四个 proof artifact。

## 最小职责

```text
EnumerateBindings(source invocation) -> anonymous coordinates
BindSource(Source,C,inputs)           -> source invocation
BindReplay(Replay,C,inputs)           -> replay invocation
Build(invocation)                     -> executable
Run(executable)                       -> observable result
Observe(executable/result)            -> Gate evidence
```

Harness 可以保存：

- concrete logical inputs 与 C values；
- source/replay 的机械调用坐标；
- grid、device、toolchain 与 compiler options；
- build/run/observation diagnostics。

Harness 不得：

- 根据参数名、`constexpr` 或 Kernel family 判断 J/C/π 归属；
- 生成 operator、predicate、formula、schedule strategy 或 Replay code；
- 把真实 Source binding、compiler diff 或 H 整体交给 Replay；
- 建立独立 `LogicalABI`、`InterfaceArtifact` 或带语义的 BindingMap。

Lift session 的只读 `CandidateCheck_v(candidate_bytes,ctx)` 只能通过显式 `ctx` 接收当前 agent 已经
获准看到的匿名机械 compiler view 与 schema/decode/typecheck version，以检查 candidate 是否篡改
参数 kind/mode/dtype/rank、C valuation 或 anonymous handle 合同。工具禁止读取 ambient host state，
也不得把 hidden inputs、真实 Source binding、Gate evidence 或额外 Harness 状态返回 agent，不得替
Lift 生成 `logical_ref`、C `subject_ref/axis_ref`、J/C 分类或任何 artifact 字段。它共享正式 Runner
的同一 source-free parser/Decode/anonymous-view/TypeCheck callable core；正式 Runner 另执行
source-identifier privacy、receipt/provenance，并在提交后重新运行该 core。不得维护
Harness-private 近似规则。

## Logical signature 与绑定

input、output、runtime scalar、destination、dtype、rank、shape、alias/inout/effect 属于 J logical
signature。adapter 若需要函数视图，只能使用 `InterfaceView(J)`，且该视图必须由 J 确定性产生。

C 是一个对象。Source 与 Replay 在同一候选中使用同一 C valuation；`MaskC(C)` 只给 Replay
提供匿名 slot/schema。non-C compile-time value 作为调用事实绑定，但其 semantic consequence
必须已经在 J 中表达，其 realization use 才可由 π 引用。

该 valuation 记为 `c0 = Bind_H(C)`。Gate1/Gate2 的主证书只覆盖实际 c0；测试多个配置时只能
报告实际覆盖的有限集合，不能据此声称所有 admissible C 都已认证。

## 三方执行权限

```text
Source  <- H binds C and inputs
Replay  <- H binds the same C and inputs
RefEval <- logical inputs only
```

`RefEval` 的函数接口不能接收 C 或 π。实验记录可以关联当前候选，但关联 metadata 不得变成
RefEval 的功能输入。

新增前端首先实现上述薄 adapter；不要求 J-to-frontend lowerer、RuleCard、source-to-π parser
或 Kernel-specific mapping table。observer 强度可以逐步增强，并必须记录 frontend、observer
profile/version、compiler/dump stage 与 canonicalization version。

## 高性能工作负载投影与有界隔离批量入口

SCRC 的高性能工作负载清单只能是现行 lowering capability registry 的确定性 Harness-side
投影。direct kernel 的 ID、ordinal 与总数继续由
`spec/lowering/high-performance-kernels.json` 及其 typed loader 唯一决定；SCRC 层不得复制第二份
28 项 truth，也不得把 scenario 29 计作 direct kernel。

投影可以附加 source module/file、entry symbol、frontend、observer/Q 版本、现有 lowering evidence、
adapter readiness 与代表性 tier。这些字段只用于定位真实 Source、选择 Harness adapter 和组织稳定
运行，不是 `(J,C,π)`、TypeCheck 或 Gate predicate 的输入。特别是，source path、Harness factory、
concrete C valuation、BindingMap、历史 diff 和其他 case 的会话内容都不得进入 Replay packet。

离线 manifest 校验必须做到：

- 从现行 registry 逐项构造投影，并机械检查 `1..28` ordinal 与 ID 唯一性；
- source file 可解析且目标函数定义恰好出现一次；
- optional dependency 或 Dataset 缓存缺失记为 `CASE_UNAVAILABLE`；
- source/schema 不一致记为 `MANIFEST_INVALID`，不得伪装成 Gate 失败；
- list、validate 与 dry-run 不调用 Lift/Replay provider。

批量入口的逻辑顺序由 registry ordinal 唯一决定；实际执行可以串行，也可以在明示资源上限内对
多个 case 做有界并行。无论调度方式如何，每个 case 都必须使用独立的 fresh Lift/Replay 生命周期和
独立 evidence 目录。单个 case 的准备、编译或认证异常必须收敛为该 case 的结构化结果，不得
取消、污染或改写其他 case 的结果。汇总层最终按 registry ordinal 稳定排序，只统计既有
TypeCheck、Gate1、Gate2、main status、J-necessity status 与真实调用预算，不能重新定义认证条件。

跨 case 并行的最小隔离单位是完整 case lifecycle：每个 case 分别新建 Harness 可变执行状态、Lift
session、provider session、Replay writer、TypeCheck/Gate runner、relift history、输出 buffer 与 evidence
目录。只读 registry、source file 和不含 case 状态的 adapter 定义可以共享；candidate artifact、provider
response/conversation、Replay、diff、winner、C valuation 和任何可变 Harness 状态不得跨 case 共享。
资源争用或 provider 限流必须记为相应 case 的执行证据，不得通过交换候选或压缩隔离边界
规避。

同一 case 内的候选 fan-out 仍按同一 Source、同一 Harness 合同与同一 `c0` 建立 `B` 个互不
重叠的 branch evidence 目录。每个 branch 新建独立 Lift、Replay writer、TypeCheck/Gate1 runner 与
relift history；Source slice 和只读 adapter 描述可以共享，但 candidate、provider response、Replay、
diff 与输出 buffer 不得在 branch 间共享。

Harness 不能因此升级为 `HarnessAgent`：它不选择 operator、修 J/π 或组合候选。fan-out 的
pre-Gate2 wave barrier 完成后，由受信任 runner 只依据 TypeCheck/Gate1 evidence，按固定
`(branch_id, formal_round)` 顺序选择一个 winner，再由 Harness 为该 winner 生成隐藏 Gate2 输入；
wall-clock 完成先后不参与选择，其余 branch 不读取这些输入。

## 真实 Triton Source 投影与声明式 adapter

真实工作负载进入统一 SCRC 生命周期前，Harness 可以从已解析的 source file 机械构造 fixed-C
Source 投影。该投影只允许：

- 选择 registry 指定且唯一的顶层 Triton entry；
- 保留其必要 import、模块常量与同模块/可解析 imported `@triton.jit` helper 闭包；
- 去除只负责搜索 C valuation 的 autotune/heuristics decorator；
- 对 source AST 的参数顺序、runtime/constexpr 分类和 adapter formals 做 exact ABI 校验。

它不得改写 Kernel body、预先解释 operator/predicate/formula，也不得把搜索 decorator 的配置变成
新的 artifact。若 helper、decorator 或 ABI 不能机械收敛，case 必须报告 `HARNESS_INVALID` 或
`CASE_UNAVAILABLE`，不能带着猜测进入 Lift/Gate。

Source 投影不能因当前 `c0` 恰好让某个 grid axis 的 extent 为 1，就删除该 axis 的
`program_id`、address-base 或其他 compiler-visible realization topology；Gate1 比较的是本次真实
编译实例，但 fixed-C 只消除调度搜索外壳，不授权语义化简 Kernel body。

真实 adapter 使用声明式 buffer/scalar/constexpr/C/grid binding。compiler view 可以为 runtime
argument 附带 `input`、`output`、`inout`、`value` 等机械调用模式；这些模式只来自实际 Harness
buffer role，用于检查 J logical signature 是否覆盖真实调用 ABI，不表示 operator 语义，也不能
进入 Replay packet。adapter 不能拥有 per-kernel Replay、Python semantic oracle 或第二份 J。

scalar/constexpr 的归属按以下机械边界处理：

| 调用事实 | 归属/锚点 | 三方执行 |
|---|---|---|
| logical runtime scalar 或 semantic constexpr | `J.signature` 的 rank-0 handle | Source/Replay 按 ABI 接收 scalar；RefEval 接收同值 rank-0 input |
| logical dimension/extent | J 的 dimension handle | 只绑定 shape/domain，不伪装成 rank-0 data tensor |
| schedule constexpr | C slot | Harness 绑定实际 `c0`；Replay 只读取 `MaskC(C)` |

六个代表项 adapter 只是现行 28 项 registry 上的 Harness 配置视图；它们不得演化成另一份 Kernel
清单。每次准备仍须重新解析 registry source、检查 entry/ABI，并在真实 CUDA/Triton frontend 上
使用同一份 Source 投影与实际 `c0`。
