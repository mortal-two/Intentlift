# IntentIR：J 主导的 π_min 与 Episode-Local Composer Replay 最终设计

> 状态：Replay 子系统权威设计增补（2026-07-31）。
>
> 本文服从并细化四份 SCRC 权威设计，不改变其数学对象、输入因子化或认证谓词：
> [C 所有权](IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
> [π 依赖型实现见证](IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)、
> [Lift/Replay/Gate 执行流程](IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与
> [SCRC 数学总述](IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。
>
> Production backend 仍以
> [J-Grounded、Q/D-Certified Lowering-State Search](IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)
> 为权威；它只消费 accepted `J+C`，不读取 π 或 Composer 状态。

## 0. 裁决

本次重构采用以下唯一版本：

```text
Lift_H(Source) -> (J,C,π_min)

Replay_f = FreshEpisode_f(
    exact J,
    exact π_min,
    MaskC(C),
    fixed public U_f
)

FreshEpisode_f 内部：
    source-blind Codex
      -> episode-local plan/build script
      -> syntax-only composer
      -> exact replay.source
      -> replay check / compile / inspect own compiler IR
      -> local edit / rollback / resubmit

SCRC_CERTIFIED
  iff TypeCheck(J,C,π_min)
  and Gate1(Source(C),Replay(C))
  and Gate2(Source(C),Replay(C),RefEval(J))
```

这里的 `π_min` 不是第四种 artifact，也不是新 wire 名称。当前 `artifact.v12` 已经满足首版的关键
所有权条件：π 只有 closed、J/C-dependent realization relations，没有 op、operand graph、dtype、shape、
predicate、formula、Source/API 或 target recipe。因此第一阶段直接把 v12 作为 `π_min-v1` 的兼容基线，
不为了“看起来是重构”而无证据地重写 schema。

本次真正改变的是 Replay 的构造机制：允许 source-blind agent 在自己的 fresh episode 内使用一个
**纯语法、可局部编辑、可回滚的 Composer**，但固定宿主仍不得把 J/π 解码成目标代码或算法。

## 1. 第一性边界

三个问题必须始终分开：

```text
J       回答：程序计算什么？
π_min   回答：Source realization 必须满足哪些 J-dependent 物理约束？
Composer回答：当前 agent 决定的目标源码怎样稳定地组织、编辑和渲染？
```

任何一层替代另一层都会退化：

- π 复制 operator/dataflow，会成为第二份 J；
- π 保存 `tl.dot`、`T.Kernel` 或 CUDA recipe，会成为 target lowerer；
- Composer 根据 J/π pattern 自动选择算法，会成为隐藏 deterministic lowerer；
- J 保存 Source 的 loop、thread、materialization recipe，会丢失跨前端语义独立性；
- backend 读取 π，会退化为复刻 Source realization。

因此本设计的最重要不变量是：

> **J 是唯一显式语义；π_min 是只在 J/C 环境中有意义的 realization delta；Composer 只操作 agent 已经作者化的语法。**

## 2. π_min 的严格定义

### 2.1 Dependent type

```text
Γ_(J,C) ⊢ π_min : RealizationConstraint(J,C)
```

删除 J/C 后，π 中的 `n* / v* / a* / e* / d* / c*` 只是不透明 handle，不获得独立逻辑含义。

### 2.2 可以拥有的事实

π_min 可以表达且只表达：

- physical region 与 lexical placement；
- 一个 J/C handle 有几个 compiler-visible realization instance；
- instance 间 route、operand slot、order 与 sharing；
- materialization、duplication、reuse 与 representation-only staging；
- program-coordinate domain 到物理 axis 的 identity；
- C slot 在 source realization 中的匿名 attachment/use；
- structured J loop 的 physical induction basis；
- J 外 physical iteration 与 typed temporal carrier topology；
- terminal arm 与 lexical continuation 的 topology；
- 已有 closed route 上的受限 physical extent partition；
- 由 J/B authorization 约束的 sensitive representation refinement。

### 2.3 永远不能拥有的事实

π_min 不得包含：

- J operator、operand graph、reducer、subscripts 或 op attrs 的副本；
- dtype、shape、logical domain、predicate、fill、index expression 或 result relation 的副本；
- Source/TTIR/TIR/PTX 文本、名字、路径、span 或 kernel/family ID；
- Triton、TileLang、CUDA 或其他 target API/recipe；
- address formula AST 或一般表达式语言；
- C concrete value；
- 可执行自由文本或能脱离 J 独立恢复 dataflow 的 plan。

允许：

```json
{"id":"i7","semantic_ref":"v17","region":"r2","mechanism":"realize"}
```

不允许：

```json
{
  "semantic_ref":"v17",
  "op":"contraction",
  "subscripts":"bhij,bjhd->bihd",
  "emit":"tl.dot(ds, k)"
}
```

### 2.4 “min”的工程含义

`min` 表示 **相对于 exact J 的 realization delta**，不是以最少 JSON byte 为目标的压缩竞赛。

默认不应在 π 重复：

- J 已确定的 op、operand、type、shape、predicate、effect；
- 一对一且无额外 realization 意义的语义描述；
- target-independent replay 可以从 J 唯一读取的逻辑 dataflow；
- frontend 公开 API 的具体拼写。

但只要 Gate1 需要区分 instance multiplicity、region、order、sharing、physical time 或
materialization，该 relation 就不是冗余。不能为了减小 π 把 Source realization topology重新交给
Replay 猜测。

首版不删除 v12 relation。先对真实 kernel 记录每类 relation 的消费与 Gate1 必要性；只有跨样本证据
表明某字段可由 J/C 全局唯一推导或应由 observer 统一 NormalizeAway，才做版本化 schema 收缩。

## 3. Composer 的准确身份

### 3.1 它是什么

Composer 是 fresh Replay episode 内的 proof-irrelevant editing mechanism：

```text
ComposerState_f
  = ordered syntax nodes
  + stable node ids
  + parent/block relation
  + source map
  + local revision/checkpoint
```

它只帮助 agent 完成：

- 创建函数、参数、statement 与 nested block；
- insert/delete/replace/move；
- indentation/braces 与稳定渲染；
- node id 到 rendered line 的 source map；
- exact render、compile、checkpoint、rollback；
- `raw_expr/raw_stmt/raw_block` 逃生。

最终进入 Gate 的仍只有 `output/replay.source`。Composer state、plan、builder script、source map 和编辑
trajectory 都不进入 `(J,C,π)`、Replay packet、Gate verdict、Gate2、certificate 或 production backend。

### 3.2 最小实现，不造第二个前端

Composer v2 只提供一个很小的 structured-text core。agent可以直接作者化严格JSON plan，也可以在
本次episode自己的`work/build_replay.py`中调用同一组通用构造器；两者产生同一plan/render结果：

```text
document
  line(exact agent-authored text)
  block(opener, closer, children)
  raw_block(exact agent-authored lines)

edit
  insert / replace / delete / move
  render / checkpoint / rollback
```

Python-like 与 C-like profile 只定义 indentation、block closing 与基本 lexical validation。它不尝试
覆盖完整 Python、Triton、TileLang 或 CUDA AST；新语法直接使用 raw escape。这样新增 target feature
不要求先扩建 Composer grammar。

### 3.3 允许的通用语法 convenience

在不读取 J/π 的前提下，可以提供：

```text
create_function
add_parameter
assign
call
for
if / else
return
load/store 形式的普通 call node
render
```

例如 agent 明确给出 `call("tl.dot", ...)` 是允许的，因为 target API 和参数选择完全由 agent 作者化；
Composer 只是渲染一个调用。宿主看到 J contraction 后自行调用 `tl.dot` 则被禁止。

### 3.4 禁止的答案型模板

以下能力永远不进入宿主 Composer：

```text
emit_flash_attention_backward(...)
emit_online_softmax(...)
lower_pi_contraction(...)
lower_update_privatized(...)
select_cuda_reduction_strategy(...)
```

也禁止：

- `J op kind -> target API` 映射表；
- `π mechanism graph -> source skeleton` 展开器；
- kernel/family template registry；
- 按历史 failure 选择 recipe；
- 跨 episode 保存 semantic helper 或成功 Replay；
- Composer 读取 Source、Gate1 diff、C values 或正常臂 Replay 后构造 masked 臂。

Replay agent 在自己的 episode 中临时写 `emit_online_step()` 或任何 helper 完全允许；它是模型阅读
当前 exact J 后的本次作者行为，episode 结束即销毁。

### 3.5 `off / available / preferred`只控制作者体验

为做同artifact效率实验，Codex Replay显式记录三种proof-irrelevant模式：

```text
off        不安装Composer，agent直接作者化完整source
available  安装builder/plan/editor，agent自行选择是否使用
preferred  要求agent优先从build_replay.py开始；遇到语法摩擦仍可raw escape
```

三种模式获得完全相同的`J+π_min+MaskC(C)+U_f`，使用相同模型、effort、时间预算、ReplayCheck、compile、
Gate1和Gate2。`preferred`不能删除raw escape，也不能增加semantic scaffold。telemetry必须分别记录
`offer_mode`、成功/失败action与最终`selected_path=composer|raw_escape`；仅安装不算使用，失败的edit也
不算使用。

## 4. Fresh episode 与 exact J 绑定

### 4.1 每个 formal round 都是全新 Replay

每个 TypeCheck-valid formal round 必须创建：

- 新 provider conversation/thread；
- 新受限 workspace；
- 新 Composer state；
- 新 plan/builder script；
- 新 output source；
- 新 compile/checkpoint history。

不得继承：

- 旧 Replay source；
- 旧 Composer/plan/source map；
- 旧 compiler error 或 Gate1 diff；
- sibling branch/case 的任何状态；
- 旧模型 conversation。

同一 episode 内可以持久编辑、编译、查看自己生成的 IR并回滚，因为这些信息只来自当前
`J+π_min+MaskC(C)+U_f` 与自己作者化的源码。

### 4.2 J 改变后为什么一定跟随新 J

三层共同约束：

1. `TypeCheck(J,C,π)`：旧 π 中失效的 handle/type/scope/ownership 直接拒绝；
2. fresh Replay：没有旧 Replay/Composer/conversation 可复用，只能读取当前 exact J；
3. Gate2 三边：即使 Replay 猜回旧 Source，`Replay–J` 仍会暴露没有跟随当前 J；若当前 J 本身错，
   `Source–J` 暴露错误。

所以“证明跟着 J 走”不由 Composer 自报，而由 exact artifact、fresh construction 与三边 Gate2 共同
强制。

## 5. Mask-J 与 J 必要性

### 5.1 验收目标

Mask-J 测量的是：

> 当前模型在只得到 π topology、MaskC 与 U_f 时，能否稳定恢复 Gate1-closing Replay。

它不要求 topology 对 kernel family 完全零统计信息，也不取代 Gate2。

### 5.2 必须独立 fresh

正常臂与 masked 臂必须分别创建独立 Replay episode：

```text
normal episode:
  exact J + exact π + MaskC(C) + U_f

masked episode:
  MaskJ(J) + exact π + MaskC(C) + U_f
```

masked 臂不得复用 normal writer、thread、workspace、Composer、draft、own-IR、compiler diagnostic 或
checkpoint。两臂只在外部 evidence 中关联同一 exact artifact/case，不共享构造状态。

### 5.3 状态解释

```text
normal Gate1 close + masked hard mismatch -> J_NECESSARY=PROVED
normal Gate1 close + masked hard close    -> J_NECESSARY=NOT_PROVED
masked generation/tool failure           -> INCONCLUSIVE
TypeCheck failure                         -> artifact invalid
```

`NOT_PROVED` 不自动否定主认证。论文可以另外报告 normal/masked closure gap；不能把 masked close 直接
写成语义泄漏，也不能把 masked mismatch 写成对所有模型的不可替代性定理。

## 6. 三前端 Composer

共享核心只操作 syntax nodes；每个 frontend 只有薄 profile、公开 `U_f` 与已有编译/观察适配。

### 6.1 Triton

```text
source form: Python + @triton.jit
hard observer: normalized TTIR
compile feedback: replay-check + synthetic compile + own TTIR
```

profile 只处理 Python block/indent、允许 import/decorator/function surface 和公开 Triton 安全政策；
`tl.dot/tl.load/tl.store/tl.trans/...` 的选择仍由 agent 读取 J/π 后完成。

### 6.2 TileLang

```text
source form: Python + @T.prim_func + T.Kernel
hard observer: alpha-normalized structural TIR
diagnostic observer: normalized PTX
compile feedback: TIR first, PTX diagnostic
```

TileLang closure 以 TIR 为准，PTX 不要求逐指令或逐地址拼写相同。Composer 只帮助组织 Python/TIR
surface；它不把 π loop 自动映射成 `T.serial/T.parallel/T.vectorized`，也不选择 fragment/shared layout。

### 6.3 CUDA

```text
source form: standalone CUDA C++
hard observer: declared normalized PTX semantic topology profile
Gate2: native CUDA execution + three edges
```

profile 只处理 C-like braces、function/helper surface与现有安全 allowlist；不选择 thread/block mapping、
shuffle、shared memory、barrier、atomic、MMA 或 reduction recipe。CUDA PTX observer按现有 contract 放松
cache、register、label与纯 address spelling，但保留 observable value、memory/effect、control、coordinate、
collective、ABI 与 target hard evidence。

## 7. Runtime 数据流

```text
exact typed artifact
  -> build source-blind packet
  -> create fresh isolated Replay workspace
  -> install immutable packet / U_f / syntax composer API + writable builder scaffold
  -> start one source-blind Codex owner
  -> agent authors plan or build_replay script
  -> composer renders candidate bytes
  -> existing exact patch/check/compile/checkpoint/rollback
  -> exact submit_replay
  -> destroy workspace after evidence persistence
  -> Harness binds C
  -> Gate1
```

Composer 不增加新的认证步骤。`render` 只产生候选 bytes；`submit_replay` 仍必须对 exact bytes运行已有
ingress、ReplayCheck与compile preflight。Gate1始终独立重编译并比较 Source/Replay observation。

## 8. 静态反泄漏与运行审计

### 8.1 π 静态审计

现有 closed schema 与 TypeCheck 继续作为权威。额外 audit 只能做确定性报告，不能成为第二套宽松
validator。至少记录：

- π 中所有 key/value 是否属于当前 closed grammar；
- 所有 semantic refs 是否为 opaque J/C handle；
- 是否出现 Source/API/kernel/family/free-text 字段；
- 是否出现 J op/attrs/type/shape/dataflow 的显式副本；
- route-extent closed constructor 是否满足既有 dependent contract。

若审计只是重述 TypeCheck，不必升 artifact/typecheck protocol；若改变接受集合，必须窄升版本并对
fresh run生效，历史证书不重判。

### 8.2 Composer 审计

每个 episode 记录 proof-irrelevant telemetry：

- frontend profile、Composer protocol与`off/available/preferred` offer mode；
- node/edit/render/compile/rollback 次数；
- 成功/失败action、最终Composer/raw escape选择；
- time-to-first-render、first-compile 与 exact submit；
- 最终 source bytes与 source map；
- workspace fresh/cleanup 状态；
- normal/masked episode 是否物理隔离。

这些字段用于检查机制是否真正降低构造成本，不决定 candidate 合法性，不使用 hash/commitment 作为
进度或 acceptance gate。

## 9. 验收矩阵

### 9.1 机制级负例

必须证明宿主 Composer不能：

- 只给 J/π 就自动生成任意 target source；
- 从 J op kind 选择 target API；
- 从 π topology 选择 kernel template；
- 跨 episode恢复 draft/helper；
- 让 masked 臂读取 normal 臂状态；
- 把 Composer state送入 Gate、RefEval或backend。

### 9.2 真实纵切

不用 archived `python/tests`。首批使用真实 dataset/registry与真实 provider，至少覆盖：

1. `add2d`：elementwise、二维 index、load/store；
2. `row_sum`：reduction、physical/local axis；
3. `gather2d` 或 masked access：predicate/fill/index boundary；
4. 一个 sharing/materialization 或 physical-loop case；
5. 一个 effect/control case（相应前端真实可运行时）。

每个可用 frontend 都必须报告：

- exact artifact 与 TypeCheck；
- fresh source-blind provider运行；
- Composer trajectory；
- compile/observer结果；
- Gate1；
- Gate2三边；
- masked-J独立episode结果；
- 失败分层：Lift信息缺失、Replay消费失败、frontend compile/observer不足或Gate2 consumer不足。

### 9.3 性能与能力指标

核心比较不是“生成源码行数”，而是：

```text
time-to-first-compilable Replay
compile/fix cycles
Replay provider turns and input tokens
closure@1 / closure@2 / closure@3
normal-vs-masked closure gap
cross-frontend shared composer LoC vs thin adapter LoC
raw escape rate
```

同artifact A/B必须通过diagnostic-only fresh Gate1 recheck运行，不重跑Lift、不复用Replay会话、不进入
Gate2，也不改变parent verdict。若 Composer 只增加工具调用、token或固定规则而不提高 first-compilable/closure，应删除或继续收缩，
不能因已经实现而保留。

## 10. 实施顺序

### P0：文档与协议对齐

- 建立本文与 Task 24；
- 修订旧文档/spec中“禁止 composer”的歧义：禁止的是 deterministic semantic composer，允许
  episode-local syntax-only composer；
- 明确 v12 是 `π_min-v1` 兼容基线；
- 不调用 provider，不产生 closure claim。

### P1：共享 Composer core

- 实现最小 structured-text tree、stable id、edit、render、source map、checkpoint/rollback；
- 复用现有 Replay workspace 的 exact patch/commit/compile/rollback；
- raw escape覆盖未知语法；
- 无 J/C/π/Source/Gate import 或 semantic callback。

### P2：三前端薄 profile

- Triton Python profile；
- TileLang Python/TIR profile；
- CUDA C-like profile；
- 接入各自已有 compile/observe，而不复制 target semantics。

### P3：fresh normal/masked isolation

- normal 与 masked 分别构造 writer/workspace/conversation/composer；
- 记录独立 episode evidence；
- 保持 masked-J 非主 Gate 的状态语义。

### P4：真实 pilot 与横向闭合

- 先跑 add2d/row_sum/gather2d 三前端纵切；
- 按共同失败实现通用机制，不按 kernel 加模板；
- pilot稳定后闭合 TileLang/CUDA选定组；
- 若机制明显降低成本，再用 fresh Replay诊断或重跑Task 15剩余项/已闭合代表项，验证对Triton也有
  真实收益。

## 11. 明确非目标

- 不新增第四 proof artifact 或 Composer IR；
- 不修改 `SCRC_CERTIFIED = TypeCheck ∧ Gate1 ∧ Gate2`；
- 不让 π 或 Composer进入production backend；
- 不建立 J→Triton/TileLang/CUDA deterministic lowerer；
- 不恢复 RuleCard、RuleRegistry、`C.strategy` 或 kernel template；
- 不用 PTX 代替 TileLang TIR hard closure；
- 不用compile readiness冒充Gate1/Gate2；
- 不用masked-J单独判断semantic correctness；
- 不用测试mock、虚构结果或 archived unit tests 代替真实provider/evidence。

## 12. 最终一句话

> **π_min 告诉 Replay“实现必须满足什么物理约束”，J 告诉 Replay“程序计算什么”，fresh source-blind agent 在本次 episode 内用 syntax-only Composer决定“具体怎样写”；Composer只放大 agent 的编辑能力，不获得从 J/π 生成答案的语义权力。**
