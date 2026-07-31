# IntentIR：J 主导的 π_min 与 Episode-Local Composer Replay 最终设计

> 历史状态：Task 24时期的Replay设计检查点（2026-07-31）；已由
> [Task 25权威设计](../../IntentIR_J主导_最小约束pi与Replay内部Composer重构设计_2026-07-31.md)
> 取代，不再属于active spec。本文只保留Task 24实现与实验的可追溯语境，不能覆盖后继设计。
>
> 本文服从并细化四份 SCRC 权威设计，不改变其数学对象、输入因子化或认证谓词：
> [C 所有权](../../IntentIR_C_所有权_绑定_消费与三方验证设计_2026-07-22.md)、
> [π 依赖型实现见证](../../IntentIR_pi_依赖类型实现见证与Replay安全设计_2026-07-22.md)、
> [Lift/Replay/Gate 执行流程](../../IntentIR_Lift_Replay_Gate1_Gate2_实验与系统改进最终执行流程_2026-07-22.md) 与
> [SCRC 数学总述](../../IntentIR_SCRC_最终数学与执行流程_2026-07-22.md)。
>
> Production backend 仍以
> [J-Grounded、Q/D-Certified Lowering-State Search](../../IntentIR_Contract_Preserving_Multi_Lowering_后端最终设计_2026-07-24.md)
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

这里的 `π_min` 不是第四种 artifact，也不是新 wire 名称。`artifact.v12` 是首版兼容基线；它已经满足
closed、J/C-dependent、无 op/dataflow/type/shape/Source/API/target recipe 副本的所有权条件。真实
CUDA `row_sum` 纵切随后给出了升版所需的反例：exact J 拥有 16-lane logical domain，C 拥有外部
participant extent，但 v12 无法表达多个物理 participant 循环覆盖该 logical domain。于是当前
`artifact.v13` 只增加一类窄的 `participant_mappings` realization delta；它不增加计算语义或 target
recipe，旧 v12 artifact 继续按原字段集合读取，不被静默升级或重判。

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
- J reducer synchronize 前一次显式identity materialization的closed route topology；
- program-coordinate domain 到物理 axis 的 identity；
- 物理 participant axis 对 exact J domain 的 closed cyclic projection；
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

不删除 v12 relation。先对真实 kernel 记录每类 relation 的消费与 Gate1 必要性；只有跨样本证据
表明某字段可由 J/C 全局唯一推导或应由 observer 统一 NormalizeAway，才做版本化 schema 收缩。

### 2.5 `artifact.v13` 的 participant projection

唯一 wire form 是：

```json
{
  "domain_instance": "i2",
  "axis": 0,
  "extent_slot": "c0",
  "projection": "cyclic"
}
```

其 dependent denotation 只在 `Γ_(J,C)` 中成立：`domain_instance` 必须是 program region 中锚定某个
J boundary domain、沿 route 到达 `address_lane` 而不到达 `address_base` 的既有 `realize` instance；
该domain必须唯一规范为单逻辑轴`[0,E)`，其中`E`是positive literal或J中declared `x*`；多轴、非零
下界、额外predicate或无法唯一得到extent的domain fail closed；
`axis` 是公开物理 participant axis；`extent_slot` 必须是 C 中同轴、required、positive-integer 的
外部 launch extent；`cyclic` 表示该物理 coordinate 循环投影到 **J 自己拥有的 exact domain extent**。

该行刻意不携带 logical extent、replica count、bitmask、modulo/address formula、C value、Source
spelling、frontend API 或 target recipe。删除或 Mask J 后，`i2` 不再给出周期或计算含义；π 只泄漏
“存在循环物理覆盖”这一 realization topology，不能独立恢复 reduce/add/dot 等计算语义。Replay agent
必须读取 exact J 才能得到 logical period，再自行选择 CUDA、TileLang 或 Triton 的公开 target spelling。

这次升版来自真实反例，而不是预先扩张 grammar：v12 CUDA `row_sum` 的三轮 Replay 都直接使用物理
participant coordinate，导致错误 row/OOB/race；Composer 每轮均实际使用，却无从恢复缺失约束。
因此根因是 Lift 表达能力与 observer coverage 的共同缺口，不是给 Composer 增加 reduction 模板的理由。

### 2.6 J-dependent identity materialization

`identity_materialization`仍是已有closed refinement，不增加operator、constant或target recipe。只有当
被修饰的`realize`恰有一条slot-0 incoming route，且唯一slot-0 outgoing route进入一个通过exact J解析为
`reduce`的`synchronize` instance时，它才具有下面的窄denotation：

```text
incoming physical carrier
  -> one explicit combine with the exact J reducer identity
  -> downstream reduce synchronize
```

reducer、typed identity、carrier dtype与逻辑shape全部只从J读取；π只声明该compiler-visible stage必须
存在，不保存`sum/max`、`0/-inf`、Source拼写、CUDA shuffle、TileLang copy或任何frontend API。删除或
Mask J后，route topology不能独立决定combine为何种计算。若Source没有可见的identity初始化与combine，
Lift不得仅因J含reduction就作者化该refinement。

该拓扑不会覆盖TileLang已有的shaped representation materialization：未以唯一route进入J-reduce
`synchronize`的`identity_materialization`仍只表示其原有routed representation stage。Replay依据当前
frontend公开`U_f`自行写出实现，最终由TileLang结构TIR或CUDA PTX hard observer判断是否重建Source。

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
surface；它不把 π loop 或 participant projection 自动映射成
`T.serial/T.parallel/T.vectorized/thread_binding`，也不选择 fragment/shared layout。若 π 含
`participant_mappings`，agent从 exact J 读取 logical extent并自行选择公开 TileLang/TIR spelling；
最终仍由结构 TIR hard observer 裁决。

### 6.3 CUDA

```text
source form: standalone CUDA C++
hard observer: declared normalized PTX semantic topology profile
Gate2: native CUDA execution + three edges
```

profile 只处理 C-like braces、function/helper surface与现有安全 allowlist；不选择 thread/block mapping、
shuffle、shared memory、barrier、atomic、MMA 或 reduction recipe。CUDA PTX observer按现有 contract 放松
cache、register、label与纯 address spelling，但保留 observable value、memory/effect、control、coordinate、
collective、ABI 与 target hard evidence。`cuda-ptx-semantic-effects-v5` 沿每个 memory address 的
reaching definitions保留 information-losing coordinate fold，并把byte-domain中的
`(coordinate << k) & ((period-1) << k)`与logical-domain的`coordinate & (period-1)`归一为同一closed
cyclic projection。该归一只跨纯power-of-two scale、整除mask或positive remainder，以及可证明精确的
同signedness非窄化整数conversion；predicated definition、control boundary、address-space conversion、
窄化或其他opaque mixing均保守停止传播。普通affine address spelling继续放松，但不删除value/effect/
control差异。该观察发生在 Gate1，结果不会返回 Replay或写入π。

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

同一source-aware Lift owner的relift feedback使用标准、每轮原子覆盖的closed `feedback.json`作为
canonical query source；它在挂载前已经删除重复previous artifact，并把compiler大文本移到单独只读
counterexample文件。stable category index只作冗余，二者文件身份都在首次episode固定，因此新增
CUDA/TileLang observer label不会改变persistent workspace文件集合，category消失时旧内容也不会残留。
owner先读bounded frontier，再按exact label查询单一category；超长TIR/PTX string只按原JSON path与
UTF-8 byte offset分页，array/object按item offset递归有界分页，并用whole/chunk receipt核验。工具不
解释差异或建议repair，也不注入固定dataflow repair hint。该transport只搬运既有Gate1 comparison，
不选择π refinement，也不把feedback送给fresh source-blind Replay。

这里的“可查询”还要求mount identity闭合：copied query tool必须从自身位置区分MiniSWE的`/input`与
Codex的`/workspace/input`，并在生成next/full query时回显本次exact roots。否则不存在的绝对path会被
错误解释成“无category”，使agent在没有exact difference的情况下猜π。Codex Lift v33把frontier与
continuation命令固定为显式workspace-relative roots；这只是evidence transport，不是语义提示。

## 8. 静态反泄漏与运行审计

### 8.1 π 静态审计

现有 closed schema 与 TypeCheck 继续作为权威。额外 audit 只能做确定性报告，不能成为第二套宽松
validator。至少记录：

- π 中所有 key/value 是否属于当前 closed grammar；
- 所有 semantic refs 是否为 opaque J/C handle；
- 是否出现 Source/API/kernel/family/free-text 字段；
- 是否出现 J op/attrs/type/shape/dataflow 的显式副本；
- route-extent closed constructor 是否满足既有 dependent contract。
- participant mapping 是否只含闭合四字段、锚定 exact J lane domain 与 matching C launch extent。

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
- 明确 v12 是兼容基线，v13 只接受由真实 cyclic-participant 反例支持的窄扩展；
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
