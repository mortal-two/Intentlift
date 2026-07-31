# IntentIR：共享J语义物化与π约束Replay编译Agent最终设计

> 日期：2026-07-31
>
> 状态：Task 26的Replay子系统权威实现依据。
>
> 本文服从C、π、Lift/Replay/Gate执行流程与SCRC数学总述，不增加正式输入、证明对象或认证Gate；
> production backend仍只消费accepted `J+C`，不读取π或Replay内部状态。
>
> 本文取代Task 24的固定syntax Composer与Task 25的route/binding Composer设计。两代真实artifact与
> 成本轨迹保持原verdict，只退出active architecture。

## 0. 最终裁决

固定代码可以而且应该完整、确定性地消费J，但它只产生前端无关、只读、参数化的semantic program：

```text
S_J := SemanticMaterialize(J)
```

fresh Replay agent不是填洞器、binding作者或fragment拼装器，而是一个完整的source-blind编译Agent：

```text
Replay_f := ReplayCompilerAgent_f(
    exact J,
    read-only S_J,
    exact π,
    MaskC(C),
    public U_f
)
```

Agent全局理解`S_J+π`，自由编写、编译、查看和重构一份完整Triton、CUDA、TileLang或其他前端源码。
宿主不拼接最终源码，不建立per-node holes/bindings，不沿route生成代码，也不提供target recipe。

外部数学接口仍然只有：

```text
Replay_f = W_f(J,π,MaskC(C);U_f)

SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1_f(Source(C),Replay_f(C))
  and Gate2(Source(C),Replay_f(C),RefEval(J))
```

`S_J`是J的确定性内部视图，不是第四输入、artifact、证书字段、Gate输入或production输入。

## 1. 三种权力必须分开

### 1.1 J：唯一计算语义

J拥有且只由J决定：

- signature、logical tensor/value与类型；
- operator、operand、dataflow与result；
- logical domain、index、predicate与fill；
- reduction、recurrence、semantic constant与result relation；
- output、alias、update与observable effect；
- structured control和conditional post-state的逻辑含义。

### 1.2 π：Source realization见证

π只保留无法由J/C唯一恢复、而Gate1重建Source realization需要的事实：

- occurrence/multiplicity；
- physical region、iteration、carrier与terminal topology；
- program axis、participant projection与program linearization；
- route、order、sharing与materialization；
- representation-only staging和匿名C-slot attachment。

π不能保存operator、算式、predicate文本、semantic constant、target API、Source片段、kernel identity或
algorithm recipe。删除J后，π最多暴露realization topology，不能独立恢复计算。

### 1.3 Replay Compiler Agent：π约束下的完整target compiler

Agent读取`S_J+π`，拥有target instruction selection和整个源码工程组织，但不拥有新的计算语义。它的
输出必须属于：

```text
Replay_f ∈ Realize_f(S_J,π;U_f)
```

而不是production backend所允许的任意：

```text
AnyImplementation_f(J)
```

## 2. SemanticMaterializer Core

### 2.1 它做什么

`SemanticMaterialize(J)`把accepted、canonical J物化为一份正常编译器可以读取的typed semantic
program。它应显式呈现：

- declaration-order function/signature；
- typed values、semantic tensors和symbolic dimensions；
- op、operand/result SSA与完整dataflow；
- domain/access/index/predicate/fill；
- reduction/recurrence/control/effect；
- output binding、alias/update和result relation；
- 每个semantic entity回指exact J handle的稳定reference。

首选表示是参数化canonical Intent MLIR或与其同构的只读typed program；它不是普通target source，
也不包含π realization、C concrete value或target API。

### 2.2 它不能做什么

SemanticMaterializer不得读取：

- π；
- 真实C valuation或Harness runtime value；
- Source、TTIR/TIR/PTX或Gate diff；
- frontend/kernel identity；
- historical Replay或target compiler feedback。

它不得选择：

- target API和语法；
- thread/block/program mapping；
- pointer/address spelling；
- memory/layout/synchronization；
- production search state或C attachment；
- 某个frontend的algorithm recipe。

### 2.3 参数化而不是production M0

production materialization可能包含concrete extent、constexpr specialization、C attachment、loop form
选择或search state。Replay的`S_J`必须保留J中的symbolic handles与全部semantic anchors，不能直接复制
一个已针对production实例化的M0。

工程上允许把后端分支已经验证过的canonical J elaboration、type/shape、boundary和Intent MLIR逻辑
选择性迁入一个自包含Replay SemanticMaterializer Core；当前不为跨worktree API共享增加工程负担。
但语义规则必须同源、可追溯，不能重新用自然语言或另一套手写规则解释J。

当前可复用基线是：

- `intentir.ir.canonical_j.elaborate_j`的canonical J elaboration；
- `intentir.frontend.canonical_j.materialize_j_function/emit_j_module`的target-neutral materialization；
- `intentir.frontend.emit_mlir`的Intent MLIR emission；
- production `backend.materialize`中J-only且不依赖C/search的closed projection。

未支持J形式必须fail closed并报告`semantic_materializer_unsupported`；不能让π或Agent补造缺失语义。

### 2.4 `S_J`不可修改的准确含义

- `S_J`由宿主从当前exact J fresh生成；
- 以只读文件和只读query挂载；
- Agent不能修改该文件或把修改版提交为输入；
- 最终target source不要求逐行保留`S_J`；
- Agent可以融合、分块、重排、改变目标算法结构和整体重构源码；
- Agent输出的是`S_J`的一个编译结果，而不是修改后的`S_J`。

这与普通编译器读取只读AST/IR相同，不需要给semantic nodes建立holes。

## 3. Replay Compiler Agent的完整职责

Agent必须全局完成：

- target function/kernel declaration与ABI；
- J semantic program到target expression/API的instruction selection；
- π约束下的program/thread/block映射；
- logical index到pointer/address expression；
- load/store/mask/fill的target表达；
- reduction、recurrence、loop/carrier与control的target实现；
- layout、memory space、temporary、sharing与synchronization；
- materialization、route/order和region nesting；
- frontend dtype、syntax、compiler限制与全局源码重构；
- 编译、查看own IR、局部或整体修复，最终提交完整source。

Agent不是逐J-node映射器。它可以用一个融合实现覆盖多个semantic op，可以选择不同但等价的physical
algorithm，也可以在本episode内自己写任意helper、metaprogram或composer。前提是：

- 不能从π补造J中不存在的公式；
- 不能忽略π已声明的occurrence/control/route/order/sharing/carrier/materialization；
- 不能增加改变observable结果/effect的额外语义；
- 必须接受最终Gate1/Gate2对完整源码的认证。

普通softmax降低为online recurrence不是默认许可：update/rescale/sum等公式必须从J/S_J可推导，π只
约束recurrence/carrier/route等realization。若J只表达extensional softmax，这是一项真实algorithm
lowering，是否同时满足Source realization由Gate1/Gate2裁决，不能由π形状暗示答案。

## 4. 宿主只提供编译工作环境

允许的固定能力：

- exact J、read-only `S_J`、exact π、MaskC与U_f文件；
- J/S_J的按声明顺序查询与exact回指；
- π按region/instance/route/control的无损分页、邻接和CFG查询；
- 普通文件编辑、exact patch、checkpoint/rollback；
- frontend compile、ReplayCheck、own compiler IR与诊断；
- 一个完整target source的原子submit；
- fresh workspace、normal/masked隔离与清理。

禁止的固定能力：

- J op到target API映射；
- π relation到target skeleton/statement；
- per-node hole/binding表；
- fixed route renderer或fragment assembly；
- kernel/family模板和历史winner；
- 根据Gate diff、Source或C concrete value修复Replay；
- 强制Agent沿某个固定AST或目标算法逐格填空。

现有syntax Composer可以作为Task 24历史工具保留到迁移完成，但退出新主路径。新路径默认由Codex像
普通编译器工程师一样维护完整source；若Agent自行建立composer，它只存在于当前episode并随workspace
销毁。

## 5. “生成后认证”而不是verified compiler

只读`S_J`能固定输入语义，却不能静态证明Agent写出的任意target program忠实实现了它。系统不新增
verified lowering proof，而继续使用：

1. `TypeCheck(J,C,π)`检查artifact与realization witness；
2. ReplayCheck/compile拒绝接口、构造与已声明义务的明显错误；
3. Gate1比较Source与最终Replay的compiler observation和realization；
4. Gate2分别检查Source--Replay、Source--J、Replay--J三条执行关系。

因此“不可修改S_J”是输入所有权与认知负担边界，不是假装已经证明target equivalence。

Gate1失败仍只返回同一个source-aware Lift owner做relift。fresh Replay不获得Gate diff，也不patch旧
Replay；新formal round重新生成`S_J`并启动新编译Agent。

## 6. Mask-J必须重新物化

正常臂：

```text
S_J = SemanticMaterialize(J)
```

masked臂必须使用：

```text
S_J_mask = SemanticMaterialize(MaskJ(J))
```

若masked J不能形成完整semantic program，就只提供相应masked view，不能先从normal J生成完整`S_J`
再仅遮盖原JSON。否则完整semantic skeleton本身已经泄漏J。

normal/masked必须使用不同provider conversation、workspace、compiler history、source、query state和
agent-local helper。两臂可以共享固定materializer代码，但不能共享任何一次物化结果或缓存内容。

Mask-J继续只衡量当前模型和packet下J的操作性必要性，不替代主证书。

## 7. 多前端边界

所有前端共享同一个：

```text
exact J -> parameterized S_J
```

不同前端的开放世界复杂度由Replay Compiler Agent承担：

```text
S_J + π + U_Triton   -> complete Triton source
S_J + π + U_CUDA     -> complete CUDA source
S_J + π + U_TileLang -> complete TileLang source
```

新增前端仍需真实compiler/ingress、ABI adapter、observer profile和Gate2 runner；不能把这些基础设施隐去
并声称零接入成本。但无需预先实现覆盖所有J op×π组合的deterministic target backend。

不同Source frontend不要求J字节一致：若ABI、mask/fill、runtime dimension或observable semantics不同，
exact J/S_J可以不同；共享的是同一semantic materialization规则，不是强制同一artifact。

TileLang仍以结构TIR为hard Gate1、PTX为diagnostic；CUDA仍以声明的PTX semantic profile为hard
observer；Triton仍以normalized TTIR为hard observer。`S_J`不替代任何frontend compiler observation。

## 8. Replay与production backend的区别

production backend：

```text
accepted J+C -> 任意满足J/Q/D的高性能实现
```

它不读取π，目标是搜索正确且高性能的实现。

Replay：

```text
J+S_J+π+MaskC+U_f -> 重建满足π所见证Source realization的目标前端程序
```

它不是任意optimizer。target语法和工程实现可以自由变化，但不能忽略π的realization义务。二者可共享
J elaboration/Intent semantic materialization语义，不共享C search、target lowering或π消费。

## 9. Fresh episode生命周期

每个TypeCheck-valid formal round：

1. 从当前exact J fresh生成参数化`S_J`；
2. 创建只挂载本轮J/S_J/π/MaskC/U_f的空workspace；
3. 启动一个fresh source-blind Replay Compiler Agent；
4. Agent按需查询、编写完整source、多次compile/inspect/rewrite；
5. Agent原子提交一个完整Replay source；
6. runner在Agent外重新编译并执行正式Gate1；
7. accepted normal candidate才运行独立Mask-J与Gate2；
8. 保存proof-irrelevant trajectory后销毁workspace。

资源限制使用wall-time、上下文窗口和formal round上限；不把内部编译/编辑次数误记为Gate1 round。

## 10. 实现目录边界

目标新增：

```text
python/intentir/replay_compiler/
  __init__.py
  semantic_program.py      # 只读参数化S_J数据模型与J anchors
  semantic_materializer.py # canonical J -> S_J；不读π/C/Source/Gate
  semantic_query.py        # bounded exact query与回指
  packet.py                # J/S_J/π/MaskC/U_f只读mount描述
```

保留原所有者：

- `intentir.ir.canonical_j`：accepted J的唯一canonical elaboration；
- `intentir.frontend.canonical_j/emit_mlir`：共享semantic materialization基础；
- `verify/gate1/replay.py`：Replay ingress、compile probe与formal packet；
- `verify/gate1/codex_replay.py`：fresh Codex编译Agent生命周期；
- frontend compiler/observer与Gate2 runner：各自真实执行边界；
- production backend：C attachment、search和target lowering。

迁移完成后删除固定host `replay_composer.py`主路径及其强制plan/scaffold；不保留新旧双栈。普通
patch/checkpoint/compile工具移交Codex Replay workspace，不依赖semantic Composer。

## 11. 真实验收

不恢复`python/tests`，不用mock、smoke或协议自测代替真实closure。每个声称成功的case必须保留：

- exact artifact与current TypeCheck；
- 本轮fresh `S_J` receipt和只读mount证据；
- fresh Replay Compiler trajectory与完整target source；
- frontend compiler observation和Gate1；
- fresh masked materialization/episode与J-necessity结果；
- Source--Replay、Source--J、Replay--J三边Gate2。

这里的`S_J receipt`只记录本轮materializer版本、J-only依赖、参数化状态与只读mount事实，是
proof-irrelevant运行证据；它不携带独立语义，不进入artifact schema、TypeCheck、Gate或certificate。

核心成本指标：

```text
raw J bytes / S_J bytes / agent实际读取bytes
time-to-first-compilable
compile/fix cycles
Replay elapsed / provider segments / input tokens
closure@1 / closure@2 / closure@3
semantic reconstruction failure数
π realization failure数
raw complete-source rewrite次数
normal-vs-masked closure gap
新增frontend基础设施LoC / persistent kernel-specific semantic LoC
```

首先用Task 24已有accepted artifact做同条件fresh A/B：旧J+π direct Replay对比新J+S_J+π Replay。新路径
不能增加formal Gate1 rounds，且应在多个case降低semantic reconstruction错误或至少一项主要成本。

随后覆盖row_sum、gather/rowmask、transpose和一个Task 15高occurrence/loop/carrier/terminal CFG难例。
只有机制确实降低成本后才横向重放原22项。

## 12. 最终非目标

- 不建立per-frontend deterministic J compiler；
- 不让Agent填per-node hole或binding表；
- 不建立fixed route/source assembler；
- 不让SemanticMaterializer读取π或生成target skeleton；
- 不把`S_J`持久化为artifact或送入production backend；
- 不让π复制J语义或target recipe；
- 不把Gate diff交给Replay；
- 不以hash、digest、commitment或文件冻结代替真实认证；
- 不用新设计静默重判Task 24/25历史结果。

最终原则：

> **固定Core把J变成只读、参数化、前端无关的semantic program；fresh Replay agent作为π约束下的完整编译器，自由生成整份目标源码；宿主不替它填空或拼程序，Gate1/Gate2认证它是否真的编译正确。**
