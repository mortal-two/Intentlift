# Task 25：J主导π_min与Replay Agent内部Route Composer重构

> 历史状态：`ARCHIVED_AFTER_P0_SUPERSEDED_BY_TASK26`（2026-07-31；未开始运行时代码迁移）。
>
> Task 25时期设计：
> `doc/archive/2026-07-31-pre-shared-j-semantic-replay-compiler/IntentIR_J主导_最小约束pi与Replay内部Composer重构设计_2026-07-31.md`。
>
> 当前权威设计：
> `doc/IntentIR_共享J语义物化与pi约束Replay编译Agent最终设计_2026-07-31.md`。
>
> 前驱：Task 24保留9项`SCRC_CERTIFIED`、1项`NO_CLOSURE`及Composer成本轨迹，但其固定
> line/block Composer不再扩建。Task 25不接管production backend；backend继续只消费accepted `J+C`。

## 0. 本任务只解决什么

Task 25只解决Replay作者化的结构性成本：让fresh source-blind Codex不再从扁平J/π逐occurrence
手写整份target source，而是在同一episode内作者化一个可复用、可编译、可局部修正的Composer
scratch。固定宿主承担语法、显式route链接、稳定local、source map、编译与隔离，不替agent选择
operator、predicate、target API、control写法或算法。

本任务不以继续闭合单个kernel为起点。先让机制在现有exact accepted artifact上证明：

1. agent输出规模从physical occurrence/source长度转为semantic choice与少量override；
2. first-compilable时间、compile/fix cycle或input tokens出现真实下降；
3. Mask-J后固定宿主不能仅凭π shell产生计算；
4. 相同核心可用于Triton、TileLang和CUDA，而不建立三套J lowerer。

## 1. 固定数学与对象边界

```text
Lift_H(Source) -> (J,C,π)

K := AgentCompose_f(J,π,MaskC(C);U_f)
R_K := AgentDeclaredPlaceholders(K)
Γ_π ⊢ R_K : ExplicitUniqueRoutes(π)
Replay := SyntaxRouteLinkRender_f(K,R_K)

SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1_f(Source(C),Replay(C))
  and Gate2(Source(C),Replay(C),RefEval(J))
```

`K`、requirements view、π行式投影、binding、fragment、source map和checkpoint全是episode-local、
proof-irrelevant派生状态；它们不成为artifact、第四IR、certificate、Gate输入或backend输入。

唯一语义权威仍是J：

```text
Γ_(J,C) ⊢ π : RealizationWitness(J,C)
SemAtoms(π) = ∅
```

## 2. Task 24为什么不够

Task 24已证明Composer core可以跨TileLang/CUDA安装、render、compile并参与真实closure，也在一个
row_sum同artifact A/B上观察到21.02% elapsed与18.23% input-token下降。但更复杂轨迹仍显示：

- requirements主要以巨大packet/critical-facts/ledger呈现；
- agent仍逐physical instance维护locals、route、region与整份source；
- Composer node数量反映源码行数，而不是semantic binding数量；
- formal relift后fresh agent必须重新完成相同机械编译工作；
- `selected_path=composer`只证明使用了工具，不证明J/π消费复杂度降低。

因此Task 25不增加kernel模板，也不继续扩建固定semantic lowerer；它把“agent在本轮写生成器”变成
一等作者化路径。

## 3. 三层严格分工

### 3.1 固定packet views：exact重索引

允许固定代码从同一canonical packet产生按需、分页、可回溯的只读视图：

- declaration-order J requirements：`id/op/operands/attrs/result/type`；
- π regions/instances/routes/order/sharing/time/control的显式列；
- J semantic ref到π occurrence的反向索引；
- lexical region内incoming/outgoing/empty-slot route清单；
- exact source path与continuation query。

视图必须lossless round-trip或逐字段回指canonical packet；不得加入target API、推荐statement、
algorithm label、Source/Gate/历史Replay信息，也不得宣告某节点已经实现。

### 3.2 固定Composer与route linker：只做机械构造

允许：

- `line/block/raw_block`与通用function/assign/call/if/for/return语法节点；
- stable node/local ID、insert/replace/delete/move；
- `{{out:iK}}`、`{{slot:iK:N}}`、`{{dep:iK:N}}`精确占位符；
- 沿agent明确引用的唯一route代入local；
- sharing/order、scope、dominance、def-before-use与重复/遗漏slot拒绝；
- render/source-map/checkpoint/rollback/compile/own-IR/diagnostic mapping；
- raw escape与完整source直接作者化。

禁止：

- `J op -> target API`；
- `π relation -> source fragment/control skeleton`；
- 自动生成predicate、effect、route、loop body或算法；
- kernel/family matcher、历史winner与failure recipe；
- 读取Source、Gate diff、C concrete value或sibling Replay。

### 3.3 Agent-owned K：本轮可以很强

fresh Replay agent可以在本轮：

- 为一个J semantic node或一组exact matcher作者化一次target binding；
- 把同一binding显式应用到多个π occurrence；
- 按lexical region组织包含多个occurrence的fragment；
- 为少数instance写override；
- 建立任意target-local helper或metaprogram；
- 根据自己的compile/own-IR诊断局部修复binding/fragment；
- 必要时退回raw target source。

这些matcher/binding/helper不跨formal round保存。固定宿主只执行agent给出的精确匹配与实例列表，
不能依据`semantic_ref`自动选择target statement。

## 4. 目标代码目录

Composer不属于Gate语义，也不属于provider launcher。迁移后的目标边界是：

```text
python/intentir/replay_authoring/
  __init__.py
  composer/
    model.py          # syntax node、stable id、closed plan grammar
    render.py         # Python-like/C-like纯序列化与source map
    edit.py           # insert/replace/delete/move、revision/checkpoint
    route_link.py     # 显式placeholder与exact route机械链接
    diagnostics.py    # rendered line/node/placeholder诊断映射
  views/
    requirements.py   # exact J requirements重索引
    realization.py    # canonical π的lossless行式投影与反向索引
    packet_query.py   # bounded paging与continuation
  episode.py          # fresh安装、状态生命周期、telemetry与销毁
```

保留原所有者：

- `verify/gate1/replay.py`：formal packet、ReplayCheck与Gate调用；
- `verify/gate1/codex_replay.py`：Codex writer和formal round orchestration；
- `extraction/providers/codex_cli.py`：通用隔离provider/tool bridge；
- frontend compiler/observer/Gate2 runner：不因Composer搬迁而复制。

迁移必须一次只移动一个责任面，删除旧实现后再进入下一面；禁止长期保留新旧双栈或兼容旁路。

## 5. 实施阶段

### P0：权威与工作区整理（本检查点）

- 固化Task 24横向成功/失败证据与PTX v6待fresh验证事实；
- 将Task 24设计移入历史目录；
- 建立Task 25和唯一active设计入口；
- 记录目标代码目录、迁移顺序和真实验收；
- provider调用为0，工作区clean。

### P1：行为不变的目录拆分

- 将现有`replay_composer.py`按model/render/edit/episode职责迁出Gate目录；
- 保持Composer v2接受集合、render bytes与Codex workspace surface不变；
- 删除旧module，不保留双实现；
- 用一个既有accepted exact artifact运行真实fresh Replay recheck，确认目录搬迁没有破坏closure。

### P2：exact requirements与π realization views

- 从`replay.py`巨型packet说明中抽离可分页的requirements/realization query；
- 每个view row回指canonical JSON path，超长值按exact byte/item分页；
- π投影decode后与canonical relation逐字段相等；不存在唯一值时fail closed，不猜默认route；
- agent启动prompt只给路径、大小、短边界和首个查询入口。

### P3：grouped fragment与route linker

- 实现`out/slot/dep`placeholder、stable locals、显式route代入；
- 支持fan-out、multi-result、empty-slot dependency、sharing/order和普通acyclic cross-region route；
- loop backedge、terminal fork、carrier tuple与conditional post-state逐类由真实kernel推动，未支持时明确
  typed hole/fail closed，不隐式降级；
- diagnostic回指fragment/node/placeholder/instance，而不是让agent重读整份source。

### P4：agent-local semantic binding

- 提供无预装规则的binding/matcher执行面；
- agent可按exact J字段写本轮matcher并显式列出适用semantic refs/instances；
- 一个binding可复用到多个occurrence，少数override保持显式；
- formal round结束销毁K，normal/masked建立完全不同的K/workspace/conversation。

### P5：真实A/B与横向闭合

先在相同accepted artifact上运行fresh `off`与新Composer `preferred`：

1. TileLang row_sum：reduction/materialization；
2. CUDA row_sum：participant projection/identity stage；
3. TileLang/CUDA gather或rowmask：predicate/index/control；
4. TileLang transpose与CUDA transpose：跨前端表示差异；
5. Task 15一个高occurrence、loop/carrier/terminal CFG难例。

只有机制A/B显示真实下降后，才用新流程横向重放原先22项；不因“已经写完Composer”扩大provider
批次。

## 6. 真实验收与成本指标

不添加、恢复或生成`python/tests`，不用mock/smoke/protocol test替代closure。每个声称closed的case必须
保存：exact artifact、current TypeCheck、fresh Replay trajectory、frontend compiler observation、Gate1、
Mask-J以及Source--Replay/Source--J/Replay--J三边Gate2。

主要指标：

```text
time-to-first-compilable
compile/fix cycles
Replay elapsed / provider segments / input tokens
closure@1 / closure@2 / closure@3
agent-authored binding bytes与最终source bytes
binding count / J semantic node count / π occurrence count
route/local/region类construction failure数
normal-vs-masked closure gap
raw escape rate
shared core LoC / target profile LoC / kernel-specific persistent LoC
```

成功不以某一个数字门槛伪装统计显著性。至少要求：

- 新Composer在多个真实case中不增加formal Gate1 rounds；
- binding数量不再随同semantic_ref的π occurrence近似线性增长；
- 至少一项主要成本指标在多数选择case上下降；
- persistent kernel-specific semantic LoC保持0；
- Mask-J不能借固定matcher/renderer恢复可执行计算。

若只增加工具调用和中间文件、没有降低构造成本，就继续收缩或删除该能力。

## 7. 执行纪律

- 默认使用Codex `high`；只有机制、packet、工具、consumer和observer已确认正确，而exact artifact下仍
  重复暴露模型推理不足时才升`xhigh`；
- 每次系统修改后都用fresh episode，不patch旧Replay、不重判历史artifact；
- 共同失败只沉淀syntax/route/view能力，不增加kernel模板；
- Replay看不到Gate1 diff；diff只返回同一source-aware Lift owner；
- PTX v6与CUDA transpose是独立observer检查点，不得反向塑造Composer接口；
- 不以hash、digest、commitment或冻结门代替真实工作进度。

## 8. P1开始前的干净入口

P1第一步只做现有Composer职责切分，并以真实accepted artifact证明行为不变。此后才实现exact views和
route linker。开始实现时不得同时修改π schema、Gate1 observer、RefEval或production backend；否则
无法区分性能变化来自作者化机制还是接受集合变化。

最终原则：

> **J决定语义，π只约束realization；fresh Replay agent在本轮编写足够强的composer/binding，固定宿主只替它稳定编辑和连接显式route，不替它写答案。**
