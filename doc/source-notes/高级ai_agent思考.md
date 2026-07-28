# 高级 AI 对 Agent 工作流的思考（来源材料）

你的感觉基本是对的：**现在卡住的主要不是“模型完全理解不了 Kernel”，而是工作流让模型把大量能力浪费在协议、JSON 机械闭合和多人交接上。**

按你贴出的三核日志，实际上只有 Linear 正在真正检验“Lift 语义质量”；另外两个失败主要是编排失败。

## 一、这三条结果分别说明了什么

| Kernel        | 实际发生了什么                                                                                                      | 暴露的问题                |
| ------------- | ------------------------------------------------------------------------------------------------------------ | -------------------- |
| Flash forward | planner 已形成了较好的 online-softmax/loop 计划，但因 submit marker 没单独放在第一行而超预算                                         | **提交协议失败，不是语义失败**    |
| GroupNorm     | planner 成功；builder 用 32 轮生成了一个 `CandidateCheck PASS` 的完整 artifact；reviewer 又用 16 轮改坏 handle/node 顺序，最后整份候选丢失 | **单一所有者和事务机制缺失**     |
| Linear        | 923 秒后形成 TypeCheck 通过的 artifact，Gate1 尚在运行                                                                   | **终于能测语义质量，但速度没有改善** |

GroupNorm 尤其说明问题：总共大约 `9 + 32 + 16 = 57` 轮，**第 41 轮时其实已经有了一个机械闭合的候选，后面 16 轮不是没有贡献，而是负贡献**。

这不是“再多给 reviewer 一点预算”能解决的。

---

# 二、为什么当前流程没有发挥 LLM 的能力

LLM 真正擅长的是：

* 从长 Source 中抽象逻辑计算；
* 判断 J/C/π 的信息归属；
* 理解 online recurrence、consumer、effect、sharing；
* 根据 Gate1 结构差异重新推断遗漏事实。

现在却让它大量处理：

* 猜工具路径；
* 记 submit marker 的特殊位置；
* 手工维护 `n0/n1/i0/i1` 等引用；
* 手工保持声明顺序；
* 在几十轮后重写一份巨大 JSON；
* 不同 agent 反复重新读取和理解同一 Source；
* reviewer 直接修改一个已经 dependent-typed 的全局 artifact。

所以现在的系统实际上是：

> **让 LLM 做编译器前端最机械、最不擅长的工作；却把它最擅长的端到端语义判断拆散到多个独立上下文里。**

`J/C/π` 又不是三个互不相关的文档。它们是强耦合对象：

* J node 改了，π 的 ref 可能全变；
* signature 改了，C slot 和 route 可能受影响；
* node 顺序改了，opaque handle 可能重新编号；
* consumer/effect 改了，多个 B 和 π relation 都要同步。

因此，“planner 写一版、builder 写一版、reviewer 再直接修改成品”天然不稳定。

---

# 三、planner → critic → builder 只修了一半

你们 agent 现在准备改成：

```text
planner → source-aware critic → builder
```

这比：

```text
planner → builder → reviewer 直接改成品
```

好，因为最后只有 builder 写 artifact。

但它仍然没有解决两个核心问题：

1. 三个独立 agent 仍然要重复理解 Source，存在语义交接损失；
2. builder 仍然承担大量 handle、顺序、引用和 JSON 维护工作。

而且 critic 在构造前只能审查“计划”，不能看到计划最终是否被正确落进 J/C/π。

更好的原则是：

> **一个 Lift owner 对最终 artifact 负全责；其他角色只能提供只读意见，不能修改候选。**

---

# 四、正确的工作流：单一作者 + 事务式机械工具

我建议把 staged workflow 改成下面这样：

```text
Source + compiler view
          │
          ▼
一个 source-aware Lift owner
          │
          ├─ 1. 形成简洁 semantic ledger
          │     signature / A / B / C / π obligations / risks
          │
          ├─ 2. 构造并编辑 working artifact
          │
          ├─ 3. 调用 CandidateCheck
          │
          ├─ 4. 可选只读 critic 返回 issue list
          │
          ├─ 5. owner 逐项修改并重新 CandidateCheck
          │
          ▼
last-known-good 完整 (J,C,π)
          │
          ▼
正式 TypeCheck → source-blind Replay → Gate1
```

这里的 planner、builder、review 可以保留为**同一个 agent session 内的阶段标签**，但不要再是三个互相交接的独立作者。

一句话：

> **阶段边界可以是日志边界，但不应该是语义上下文边界。**

---

# 五、最关键的是建立“last-known-good”事务机制

GroupNorm 不应该失败。

既然 builder 已经得到：

```text
CandidateCheck = PASS
```

这份候选就应该形成不可变 checkpoint：

```text
last_good_candidate.json
```

之后任何修改都必须走：

```text
复制 last_good → working draft
修改 working draft
运行 CandidateCheck
    PASS → 替换 last_good
    FAIL → 丢弃 working draft，回滚 last_good
```

如果 critic、builder 或模型预算耗尽：

```text
有 last_good → 提交 last_good
没有 last_good → 才报告 no candidate
```

现在系统的行为相当于：

```text
已经得到合法候选
→ reviewer 原地修改
→ 修改失败
→ 把原合法候选也一起丢掉
```

这是明显的编排错误。

尤其要冻结一条硬规则：

> **critic/reviewer 永远只输出结构化问题列表，不能输出或覆盖最终 artifact。**

---

# 六、机械工作必须交给固定工具，但不能替 LLM 生成语义

这不违反 LLM-first。

LLM-first 的含义应该是：

> **LLM 决定语义和信息归属；固定工具负责机械一致性。**

固定工具完全可以安全负责：

* 分配 opaque handle；
* 保持 topological declaration order；
* 更新重命名后的引用；
* 检查 route slot；
* 规范化 JSON；
* snapshot / rollback；
* schema validation；
* 调用 CandidateCheck；
* 原子提交。

它不应该负责：

* 选择 operator；
* 判断 mask 语义；
* 推断 reduction；
* 决定事实属于 J/C/π；
* 自动补 effect 或 result relation。

最小实现甚至不需要建设一个很大的新 editor。先做四个操作就够：

```text
save_working_candidate
candidate_check
checkpoint_if_pass
rollback_to_last_good
```

后续再考虑 typed node editor。

---

# 七、Flash 的 submit marker 应该彻底删除

Flash 的失败完全没有科研意义。

模型已经写出了计划，却因为：

> marker 必须单独位于第一行

而整个 planner 被判 `LimitsExceeded`。

这类协议非常不 agent-friendly。应该改成真正的工具接口：

```text
submit_lift_plan("/output/lift_plan.json")
```

工具负责：

1. 检查文件存在；
2. 解析 JSON；
3. 校验 protocol；
4. 原子结束阶段。

不要再依赖某一行 shell 文本的位置。

如果暂时无法增加工具，至少做到：

* 输出路径在 prompt 第一屏明确给出；
* 不允许 planner 猜工具目录；
* 提供唯一的提交命令；
* 提交命令不再与其他 shell 输出混合；
* 合法 plan 已落盘时，协议失败单列 `submission_protocol_failure`，不能算 semantic Lift rejection。

目前把 Flash 写成：

```text
IntentIR staged Lift planner failed
```

会污染 Lift 能力统计。它应归类为：

```text
ORCHESTRATION_FAILED
```

---

# 八、planner 不应该用 6–9 轮发现工具

planner 应该只做语义计划，不应该浏览运行环境。

它开始时就应该拿到：

```text
Source 文件
source-outline
可用的精确读取工具
精确输出路径
精确 submit 工具
计划 schema
```

理想情况下，planner 的工作是：

1. 阅读 outline；
2. 定向读取关键 Source 段；
3. 写一个紧凑计划；
4. 提交。

而不是：

```text
猜路径
列目录
试命令
找输出位置
理解 marker 协议
```

另外，plan 不要膨胀成第二份 IR。它只需要保存真正需要 LLM 判断的东西：

```text
logical signature
A dataflow / recurrence / state
B domain / effect / consumer boundary
C candidate slots
π realization obligations
unresolved risks
```

它是 proof-irrelevant scratchpad，不是新的 artifact。

---

# 九、下一轮不要急着增加候选或 Gate1 重试

现在：

```text
candidates = 1
gate1-rounds = 1
```

反而是对的。

因为你们当前要测的是：

> 一个干净的 Lift invocation 到底能不能一次形成高质量 artifact。

增加 batch、branch 或 relift 只会把工作流缺陷隐藏起来。

下一轮仍然用同样的三个 Kernel：

* Linear：time/carrier；
* Flash：online recurrence；
* GroupNorm：reduction/consumer/effect。

但比较两个流程：

### A：当前 staged multi-agent

```text
planner → builder → reviewer
```

### B：single-owner transactional Lift

```text
plan → build → check → optional read-only critique → repair
```

保持：

* 同一模型；
* 同一 effort；
* 同一 Source；
* 同一 CandidateCheck；
* 一个候选；
* 一个 Gate1 round。

---

# 十、真正应该记录的指标

不能只记录最终 `PASS/FAIL`。

每核至少记录：

| 指标                               | 说明                                          |
| -------------------------------- | ------------------------------------------- |
| time-to-semantic-plan            | 形成可用语义计划的时间                                 |
| time-to-first-valid-artifact     | 第一次 CandidateCheck PASS 的时间                 |
| last-known-good retained         | 合法候选是否因后续操作丢失                               |
| planner / builder / critic turns | 每阶段真实轮数                                     |
| protocol/tool-discovery turns    | 浪费在协议上的轮数，目标应为 0                            |
| CandidateCheck calls             | 机械修正成本                                      |
| TypeCheck result                 | 最终 artifact 是否合法                            |
| Gate1 frontier                   | missing/extra op、route、predicate、effect 等差异 |
| Gate1 pass                       | 一轮能否闭合                                      |
| wall time                        | 总墙钟时间                                       |

尤其是 `Gate1 frontier`。

Linear 即使最终 Gate1 没过，只要相比旧流程：

* K-loop 已进入 J；
* carrier 路径更完整；
* missing ops/edges 大幅下降；

就说明 semantic planning 有价值。

反过来，如果 TypeCheck PASS，但 Gate1 差异完全没改善，那就说明 planner 的好计划没有真正传递到 artifact。

---

# 十一、现在能得出的准确结论

目前还不能说：

> staged Lift 已经提高了 LLM 语义恢复能力。

因为真正的 Gate1 证据还不足。

但也不能说：

> LLM 根本没有能力。

因为：

* Flash 失败于提交协议；
* GroupNorm 已经产生 CandidateCheck PASS artifact，后来被 reviewer 破坏；
* Linear 已经形成 TypeCheck PASS artifact。

当前证据最支持的是：

[
\boxed{
\text{模型可能有足够的语义能力，}
\quad
\text{但工作流尚不能稳定保存和交付它的成果。}
}
]

这是一个比“换模型”或“增加预算”更根本的问题。

---

# 十二、可以直接发给 agent 的裁决

> 当前三核 pilot 暂不判定为模型语义能力失败。Flash 是提交协议失败；GroupNorm 是已通过 CandidateCheck 的候选被后置 reviewer 破坏；Linear 才是尚待 Gate1 裁决的语义样本。
>
> 停止 `planner → builder → reviewer直接修改成品`。改为 single-writer transactional Lift：一个 source-aware owner 负责最终完整 `(J,C,π)`；planner/critic 只生成 proof-irrelevant plan 或 issue list，不得覆盖 artifact。
>
> builder 每次修改必须在 working copy 上执行 CandidateCheck；只有 PASS 才更新 `last_known_good`，失败立即回滚。任何预算耗尽或 critic 终止时，只要存在 `last_known_good`，必须提交该候选，禁止丢弃。
>
> 删除文本 submit marker，改为专用 `submit_plan/submit_candidate` 工具；planner 不得自行猜工具路径。固定工具负责 handle 分配、引用更新、声明排序、JSON 规范化、checkpoint 和 rollback，但不得选择 operator、B relation 或 J/C/π 所有权。
>
> staged 仍可保留为内部阶段与 receipt 标签，但语义规划、构造和修复应尽量处于同一长生命周期 agent context 中。下一轮仍使用三核、单候选、单 Gate1，对比当前 multi-agent 与 single-owner 流程，核心指标是 time-to-first-valid-artifact、合法候选保留率、总轮数和 Gate1 frontier，而不是增加重试碰运气。

最重要的一句话是：

> **现在不是 LLM 思考得不够多，而是 LLM 思考出的正确东西没有被系统以事务方式保存下来。**


## 先给裁决

这三条 pilot **目前更像是 Lift 工作流设计失败，而不是 J 的思想失败，也不是模型理解不了 Kernel**。

甚至从日志看，LLM 的语义能力已经部分表现出来了：

* Flash 的 planner 在第 10 轮已经识别出 loop 和 online-softmax，却因为提交 marker 的格式错误被判 `LimitsExceeded`；
* GroupNorm 的 builder 已经产出一份 **CandidateCheck 通过的完整 artifact**，却被后置 reviewer 改坏了 handle/node 顺序；
* Linear 最终能生成 TypeCheck 通过的 artifact，只是用了 923 秒。

所以当前三例分别暴露的是：

| 样例        | 真正失败点                   | 暂时不能得出的结论                    |
| --------- | ----------------------- | ---------------------------- |
| Flash     | 提交协议脆弱                  | 不能说 LLM 不懂 online recurrence |
| GroupNorm | reviewer 破坏已闭合 artifact | 不能说 builder 生成不了 J           |
| Linear    | 调用轮数和机械构造成本过高           | 不能说 J 表达能力不足                 |

一句话概括：

> **模型在做语义理解，但系统把它的大部分时间消耗在找路径、满足提交协议、手工维护 JSON 引用和多阶段反复重读上。**

---

# 一、当前流程为什么没有发挥 LLM 的能力

你现在虽然叫它“agent”，实际上却在把它当成三个相互断开的远程 JSON 作者：

```text
Planner
   ↓ 写一个 plan 文件
Builder
   ↓ 重新理解并手工构造完整 artifact
Reviewer
   ↓ 再次重写完整 artifact
```

这不是 Codex 式 agent loop，而更像：

```text
三个独立模型依次接管同一个脆弱文件
```

它有四个根本问题。

## 1. Planner 的理解没有真正减少 Builder 的工作

Flash planner 已经花了近 10 轮理解 online softmax；但 builder 仍需要：

* 再读 Source；
* 再理解 schema；
* 再构造 J；
* 再分类 C；
* 再构造 π；
* 再维护 opaque handle；
* 再满足所有引用和顺序约束。

如果 planner 输出只是一份文字或半结构化 plan，而 builder 仍然必须重新完成绝大多数推理，那么 staging 只是把一次任务变成两次任务。

真正有效的 plan 至少应该让 builder 不再重新判断：

```text
logical signature
主要 A nodes 与 dataflow
reduction / recurrence state
B 中的 domain、predicate、effect
C 的候选 slot
π 需要保留的 realization stages
已知风险
```

否则 planner 只是额外增加调用数。

---

## 2. Reviewer 不应该拥有“重写已通过 artifact”的权限

GroupNorm 是最明确的反例：

```text
builder:
    完整 artifact
    CandidateCheck PASS

reviewer:
    修改 16 轮
    破坏 handle/node order
    最终没有 candidate
```

这说明 reviewer 的角色设计是错的。

一个 artifact 一旦通过机械检查，后置 reviewer 最多只能：

```text
输出只读问题清单
```

不能：

```text
重新生成或重写整份 artifact
```

否则语义 reviewer 与机械闭合之间互相踩踏。

把顺序改成：

```text
planner → critic → builder
```

比当前版本好，但我认为仍然不够彻底，因为它还是三个独立上下文。

更合理的是：

```text
一个持续存在的 Lift agent
    内部先规划
    调用只读 critic/checker
    自己修正
    最后只提交一次完整 artifact
```

或者：

```text
planner/critic 只负责建议
唯一 builder 拥有写权限
```

---

## 3. 提交 marker 和路径猜测是“agent-hostile protocol”

Flash 不是没完成任务，而是：

> 高质量计划已经写出，但 marker 没有单独位于第一行。

这类失败不应该消耗整个语义任务。

不要再让 agent 用 shell 文本协议表达：

```text
第一行必须是某个 marker
然后不能有其他输出
文件必须在某个路径
```

应当直接提供结构化工具：

```text
submit_lift_plan(plan_object)
submit_candidate(candidate_object)
```

工具成功就返回：

```text
accepted / rejected + structured diagnostics
```

不能让一个换行或 shell 输出位置摧毁 192 秒的语义推理。

---

## 4. “一个 Gate1 round”被错误地等同于“只准一次内部修正”

你们的设计本来允许 source-aware Lift 在一次生成 episode 内调用有界的 `CandidateCheck`；它调用正式 parser、Decode 和 TypeCheck，只返回机械 diagnostics，不生成语义、不签发证书，也不改变最终 SCRC 谓词。

并且只有最终完整候选通过 TypeCheck、真正进入 fresh Replay 后，才算一次正式 Gate1 round；Replay 前的 JSON、引用和类型修正不属于 Gate1 round。

所以完全可以同时保持：

```text
正式候选数 = 1
正式 Gate1 round = 1
```

但允许这一个 Lift agent 内部：

```text
CandidateCheck 3–5 次
```

并如实记录：

```text
candidate_check_calls
model turns
tool calls
time-to-TypeCheck
```

这不是用重试掩盖质量，而是在发挥 agent 最基本的：

> 写—检查—修正

能力。

你现在把这条循环压得太死，等于让 agent 用第一版复杂 JSON 直接参加正式考试。

---

# 二、Codex 为什么看起来可能会快很多

你的直觉有一部分是对的。

Codex 这类 coding agent 的工作模式就是：

```text
持续读取代码库
编辑文件
运行命令和测试
观察失败
继续修改
```

官方说明也把 Codex CLI 定义成能读取、修改并运行本地代码的 coding agent；Codex 的软件工程训练和运行方式强调在实际环境中反复运行测试直至获得通过结果。([OpenAI Help Center][1])

这与“写一份复杂 typed artifact，再运行 CandidateCheck 修正”的任务非常匹配。

但真正有用的不是 `Codex` 这个名字，而是它背后的工作方式：

* 一个持续上下文；
* 真实文件系统；
* 显式工具；
* 可以反复运行 checker；
* patch 而不是全文重写；
* 测试失败后继续；
* 不需要靠特殊 marker 交付。

Codex 还面向长时间、sandbox 内的完整工程任务，并支持让多个 agent 在独立工作区中并行。([OpenAI][2])

不过，**直接把 miniSWE 换成 Codex，不改当前协议，也未必解决问题**。如果仍要求 Codex：

```text
猜工具路径
维护特殊 marker
planner/builder/reviewer 三次重开上下文
reviewer 重写通过的 artifact
```

它一样会浪费时间。

所以最准确的判断是：

> **Codex 很可能是更适合的 Lift execution engine，但真正必须先改的是 agent–artifact interface。**

---

# 三、生成 J 的思想有没有问题

## 目前没有证据说明 J 的语义设计错了

相反：

* Flash planner 已识别 online recurrence；
* GroupNorm builder 已写出 CandidateCheck 通过的 artifact；
* Linear 已 TypeCheck 通过。

这些都说明当前模型至少能够理解：

* recurrence；
* reduction；
* consumer/effect；
* carrier；
* J/C/π 的基本分工。

真正还没回答的是：

> 这些 mechanically valid artifact 是否足以通过 Gate1，并在 Gate2 中正确？

所以现在不能因为流程慢，就修改 J schema。

最应该做的是：

1. 保存 GroupNorm **reviewer 修改前**的 builder artifact；
2. 直接对它运行正式 TypeCheck、Replay、Gate1；
3. 保存 Flash planner 的完整 plan；
4. 让唯一 builder 从该 plan继续，而不是把 plan 作为失败任务丢弃；
5. 等 Linear 的 Gate1 结果。

如果 builder artifact 能通过 Gate1，那么当前问题几乎可以确定是 orchestration，而不是 J。

---

## 但 J 的“作者化接口”确实设计得太低层

模型现在不只需要决定语义，还要手工处理：

```text
opaque IDs
declaration order
topological order
cross-object refs
node/result handles
π instance refs
route slots
sharing/order
JSON serialization
submit protocol
```

这相当于让人先理解 FlashAttention，再手写一份带全局 SSA 引用的编译器 IR JSON，而且一次不能出错。

LLM 应负责的是：

```text
operator 是什么
axis 是什么
dataflow 是什么
domain/predicate/effect 是什么
什么属于 C
π 中有哪些 realization stages 和关系
```

不应把大量推理预算用于：

```text
n7 应该先于 n8 声明
i12 应该引用 v4 还是 n4.result
这个 JSON 数组顺序是否满足 parser
```

---

# 四、最应该增加的不是新 J 字段，而是 typed authoring tool

这不会违反你们的 LLM-first 哲学。

最终 proof artifact 仍然只有：

```text
(J, C, π)
```

工具也不替 LLM判断语义。

它只提供类似：

```text
declare_input(...)
add_node(op, operands, attrs)
add_domain(...)
add_effect(...)
add_c_slot(...)
add_pi_instance(ref, region)
add_route(source, target, slot)
candidate_check()
submit()
```

由工具机械负责：

* 分配 opaque ID；
* 保证 declaration order；
* 维护引用；
* 生成规范 JSON；
* 在类型能唯一推出时物化类型；
* 报告 closed diagnostics。

LLM 仍然决定每一个语义内容和关系。

可以理解成：

```text
现在：
LLM 手写 MLIR bytecode

改后：
LLM 使用 IRBuilder
```

这不是 deterministic semantic composer，因为工具不会决定：

* 这是 sum 还是 max；
* axis 是什么；
* predicate 是什么；
* effect 是什么；
* C/π 如何分类。

它只是避免模型手工维护序列化细节。

你们现有设计已经允许确定性的 Decode/Elaborate 和同一规则集的 CandidateCheck，只要它不补写或发明语义。 

---

# 五、我认为最合理的新 Lift 工作流

不是当前：

```text
planner → builder → reviewer
```

而是：

```text
一个 persistent source-aware Lift agent
        │
        ├─ inspect_source()
        ├─ inspect_source_outline()
        ├─ inspect_compiler_view()
        │
        ├─ 在同一上下文中形成 semantic sketch
        │
        ├─ 用 typed builder 构造 J
        ├─ 用 typed builder 补 C
        ├─ 用 typed builder 构造 π
        │
        ├─ CandidateCheck
        ├─ 根据 diagnostics 局部 patch
        ├─ CandidateCheck
        │
        ├─ read-only self-review / critic
        ├─ 必要时再次 patch + CandidateCheck
        │
        └─ submit one complete (J,C,π)
                         │
                 external TypeCheck
                         │
                   fresh Replay
                         │
                      Gate1
```

几个关键点：

## 一个 agent 拥有最终 artifact

不能多人轮流全文重写。

## Critic 只读

输出：

```text
issue_code
artifact_path
reason
severity
```

不直接修改 artifact。

## Checker 可以多次调用

但最终：

```text
formal artifact = 1
formal Gate1 round = 1
```

## 所有中间版本不可变保存

至少保存：

```text
semantic sketch
pre-check candidate
post-check candidate
critic report
final candidate
```

否则像 GroupNorm 一样，builder 的成功结果会被 reviewer 覆盖掉，无法分析。

---

# 六、为什么会这么慢

GroupNorm 已经用了：

```text
planner 9 turns
builder 32 turns
reviewer 16 turns
```

总计约 57 个模型回合。

即使每回合平均只需要 10 秒，也已经接近 10 分钟；如果每回合需要重新发送：

* Source；
* schema；
* protocol；
* plan；
* artifact；
* tool history；

延迟会更高。

所以“staged”没有自动变快，是因为你们做的是：

```text
多阶段 × 多轮 × 重复上下文
```

而不是：

```text
一次高密度推理 + 快速本地工具循环
```

应当逐调用记录：

```text
queue time
time to first token
generation time
prompt tokens
cached prompt tokens
output tokens
tool execution time
full-context bytes resent
```

否则 923 秒究竟是：

* provider 慢；
* xhigh reasoning 慢；
* 30 次网络 round trip；
* 每轮重发巨大 context；
* 共享并发队列；

目前无法区分。

先用单个 Linear case 串行跑一次，能排除三核并发对时延的影响。

---

# 七、开发模式和论文评测模式必须分开

你现在为了“不用重试掩盖质量”，把生产工作流也人为限制成 one-shot，这会伤害系统本身。

建议同时保留两个指标。

## One-shot capability

```text
一个 agent response
不调用修正工具或只调用一次
TypeCheck@1
Gate1@1
```

用于测模型裸能力。

## Agentic bounded lifting

```text
一个 persistent agent episode
最多 k 次 CandidateCheck
一个最终 artifact
一个正式 Gate1 round
```

用于测实际系统能力。

报告：

```text
one-shot TypeCheck
bounded TypeCheck
time-to-first-valid
CandidateCheck calls
Gate1 closure
total wall time
```

这样既不会用无限重试粉饰质量，也不会故意废掉 agent 的核心能力。

---

# 八、是否应该直接做一个 Codex A/B pilot

我认为应该，而且只需要当前三个 Kernel。

## A：当前 staged miniSWE

```text
planner → builder → reviewer
```

## B：Codex-style single persistent agent

```text
one agent
+ named tools
+ artifact file
+ CandidateCheck loop
+ no reviewer rewrite
```

为了定位原因，最好再加：

## C：同一个模型，但使用 B 的工作流

这样可以区分：

* 是模型/提供商差异；
* 还是 agent orchestration 差异。

统一比较：

| 指标                   |  A |  B |  C |
| -------------------- | -: | -: | -: |
| wall-clock           |    |    |    |
| model calls          |    |    |    |
| tool calls           |    |    |    |
| prompt/output tokens |    |    |    |
| protocol failures    |    |    |    |
| time-to-TypeCheck    |    |    |    |
| TypeCheck pass       |    |    |    |
| Gate1 pass           |    |    |    |
| Gate1 diff size      |    |    |    |
| Gate2 pass           |    |    |    |

只看三个案例就能快速判断。

---

# 九、我对当前推进的具体裁决

现在不要：

* 改 J schema；
* 增加新的 planner 字段；
* 增加 reviewer 预算；
* 把 Gate1 rounds 从 1 提高到很多；
* 再跑大 batch；
* 因为慢就判断 LLM 不具备能力。

现在应该：

1. **保留 46c54aa6 作为当前对照。**
2. 取回 GroupNorm 的 pre-review builder artifact，直接运行 Gate1。
3. 修复 Flash submit 为结构化工具调用，不再使用 marker。
4. 删除后置可写 reviewer。
5. 做一个 single persistent agent Lift backend。
6. 在一次 Lift episode 中允许 3–5 次 CandidateCheck，但正式 Gate1仍为 1。
7. 先用 Linear、Flash forward、GroupNorm 做 A/B/C。
8. 等看 Gate1 diff，再判断是 J 语义、π realization、Replay 消费还是纯 orchestration 问题。

---

# 最终判断

你说“没有发挥 LLM 的能力”，我基本同意，但原因不是模型一定不行，而是：

[
\boxed{
\text{当前系统把 LLM 当成分阶段的脆弱 JSON 生成器，}
\quad
\text{而没有把它当成可持续读写、调用 checker、局部修复的 agent。}
}
]

Codex-style agent 很可能显著改善这个问题，因为这种任务天然适合“编辑—检查—修正”的 coding-agent loop；但最重要的不是换模型，而是：

[
\boxed{
\text{一个 artifact owner}
+
\text{typed authoring tools}
+
\text{结构化提交}
+
\text{有界 CandidateCheck}
+
\text{只读 critic}
}
]

从当前三条证据看，**我暂时不会怀疑 J 的科学思路；我会优先判定当前 staged Lift 的工程接口设计错了。**

[1]: https://help.openai.com/en/articles/11096431?utm_source=chatgpt.com "OpenAI Codex CLI – Getting Started | OpenAI Help Center"
[2]: https://openai.com/index/introducing-the-codex-app/?utm_source=chatgpt.com "Introducing the Codex app | OpenAI"
