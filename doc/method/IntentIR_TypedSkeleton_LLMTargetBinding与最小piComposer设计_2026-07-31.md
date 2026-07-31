# IntentIR：Typed Skeleton、LLM Target Binding 与最小 π Composer

> 日期：2026-07-31  
> 状态：Task 17 的实现依据；接管 Task 16 的 flat/grouped-source composer 路线。

## 1. 裁决

Replay 不应在两个极端中选择：

- 不能继续让 LLM 按每个 π occurrence 手写完整 Triton；
- 也不建设一个固定、语义完备的 J→Triton 编译器。

采用中间结构：

```text
S = Skeleton(J, π_min, MaskC(C))
B = AgentBind_f(S, J, π_min; U_f)
Replay = Render_f(S ⊗ B)
```

- `S` 是固定代码对既有 J/π 的 target-neutral 结构化重索引；
- `B` 是 fresh source-blind Replay agent 作者化的紧凑 target binding；
- `Render_f` 只实例化、连线、维护作用域并序列化，不选择语义 lowering；
- `B` 与 composer 工作区均为 proof-irrelevant episode scratch，结束即销毁。

认证公式不变：

```text
SCRC_CERTIFIED
  iff TypeCheck(J,C,π)
  and Gate1(Source(C), Replay(C))
  and Gate2(Source(C), Replay(C), RefEval(J))
```

## 2. 三方职责

### 2.1 J：唯一计算语义

J 继续唯一声明 operator、operand、type、shape、domain、predicate、access、effect、reduction、
control 和 result relation。任何会改变 logical value 的事实都不能迁往 π 或 binding shell。

### 2.2 π_min：最小 realization 见证

π 只保留不能由 J 与 closed structural convention 唯一恢复的事实：

- occurrence multiplicity 与 semantic occurrence placement；
- region tree、physical iteration、carrier、program axis/linearization；
- terminal、materialization、sharing、order 与跨 region topology；
- 非 identity、跨 occurrence 或存在多候选端点的 route；
- representation-only opaque attachment。

以下内容不再要求 Lift 重抄：

- J operand graph 已声明且同 region 中 occurrence 唯一的 1:1 data route；
- 唯一的 declaration/topological order；
- 可由 typed endpoint 唯一确定的 local/result kind；
- 结构化 wire 中纯机械的全局 opaque ID。

省略只允许发生在 closed、可证明唯一的情形。存在 multiplicity、branch、cross-region、carrier、
sharing 或多个 endpoint 时必须由 Lift 显式作者化，不能由固定代码猜测。Decode 只展开这一无损
编码，不修复或补造语义。

π 仍禁止 operator、formula、predicate 文本、fill、target API、Source、Gate diff、Kernel 身份或
recipe/template 标签。

### 2.3 Replay agent：target binding 作者

Agent 不再写整份目标源码。它为 skeleton 中不同的 semantic/target choice 写 binding：

- target operator、callee 与 expression；
- target predicate/mask/fill 表达；
- loop/branch 的 target spelling；
- cast、layout、reshape、broadcast 与 target representation；
- unsupported/new IntentOp 的开放世界实现；
- 编译失败后的 occurrence override。

一个 binding 可由 agent 显式匹配多个 J semantic nodes，再由 composer 沿 π 展开为多个 physical
occurrences。只有真实 realization 不同的 occurrence 才写 override。

## 3. Composer 的复杂度边界

固定 composer 可以很复杂，但只能复杂在结构和机械性上：

- 构造 typed semantic/realization skeleton；
- region/block tree、scope、dominance、stable local/SSA；
- instance expansion、route binding、carrier tuple、order/sharing；
- target-neutral `expr/assign/call/if/loop/return/effect/raw` AST；
- binding matcher 的精确执行、参数检查和 occurrence override；
- source map：目标行回指 binding、J node、π instance/region；
- 局部 render、compile/check、checkpoint 与原子 submit。

固定 composer 不得内置：

- `J op -> Triton API/operator` 映射；
- `π relation -> target statement` 映射；
- per-kernel helper、历史 winner 或共享 semantic template；
- 缺失 predicate、formula、effect、loop bound 或 update 的默认补写。

因此它相当于“没有 instruction selection 的编译器”。instruction selection 与新 target 表达仍由
Replay agent 完成。

## 4. Binding 的最小形式

Binding 是 episode-local 的 closed authoring object，概念形式为：

```json
{
  "templates": [
    {
      "id": "b0",
      "parameters": ["arg0"],
      "target_ast": {"kind": "call", "callee": "tl.exp2", "args": ["arg0"]}
    }
  ],
  "applications": [
    {"binding": "b0", "semantic_nodes": ["n17", "n29"]}
  ],
  "overrides": []
}
```

具体 schema 以实现为准，但必须满足：

- agent 显式提供 target callee/operator/control 与所有 semantic parameters；
- matcher 只能匹配当前 J 的 typed fields，并在 evidence 中展开为精确 node IDs；
- fixed code 不选择 binding，也不从 op 名生成 target AST；
- 每个需要 target 表达的 skeleton hole 恰好一个 binding，缺失或重复均 fail closed；
- raw escape 保留开放世界能力，但仍绑定明确的 inputs、outputs、region 和 J provenance。

## 5. Lift 的新负担

Lift 仍由一个 persistent source-aware owner 完整作者化 `(J,C,π_min)`，但不再维护庞大的 flat route
ledger。它只回答：

- 计算语义属于 J 的什么节点与 boundary relation；
- 非唯一 physical occurrences 位于哪个结构化 region；
- 哪些 realization 需要显式 carrier/route/order/terminal/materialization 见证。

固定 CandidateCheck/TypeCheck 只验证 closed dependent relations，不补写 Lift 遗漏。结构化 wire 的
canonical ID、唯一 1:1 edge 展开和 declaration order 属于 lossless Decode，不改变证明对象信息量。

## 6. Source-blind 生命周期

每个 formal Gate1 round：

1. runner 由 current `(J,C,π_min)` 构造 target-neutral skeleton；
2. fresh Replay agent 只得到 skeleton/J/π_min/MaskC/U_f 的文件路径；
3. agent 作者化/局部修改 binding，而非完整源码；
4. composer 展开 occurrence、route、region 和 SSA，生成 Replay；
5. agent 根据自身 compile/check 与 own-TTIR 修 binding；
6. exact Replay bytes 通过外部 ingress 后进入 Gate1；
7. episode workspace、binding 和 helper 全部销毁。

Gate1 失败若属于 J 缺失，反馈给同一 Lift owner；属于 target binding/realization，则启动 fresh Replay，
不重新 Lift 正确 J。

## 7. No-leak / J-authority

- Composer 输入不得包含 Source、Harness concrete value、Gate、历史 Replay 或 Kernel identity；
- 每个 semantic target AST node 必须携带 J provenance；
- π-only skeleton 只能形成 region/occurrence shell，不能形成可运行计算；
- Mask-J 必须 fresh 生成 binding，禁止复用正常 J 下的 `B`；
- 改变一个闭合 J fact 后，fresh binding/Replay 必须随 J 改变；
- production backend 仍只读取 `J+C`，不读取 π 或 Replay binding。

## 8. 实施顺序

1. 定义 target-neutral skeleton、hole 与 binding schema；
2. 将 flat π wire 改为结构化最小 wire，并证明 Decode 无损；
3. 实现 skeleton builder 与结构 renderer，不加入 semantic lowering table；
4. 将 Codex Replay 改为 binding authoring loop；
5. 用同一 Flash-forward accepted J 对比 v2：输出规模、首次编译时间、route/region/SSA frontier；
6. 闭合 Llama、SplitK forward、Flash forward/full backward；
7. 对原 22 项 fresh Replay，并执行 Mask-J/J-mutation 验收。

最终原则：

> **固定 composer 负责完整的结构化程序构造，fresh LLM 负责紧凑的 target instruction selection；
> π 只保留非唯一 realization，J 始终是唯一计算语义。**

