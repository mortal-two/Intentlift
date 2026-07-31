# IntentIR：Typed Skeleton、LLM Target Binding 与最小 π Composer

> 日期：2026-07-31  
> 状态：历史失败方案；Task 18 已停止 hole/binding/fixed structural renderer 路线。

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
- `B` 是 fresh source-blind Replay agent 作者化的少量 target source blocks；
- `Render_f` 固定生成 ABI、program coordinate、region/loop/carrier 等 target plumbing，
  再嵌入 `B`；它不选择 J 的值语义 lowering；
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

第一阶段先落地严格双射的 nested wire：region 递归嵌套，occurrence 继承所在 region，连续 `r*/i*`
由 ordinal 机械恢复，其余 relation 完整保留。真实 Flash-forward 的 π 从 9064 个 compact JSON
字符降至 4186，Decode 后与 canonical π 完全相等。这个结果只证明序列化减负；尚未实现的唯一
route 省略不能提前计入 π_min，也不能用字节减少冒充语义简化。

π 仍禁止 operator、formula、predicate 文本、fill、target API、Source、Gate diff、Kernel 身份或
recipe/template 标签。

### 2.3 Replay agent：target binding 作者

Agent 不再从空文件写整份目标源码，也不再手工维护 target AST、application、region 和 result
alias。固定代码预生成不可变 manifest 与普通 `.source` 文件；agent 只在少量 region block 中写：

- target operator、callee 与 expression；
- target predicate/mask/fill 表达；
- J 值计算与必要的非唯一 branch spelling；
- cast、layout、reshape、broadcast 与 target representation；
- unsupported/new IntentOp 的开放世界实现；
- 编译失败后的 occurrence override。

每个 block 的 J/π provenance、lexical region 与前后顺序均由 scaffold 预填，agent 不能改；同一
block 可以一次实现多个连续 J 节点和 access stage。只有目标实现确有必要时才进一步拆成局部
override，不能退回 per-occurrence 全文重写。

## 3. Composer 的复杂度边界

固定 composer 可以很复杂，但只能复杂在结构和机械性上：

- 构造 typed semantic/realization skeleton；
- region/block tree、scope、dominance、stable local/SSA；
- instance expansion、route binding、carrier tuple、order/sharing；
- logical signature alias、program axis/linearization 的 target plumbing；
- 由 J trip 与 π iteration 唯一确定的 loop header、carrier init/update/result；
- 不可变 block manifest、region sequence、scope/dominance 与 stable locals；
- source map：目标行回指 binding、J node、π instance/region；
- 局部 render、compile/check、checkpoint 与原子 submit。

固定 composer 不得内置：

- `J op -> Triton API/operator` 映射；
- 从 `π` 猜测任何 value/domain/predicate/address/mask/effect；
- per-kernel helper、历史 winner 或共享 semantic template；
- 缺失 predicate、formula、effect、loop bound 或 update 的默认补写。

允许的 target plumbing 必须满足：删除 J 后只能得到 ABI、坐标和空 control shell，不能得到任何
计算结果或 memory effect。因而 composer 是“拥有结构 lowering、没有 value instruction
selection 的不完整编译器”；计算、地址、mask、load/store 与开放世界 target 表达仍由 Replay
agent 完成。

## 4. Binding 的最小形式

Binding 是 episode-local 的 closed manifest，概念形式为：

```json
{
  "protocol": "intentir.replay-compact-binding.v2",
  "blocks": [
    {"id": "b0", "semantic_refs": ["n0", "n1"],
     "realization_refs": ["access:a0@r1", "rh0"],
     "source_path": "binding/b0.source"}
  ]
}
```

具体 schema 以实现为准，但必须满足：

- manifest 完全由 fixed scaffold 产生；agent 只能编辑其指向的普通 `.source` 文件；
- 每个 block 固定绑定连续 J refs、access/realization refs 和 lexical region；
- fixed code 不从 op 名生成 value expression、地址、mask 或 memory effect；
- 缺文件、空文件、增删 block、改 provenance/path 均 fail closed；
- raw block 保留开放世界能力，exact bytes 与 block-level J/π source map 一同留证。

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
2. runner 预生成不可变 manifest 与少量空 `.source` block；
3. fresh Replay agent 只按路径查询所需 J/π workset，并直接编辑这些普通源码文件；
4. composer 固定生成 ABI/program/region/loop/carrier shell，嵌入 block 生成 Replay；
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
4. 将 Codex Replay 改为 file-native block edit → render → compile/check loop；
5. 用同一 Flash-forward accepted J 对比 v2：输出规模、首次编译时间、route/region/SSA frontier；
6. 闭合 Llama、SplitK forward、Flash forward/full backward；
7. 对原 22 项 fresh Replay，并执行 Mask-J/J-mutation 验收。

最终原则：

> **固定 composer 负责 target plumbing 与结构化程序构造，fresh LLM 负责全部 value/address/effect
> target code；
> π 只保留非唯一 realization，J 始终是唯一计算语义。**
