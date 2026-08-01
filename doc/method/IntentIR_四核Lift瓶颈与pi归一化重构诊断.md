# IntentIR 四核 Lift 瓶颈与 π 归一化重构诊断

> 性质：当前方法诊断，不是证书、实验结论或修改流水账。
> 范围：最后四个未闭合 Kernel，以及 one-col 辅助样本暴露的共性问题。

## 1. 结论

最后四项是：

1. `high_performance:liger_llama4_rope`；
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`；
3. `high_performance:flash_attn.flash_attn_triton._fwd_kernel`；
4. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`。

它们当前均没有 Gate1 closure，也均未进入 Gate2。TypeCheck 通过只说明 JSON 与
dependent contract 机械合法，不构成进展，也不能证明 J 正确。

当前主要矛盾不是 Source 编译慢，也不是 compact π 太大，而是：

- Lift Agent 仍手工作者化一份 30--46 KiB、跨字段强依赖的完整 `(J,C,π)`；
- 输入工作区重复提供完整 schema、contract、IntentOps、preflight、示例与工具说明；
- 工作轨迹和历史工具输出常比 candidate 大一个数量级；
- 当前 π 只表达 program-axis binding，而固定 normalizer 实际是 identity；
- Gate1 因而继续比较物理 loop、carrier、address、control 与 program topology；
- 即便 Source 和 Replay 都实现同一个正确 J，不同合法 TTIR realization 也会被拒绝。

因此，需要同时修复两件事：

1. source-aware Lift 必须正确作者化 J/C，特别是 logical loop 与 C tile attachment；
2. π 必须恢复少量、closed、typed realization bindings，让固定 unary normalizer
   能把不同物理实现归一到同一个 J/C-owned structure。

不应恢复旧的 region/instance/route 大图，也不能靠放宽 Gate 或 pairwise diff 白名单闭合。

## 2. Lift 时间花在哪里

### 2.1 四核观测

| Kernel | 三个首次 Lift 的量级 | 后续 relift 量级 | 代表 artifact 大小 | 当前结果 |
|---|---:|---:|---:|---|
| Llama4 RoPE | 654--706 s | 112--269 s | 29--40 KiB | Gate1 0 closure |
| FlashAttention forward | 872--1516 s | 35--174 s | 约 35 KiB | Gate1 0 closure |
| FlashAttention full backward | 1438--1785 s | 20--246 s | 约 46 KiB | Gate1 0 closure；另有基础设施失败 |
| SplitK forward | 646--900 s | 80--832 s | 约 46 KiB | Gate1 0 closure |

代表性 Source TTIR 编译约为数秒，CandidateCheck 和普通工具调用通常是毫秒到百毫秒量级。
十几分钟至半小时的首次 Lift 主要消耗在模型回合、上下文处理、完整 artifact 生成和反复修复，
不是编译器执行。

### 2.2 机械修复占用了大量回合

真实失败反复集中在：

- node/handle declaration order；
- scope definitions、captures 与 storage lifetime；
- signature dtype/rank/logical_ref；
- value、dimension 与 node 的交叉引用；
- conditional arm-local value 与 effect membership；
- inout pre-state/post-state、output binding；
- C role、subject_ref 与 axis_ref；
- authorization/refinement dtype；
- 完整 schema 中的闭合枚举和必填字段。

这些错误中只有一部分涉及语义判断；大量错误是模型手工维护 dependent JSON 的机械成本。
某些首次 Lift 在最终提交前产生十余个 working checkpoint，并触发数十次 CandidateCheck；
Source 编译与 checker 时间相对于总墙钟可以忽略。

### 2.3 输入与历史上下文过重

当前 Lift 工作区通常同时挂载：

- 匿名 Source；
- Harness/compiler view；
- Source TTIR 或其查询入口；
- 完整 artifact schema；
- Lift contract；
- IntentOps 说明；
- preflight；
- minimal example；
- schema/context/check 工具和重复的使用说明；
- persistent continuation 与历史工具轨迹。

candidate 本身通常只有几十 KiB，但历史 trajectory 可达到数百 KiB。当前主要注意力浪费在：

1. 同一规则在 system prompt、instance prompt、README、contract 与 continuation 中重复；
2. Agent 先浏览大 schema，再寻找局部字段，而不是按当前错误查询一个 closed fragment；
3. 每次局部语义修改都可能迫使模型同步多个 handles、scopes、effects 和 bindings；
4. 历史工具输出虽被部分压缩，仍持续占据长生命周期 session 的注意力；
5. 失败诊断混合机械错误、J/C semantic error、Replay fidelity 与 Gate realization residual。

### 2.4 compact π 不是当前耗时来源

当前空 π 只有 schema 与空 `program_axes`，体积不足百字节。每条 axis row 也只是几十字节。
它不会解释十几分钟的 Lift 时间。

真正的问题相反：π 被压缩到几乎没有消费能力，导致本来应由 normalization 处理的物理差异，
最终全部回到 Replay 猜测和 Gate1 hard comparison。

## 3. 四个 J/C/π 的静态判断

以下判断只基于 Source、adapter 与 artifact 的直接对应。没有 Gate2 的对象不能称为正确，
因此使用“核心图合理”“存在明确缺口”“无法判定”三种表述。

### 3.1 Llama4 RoPE

J 已表达：

- Q/K 两条 rotary 路径；
- 前后半维 gather/scatter；
- cos/sin 数据流；
- f16/f32 casts；
- Q/K inout read/write effects；
- 两个输出及 exact result relation。

静态判断：核心 J 很可能合理。目前没有直接证据表明 rotary 公式或 output ownership 错误。

当前主要缺口：

- Source 使用两个 program axes，Replay 候选曾生成额外 program coordinate；
- π 虽记录 domain-to-axis rows，normalizer 不消费它；
- Gate1 因而把 `get_program_id`、`addptr` 数量与完整 address dataflow 当成 hard；
- C 只有一个 tile slot，尚未与 program/domain realization 形成可消费的闭合关系。

本核更像“J 基本正确，Replay realization 与 Gate normalization 未闭合”，但仍需 Gate2
证明数值结果，尤其是 casts 与 exact relation。

### 3.2 FlashAttention forward

J 已表达 online softmax 的核心结构：

- logical key-axis loop；
- running maximum、normalizer 与 accumulator 三个 carried states；
- QK contraction、max、exp、sum 与 accumulator rescale；
- output、LSE 与 scratch materialization effects；
- query/key/feature extents 与 runtime scalar equality；
- tolerance result relations。

静态判断：online recurrence 核心 J 很可能合理。当前没有证据支持把它重新压成普通全局 softmax。

当前主要缺口：

- C 记录 query/key tile，却没有形成 `J loop + logical domain + tile slot` 的直接消费关系；
- π 只有 program axes，没有 iteration、carrier 或 scratch materialization binding；
- Source 的 tiled loop、pointer advance、scratch store/load 与 Replay 的合法替代实现产生大量 TTIR diff；
- RefEval/functional probe 对非 direct-J-operand load 仍不能给出判定，这既可能是 boundary consumer
  能力缺口，也可能暴露 access mapping 不够明确。

本核不能简单归为 J 错；应先补 iteration/materialization normalization，再用 Gate2 判断 J。

### 3.3 FlashAttention full backward

当前 specialization 的 Source 包含：

- program-level batch/head mapping；
- key-block outer iteration；
- one-col 内部 query-block recurrence；
- DK/DV accumulators；
- DQ inout update；
- conditional dispatch 与多输出 effects。

J 已表达主要 contractions、casts、DQ update、DK/DV outputs、一个 logical query recurrence 和
effect control。该归纳在算法上是合理的，但比 forward 更难确认，因为 Source 的两个物理循环层次
被归并为 semantic recurrence 加 tiling。

当前明确缺口：

- query/key tile C slots 缺少稳定的 subject_ref/axis_ref attachment；
- π 只记录一个 program-axis row；
- 没有 relation 将 outer tiled iteration、inner recurrence、pointer-only carriers 与 terminal control
  锚定到 J loop/control；
- Gate1 仍比较 Source helper inlining、physical loops、unused/used ABI、address and effect region topology；
- Replay 本身还需要数分钟，并多次出现 compile/API 修复。

静态判断：J 的数值核心“可能正确但证据不足”；C/π 明显不够，不能把当前失败全部归因给模型。

### 3.4 xFormers SplitK forward

Source 的核心是：

- 三个 program axes，其中一个选择 split；
- split chunk low/high；
- tiled K/V iteration；
- running max、normalizer 与 accumulator 三个 numerical states；
- split output 与可选 LSE；
- 多种 specialization 下的 early exit、mask、paged/quantized branches。

当前 J 已保留三状态 online recurrence、QK/exp/sum/PV、split outputs 与三条 program-axis rows，
所以不能说模型完全没理解算法。

但当前 artifact 仍有明显风险：

- 多个 C slots 没有 semantic attachment，仅以通用 role 保存；
- split count、padded query capacity 等出现在 concrete output shapes，参数化身份尚未闭合；
- π 没有 split-axis、mixed-radix、iteration、early-exit/control binding；
- 当前 artifact 对复杂 specialization 的 boundary facts 只覆盖实际固定分支，尚不能证明所有
  representation-only constexpr 与语义参数的所有权划分正确；
- 历史 Lift 反复出现 unknown logical refs、functional axis mismatch、C type/axis 与 runtime carrier
  ownership 错误。

静态判断：online recurrence 核心合理，但 J boundary 参数化和 C/π 尚不成熟；四核中它最不能被称为
“J 已正确，只差 normalizer”。

## 4. one-col 辅助样本说明什么

one-col 不属于最后四项，但它给出了清晰的分层证据：

- `value_dimension_relation(t9,x0,eq)` 实际已经存在；
- QK、softmax backward、DK/DV accumulators、DQ inout 的核心 J 图基本对应 Source；
- J loop 的 `trip=x0` 表示完整 logical axis 上的 semantic recurrence，物理 tile 由 C/π 解释，
  不能直接把 Source 的 `ceil_div(x0,BLOCK_M)` 物理循环次数当作 J trip 错误；
- 但 `BLOCK_M` 对应的 C slot 被附着成 contraction reduction tile，而不是 logical loop 的
  `tile_extent`，这会直接误导 Replay；
- π 又没有 iteration/control binding，Gate1 最终只能比较不同物理 TTIR。

它说明必须先区分：

1. J-owned semantic error；
2. C attachment error；
3. π normalization relation 缺失；
4. Replay 没有忠实消费现有 J/C；
5. Gate observer 把合法 realization difference 留成 hard。

不能将五类问题统一反馈成“重新 Lift J”。

## 5. π 被过度压缩

当前 π 只包含 `program_axes`，而固定 normalizer返回原 observation：

```text
N^π(O) = O
```

所以 π 当前：

- 不帮助合法 TTIR realization 闭合；
- 不能支撑有/无 π 的有效消融；
- 不会减少 Replay 负担；
- 只增加一个很小的 Lift/TypeCheck 对象；
- 无法解释四核共同出现的 loop、carrier、address、control 与 materialization residual。

压缩旧 π 的方向是正确的：旧 realization graph 太大、交叉引用太多，并曾错误进入 Replay。
错误在于只完成了“删除复杂度”，没有保留完成 Gate1 所需的最小 relation algebra 与 consumer。

## 6. 适度复杂而非完整 realization graph 的 π

π 应只包含少量直接绑定，不包含 region tree、opaque instance graph、route assembly、allowed diff、
free-text equivalence 或 target recipe。建议首先支持三类 relation：

```json
{
  "program_bindings": [
    {
      "domain_ref": "d2",
      "axis": 0
    }
  ],
  "iteration_bindings": [
    {
      "loop_ref": "n3",
      "domain_ref": "d4",
      "tile_slot": "c5",
      "form": "serial_tiled"
    }
  ],
  "control_bindings": [
    {
      "control_ref": "n2",
      "form": "terminal_guard"
    }
  ]
}
```

字段只选择预定义 relation kind 与既有 J/C handles；候选不能指定删除哪些 observation fields。

### 6.1 program binding

固定 normalizer 将唯一识别的 physical program coordinate 归一为：

```text
program-coordinate(domain_ref)
```

它必须保留 coordinate multiplicity、logical-domain identity、所有 predicate/effect users。
额外 program coordinate、错误 domain binding 或歧义 mapping 继续失败。

### 6.2 iteration binding

固定 normalizer将：

```text
physical loop bounds/step
+ C tile
+ value carriers
+ pointer-only carriers
```

归一为：

```text
semantic-loop(loop_ref, trip-domain, carried-value transitions)
```

它必须机械检查：

- physical bounds/step 与 J trip domain、C tile 的 closed 关系；
- J value states 的 init/next/result；
- body operator、predicate、reduction 与 effect；
- 被消去的额外 carrier 只用于 address evolution，不流入 value、predicate、store value 或 output；
- pointer advance/recompute 只改变 representation，不改变 J access tensor/index/domain/effect。

任何一项不唯一就返回 unsupported，而不是 PASS。

### 6.3 control binding

固定 normalizer 可将满足 closed 前提的 early return 与 terminal structured if 归一为同一
J `effect_if` anchor。它必须保留：

- predicate value 与 polarity；
- positive/negative arms；
- arm-local effects 与 order；
- continuation 是否为空；
- post-state 与 outputs。

一般 CFG、跨臂 SSA、不同 effect set 或无法证明 terminal 的 control 继续 hard/unsupported。

### 6.4 address 与 materialization

第一阶段不为每个 addptr 建 π row。artifact-aware hard projection 应依据 boundary tensor、J access、
logical index/domain、predicate/fill 与 effect，将纯 address arithmetic 压成 opaque representation edge。
只有出现无法唯一锚定的 materialization 时，才考虑增加一个直接、typed、已有固定 consumer 的
materialization binding；不得预先恢复旧 graph。

## 7. 为什么该 π 不会让 Lift 自己批准自己

安全性不来自 LLM 声称 relation 正确，而来自固定 consumer 对每个 relation kind 的 closed recognizer：

- π 只能选择 relation kind 与 J/C handles；
- relation 的 denotation、scope 与 canonical replacement 在候选之外冻结；
- normalizer 对 Source 和 Replay 每一臂独立运行，不读取 peer observation 或当前 diff；
- J/C-owned operator、value dataflow、predicate、reducer、value recurrence、effect、post-state 与 output
  在 normalization 前后必须保持；
- 错误、放大的或无法唯一匹配的 π 只能导致 unsupported/failure，不能扩大接受集合；
- π 永远不进入 Replay。

因此 π 是 typed realization witness，不是 comparator permission。

## 8. 下一步重构顺序

在再次运行最后四核前，应按以下顺序推进：

1. 精简 Lift workspace：Source、Harness view、唯一 working candidate、bounded feedback 与按需查询入口
   保留；完整 schema/ops/preflight/example 改为查询，不再重复无条件挂载。
2. 提供只做机械引用维护的 typed artifact editor；LLM 仍决定所有 J/C/π 语义、relation kind 与 handles，
   固定工具不得补语义。
3. 修复四核已发现的 C attachment 与 boundary 参数化问题，特别是 loop/domain/tile ownership。
4. 实现上述最小 π grammar、dependent checks 与每前端固定 unary normalizer。
5. 完成 artifact-aware J/C hard projection；未知或歧义 atom fail closed。
6. fresh 运行四核。只有 Gate1 closure 后才进入三边 Gate2；Gate2 失败按三边定位，不能盲目重新 Lift。

完成前，不应继续通过增加 Agent预算、重复 Replay 或增加 branch 数碰运气。当前失败首先是系统输入、
artifact ownership 与 normalization 能力未闭合。
