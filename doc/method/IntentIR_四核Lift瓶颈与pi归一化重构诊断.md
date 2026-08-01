# IntentIR 四核 Lift 瓶颈与 π 归一化重构诊断

> 性质：当前方法诊断，不是证书、实验结论或修改流水账。
> 范围：原最后四个 Kernel 的纵向诊断；RoPE 已闭合，当前剩余三个 Kernel，以及 one-col 辅助样本暴露的共性问题。

## 1. 结论

原最后四项是：

1. `high_performance:liger_llama4_rope`；
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`；
3. `high_performance:flash_attn.flash_attn_triton._fwd_kernel`；
4. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`。

其中 Llama4 RoPE 已由自动 Lift 路径完成 Gate1 与 Gate2 三边闭合；当前剩余
SplitK forward、FlashAttention forward 和 FlashAttention full backward 三项。
TypeCheck 通过只说明 JSON 与 dependent contract 机械合法，不构成进展，也不能证明 J 正确。

RoPE 闭合前的主要矛盾不是 Source 编译慢，也不是 compact π 太大，而是：

- Lift Agent 仍手工作者化一份 30--46 KiB、跨字段强依赖的完整 `(J,C,π)`；
- 输入工作区重复提供完整 schema、contract、IntentOps、preflight、示例与工具说明；
- 工作轨迹和历史工具输出常比 candidate 大一个数量级；
- 当时 π 只表达 program-axis binding，而固定 normalizer 实际是 identity；
- Gate1 因而继续比较物理 loop、carrier、address、control 与 program topology；
- 即便 Source 和 Replay 都实现同一个正确 J，不同合法 TTIR realization 也会被拒绝。

RoPE 已完成轻量 workspace、persistent owner、精确 TTIR、typed diagnostics 与
`address_recomputed` fixed unary consumer，证明这条改造有效。剩余三核仍需逐核完成两件事：

1. source-aware Lift 必须正确作者化 J/C，特别是 logical loop 与 C tile attachment；
2. π 必须恢复少量、closed、typed realization bindings，让固定 unary normalizer
   能把不同物理实现归一到同一个 J/C-owned structure。

不应恢复旧的 region/instance/route 大图，也不能靠放宽 Gate 或 pairwise diff 白名单闭合。

## 2. Lift 时间花在哪里

### 2.1 四核观测

| Kernel | 三个首次 Lift 的量级 | 后续 relift 量级 | 代表 artifact 大小 | 当前结果 |
|---|---:|---:|---:|---|
| Llama4 RoPE | 654--706 s | 112--269 s | 29--40 KiB | Gate1 与 Gate2 三边已闭合 |
| FlashAttention forward | 872--1516 s | 35--174 s | 约 35 KiB | Gate1 0 closure |
| FlashAttention full backward | 1438--1785 s | 20--246 s | 约 46 KiB | Gate1 0 closure；另有基础设施失败 |
| SplitK forward | 646--900 s | 80--832 s | 约 46 KiB | Gate1 0 closure |

代表性 Source TTIR 编译约为数秒，CandidateCheck 和普通工具调用通常是毫秒到百毫秒量级。
十几分钟至半小时的首次 Lift 主要消耗在模型回合、上下文处理、完整 artifact 生成和反复修复，
不是编译器执行。

### 2.2 闭合前机械修复占用了大量回合

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

### 2.3 闭合前输入与历史上下文过重

原 Lift 工作区通常同时挂载：

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

### 2.4 compact π 不是当时耗时来源

当时的空 π 只有 schema 与空 `program_axes`，体积不足百字节。每条 axis row 也只是几十字节。
它不会解释十几分钟的 Lift 时间。

真正的问题相反：π 被压缩到几乎没有消费能力，导致本来应由 normalization 处理的物理差异，
最终全部回到 Replay 猜测和 Gate1 hard comparison。RoPE 现已补上保守的 address
representation consumer；iteration、materialization、program-linearization 与 terminal-control
仍必须由剩余 Kernel 的真实 residual 驱动实现，不能预先放宽。

## 3. 四个 J/C/π 的静态判断

以下判断只基于 Source、adapter 与 artifact 的直接对应。没有 Gate2 的对象不能称为正确，
因此使用“核心图合理”“存在明确缺口”“无法判定”三种表述。

### 3.1 Llama4 RoPE：已闭合纵切

J 已表达：

- Q/K 两条 rotary 路径；
- 前后半维 gather/scatter；
- cos/sin 数据流；
- f16/f32 casts；
- Q/K inout read/write effects；
- 两个输出及 exact result relation。

最终自动闭合证明：核心 J、C attachment 与结果关系在本次执行域内成立。可复用闭环为：

1. 用精确 Source TTIR 区分 J-owned 数值语义与纯 realization；
2. 只局部修正 J 中的 FMA、cast、sub 与 static affine inout store 表达，不重写已正确部分；
3. J 或 MaskC 改变后执行 fresh source-blind Replay；
4. Gate1 residual 先分流：额外 program axis、错误 cast/dataflow 属于 Replay fidelity，不能由 π 掩盖；
   唯一可证明的 address recomputation 才由直接绑定 J/C 的 `address_recomputed` relation 归一；
5. Gate1 通过后运行 Source–Replay、Source–J、Replay–J 三边 Gate2；
6. Source–J 暴露的 RefJ store/post-state 缺口作为 J-only consumer bug 修复，禁止回灌 Source、Replay、π 或 C 值；
7. 最后让轻量、persistent 的同一 Lift owner 从 Source/TTIR 与局部 checker 反馈重新作者化完整
   `(J,C,π)`，fresh Replay 后复现全部闭合，排除手写 oracle 依赖。

这次闭环没有依靠 per-Kernel matcher、allowed diff、Gate 放宽或 π→Replay 信息通道。Gate1
实际只启用了 `jc_access_address_representation`；Source/Replay 分别归一 10/5 个
`tt.addptr` occurrence，所有 J/C-owned hard categories 保持 exact。最终自动证据为
`/tmp/intentir-rope-auto-certified-20260801-final/liger_llama4_rope/gate2-recheck-001`，
Q/K 三边最大误差均为 0；Lift/fresh Replay 父证据为
`/tmp/intentir-rope-local-patch-20260801/liger_llama4_rope/run-001/attempt-02`。

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

- `c0=BLOCK_HEADDIM` 绑到 feature domain `d3/x2`，`c4=BLOCK_M` 绑到 query
  domain `d0/x0`，两者基本正确；
- `c5=BLOCK_N` 虽已指向 key-axis loop `n3/x3`，却被写成
  `reduction_tile_extent`；Source 实际用它作为 serial key-tile loop step，应改为
  `tile_extent`，并由 `n3` 的 trip domain `d4/x3` 给出逻辑轴；
- `EVEN_M`、`EVEN_N`、`EVEN_HEADDIM` 三个 specialization slot 尚未绑定 J
  subject，应分别附着到 `d0/x0`、`d4/x3`、`d3/x2`；J 仍必须自己保留
  完整 logical predicate/fill，不得依靠这三个 C 值恢复语义；
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

J 已表达主要 contractions、casts、DQ update 与 DK/DV outputs，但最新可读候选中
没有 structured loop/trip，因而尚未能确认 inner query-block 上的 DK/DV carried-state
recurrence 已由 J 拥有。outer key-block 枚举可以是 domain/program realization，但 inner
query recurrence 不能只靠 C/π 补成。

当前明确缺口：

- `c0=BLOCK_HEADDIM`、`c5=BLOCK_M`、`c6=BLOCK_N` 的值与 domain attachment
  数值上合理；其中 `c5` 必须在 J 补出 inner semantic loop 后改绑到该 loop，
  `c6` 可以继续附着到 key-output domain，因为 outer key-block 枚举可以只是该
  domain 的 realization，不必伪造成第二条 semantic recurrence；
- `c1=SEQUENCE_PARALLEL` 作为 implementation-selection C 的分类是合理的，其
  `d4/x1` attachment 也可保留；真正缺口是 closed consumer 尚不能用该 slot
  区分 key domain 的 serial outer iteration 与 program-axis distribution。这应由 π
  iteration/program binding 引用 `c1`，而不是让 C 反向引用 π 中的物理 mapping；
- `EVEN_M/N/HEADDIM` 可作为附着到对应 domain 的 specialization C，但 causal、
  bias interpretation、logical guard、atomic/inout effect 必须由 J 的 predicate/access/effect
  闭合，不能通过新增 C slot 隐藏；
- π 没有可消费的 outer program/iteration、inner recurrence realization、pointer-only
  carrier 与 terminal-control bindings；
- 没有 relation 将 outer tiled iteration、inner recurrence、pointer-only carriers 与 terminal control
  锚定到 J loop/control；
- Gate1 仍比较 Source helper inlining、physical loops、unused/used ABI、address and effect region topology；
- Replay 本身还需要数分钟，并多次出现 compile/API 修复。

静态判断：J 的局部数值图可能正确，但 inner recurrence 仍是 J-owned 明确缺口；
C 则同时存在“已有 tile/mode 值基本正确”和“inner-loop attachment 与
parallelization consumer 未闭合”两个事实。
本核必须先修 J，之后重绑 C 并 fresh Replay，不是 π-only 修复。

### 3.4 xFormers SplitK forward

Source 的核心是：

- 三个 program axes，其中一个选择 split；
- split chunk low/high；
- tiled K/V iteration；
- running max、normalizer 与 accumulator 三个 numerical states；
- split output 与可选 LSE；
- 多种 specialization 下的 early exit、mask、paged/quantized branches。

历史候选已保留三状态 online recurrence、QK/exp/sum/PV 与 split outputs，所以不能
说模型完全没理解算法。最新运行在形成可审计 artifact 前即是 infrastructure failure，
因而下面的 C 裁决以 Source 和最近可读候选为依据，不把该次失败写成 C 证据。

但当前 artifact 仍有明显风险：

- `BLOCK_M` 是 query-domain tile，应作为 C `tile_extent` 附着到 J query domain；
- `BLOCK_N` 是 split chunk 内部 key iteration 的 tile，应作为 C `tile_extent` 附着到
  J key loop/domain；
- `split_k` 同时决定 program axis 和可观察 output split dimension，因而必须参数化到 J，
  不能作为 C；
- `BLOCK_N_PER_SPLIT` 决定每个可观察 partial-output slice 对应的 key chunk
  边界；对当前将 partial outputs 直接视为 Kernel outputs 的 J，它也必须由 J 的
  split-partition relation 拥有，不能仅当作纯 tile C；
- padded query capacity 只能在不改变 logical output domain 时作 representation capacity；如果它
  出现在可观察 output shape，J 必须同时声明 logical extent 与 capacity/valid-domain 关系；
- `num_warps/num_stages` 若只是 adapter/compiler launch context，应留在固定
  frontend/Harness context；不能因 adapter 中有值就凭空生成 C slot；
- π 没有 split-axis、mixed-radix、iteration、early-exit/control binding；
- 当前 artifact 对复杂 specialization 的 boundary facts 只覆盖实际固定分支，尚不能证明所有
  representation-only constexpr 与语义参数的所有权划分正确；
- 历史 Lift 反复出现 unknown logical refs、functional axis mismatch、C type/axis 与 runtime carrier
  ownership 错误。

静态判断：online recurrence 核心合理，但 split output/partition 是 J-owned boundary，
`BLOCK_M/BLOCK_N` 才是纯 C tile；这一分界未闭合前，它最不能被称为
“J 已正确，只差 normalizer”。

### 3.5 四核 C 收尾裁决

C 的修复不是增加 source recipe，而是把纯 implementation-selection coordinate 附着到它
所作用的稳定 J subject/axis。统一判断为：

| Kernel | C 现状 | 最小修复 | 是否改 J | Replay 处理 |
|---|---|---|---|---|
| Llama4 RoPE | `c0=BLOCK_SIZE -> d0/axis 3` 正确 | 已完成；C 未扩张 | 已局部修正 FMA/cast/static affine store 表达 | 已 fresh Replay 并完成三边 Gate2 |
| Flash forward | feature/query tile 正确；key tile role 错；三个 EVEN slot 无 attachment | `c5` 改为 loop `tile_extent`；EVEN slots 附着 query/key/feature domains | 暂不改 recurrence J | C schema/attachment 改变后 fresh |
| Flash full backward | tile/mode 值基本对；没有可附着的 inner loop，也没有 `SEQUENCE_PARALLEL` consumer | 先补 J inner recurrence anchor，再将 query tile 绑到该 loop；π 以 `c1+d4` 区分 serial/program realization | **需要** | fresh Lift + fresh Replay |
| SplitK forward | 最新运行无可审 artifact；历史归类混合 split 语义与 tile | 只将 `BLOCK_M/BLOCK_N` 作 C；`split_k` 与 observable chunk partition 收回 J | **需要 boundary 参数化** | fresh Lift + fresh Replay |

为避免再次出现 one-col 式错绑，Lift workspace 只需增加一个按需查询的非证明视图：

```text
anonymous coordinate q
  -> raw SSA uses: loop-step / range-width / mask-bound / address-stride / launch-only

current J
  -> valid subject_ref and AxisRefs(subject_ref)
```

该视图不声称 q 属于 J/C/π，也不自动选择 subject。LLM 仍负责分类和绑定；
fixed checker 只检查 handle、axis、role/type 与 dependent references 是否机械合法。正确性仍由
fresh Replay、Gate1 与三边 Gate2 闭合，不新增 C composer 或自动所有权分类器。

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

## 5. 诊断起点：π 被过度压缩

诊断开始时 π 只包含 `program_axes`，而固定 normalizer 返回原 observation：

```text
N^π(O) = O
```

所以该起点下的 π：

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

## 8. 剩余三核执行顺序

RoPE 已证明轻量 persistent Lift、精确 TTIR、局部修 J、fresh Replay、Gate1 residual 分流、
π-only normalization 与 J-only RefJ consumer 修复可以形成完整闭环。剩余 Kernel 不再并行散开，
而是逐核复用同一纵向流程：

1. FlashAttention forward：先修 key tile 与 EVEN attachments，再判断 online recurrence J、Replay fidelity、
   iteration/materialization normalization 和 RefJ consumer；完成 Gate1 与三边 Gate2 后再进入下一核。
2. FlashAttention full backward：先补 J-owned inner query recurrence，再重绑 C，fresh Replay；不得用 π
   替代 recurrence、conditional post-state 或多输出语义。
3. xFormers SplitK forward：先闭合 split output/partition 的 J-owned boundary 参数化，只把
   `BLOCK_M/BLOCK_N` 留在 C，再处理 mixed-radix program、iteration 与 terminal control realization。

每核都保留 best exact candidate，但 J/MaskC 改变必须 fresh Replay；π-only 或 normalizer 修复可以复用
Replay 源码并重算 Gate1。Gate1 通过后才运行三边 Gate2；Gate2 按三边归因，禁止把所有失败盲目送回 Lift。
未知或歧义 relation 一律 fail closed，不通过增加预算、branch 或重复 Replay 碰运气。
