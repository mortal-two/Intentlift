# IntentIR：π 在 Gate1 归一化中的真实作用

## 1. 一句话结论

`π` 有真实作用，但作用范围很窄：它不向 Replay 提供实现信息，也不定义 J 的语义；它只在
Gate1 中，以 typed binding 方式选择固定的一元归一化规则，使 Source 与 Replay 对同一个 J access
采用不同地址构造形式时，能够比较其 J/C-owned 结构，而不是机械比较 `tt.addptr` 的写法。

当前实现真正启用的 π 规则只有 **access representation normalization**。其他已经能解析的
`program_axes`、iteration、control 和 materialization relation 仍然保持 hard，不能放宽 Gate1。

## 2. π 在完整流程中的位置

```text
Lift(Source) -> (J, C, π)

ReplayAgent(J, MaskC(C)) -> Replay       # 不读取 π

Source ──Compile/Observe──┐
                          ├─ 固定 Canon ── 一元 N^π ── exact compare ── Gate1
Replay ──Compile/Observe──┘

Gate2: Source–Replay && Source–J && Replay–J
```

因此 π：

- 不进入 Replay prompt、workspace 或工具；
- 不进入 RefEval(J)；
- 不进入 production backend；
- 不携带 Source recipe、Triton API、目标代码或允许差异列表；
- 只由 Gate1 的固定 normalizer 消费。

对 Source 和 Replay，`N^π` 分别独立执行。一次执行看不到另一臂，也看不到 Source–Replay diff。
最后仍由原有 comparator 精确比较两份归一化 fingerprint。

## 3. 固定 Canon 与 π-normalizer 的分工

Gate1 有两层容易混淆的处理。

### 3.1 固定 Canon：不读取 π

固定 TTIR observer 负责通用规范化，例如：

- SSA/局部打印名称消除；
- 交换律表达和比较极性规范化；
- 纯 representation 节点的保守收缩；
- 整数地址、索引多项式与 pointer form 规范化。

这些能力属于固定 observer，不应归功于 π。

### 3.2 `N^π`：决定能否对 access address representation 取商

当前 `normalize_ttir_for_gate1` 不根据 π 重写目标程序，也不生成新语义。它只检查 π 是否满足
固定规则的启用条件；满足时向 TTIR fingerprint 阶段提供：

```text
representation_anchor_exemptions = {"tt.addptr"}
```

这表示：`tt.addptr` 可以作为地址 realization 的中间表示被收缩，不再要求 Source 和 Replay 使用
完全相同数量、相同拓扑的 pointer-advance 节点。最终 base、规范化 offset、access kind、predicate、
value 和 effect 仍然可见。

代码入口：

- `python/intentir/verify/gate1/normalize.py:58-138`：一元 π normalizer；
- `python/intentir/verify/gate1/orchestrate.py:479-505`：Replay 臂独立归一化后 fingerprint；
- `python/intentir/verify/gate1/orchestrate.py:595-616`：Source 臂独立执行同一过程；
- `python/intentir/verify/gate1/ttir.py`：固定 Canon 和 address/pointer fingerprint。

## 4. π 具体保存什么

当前 compact witness 中，与已启用规则直接相关的字段是：

```json
{
  "schema": "intentir.pi.normalization-witness.v1",
  "program_axes": [],
  "access_representation_bindings": [
    {
      "relation": "address_recomputed",
      "access_ref": "a0"
    },
    {
      "relation": "pointer_advanced",
      "access_ref": "a1"
    }
  ]
}
```

其中：

- `access_ref` 必须引用 J 中已经声明的 access；
- relation 只能是关闭集合中的 `address_recomputed` 或 `pointer_advanced`；
- 每个 `(relation, access_ref)` 不能重复；
- 不允许 free text、wildcard、`allowed_diff`、tolerance 或额外语义字段。

这两种 relation 当前具有相同的 Gate1 权限：允许地址由重新计算 offset 或逐步推进 pointer 的方式
实现。它们都不允许改变这个 access 读写的逻辑位置和语义。

## 5. 一条 π 什么时候真正生效

当前代码只有同时满足以下条件才激活 access normalization：

1. J 至少声明一个 boundary access；
2. π 中合法 relation 所绑定的 `access_ref` 集合，必须**恰好等于** J 的全部 access 集合；
3. 当前单臂 TTIR 中 load/store/update 的分类数量，必须与 J 声明精确一致；
4. 所有 binding 已通过 parser 和 TypeCheck，确实指向 J-owned access。

形式上，关键条件是：

```text
bound_accesses == declared_J_accesses
```

并且：

```text
observed_memory_multiplicity == J_memory_multiplicity
```

如果少绑定一个 access，规则完全不激活，而不是只对剩余区域宽松比较。如果 memory multiplicity
不一致，normalizer 直接 fail closed。错误 anchor 由 TypeCheck 拒绝；未知 relation 或额外 payload
由 closed parser 拒绝。

## 6. 它消除什么，又绝不消除什么

### 可以消除

同一个 J access 的纯地址构造差异，例如：

```text
Source:  base -> addptr(row_offset) -> addptr(lane)
Replay:  base -> addptr(row_offset + lane)
```

或者 Source 重新计算完整 offset，而 Replay 逐次推进 pointer。只要固定 Canon 得到的 base、有效 offset、
access attachment 和 memory effect 闭合，`tt.addptr` 的中间拓扑不再形成假差异。

### 仍然保持 hard

- base pointer 或最终逻辑 offset 改变；
- load/store/update 的种类或数量改变；
- access predicate、mask polarity 或 fill 改变；
- 被读写的 value、tensor、effect target 或 effect order 改变；
- operator、operand dataflow、reducer、recurrence、post-state 或 output 改变；
- loop/carrier/terminal control；
- 当前尚无 sound consumer 的 program-axis、iteration、control、materialization relation；
- 任何 unknown 或无法唯一绑定的 observation atom。

所以 π 不是“允许 Source 和 Replay 不同”的白名单。它只授权一个预先冻结、局部且一元的
address-representation quotient。

## 7. 为什么这不构成泄漏或自我审批

π 是 Lift 生成的 witness，但是否放行不由 Lift 决定：

1. relation language 和语义由固定代码预先定义；
2. Replay 完全看不到 π，因此不能照着 Source realization 写目标代码；
3. normalizer 每次只看一臂，不能读取 pairwise diff 后量身定制规则；
4. TypeCheck 只接受 J-owned typed anchor；
5. all-access coverage 与 exact memory multiplicity 防止用局部 binding 吞掉未知 access；
6. J/C-owned 语义类别始终 hard；
7. 最终还有三边 Gate2 独立检查行为。

LLM 能做的是选择一个预定义 relation 并给出 typed witness；它不能改变 relation 的 denotation，
也不能扩张 Gate1 的接受集合。

## 8. 24-Kernel 和大规模变体证据

24-Kernel 受控消融中：

- correct π：24/24 通过；
- no π：0/24 通过；
- narrow/wrong π：0/24 通过；
- semantic mutant + correct π：0/24 通过；
- relation region 内的 semantic mutant + correct π：0/24 通过。

其中 23 个 Kernel 的主 relation family 是 access representation。Liger RoPE 的 `program_axes`
结果来自受控历史实验；当前活动 normalizer 已将 program-axis relation 收紧为 hard，因此不能把该
一行继续表述成当前已发布的 program-axis normalization 能力。

更大的 π variant bank 共真实裁决 2,785 个变体：

| 变体 | 结果 |
|---|---:|
| correct | 162/162 Gate1 pass |
| equivalent binding order | 161/161 pass |
| equivalent closed relation encoding | 161/161 pass |
| no π | 162/162 Gate1 reject |
| delete one binding | 888/888 Gate1 reject |
| wrong anchor | 605/605 TypeCheck reject |
| unknown relation kind | 161/161 parser reject |
| over-broad payload | 161/161 parser reject |
| semantic smuggling under correct π | 162/162 Gate1 reject |

这些结果同时证明两点：π 的存在确实改变合法 realization variant 的 Gate1 结果；但合法 π 不能
掩盖已经注入的语义错误。

当前证据也保留一个真实边界：158 个 observable nonzero-address controls 中有 5 个穿过了现有
address quotient。它们是 observer blind spot，不能删除或伪装成正确；这说明当前规则有作用但不完备，
并说明三边 Gate2 仍然必要。

证据位置：

- `papers/intentir-ppopp27/experiments/raw/frontend_scrc_24/aggregate/pi-normalization-ablation.csv`；
- `papers/intentir-ppopp27/experiments/raw/frontend_scrc_24/pi_variant_bank/results.json`。

## 9. “人工压缩 π”到底意味着什么

24-Kernel 实验中的人工处理只做了 witness 最小化与受控变体构造：

- 从 accepted J 读取已有 access IDs；
- 只保留当前 normalizer 能消费的 closed relation 与 typed anchor；
- 构造 correct/no/delete/wrong/over-broad 等消融臂；
- 对固定 Source/Replay observation 运行真实 Gate1。

它没有把 Source 地址公式、目标代码、Replay 模板或 pairwise diff 写入 π，也没有替 Replay 生成程序。
因此这可以用于回答“π 是否必要、是否过宽”的机制消融，但不能冒充自动 Lift 生成 π 的成功率。

## 10. 最终可以声称什么

准确、可防守的结论是：

> π 是 Gate1 的 certification-only typed normalization witness。当前实现中，它以完整 J-access
> binding 激活固定的一元 address-representation quotient，使合法的 pointer construction 差异闭合；
> closed schema、J-owned anchor、exact memory multiplicity 与 hard semantic categories 限制其权限。
> 受控消融证明 π 对合法 realization closure 是必要的，同时不能掩盖 J-owned semantic mutation。

不能声称：

- π 已经归一化所有 program/loop/control realization；
- π 正确性或完整性已被形式化证明；
- Gate1 或 Gate2 单独证明全输入域语义等价；
- 当前 address quotient 没有 blind spot。
