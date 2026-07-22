# 结构化循环(带状态携带)

## 为什么 IR 必须有循环:FlashAttention

把 FlashAttention 用整张量视角写出来,问题一眼可见:

```
m = -inf;  l = 0;  O = 0                    # 累加器(携带状态)
for Kj in K 的每一块:                        # ← 带状态的循环
    S      = Q @ Kj^T                        # contraction
    m_new  = max(m, rowmax(S))               # 更新 running max
    P      = exp(S - m_new)                   # map
    l      = l*exp(m-m_new) + rowsum(P)       # 这一轮依赖上一轮 ← 递推
    O      = O*exp(m-m_new) + P @ Vj          # 这一轮依赖上一轮 ← 递推
    m      = m_new
O = O / l
```

FlashAttention = 两个 contraction + 一个"边扫 K、边更新累加器 (m, l, O)"的 softmax。那个"边扫边更新"就是**带状态的循环**。没有循环构造,这个算子写不出来。所以 IR 必须有一等公民的结构化循环。

## `loop` 构造

- 形态:`scf.for` 风格的结构化循环,带 **iter_args / carry**(循环携带的状态),循环体是一个 region。
- 是结构化的(有明确的归纳变量 / 携带状态 / region 体),**不是任意 goto**。
- 默认 trip count 静态已知(覆盖 scan / FlashAttention);放宽成"条件由数据决定"即覆盖数据依赖迭代(`scf.while` 风格)。**一个构造通吃。**

### 已实现(2026-06-25):`intent.loop` + `intent.yield`

C++ TableGen 算子已落地(`include/Intent/IntentOps.td`,verifier 见
`lib/Intent/IntentOps.cpp`):`intent.loop scan(K,V) init(m,l,Oacc) {scan_axes,
tile_hint} -> ... do { ^bb0(Kj, Vj, m, l,Oacc): ... intent.yield ... }`。

这里 `tile_hint` 是现有 dialect/lowering 兼容 attribute，不是 canonical J 的 semantic
trip 真值。新 SCRC J 的 loop 只持久化 semantic axis 与 carry；tile/unroll/pipeline
必须由 C 或 target schedule 提供。
- **scan 输入**:lowering 可按 C 中的 tile 沿 `scan_axes` 切片 K/V；改变 tile 不得改变
  semantic recurrence。
- **carry(init/yield)**:循环携带状态;result 类型 == init 类型(carry 类型循环不变,verifier 强制)。
- **body block args** = scan 切片(按序)++ 当前 carry。
- **lowering**:`intent.loop` → `scf.for`(trip)+ tensor `iter_args` + scan→
  `tensor.extract_slice` + body→linalg(`LoopLowering`)。这是张量级自然下降。
- **前端**:`python/intentir/frontend/emit_mlir.py` 从 `flashattention_fwd.json`
  发射该 op(及可执行 body 函数)。flash 递推由发射的 body 在 5090D 上 host 驱动
  trip 循环执行,与解释器一致(见 [../lowering/index.md](../lowering/index.md))。
- **残项**:从 `intent.loop` 直接产出 *fused* 单 kernel(query-outer/per-query-carry
  schedule)需要 interchange + carry privatization;今天 fused 形态由
  `python/intentir/backend/cuda/flash.py` 发射。

## 三类循环的处理

| 循环种类 | 归属 | 处理 |
|---|---|---|
| **展开 / tiling 循环**(静态次数) | **Layer C** schedule | LLM 把循环卷回成整张量算子,循环消失,tile/unroll 因子进 C 层 hint |
| **递推**(loop-carried,次数静态) | **Layer A** | 用 `scan` 或带 carry 的 `loop` 表达(online-softmax / prefix-sum / RNN step) |
| **数据依赖迭代**(while / 收敛 / 动态 trip count) | Layer A,**门控逃生舱** | 同一 `loop` 构造,条件由数据决定;属能力边界前沿,见 [../capability/boundaries.md](../capability/boundaries.md) |

## 循环展开(unroll):提取鲁棒性问题

- unroll 是**纯 schedule**,语义上是 no-op。
- 你输入清单里的手写高性能 CUDA **大量手工 unroll / strip-mine**,所以"扛得住 unroll"本质是**提取鲁棒性**要求。
- LLM 的活:把展开 / strip-mine 的重复体**卷回去(re-roll)**,认出"这其实是 N 次同构计算"。识别重复模式正是 LLM 强项,脆弱的规则匹配做不好。
- **re-roll 必须留在 LLM 提取阶段,不能做成下游 pass**——否则把"原子化分解交给 pass"的禁忌又带回来了(见 [../project/constitution.md](../project/constitution.md))。

## 因式分解(再次强调)

FlashAttention 的 K 循环**同时**是 tiling(C)和 online-softmax 递推(A scan)。必须把这一个源循环拆成 `C 层 tiling hint` + `A 层递推语义`,不能简单归类进单一桶。详见 [layers-abc.md](layers-abc.md)。
