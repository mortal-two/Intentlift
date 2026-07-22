# Task08 v5 Canary：合法 π 到达 Gate1 后暴露 Address Realization 缺口

## 结果边界

唯一 protocol-v5 Swiglu canary 完整跨过了 Task08 要修复的边界：

```text
Lift             completed, attempts=1
normal Replay    completed, attempts=1
strict Gate1     no_closure
masked-J         calls=0
```

因此 graph foreign-key 可生成性问题已经关闭；本次不是 Lift schema rejection。normal 未闭合后
立即停止，符合预算，没有重试或扩散其余 27 case。

## 冻结 provenance

- v5 case receipt SHA：`5a286725e22f1fadc83771eb460a26e823e0c9675606bde311412563f2921b71`
- Lift response SHA：`c22ce3c3ef585ae8d635ea8ef8bfee74a13ec8d6fa9c65ace93abdbd946cffac`
- canonical artifact digest：`a3ec791ee55ffbac51aac2dc68fe06d7cb45a85a3a2b497e2bbfb77b931ff68e`
- public artifact file SHA：`6f54812ce0623c113a4cfd80ae6397c0d7a36dc90687a2ffa07aa78829ac749f`
- Replay code SHA：`95951e292cc76b4281e97b6999ef3b616ae2b660bb3107b72327b043ebdd9c4f`
- Gate1 canonical receipt SHA：`e77401db2db33db8b4084f0cc8aa3eb5380d896c89fcd321eeb32c89ef4e3583`
- Lift usage：input `10038`，output `18251`（reasoning `15743`），total `28289`
- Replay usage：input `3589`，output `1150`（reasoning `947`），total `4739`

## J/C/interface 没有回退

- J inputs/output 都是 `[128,1024]`；
- interface 顺序是 `p0,p1,p2,p3,p4,c0`；
- non-C `p4` 没有进入 C；
- C 只有 `c0=1024` 与 `c1=4`；
- Gate1 的 interface、reduction、predicate/effect、schedule-only components 全部通过。

所以失败不再是 n_cols 所有权、logical rank、mask 或调用绑定问题。

## 合法 π 已经表达了什么

v5 π 有 21 instances、31 routes、5 order edges、6 sharing relations。关键子图为：

```text
i1 anchor=q0 (program instance) ─┬─> i4 anchor=a0 realize ─> i13 transfer_direct
i2 anchor=p3 (runtime stride)  ──┤
i3 anchor=p0 (input pointer)   ──┘

i1/i2 同时 fanout 到 i6(a1) 与 i8(a2)
i2 sharing consumers = [i4,i6,i8]
```

这满足 Task07 设定的升级门槛：non-C/runtime dependency、三路共享、base-stage 到 transfer-stage
的多级 chain 都存在，并原样进入 source-blind Replay packet。π 中没有 source name、formula、dtype、
operator 或 predicate 泄漏。

## Replay 做了什么

Replay 确实消费了 p3，但选择了 flat vector-offset topology：

```python
v5 = tl.program_id(0)
v6 = tl.arange(0, c0)
offsets = v5 * p3 + v6
x = tl.load(p0 + offsets, ...)
gate = tl.load(p1 + offsets, ...)
tl.store(p2 + offsets, ...)
```

编译后是：

```text
i32 program_id * i32 stride
  -> splat scalar offset
  -> vector add with lane
  -> splat each raw pointer
  -> 3 vector tt.addptr
```

source 则明确先建立 program-uniform pointer bases，再加入 lane offsets：

```text
program_id i32 -> extsi i64
stride i32     -> extsi i64
  -> scalar i64 multiply shared once
  -> 3 scalar tt.addptr on p0/p1/p2
  -> splat the 3 adjusted pointers
  -> 3 vector tt.addptr with lane offsets
```

两者数值等价且都使用 p3，但 strict Gate1 正确地区分。

## Gate1 精确差异

- score：`0.6407238924050632`；
- op multiset：Replay 多 1 `arith.addi`、多 1 `tt.splat`，少 2 `arith.extsi`、少 3
  `tt.addptr`；
- dataflow differences：77；
- interface/reduction/predicate-effect/schedule-only：全部通过；
- normal 不闭合，所以 masked-J 为 0。

与 v3 的 score `0.426893...`、81 dataflow differences 和 predicate/effect failure 相比，v5 已
显著收窄到同一个 address staging 差异，但仍不能 hard-close。

## 为什么现有 `realize` 不够

当前 π 能说“存在一个锚定 access 的 realization event，并依赖 pointer/program/stride”，却不能
区分这个 event 是：

1. program-uniform scalar pointer base；还是
2. lane-varying vector address；或者
3. 把两者合并成一个 flat vector offset。

Replay 没有违反任何可判定的 mechanism tag，因为三者目前都只能写成 `realize`。要求模型从
route pattern 猜出“这里必须是 pointer base”会把未定义的图形编码当规则卡使用，不符合简洁、
多来源、LLM-direct 的设计。

## 下一能力上限

现在达到 Task07 预先冻结的 escalation gate。下一任务只允许给现有 `mechanism` enum 增加两个
source-neutral tags：

```text
address_base = program-uniform pointer/base realization stage
address_lane = lane-varying final address realization stage
```

它们只给已有 instance 分类，不增加 formula、offset value、integer width、cast op、dtype、shape、
frontend API 或 kernel family。pointer-width cast 应由公共 frontend 在实现 `address_base` 时自然
产生，而不是泄漏到 π。若这两个 tag 仍不足，必须由新的合法 artifact + Replay + strict Gate1
反例再次证明，不能预先扩表。
