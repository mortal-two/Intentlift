# Task 27 实验报告：手工 J/C/π 与 J-only CUDA/TileLang add2d

> 状态：`PAIR_CLOSED`（2026-08-01）
>
> 机器可读指标：`pair_metrics.json`

## 1. 这个实验只回答什么

研究问题被刻意限制为一个最小纵切：主代理直接从当前 Dataset Source 与编译观察作者化
`(J,C,π)`，完全不调用 Lift agent；随后两个 fresh、source-blind Codex 分别只消费
`J + MaskC(C) + U_f`，能否独立生成 TileLang/CUDA，并依次通过各自 Gate1 与三边 Gate2？

这不是 kernel 性能实验，也不是 J necessity、π ablation 或多算子泛化实验。add2d 的目的，是先把
J/C ownership、匿名 C binding 和 frontend observer 三个变量分开，避免把复杂算法能力混入协议判断。

## 2. 固定协议

```text
主代理读取 Source/compile view -> 手工 (J,C,π) -> TypeCheck
fresh Replay_f(J,MaskC(C);U_f) -> exact R
Gate1_TileLang = alpha/canonical TIR hard + PTX diagnostic
Gate1_CUDA     = semantic PTX hard + raw PTX diagnostic
Gate2          = Source–Replay + Source–J + Replay–J
```

- 两次正式成功 run 的 `lift_agent_calls` 均为0；
- Replay receipt 的 visible root 均恰为 `c,j,u_f`，`pi_visible_to_replay=false`；
- 两次均使用 fresh `gpt-5.6-sol/xhigh`，各1次 provider call，无旧 Replay、Source、Gate diff或 C value；
- π 为95 canonical bytes，两个前端 byte-identical；当前 normalizer 为 identity/fail-closed，未声称π收益。

## 3. 跨前端作者化结果

两份 artifact 不是 byte-identical full J。它们共享完全相等的数学部分：`intent`、`node_ids`、
`boundary`、`scopes`、`storage` 与 dimension relations；canonical 后为1484 bytes，SHA-256 为
`c9d1d21d62f30ba86e248edd08f633382edb6a4f3d2d063ee824b5b8b7e0ba67`。

真实前端差异保留在 ABI/C：

| 项目 | TileLang | CUDA |
|---|---:|---:|
| artifact 文件 | 4113 B | 4291 B |
| full J canonical | 1797 B | 1887 B |
| C canonical | 123 B | 121 B |
| runtime ABI | 3个 rank-2 tensor | 3个 rank-2 pointer + 1个 i32 逻辑维度 |
| C | `c0:int32=16` | `c0:i32=16` |
| π canonical | 95 B | 95 B |

因此论文应写“共享数学 J 核心、保留真实 frontend ABI delta”，不能写“两个前端得到完全相同的 J”。

## 4. 正式闭合结果

| 指标 | TileLang add2d | CUDA add2d |
|---|---:|---:|
| TypeCheck | PASS | PASS |
| Replay source | 13 LoC / 323 B | 10 LoC / 385 B |
| Replay episode | 158.57 s | 96.28 s |
| provider calls | 1 | 1 |
| construction tool calls | 3 | 2 |
| live checkpoints | 2 | 1 |
| Gate1 profile | TIR semantic structure v5 | PTX semantic effects v6 |
| Gate1 hard | PASS | PASS |
| raw/lower diagnostic | 0 differences | 3 raw-PTX differences |
| Gate2 Source–Replay | PASS，max error 0 | PASS，max error 0 |
| Gate2 Source–J | PASS，max error 0 | PASS，max error 0 |
| Gate2 Replay–J | PASS，max error 0 | PASS，max error 0 |

TileLang 的 Gate1/Gate2 wrapper wall time因旧 recorder 在 `gate2.json` 已落盘后才误读 suite `.status`
而使用文件 mtime近似；158.57 s 的 provider episode、Gate1/Gate2 verdict与三边误差均来自原始精确证据，
没有重跑或改写。CUDA 的 Gate1 100.94 s、Gate2 5.25 s为 runner 直接记录。

CUDA 的三个 diagnostic difference 是 raw instruction topology、raw opcode multiset 与 raw memory string；
semantic value/memory/address/control/coordinate/ABI/target 全部 hard PASS。这正好说明 CUDA 不应比较 raw
PTX 文本。TileLang 则在前端公开的 TIR boundary 上 structure/topology/op multiset 全部精确闭合，PTX 不
反向支配 TileLang Gate1。

## 5. 两个失败基线真正说明了什么

### 5.1 TileLang：MaskC 不是让 agent 猜常数

`run-001` 中 Replay 已生成正确加法，却不知道现有 worker 会按 exact `c*` 名称注入 Harness schedule，
于是经历15个 checkpoint、18次工具调用并最终重绑 `c0=128`，加入 carrier 与两个 guard；identity TIR
Gate1 正确拒绝。公开 host-global binding convention并禁止 slot shadow 后，fresh `run-002` 直接引用
`c0`，checkpoint降至2、工具调用降至3，episode 从555.32 s降至158.57 s（约3.50×）。

### 5.2 CUDA：domain 不是 access predicate

CUDA `run-001` 首次即编译通过，但把 logical domain 重复写成 `if (row < dim)`，再用 `blockDim.x` 构造
grid-stride loop。PTX 因此多出 scalar param load、比较、分支与额外 coordinate，Gate1 在
value/memory/control/coordinate 四类 hard 字段拒绝。明确“domain 是迭代集合；只有 J access predicate、
access-guard relation 或 value/effect control 才授权 guarded effect”后，fresh `run-002` 生成直接无 guard
CUDA，并完成 Gate1/Gate2。

这两轮表明主要瓶颈不是 LLM 无法理解 `add`，而是 J/C/target contract 中两个通用 ownership 语义没有
被明确表达。修复没有读取失败 Gate diff来生成答案，没有增加 add2d 模板，也没有放松 observer。

## 6. 工程量口径

- 本 pair 新增 kernel-specific production lowering：**0 LoC**；
- 两次基线暴露的通用协议修复：122 Python added / 33 deleted，net +89；
- experiment-only 手工作者化与 runner：282 + 258 = 540 physical LoC；它们用于可审计实验，不是
  production frontend integration cost；
- 更早的 J-only core selective sync 为2396 added / 1263 deleted（net +1133），包含共享 Replay/schema/
  provider重构，不能伪装成“增加一个 frontend只需多少行”；
- TileLang TIR observer 与 CUDA semantic PTX observer在本 Task 前已存在，不计为 Task27新增。

最稳健的论文表述不是一个过度压缩的单一 LoC，而是同时报告：共享 core delta、每 profile已有 observer、
本 pair 的 kernel-specific production LoC=0，以及 experiment-only authoring成本。

## 7. 可以与不可以写进论文的结论

当前可以写：

1. 对这一对真实 add2d，J-only Replay在两个不同 frontend/compiler boundary 上都能独立闭合；
2. π、Source、Gate与 C valuation不进入 Replay，仍能完成 Gate1与有限域三边Gate2；
3. 同一数学 J 核心可复用，但 ABI/C差异被诚实保留；
4. Gate1 能在 Gate2前拒绝“数值看似安全但 compiler structure 已改变”的额外 guard/carrier；
5. 多前端差异落在 `U_f + observer profile`，没有增加 per-kernel production recipe。

当前不能写：

- J 是必要或最小的；本轮未做 fresh Mask-J necessity实验；
- 结果已泛化到复杂 kernel；目前只有 add2d pair；
- π normalizer提高了闭合率；当前是 identity基线；
- Replay kernel性能等同或优于 Source；本轮 Gate2只测正确性；
- TileLang/CUDA full J byte-identical；它们的 ABI确实不同。

## 8. 最小下一步

若论文预算只允许再做一组，应做 `gather2d` pair，而不是扩到许多简单算子：它增加 data-dependent
indirect address，能直接检验 TileLang TIR 与 CUDA semantic PTX observer对合法 address realization 的
误拒率。随后只需对 add2d 做 wrong-op、wrong-index、wrong-effect 三个受控 mutation，量化 Gate1 在
Gate2前的拒绝价值。这样一张主表加一张小 mutation表，已经足以支撑5.2节，不必追求无限 kernel数。
