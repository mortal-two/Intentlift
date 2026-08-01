# Final Task：剩余三核逐核 SCRC 闭合

> 状态：`IN_PROGRESS`
> 当前 Kernel：`high_performance:flash_attn.flash_attn_triton._fwd_kernel`
> 方法依据：`doc/method/IntentIR_四核Lift瓶颈与pi归一化重构诊断.md`

## 1. 唯一目标

按以下顺序逐个完成剩余三个 Kernel 的真实 Gate1 与 Gate2 三边闭合；一个 Kernel 完整闭合后，
才进入下一个：

1. `high_performance:flash_attn.flash_attn_triton._fwd_kernel`
2. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`
3. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`

TypeCheck、schema、工具、prompt、文档或中间 artifact 不计进展。

## 2. 不可改变的合同

```text
Lift(Source) -> (J,C,π)
ReplayAgent(J,MaskC(C);U_f) -> R
Gate1: N^(J,C,π)(Observe(Source,c0)) == N^(J,C,π)(Observe(R,c0))
Gate2: Source-Replay && Source-J && Replay-J
```

- Replay 不读 Source、π、C values、Gate diff 或历史 Replay；
- raw observer 不读 π；normalizer 对 Source/Replay 两臂独立执行，不读 peer 或 pairwise diff；
- π 只能选择冻结的 typed relation 并绑定 J/C handles；unknown、ambiguous 或无法证明时 fail closed；
- π 不得消除 J/C-owned operator、value dataflow、predicate、reducer、recurrence、effect、post-state 或 output；
- 禁止 per-Kernel matcher、allowed diff、free-text equivalence、Source recipe、composer 或 Gate 放宽；
- J 或 MaskC 改变必须 fresh Replay；π-only/normalizer 修复可复用 Replay 源码但必须重算 Gate1；
- Gate1 通过后才运行三边 Gate2；Gate2 失败按三边归因，不盲目 relift。

## 3. 逐核闭环方法

每个 Kernel 都执行同一个纵向闭环：

1. 从匿名 Source、Harness/compiler view 与精确 TTIR 核对现有 best `(J,C,π)`；
2. 把问题严格分为 J-owned semantic、C ownership/attachment、Replay fidelity、π-normalizable realization、
   observer/RefJ consumer 五类；
3. 只修改所属层，保留其余已正确 artifact；
4. fresh source-blind Replay 后运行 Gate1；
5. 对 Gate1 residual 再分流，禁止用 π 掩盖 J/Replay 错误；
6. Gate1 closure 后运行 Source–Replay、Source–J、Replay–J；
7. 最终候选必须由轻量 persistent Lift owner 自己作者化并复现闭合，手工候选只作诊断 oracle。

## 4. 当前 Kernel：FlashAttention forward

已知但尚待真实证据裁决：

- online-softmax 的 key-axis loop 与 running max/normalizer/accumulator 核心图可能合理；
- `BLOCK_HEADDIM` 与 `BLOCK_M` attachment 基本合理；
- `BLOCK_N` 应是 key-loop `tile_extent`，不是 `reduction_tile_extent`；
- `EVEN_M/N/HEADDIM` 只可作为 specialization C 附着对应 logical domain，J 自己保留 predicate/fill；
- tiled iteration、pointer-only carrier、scratch materialization 只有在 fixed unary consumer 唯一证明时才能归一；
- RefJ 无法消费非 direct-J-operand load 时，先区分 J access 不足与 J-only consumer 缺口。

### 当前动作

- [ ] 定位最新 best artifact、Replay 与 raw Gate1 residual；
- [ ] 逐项核对 J、C、π 与精确 TTIR，不把 TypeCheck 当作进展；
- [ ] 修复真实所属层并 fresh Replay；
- [ ] Gate1 closure；
- [ ] Source–Replay、Source–J、Replay–J 全部通过；
- [ ] 自动 Lift 路径复现闭合，并记录最终 evidence root。

## 5. 后续 Kernel 的硬边界

- Flash backward：inner query-block DK/DV recurrence、conditional DQ post-state 与多输出属于 J；
  outer key-block program/iteration realization 才可能属于 C/π。
- SplitK forward：split output/partition 与 observable chunk boundary 属于 J；只有纯
  `BLOCK_M/BLOCK_N` tile 属于 C；mixed-radix program 与 terminal control 需 fixed consumer。

## 6. 当前真实证据

| Kernel | Gate1 | Source–Replay | Source–J | Replay–J | evidence |
|---|---|---|---|---|---|
| FlashAttention forward | pending | pending | pending | pending | pending |
| FlashAttention backward | pending | pending | pending | pending | pending |
| xFormers SplitK forward | pending | pending | pending | pending | pending |

## 7. 完成条件

- 三个 Kernel 均有可审计的 Gate1 pass 与 Gate2 三边 pass artifact；
- 最终 `(J,C,π)` 满足上述信息隔离与 ownership 合同；
- 没有 Gate 放宽、Kernel 特化、Source/π 泄漏、假闭合或以 TypeCheck 代替进展；
- authoritative evidence 写回本任务，代码与文档提交，工作区干净。
