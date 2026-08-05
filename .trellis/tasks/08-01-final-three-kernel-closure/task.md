# Final Task：剩余 Kernel 逐核 SCRC 闭合

> 状态：`IN_PROGRESS`
> 当前 Kernel：`high_performance:flash_attn.flash_attn_triton._bwd_kernel`
> 方法依据：`doc/method/IntentIR_J-only_Replay与JC结构Gate1_2026-07-31.md`、
> `doc/method/IntentIR_Gate1归一化现状_证据归集与论文优先执行计划_2026-08-01.md`

## 1. 唯一目标

逐个完成剩余 Kernel 的真实 Gate1 与 Gate2 三边闭合；一个 Kernel 完整闭合后才进入下一个。

**已闭合（2026-08-01）**：

- ✅ `high_performance:liger_llama4_rope`
- ✅ `high_performance:flash_attn.flash_attn_triton._fwd_kernel`

**剩余 4 项未闭合**（按当前证据重算；此前"剩余三核"的表述漏了 Flash linear）：

1. `high_performance:flash_attn.flash_attn_triton._bwd_kernel`
2. `high_performance:xformers.ops.fmha._triton.splitk_kernels._fwd_kernel_splitK`
3. `high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block`
4. `high_performance:flash_attn.ops.triton.linear.kernel_fwd`

因此当前独立覆盖为 **24/28**（按 `/tmp` 证据文件重算，非任务卡自述）。

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

## 4. 当前 Kernel：FlashAttention backward

FlashAttention forward 已于 2026-08-01 17:39 闭合（见 §6）。其可复用闭环：
用精确 Source TTIR 分流 J-owned 语义与纯 realization → 只修所属层 → fresh source-blind Replay
→ Gate1 → 三边 Gate2。forward 最终只需 `access_representation_bindings` 的
`address_recomputed`（9 条 access 全绑定），Source/Replay 各归一 18 个 `tt.addptr`，
其余五个 hard 类别保持 exact。

backward 的已知边界：

- inner query-block DK/DV recurrence、conditional DQ post-state 与多输出属于 J；
  outer key-block program/iteration realization 才可能属于 C/π；
- `BLOCK_HEADDIM`/`BLOCK_M`/`BLOCK_N` 数值合理，但 `BLOCK_M` 须在 J 补出 inner semantic loop
  后改绑到该 loop；
- `SEQUENCE_PARALLEL` 作为 implementation-selection C 分类合理，缺口在 closed consumer 尚不能
  用该 slot 区分 serial outer iteration 与 program-axis distribution；
- causal、bias interpretation、logical guard、atomic/inout effect 必须由 J 的
  predicate/access/effect 闭合，不得通过新增 C slot 隐藏。

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
| FlashAttention forward | pass | pass（out 2.05e-3） | pass（out 2.06e-3、lse 2.86e-6） | pass（out 2.06e-3、lse 2.86e-6） | `/tmp/intentir-flash-fwd-q-closure2-20260801/flash_attn_flash_attn_triton_fwd_kernel/gate2-recheck-001` |
| FlashAttention backward | pending | pending | pending | pending | pending |
| xFormers SplitK forward | pending | pending | pending | pending | pending |
| Flash backward one-col | pending | pending | pending | pending | pending |
| Flash linear kernel_fwd | pending | pending | pending | pending | pending |

FlashAttention forward：observer profile `triton-ttir-3.6-gate1-v14`，五个 hard 类别
（interface / op_multiset / dataflow / reduction / predicate_effect）differences 全空；
Gate2 输入 profile `flashattention-B4-H16-S8192-D128-f16-v1`，f16 tolerance
`rtol=0.02 / atol=0.003`。parent evidence
`/tmp/intentir-flash-fwd-manual-access-pi-20260801/.../run-001/attempt-04`，recheck `model_calls=0`。

所有 evidence 已于 2026-08-01 快照到
`/mnt/hdd/kingdom/intentir-evidence/20260801-snapshot/`（`/tmp` 原件保留未动）。

## 7. 完成条件

- 剩余 4 个 Kernel 各有可审计的 Gate1 pass 与 Gate2 三边 pass artifact；
- 最终 `(J,C,π)` 满足上述信息隔离与 ownership 合同；
- 没有 Gate 放宽、Kernel 特化、Source/π 泄漏、假闭合或以 TypeCheck 代替进展；
- authoritative evidence 写回本任务，代码与文档提交，工作区干净。

若某个 Kernel 最终无法闭合，按
`doc/method/IntentIR_Gate1归一化现状_证据归集与论文优先执行计划_2026-08-01.md` §2.6 的判断，
如实报告为"Gate1 当前对 loop tiling / carrier / terminal control 无一元归一化，保持 hard，
因而拒绝合法 realization 差异"的**测量结果**，不得为闭合而放宽 Gate 或加 per-Kernel 规则。
