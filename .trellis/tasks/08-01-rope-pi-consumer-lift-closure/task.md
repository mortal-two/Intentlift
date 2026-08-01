# Final Task 子任务：RoPE π Consumer 与自动 Lift 闭合

> 状态：`ACTIVE`
> 性质：唯一活动执行记录；先完成 Llama4 RoPE 纵切，再将同一能力横向用于剩余 Kernel。
> 方法依据：`doc/method/IntentIR_四核Lift瓶颈与pi归一化重构诊断.md`。

## 1. 唯一目标

用当前最可信的 Llama4 RoPE `J+C` 完成两次真实闭合：

1. 冻结 `J+C`，手工作者化足够丰富、直接绑定 J/C 的 π，实现 fixed unary consumer，
   通过 Gate1 与 Gate2 三边；
2. 在不改变数学边界的前提下精简 Lift workspace，让同一 persistent Lift Agent
   自己生成等价的 `(J,C,π)`，fresh Replay，再次通过 Gate1 与 Gate2 三边。

TypeCheck、schema 改动、工具与文档都不计进展。

## 2. 不可改变的合同

```text
Lift(Source) -> (J,C,π)
ReplayAgent(J,MaskC(C);U_f) -> R
Gate1: N^(J,C,π)(Observe(Source,c0)) == N^(J,C,π)(Observe(R,c0))
Gate2: Source-Replay && Source-J && Replay-J
```

- Replay 不读 Source、π、C values、Gate diff 或历史 Replay；
- raw observer 不读 π；
- normalizer 在 Source/Replay 两臂独立执行，不读 peer observation 或 pairwise diff；
- π 只选择冻结 relation kind，并直接绑定 J/C handles；
- J/C-owned operator、value dataflow、predicate、reducer、recurrence、effect、post-state 与 output
  在 normalization 前后必须不变；
- unknown、ambiguous、绑定错误或 consumer 无法证明时 fail closed；
- 禁止 per-Kernel matcher、allowed-diff、free-text equivalence、Source recipe 和旧
  instance/region/route 大图。

## 3. RoPE 冻结起点

当前静态判断：

- J 已包含 Q/K rotary、gather/scatter、cos/sin、casts、inout effects 与两个 outputs；
- C 的 `BLOCK_SIZE=64 -> d0/axis 3` attachment 与 Q feature domain 一致；
  手工纵切先冻结 C，并要求 iteration binding 同时覆盖 Q/K 对应的同一逻辑 feature 轴；
- 当前 Gate1 同时混有 program/address realization residual 和 Replay cast/dataflow fidelity residual；
- 只有前者可由 π consumer 归一；后者必须修 Replay，若 J 本身缺失才返回 relift。

手工纵切只能改 π、固定 consumer 和 source-blind Replay；不得根据 Source 修补 J 或 Replay。

## 4. π 的活动表达能力

π 不是“只有 program axis 的极小对象”，也不恢复旧 realization graph。活动 grammar
按四核真实需求支持以下直接 relation：

1. `program_bindings`：direct、product-linearized 与 program-distributed logical domains；
2. `iteration_bindings`：domain tiling 与 structured-loop tiled recurrence，可引用 C tile/mode slot；
3. `control_bindings`：terminal guard/early-return 与 J effect control anchor；
4. `materialization_bindings`：J value 的 scratch roundtrip/recompute 表示；
5. `access_representation_bindings`：只在 tensor、logical index、domain、predicate 和 effect
   全部唯一时归一 pointer/address representation。

每条 row 只含 relation kind、J anchor 与可选 C slot/axis；不含 Source/Replay node ID、
代码片段、要删除的 observation field 或自定义 comparator。

RoPE 纵切至少需要：

- batch×sequence domain 的 product-linearized program binding；
- Q/K guarded head domains 的 shared/guarded-union program binding；
- `d0/axis 3 + c0` 为主锚点、并显式覆盖 `d1/axis 3` 的 feature-domain tiling binding；
- 只在观测确有 terminal/address/materialization residual 且 fixed consumer 能保守证明时，
  加入对应直接 binding。

## 5. Lift Agent 如何生成 π

仍由同一 source-aware Lift owner 完整作者化 `(J,C,π)`，不增加 planner/reviewer/
composer 或第四 artifact。同一 session 内顺序为：

```text
author J
  -> classify and attach C to J
  -> query raw realization ledger
  -> bind closed π relations to existing J/C
  -> CandidateCheck
  -> submit one complete artifact
```

Lift workspace 只新增按需查询的 raw realization ledger：

- physical program-coordinate 数量与 div/rem/mul/add 使用；
- physical loops 的 bound/step 及匿名 coordinate use；
- loop carrier/yield 的 typed use categories；
- terminal control 的 predicate/effect shape；
- scratch roundtrip 与 address-progression 的 typed raw facts；
- 当前 J 可用 `subject_ref/axis_ref`，以及 C attachments。

ledger 不判断这些事实属于 J/C/π，不建议 relation 或 target；LLM 完成所有
分类与绑定。fixed checker 只检查 schema、handle/type/axis 与 C–π attachment consistency，
不生成、改写或自动补全 relation。

## 6. 执行顺序

### A. 手工 RoPE 纵切

- [x] 选定 RoPE J/C 并复制到新 evidence root；旧 evidence 未改；
- [x] 手工作者化直接绑定的 RoPE π；
- [x] 实现对应 schema/Decode/dependent checks；
- [x] 实现 access/address 的 Triton fixed unary consumer 与仿射 hard-preservation；
- [x] 真实 Gate1 将错误第三 program axis 判为 Replay fidelity，而非用 π 掩盖；修复后闭合；
- [x] Gate1 closure 后完成 Source–Replay、Source–J、Replay–J 三边 Gate2。

### B. 精简 Lift 并自动作者化 π

- [ ] 只保留 Source/Harness/working candidate/bounded feedback 与按需查询入口；
- [ ] 完整 schema、IntentOps、preflight 和示例不再重复无条件挂载；
- [ ] 提供 raw realization ledger 与 J/C attachment query；
- [ ] 同一 persistent Lift owner 作者化完整 `(J,C,π)`；
- [ ] fresh Replay、Gate1 和三边 Gate2；
- [ ] 只有自动路径再次全部通过，RoPE 才计入最终闭合 corpus。

## 7. 当前阻塞

1. RoPE 所需 access/address consumer 已闭合；program/iteration/control relation 在本次
   Source/Replay 无对应 residual，仍保持 hard/unused，不能宣称已有通用 quotient；
2. Lift Agent 尚未自动作者化新的直接绑定 π 与显式 inout output mapping；
3. Lift workspace 仍承担重复上下文与 dependent JSON 机械成本；
4. fresh 自动 Lift/Replay 尚未给出 Gate1/Gate2 证据。

## 8. 真实证据

| 阶段 | Gate1 | Source–Replay | Source–J | Replay–J | evidence |
|---|---|---|---|---|---|
| 手工 π 纵切 | pass | pass（误差 0） | pass（最大误差 4.8828125e-4） | pass（最大误差 4.8828125e-4） | `/tmp/intentir-rope-pi-consumer-20260801/manual-closure/liger_llama4_rope/recheck-001` |
| 自动 Lift 闭合 | pending | pending | pending | pending | pending |

## 9. 完成条件

- RoPE 手工 π 纵切 Gate1 closure 且 Gate2 三边全部通过；
- 轻量 Lift Agent 自己生成同等表达力的完整 `(J,C,π)`；
- 自动路径 fresh Replay、Gate1 与 Gate2 三边全部通过；
- 没有放宽 Gate、Kernel 特化、Source/π 泄漏、第四 artifact 或 fixed semantic construction；
- 真实 evidence 路径写回本任务，工作区提交且干净。
