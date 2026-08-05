# IntentIR：Gate1 归一化现状、证据归集与论文优先执行计划

> 日期：2026-08-01
> 性质：现状诊断 + 执行计划。不是证书、不是新协议、不引入新 artifact 或 Gate。
> 目的：回答三个问题——(1) 证据往哪里归集；(2) Gate1 的归一化到底是怎么实现的、是不是太弱；
> (3) 前端 SCRC 与后端 J-consumption 两半合起来现在能写出什么论文。

---

## 0. 一页结论

**手上的东西比任务卡上写的多。** 按证据文件（不按 task.md）重算：

| 项目 | 真实数字 | 依据 |
|---|---|---|
| 有 `scrc_certified` 收据的 Kernel | **24 / 28** | `/tmp/**/recheck.json` + `/tmp/**/batch-report.json` 扫描 |
| 其中在当前 v13/v14 profile 下 | 5 | RoPE、Llama4 RoPE、Qwen2VL mRoPE、SplitK reduce、Flash forward |
| 从未闭合 | **4** | Flash bwd、Flash bwd one-col、Flash linear kernel_fwd、SplitK forward |
| 后端已消费的 accepted J | **24 / 24** | 后端分支 `已通过24个J_三目标后端总览_2026-07-31.md` |
| 后端 GPU 完整矩阵 | 5090D 24/24、H100 24/24 | 同上 §3 |

**FlashAttention forward 已于 2026-08-01 17:39 闭合**，`.trellis/tasks/08-01-final-three-kernel-closure/task.md`
仍写 pending，落后于实际进展。证据：`/tmp/intentir-flash-fwd-q-closure2-20260801/flash_attn_flash_attn_triton_fwd_kernel/gate2-recheck-001/`。

因此**论文的瓶颈不是证据不够，是证据没有归集、叙事没有对齐**。剩下三核（Flash bwd、SplitK fwd、
Flash linear）即使永远不闭合，24/28 + 三目标后端已经足够支撑 PPoPP 的三条 claim。

---

## 1. 证据归集：从 `/tmp` 到 `papers/intentir-ppopp27/experiments`

`/tmp` 不会丢失（已确认），但当前有 1010 个 `/tmp/intentir-*` 条目互相覆盖，**没有索引**，
论文表格无法从中直接生成。目标目录已存在且已定义 schema，但 `raw/normalized/derived/receipts/figures`
五个目录全空。

### 1.1 需要归集的 24 份权威证据

按 case 取**最新且状态为 `scrc_certified`** 的那一份。已定位的独立 recheck 收据（17 项）：

| Kernel | evidence root | 日期 | profile |
|---|---|---|---|
| liger_llama4_rope | `/tmp/intentir-rope-auto-certified-20260801-final/liger_llama4_rope/gate2-recheck-001` | 08-01 | v14 |
| flash_attn._fwd_kernel | `/tmp/intentir-flash-fwd-q-closure2-20260801/flash_attn_flash_attn_triton_fwd_kernel/gate2-recheck-001` | 08-01 | v14 |
| liger_qwen2vl_mrope | `/tmp/intentir-task15-qwen-contract-indexed-gate2-recheck-20260730-01/...` | 07-30 | v13 |
| splitK_reduce_varargs | `/tmp/intentir-task15-v12-splitk-reduce-exact-current-gate1-gate2-20260730-01/...` | 07-30 | v13 |
| liger_rope | `/tmp/scrc_task11_v116_rope_gate2_recheck_20260726_01/liger_rope/gate2-recheck-001` | 07-26 | v13 |
| liger_cross_entropy | `/tmp/intentir-task14-wave33-crossentropy-compressed-gather-recheck-20260729/...` | 07-29 | v11 |
| liger_group_norm | `/tmp/intentir-task14-wave31-final-consumer-rechecks-20260728/liger_group_norm/...` | 07-29 | v11 |
| flash layer_norm 1pass | `/tmp/intentir-task14-gate2-capability-rechecks-20260728/...layer_norm.../gate2-recheck-002` | 07-28 | v11 |
| xformers rms_norm_add | `/tmp/scrc_task11_v14_rmsnorm_add_canonical_consumers_recheck_20260725_01/...` | 07-28 | v11 |
| xformers rms_norm | `/tmp/scrc_task11_v4_gate2_rechecks_20260725_01/...rms_norm_kernel/gate2-recheck-004` | 07-25 | v11 |
| flash bwd_preprocess | `/tmp/scrc_task11_v4_gate2_rechecks_20260725_01/...bwd_preprocess_do_o_dot/gate2-recheck-001` | 07-25 | v11 |
| liger_layer_norm | `/tmp/scrc_task11_v52_layernorm_gate2_recheck_20260725_01/...` | 07-26 | v11 |
| liger_softmax | `/tmp/intentir-task13-v52g-miniswe-softmax-identity-gate2-recheck-20260727/...` | 07-27 | v11 |
| liger_sparsemax | `/tmp/scrc_task11_gate2_capability_rechecks_v26_20260725_01/liger_sparsemax/gate2-recheck-003` | 07-26 | v11 |
| liger_jsd | `/tmp/scrc_task11_v5_jsd_loopcarry_recheck_20260725_02/liger_jsd/gate2-recheck-001` | 07-25 | v11 |
| liger_tvd | `/tmp/scrc_task11_v44_tvd_loop_capture_refeval_20260725_01/liger_tvd/gate2-recheck-001` | 07-26 | v11 |
| liger_poly_norm | `/tmp/scrc_task11_v13_polynorm_after_direct_mixed_20260725_01/...` | 07-25 | v11 |

另 7 项只有 batch-report 内的 `classification=scrc_certified`（无独立 recheck 目录）：
DYT、SwiGLU、GeGLU、RMSNorm、fused-add RMSNorm、KLDiv、SplitK reduce backward，
主要在 `/tmp/intentir-task14-wave1-ready-{a,b}-20260728/` 与 `wave2-ord08-14-terra-high-20260728/`。

### 1.2 归集动作（一次性脚本，不进 SCRC 路径）

写一个 `tools/collect_evidence.py`，做且只做搬运与索引：

```
papers/intentir-ppopp27/experiments/
  raw/20260801_claim1_triton_sm120a_v14/<case_id>/
      artifact.json  gate1.json  gate2.json  replay.py  recheck.json  typecheck.json
  receipts/20260801_claim1_.../index.csv      # case_id, artifact_digest, profile, 日期, 源路径
  normalized/20260801_claim1_.../claim1-certification.csv   # 按已有 schema 填
```

关键要求：**保留 `artifact_digest` 与原 `/tmp` 路径**，让每一行可回溯。不重跑、不改判、
不把 v11 的收据升级成 v14。表格里如实标注 profile 版本列——这一列本身就是论文可以正面写的
"protocol 演进过程中旧证书不被静默追认"的证据。

**这件事优先级最高，因为它是纯搬运、零风险、且论文的每张表都依赖它。**

---

## 2. Gate1 的归一化到底是怎么实现的

这是你问的核心。**结论：归一化是真实存在且相当强的，但它不在 π 那一层。**

### 2.1 实际有两层归一化，不是一层

设计文档写的是单层：`Φ^π = N^π(Canon(Observe(Compile(P,c))))`，π 参数化那个 `N^π`。
实现里是两层，而且**重活全在第一层**：

```
                    ┌──────────────────────────────────────────────┐
  TTIR 文本  ──────► │ 第一层：raw Canon（π-blind，ttir.py）         │
                    │  · alpha/SSA 名消除 → 内容 digest 化 node_ref │
                    │  · 交换律算子操作数排序、cmp 极性翻转          │
                    │  · FMA ≡ mul+add（v13）                       │
                    │  · representation_only 商（★ 最重的一步）      │
                    │  · integer_polynomial 地址/下标多项式规范化    │
                    │  · pointer_form = (base, offset 多项式)        │
                    └──────────────────┬───────────────────────────┘
                                       │  TTIRFingerprint
                    ┌──────────────────▼───────────────────────────┐
  π  ─────────────► │ 第二层：N^π（normalize.py，当前 只有 1 条规则）│
                    │  · jc_access_address_representation           │
                    │    ＝ 把 tt.addptr 从 anchor 集合里豁免        │
                    └──────────────────────────────────────────────┘
```

第二层的全部代码是 `python/intentir/verify/gate1/normalize.py:56-140`，激活规则只有一条，
而且它做的事只是一个**开关**：把 `tt.addptr` 加进 `representation_anchor_exemptions`，
从而允许第一层的 representation 商把 addptr 链也收缩掉。

### 2.2 第一层的商到底擦掉什么（`ttir.py:2045-2180`）

`_representation_only_nodes` 保三个硬切片不动：

1. **进入返回值 / store 值 / atomic 更新值**的所有计算（`mark_value` 递归）；
2. **可观察谓词的布尔树及其直接常量**（`mark_predicate`，只沿 `_PREDICATE_VIEW_OPS` 递归）；
3. **effect、structured region、transfer、以及显式 address/program anchor**
   （`tt.get_program_id` / `tt.get_num_programs` / `tt.addptr`）。

剩下的、且在 `_KNOWN_PURE_OPS`（约 50 个 arith/math 算子的**手工白名单**）内的算子，
被收缩成一条带类型的 representation 边。未知算子**永不**因为"没到输出切片"而变噪声——这条守卫是对的。

### 2.3 真正做事的是整数多项式规范化（`ttir.py:4395-4478`）

这才是 RoPE 能闭合的原因。`integer_polynomial` 把地址/下标算术**代数地**规约成
原子上的多项式：

- 原子集合 = `{program_axis:x/y/z, lane:<make_range 签名>, 运行时常量}`；
- `addi/subi/muli` 真正做多项式加减乘（`polynomial_add` / `polynomial_mul`），系数为 0 的项删除；
- `extsi/trunci/splat/broadcast/expand_dims/reshape/trans` 透明穿透；
- `divsi/remsi` 不展开，规约成一个不透明的 `integer_relation:<digest>` 原子（保守）；
- 指针 = `(base_leaf, offset 多项式)`。

**后果**：Source 写 `base + (m*sm + d)`、Replay 写 `base + m*sm + d`、甚至先算 `m*sm` 再加，
只要多项式相同就同一条边。这就是为什么 RoPE 的 Source **10 个** `tt.addptr` 能和 Replay 的
**5 个** 闭合（`gate1.json`: `source_normalization.transformed_occurrences=10`,
`normal.normalization.transformed_occurrences=5`），Flash forward 两边各 18 个也闭合。

### 2.4 所以"归一化太弱"这个判断——一半对

| 层 | 强度 | 问题 |
|---|---|---|
| 第一��� raw Canon | **强**。地址/下标/调度算术已被代数规约 | 它是 π-blind 的，π 拿不到功劳 |
| 第二层 N^π | **几乎为零**。1 条规则，且只是个 bit | 设计文档说 π 承载归一化，实现里 π 只有开/关 |

**π 目前不是"太弱的归一化器"，π 根本不是归一化器**——它是一个布尔开关，
外加一堆被 `normalize.py:115-129` 标成 `hard_relations` 的、**没有消费者的惰性字段**
（`program_axes` / `program_bindings` / `iteration_bindings` / `control_bindings` /
`materialization_bindings` 一旦非空就直接进 hard 列表，等于"声明了但不消费"）。

**真正卡住剩余三核的，不是归一化太弱，是归一化的位置不对：**
第一层能规约的是**纯算术**（地址、下标、调度）；卡住 Flash bwd / SplitK fwd 的是
**loop tiling 结构、carrier 布局、terminal control 形状**——这些既不在第一层的 pure-op 白名单里
（`scf.for` 有 region、有 effect，永远进不了 representation 商），也没有第二层的规则。
所以两边都管不到，直接作为 hard dataflow 边参与精确比对。

### 2.5 这对论文意味着什么（**重要，且不需要改代码**）

现在的实现完全可以用一个**与代码一致且更强**的数学表述来写：

> 令 `Canon_{f,p}` 是 profile 冻结的、π-independent 的基线规范化（α-等价、交换律、
> FMA、以及 realization-carrier 的类型化商 + 整数多项式规约）。
> 令 `N^π_{f,p}` 是同一 profile 冻结的、由 π **选择激活子集**的一元规则族。
> `Φ^π = N^π ∘ Canon`。当前 profile 的激活集 = `{jc_access_address_representation}`，
> 其余已声明 relation 保持 hard（fail-closed）。

这段话每一句都能在代码里指出行号，而且它把第一层的功劳正确地归给"profile 的基线规范化"，
把 π 正确地降格为"激活子集的选择器"。论文里**不要**写成"π 承载全部归一化能力"——那句和代码对不上，
而且是 reviewer 一 grep 就能发现的。

反过来，这个表述有一个额外好处：**它让"π 目前只激活一条规则"从缺陷变成设计声明**——
"我们只发布通过 hard-preservation 论证的规则，其余保持 hard"，这正是 7-31 撤销
`x/y/z` label quotient 那次决策的一般化。

### 2.6 如果要让剩余三核闭合，需要加的规则（按性价比排序）

不建议为了论文去做，但列出来供判断：

1. **`iteration_tiling`**：把 `scf.for(0, N, T)` + 内部 `tile` 索引与 J 的 logical loop 对应，
   要求 trip domain 唯一回指 J domain。这是 Flash bwd 与 SplitK fwd 共同的缺口。
   风险：`scf.for` 带 region/effect，商它需要独立的保序论证，工作量大。
2. **`carrier_layout`**：loop carrier 的 tuple 顺序归一。风险低但收益也低（当前 v28 已按声明序对齐）。
3. **`terminal_control`**：早退/守卫的两臂形状归一。Flash bwd 需要。

三条都要先有"Kernel 无关的一元 normalizer + hard-preservation 论证"才能进 schema——
这是 `doc/method/IntentIR_J-only_Replay与JC结构Gate1_2026-07-31.md` §5 已经定下的门槛，
不应为了闭核而绕过。

---

## 3. 后端：两半合起来是什么

后端在**独立分支** `/home/kingdom/phdworks/TianchenIR-backend-certified-j-search`，
权威报告 `doc/已通过24个J_三目标后端总览_2026-07-31.md`（+ 08-01 的 RVV 更新）。
本分支 `corpus/reports/PERF.md`（06-26）与 `spec/lowering/high-performance-kernels.json`（07-20）
**已过时，不要用于论文**——PERF.md 里 FlashAttention 那三行还是 875×–983× 慢于 Triton 的旧数据。

### 3.1 后端真实结果（可直接进论文）

四柱：`C-Fixed / Schedule-Only / Full / Source(GPU) 或 Native(RVV)`，
`G_schedule = C-Fixed/Schedule-Only`，`G_full = Schedule-Only/Full`。

| Target | 范围 | `G_schedule` | `G_full` | Source/Full |
|---|---|---:|---:|---:|
| 5090D | 24 项合并 | ≈1.266× | ≈1.055× | **≈1.291×** |
| H100 | 24 项合并 | ≈1.366× | ≈1.138× | **≈1.409×** |
| RVV | 阶段二 8 项 | 1.769× | 1.792× | 0.863× |

RVV 六项正式结果（08-01 更新）：Gather 1.608×、TVD 6.522×、Softmax 9.852×、
xFormers RMSNorm 4.888×、Flash bwd preprocess 28.771×。

**这一半的证据质量高于前端一半**：code-warm/data-cold、每样本 L2 eviction、CUDA event 计时、
RVV 固定 affinity 单线程，而且报告自己废弃了旧的 `52.201×/89.373×`（归因边界错误）——
这种自我更正在论文里是加分项，可以在 §5 明写一句"我们发现并更正了一次归因错误"。

### 3.2 两半怎么接成一条故事线

它们**共享同一组 24 个 accepted J**，这是唯一也是足够的接口：

```
   Triton Source ──Lift(LLM)──► (J, C, π)
                                  │
                      TypeCheck + Gate1 + Gate2  ──► accepted J
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              GPU consumer                RVV consumer
           (5090D / H100)              (单线程 scalable vector)
           1.29× / 1.41× vs Source        跨目标复用实例
```

论文的第三条 claim（"accepted J 是 source-independent compiler interface"）**已经有证据**：
同一个 J 在 GPU 与 RVV 上被选出**不同的 concrete MLIR 路径**（报告 §4 的表格逐项列了）。
这比"我们的后端很快"强得多——快是次要的，**同一语义资产被两个独立 consumer 复用**才是论点。

### 3.3 两半之间必须补的一件事

后端报告 §6 承认："阶段一 14 项的 headline shape/dtype 没有逐项汇总"。
论文表格需要每一项的 shape/dtype，否则 reviewer 会问"你们在什么规模上测的"。
这是**已有数据的元信息补全**，不是重跑——从当时的 workload report 里捞即可。

---

## 4. 论文优先的执行顺序

按"最短路径到一份可投稿的稿子"排，不按"把系统做完"排。

### P0（本周，纯归集，零技术风险）

1. **`tools/collect_evidence.py`**：把 §1.1 的 24 份证据搬进
   `papers/intentir-ppopp27/experiments/raw/` + `receipts/index.csv`，保留 digest 与源路径。
2. **更新 `task.md`**：Flash forward 已闭合，剩 3 核（bwd / bwd one-col / SplitK fwd / Flash linear
   —— 实为 4 项未闭合，任务卡写的"三核"漏了 Flash linear）。
3. **后端数据落地**：把后端分支 `doc/results/stage2-strong-canonical-baseline-23/{5090d,h100}.json`
   与 `stage3-*`、`rvv-external-baselines-20260801` 复制进同一个 experiments 树（`raw/claim3_*`）。
4. **补后端 shape/dtype 元信息**（§3.3）。

### P1（下周，把已有证据变成论文数字）

5. **normalized/derived 两级 CSV**：按已定义 schema 填。Claim 1 用 §1.1 的 24 行；
   Claim 3 用后端三目标矩阵。**分母如实写 28，分子 24，另 4 项单列失败类别。**
6. **Gate1 表述改写**：按 §2.5 的两层公式写 §3.2，每句话标行号。
   删除任何"π 承载归一化"的措辞。
7. **写 §5 的 coverage 表**：24/28 + profile 版本列 + 4 项未闭合的**结构原因**
   （不是"没做完"，是"Gate1 当前对 loop tiling/carrier/terminal control 无一元归一化，
   保持 hard，因此合法 realization 差异被拒"——这是一个**结果**，不是一个 TODO）。

### P2（有余力再做，能显著加分但不是必需）

8. **Mutation sensitivity**：拿已闭合的 24 个 J，程序化注入错误
   （reducer max→sum、mask 极性翻转、axis 错位、dtype 错误），量化 Gate1 的拦截率。
   **这个实验不依赖剩余核闭合，且直接支撑 Claim 2（authority）**，是当前性价比最高的新实验。
9. **|T| 从 1 提到 5–10**：当前每个 Kernel 的 Gate2 只跑**一个** hidden case
   （`scrc_workload_run.py:459`，单 shape 单 seed）。加到 5–10 并含边界输入，
   把"三边行为闭合"从一个样本变成一个测试域。代价极小，抗打击力提升明显。
10. **π ablation**：当前 `jc-normalizer.v2` 只有一条规则，可以做 with/without 该规则的
    closure 率对比——这是论文里"π 确实有用"的唯一直接证据，成本是重跑 24 份 Gate1（零模型调用）。

### 明确不做

- 不为了闭合剩余 4 核而放宽 Gate、加 per-Kernel 规则或 allowed-diff。
  它们闭不上这件事本身就是可报告的结果（见 P1.7）。
- 不重跑 v11 的 17 份证书去追 v14。协议演进过程中旧证书按原版本解释，这是既有纪律，
  且在论文里比"全部统一到最新版"更可信。

---

## 5. 论文能写什么 / 不能写什么

**能写（有证据）**：

- 24/28 Triton Kernel 通过 TypeCheck + source-blind Gate1 + 三边 Gate2；
- 同一组 24 个 accepted J 被 GPU 与 RVV 两个独立 MLIR consumer 消费，
  GPU 相对 Source 几何均值 1.29×（5090D）/ 1.41×（H100）；
- `C` 解冻带来 1.27×/1.37× 的 target retile 收益，证明 C 不是 legality 的一部分；
- Replay 在只见 J + MaskC(C) 的条件下，复现出 Flash forward 的 online-softmax 三状态递推
  与 scratch store/load 往返（后者是 `flash_attn_triton.py:224-225` 的真实实现细节）；
- 协议演进过程中主动**收紧**了一次门（7-31 撤销 program-axis label quotient），
  代价是当时闭不了核。

**不能写（会被 grep 到）**：

- "π 承载 Gate1 的归一化能力"——实现里 π 只选激活子集，重活在 π-blind 的基线规范化；
- "28/28 全部认证"——是 24/28，且其中 17 项停在 v11 profile；
- "Gate2 是行为等价证明"——每 Kernel 只有 1 个 hidden case，1 个 shape，1 个 seed；
- "RVV 有 24 项完整矩阵"——阶段一 Fresh-14 只有 4 项正式结果；
- 本分支 `corpus/reports/PERF.md` 的任何数字——06-26 的旧数据，FlashAttention 那三行是 875×–983× **慢于** Triton。

---

## 6. 一句话

**证据够了，缺的是归集和叙事对齐。** 优先做 §4 的 P0/P1（纯搬运 + 表述改写，两周内可完成），
P2 里的 mutation sensitivity 是唯一值得新做的实验。剩余 4 核不闭合不影响论文成立——
把它们作为"Gate1 在 loop/carrier/control 上缺一元归一化因而拒绝合法 realization 差异"的
**测量结果**报告，比挂着当 TODO 更有信息量。
