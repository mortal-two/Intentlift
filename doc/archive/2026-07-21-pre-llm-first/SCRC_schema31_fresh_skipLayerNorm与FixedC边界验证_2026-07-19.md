# SCRC schema-31 fresh SkipLayerNorm 与 fixed-C 边界验证

日期：2026-07-19
模型：`gpt-5.6-sol/xhigh`
协议：`gate1_fixed_c_family_anti_leakage_diff_guided_pi_relift_v11`
TTIR fingerprint：`triton-ttir-3.6-gate1-v3`

本轮没有使用 Trellis task/workflow，也没有恢复旧 verifier、JSON interpreter 或
replay repair。没有新增 IntentIR op、J 字段、C.strategy family、RuleCard 或新的
Gate1/Gate2 verdict。

## 1. 为什么选择该样本

`fg_skip_layer_norm` 是活动 20-ready 队列中尚无 fresh SCRC 证据的既定样本。它覆盖：

- residual add；
- 两次 row reduction（mean/variance）；
- epsilon、sqrt、reciprocal；
- weight/bias affine；
- 四个输入 tensor 和独立 row/column stride；
- masked load/store。

因此它不是继续增加同类单 reduce，而是 fused norm、多输入访问和数值稳定边界的真实
family。

## 2. Fresh 完整流程

报告：

```text
/tmp/scrc_schema31_fresh_skip_norm_20260719/fg_skip_layer_norm.json
```

### 2.1 Lift

- source preflight：通过；
- initial Lift：1 个真实响应；
- response id：
  `resp_0feecaf4e5011d29016a5bb884e018819b9c2eb3eaff6efbd2`；
- initial Lift 墙钟：约 203.0 秒；
- J/interface/C/π schema、deterministic compose equality 和 provenance 均通过。

### 2.2 首轮 Gate1 与 Relift

第一份 source-blind replay 可编译，但 Gate1 在以下 hard categories 不闭合：

```text
op_multiset + dataflow
```

exact hard-diff digest：

```text
4ce7ea6406a8051d412de354211cefc54a13426982f2e4b604c6c6c9c230f25b
```

该新鲜差分进入一次 source-aware Relift：

```text
resp_0e129f8722144abb016a5bb9de5e64819b9996907c594f78b4
```

interface 与 C.schedule 保持冻结；Relift 返回完整新 artifact 并重新 finalize π。随后从
source compile、静态检查和 fresh writer 全流程重跑。第二份 normal replay 可编译，
全部 Gate1 hard categories 闭合。

这证明当前 Relift 通道不是一次性 Lift，也没有修补旧 replay code。

### 2.3 Paired semantic intervention

normal control 闭合后执行 3 个真实 paired interventions：

| intervention | 结果 | 观察 |
|---|---|---|
| `masked_fill:a2:0_to_1` | PASS | exact registered TTIR delta |
| `masked_fill:a3:0_to_1` | PASS | exact registered TTIR delta |
| `binary_fn:n11:add_to_sub` | `off_target_drift` | 除目标 `addf→subf` 外，又删除 2 个 `divf` 和 1 个 `sitofp` |

第三个 mutant writer 将 mean/variance 中的 `/N` 一并丢失，因此不是“只响应目标 J
变异”。Gate1 按既定 deterministic causal policy 返回：

```text
LEAKAGE_REJECTED
anti_leakage_basis = paired_intervention
```

这不是 π topology hard closure：

- normal replay 已闭合；
- 两个 fill intervention 精确通过；
- 失败是 fresh mutant writer 的 off-target semantic drift；
- Gate1 在 paired 阶段已经拒绝，因此没有继续运行 masked-J/full-π terminal attack；
- π ablation 也因没有合法 `REPLAY_CLOSED` control 而跳过。

本轮没有据此修改 π 拓扑，也没有触发“跨 family 大量 π 拓扑泄漏”警报。

## 3. 独立 Gate2 结果

虽然 Gate1 为 `LEAKAGE_REJECTED`，与最终 artifact digest 对齐的 normal replay 仍可作为
rejected-sample diagnostic 独立执行。原始 contract 上：

```text
3 seeds × 1 shape × 1 value = 3 cases
source ↔ replay   3/3 PASS
source ↔ Lower2C 3/3 PASS
replay ↔ Lower2C 3/3 PASS
```

Gate2 PASS 不覆盖 Gate1 rejection；该样本属于：

```text
Gate1 not closed + Gate2 pass
```

## 4. fixed-C width family 与数值边界

在不改变 `BLOCK_SIZE=512` 的条件下，为该真实 Kernel 增加四个 width profiles：

```text
N = 1, 257, 511, 512
rows = 7
grid = (7,)
```

只改变 runtime tensor shape、stride 和 N；J/interface/C/π/replay 均保持同一份。另声明
三个 Gate2 value profiles：

- `normal`；
- `large_dynamic`；
- `near_constant`。

每个 value/shape 使用 3 个固定 seed，共 36 个 Gate2 cases。

保存真实 artifact/replay 的当前 stack 无模型重检位于：

```text
/tmp/scrc_schema31_skip_norm_family_recheck_20260719/fg_skip_layer_norm.json
```

### 4.1 Gate1 family

同一 replay 在全部四个 fixed-C profiles 上：

```text
source compile  4/4
replay compile  4/4
Gate1 hard pass 4/4
```

这补上了 fused norm 的 fixed-C multi-specialization normal evidence。

### 4.2 Gate2 matrix

结果：

```text
gate2_pass       27
lowering_diverged 9
```

normal 与 large-dynamic cases 全部通过。9 个失败全部来自 `near_constant`，分布在
`N={257,511,512}`：

```text
source ↔ replay   PASS
source ↔ Lower2C FAIL
replay ↔ Lower2C FAIL
```

最大绝对差约为 `9.7e-5～4.3e-4`。额外使用 float64 whole-tensor reference 检查一个
固定 seed：

| N | source↔float64 max abs | Lower2C↔float64 max abs |
|---:|---:|---:|
| 257 | `7.6e-5` | `1.1e-4` |
| 511 | `7.2e-5` | `4.4e-4` |
| 512 | `6.6e-5` | `3.3e-4` |

near-constant 输入令 variance 接近 epsilon，mean/variance 的 f32 reduction order 差异
会被 reciprocal standard deviation 放大。当前证据表明 production Lower2C 比
source/replay 更偏离高精度参考，不能把失败归因给 replay。

本轮保留 `lowering_diverged`，没有通过扩大统一 atol、删除 adversarial value profile
或改写正确 J 来制造 PASS。后续若要引入 value-conditioned tolerance 或改变 Lower2C
reduction accumulation，必须先单独论证；本轮没有实施。

## 5. 代码与验证

本轮计划内代码变化只在真实 held-out contract：

- `fg_skip_layer_norm` 增加 4 个 fixed-C width profiles；
- 增加 normal / large-dynamic / near-constant value profiles；
- Gate1 family 与 Gate2 matrix 复用现有 runner，没有新增验证模式。

聚焦回归：

```text
128 passed, 33 warnings
```

其中覆盖 held-out identity、family contract、Gate1 specialization、live harness 和 Gate2。

## 6. 结论与后续归因

1. fresh normal replay 在一次 Relift 后真正闭合，证明 hard-diff guidance 对该 fused norm
   样本产生了有效下一轮 artifact。
2. Gate1 rejection 来自 paired mutant 的 off-target drift，不是 normal replay 失败，
   也不是 masked-J/full-π topology closure。
3. 同一 normal replay 已在四个 fixed-C width profile 上 Gate1 4/4 闭合。
4. Gate2 normal/large-dynamic 全部通过；near-constant 暴露 production Lower2C 数值路径
   的稳定性边界。
5. 下一步应继续按现有 taxonomy 分开处理 writer causal adherence 与 Lower2C numerical
   accuracy；不能用 Gate2 PASS 覆盖 Gate1，也不能用放宽 Gate2 统一容差掩盖第三腿差异。
