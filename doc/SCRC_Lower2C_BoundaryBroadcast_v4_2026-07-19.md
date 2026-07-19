# SCRC Lower2C Boundary Broadcast v4

日期：2026-07-19
held-out identity：schema 33
Lower2C protocol：
`intent_linalg_cpu_symbolic_iota_integer_readback_boundary_broadcast_v4`

本轮没有使用 Trellis task/workflow，没有修改 π、Gate1、replay code 或 J schema。

## 1. 真实触发问题

schema-32 fresh `fg_eye` 已满足：

- normal Gate1 hard-close；
- 四个 fixed-C profile 全部 hard-close；
- masked-J family 全部 hard mismatch。

但旧 Lower2C 的 12 个 Gate2 case 为：

- 6 `lowering_diverged`；
- 6 `execution_failed`。

两个表象来自同一根因：J 的 `Boundary.broadcasts` 已明确记录 row/column operand 的
rank-insertion axes，但 `Lower2CSpec` 只把 `IntentFunction` 交给 MLIR emitter，丢掉了
Boundary 语义。

因此旧路径把：

```text
idx_i[N]  axes=[1]
idx_j[M]  axes=[0]
```

当成两个普通一维输入。`N==M` 时结果退化成一维 zip；`N!=M` 时 linalg input map 与
operand shape 不一致。

## 2. 为什么不需要新 schema/op

现有信息已经充分：

- `BroadcastSpec.node_ref` 指定 consumer；
- `BroadcastSpec.operand` 指定 consumer operand；
- `BroadcastSpec.axes` 指定语义新增轴；
- Intent op 的显式 result type 给出目标 domain。

缺口只是 Gate2 没有消费这些已有字段。因此本轮不增加 `intent.broadcast`、
`broadcast_in_dim`、π 字段或 writer 规则。

## 3. Lower2C v4 实现

### 3.1 Boundary 解析

`Lower2CRunner` 从 `SemanticArchive.node_ids` 确定性解析 broadcast consumer，并构造：

```text
consumer_result -> operand -> inserted_axes
```

当前支持已有真实触发的 `map_binary` 和 `select`。以下情况 fail-closed：

- target 无 node path；
- operand 不是 consumer 输入；
- 多结果 consumer；
- 非整数 axes；
- axis 越界或重复；
- 同一 result/operand 重复或冲突声明；
- rank insertion 后 operand projection 不兼容。

### 3.2 MLIR legalization

frontend emitter 在原 Intent op 前插入标准 `tensor.expand_shape`：

```text
tensor<N> -> tensor<Nx1>
tensor<M> -> tensor<1xM>
```

随后继续走现有通用路径：

```text
intent.map_binary/select
  -> linalg.generic with affine broadcast input maps
  -> loops
  -> LLVM
  -> mlir-runner
```

rank-0 scalar仍直接使用空 affine input map，不需要物化无意义的 expand_shape。

这不是 eye 特判；任何带相同 Boundary 合同的 flat `map_binary/select` 都复用同一逻辑。

### 3.3 Shape inference

Lower2C 的输出 shape inference 与 MLIR emission 使用同一 broadcast axes：

- 不再把 `[N]` 和 `[M]` 简单取 `max`；
- 显式 rank insertion 后按严格 NumPy size-1 broadcast 检查；
- 没有 Boundary facts 时保留原有行为，不猜 outer broadcast。

## 4. 单元与边界测试

新增执行测试覆盖：

- eye：`1×1`；
- 等长、最容易误退化为 zip 的 `4×4`；
- 非对称 `3×5`、`5×3`；
- 非交换 outer-sub：`lhs[:,None] - rhs[None,:]`，验证轴方向；
- 无 Boundary facts 时不得猜 outer broadcast；
- axis 越界；
- symbolic axis；
- duplicate/conflicting axes；
- 不兼容 projection。

专项测试：`12 passed`。
Gate2/held-out/recheck 扩大回归：`134 passed`。

完整 `python/tests`：`713 passed, 3 skipped, 3 failed`。三个失败与本轮前一致：两个
xFormers split-k binder 用例隔离复跑 `2 passed`；另一个是旧 universe
`count_nonzero` manifest 漂移。

## 5. 保存真实 `fg_eye` 三腿复验

源报告：

```text
/tmp/scrc_schema32_final_eye_20260719/fg_eye.json
```

v4 无模型复验：

```text
/tmp/scrc_lower2c_v4_eye_recheck_20260719/fg_eye.json
```

复验重用保存的真实 LLM artifact/replay/masked replay，只重新 compose、编译、计算
TTIR 和执行 Gate2；没有重新 Lift 或生成 replay。

结果：

- Gate1：`replay_closed`；
- normal source/replay hard comparison：PASS；
- fixed-C family：4/4 PASS；
- masked-J family：证据完整，4/4 hard mismatch；
- Gate2：12/12 `gate2_pass`；
- source↔replay、source↔Lower2C、replay↔Lower2C 全部一致。

因此该修改把 `fg_eye` 从：

```text
Gate1 closed + Gate2 incomplete
```

推进为：

```text
Gate1 closed + Gate2 pass
```

Gate1 verdict 本身没有变化。

## 6. 当前限制

本轮只实现已有 Boundary facts 能确定表达的 rank insertion。以下情况未实现，也没有
提前扩 schema：

- 任意维度置换；
- 同一 source dim 多次映射；
- diagonal/index coupling；
- tile/repeat；
- 无法从 consumer result domain 得到的动态目标 extent；
- semantic loop body 内的 Boundary broadcast Lower2C。

只有真实 Kernel 触发且现有 `(node_ref, operand, axes)` 无法表达时，才提交新的 schema
必要性报告。
