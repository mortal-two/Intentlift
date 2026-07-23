# Task 05：Gate2 dense pointwise RefEval 合同闭合

> 状态：执行中
>
> 类型：SCRC 系统能力新增/收紧；不是在失败实验内反馈 Gate2，也不是 per-kernel oracle。

## 直接目标

从 Task04 `liger_dyt/run-003` 的真实结果出发，统一 Lift wire knowledge 与现有 `E_J` dense pointwise
能力，使新的完整 `(J,C,pi)` 能在 Gate1 closure 后进入三边 Gate2，而不是因可前置说明的 direct
index/type/output mapping 拼写差异返回 `REFERENCE_UNSUPPORTED`。

正式认证谓词不变：

```text
SCRC_CERTIFIED
iff TypeCheck(J,C,pi) and Gate1 and Gate2.
```

Gate2 失败仍不回流同一次运行。run-003 已结束；所有修改都发生在实验外，随后从 Source 开始新的
Lift/Replay/Gate run。

## 真实触发证据

Task04 run-003 已获得：

- canonical decode 3/3；
- TypeCheck 3/3；
- accepted formal Gate1 round 3；
- normal Replay hard-close；
- masked-J mismatch，`J_NECESSARY=PROVED`；
- Gate2 Source--Replay 通过，`max_abs_error=0`；
- RefEval 首个错误：二维 dense access 使用 `"k1,k0"`，而现行 `E_J` 只接受 bare rank-1 axis
  或 bracketed `"[axis0, ...]"`。

对同一 accepted J 做不调用 provider 的内存探测得到：

```text
仅 bracket direct index
  -> E_J store producer type differs from output

+ observable producer type
  -> E_J output destination relation is incomplete

+ 删除 direct same-handle output 的冗余 output_mapping
  -> RefEval PASS, profile=logical_domain_mask_erasure_v1
```

因此这不是要扩充新 IR，而是把三个已经存在的消费者合同明确给 Lift。

## 系统修改范围

1. 在 source-neutral Lift schema description、Lift contract 和 MiniSWE preflight 中明确：
   - rank-1 direct access 使用 bare axis，例如 `k0`；
   - rank-0 使用 `0` 或 `[]` 的现行合法形式；
   - rank >= 2 的 direct dense access 使用 bracketed ordered axes，例如 `[k1,k0]`；
   - 不使用未括号化逗号列表。
2. 对存在 predicate/fill 的 pure functional boundary，observable store producer 必须携带与 J output
   完全一致的 `type.dtype/type.shape`，使 `E_J` 能机械检查 mask erasure 后的输出类型。
3. 当 op 已直接产生同名 J output handle 时，`boundary.output_mappings` 保持空；只有
   `IntentFunction.returns` 存在真实 output-to-value 重绑定时，才声明完全一致的 mapping。
4. 更新过时的 lowering spec：当前 `logical_domain_mask_erasure_v1` 不只覆盖 rank-1 identity，也覆盖
   通过 direct access/effect/full-domain predicate/producer-type 检查的 pure functional pointwise J。
5. 不放宽 `E_J` 去猜测 `k1,k0`，不由 Harness 重写 J，不增加 Gate、hash、RuleCard 或 DYT 专用逻辑。

## 验证

1. schema query 的 `section access` 能直接看到 bracket 规则；preflight/contract 有机械一致措辞；
2. 正反例覆盖 bracket、producer type、冗余 mapping；
3. run-003 accepted artifact 的纯内存投影在三项通用修正后 `ScalarLoweringRunner` 成功；
4. 定向与全量测试、compileall、diff check 通过；
5. 新的 `liger_dyt` canary 从零开始，不读取 run-003 candidate/Gate2 output；若 Gate1 再闭合，Gate2
   必须真实执行三条边。只有三条边全部通过才能报告 `SCRC_CERTIFIED`。

## 非目标

- 不把 run-003 Gate2 数值或 hidden input 反馈给 Lift；
- 不加入 DYT 语义实现、per-kernel comparator 或 reference formula；
- 不让 TypeCheck 承担一般 affine expression 证明；
- 不把 RefEval 变成 production lowerer；
- 不因本任务恢复向量 RAG、额外 Agent 或新的 artifact。

## 2026-07-23 实现 checkpoint

- knowledge profile 已升级为 v4；canonical boundary grammar、Responses Lift contract 与 MiniSWE
  preflight 已统一 direct-index、predicated producer type 与 output mapping 三项规则；
- `section access`/`section output_mapping` 的 source-neutral 叶子查询直接给出相同规则；
- lowering spec 已按当前实现修正：`logical_domain_mask_erasure_v1` 覆盖满足 full-domain relation 的
  pure functional pointwise J，不再误写为仅 rank-1 identity；
- 新增二维 masked pointwise 正例，以及未括号化 index、缺 producer type、冗余 mapping 三个
  fail-closed 反例；定向回归 `67 passed`，全量 Python 回归 `831 passed`，`compileall` 与 diff
  check 通过；
- 随后从 Source 启动的 Task05 `run-001` 使用 knowledge-v4：3 次 fresh Lift 共 21 次物理 provider
  call；第 1 次在 canonical C slot 拒绝，第 2、3 次通过 TypeCheck，formal Gate1 round 2 获得
  normal Replay closure，masked-J mismatch 得到 `J_NECESSARY=PROVED`。Gate2 Source--Replay
  继续以 `max_abs_error=0` 通过，但 accepted J 的三个 unpredicated load 带有 `fill_value=0`，因此
  RefEval fail closed：`E_J unpredicated access must not declare fill`。
- 对该 accepted J 仅在内存中删除三个 orphan fill 后，现有 RefEval 立即成功并识别为
  `dense_functional_v1`。这证明没有新的 DYT operator/lowering 能力缺口，而是 static schema 与
  TypeCheck 漏掉了全局 access invariant：`fill != null => kind=load and predicate != null`。
- knowledge profile 已升级为 v5；Lift schema、canonical grammar、Responses prompt、MiniSWE
  preflight、三份派生规范和 TypeCheck 已统一上述 invariant。TypeCheck 权威拒绝，schema 只提前
  诊断，RefEval 不静默归一化。新增 orphan/non-load fill 正反例；聚焦回归 `136 passed`、全量
  Python 回归 `833 passed`，`compileall` 与 diff check 通过。
- `run-001` 已结束且 Gate2 结果不会回流；下一步提交本 checkpoint，再从 Source 启动全新 v5
  canary。只有新的完整 artifact 自行通过 TypeCheck、Gate1 与 Gate2 三边才完成本任务。
