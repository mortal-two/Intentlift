# Task 09：退役 RefEval 重复 Layer-B 解释器

> 状态：已归档（原任务已结束；后续工作不恢复本任务）
>
> 类型：新架构收敛清理；不是新增语义能力、实验或兼容层。

## 问题

Task07 已使 RefEval 与 production CUDA 都以
`intentir.ir.functional_boundary.project_functional_boundary(J)` 为唯一 Layer-B projection 权威，但
`verify/gate2/scalar.py` 仍保留约 700 行旧的 domain/predicate/access/effect/reduction validator。
运行路径不再调用它们，测试却仍直接 import 私有 helper。这会造成：

- 活动代码表面上仍有两套 B interpretation；
- 新修改可能误改旧 helper、让测试通过但正式共享 projection 未被覆盖；
- `scalar.py` 继续持有 ContractExpr parser 与 boundary topology 职责，违背 lowering spec；
- 未来维护者可能重新调用旧 `_validate_dense_functional_boundary`，恢复 consumer drift。

## 目标

```text
J -> project_functional_boundary(J)
       |-> ScalarLoweringRunner
       `-> CUDA analyzer/lowering
```

活动 Gate2 scalar 模块只保留：J-only CPU execution request、runtime index-bound input check、共享
projection 调用与 execution result。direct-index、predicate、full-domain、effect/output/reduction
解释全部只存在于 `ir/functional_boundary.py`。

## 实施

1. 删除 `scalar.py` 中旧 `_parse_*`、`_validate_*functional_boundary`、旧 masked-sum validator及其
   只为旧路径存在的 imports/helpers；
2. 测试直接针对 `parse_direct_axes`、`project_predicate_text` 与
   `project_functional_boundary`，不再把 Gate2 私有兼容 wrapper当公共合同；
3. 删除无外部调用的 `scalar_index_bounds`/`scalar_broadcast_axes` compatibility exports；runner继续
   一次计算 projection并复用其 broadcast/index-bounds；
4. 保留 `validate_index_bounds_inputs`，因为它检查真实 runtime challenge，而不是重新解释 J wire；
5. 运行 canonical-J、functional boundary、Gate2、typed relation、CUDA direct-J focused regression。

## 验收

1. `scalar.py` 不再 import ContractExpr/parser、AccessKind/EffectKind 或自行遍历 B topology；
2. `rg` 证明旧 helper名只在历史 archive 中存在，活动 Python 无引用；
3. 正反例仍由共享 projection覆盖，诊断与 Task07 行为不退化；
4. RefEval runner每次只构造一次 projection并传给 emitter；
5. 无 provider、Gate1/2 batch、全量实验或新 compatibility shim。

## 非目标

- 不扩展 functional boundary支持域；
- 不改变 Gate2/Q/SCRC 接受谓词；
- 不新增 J、IR、Agent、Gate、hash 或 sidecar；
- 不重判任何历史 run。
