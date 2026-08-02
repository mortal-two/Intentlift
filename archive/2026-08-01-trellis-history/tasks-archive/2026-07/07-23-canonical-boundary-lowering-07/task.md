# Task 07：J Functional Boundary Projection 与 Production Lowering 消费闭合

> 状态：已完成
>
> 类型：SCRC/IntentIR 系统能力重构；不是新增实验 Gate，也不是继续调 Lift prompt。

## 直接目标

Task06 已建立唯一：

```text
response JSON -> Decode/Elaborate -> J -> Intent MLIR
```

但当前 complete Layer-B access/domain/effect 的消费证明仍私有地位于 `gate2/scalar.py`，CUDA
production analyzer 看到任何显式 access/domain/effect 都直接 `unsupported`。因此两个入口虽然共用
J/MLIR emitter，仍没有共用“哪些 B relation 已被完整消费”的确定性投影。

本任务建立：

```text
J
   -> ProjectFunctionalBoundary
      -> ephemeral FunctionalBoundaryProjection | fail-closed capability reason
           |-> RefEval scalar execution
           `-> production tensor lowering(J,C,target)
```

让 direct full-domain、logical-domain mask erasure 与窄 reduction padding 的证明只存在一份；
RefEval 与 production backend 只能消费该 typed projection，不能分别复制字符串/parser/判定逻辑。

## 所有权边界

- projection 只读取 J，不读取 Source、Replay、π、Gate result、Harness concrete value或 C；
- 它证明 B 可确定性投影到 J 的完整 logical tensor semantics，不改写 J，也不猜遗漏 predicate；
- production lowerer仍单独读取 C/target 做实现选择；projection 本身不包含 schedule；
- unsupported relation 明确拒绝，不能静默忽略 access/effect/alias/update；
- projection 是内部 typed view/evidence，不是第四 artifact、Agent、Gate、hash/commitment。

## 实施步骤

1. 从 `verify/gate2/scalar.py` 提取 canonical domain/predicate/direct-access/effect/output-binding 消费为
   backend-neutral `FunctionalBoundaryProjection`；删除 RefEval 私有 interpretation 权威。
2. RefEval 先取得 projection，再执行同一个 `J -> Intent MLIR -> scalar CPU`。
3. CUDA production analyzer 使用同一 projection；对已经证明等价于完整 logical tensor execution 的
   dense/mask/reduction profile，记录已消费 relation 后进入 Intent MLIR，不再把所有显式 B 一律拒绝。
4. alias/inout、一般 mutable storage、partial/affine/gathered access、未支持 update 继续 fail closed；
   不通过“忽略 B”获得 codegen。
5. production conformance record 明确记录 projection profile 与逐类 consumed relation。

## 验收

1. RefEval 与 CUDA analyzer 不再各自解析 `index_expression`/predicate/domain wire string；
2. 同一 J 在两条路径获得 byte-equivalent projection record；
3. direct full-domain、pointwise mask erasure、sum padding 正例可进入 production MLIR；
4. affine-as-dense、predicate/domain mismatch、orphan fill、effect mismatch、alias/inout/update 反例在
   codegen 前 fail closed；
5. Task04 DYT real artifact 可在不调用 provider的情况下通过 shared boundary projection，并生成
   production Intent MLIR；不重判历史 SCRC 证书；
6. 只运行相关 focused tests与必要 compile check，不重复全量回归或昂贵 Lift/Replay canary。

## 非目标

- 不在本任务实现一般 affine memory codegen、SMT、集合值并发语义或 backend optimizer；
- 不让 production lowering读取 π；
- 不把 RefEval 改成 CUDA production executor；
- 不修改 Gate1/Gate2 接受谓词、Q 或 masked-J 状态；
- 不新增 per-kernel RuleCard、LoweringAgent、hash gate 或实验“冻结”设施。

## 2026-07-23 completion checkpoint

- 新增 backend-neutral `intentir.ir.functional_boundary`。它只从唯一 accepted J 机械派生 ephemeral
  `FunctionalBoundaryProjection`，统一解释 typed domain/predicate/direct axes、access/effect、output
  mapping、broadcast/index-bounds 与窄 reduction padding；不包含或序列化第二份 J；
- `ScalarLoweringRunner` 与 CUDA analyzer 均调用同一个 `project_functional_boundary(J)`；RefEval
  不再以自己的 complete-boundary validator作为运行时权威，CUDA 也删除了对所有显式
  domain/access/effect 的 blanket rejection；
- production analyzer只在完整 projection 成功后进入共享 `emit_j_module(J)`；projection profile、
  structured axes 与逐 relation consumption 进入诊断/conformance metadata。affine-as-dense、
  predicate/domain mismatch、effect mismatch、alias/inout 与未支持 mutable relation 继续在 codegen 前
  fail closed；
- 聚焦正例覆盖 dense functional、pointwise full-domain mask erasure 与单轴 sum padding，负例覆盖
  affine index、effect mismatch 与 inout；RefEval/CUDA 对同一 J 的 projection record完全一致；
- Task04 DYT `run-003/attempt-03` 未调用 provider、未修改 artifact：唯一 J 的 `a1/a4` 均把
  `k1,k0` 消费为 `("k1","k0")`，得到 `logical_domain_mask_erasure_v1`，CUDA analyzer supported，
  与直接 projection record一致，并生成 12 行 production Intent MLIR；
- 相关 canonical-J、Gate2、backend、typed relation、TypeCheck 与 SCRC runner 回归为
  `125 passed`；另完成相关 Python compile check。未运行全量仓库回归，也未重跑 Lift/Replay。
