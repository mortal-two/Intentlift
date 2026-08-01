# Task 29：TileLang / generated-CUDA 二十对 J 分层差异实验

> 状态：`COMPLETE`（2026-08-01）
> 上游证据：Task28 的 31/31 accepted same-origin pairs、62/62 frontend runs  
> 目标：为论文 §5.1 / Fig.2(a) 生成可复现的跨前端 J correspondence/difference 数据

## 1. 问题

Task28 已证明 31 对 TileLang/CUDA Source 可以各自走完
`TypeCheck -> J-only Replay -> frontend-specific Gate1 -> three-leg Gate2`。本 Task 不重复调用 Replay，
而回答下一层问题：

> 对同一 TileLang case 及其 generated CUDA snapshot，独立 accepted artifact 中哪些 J 结构稳定，
> 哪些差异来自 ABI、runtime extent、typed representation、boundary/effect 或 C？

分析必须保留真实差异；full-J equality 从不作为闭合条件，也不能为了提高相似率修改任一 artifact。

## 2. 分母

- Primary paper cohort：20 个 unique same-origin pair；
- Robustness cohort：Task28 全部 31 个 unique pair；
- 每个 pair 只读取 Task28 `results_current.json` 指向的两份 latest accepted exact artifact；
- 20 是 pair 数，不是 frontend run 数；对应 40 个 accepted frontend artifacts；
- 31-pair robustness 对应 62 个 accepted frontend artifacts。

## 3. Primary-20 冻结规则

选择只读取 stem 名称与 Task28 预先冻结的 cohort membership，不读取 J、artifact、Gate diff、
Replay、运行时间或本 Task 的任何结果。

1. 先固定八个论文语义锚点：pointwise、irregular access、reduction、normalization、attention、
   contraction、typed fused epilogue、multi-stage fusion；
2. 对剩余 stem，计算它对当前七个 cohort 的逆频率覆盖增益
   `sum(1 / (1 + selected_count[g]))`；
3. 每轮选择增益最大者；并列时先选总 membership degree 更高者，再按 stem 字典序；
4. 填充到 20 个 unique stem 后停止；
5. 正式分析器必须重新执行该算法并断言结果与 `cohort20.json` 完全一致。

该规则冻结的是“详细主表放哪 20 对”，不是在闭合前预注册成功率。成功率仍使用 Task28 全部 31 对，
不得把 20/20 写成新的 closure denominator。

## 4. 分层指标

每对至少报告：

1. canonical full-J exact；
2. identity/signature-free J exact；
3. Layer-A typed intent exact、op sequence、op-family set Jaccard；
4. boundary 总体与每个子字段 exact，单列 access/effect/result relation；
5. dimension constraints/value-dimension/dimension relations exact；
6. raw logical signature exact；
7. tensor/pointer carrier 归一后的 typed buffer profile exact；
8. CUDA extra runtime scalar 数、buffer permutation、bool i1/i8 等 typed delta；
9. C slots structure、C values 与 compact pi 是否 exact；
10. canonical J byte size、node/access/effect counts；
11. 每项差异的有限 taxonomy，不生成 free-text 相似度 oracle。

比较器只做 deterministic projection。它可以删除 `kernel_id`、IntentFunction name 等 frontend identity，
并将 boundary 的 node refs 通过 `node_ids` 映射到 canonical body path；不能删除 operator、operand、dtype、
predicate、fill、effect、result relation 或 dimension fact。

## 5. 论文需要的数据

正文最多保留：

- 一个 20-pair aggregate table：semantic correspondence 与真实 delta；
- 一个 compact difference taxonomy：ABI carrier / runtime scalar / typed representation /
  boundary-effect / C；
- Fig.2(a) 选择一对 normalization 或 contraction，展示同语义、不同 frontend realization；
- 31-pair robustness 数字放 appendix/artifact。

工程量另行报告：

- 新增 frontend adapter/runtime/observer LoC；
- kernel-specific production semantic rule 数；
- existing backend 对同一 J feature 的 frontend-specific delta；
- 新 J feature 才产生的 shared consumer delta。

J-diff 不能替代 backend lower evidence；后端是否零修改消费及性能必须引用 backend 分支的独立 receipt。

## 6. 完成条件

- [x] 冻结 20-pair selection 与规则；
- [x] 实现 deterministic pair analyzer；
- [x] 生成 primary-20 与 full-31 JSON/Markdown；
- [x] 逐项核验 exact accepted run provenance；
- [x] 形成论文可直接使用的实验解释与 claim boundary；
- [x] 提交并保持工作区 clean。

## 7. 最终机器结果

- Primary：20 unique pairs / 40 exact accepted frontend artifacts；
- Robustness：31 unique pairs / 62 exact accepted frontend artifacts；
- signature-free semantic core exact：15/20；full corpus 26/31；
- Layer-A op-kind set exact：18/20；full corpus 29/31；
- boundary exact：15/20；full corpus 26/31；
- carrier-normalized typed buffer profile exact：16/20；full corpus 27/31；
- raw full-J 与 raw signature exact 均为 0，因为 frontend identity 与 tensor/pointer ABI 被诚实保留；
- C 与 compact pi：20/20；full corpus 31/31 exact；
- 31 对全部存在 tensor-to-pointer carrier delta；27 对 CUDA ABI 有额外 runtime scalar；
- 五个 semantic-core delta 均有 Source 依据：四个 i1/i8 typed carrier/cast，另一个是 CUDA gather
  显式 bounds/zero-fill 与 TileLang in-bounds direct gather 的真实 boundary 差异；
- 62 个 exact accepted Replay 均为每 run 一次 provider call；full corpus Gate1 median：TileLang
  198.701 s、CUDA 128.694 s；Replay source median：TileLang 690 B、CUDA 916 B。
