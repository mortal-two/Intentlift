# 能力矩阵

每行回答 **可表示 / LLM 可提取 / 可验证(到什么保证)/ v1 是否纳入**。`v1` 列就是规范第一版对外承诺的能力边界。

> 注:lower 到哪个标准 dialect 是后续阶段的事(见 [../lowering/index.md](../lowering/index.md)),当前矩阵不含该列。

| kernel 模式 | 可表示 | LLM 可提取 | 可验证(保证级) | v1 |
|---|---|---|---|---|
| elementwise / map | ✅ | ✅ 强 | 执行一致(**强**) | ✅ |
| reduce | ✅ | ✅ 强 | 执行一致 + 结合性义务 | ✅ |
| contraction / matmul | ✅ | ✅ 强(dot anchor) | 执行一致(**强**) | ✅ |
| scan / recurrence | ✅(scan / loop+carry) | ⚠️ 中(需识别 carry) | 执行一致 | ✅ |
| layout(transpose/reshape/broadcast) | ✅ | ✅ 强 | 执行一致(强) | ✅ |
| gather / scatter(数据依赖地址) | ✅(地址即 operand) | ✅ 中 | **仅动态**执行一致 | ✅ |
| select / where(数据依赖值) | ✅ | ✅ 强 | 执行一致(强) | ✅ |
| segment-reduce(数据依赖分段) | ✅(segment_ids) | ⚠️ 中 | 动态执行一致 | ✅ |
| atomic-reduce(并发归约) | ✅(归约意图 + 结合性) | ✅ 中 | 动态 + **tolerance**(浮点非结合) | ✅ |
| sync / barrier | ✅(B 层约束,不入 A) | ✅ | N/A(切分合法性) | ✅(识别) |
| **control-flow(数据依赖结构)** | ⚠️ 需 `loop` 逃生舱 | ⚠️ 难 | 动态执行一致 | 🔶 **门控纳入** |
| pointer / aliasing / in-place / 多输出 | ⚠️ 需 destination-passing | ✅ | 执行一致 | ✅ |

## 三元组要求

每新增一行(一种 kernel 模式),必须同时回答三个问题,不能只答"可表示":

- **represent**:IR 里用什么构造表示?
- **extract**:LLM 能从混淆源码里恢复出来吗?难度?
- **verify**:验证到什么保证?数据依赖行**全靠动态执行一致性**,这一点必须写明。

## FlashAttention 自检

FlashAttention 落在 `contraction × 2` + `scan/recurrence` + `softmax(map/reduce)` 这几行的组合上,其 K 循环按 [../ir/layers-abc.md](../ir/layers-abc.md) 因式分解。它是 v1 必须支持的旗舰用例,不得破功。
