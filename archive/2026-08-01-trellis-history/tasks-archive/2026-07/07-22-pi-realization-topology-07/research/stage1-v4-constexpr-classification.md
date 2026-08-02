# Task07 Stage 1：Protocol-v4 constexpr 所有权分类

## 全局合同

旧 manifest 把所有 source `constexpr` 放入 `specialization.schedule`，这等价于把
`constexpr == C` 写死。Protocol-v4 改为一个按源码声明顺序排列的 closed parameter 数组：

```json
{"name": "p", "binding": "runtime", "kind": "tensor", "...": "..."}
{"name": "n", "binding": "constexpr", "class": "binding", "value": 1024}
{"name": "b", "binding": "constexpr", "class": "schedule", "value": 1024}
```

- `runtime`：普通 callable parameter；
- `constexpr/binding`：不属于 C，匿名化成 `p*`。Lift 再依据源码判断其 consequence 属 J，
  或其 representation-only use 属 pi；
- `constexpr/schedule`：在合法选择中不改变 J denotation，匿名化成 `c*` 并进入 pure C。

这里只冻结 28-case invocation evidence。系统没有按名字判断的 runtime registry；投影代码只
消费两个全局 class。完整有序数组避免 LayerNorm 中 `binding -> schedule -> binding` 交错时
被两张 map 重排，也不另建 `parameter_order`/LogicalABI。

## 28-case source audit

下表的 “pure C” 逐一对照冻结 source symbol；同一 case 其余 source constexpr 全为
`binding`。

| case | pure-C source constexpr |
|---|---|
| liger_swiglu_forward | `BLOCK_SIZE` |
| liger_geglu_tanh_forward | `BLOCK_SIZE` |
| liger_rms_norm_forward | `BLOCK_SIZE` |
| liger_softmax_forward | `BLOCK_SIZE` |
| flash_layer_norm_forward | `BLOCK_N` |
| liger_sparsemax_forward | `BLOCK_SIZE`, `num_warps` |
| liger_jsd_forward | `BLOCK_SIZE` |
| xformers_index_select_forward | `BLOCK_SIZE_INDEX`, `BLOCK_SIZE_COL` |
| liger_embedding_forward | `BLOCK_SIZE_M`, `BLOCK_SIZE_N` |
| liger_llama4_rope | `BLOCK_SIZE` |
| liger_multi_token_mask_forward | `BLOCK`, `num_warps` |
| flaggems_cat_copy | `BLOCK_X` |
| cute_async_bulk_copy | none（无 source constexpr） |
| cute_async_tma_copy | none（无 source constexpr） |
| flaggems_matmul | `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `GROUP_M` |
| flaggems_bmm | `TILE_M`, `TILE_N`, `TILE_K`, `GROUP_M`, `DIVISIBLE_M/N/K` |
| flash_attention_forward | `BLOCK_HEADDIM`, `EVEN_M/N/HEADDIM`, `BLOCK_M/N` |
| flaggems_normalized_cumsum | `BLOCK` |
| liger_mhc_sinkhorn_forward | none |
| liger_mhc_sinkhorn_history_backward | none |
| liger_neighborhood_attention_qk | `BLOCK_SIZE_M/N/K`, `num_stages`, `num_warps` |
| xformers_scaled_index_add_forward | `BLOCK_SIZE_INDEX/ROW/COL` |
| liger_embedding_backward | `BLOCK_SIZE_M`, `BLOCK_SIZE_N` |
| triton_racing_overwrite | none |
| flaggems_scatter_add | `BLOCK_SIZE`, `LOOP` |
| flaggems_masked_scatter | `BLOCK_SIZE` |
| flaggems_reshape_and_cache | none |
| flaggems_dropout_forward | `BLOCK` |

关键 non-C 例子：

- `n_cols/embedding_dim/head_dim/N/HC` 改变 logical extent，consequence 属 J；
- `beta/ignore_index/mask_val/IS_CAUSAL/HAS_*` 改变值、predicate 或 effect，属 J；
- Sinkhorn 的 `stride_*` 改变物理 address realization，作为 `binding` 进入 interface/pi，不能
  因为不属 J 就退回 C；
- `DIVISIBLE_*`、`EVEN_*` 是从 shape 推导的合法 specialization guard。在满足其合法性前提
  的 C 选择中只改变 mask/control realization，因此冻结为 schedule；错误 guard 不属于合法 C。

## Swiglu canary specialization

v4 将三个 tensor compile shape 从 `[131072]` 改为 `[128,1024]`，保留 whole-tensor row/col
因子；`n_cols` 是 non-C binding，`BLOCK_SIZE` 与 num_warps 是 C。匿名 Lift view 为：

```text
signature: p0,p1,p2,p3,p4:constexpr,c0:constexpr
bound_constexpr_values: p4=1024
pure C: c0=1024, c1(num_warps)=4
```

离线实编已确认 source TTIR/TTGIR 与原 flat fake allocation 字节相同：TTIR
`92e14ee61cbc4adce23bc6bf6813853d3983ff167732d40b1e2a934ce514f5fb`，TTGIR
`75bc6af49a8d8d5b9e219d0036e60a29304029227f2d867b74aa156a516f4a16`。
