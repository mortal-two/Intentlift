# Backend J Development Variant v1

`intentir.backend.dev-j-variant.v1` 是后端能力开发协议，不是 SCRC artifact schema，也不改变 accepted J。
它用于把上游真实 shape/dtype invocation 显式写成一个物理分离、可审计但未认证的 J 变体。

## 对象与所有权

```text
certified ProductionSnapshot(base J, optional C)
  + exact JDevEdit[]
  -> canonical re-elaboration
  -> J_dev + optional C_dev
  -> ordinary materialize_j -> MLIR -> target pipeline
```

- shape、boundary dtype、显式 cast 与 domain 属于 J；C value 只可作为单独 edit 修改；
- accepted snapshot 不可原地修改；variant 必须有新的 snapshot/J 或 C digest；
- Python 只创建/验证语义输入与调度编译流程，不生成 target kernel；
- Source/Harness invocation 由独立的 `intentir.backend.dev-workload.v1` 配置拥有，compiler 不读取它。

## Receipt 必需字段

`*.recipe.json` 只是带 authoritative base snapshot digest 与 exact edits 的作者化输入；下列字段属于
`tools/create_backend_j_dev_variant.py` 生成并存入 `corpus/backend-j-dev-v1/` 的 variant receipt，不要求
在 recipe 中重复。consumer 只信生成的 receipt，并对 base 重放 recipe edits。

```text
protocol = intentir.backend.dev-j-variant.v1
base_snapshot_digest
base_j_digest
base_c_digest
base_certification_status
source_kernel_id
shape_config_id
dtype_config_id
edits[]
variant_j
variant_c
variant_j_digest
variant_c_digest
variant_snapshot_digest
certification_status = uncertified_backend_dev
allowed_claims = [backend_capability,
                  target_correctness_on_recorded_QD,
                  performance]
j_unchanged = false
receipt_digest
```

每个 edit 必须包含 exact JSON pointer、`old_value`、`new_value`、分类和非空 upstream evidence。v1 只
允许现有 typed shape/dtype 字段、boundary domain string 与 `/c/values/<slot>` 的替换；不允许修改
kernel identity、B result relation、alias/effect 或任意 body 结构。需要改变 A graph/cast topology 的
dtype 变体必须升级协议，不能用 v1 绕过。

## 验证顺序

1. 核对 authoritative base snapshot/J/C digest；
2. 对 base 深拷贝逐项核对 old value，再应用 edit；
3. 递归删除普通 op 的派生 `type` 并由 canonical `elaborate_j` 重建；`value_if` result type 与 loop
   carry type 是 region interface，必须保留或由 recipe 显式修改；半更新 shape/dtype 必须在 compiler 前失败；
4. canonicalize C 并计算 variant digests；
5. consumer 必须对 authoritative base 重新执行全部 edits，验证 payload 等于 replay 结果；
6. paired dev-workload 必须匹配 kernel/shape/dtype config id，并 digest-pin real Source adapter；
7. Source--RefDev 与每个计时候选的 Target--RefDev/QD 必须先通过，性能结果才有效。

## Claim 边界

J_dev 可以证明当前 compiler 能消费该 typed workload、候选在记录的 Q/D 上正确，以及对应性能；它不
继承 base 的 Gate1/Gate2，不进入 SCRC closure count，也不能写 `SCRC_CERTIFIED`。若该变体要成为完整
SCRC→backend 论文证据，前端必须产生新的 accepted J digest并重新执行 Lift/TypeCheck/Replay/Gate1/Gate2。

当前首批 recipe 包含两类：accepted FP32 J 的自然 shape capability slice；以及 Flash one-pass
LayerNorm 的 BF16 envelope slice。后者复用 base J 已显式存在的 half↔FP32 cast graph，只把全部相关
boundary/signature/cast authorization/type carrier 从 FP16 一致替换为 BF16，并明确区分“上游声明的
BF16 dtype envelope”与 archived concrete FP16 specialization。若 base J 没有这类显式 cast/A graph，
仍必须升级协议或 fresh Lift，不能只改 Harness tensor dtype。

BF16 host transport 使用 `ml_dtypes.bfloat16` 保存真实 16-bit storage/rounding，并以 raw bits 在
NumPy↔Torch、CPU memref ABI 与 portable bundle 之间转换；RefDev、Source 与 target 输出保持 BF16 dtype，
不再用 FP32 数组冒充 BF16。目标计算仍全部来自 MLIR/C++/target compiler。
