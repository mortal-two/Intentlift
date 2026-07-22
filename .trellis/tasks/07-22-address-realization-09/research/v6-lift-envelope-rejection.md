# Task09 v6 Canary：Address Tags 生成成功，但 Artifact Envelope 错位

## 结果边界

protocol-v6 唯一 Swiglu canary 在 Lift parser 边界停止：

```text
Lift          failed after one completed provider response
Replay        0
Gate1         0
masked-J      0
```

本地 diagnostic：

```text
Gate1Artifact is missing required key(s): interface, location_map, pi, schedule
```

这不是 address tag 被拒绝，也不是 Gate1 no-closure；不能在 v6 内重试。

## 冻结 provenance

- v6 receipt SHA：`21fb0a2e548f75c619e0339664092defb554dacef5b6a132b54d5db0aed04b9d`
- request SHA：`cc44b107c1bad9ffa89f9489de539dc078bd81f167d297ec149f568f30cf3d9f`
- response SHA：`4ec0e7f4b8ef75ac0e0794ece900e8cb00e7e7575d2823a0aec74ce59f7d01fb`
- raw response SHA：`60c6242c419c79bbd2215842b8e88af8d93ef4fa0588915601fc669ffa1f0700`
- usage：input `10567`，output `18349`（reasoning `16335`），total `28916`
- provider attempts：1

raw provider response 明确为 `status=completed`、`incomplete_details=null`、`error=null`，所以不是
token truncation、network failure 或 provider refusal。

## 精确结构错误

返回 JSON 可以正常解析，但 root 只有：

```text
[schema, j]
```

本应是 root siblings 的四个 section 被放进 `j`：

```text
j.interface
j.schedule
j.pi
j.location_map
```

也就是说模型少关闭了一层 `j` object。Responses format 使用 `strict=false`，schema 虽要求 root
六个字段，却不构成 provider-side hard guarantee；本地 parser 正确 fail closed，且没有 hoist、
补 bracket 或重新包装 response。

## Address contract 实际到达模型

misnested `j.pi` 本身通过 `RealizationWitness` parser：

- 15 instances、15 routes、10 order edges、2 sharing relations；
- 3 个 `address_base`；
- 3 个 `address_lane`；
- 3 个 `transfer_direct`；
- 每条 access 都形成 base→lane→transfer chain；
- stride instance 共享给三个 base，lane extent instance 共享给三个 lane stages。

因此 Task09 的 enum/schema/shared prompt 投影已被真实模型消费。由于完整 artifact 无效，没有
Replay/Gate1，不能声称 address tags 已经使 compiler topology 闭合。

## 第二个离线诊断

仅为定位、未保存也未用于 Replay：把四个 misplaced sections 在内存中移回 root 后，parser 的
下一个拒绝是：

```text
BoundaryAccess 'a0' requires index_expression
```

v6 J 把 rank-2 accesses 的 `index_expression/predicate/fill` 写成 null，并把 `domains` 留空；v5
合法 J 则保留 `[x0,x1]`、`x1 < p4` 与 load fill 0。故根因不只是一个括号，还是 root/J-boundary
完整合同没有被当前自然语言投影稳定生成。

这项诊断不授权 deterministic repair：hoist 或从 v5/J/source 补 boundary 都会猜测模型本应输出
的语义，并破坏 Lift proposal 的证据意义。

## 下一步边界

下一任务应只闭合两个已经观测到的通用生成合同：

1. root envelope 固定为 `schema,j,interface,schedule,pi,location_map`，`j` 在 `storage` 后结束；
2. rank>0 access 必须有 logical index expression；源码存在 logical mask/fill/domain 时不得因 frozen
   specialization 恰好全覆盖而删除。

应由一个共享 envelope/boundary grammar 投影 schema description 与 Lift prompt；parser 继续拒绝，
不 hoist、不 repair、不增加 kernel rule。Address mechanism 不再扩容，v7 仍只检验相同 Swiglu。
