# Task07 v4 Canary：Lift 在 π 外键闭合处被拒绝

## 结论

Protocol-v4 的唯一 Swiglu canary 已按冻结预算执行，并在 Lift 本地验证阶段终止：

```text
status = lift_failed
diagnostic = SchemaError: pi route instance is unknown
Lift calls = 1
normal Replay calls = 0
masked-J calls = 0
Gate1 runs = 0
```

这不是 Gate1 `no_closure`，也不是现有 π 表达能力不足的证据。LLM 返回的 π 把
Interface/C handle 直接写入 `routes[*].source`，而冻结的 `RealizationWitness` 合同要求
`routes`、`order`、`sharing` 的所有端点均为已声明的 `i*` realization instance。parser
正确地 fail closed；不得自动补 instance、按 `semantic_ref` 猜映射或在同一 v4 epoch 重试。

## 冻结 provenance

- manifest digest：`0da031b95049acf5884e1adb60126253ab5441d6f7de142a0373e2659ab98b18`
- manifest file SHA：`3d8be701069c404cce043779ca953113ec3c8cf2655dd1df6140430cebe800a9`
- preflight SHA：`74862a1402776d68988f3e704b2ceb22d61e89a5849e6a8c1da6c278e585fa30`
- predecessor v3 receipt SHA：`ae69d449cc70d7e812dbd4de4382aa164e837b6bcd8b1246211fc982de8df7c1`
- v4 receipt SHA：`98a3efbc9b9027e2480e68347e053ea62ce7014b6f34db95b707fd39364573d2`
- request SHA：`cbf795fc64739c338cd6afcf9581ad636622d130f53dc987fdec727fda840289`
- raw response SHA：`b6364cf13ffbebfdc33a4a10fb27400165fbc7a73666953f2c81ea4c8ef81b0e`
- extracted response SHA：`289aa3b37fb04c198941f31f76e31a4d05b118049dcf149e4593e8bde8e7db0a`
- provider/model：`responses` / `gpt-5.6-sol`，reasoning `xhigh`
- provider attempts：`1`
- token usage：input `9555`，output `14152`，total `23707`

原始请求、响应和 ledger 均保存在：

```text
artifacts/real_llm_batch_28_v4/recovery/liger_swiglu_forward/
```

## 返回对象实际做对了什么

该响应并非回退成 v3 的错误所有权：

- J 的 `t0/t1/t2` 都保留 `[128, 1024]`，没有再次扁平化为一维；
- `p4` 是 non-C constexpr interface binding，没有进入 C；
- C 只有 `c0=1024` 与 `c1=4`，分别对应 pure schedule constexpr 和 `num_warps`；
- π 显式尝试使用 `p3` 作为三个 base realization 的共享 stride 输入，并使用 `p4/c0`
  区分 logical extent 与 block extent；
- π 还尝试保留 load/compute/store chain 与 sharing。

因此 v4 证明了阶段 0--3 的 J/C/π 所有权信息能到达模型；失败发生在 π graph 的
**命名空间与外键写法**，不是 Swiglu 语义识别或 p3 信息缺失。

## 精确的 wire 违反

响应声明 `i0..i16` 共 17 个 instances、30 条 routes、16 条 order edges 和 4 条 sharing
relations。其中 9 条 route 的 source 不是已声明 `i*`，涉及 7 个 raw handles：

```text
c0, c1, p0, p1, p2, p3, p4
```

典型片段是：

```json
{"source":"p3","target":"i1","slots":[1]}
{"source":"c0","target":"i4","slots":[0]}
{"source":"p4","target":"i5","slots":[1]}
```

而当前合同唯一合法的写法是先声明一个以该 J/C/interface handle 为 `semantic_ref` 的
realization instance，再用这个 instance 的 `i*` ID 作为 route source。`semantic_ref`
是跨层 anchor；`i*` 才是 π 内图节点。二者不能混用。

## 根因分类

1. validator 没有错误：它保持了外键闭合，也没有做拓扑猜测；
2. model 输出确实无效：Responses 请求使用 `strict=false`，JSON Schema pattern 并不构成
   服务端强保证；本地 parser 才是最终权威；
3. prompt/schema 的可生成合同仍有歧义：提示要求 Interface/C handle “occur in pi and route”，
   但没有用一句显式规则说明 route 端点只能是 `i*`，研究文档还使用了容易误读的
   `p* -> access` 简写；
4. 该失败尚未达到增加 `address_base/address_lane` 等 π capacity 的升级门槛，因为我们还没
   得到一个合法 π artifact，更没有执行 Replay/Gate1。

## 下一步边界

下一 epoch 只能做通用的 realization-ID referential-integrity 去歧义：

- 明确 `semantic_ref` 与 `instance.id` 是两个不相交角色；
- 明确 route/order/sharing 的端点集合必须是 `instances[*].id` 的子集；
- 用最小 source-neutral 正反例覆盖 raw `p*/c*/n*` 不能直接作为图端点；
- 改善 validator 的 path-specific 诊断，但绝不修复模型输出；
- 冻结新 wire/manifest/preflight 后，才允许一次绑定本 v4 failed receipt 的 fresh canary。

不得在该阶段加入 composer、RuleCard、kernel-specific rule、source-to-π extractor 或自动 retry。
