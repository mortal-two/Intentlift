# Task10 Protocol-v7 Canary：Envelope 闭合，Gate1 未闭合

## 正式结果

唯一 `liger_swiglu_forward` recovery canary 严格按预算运行：

```text
Lift          completed, attempts=1
normal Replay completed, attempts=1
Gate1         no_closure
masked-J      0
retry         0
```

receipt SHA：

```text
8c48dd721c876beda59d4d5c0320a979c49ba5c1d02b6b59555fd085728fa2a3
```

它精确绑定 v7 preflight SHA
`0e7a6df61b2d11cfab4fb6e8092fed453c9a14897a474fe3777ec1654e988e2c` 和 v6 predecessor
receipt SHA `21fb0a2e548f75c619e0339664092defb554dacef5b6a132b54d5db0aed04b9d`。

## 调用证据

### Lift

- request SHA：`578c1d1b06bf58fc227aa568174687be5e92b984f5229577d5b5400a45b16bfe`
- response SHA：`780867ae8a20ee8dbbc613b8ac1c5c663d4de287b9369f770bdaf236580d9645`
- raw response SHA：`f3572b19c2f8382ea231ced6acfc9f7ecba881f62940165d9c288c9adb5ee664`
- usage：input `10998`、output `15979`、reasoning `13586`、total `26977`
- provider status：`completed`，无 incomplete/error

### normal Replay

- request SHA：`b6b43c5d19aab4f2e3dfb3c89aad0ecd1690d46173e90800bcff918930143c2b`
- response SHA：`399325b541a034cb88462589d5b66c8f9a75375a9041c6df637afad50b2db080`
- raw response SHA：`64938caebbfcbc9ecbe51382ce82ad117198b67df3218317a2f7366d5b53aa8f`
- replay code SHA：`99342e6d8e5b684263db74800d69d979bb7fa6b585ff4b34d0f6d766662e62fe`
- usage：input `3367`、output `3868`、reasoning `3614`、total `7235`
- provider status：`completed`，无 incomplete/error

## Task10 合同真实到达

模型原始 Lift response 的 root 恰为：

```text
schema, j, interface, schedule, pi, location_map
```

`j` 恰为：

```text
kernel_id, intent, node_ids, boundary, scopes, storage
```

三个 access 均有非空 `(x0,x1)` index expression；两个 masked load 都保留 predicate 与 fill `0`，
store 保留 predicate；domain 也存在。π 通过完整引用校验，包含 14 instances、19 routes、14 order
edges、3 sharing，且有各三个 `address_base/address_lane/transfer_direct`。这些字段由一次 Lift 直接
生成，不是固定代码 hoist、补全或从 v5 复制。

所以 Task10 的 root ownership 与 access completeness 目标已经完成；Gate1 未闭合不能反推
Task10 失败。

## Gate1 边界

Gate1 score 为 `0.6784552845528454`：

- interface：通过；
- predicate/effect：通过；
- schedule-only：通过；
- op multiset：未闭合，source 有两个 `arith.extsi`，Replay 为 0；
- dataflow：未闭合，80 项差异；
- reduction：不适用。

Replay 的数学计算、mask、load/store 数量和六个 `tt.addptr` 已经一致；两个 integer extension 缺失
造成节点 fingerprint 与大量下游 dataflow edge 连锁变化。但不能因此立即添加 `integer_cast` π tag：
这会继续按单次 TTIR diff 打补丁。

## 新暴露的架构边界

冻结 Replay request 的输入实际为：

```text
schema + mode + J + C + π + harness + frontend_context
```

而新的设计判断是：Replay agent 的构造输入应收敛为 `J+π`；π 可以丰富到描述每个 J op 的
realization 用法，C 则由一个前端通用、非 kernel-specific 的吸收规则应用，ABI/harness 只负责
调用边界，不应整体进入 Replay 语义上下文。

因此下一任务不是 `extsi` capacity，而是 Replay minimal surface / C absorption 的架构回收。
