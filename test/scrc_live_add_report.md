# SCRC 真实 add 闭环报告

日期：2026-07-14

## 结论

首个真实 `add` 已完整打通：

```text
Triton 源码
  -> gpt-5.6-sol/xhigh Lift(J/π/C/接口/定位表)
  -> gpt-5.6-sol/xhigh source-blind Replay
  -> Gate1 source/replay 独立编译与 TTIR 比较
  -> masked-J 终检
  -> Gate2 source/replay/production lowering 三方执行
```

最终状态：

- Lift：成功；
- Gate1：`REPLAY_CLOSED`；
- Gate2：`GATE2_PASS`。

本结果不是手写 J、手写 Intent MLIR 或手写 replay。J/π/C 与 replay 均来自
真实 provider 调用；裁决、编译和执行由项目代码完成。

## 为什么此前没有闭合

本轮先复现了两个实际接口问题。

1. 使用错误模型名 `gpt-5.6` 会得到 `unknown provider`；正确名称是
   `gpt-5.6-sol`，推理强度单独传 `xhigh`。最小连通性调用确认请求模型与响应
   模型均为 `gpt-5.6-sol`。
2. 原 lifter 只给模型顶层字段说明，没有完整 Gate1Artifact wire schema。
   真实响应曾发明 `KernelParameter.referenced_dimension`。接入 Responses JSON
   Schema 后，固定协议对象不再发生未知字段漂移。
3. 只传 kernel 函数体无法恢复真实 C：`BLOCK/num_warps/num_stages` 来自
   compile/launch case，而不一定写在函数体。方案本来就规定 Lift Agent 可看
   编译器视图摘要；补入可信 compiler view 后，不再让模型猜 C。

接入 wire schema 后、补 compiler view 前的两次有界响应也给出了清楚证据：

- 第一次把 `boundary.domains` 的值写成对象；本地严格 parser 拒绝，repair
  随后改成字符串；
- 第二次把 constexpr `BLOCK` 错列为 `compiler_slots`，并缺失实际 C 值；这正是
  compiler-view 输入缺失，而不是 IntentIR op 缺失。

补入 compiler view 后曾出现一次 Gate1/Gate2 都通过的结果，但审计发现模型把
`{"tile_size": 256}` 同时写进 `J.intent.layer_c`。这会让具体 C 经 J 进入
rewriter view，违反 `J=A+B、C 单独注入`。该结果被明确废弃，wire schema 与
本地 parser 随后共同要求 `J.intent.layer_c={}`，再从头重跑完整链路。以下数据
全部来自修正后的第二次完整运行。

## 最终配置

- model：`gpt-5.6-sol`；
- reasoning effort：`xhigh`；
- shape：`N=1003`；
- C：`BLOCK=256, num_warps=4, num_stages=2`；
- grid：4 个 program，覆盖非整除尾部 mask；
- alias contract：本 case 三个数组互不别名；
- Lift repair 上限：2；
- Replay repair 上限：2；
- 实际 Lift/Replay 均首轮成功，未消耗 repair。

## Lift 结果

- 实际模型：`gpt-5.6-sol`；
- attempts：1；
- repairs：0；
- response id：
  `resp_0e733c5ceb6bc149016a5662f4a6d081998280c8d4e1c7877f`；
- `J.intent.layer_c`：空对象；具体 C 只存在于独立 schedule。

J 正确表达两个 `f32[N]` 输入的逐元素 add；B 包含两次 masked load、一次
masked store、`i < n_elements` predicate、load fill `0.0`、读写 effect 与输出
映射。π 只使用注册规则 id 和 J/B 引用。C 与可信 compiler view 精确一致。

## Gate1 结果

- 正常 replay 候选数：1；
- replay response id：
  `resp_04243b3fe559a5d4016a56634fa2788191b19354eb4a9b7e9e`；
- replay compile：成功，且 `cache_hit=false`；
- source/replay 总分：1.0；
- interface、op multiset、dataflow、reduction、predicate/effect 五个硬项：
  全部通过且无 difference；
- C soft 项：通过；
- 正常 replay 与源实现结构一致，但代码由 source-blind rewriter 独立生成。

masked-J generation 也由同一模型独立产生，response id 为
`resp_015b0c4080eeba52016a56635f6578819aab638cd1f3d7ef5b`。它生成参数签名后
立即触发静态失败，Triton compile 在 `tl.static_assert(False)` 处终止；因此
masked comparison 不成立，泄漏比为 0，反作弊终检通过。这是期望的“遮掉 J
后无法恢复语义”结果。

## Gate2 结果

三条腿均实际执行成功：

- source Triton：成功；
- replay Triton：成功；
- J -> intent MLIR -> linalg/loops -> LLVM -> mlir-runner：成功。

两两输出：

| 比较 | 通过 | 最大绝对误差 |
|---|---:|---:|
| source ↔ replay | 是 | 0 |
| source ↔ lowering | 是 | `5.0067901611328125e-6` |
| replay ↔ lowering | 是 | `5.0067901611328125e-6` |

在 `rtol=1e-5, atol=1e-6` 下三方一致，机械归属为 `GATE2_PASS`。

## 本轮代码落点

- `python/intentir/extraction/providers/client.py`：Responses 可选
  `text.format`；
- `python/intentir/extraction/lifters/scrc_schema.py`：Gate1Artifact wire
  JSON Schema；
- `python/intentir/extraction/lifters/scrc.py`：可信 compiler view、完整失败
  响应保留与跨字段一致性检查；
- `test/scrc_live_add_e2e.py`：真实单入口端到端执行；
- `python/intentir/verify/gate1/orchestrate.py`：保存 masked generation
  provenance，后续报告可直接审计其模型/response id/digest。

## 边界

这只证明一个真实 add、一个非整除 shape 的闭环。它没有证明 row-sum、softmax、
RMSNorm、matmul 已经通过真实 LLM extraction，也不能外推为数百 Kernel 覆盖。
下一步应沿同一入口推进 row-sum，并由真实失败决定补 IntentIR A/B/C、规则卡、
TTIR 规范化还是 lowering；不预先枚举新特性。
