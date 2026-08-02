# Task10 Protocol-v7 Wire Freeze

## 冻结结果

protocol-v7 保持 v6 的 28 个 case、源码 pin、specialization、provider、Gate1/Gate2/performance
protocol 和在线预算不变，只替换 source-aware Lift wire：

```text
manifest schema  intentir.real-llm-batch-28.v7
batch id         tianchenir-formal-real-kernels-protocol-v7-2026-07-22
Lift protocol    llm_direct_pi_source_aware_lift_v7
Lift wire        intentir.lift-wire.v7
```

- manifest canonical digest：`66bf2db6af8b7c26d6ea04d045bc3e50eb1f64dcf667890e801c938621312a34`
- manifest file SHA：`fc9e8902b07da4687b67f1b5c11884e8dd73d3f3785a11b2f1e1e820a2df2c50`
- Lift schema SHA：`f7612a64c587cdcdfd07b754b9f8f881443520a4dc2dafb716aaadd1360444b7`
- Lift prompt SHA：`8f97ce5928725e837e69bf3a22cfea35a2ef8d42439e6344db4c52536115085a`

v7 client identity 首次显式包含 `python/intentir/verify/gate1/model.py`。原因不是扩大 fixed
pipeline，而是 root/J required declarations 与最终 response parser 都由该文件拥有；若不 hash 它，
Lift 接受域可能变化而 wire identity 不变。

## 相对 v6 的单变量

v7 wire 的有效变化只有：

1. model-visible root/J ownership grammar；
2. Boundary access completeness/ownership grammar；
3. schema description 与 Lift prompt 的共享投影；
4. `index_expression` schema 从 nullable required 收紧为 non-empty string；
5. persisted J 的 `scopes/storage` 与既有 schema/canonical output 对齐为 required；
6. parser owner 文件进入 Lift client hash。

没有新增 operator、mechanism、RuleCard、composer、response hoist、字段补全、source recognizer 或
retry。Address mechanism enum 与 v6 相同。

## v7 对 v6 的精确 recovery lineage

v7 的唯一 recovery 必须读取以下原始 receipt，且 SHA 完全匹配：

```text
artifacts/real_llm_batch_28_v6/recovery/liger_swiglu_forward/receipt.json
21fb0a2e548f75c619e0339664092defb554dacef5b6a132b54d5db0aed04b9d
```

runner 还结构化验证：

- v6 manifest digest/file SHA：
  `c3144d17dc7a691dbe7127e8fea2aeb6be526cfb8788239ecf9466b27142212d` /
  `162c86ef069434ae74c993d6003eb127de0b75382910481861491b06e4e84e95`；
- v6 preflight SHA：`ffda048d...52`；
- 恰好一次 `lift/failed/attempts=1`；
- request/response/raw SHA 分别为
  `cc44b107c1bad9ffa89f9489de539dc078bd81f167d297ec149f568f30cf3d9f`、
  `4ec0e7f4b8ef75ac0e0794ece900e8cb00e7e7575d2823a0aec74ce59f7d01fb`、
  `60c6242c419c79bbd2215842b8e88af8d93ef4fa0588915601fc669ffa1f0700`；
- diagnostic 是 Gate1Artifact root missing-key rejection；
- artifacts 为空、Replay/Gate1/masked-J 均未执行；
- v6 lineage 是 `protocol_v6_recovery`，并继续绑定精确 v5 receipt；
- source pin 与 provider identity 和 v7 manifest 一致。

任何一项漂移都在 provider resolution 和 ledger creation 前 fail closed。

## 历史身份规则

历史 epoch 身份由冻结请求字节、记录 hash 和对应 Git 提交共同锚定。当前 Lift 实现只认证当前
v7，不能把 current builder 换一个 `v6` 标签后称为重新认证 v6。因此测试已拆成：

- v6：只检查冻结 manifest/preflight/canary bytes 与历史 outcome；
- v7：使用 current builder 复算并严格匹配 manifest wire identity。

没有建立多版本 runtime registry 或历史 wire dispatcher。

## 离线测试

在冻结 manifest 后：

```text
manifest + formal runner targeted  90 passed
wire/isomorphism targeted          106 passed
online LLM calls                   0
```

共享 grammar 提交后的第一次全量回归为 `851 passed, 4 skipped, 3 failed`；三个失败全是 current
builder 已漂离 frozen v6 wire 的 epoch-lock 断言，没有功能或 parser failure。v7 freeze 将这三条
职责拆成 v6 historical lock 与 v7 current lock；最终全量回归必须重新执行并全绿后才能 preflight。
