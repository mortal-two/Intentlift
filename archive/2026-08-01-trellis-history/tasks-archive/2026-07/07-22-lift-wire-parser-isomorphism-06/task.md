# Lift Wire / Parser Isomorphism 06：消除结构接受域漂移后再开 v3 Canary

> 状态：已结束（wire/parser 已闭合；v3 canary 暴露下一层 pi 信息缺口）
> 归档原因：Task06 的结构合同目标已完成，唯一 v3 canary 已耗尽并完整留证；Gate1 缺口转入 Task07
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

## 为什么必须单独成任务

Task05 的 protocol-v2 Swiglu recovery 已证明一个全局契约错误：实际发送的 JSON Schema 对
任意 IntentOp 都允许 nullable `scope_id`，权威 parser 却只允许 `op == "loop"` 声明它。
模型响应满足 wire shape 仍被 parser 拒绝。这不是 kernel 语义缺口，也不能靠重试解决。

Task04/v1 与 Task05/v2 的 response、receipt、manifest、preflight 和 progress 都是冻结实验
证据，必须原样保留。Task06 不改写失败，只消除下一协议的系统性 wire/parser 漂移。

## 最终目标

建立一个简洁、全局、与 kernel 无关的 IntentOp 结构合同，使：

```text
canonical parser-serializable shape  ⊆  wire-accepted shape
wire-accepted structural shape       ⊆  parser structural shape
full parser                           = structural shape + cross-reference/semantic invariants
```

JSON Schema 无法表达的跨节点引用、scope ownership、storage lifetime 等继续由权威 parser
验证；但字段集合、required/forbidden 字段、loop/non-loop 判别等结构条件不得再出现双重真相。

## 架构不变量

- Canonical Intent 仍只有 `J(A+B) + pure C`；pi 在 Intent 外，由 source-aware Lift LLM 直接给出。
- 不引入 composer、RuleCard/Registry、IntentProgram、LogicalABI、kernel-name rule 或 response repair。
- 不降低 parser、Gate1、Gate2；失败 response 不做 post-hoc 删除字段或自动修正。
- 规则只按全局 IntentOp variant/字段合同定义，不按 Swiglu、LayerNorm、Triton/CuTe case 定义。
- 在新 manifest、wire SHA、preflight 提交前，在线 LLM 调用必须保持 0。

## 阶段与提交节奏

### 0. 冻结反例并完成差异审计（0 次 LLM）

- 把 v2 raw response 作为不可修改 conformance 反例；验证 request/response/raw/receipt SHA；
- 不只停在第一个异常，离线枚举该 response 中所有“wire 接受、parser 结构拒绝”的字段条件；
- 明确区分 structural mismatch 与合法的 cross-reference/semantic parser rejection。

完成后提交。

### 1. 单一全局结构合同（0 次 LLM）

- 为 IntentOp 建立 loop/non-loop discriminated shape；只有 loop variant 暴露并要求
  `scope_id/carry/trip/scan/body/yield`；普通 op 不出现这些字段；
- parser 的结构预检与 Responses schema 从同一组 variant/field declarations 构造或核验，
  不再分别手写两套允许字段；
- 保持 operation-specific `attrs` 的现有开放边界，不借机建立 op catalog/composer。

完成后提交。

### 2. 双向 conformance（0 次 LLM）

- 所有 canonical typed fixtures 的 `to_json()` 必须通过 wire schema；
- 对每个 conditional 字段做 positive/negative mutation，证明 schema 与 parser 结构判定一致；
- 覆盖嵌套 loop、required loop fields、non-loop forbidden fields、null/absent 区别；
- v2 失败 response 必须在 provider 之前即可被新 wire schema 拒绝；
- 全量回归通过，且 v1/v2 frozen digest/evidence 不变。

完成后提交。

### 3. 冻结 protocol-v3 与唯一 canary

- 新建完整自包含 v3 manifest/batch id/wire digest/preflight；v1/v2 不修改；
- Swiglu 标记 `protocol_v3_recovery`，绑定 v2 failed receipt SHA；
- 只允许 1 次 Lift；合法后 1 次 normal Replay；normal Gate1 闭合后最多 1 次 masked-J；
- 若再出现全局 wire/parser failure，立即停止并归档，不运行其余 27 项。

成功后归档 Task06，并创建 Task07 执行剩余正式 batch/Gate2/backend/performance。

## 验收标准

1. v2 response 的结构差异清单完整且证据 SHA 可复核；
2. loop/non-loop wire shape 与 parser structural shape 有单一全局合同；
3. 双向 conformance mutation 全通过，不使用 per-kernel 条件；
4. Task04/Task05 evidence 一个字节不改；
5. v3 manifest/preflight 在调用前冻结且 28/28 ready；
6. 最多一次 v3 canary 调用链，完整保存 provenance；
7. 每个阶段独立提交，结束时归档并创建下一工作项。

## 当前第一步

只做阶段 0：以 Task05 原始 response 为 fixture，审计 wire schema、IntentOp 结构预检和完整
parser 的所有分歧。此阶段禁止修改 response、禁止放宽 parser、禁止调用 LLM。

## 阶段 0 完成记录（2026-07-22）

- 已冻结 request/response/raw/receipt 四个 SHA，并在
  `research/v2-wire-parser-audit.md` 记录可复核的分层审计；历史 evidence 未修改。
- 实际 response 的 structural mismatch 完整集合是五个 non-loop `scope_id` 路径；v2 wire
  接受，完整 parser 的 scope ownership 合同拒绝。
- 已明确分离后续 semantic rejection：`q0.definitions` 多出逻辑 output handle `t2`，以及
  `b3..b6` 把普通 SSA 临时值误声明为缺少 semantic write event 的 write-once storage。
- 同时审计出 v2 的 latent variant drift：non-loop loop-fields、incomplete loop、nullable/empty
  loop contract、`id` completeness 与 non-loop body，不再只修 canary 撞到的一项。
- 新增不可修改真实反例测试；阶段 0 定向回归 `10 passed`，在线 LLM 调用新增为 0。

## 阶段 1 完成记录（2026-07-22）

- 新增 `intent_op_contract.py` 作为唯一结构真相；它只识别全局结构判别
  `op == "loop"`，其余任意现在或未来 opcode 一律走同一个 non-loop shape，没有 op catalog。
- Responses JSON Schema、Gate1 strict parser 结构预检和 Lift wire grammar 均由该合同投影；
  删除原 schema/parser 分别维护的 loop 字段集合，没有引入 composer、registry 或 repair。
- canonical `IntentOp.to_json()` 保留合法 loop 的空 `scan`，消除 parse/serialize 后把 required
  字段折叠为 absent 的反向漂移；`id` 也纳入 schema，补齐 canonical node-id 表达。
- v2 raw response 现在会在新的局部 wire/结构边界拒绝，而不是进入 archive ownership 验证后
  才发现五个 non-loop `scope_id`；历史 v1/v2 evidence 未改写，只明确标记 wire epoch 已退役。
- 阶段 1 定向编译与回归：`89 passed`；在线 LLM 调用新增为 0。

下一步进入阶段 2：对真实 canonical fixtures 与确定性 mutation 建立双向接受域 conformance，
然后运行全量回归；在这些离线证据闭合前不冻结 v3、不调用 LLM。

## 阶段 2 完成记录（2026-07-22）

- 对全体 IntentOp 字段的 presence 子集、`loop`/任意 non-loop/null discriminator 做确定性
  穷举，逐项断言 Responses schema 与共享结构 validator 接受域一致；没有 kernel/opcode 特例。
- 覆盖真实 copy/reduce typed artifact、三层 nested loop、canonical node id、required/forbidden、
  unknown key、null/absent、数组唯一性、递归 body、bool/integer、非整值与非有限数 mutation。
- conformance 发现并消除了第二个真实漂移：JSON Schema 按规范接受数学整数 `1.0`，旧 Python
  parser 只接受 `int`。现在有限整值在 wire/parser 两侧一致接受并规范化为 canonical `int`；
  bool、`1.5`、NaN/Inf 仍拒绝，Lift JSON parser 也显式拒绝非 RFC-8259 常量。
- typed `parse -> to_json -> parse` 双轮幂等，新 schema 在完整 parser 前拒绝冻结 v2 反例；
  v1/v2 manifest、receipt、raw response 与 digest 均未修改。
- 阶段 2 定向回归 `128 passed`；全量回归 `715 passed, 4 skipped`；在线 LLM 调用新增为 0。

阶段 0 同时暴露的 scope/storage semantic rejection 也在冻结 v3 前用一个全局、kernel 无关的
archive grammar 关闭了“合同不可见”问题：scope definitions/captures/effects 必须从 J 精确推导，
逻辑 output 不能凭声明多列；普通 SSA 不创建 StorageObject，storage 可以为空，只有 alias、
destination identity 或 loop state 等真实语义身份才创建。该 grammar 由 schema description 与
prompt 共用，不修改 parser、不做 response repair，也没有 Swiglu/opcode 分支；定向回归 `88 passed`。

下一步进入阶段 3：先冻结自包含 protocol-v3 manifest 与 28/28 preflight，再绑定 v2 failed
receipt 执行唯一一次 Swiglu recovery canary。冻结前仍不得调用 LLM。

## 阶段 3A 冻结记录（2026-07-22，调用前）

- 新建完整自包含 `test/real_llm_batch_28_v3.json`，28 个 case 与 v2 完全相同，没有
  `base_manifest` 引用；manifest digest 为
  `5f0b1ee853e2821a299753165145023562df548810468c0110bdbc6895823222`。
- v3 Lift wire 为 `intentir.lift-wire.v3`，同时绑定 provider client、SCRC prompt/parser、schema
  与共享 `intent_op_contract.py`；v1/v2 仍可审计，但 current lock 只接受 v3。
- Gate1 preflight identity 新增 strict parser、typed Intent model 与共享 contract，修复 v2 receipt
  未覆盖权威 parser 的证据缺口；v2 preflight 不能授权 v3。
- v3 recovery lineage 只接受精确的 v2 recovery `lift_failed` receipt：校验 receipt SHA、v2
  manifest digest/file SHA、case/source/provider、attempt class，以及唯一一次失败 Lift、零 Replay。
- 冻结 `artifacts/real_llm_batch_28_v3/preflight.json`，SHA
  `815d43b31635892e67aa3fddcd8f7843fffafa65d3064467b99c9afa5d3744f4`；26/26 Triton、
  2/2 CuTe、28/28 source observer/replay arm ready，`online_llm_calls=0`。
- v3 manifest/preflight/lineage 定向回归 `99 passed`；冻结后全量回归
  `727 passed, 4 skipped`；尚未进行 v3 canary。

下一步先完成冻结后的全量离线回归并提交 3A；只有该提交存在且 worktree 干净后，才执行唯一
一次 `protocol_v3_recovery` Swiglu canary。

## 阶段 3B 唯一 canary 结果（2026-07-22）

- `liger_swiglu_forward` 只消耗一次 v3 Lift：输入 9250、输出 9793（reasoning 8061）、
  总计 19043 tokens，elapsed 178585.940 ms。response 被完整 schema/parser 接受，canonical
  J/C/pi/interface 与 privileged binding 均成功落盘；Task06 的原 wire/parser failure 已关闭。
- 随后只消耗一次 source-blind normal Replay：输入 2331、输出 1635（reasoning 1447）、
  总计 3966 tokens，elapsed 31009.234 ms；Replay 代码通过 validator 并真实编译。
- strict Gate1 返回 `no_closure`，score `0.42689393939393944`。interface、schedule-only、
  reduction 通过；op multiset、dataflow、predicate/effect 不通过。由于 normal 未闭合，masked-J
  调用为 0，符合状态机与预算。
- 根因不是 J 数值语义或 parser：source 保留 `program_id/stride` 的两次 `extsi`、runtime
  row-base、三次 scalar pointer add 再加三次 vector add；Replay 把 specialization 扁平化为
  `program_id*1024 + arange`，完全不使用 `p3`。精确 op delta 是 Replay 多 1 `addi`、1
  `constant`、1 `splat`，少 2 `extsi`、3 `tt.addptr`，并产生 81 个 dataflow 与 6 个
  predicate/effect 差异。
- 当前 J、C、interface 都正确；现有 pi 只有 region/J-instance/routes/order/sharing，没有
  `p3 -> row base -> pointer`、scalar/vector address decomposition、cast width 或 predicate
  placement。因此同一 artifact 同时允许 source 与 Replay 两种 TTIR，source-blind writer 无法
  唯一恢复。严格 Gate1 不应把它们归一成相同。
- exact request/response/raw、artifact、Replay、compiler cache 与 Gate1 receipt 保存在
  `artifacts/real_llm_batch_28_v3/recovery/liger_swiglu_forward/`；聚合统计在 v3
  `progress.json`，审计在 `research/v3-gate1-no-closure-audit.md`。post-canary evidence 回归
  `87 passed`，未重试、未 repair、其余 27 项未调用。

## 结束判定

Task06 的目标——让 IntentOp wire 与 parser 结构接受域同构，并以新 epoch 验证——已经完成；
v3 Lift 的真实成功解析是直接证据。更广的 28-case 目标尚未完成，但当前阻塞已不属于 Task06：
pi 对 source-realization topology 的信息容量不足。继续运行剩余 27 项只会扩散已知缺口，因此
本 epoch 在 canary 后停止，不降低 Gate1、不向 Replay 泄露 source/compiler fingerprint，也不
通过 prompt 猜测或 kernel rule 补洞。

后续 Task07 将只处理最小、匿名、LLM-direct 的 pi realization topology：使 Lift 能表达地址/
predicate/placement 拓扑，Replay 能消费，同时用 masked-J 证明 pi 仍不足以承载完整语义。
