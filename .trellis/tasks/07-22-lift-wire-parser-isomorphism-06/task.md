# Lift Wire / Parser Isomorphism 06：消除结构接受域漂移后再开 v3 Canary

> 状态：进行中
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
