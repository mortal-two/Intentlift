# Task 12：Single-Owner Transactional Agent Runtime

> 状态：执行中
>
> 类型：Task 11 的独立执行引擎能力任务；不终止、不归档或替代 Task 11。
>
> 范围：同时覆盖 source-aware Lift owner 与 source-blind Replay owner；不做 backend、论文实验或
> 新 artifact 设计。

## 目标

把当前“每个小 shell 动作触发一次全历史 xhigh 请求、多个 fresh writer 轮流改同一 artifact”的
MiniSWE episode，改造成接近 Codex 工作方式的单一 owner：在一个持续上下文中读取、编辑、检查、
局部修正并提交一个完整结果。完成后将该 runtime 作为 Task 11 的执行引擎继续推进剩余 Kernel
闭合。

本任务只改变 Agent 如何工作和如何保存其成果，不改变 SCRC 数学对象或判定：

```text
Lift LLM authors complete (J,C,pi)
  -> external Decode/TypeCheck
  -> source-blind Replay LLM authors complete Replay
  -> Gate1
  -> Gate2
```

## 根因基线

v124 staged pilot 给出了三类互相独立的反例：

- Flash forward planner 已写出 online-softmax/loop 计划，却因文本 submit marker 位置与 turn limit
  组合而丢失；这是 orchestration failure，不是语义反例；
- GroupNorm builder 已取得 CandidateCheck PASS，后置 fresh reviewer 又改坏 tensor handle 与 node
  declaration order；57 个模型调用中的最后 16 个调用是负收益；
- Linear Lift 使用 53 个模型调用、累计约 916k input token；Replay 使用 31 个模型调用、累计约
  932k input token。Lift 52 个实际动作中 24 个是读取/查询、17 个是检查；Replay 31 个动作中
  22 个是读取/查询。单轮输入分别从 708 增长到 44,334、从 1,219 增长到 53,177 token；
- 当前 Responses worker 硬限制每轮恰好一个 bash call，并在下一轮重新发送增长的完整历史。
  staged workflow 只是把这一低效内循环复制三次，没有形成能力提升。

隔离 smoke test 证明本机 Codex CLI 能在一个顶层 session 内完成文件编辑、执行校验和提交；这只
证明 Codex-style interface 可行，不授权直接绕开现有 bubblewrap、Replay privacy 或 provider secret
边界。

## 不可破坏的数学与所有权边界

1. Lift LLM 仍完整 author 每个 J、C、pi 字段；Replay LLM 仍完整 author Replay source。
2. 固定代码可以保存、hash、复制、回滚或提交 **exact LLM-authored bytes**，并运行既有 closed
   checker；不能新增节点、分配/改写 handle、重排声明、更新引用、补类型、补 relation 或选择
   operator。
3. CandidateCheck/ReplayCheck 仍只诊断，不生成 patch，不签发证书；正式 Runner 必须独立重跑
   Decode/TypeCheck/Replay ingress/Gates。
4. Replay 可见性保持为当前 case 的 `J + MaskC(C) + pi + U_f` 与自己的当前 source；不得看到
   Source、Harness、Gate、prior Replay、其他 candidate 或历史 runtime facts。
5. 多 action、checkpoint、rollback 和 session-local semantic ledger 全部 proof-irrelevant，不进入
   `(J,C,pi)`、证书谓词或 production backend 输入。

## 对两份高级 AI 建议的裁决

外部建议原文保存在
[`doc/source-notes/高级ai_agent思考.md`](../../../doc/source-notes/高级ai_agent思考.md)，只作为自省与
设计来源，不具有规范权威；本节记录按五份权威设计和真实日志作出的取舍。

接受：

- 一个 persistent source-aware Lift owner 对最终 artifact 负全责；阶段可作为同一 session 的私有
  工作标签，不能再成为三个 writer 的语义上下文边界；
- critic/reviewer 若存在只能给只读 issue list，不能覆盖 artifact；当前首先使用 owner self-review，
  不额外引入 fresh critic；
- LLM 显式 commit 的 exact checked bytes 可作为 last-known-good；后续 working draft 失败时可以
  exact rollback，预算耗尽时只允许回退到该显式 committed choice；
- 用直接 submit action 替代 stdout marker；准确记录 time-to-first-valid、model/provider calls、
  check calls、commit retention、wall time 与 Gate1 frontier；
- bounded Agentic Lift 与 formal Gate1 round 分离：一次 Lift episode 可多次调用有界 CandidateCheck，
  但最终 artifact 和正式 Gate1 round 仍按现有 Runner 计数。

拒绝或延后：

- 固定 typed builder 自动分配 opaque ID、重排 topological order、批量更新引用或物化类型。按当前
  定稿哲学，这已让固定代码参与 artifact construction，容易重新成为 composer；
- 看到任意 CandidateCheck PASS 后由 harness 擅自挑选该版本。只有 LLM 显式 commit 的 exact
  bytes 才有 conditional-submit 含义；
- 因 runtime 慢而增加 J schema、Kernel 专用 recipe、更多 Gate1 rounds 或盲目扩大 batch；
- 未证明 host filesystem/provider secret/Replay privacy 隔离前，直接把 Codex CLI 当 production
  backend 接入。

## 设计

### P0：Single owner

- `single` Lift 保持一个 source-aware owner 的完整生命周期；`staged` 只保留为 v124 对照，不再作为
  Task 11 默认推进路径；
- owner 可在 `/work` 保存 proof-irrelevant semantic ledger，但最终唯一输出仍是完整
  `/output/candidate.json`；
- Replay 对称保持一个 source-blind owner，最终唯一输出仍是 `/output/replay.source`。

### P1：Exact-byte transaction

提供窄虚拟 action：

```text
CandidateCheck / ReplayCheck
commit exact checked bytes
rollback exact committed bytes
submit current bytes or explicitly committed bytes
```

commit 必须绑定 checker PASS 的同一字节 digest。rollback 只能恢复该 digest；固定代码不能解析后
修改内容。若 episode 以 limit/time/format terminal 结束，只有 owner 已显式 commit 时才可交付
`SubmittedCheckpoint`，并在 receipt 中保留原 terminal reason、digest 与 byte count。

### P2：减少 provider round-trip

- 每个模型响应最多四个独立 bash action；多 action 仅用于互不依赖的只读查询，任何写入、check、
  commit、rollback 或 submit 必须单独串行；
- runtime 在执行前原子检查整个 multi-action response；Lift 仅允许直接调用只读
  `context_query.py/schema_query.py`，Replay 仅允许 `intentir_packet_query`。只要混入一个其他命令，
  整批零执行并返回闭合诊断，不能只靠 prompt 约束；
- Lift 首次 bootstrap 后可同轮读取 Source、Harness、syntax-only outline 与允许的 feedback；Replay
  可同轮读取 packet overview、ABI、graph 与 realization；
- provider 不支持 server-side conversation 时仍保持完整审计历史，但通过减少模型轮数控制累计
  context；不得用截断语义输入换速度；
- 后续只有在 current provider/relay 证明支持且 receipt 可闭合时，才实验 previous-response/session
  transport；不得把 provider 隐式状态变成认证输入。

### P3：Codex-level engine A/B

先以改造后的单 owner MiniSWE 作为 B，v124 staged 作为 immutable A。若 B 仍无法达到显著的
turn/wall-time 改善，再实现隔离的 Codex-style engine C；C 必须复用同一 allowlisted workspace、
checker、预算与 evidence contract，并证明 model tool process 无法读取 host repo、secret 或 Replay
禁区。

固定 pilot：

- Linear：time/carrier 与 Gate1 structural frontier；
- Flash forward：online recurrence 与 protocol-retention；
- GroupNorm：reduction/consumer/effect 与 last-known-good retention。

同模型 `gpt-5.6-sol/xhigh`、同 Source/Harness、单 candidate、单 formal Gate1 round。A 的 v124 evidence
不回写；B/C 必须 fresh run。

## 验收

1. Flash 类合法落盘输出不再因 stdout marker 位置丢失；protocol failure 与 semantic rejection 分类
   分离；
2. GroupNorm 类 PASS candidate 在后续 working draft 失败或预算耗尽后仍能以 owner 显式 commit 的
   exact bytes 交付；receipt 能证明没有 fixed repair；
3. Lift/Replay multi-action、transaction、limit fallback、exact-byte mismatch、额外 output、sandbox 与
   privacy 均有正反例；
4. 三核 B 相比 v124 A 显著减少 model calls、累计 input 与 wall time；至少不降低 first-valid rate、
   TypeCheck 结果或 Gate1 frontier。若不满足，Task 12 不完成，转入 engine C 而不是宣称成功；
5. 正式 SCRC 谓词、artifact wire、TypeCheck、Replay visibility、Gate1/Gate2 均无 acceptance 漂移；
6. Task 12 完成后提交并归档，Task 11 保持活动并立即恢复横向闭合。

## 2026-07-27 实施进度

P0–P2 已形成可执行实现，P3 三核 fresh A/B 尚未开始，因此本任务仍是“执行中”，不能提前宣称
完成：

- Lift 与 Replay 均由一个 session 内的 sole owner 持续构造、自审、检查和交付；旧 staged
  planner/builder/reviewer 仅保留为 v124 对照入口；
- 新增 direct submit、LLM 显式 exact-byte commit、rollback 与 bounded-terminal checkpoint；commit
  只有在同一字节已取得对应 Check PASS 时成立，checker PASS 不会自动替 LLM 选择版本；
- 每个 provider response 最多四个独立只读调用；runtime 对整批 action 做执行前 allowlist 与数量
  检查，混入写/check/submit 或超过四个时整批零执行；
- rollback 使用 host 创建的唯一排他临时文件并原子替换，sandbox 内预置同名 symlink 不能影响被
  保存字节或主机 canary；
- Replay receipt 绑定 multi-action 配置、实际 termination status 与 transaction audit；所有
  checkpoint 仍须经过后续正式 ingress/TypeCheck/Gates；
- 定向及相邻 Runner/Batch 回归共 `191 passed`；尚无三核 B 的 model-call、累计 input、wall-time、
  first-valid、TypeCheck 或 Gate1-frontier 证据。

下一动作固定为 Linear、Flash forward、GroupNorm 的 fresh B pilot；若 B 不满足验收 4，继续 P3
实现隔离 engine C，而不是将 Task 12 标记完成。

## 非目标

- 不实现 typed skeleton、deterministic elaborator、IRBuilder/composer 或 semantic patcher；
- 不修改 A/B/C、J/C/pi schema、IntentOp 或 production backend；
- 不引入 Kernel 名称、模板 artifact、Source-derived fixed repair 或跨 candidate semantic cache；
- 不以更多 token、更多 reviewer、更多 formal Gate1 round 或跨 session 私有思维传递代替能力提升。
