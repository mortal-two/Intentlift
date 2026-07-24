# Task 11：SCRC v4 的 28-Kernel 有界并行全闭合

> 状态：执行中
>
> 类型：Task 10 完成后的 fresh SCRC corpus 构建；不是 backend 性能搜索、论文实验或旧 Task 03
> 恢复执行。
>
> 活动性：Task 01–10 已完成并物理归档；当前只有 Task 11 活跃。

## 目标

使用 `artifact.v3 / decode.v3 / typecheck.v7 / candidate-check.v1`、
`j_scalar_cpu_mlir_v9` 与 Gate1 observer v5，为尚未认证的 26 个 Kernel 建立各自独立的 fresh
accepted `(J,C,π)`、Replay、Gate1、Gate2 与预算证据；active registry 的最终 28/28 覆盖由这 26 份
Task 11 evidence 加上 DYT、SwiGLU 两份原版本独立证书组成，后二者不冒充 fresh v3。执行必须真正
支持多个语义族 wall-clock 并行，同时保持每个 case 的证书谓词与 evidence 完全隔离。

稳定 J corpus 完成后，production backend B0/B1 作为独立任务消费这些 J；本任务不要求每闭合一个
Kernel 就紧邻实现其 backend。

## 第一性原理

1. LLM 完整生成 `(J,C,π)`；就 artifact 构造而言，固定代码只做 closed Decode/Elaborate、dependent
   checking、拒绝与 diagnostics，不从 Source、Gate、其他 branch 或历史 evidence 自动补 artifact。
   受信任 runner 另负责执行与比较，但同样没有补写 artifact 的权限。
2. 不同 case 可以并行，但不能混成一次认证；candidate、conversation、Replay、diff、winner、C
   valuation、hidden Gate2 input 和 Harness mutable state 都是 case-local。
3. 横向覆盖优先于围绕一个 Kernel 无限重试。每个执行波至少覆盖多个语义族；通用缺口修复后才
   fresh rerun 受影响的 case。
4. 所有历史 artifact/receipt 只按其落盘版本解释；后续 wire、TypeCheck、RefEval 或 observer 升级
   不静默追认、改写或重判旧 verdict。
5. 一个 case 的 infrastructure failure、NO_CLOSURE 或 unsupported 不取消其他 case，也不改变其他
   case 的证书状态。

## 并发合同

当前 outer batch protocol 为 `intentir.scrc.batch.v3`。v3 只新增
`observer_unsupported` terminal classification，使固定 Source observer 的能力缺口不再被误记为
`harness_invalid`、Replay failure 或一般 `no_closure`；调度、winner、Gate2 barrier 与 evidence
隔离合同不变。历史 batch.v2 report 保持原 protocol/verdict。

设：

- `W`：同时运行的独立 case lifecycle 数；
- `B`：每个 case 的 same-case candidate branch 数；
- `G`：全局 branch/provider/Gate1 in-flight 上限；
- `G2`：同时执行 hidden Gate2 的设备上限。

必须满足 `W >= 1`、`1 <= B <= 5`、`G >= 1`、`G2 >= 1`、`W*B <= G`。外层 scheduler 只改变 wall-clock 调度，不进入
TypeCheck、Gate1、Gate2 或 winner predicate。每个 case：

```text
fresh prepare + isolated evidence root
        |
        v
B independent Lift/Replay/Gate1 branches
        |
        v
pre-Gate2 wave barrier
        |
        v
fixed lexicographic (branch_id, formal_round) winner
        |
        v
one hidden Gate2 under G2 semaphore
```

最终 aggregate 必须按 registry ordinal 稳定排序，与线程完成顺序无关。证据目录必须在 worker
启动前或通过原子、无冲突的 case-local 分配建立；不得使用跨 case 的 `max(existing)+1` 竞态。
每个 case/branch 必须各自新建 provider conversation、runner、writer、buffers 与 evidence state；
只有只读 `ProviderConfig`、registry、source 与 stateless adapter 可以共享。单 case 失败必须结构化
隔离，不得取消或污染其他 case。

same-case 仍先完成 pre-Gate2 wave barrier，再按固定 lexicographic
`(branch_id, formal_round)` 选择一个 winner，只为该 winner 在 `G2` semaphore 下执行唯一 Gate2；
Gate2 失败不 fallback。若没有 winner，只有全部 `B` 个 branch 均带完整 evidence 终止为
`NO_CLOSURE` 时才能报告 `NO_CLOSURE_(B,K)`；若混有 infrastructure-failed、superseded 或
incomplete，aggregate 必须为 `infrastructure_failed` 并保留所有 branch statuses。

live `--run` 默认在 stderr 输出 `intentir.scrc.progress.v1` JSONL observability。除固定
protocol/event envelope 外，其 closed 字段只有 case identity、branch、stage、attempt、formal round、
elapsed 与 closed status，并约每 60 秒为活跃 case/branch heartbeat；绝不输出 Source、artifact、
diff、diagnostic、provider response/conversation 或 hidden input。该旁路 proof-irrelevant，任何写入或
heartbeat 异常都不能改变 verdict；`--no-progress` 只关闭旁路。

## Phase P0：实现真实 case-level bounded parallel

- batch runner 增加显式 `case_workers`，串行路径保持 `W=1`；
- CLI 允许 `--tier` 或显式 case 集在 `W>1` 下运行；禁止无选择地隐式启动 28 项；
- CLI 默认 evidence root 为 `/tmp/intentir-scrc-task11`，同时允许显式覆盖；
- 明示并校验 `W*B <= G`，报告 `W/B/G/G2` 作为 proof-irrelevant audit metadata；
- provider config 可以只读共享，provider conversation/session、runner、writer、buffers 与 evidence
  state 必须每 case/branch 新建；
- Gate2 在 case winner 选定后执行，并受独立 `G2` 限制；Gate2 结果不回流 sibling 或其他 case；
- 单 case 异常结构化隔离，最终 summary 与串行执行具有相同 ordinal 顺序和既有 classification。

### P0 当前实现/测试 checkpoint（2026-07-24；不是 Task 11 完成标记）

- 该 2026-07-24 checkpoint 当时使用 `intentir.scrc.batch.v2`，并携带
  `case_workers/candidates/max_branch_workers/gate2_workers`；当前 live protocol 已按前述窄升 v3，
  历史 report 不回写；
- live CLI 已有显式重复 `--case`/`--tier` 选择、仓库外默认 evidence root、`W*B<=G` fail-closed
  校验、case-level bounded worker pool 与独立 Gate2 semaphore；
- case/branch evidence 已使用原子目录分配，worker 异常收敛为 case-local
  `infrastructure_failed`，aggregate 按 registry ordinal 稳定排序；
- same-case fan-out 保留 pre-Gate2 barrier、固定 winner、唯一 Gate2 与失败不 fallback；无 winner 时
  只有全分支完整 `NO_CLOSURE` 才聚合为 `NO_CLOSURE_(B,K)`，混合失败保留 branch statuses 并聚合为
  `infrastructure_failed`；
- progress side channel 已使用 `intentir.scrc.progress.v1`，覆盖 closed fields、约 60 秒 heartbeat、
  `--no-progress` 与 broken-stderr non-fatal 正反例，且不污染最终 stdout JSON；
- P0 定向回归命令
  `pytest -q python/tests/test_scrc_batch.py python/tests/test_scrc_batch_cli.py python/tests/test_scrc_workload_run.py`
  当前结果为 `42 passed`；
- 该 checkpoint 尚未产生 Task 11 所需的 26 份 fresh `SCRC_CERTIFIED` evidence，也不能与 DYT、
  SwiGLU 两份原版本证书合成最终 28/28 覆盖，因此 Task 11 继续保持活跃，不能归档或标记完成。

## Phase P1：横向 canary 与 B1 terminal checkpoint

### 旧 v2 波的已完成诊断 checkpoint

旧 `public_frontend_context_v2` 波以 `W=4,B=1`、MiniSWE xhigh 运行 Softmax、RMSNorm、attention row
reduction 与 linear contraction，四项均形成完整 `NO_CLOSURE_(1,K)` evidence；所有正式 candidate
均通过 TypeCheck。RMSNorm、Softmax 与 attention row reduction 的共同根因已经收敛：π 中 pointer
carrier 直接进入 `address_base`，v2 Replay 却把 base/lane 错误合成整数 offset，导致 `tt.addptr` 与
pointer-carrier dataflow 丢失。通用修复已形成 `public_frontend_context_v3` 与
`intentir.lift-knowledge.v11`；旧 v2 verdict 仍保持 `NO_CLOSURE`，不得用新协议追认。

### v5 / CandidateCheck / adapter checkpoint（2026-07-24；不是 wave 终态）

- TypeCheck 因接受语义收紧而从 v4 窄升 v5：已知 `reduce` 必须显式使用合法
  `fn + axes + assoc`；已知 binary `contraction` 只接受 canonical `subscripts`，并在 typed J 可
  机械判定范围检查 operand rank、label extent 与 result shape。`kind` 与
  `lhs_contract_axes/rhs_contract_axes` 等别名、缺失/冲突字段均 fail early；fixed code 不翻译或
  补齐 artifact。可选 boolean `keepdims` 继续受既有 declared result shape 校验；
- `candidate-check.v1` 的 acceptance path 与 closed response envelope 不变。同一权威 strict
  Decode/TypeCheck 已经判定的独立机械错误按声明/字段稳定顺序一次展开，统一最多 20 项、每段文本
  最多 2048 字符；checker 只诊断，不补默认值、不改别名、不删除冲突、不返回 repaired artifact。
  前置对象尚未合法解析时，不猜测依赖其修复结果的 dependent error；
- `intentir.triton.fixed-c-source-slice.v3` 只从显式 source roots 解析实际引用的唯一顶层、无 decorator
  同步 plain helper 及其依赖闭包；只允许 stdlib/Triton absolute import、受审计 helper 与 static
  literal，拒绝第三方依赖、dynamic initializer、`global/nonlocal`、local import、decorator、
  `async/await`、`yield`、歧义/冲突/未解析 binding。它直接读 source file 而不 import package，且没有
  xFormers、`libdevice_find`、`rsqrt` 或 Kernel-family 特判。v3 另按精确 `VAR_ARGS_ARRAY` marker 与
  显式 bounded arity 展开 positional/local arrays 和 `range(len(array))`，残留/动态/foreign marker
  fail closed，不执行外部 `unroll_varargs`/`exec`/monkey patch；
- `triton-workload-adapter.v3` 当前 declared 数为 28：原六项、full FlashAttention fwd/bwd、
  fused-add RMSNorm、Sparsemax、one-column-block backward，以及 Qwen2-VL MRoPE、GroupNorm、JSD、
  CrossEntropy、GEGLU、PolyNorm、Llama4 RoPE、FlashAttention LayerNorm、RoPE、LayerNorm、KLDiv、
  TVD、xFormers RMSNorm/RMSNorm-add 与三个 SplitK entry；ordinal 26 的 8 个 optional carrier 均以
  真正 `None` 贯穿 compiler/Gate2 并映射为 J opaque handle，绝无 dummy pointer；ordinal 26/27/28 的
  vararg arity 分别为 1/2/2，expanded ABI 分别为 82/38/51 参数。全部 28 项已通过 static prepare 与
  真实 Triton CUDA frontend compile；fresh MiniSWE Lift knowledge 因 `none -> opaque(dtype=none)` 及
  canonical contraction/direct-vs-decomposition multiplicity 构造提示当前升为 v13，旧 receipt 保持
  原版本；这只是 Harness readiness，不是 SCRC verdict；
- 当前 Replay packet 为 `intentir.scrc.replay-input.v1`、Triton public context 为 v5。它继承 carrier
  routing，并新增 source-neutral contraction direct-`tl.dot` support boundary 与同一 `d*` 的 program
  axis identity；重复 instance 仍分别 materialize，但不得仅因 multiplicity 制造 axis 1。旧 linear
  三个 artifact 的 contraction aliases 已由 v5 实测在 Replay 前一致拒绝，必须 fresh Lift；
- Lift/preflight v13 已明确：compiler-visible loop-carried semantic recurrence 必须由 J structured `loop`
  完整声明 trip/scan/carry/init/update/yield/result，π 不能补公式；一个只定义一次并被多 access 使用的
  Source SSA coordinate 必须是一个 producer 加 routed consumers/one-producer sharing，不能按 use 次数
  复制 instance；
- `intentir.scrc.gate1-diagnostic-recheck.v1` 可对 parent evidence 中固定 artifact 执行 current TypeCheck、
  一个 fresh normal Replay 与 Gate1，并写入独立 `recheck-NNN`；它不 Lift、不跑 masked-J/Gate2、不参加
  winner，且 `changes_prior_verdict=false`。provider 前必须精确绑定 parent case/request/source
  projection/source bytes 与 artifact digest；错配或缺字段 fail closed；
- 所有已落盘 evidence 按启动时的 version receipt 保存，不被 v5 静默追认；需要 v5 结论的 case
  必须 fresh rerun。下方 B1 只记录已终止的旧 receipt 波，不把它改写成当前协议结果。

### B1 immutable terminal checkpoint（2026-07-25）

B1 使用六个显式 `--case`、`W=6,B=1,G=6,G2=1`、MiniSWE xhigh 与 `--skip-masked-j`。该进程在
`typecheck.v4 / RefEval v8 / public_frontend_context_v3 / intentir.lift-knowledge.v11` receipt 下启动，
不是当前 `typecheck.v7 / U_f v6 / replay-input.v1 / lift-knowledge.v15` 结果。六个 case-local run 均已
写出 terminal `report.json` 与 `SUMMARY.md`；outer CLI 正常终止并写出 aggregate report
`/tmp/intentir-scrc-task11/wave-20260724-six-v4-v8-ufv3-xhigh-01-report.json`（进程 exit 2 只反映该批次
存在非通过分类，不表示 evidence 中断）。`B=1` 的失败只表示本次诊断宽度，不能写成
`NO_CLOSURE_(5,K)`。

统一 evidence root：
`/tmp/intentir-scrc-task11/wave-20260724-six-v4-v8-ufv3-xhigh-01`。

| ordinal / case | immutable classification | 终态与证据 |
|---|---|---|
| 2 `high_performance:liger_rms_norm` | `no_closure` | 3 个 formal round 均 TypeCheck PASS、Gate1 `no_closure`，`liger_rms_norm/run-001/report.json` |
| 8 `high_performance:liger_softmax` | `gate2_failed` | formal round 2 Gate1 `replay_closed`；Gate2 J leg 因旧 v4 接受的 noncanonical reduce attrs 报 `KeyError: fn`，`liger_softmax/run-001/report.json` |
| 18 `high_performance:flash_attn.ops.triton.linear.kernel_fwd` | `replay_compile_failed` | 3 个 TypeCheck PASS formal rounds；terminal Replay compile failure，`flash_attn_ops_triton_linear_kernel_fwd/run-001/report.json` |
| 19 `high_performance:flash_attn.flash_attn_triton._fwd_kernel` | `replay_response_rejected` | 3 个 TypeCheck PASS formal rounds；terminal Replay provider response rejection，`flash_attn_flash_attn_triton_fwd_kernel/run-001/report.json` |
| 21 `high_performance:flash_attn.flash_attn_triton._bwd_kernel` | `lift_response_rejected` | 3 个 Lift invocation 均未提交可接受 artifact；TypeCheck/Replay/Gate1/Gate2 均 0，`flash_attn_flash_attn_triton_bwd_kernel/run-001/report.json` |
| 22 `high_performance:flash_attn.flash_attn_triton._bwd_preprocess_do_o_dot` | `no_closure` | 1 次 pre-Replay rejection 后 3 个 TypeCheck PASS formal rounds，`flash_attn_flash_attn_triton_bwd_preprocess_do_o_dot/run-001/report.json` |

因此 B1 只回答旧 receipt 下的横向 failure classes。凡要对当前 `typecheck.v7 / U_f v6` 作结论，必须
使用 v15 knowledge 重新 fresh Lift/Replay；不得用 fixed recheck 或新 checker 追认 B1 artifact。

两个固定 artifact 的 diagnostic recheck 另行保留，但都不改变上表：

- full FlashAttention forward attempt-03：
  `/tmp/intentir-scrc-task11/fixed-gate1-rechecks-fwd-20260724/flash_attn_flash_attn_triton_fwd_kernel/recheck-001`。
  artifact digest 不变、TypeCheck PASS、fresh Replay compile PASS，Gate1 仍 `no_closure`；interface、
  reduction、predicate/effect 与 schedule-only PASS，op multiset 8 项、dataflow 241 项失败。证据表明 J
  只给 final/global attention，而 Source 是 online loop-carried recurrence；π 不能补 init/update/yield，
  contraction result orientation 也必须由下一 fresh Lift 在 J 中闭合。
- attention row reduction attempt-04：
  `/tmp/intentir-scrc-task11/fixed-gate1-rechecks-attention-zero-v5-20260725/flash_attn_flash_attn_triton_bwd_preprocess_do_o_dot/recheck-001`。
  artifact digest 不变、TypeCheck PASS；该次后来撤回的临时 heuristic 让 Replay 不再输出额外
  `where/select`，但随后严格审查证明该 heuristic 不 sound：artifact 只有 symbolic `x1`、literal
  `hi=31` 与 static output `64`，J/π 内没有 `x1+31=64` 的 dependent equality，因而不能授权省略 pad。
  该 packet 扩展已经撤回，当前协议回到 `replay-input.v1`。这份 recheck 只保留为 soundness 反例；
  它同时观察到旧 π 把一次 coordinate producer 拆成多个未共享 instances，Replay 用代数恒等式显式
  满足，下一 fresh Lift 必须自行给出可检查的 J shape/pad relation 与 one-producer sharing。

两项 recheck 都明确 `diagnostic_only=true`、`enters_gate2=false`、
`changes_prior_verdict=false`，不得作为当前协议的 SCRC verdict。

### B3 immutable terminal checkpoint / v6+v14 response（2026-07-25）

B3 使用三个显式 case、`W=3,B=3,G=9,G2=1`、MiniSWE xhigh 与
`--skip-masked-j`，并已完整终止。aggregate report 为
`/tmp/intentir-scrc-task11/wave-20260725-b3-softmax-linear-fullbwd-v5-ufv5-kv13-xhigh-01-report.json`，
统一 evidence root 为同名去掉 `-report.json` 的目录。该进程以
`typecheck.v5 / public_frontend_context_v5 / lift-knowledge.v13` 启动；以下 verdict 与预算均保持
immutable，后续 v6/v14 只能通过 fresh run 产生新结论，不能回写本批次。

| ordinal / case | immutable classification | 本批次完整预算与终态 |
|---|---|---|
| 8 `high_performance:liger_softmax` | `gate2_failed` | 3 个 branch 均获 Gate1 closure；合计 6 Lift / 74 physical / 6 TypeCheck / 6 Replay / 6 Gate1，唯一 winner Gate2 执行 1 次并在 J functional leg 失败 |
| 18 `high_performance:flash_attn.ops.triton.linear.kernel_fwd` | `no_closure` | 3 个 branch 均 `no_closure`；合计 9 Lift / 144 physical，TypeCheck、Replay、Gate1、Gate2 全部 0 次 |
| 21 `high_performance:flash_attn.flash_attn_triton._bwd_kernel` | `no_closure` | 3 个 branch 均 `no_closure`；合计 9 Lift / 144 physical，TypeCheck、Replay、Gate1、Gate2 全部 0 次 |

Linear 与 full-bwd 的 `B=3` 失败都发生在 worker 内，不是 contraction op、Replay、U_f 或 observer
失败。18 个复杂 Lift episode 全部恰好使用旧上限 16 turns。Linear 的最后可见机械层依次覆盖
loop-state storage、scope capture/definition、symbolic dimension 连续性、rank 与 J body producer
result type；full-bwd 的最后可见层覆盖 logical tensor handle 完整声明、Harness handle 保持、C slot
纯 scheduling role、symbolic dimension 连续性与 op carrier type 冲突。多个 episode 已从首次最多
20 条 closed diagnostics 收敛到最后 1–2 条，但随即达到 4-check/16-turn 边界；其中有候选在最后
修正后才提交，却因 ordinary CandidateCheck 预算已经耗尽而只得到 `check_budget_exhausted`。

因此此处先修复通用 bounded episode，而不直接把相同截断复制成 B5：B 增加的是独立候选宽度，
不能替代单个候选跨 dependent construction 层的诊断深度。v14 将 episode 改为下述 24 turns、
7 shared + 1 submission reserve，并保留 source-free rule/path 跨相邻 fresh attempt 摘要；完成该通用
修正后，再对未闭合 case 使用 `B<=5` 才是在增加生成多样性。这里没有 typed skeleton、自动
elaboration、artifact patch 或 Kernel 专用 prompt。

- B3 的 Softmax fanout 在
  `/tmp/intentir-scrc-task11/wave-20260725-b3-softmax-linear-fullbwd-v5-ufv5-kv13-xhigh-01/liger_softmax/fanout-001`
  以 `typecheck.v5 / public_frontend_context_v5 / lift-knowledge.v13` 启动。三个 branch 都得到
  Gate1 closure，固定 winner 是 branch 1/formal round 1；唯一 Gate2 的 J leg 随后以
  `ValueError: functional predicate differs from the full logical domain` 失败。该 v5 fanout、winner 与
  `GATE2_FAILED` 保持不可变，不能用新 checker 追认；
- winner J 的 direct accesses 都是 `[k0,k1]`，padding predicate 都是 `k1 < x0`，但
  `boundary.domains` 只声明 `0 <= k0 and k0 < 2`，没有声明 `k1` 的 logical domain。这不是让
  RefEval 从 tensor shape/C/runtime 补一个 domain 的许可，而是完整 J 内部已经可机械判断的
  functional-boundary 缺口；
- fresh TypeCheck 因接受集合收紧窄升为 v6，并与 CandidateCheck 共享同一个 J-only closure core。
  对上述固定 artifact 的只读回放稳定返回两条 `functional_boundary_incomplete`，路径分别为
  `j.boundary.accesses[0].predicate` 与 `[1].predicate`；因此下一 fresh run 会在 Replay/Gate1/Gate2 前
  fail early，由 Lift LLM 完整修订 J。fixed code 不增补 domain、不改 predicate、不碰 Q/RefEval/emitter；
- v6 只覆盖可识别的零起点矩形 logical-domain/direct-access/padding profile；general affine/交叉轴
  logical predicate 不会因为当前 RefEval 未实现而被判为 J 非法。合法 masked/padded
  predicate/fill/effect 正例仍通过；
- fresh MiniSWE receipt 合并升为 `intentir.lift-knowledge.v14`。worker 最多 24 model turns；
  CandidateCheck 使用 7 次普通 shared allowance（显式调用与提前 submission 共用）和 1 次仅
  submission 可用的 reserve，总 evaluation 不超过 8。无 candidate worker 只把最后一次实际 checker
  的最多 20 条 `rule_code/path` 带给同 branch 下一 fresh Lift，不带 observed value、patch、
  Source/Gate/hidden/sibling 信息；submission `PASS` 仍由外部 Runner 以 v6 重检。

### v6/v14 Softmax fresh slice / v7 response（2026-07-25）

- fresh evidence 位于
  `/tmp/intentir-scrc-task11/wave-20260725-b3-v6-v14-frontier5-xhigh-01/liger_softmax/fanout-001`。
  三个 branch 再次全部 Gate1 closure，固定 winner 为 branch 1/formal round 3；Source–Replay 在唯一
  Gate2 case 上通过且误差为 0，但 J leg 以
  `functional padded reduction requires one input, output, and contract` 返回
  `REFERENCE_UNSUPPORTED`。该 v6/v14 run、winner 与 verdict 保持 immutable；
- winner 完整 J 把两个只服务 address realization 的 public `i64` runtime scalar p1/p3 分别绑定成
  rank-0 logical inputs t0/t2，但 t0/t2 不在任何 J body/output dependency、boundary predicate/index/
  effect 或 shape relation 中；它们只由 π realization route 消费。这扩大了 RefEval logical-input
  ABI，且不是 RefEval 应从 Source 名称/参数位置猜测并忽略的字段；
- TypeCheck 因接受语义再次收紧窄升 v7。v7 只对 rank-0 signature logical input 做 source-free
  liveness：必须到达 observable output dependency，或有显式 boundary/shape semantic use；direct
  identity、`returns`/`output_mapping` 与 boundary-only 正例均保留，一般 tensor liveness 不作 claim。
  对固定 winner bytes 的只读检查稳定返回两条 `logical_input_unused`，路径正是
  `j.signature.parameters[1].logical_ref` 与 `[3].logical_ref`；
- representation-only runtime scalar 的合法显式形态是
  `kind=scalar,mode=value,rank=0,logical_ref=null`：不创建 J input、不进入 J semantics，但 p* 可继续
  由 π 引用。Lift 必须根据 Source use 写出该分类，fixed code 不自动解绑、不删 input、不重编号
  tensor。一次仅用于正例验证的内存中完整 J 修订（移除死 inputs、显式清空 p1/p3 logical_ref，并由
  作者同步重编号所有 J handles）通过 v7；它不写回或追认旧 run；
- prompt/preflight 因构造合同改变升为 `intentir.lift-knowledge.v15`。旧 v6/v14 receipt 不被 v7/v15
  静默重判；Softmax 必须 fresh Lift/Replay/Gates。

### v6/v14 Softmax immutable Gate2 / RefEval v9 response（2026-07-25）

- 上述 v6/v14 Softmax winner 的唯一 Gate2 保持 `REFERENCE_UNSUPPORTED`：Source 与 Replay 都执行
  成功，Source–Replay 在外部 tolerance relation 下误差为 0；J leg 在读取任何 hidden input 数值前由
  旧 projector 以 `functional padded reduction requires one input, output, and contract` 拒绝。该 run、
  winner、v8 receipt 与 verdict 不回写；
- winner J 已显式声明两个完整零起点 logical domains、同一 tail predicate 的 load/store、`-inf`
  fill、max/sum 两个中性 reduction contracts，以及两条非尾随 broadcast。J-only 审查证明真正缺口是
  RefEval 仍把任意带 predicate 的 reduction graph 强行送入旧 single-input/single-output/single-sum
  profile，而不是 J 缺 op、Q tolerance、Source/Replay 数值或 Softmax 专用语义；
- shared projection 升为 `intentir.j-functional-boundary.v2`，scalar consumer 升为
  `j_scalar_cpu_mlir_v9`。v9 不按 Kernel 名称或固定 Softmax 公式匹配，而要求：每个 predicated direct
  access 的全部 axis 有完整 logical domain；仍携带 masked axis 的 access 使用相同 full-domain
  predicate且 load 有显式 fill；每个 logical reduce 恰有一个中性 `masked_contribution` contract；
  每个 pointwise operand 的普通 trailing 或显式 non-trailing broadcast 闭合。满足这些条件后，唯一 J
  直接交给既有 Intent MLIR scalar executor，因此通用支持 max+sum、多 reduction 与普通
  pointwise/broadcast DAG；
- J-only access/dataflow 若证明 masked axis 由 `mean` 消去，因当前 boundary 没有 logical count
  contract 继续 unsupported；缺 domain/fill/reduction/
  broadcast、非中性 contribution、update/alias/mutable relation 继续 fail closed。projector 不读取
  Source、C、π、runtime/hidden input，不补造字段，也不改 Q；
- active `typecheck.v7` 显式复用 `functional_reduction_padding_issues(J)`，把上述可机械证明的欠约束
  映射为独立 `padded_mean_count_unproven`。dense mean、无关轴 mask 与未知-op 传播不 blanket 拒绝；
  checker 不改图，v15 guidance 要求 Lift 将 Source 已有的 masked sum + explicit division 保持为
  两阶段。新增 padded-mean 拒绝、dense mean 正例与无关 mask 正例均进入 TypeCheck 定向回归；
- v9 soundness 复查补上两个 consumer-only fail-closed 边界：任何 guard-bearing access predicate 都
  unsupported，避免把 `flag` 与 `not flag` 折叠后仅凭 dead operand membership 擦除 mask；已有 masked
  provenance 一旦进入 contraction 或其他未建模 op，也在该 op 处 unsupported，不能静默停止传播后
  放行下游 mean。两者都不扩大 TypeCheck 的 J legality，不增加 Kernel 特判或 fixed artifact 修复；
- 当前固定 winner J 仅以公开自造数组做 J-only replay，既有 scalar MLIR 路径成功执行，和独立 NumPy
  max/sub/exp/sum/div 组合的最大误差为 `5.960464477539063e-08`。新增稳定通用 Softmax 正例及
  domain/fill/broadcast/reduction neutral 负例；这只是 v9 consumer 回归，不是对旧 Gate2 的追认。
  Softmax 的当前 SCRC 结论仍需 v7/v15/v9 fresh Lift/Replay/Gates。

### frontier-five B3 immutable terminal checkpoint / iteration response（2026-07-25）

本轮不是单 B：五个显式 case 均使用 `B=3`，evidence root 为
`/tmp/intentir-scrc-task11/wave-20260725-b3-v6-v14-frontier5-xhigh-01`，全部 branch 已终止且当前无
provider 进程。该 run 以 `artifact.v2/decode.v2/typecheck.v6/public_frontend_context_v5/
lift-knowledge.v14/j_scalar_cpu_mlir_v8` 启动，以下对象、winner 与 verdict 均保持 immutable：

| case | B=3 终态 | 一手边界 |
|---|---|---|
| `liger_rms_norm` | branch 2 / round 2 Gate1 closure；Gate2 `REFERENCE_UNSUPPORTED`；Lift/physical=`8/118`，TC/Replay/Gate1=`8/8/8`，Gate2=`1` | Source–Replay 两个 output 均误差 0；旧 J projector 把多 input/output、mean→rsqrt→affine 图错误限制为 single padded reduction，J 两腿未执行 |
| `liger_softmax` | 三 branch 均 Gate1 closure，winner branch 1 / round 3；Gate2 `REFERENCE_UNSUPPORTED`；Lift/physical=`6/103`，TC/Replay/Gate1=`6/6/6`，Gate2=`1` | Source–Replay 误差 0；旧 projector 同样在 J execution 前拒绝 max+sum/broadcast DAG |
| `linear.kernel_fwd` | 三 branch `NO_CLOSURE`；Lift/physical=`10/224`，TC/Replay/Gate1=`3/3/3`，Gate2=`0` | 最佳 branch 的 Source/Replay 只差物理 loop：Source 为 `[0,K)` step `BLOCK_K`，Replay 为 `[0,ceil_div(K,BLOCK_K))` step 1；旧 `(J,C,π)` 对 induction basis 欠定 |
| `_bwd_preprocess_do_o_dot` | branch 3 / round 2 Gate1 closure；outer `INFRASTRUCTURE_FAILED`；Lift/physical=`9/137`，TC/Replay/Gate1=`8/8/8`，Gate2 case=`0` | winner 选定后的 Gate2 adapter binding 发现 J 把 input seq extent 与 padded Delta output extent 错绑同一 `x1`，并把 rounded extent 当 rank-0 data；没有执行 Gate2 case，这不是 Q/数值或 provider 基础设施 |
| full `_bwd_kernel` | 三 branch `NO_CLOSURE`；Lift/physical=`12/266`，TC/Replay/Gate1=`6/6/6`，Gate2=`0` | 旧 observer v4 报告 broad interface/op/dataflow/predicate diff；后续审计证明其只观察 public `tt.call`、漏掉 private helper body，因此这些 diff 不能作为精确归因，旧 NO_CLOSURE verdict 仍保持 |

RMS/Softmax 的 shared RefEval v9 与 rank-0 TypeCheck v7 响应已在上节落地，不回写本 run。Linear
反例则揭示独立的表达缺口：J 拥有 logical loop，C 拥有 tile value，但旧 π 没有声明 source physical
induction variable 的单位。当前 wire 因此窄升 `artifact.v3/decode.v3`，加入 required closed：

```text
pi.iterations[*] = {region, loop_instance, tile_instance, induction_basis}
induction_basis = extent_step | tile_ordinal
```

前者全局表示 `[0,Trip(J.loop))` 以匹配 C tile 为 step；后者表示
`[0,ceil_div(Trip,Tile))`、step 1。entry 不携带公式、concrete value、Source/API 或 operator；Lift 完整
选择，CandidateCheck/TypeCheck 只验证，所有 frontend 由 `public_frontend_context_v6` 对同一 denotation
作公开语法映射。每个 C-tiled J-loop/slot pair 至少一条 relation；同 pair 的多个真实 physical region
可各有一条，完全被 Lift 遗漏的 Source region 仍只能由 Gate1 检出。因此这不是 typed skeleton、自动
构造或 fixed patch，也不改变 production 只读 J+C、禁止读取 π 的边界。

`artifact.v1/v2` parser/to-json/digest 保持历史 wire，不接受或暗加空 `iterations`；已闭合的 DYT 与
SwiGLU receipt 原样保留，不因 v3 升级重跑，也不得被称为 fresh v3。Replay packet envelope 与 Gate1
report 未增加字段，仍分别是 `replay-input.v1` 与既有 report version；当前合并版本为
`artifact.v3 / decode.v3 / typecheck.v7 / candidate-check.v1 / public_frontend_context_v6 /
lift-knowledge.v15 / j_scalar_cpu_mlir_v9 / Gate1 observer v5 / batch.v3`。

Gate1 observer 因上述 full-bwd 反例从 `triton-ttir-3.6-gate1-v4` 窄升 v5。v5 对 Source 与 Replay
执行相同的 source-neutral interprocedural fingerprint：reachable acyclic structured `tt.call` DAG
按 formal→actual 与返回值关系展开，private helper 名称/是否 outline 不进入证据，而 helper 的 op、
dataflow、递归 region topology、caller region 归属以及 load/store/atomic/barrier/assert 的 lexical
effect order 必须进入 hard evidence。unknown callee、递归、签名/实际 SSA 类型不闭合、structured
region 内副作用、未知 function/private-argument attr 和当前未建模的 multi-block CFG 均 fail closed；
function attr 只闭合允许布尔 `noinline` 这一组织 hint。Source capability failure 以 batch.v3
`observer_unsupported` 终止，在首个完整 TypeCheck/Gate1 attempt 后停止 relift，并且该 attempt 的
`feedback=null`、不调用 feedback 构造器；ordinary hard diff 中的 node/region/effect digest 也必须先
清洗再进入同 branch 的 bounded feedback。真实 full `_bwd_kernel` 的 helper 含 early-return
`cf.cond_br`，当前因此准确标记 observer unsupported；尚未闭合，不能把旧 v4 的 289 条假差异或新
unsupported 当作 SCRC 进展。普通 `noinline=false` helper 组织不进入 π，production 仍只消费 J+C。

DYT 与 SwiGLU 的既有通过证据不属于 B1，也未在 B1 重跑。

P1 的 adapter readiness 以本节上方 checkpoint 的 declared=28/pending=0 为准。Backward 被移除的
autotune `pre_hook` 对 DQ 的清零由通用 closed buffer initializer 保留；本地 helper keyword 的匿名化
只对静态解析的 slice-local callee 生效，不改写 `tl.*`、library、attribute 或动态调用。这里没有加入
FlashAttention 或其他 Kernel 专用 prompt、Replay、J/C/π 构造或 Intent op。

## Phase P2：28 项横向 closure

- 按 registry ordinal 组成多个语义混合 wave；简单 case 与 frontier case 同时推进；
- 已通过的 case 保留其独立 accepted evidence，不因其他 case 修复而被历史重判；协议实现改变时，
  只对受影响 case 发起新的 fresh run；
- 对生成方差导致的未闭合 case，在全局 `G` 内增加到 `B<=5`；同 case 仍等待 pre-Gate2 barrier，
  Gate2 失败不 fallback；
- 每轮汇总 TypeCheck rule、Gate1 hard category、Gate2 三边、J-necessity 与真实物理调用预算，按通用
  failure class 决定下一横向修正；
- 最终每个 registry item 都必须有自己的独立 `SCRC_CERTIFIED` evidence，aggregate 不能替代单项
  证书；其中 DYT/SwiGLU 沿用原 receipt，其余 26 项由 Task 11 fresh 产生。

## 验收

1. outer case-level bounded parallel 有实现、正反例、异常隔离和稳定 ordinal 测试；
2. `W/B/G/G2` 合同 fail closed，实际测试证明多个不同 case lifecycle 确实重叠；
3. same-case v3 barrier winner 与唯一 Gate2 规则不回退；
4. B1/B3/v6-v14 slice 的旧 receipt terminal evidence 保持不可变；受影响项若要形成当前
   artifact.v3/decode.v3/typecheck.v7/context.v6/knowledge.v15/RefEval.v9/observer.v5 结论，必须分别
   fresh rerun，且问题归因不复用旧 verdict 或 diagnostic recheck；
5. 28/28 各自具有独立 `SCRC_CERTIFIED` artifact 与完整 evidence：DYT/SwiGLU 为原版本证书，其余
   26 项为 Task 11 fresh evidence；
6. 全量回归通过，活动 task 唯一，完成后本 Task 直接迁入 archive，工作区提交且干净。

## 非目标

- 不实现 production backend materializer、Impl(J) search、cost model 或性能实验；
- 不建立分布式/持久化异步队列、复杂 resume、hash/commitment/freeze gate；
- 不创建第四 artifact、HarnessAgent、Composer、RuleCard 或 per-kernel prompt；
- 不把 Gate2 tolerance 当作 accumulator dtype、FMA、reassociation 或 cast permission；
- 不恢复或修改已归档的 Task 01–10。

## Task 10 handoff（2026-07-24）

- DYT 与 SwiGLU 已有完整认证，分别验证 rank-0/tanh 与 floating tolerance 三边；
- Softmax、RMSNorm、Attention reduction、Contraction 已各有 xhigh fresh terminal evidence；
- Softmax/RMS 历史 J 由 v4 更早拒绝缺失 broadcast，不追认 v8 能力修正；
- Contraction 历史三候选由 v4 拒绝 non-total typing，无需新增 accumulator schema 字段；
- Attention 最佳候选已闭合 reduction，剩余是通用 Replay fidelity；
- Task 10 全量回归为 `895 passed`，详细证据与版本审计见其归档 task。
