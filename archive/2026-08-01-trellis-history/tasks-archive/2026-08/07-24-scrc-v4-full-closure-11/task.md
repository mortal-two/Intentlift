# Task 11：SCRC v4 的 28-Kernel 有界并行全闭合

> 状态：执行中
>
> 类型：Task 10 完成后的 fresh SCRC corpus 构建；不是 backend 性能搜索、论文实验或旧 Task 03
> 恢复执行。
>
> 活动性：Task 01–10、Task 12–14均已物理归档；Task 11闭合主线保持活跃。Task 15承接model-level
> input、Compiler-observed Codex Lift、closed J、source-blind Replay agent与最终十项闭合；它不终止、
> 替代或归档本任务。

## 目标

Task 15保留经逐artifact审查确认参数化、ABI/variant正确的18份accepted J，并对8个未闭合case及
正式`split=4`下失效的2个SplitK旧证书进行fresh闭合。最终corpus由`18 retained + 10 fresh`组成，
不因通用schema增加该artifact不需要的relation而机械重跑18项。实际protocol版本以Task 15每次
能力提交后的fresh receipt为准。执行必须真正支持多个语义族
wall-clock 并行，同时保持每个 case 的证书谓词与 evidence 完全隔离。

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
first complete Gate1 closure; sibling cooperative stop receipts
        |
        v
first-observed winner (fan-out v4)
        |
        v
one hidden Gate2 under G2 semaphore
```

最终 aggregate 必须按 registry ordinal 稳定排序，与线程完成顺序无关。证据目录必须在 worker
启动前或通过原子、无冲突的 case-local 分配建立；不得使用跨 case 的 `max(existing)+1` 竞态。
每个 case/branch 必须各自新建 provider conversation、runner、writer、buffers 与 evidence state；
只有只读 `ProviderConfig`、registry、source 与 stateless adapter 可以共享。单 case 失败必须结构化
隔离，不得取消或污染其他 case。

same-case 使用 `first_observed_gate1_closure_cooperative_stop.v4`：首个交付完整 Gate1-closed evidence
的 branch 原子成为 winner，sibling 停止后续搜索并保存 `SUPERSEDED` receipt；只为 winner 在 `G2`
semaphore 下执行唯一 Gate2，Gate2 失败不 fallback。完成次序只作 proof-irrelevant 调度，不能读取
hidden Gate2，且 report 必须记录 selection rule/winner/stop receipts。若没有 winner，只有全部 `B` 个
branch 均带完整 evidence 终止为 `NO_CLOSURE` 时才能报告 `NO_CLOSURE_(B,K)`；若混有
infrastructure-failed 或 incomplete，aggregate 必须为 `infrastructure_failed` 并保留所有 branch statuses。

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
report 未增加字段，仍分别是 `replay-input.v1` 与既有 report version；后续受影响 case 的 active
fresh-run stack 为
`artifact.v3 / decode.v3 / typecheck.v7 / candidate-check.v1 / public_frontend_context_v6 /
lift-knowledge.v15 / j_scalar_cpu_mlir_v10 / functional-boundary.v3 / Gate1 observer v6 / batch.v3`。

此前 Gate1 observer 因上述 full-bwd 反例从 `triton-ttir-3.6-gate1-v4` 窄升 v5。该 v5 checkpoint
对 Source 与 Replay
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

### v7/v15 双 outer-batch immutable terminal checkpoint（2026-07-25）

公共合同提交 `e01c126f` 后同时启动两个互不共享 case/evidence/provider session 的 outer batch。A 使用
`W=2,B=3,G=6,G2=1`，B 使用 `W=3,B=2,G=6,G2=1`；两者都是 MiniSWE `gpt-5.6-sol/xhigh`、
`--skip-masked-j`。因此这不是单 B，也不是把不同 case 混成一次认证。两个进程均正常写出 aggregate
report 后以 exit 2 结束；exit 2 只表示批内含非通过分类。immutable evidence 为：

- A root：`/tmp/intentir-scrc-task11/wave-20260725-b3-v7-v15-affected4-xhigh-01`；
- A report：同路径加 `-report.json`；
- B root：`/tmp/intentir-scrc-task11/wave-20260725-b2-v7-v15-complex6-xhigh-01`；
- B report：同路径加 `-report.json`。

本轮运行栈是 `artifact.v3/decode.v3/typecheck.v7/candidate-check.v1/context.v6/knowledge.v15/
RefEval.v9/functional-boundary.v2/observer.v5/batch.v3`。后续 observer v6、RefEval v10 或其他修复只能
通过新鲜运行产生新结论，不能回写下表：

| case | immutable terminal 与预算 | 一手边界 |
|---|---|---|
| `liger_softmax` | **`SCRC_CERTIFIED`**；winner branch 1/round 3；Lift/physical=`9/136`，TC/Gate1=`9/9`，Gate2=`1` | 三边均为 tolerance PASS：Source–Replay=`0`，Source–J=Replay–J=`3.725290298461914e-09`；v9 native J execution 1 次；另两 branch 各跑满 3 rounds 后 no-closure，Gate2 不 fallback |
| `liger_rms_norm` | `reference_unsupported`；branch 2/round 3 与 branch 3/round 2 均 Gate1 closed，固定 winner 为 branch 2；Lift/physical=`8/126`，TC/Gate1=`8/8`，Gate2=`1` | Source–Replay 的 `y/rstd` 均误差 0；J leg 在 execution 前因 `k1 < t2` rank-0 runtime logical predicate 超出 v9 consumer 而 unsupported，另外两边未比较；该 winner J 已通过 `typecheck.v7`，且不能改选 branch 3 |
| `linear.kernel_fwd` | 三 branch `NO_CLOSURE`；Lift/physical=`11/185`，TC/Gate1=`9/9` | branch 2 由 LLM 明示 `extent_step`，Source/Replay loop lower/upper/step 已完全一致，旧 basis 欠定消失；剩余是 structured K recurrence 与 A/B pointer carry→advance→yield π topology未被稳定联合构造。branch 1 最后因 pointer 直接进入 `tl.dot` compile fail，branch 2/3 为真实 op/dataflow（branch 3 另含 predicate/effect）mismatch |
| `_bwd_preprocess_do_o_dot` | `infrastructure_failed`；已观察 Lift/physical=`6/73`、TC/Gate1=`6/6`，预算不完整 | branch 1/2 各跑满 3 rounds no-closure；branch 3 第 3 次 Lift 遭服务端 `auth_unavailable`，不能把本 case 写成完整语义 verdict，必须另开 fresh multi-batch |
| `liger_fused_add_rms_norm` | 两 branch `NO_CLOSURE`；Lift/physical=`6/98`，TC/Gate1=`6/6` | branch 1 最后为 op/dataflow/predicate-effect mismatch；branch 2 最后 Replay 把 block value store 到 scalar pointer而 compile fail。Source observer 正常，不属于 observer unsupported |
| `liger_cross_entropy` | `observer_unsupported`；Lift/physical=`3/61`，TC=`2`，Source observer 后不调用 Replay | 两个 well-typed artifact 都在 public root 的 runtime `cf.cond_br`、multi-block 与 early return 处 fail closed；attempt feedback 均为 null，不继续 relift |
| `liger_sparsemax` | `observer_unsupported`；Lift/physical=`2/37`，TC=`2`，Source observer 后不调用 Replay | 两个 well-typed J 都已使用通用 scan；合法 Source TTIR 的 quoted generic `tt.scan`/typed `tt.scan.return` 超出 v5，feedback=null；不是缺 IntentOp 或 Sparsemax prompt |
| full FlashAttention `_fwd_kernel` | 两 branch `NO_CLOSURE`；Lift/physical=`8/154`，TC/Gate1=`4/4` | branch 2 完成 3 个 formal round，最终仍有 interface/op/dataflow/predicate-effect mismatch；branch 1 两次 ingress rejection 后得到一个 well-typed formal round，后续仍以 LimitsExceeded 终止。Source observer 可用，这是 online-attention 真正 realization frontier |
| FlashAttention `_bwd_kernel_one_col_block` | 两 branch `NO_CLOSURE`；Lift/physical=`4/95`，TC=`1` | branch 1 well-typed 后在 public multi-block CFG observer 边界停止且 feedback=null；branch 2 三次 Lift 均 LimitsExceeded、无 TypeCheck。混合边界不能伪装成纯 observer verdict |
| xFormers SplitK `_fwd_kernel_splitK` | `infrastructure_failed`；aggregate 只记录不完整 observed Lift/physical=`3/44` | branch 2 三次 ingress rejection后 no-closure；branch 1 最后遭 provider `unexpected EOF`，没有完整 candidate budget，必须 fresh multi-batch |

Softmax 证书的 artifact digest 是
`610ba6bc5f49741fe17cceb496d45600e385dcd6bd37a41658d725836f35dba6`。其 Gate1
interface/op-multiset/dataflow/reduction/predicate-effect 五类全过，RefEval receipt 为
`j_scalar_cpu_mlir_v9 + intentir.j-functional-boundary.v2 + logical_domain_reduction_padding_v2`；
`run_masked_j=false/J-necessity=NOT_RUN` 只是未执行可选必要性实验，不改变主证书。严格 claim 仍只覆盖
当前 Harness、`c0=128` 与外部 Q 下的 hidden finite case，不外推为所有 Softmax shape/value 的全称证明。

这轮横向结果也给出下一共性顺序，而非 per-kernel 修补：

1. observer v6 只增加 typed、single-block、effect-restricted 的通用 `tt.scan`，随后仅 fresh rerun
   Sparsemax；v5 evidence 不重判；
2. RefEval v10 只在当前 invocation 可由 J logical valuation 证明 access predicate 对 closed domain
   全真时做 runtime tautology discharge，随后 fresh rerun RMS；不读 Source/C/π、不改 J；
3. multi-block CFG/early-return observer 先定义 closed CFG fingerprint 再实现，不能 flatten block；
4. Linear 先审查既有 π vocabulary 是否已足以表达 pointer carry topology；只有证明不可表达才增加
   最小 source-neutral relation。任何这种 future relation 都只能进入未来
   `artifact.v4/decode.v4` 并另开 fresh session；当前 `artifact.v3/decode.v3` wire 与既有 receipt 不变，
   本 checkpoint 不定义其 schema；
5. preprocess 与 SplitK 的基础设施失败在下一干净 checkpoint 另开 fresh multi-batch，不从本轮补造
   verdict。

### observer v6 / RefEval v10 active implementation checkpoint（2026-07-25；不是 fresh case verdict）

- `triton-ttir-3.6-gate1-v6` 已实现，并保留 v5 对已建模 op/reachable structured call graph 的
  interprocedural comparison rules；新增 typed、single-block、encodingless `tt.scan`，以及全局
  closed operation/effect classification。unknown op、malformed
  generic/body line、multi-block CFG/region 与未支持 control/region fail closed；TTIR 专项为
  `136 passed`，完整 Gate1/SCRC 回归为 `460 passed`。历史 191 份真实 TTIR 中 182 份继续通过、
  9 份保持既有 multi-block unsupported；77 组可比较历史 Source/Replay pair 无 verdict 漂移，另
  4 组只因同一 multi-block 边界无法比较。这不声称 v5→v6 acceptance 单调；
- `j_scalar_cpu_mlir_v10 + intentir.j-functional-boundary.v3` 已实现。它只为已选 Gate2 winner 的当前
  invocation，用 J rank-0 logical `i1`/integer values 与完整 closed dimensions 逐 access 证明 predicate
  全域为真；调用权属于已完成 winner 选择的主 SCRC runner，projection library 不自行选择 winner。
  不改 J、不读 Source/C/π、不进入 Lift feedback；v3 projection 的 invocation-local proof subreceipt
  不含 raw value/value digest，这只是 audit-data minimization，不是 confidentiality/privacy proof。
  production projection 仍没有 runtime-value 参数；Gate2 direct-J + CPU backend 回归为 `85 passed`；
- CPU host pipeline 的 `expand-rsqrt-for-libm` 只在共享 Intent-to-Linalg 之后展开 `rsqrt -> 1/sqrt`，
  不改变共享/GPU lowering；isolated CPU backend 为 `20 passed`；
- 根级受影响回归为 `484 passed`，完整 `python/tests` 回归为 `1274 passed`；
- 以上是实现与测试 checkpoint，不是 Sparsemax/RMS 或其他 case 的新 SCRC evidence。上方
  `RefEval.v9/functional-boundary.v2/observer.v5` immutable table、Softmax certificate 和历史 verdict
  均不回写；受影响 case 仍需 fresh run。

DYT 与 SwiGLU 的既有通过证据不属于 B1，也未在 B1 重跑。

### decode.v9 / RefEval v14 RMSNorm-add current certificate checkpoint（2026-07-25）

`decode.v9` 修复 canonical logical inout handle 的首次声明编号后，xFormers
`_rms_norm_add_kernel` 在 fresh `B=2` run 中由 branch 1/formal round 2 形成当前有效 Gate1 closure；
artifact digest 为 `e89066867f5918fbeb11126307d3b1b1fa53043ac0ced956f19e0c589abe340c`，
parent evidence 为
`/tmp/scrc_task11_v9_rmsnorm_add_inoutfix_20260725_02/xformers_ops_triton_rmsnorm_kernels_rms_norm_add_kernel/fanout-001/candidate-01/attempt-03`。
该 artifact/Replay 随后只做零模型 Gate2 capability recheck，不重新 Lift、不改 J、不 fallback：

- `j_scalar_cpu_mlir_v13 + functional-boundary.v12` 先消费唯一 output mapping、同 index/predicate 的
  inout producer store/load 与 store-before-load effect order，把 physical post-state reload 投影回 J 已有
  `v0`；
- `j_scalar_cpu_mlir_v14 + functional-boundary.v13` 再证明 weight masked load 的 outside-domain value
  仅流经 pointwise `map_binary/select`，且唯一 observable store 使用同 predicate，因此缺省 physical
  fill 不可观察；reduce/scan/loop/transpose/pad、不同 predicate 与未知 effect 仍拒绝；
- shared J→Intent MLIR emitter、CUDA analysis 与 structured reference v5 消费 canonical
  `JElaboration.output_bindings`。`returns` 或
  `output_mapping` 任一可唯一给出 destination，两者同时出现必须一致；这不补造 artifact 字段。

最终以全部 canonical consumers 同步后的 immutable evidence 为
`/tmp/scrc_task11_v14_rmsnorm_add_canonical_consumers_recheck_20260725_01/xformers_ops_triton_rmsnorm_kernels_rms_norm_add_kernel/gate2-recheck-001`，
`model_calls=0`、复用 exact artifact 与 replay，status=`scrc_certified`。Source–Replay 的 `x/h1`
最大误差均为 0；Source–J 与 Replay–J 的 `x` 最大误差为 0，`h1` 最大误差均为
`1.1920928955078125e-07`，exact/tolerance policy 全部通过。projection receipt 明确包含
`inout_poststate_reload:a3:t0:v0` 与 `predicate_dominated_load:a4:k1 < x0`。

因此当前严格独立覆盖为 **12/28**：此前 11 项加 xFormers RMSNorm-add。该 recheck 不回写 parent
Gate2 failure；新证书只由独立 capability-recheck evidence 承担。

后续 fresh Lift 的通用 instruction/schema/preflight 以 `intentir.lift-knowledge.v21` 明示 canonical
output binding 可由 `returns` 或 `output_mapping` 任一唯一给出、双写必须一致。正在/已经运行的 v20
batch 保留原 prompt receipt 与 verdict，不由 v21 追认；v21 也不自动插入 binding。

### decode.v9 / knowledge.v20 四批横向 terminal checkpoint（2026-07-25）

以下四个 report 均已完整落盘并终止，运行中的进程为 0；所有 verdict 保持其启动时的 v20 prompt、
decode.v9 与 observer v10，不由上节 v21 追认：

- `/tmp/scrc_task11_v9_dataflow_fresh_b3_20260725_01-report.json`；
- `/tmp/scrc_task11_v9_control_inoutfix_b2_20260725_01-report.json`；
- `/tmp/scrc_task11_v9_gate1_horizontal_b2_20260725_01-report.json`；
- `/tmp/scrc_task11_v9_remaining_horizontal_b2_20260725_01-report.json`。

| case group | immutable terminal；Lift/physical、TC/Gate1、Gate2 |
|---|---|
| KLDiv | `NO_CLOSURE_(3,3)`；`15/306`、`9/9`、`0` |
| TVD | `NO_CLOSURE_(3,3)`；`14/301`、`9/9`、`0` |
| SplitK reduce backward | `NO_CLOSURE_(3,3)`；`9/191`、`9/9`、`0` |
| Llama4 RoPE | `NO_CLOSURE_(2,3)`；`6/162`、`6/6`、`0` |
| Flash backward one-col | `NO_CLOSURE_(2,3)`；`8/167`、`6/6`、`0` |
| full Flash backward | `NO_CLOSURE_(2,3)`；`9/224`、`6/6`、`0` |
| Liger RoPE | `NO_CLOSURE_(2,3)`；`8/190`、`6/6`、`0` |
| Flash linear | `NO_CLOSURE_(2,3)`；`6/89`、`3/3`、`0`；另一 branch 无 well-typed J |
| full Flash forward | `NO_CLOSURE_(2,3)`；`7/127`、`3/3`、`0`；另一 branch 无 well-typed J |
| SplitK reduce | `NO_CLOSURE_(2,3)`；`7/170`、`6/6`、`0` |
| CrossEntropy | `NO_CLOSURE_(2,3)`；`6/128`、`6/6`、`0` |
| GroupNorm | `NO_CLOSURE_(2,3)`；`8/187`、`6/6`、`0` |
| Qwen2VL mRoPE | `NO_CLOSURE_(2,3)`；`8/199`、`6/6`、`0` |
| Sparsemax | branch 1/formal round 2 `replay_closed` 后唯一 Gate2=`REFERENCE_UNSUPPORTED`；`6/167`、`2/2`、`1` |
| SplitK forward | `NO_CLOSURE_(2,3)`；`6/115`、`3/3`、`0`；另一 branch 无 well-typed J |

Sparsemax 当前 exact parent 是
`/tmp/scrc_task11_v9_remaining_horizontal_b2_20260725_01/liger_sparsemax/fanout-001/candidate-01/attempt-03`，
artifact digest=`f6284b7c38b58089b96c9c26bc04e768fac571122289a68c5f36e533914670e1`。
Source–Replay output `o` 最大误差为 0；current v14 zero-model recheck 仍在 J execution 前以
`functional padded reduce/scan requires one local-axis algebra contract for every affected consumer` 拒绝。
artifact 的 scan `n4` 没有 `boundary.reductions` entry，且 predicate-bearing load 到 scan/reduce 的
outside-domain consumer paths 没有 `consumer_extensions`。这不是 consumer 可从 Source/运行值补造的
字段；下一 fresh Lift 必须由通用合同明确要求 reduce/scan algebra 与 per-consumer typed outside
identity，再复用既有通用 TypeCheck/RefEval 消费，不对该 kernel 写专用提示。

该通用提示现在版本化为 `intentir.lift-knowledge.v22`：每个 semantic reduce/inclusive-scan 都须由
Lift author `boundary.reductions`，predicate-bearing load 到 sum/max/min reduce/scan 的每个 affected
consumer/operand/access 须 author typed two-sided `consumer_extensions`；fill、reduction identity、C/π
均不是替代品。v22 没有 Sparsemax 名称、模板 J 或自动 field construction。

因此本 checkpoint 后的严格覆盖是 **12/28**，另有 Sparsemax 一个当前有效 Gate1-only artifact。
Flash LayerNorm 的旧 value-if artifact 在 Replay v9 diagnostic recheck 仍为 no-closure，不计入该一项。

P1 的 adapter readiness 以本节上方 checkpoint 的 declared=28/pending=0 为准。Backward 被移除的
autotune `pre_hook` 对 DQ 的清零由通用 closed buffer initializer 保留；本地 helper keyword 的匿名化
只对静态解析的 slice-local callee 生效，不改写 `tl.*`、library、attribute 或动态调用。这里没有加入
FlashAttention 或其他 Kernel 专用 prompt、Replay、J/C/π 构造或 Intent op。

## Phase P2：28 项横向 closure

### Task 12 归档 / Task 13 接力 checkpoint（2026-07-27）

Task 12 runtime实验没有新增证书；后续独立fresh closure与exact Gate2 capability recheck曾把严格覆盖
推进到18/28。Task 14的model-input审计进一步确认可保留的18项为SwiGLU、RMSNorm、fused-add
RMSNorm、GeGLU、LayerNorm、Softmax、DYT、Sparsemax、KLDiv、PolyNorm、JSD、TVD、xFormers
RMSNorm、xFormers RMSNorm-add、Flash backward preprocess、CrossEntropy、GroupNorm与Flash
one-pass LayerNorm。旧Liger RoPE与SplitK reduce不再位于这18项：前者的consumer-local half-view J
有误，后者旧证书只覆盖`vararg_arity=2`。各保留证书继续按自身artifact/Replay/Gate版本解释，不被
新实现静默重写或重判；Task 15只fresh闭合active十项。

Task 13提供历史single-owner执行基线；Task 15只使用Codex主路径，并在同一数学合同下增强匿名
Compiler观察、持久Lift owner、closed J dimension relation与fresh source-blind Replay coding episode。
Task 13 的 diagnostic pilot 不冒充新的 SCRC 证书。

Task 13 同时把 fresh Harness 升为 `triton-workload-adapter.v5`：concrete `shape` 只作本次 valuation，
source-grounded shared axes 由 Harness-only `h*` shape contract 暴露，runtime scalar/grid 只能声明与
该 coordinate 的 equality/realization。Lift 必须显式作者化双射 `x*` 与必要的 value-dimension
relation；v5 另闭合 semantic constexpr ↔ `h*` 与真实 scratch inout role。fixed code不从
literal/grid生成 J。旧18项证书不重写；只有active十项与后续证明确有profile/J错误的case fresh Lift。

v55 DYT canary 首次形成 `[x0,x1]` 且 SCRC-certified 的 fixed-valuation J，但暴露 semantic constexpr
`N` 与 `h1` 尚未闭合；因此 current knowledge v56 将 adapter 显式、valuation-matched 的
bound-constexpr/shape-coordinate relation 纳入同一 closed Harness contract，仍由 Lift 直接绑定 `x*`
或作者化 `t*=x*`。full Flash forward v55 `B=3,K=3` 的 9 个 artifact 全部 TypeCheck PASS，但最终
`NO_CLOSURE_(3,3)`；最佳 branch 已有 physical iteration，剩余是 physical loop carriers 与 tile/effect/
dataflow realization。两项 immutable evidence 详见 Task 13，均不能用 v56 追认。

- 按 registry ordinal 组成多个语义混合 wave；简单 case 与 frontier case 同时推进；
- 已通过的 case 保留其独立 accepted evidence，不因其他 case 修复而被历史重判；协议实现改变时，
  只对受影响 case 发起新的 fresh run；
- 对生成方差导致的未闭合 case，在全局 `G` 内增加到 `B<=5`；同 case 首个 Gate1 closure 后协作停止
  sibling，Gate2 失败不 fallback；
- 每轮汇总 TypeCheck rule、Gate1 hard category、Gate2 三边、J-necessity 与真实物理调用预算，按通用
  failure class 决定下一横向修正；
- 最终每个 registry item 都必须有自己的独立accepted `SCRC_CERTIFIED` evidence，aggregate不能
  替代单项证书；Task 15形成10项fresh证书，与18项保留证书组成最终corpus。

## 验收

1. outer case-level bounded parallel 有实现、正反例、异常隔离和稳定 ordinal 测试；
2. `W/B/G/G2` 合同 fail closed，实际测试证明多个不同 case lifecycle 确实重叠；
3. same-case candidate-fanout v4 first-observed winner、cooperative sibling stop 与唯一 Gate2 规则不回退；
4. 历史receipt terminal evidence保持不可变；active十项必须在Task 15声明的current protocol下fresh
   闭合，18项保留artifact只有在证明确受新能力/输入错误影响时才重新Lift；
5. 28/28各自具有独立、source-grounded且ABI/variant正确的accepted `SCRC_CERTIFIED` artifact与完整
   evidence，其中18项retained、10项fresh的谱系明确；
6. 以真实closure artifact完成验收，不新增或恢复mock tests；完成后本Task直接迁入archive，工作区
   提交且干净。

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
