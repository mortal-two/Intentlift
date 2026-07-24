# Task 11：SCRC v4 的 28-Kernel 有界并行全闭合

> 状态：执行中
>
> 类型：Task 10 完成后的 fresh SCRC corpus 构建；不是 backend 性能搜索、论文实验或旧 Task 03
> 恢复执行。

## 目标

使用 Task 10 已完成的 `artifact.v2 / decode.v2 / typecheck.v4 / candidate-check.v1`、
`j_scalar_cpu_mlir_v8` 与 Gate1 observer v4，对 active registry 的 28 个 Kernel 建立各自独立的
fresh accepted `(J,C,π)`、Replay、Gate1、Gate2 与预算证据。执行必须真正支持多个语义族
wall-clock 并行，同时保持每个 case 的证书谓词与 evidence 完全隔离。

稳定 J corpus 完成后，production backend B0/B1 作为独立任务消费这些 J；本任务不要求每闭合一个
Kernel 就紧邻实现其 backend。

## 第一性原理

1. LLM 完整生成 `(J,C,π)`；固定代码只做 closed parse/elaboration、dependent checking、拒绝、
   执行与比较，不从 Source/Gate 自动补 artifact。
2. 不同 case 可以并行，但不能混成一次认证；candidate、conversation、Replay、diff、winner、C
   valuation、hidden Gate2 input 和 Harness mutable state 都是 case-local。
3. 横向覆盖优先于围绕一个 Kernel 无限重试。每个执行波至少覆盖多个语义族；通用缺口修复后才
   fresh rerun 受影响的 case。
4. 历史 v3/v7 artifact 只作诊断输入；v4/v8 不静默追认或重判旧 verdict。
5. 一个 case 的 infrastructure failure、NO_CLOSURE 或 unsupported 不取消其他 case，也不改变其他
   case 的证书状态。

## 并发合同

设：

- `W`：同时运行的独立 case lifecycle 数；
- `B`：每个 case 的 same-case candidate branch 数；
- `G`：全局 branch/provider/Gate1 in-flight 上限；
- `G2`：同时执行 hidden Gate2 的设备上限。

必须满足 `W >= 1`、`B >= 1`、`W*B <= G`。外层 scheduler 只改变 wall-clock 调度，不进入
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

## Phase P0：实现真实 case-level bounded parallel

- batch runner 增加显式 `case_workers`，串行路径保持 `W=1`；
- CLI 允许 `--tier` 或显式 case 集在 `W>1` 下运行；禁止无选择地隐式启动 28 项；
- 明示并校验 `W*B <= G`，报告 `W/B/G/G2` 作为 proof-irrelevant audit metadata；
- provider config 可以只读共享，provider conversation/session、runner、writer、buffers 与 evidence
  state 必须每 case/branch 新建；
- Gate2 在 case winner 选定后执行，并受独立 `G2` 限制；Gate2 结果不回流 sibling 或其他 case；
- 单 case 异常结构化隔离，最终 summary 与串行执行具有相同 ordinal 顺序和既有 classification。

## Phase P1：v4 横向 canary wave

首波使用 `W=4,B=1` 同时运行四类，而不是串行等待：

1. Softmax：要求 Lift 显式声明 max/sub 与 sum/div 的 non-trailing broadcast；
2. RMSNorm：要求 total typed graph 与 row-wise rstd broadcast；
3. Attention reduction：验证 unconditional store 与 base/lane merge 的公共 U_f 澄清；
4. Contraction：要求显式 typed contraction/cast/epilogue，不依赖 Triton 默认 f32 accumulate。

这四项用于验证最终 v4 构造边界，不把 `B=1` NO_CLOSURE 当作标准搜索结论。若某类在合法 J 后
暴露 fixed consumer 缺口，只实现跨 Kernel 可复用的最小能力并补正反例：

- RMS 合法 J 后才处理 embedded mean/reduce declared shape 与 rsqrt CPU lowering；
- loop/contraction 类修复 Gate1 对 `scf.for` iter_args/region results 的 pointer liveness；
- Replay 对无 predicate access 不加 mask，含混/未锚定 C 安全忽略，并按 π 消费地址依赖；
- 不为任何具体 Kernel 增加名称、路径或 Source fragment 驱动规则。

## Phase P2：28 项横向 closure

- 按 registry ordinal 组成多个语义混合 wave；简单 case 与 frontier case 同时推进；
- 已通过的 case 保留其独立 accepted evidence，不因其他 case 修复而被历史重判；协议实现改变时，
  只对受影响 case 发起新的 fresh run；
- 对生成方差导致的未闭合 case，在全局 `G` 内增加到 `B<=5`；同 case 仍等待 pre-Gate2 barrier，
  Gate2 失败不 fallback；
- 每轮汇总 TypeCheck rule、Gate1 hard category、Gate2 三边、J-necessity 与真实物理调用预算，按通用
  failure class 决定下一横向修正；
- 最终每个 registry item 都必须有自己的 `SCRC_CERTIFIED` evidence，aggregate 不能替代单项证书。

## 验收

1. outer case-level bounded parallel 有实现、正反例、异常隔离和稳定 ordinal 测试；
2. `W/B/G/G2` 合同 fail closed，实际测试证明多个不同 case lifecycle 确实重叠；
3. same-case v3 barrier winner 与唯一 Gate2 规则不回退；
4. 首波四类均在 v4 下形成 fresh terminal evidence，且问题归因不复用 v3 verdict；
5. 28/28 各自形成 fresh `SCRC_CERTIFIED` artifact 与完整 evidence；
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
