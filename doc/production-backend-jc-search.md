# Accepted J+C production backend：真实 MLIR lowering、B 合同与 C 搜索

> 状态：本文的实现与性能数字对应已经 content-addressed 冻结的 12-kernel catalog。2026-07-26
> 最新严格 SCRC closure 已达到 16/28；新增四项尚待导入，不能把本文历史 12-kernel 数字重标为
> 16-kernel 结果。

本文记录已冻结 12 个 SCRC Gate1/Gate2 闭合 Kernel 的 production backend 实现。性能与消融数值见
[production-backend-report.md](results/production-backend-closed-12/production-backend-report.md)。

最新 16-kernel closure 中，待 append-only 导入 `corpus/production-jc-v1` 的四项为：xFormers
RMSNorm-add、Liger Sparsemax、Liger KLDiv、FlashAttention `_layer_norm_fwd_1pass_kernel`。四项都已有
独立完整 Gate1+Gate2 evidence；导入仍须复制 exact artifact/C/receipt、核验 digest 并经过同一个
catalog loader，不从 `/tmp` 路径直接长期消费，也不重新 Lift 或修改 J。

当前 intake handoff 的 raw-file SHA256 为：

| case | artifact SHA256 | closure evidence SHA256 |
|---|---|---|
| xFormers RMSNorm-add | `5e6375eb9c337fc8c30387ff97965a0bca89023432981786a6ba9f9024ba707e` | `1bc8b069c1d5ad5b6b139173144f75493dc8e59ab657793a3b118abaeb3e40f1` |
| Liger Sparsemax | `923b496353836a2815ba02dd84c9f3c1d0dd9dca0500f5ed3c9391dc34ed2876` | `710368f0c3536e4f876011e6a86e34059e08b9f8aa8221df67567ff3ee94bef4` |
| Liger KLDiv | `78ca4a70d137ce221bdca3ea304a9770ebf702da76c062211b7876f544cb1582` | `40425909c8c66ccafb489de19094b6e9ac86e010af024f9a5e487ee67bf33a8f` |
| FlashAttention one-pass LayerNorm | `28c5a97bcd5c7784e189e3edc92d3000c080b9ce2cddb26db892739a27705e1c` | `caee663241d996bc1df1d67246d6cd4fae553d235985795bfbbf4478f7c402fc` |

这些是 staging 文件 digest，不冒充 catalog 的 canonical J/C/snapshot digest；正式导入工具仍须重新
计算并交叉核验各层 digest。staging roots 依次为
`/tmp/scrc_task11_v9_rmsnorm_add_inoutfix_20260725_02`、
`/tmp/scrc_task11_v25_continuation_sparse_b1_20260725_01`、
`/tmp/scrc_task11_v60_kl_agent_discovery_20260725_01` 与
`/tmp/scrc_task11_v69_layernorm_formal_replay_fanout_20260725_01`；它们只用于本次 intake handoff。

当前 16 项均无 typed contraction；Flash `do_o_dot` 是 FP32 multiply + row reduction。因此当前性能
推进不把 TF32 误套到 row reduction；TF32/IEEE 精度状态留给后续 `tl.dot`/matmul/完整 Attention J。

## 1. 输入边界

core lowerer 的持久输入只有：

- accepted canonical `J=A+B`；
- 可选、可擦除的 `C`；
- target capability。

它不读取 Source、Replay、π、Gate verdict、Q、kernel 名或 provider 身份来决定
lowering。`corpus/production-jc-v1/` 保存 12 个 content-addressed
`intentir.production.jc-snapshot.v1` 对象和一个 stable-name catalog。catalog loader
会同时核验 snapshot、J、C digest；验收导出目录的 `<name>.json` 和冻结后的
`<sha256>.json` 使用同一个消费入口。

```text
accepted J+C snapshot
  -> Emit(J, concrete invocation)              M0 Intent MLIR
  -> ContractView(J, target)                   hard-boundary admission
  -> Attach_C(M0, C, target) / Erase_C         ordering only
  -> registered C++ MLIR pass candidates
  -> CUDA: GPU dialect -> NVVM -> PTX -> ptxas cubin
     RVV: Linalg -> bufferized loops -> LLVM -> riscv64 object
  -> generic Q/D candidate filter
  -> performance selection among passing states
  -> held-out final-winner backend closure
```

Python 只负责 snapshot transport、候选编排、CUDA Driver ABI 绑定、计时与报告。
所有目标计算和 local/structural/schedule transformations 都由注册的 C++/upstream MLIR pass 生成；不存在
Python/Triton 替代 kernel emission。

这里的 Q/D filter 是一个待实现的通用 search controller，不是每个 Kernel 的 Python emitter 或
validator。core MLIR compiler 不读取 Q；controller 只通过 signature-driven opaque harness 执行
Source/Replay/RefEval/target，并使用冻结 comparator。当前已提交实现完成了真实 MLIR compilation、
correctness checks 与性能搜索，但尚未把 reduction tree、FMA、内部精度候选统一接入该 Q/D
feasibility protocol。

## 2. J、B 与 constexpr

`materialize_j()` 先对 canonical J 做 deterministic materialization。当前支持：

- functional boundary projection；
- dense predicate/fill materialization；
- exact value-dimension relations；
- no-scan full-axis logical step，以及 J-declared constexpr 对 typed
  `effect_if` 的精确分支专化；
- J.signature 明确声明的 rank-0 `constexpr` specialization。

constexpr specialization 不是从 Source 猜常量。只有 J.signature 的
`ParameterKind.CONSTEXPR` 可以被绑定；普通 scalar/pointer 输入会被拒绝。若 constexpr
同时受 B 的 value-dimension equality 约束，materializer 必须先验证值与 dimension
一致，再删除 runtime ABI 参数。该检查曾真实拦截 xFormers RMSNorm 的旧索引投影问题。

每次 CUDA 搜索前都会复用 `analyze_j_for_cuda()` 建立 ephemeral
`CudaContractView`，记录：

- J、target 与 boundary digest；
- 已消费 B facts；
- unsupported B facts；
- deterministic boundary projection。

存在未消费的 alias/effect/access guard/storage/update/reduction/output relation 时，搜索在
候选编译前 fail-closed。C 不能绕过该入口。编译审计会向每个已闭合 kernel 的同一 M0
注入未支持 `may_alias` 作为负对照：12/12 的 with-C 与 Erase_C 两臂都在 codegen 前以
`alias_relation` 拒绝。

该负对照只证明显式 `may_alias` 不会被 C 绕过，不能单独证明 v7 的空 alias list 等价于完整
no-alias。当前 `b.readonly_nonalias_inputs.v1` 对空列表的解释属于待收紧的 B0/B1 contract gap；规范
要求后续 capability receipt 同时给出 exact J carriers、artifact/schema completeness rule 与 derivation
version。无法证明 alias relation set complete 时应返回 `missing_contract`，而不是从具体 benchmark
buffers、测试通过或空数组猜 no-alias。

## 3. C 的真实消费

`Attach_C` 把 C slot 解析为带 role、value、subject/axis attachment 的 search prior；
`BaseIR(S_C)` 始终是同一 M0，且 `Erase_C(S_C) == M0`。C 不进入 pass legality，也不
改变 candidate ID、performance-cache identity 或 candidate universe。

当前候选枚举在 `elementwise/row_reduction/padded_row_reduction` 三类 registered lowering matcher
之间 round-robin，再用 C 的 tile/thread prior 排同一路径种类内的 schedule。这三个名字是现有
receipt/API 的 legacy `family` 字段，本文只把它们解释为 `lowering_path_kind`，不把它们当作可发射的
compiler family 或 semantic object。round-robin 只避免有限预算下某种 matcher 候选被饿死；C
仍不决定哪条 path 合法。

正式消融固定 J=A+B、ContractView、target、compiler、workload、candidate universe 和
budget，只切换 `Attach_C/Erase_C`。RTX 5090 的 order-only 对照表明：

- budget=3：with-C 和 without-C 都覆盖 12/12，但 with-C 的几何平均 best latency
  低 6.9%；
- full budget=83：使用相同 latency map 时两臂完全收敛；
- 因而 C 提高有限预算搜索效率，但不改变 `ReachHard`、Q/D feasibility 或 correctness。

## 4. B-conditioned local transformation 与结构路径证据

B 不能从 J 中关闭，因为持久语义对象就是 `J=A+B`。当前 GPU 配对固定 J、完整 ContractView、C、
target、workload 和计时协议，只改变已注册的 B-conditioned local transformations 是否加入候选：

- L0（legacy CLI 名 `schedule-only`）：仍完整消费 B 并执行 fail-closed legality，但候选的
  `b_transformations=[]`；
- L1（legacy CLI 名 `contract-guided`）：候选集合严格包含 L0，并加入其
  `required_b_capabilities` 被 ContractView 满足的变换；
- 两臂分别从已通过当前 hard admission 与 canonical-J correctness 的候选中按性能选 winner；这是
  frozen 12 的历史机制，未来 observation-changing states 改用统一 D_search/D_cert protocol。

当前公共 B capability 由 canonical J 的 effect、alias、update、storage 与 signature 派生，
不是从 kernel 名或 Source recipe 猜测。CUDA 已注册 rank-0/singleton host-by-value capture、
epilogue read-only load motion；RVV 已注册 read-only non-alias whole-graph Linalg fusion。缺少所需
capability 的候选不会进入 codegen。

正式 GPU 配对结果为：RTX 5090 全 12 核 L1/L0 几何平均 **1.031x**，超过 1% 的
2 个受影响核几何平均 **1.194x**；H100 分别为 **1.019x** 与 **1.120x**。RTX 上
Flash preprocess、JSD 分别为 1.261x、1.130x；H100 分别为 1.170x、1.071x。
这些数值证明 B-conditioned local optimization，不作为 E3/E2 structural lowering-path gain。
这与 C 消融是两件事：B 决定额外变换是否被授权，C 只决定有限预算候选顺序。

RVV 的 whole-graph Linalg fusion 确实改变 materialization/loop topology，是当前首个结构性 lowering
transition；但当前实现把 `fuse_elementwise` 混在 `RiscvSchedule` 中，正式结构路径实验前应将它拆成
transformation trace，并让 fused/materialized 两条 path 分别搜索 schedule。

## 5. 注册的真实 MLIR lowering paths

### Elementwise row lowering path

`ConvertIntentElementwiseToGPU.cpp` 对完整 typed map/cast/select DAG 生成一个
`gpu.launch`。搜索轴包括 block threads、row/flat layout、vector width，以及由只读非别名
B capability 授权的 singleton/rank-0 host-by-value ABI。非 2 次幂 tail
使用安全 in-bounds 地址 + guarded store，不产生越界，也避免每个 input load 的发散
`scf.if`。

已消费：DYT、GeGLU、SwiGLU、TVD。TVD 的 optional-label 输入在 constexpr
`HAS_LABEL=false` 后成为 dead ABI input；MLIR host signature 仍保留该 memref，但 C++
lowering 不生成虚假设备 load。RTX 5090 上 TVD 为 0.7280 μs，对 Triton 0.8103 μs
（1.113x speedup）；H100 为 1.1339 μs，对 1.1912 μs（1.051x speedup）。

### Row reduction lowering path

`ConvertIntentRowReductionToGPU.cpp` 对 normalization/loss 型完整 DAG 生成 read-once
register tile、warp/shared reduction 和 typed epilogue。搜索轴包括：

- block threads；
- register tile extent；
- scalar/vec4 memory path；
- deferred/preloaded epilogue Col；
- rank-0 host-by-value / device-uniform-load ABI。

FMA 选项目前仍是旧的专项离线工程开关，尚未接入统一 production search。按最新设计，它不再等待
J 增加 mechanism permission；实现时应把 FMA concrete MLIR state 标记为 `observation-changing`，
先通过 hard verifier，再进入通用 Q/D filter。现有 `reduce.assoc`、value-dimension equality、
predicate/output/alias projection 继续参与 hard admission 或 codegen。

已消费：Liger RMSNorm、fused-add RMSNorm、LayerNorm、Softmax、JSD、PolyNorm、
xFormers RMSNorm。

### Masked short-dot + pad lowering path

`ConvertIntentPaddedRowReductionToGPU.cpp` 结构化匹配：

```text
same B-derived (p < validP && n < validN) mask
  -> masked A/B [1,M,P,N]
  -> f32 multiply
  -> reduce(axis=3)
  -> authored zero pad P -> Q
```

matcher 不读函数名。搜索 block threads、full-feature vector load 和 rank-0 scalar ABI。
将 valid-row/valid-feature 从每线程 device load 改为 host-by-value MLIR candidate 后，
Flash preprocess 在 RTX 5090 上从约 1.35× Triton 降至 1.07×，H100 为 1.06×。

## 6. 搜索与计时

`search_cuda()` 对每个 candidate 执行：

1. complete Layer-B audit；
2. 指定唯一 registered fast path 和 pass options；
3. C++ whole-graph admission；
4. GPU/NVVM/cubin compilation；
5. 已合法 candidate 的 device timing；
6. top-finalist robust tournament；
7. fastest candidate selection。

运行结果不用于放宽 memory/effect/ABI hard boundary。正确性测试与 performance selection 分离；
当前 frozen-12 winner 另外与 canonical J 比较，未来 observation-changing state 则先经过通用
Q/D feasibility、最终再做 direct-edge closure。baseline 是 SCRC 已使用的 exact declarative Triton
workload adapter/source slice，而不是新写的替代 kernel。

performance cache key 包含 M0 digest、target、candidate、toolchain 和 benchmark workload，
但不包含 C digest；同一 candidate 在 Attach_C/Erase_C 两臂可安全共享测量。

## 7. H100 分支

H100 机器不安装/调用 IntentIR compiler。`export_cuda_benchmark_bundle.py` 在本地用
`INTENT_GPU_CHIP=sm_90a` 生成 cubin、launch plan、输入、canonical-J reference 和 exact
Triton baseline；`run_cuda_benchmark_bundle.py` 在 `ssh h100` 只加载 cubin并做 CUDA Driver
launch、correctness 和 timing。

bundle 同时保存 with-C/without-C candidate order、ContractView digest 和 candidate-universe
invariant，因此 H100 的 bounded/full 搜索可以直接做预算消融。完整搜索相对 bounded 前缀
在 PolyNorm 上提升 1.159×，在 JSD 上提升 1.069×。

## 8. RVV 分支

RVV 后端不生成 C/Python kernel。它从与 CUDA 相同的 deterministic Intent MLIR M0 出发，
执行以下真实编译链：

```text
registered ConvertIntentToLinalg + ExpandRsqrtForLibm
  -> materialized/fused upstream Linalg structural arm
  -> one-shot-bufferize + buffer-results-to-out-params
  -> registered ForwardBufferResultsToOutParams
  -> registered LowerDisjointMemrefCopyToLoops
  -> Linalg/SCF/MemRef/Math -> LLVM dialect -> LLVM IR
  -> LLVM O2/O3 vectorization search -> llc riscv64 (+v,+zfh) PIC object
```

`ssh rvv` 实测机器为原生 riscv64、64 核、2.6 GHz、VLEN=128。portable bundle 在本地完成
MLIR/LLVM 编译，远端只用系统 clang 链接 object，并通过 ctypes C ABI 做 canonical-J
correctness 和固定 CPU 计时。12/12 kernel、全部候选均通过正确性；fused arm 相对 materialized
arm 的全体几何平均加速为 **1.157x**，11/12 winner 使用该变换。
Flash preprocess 为 1.495x，PolyNorm 为 1.445x，Liger RMSNorm 为 1.218x，TVD 为 1.140x。

RVV winner 不强制“必须含向量指令”：例如包含 `tanhf` 的 DYT 在该小 shape 上标量候选更快，
搜索会如实选择它；其他候选与 receipt 仍记录 RVV 指令数。当前没有与 RVV 上的 Source/Triton
实现进行配对，因此 1.157x 只能解释为 whole-graph fusion path 的内部因果消融，不能解释为超越
Source，也暂不冒充完整的 E3/E2 structural-path search。

## 9. 复现

```bash
export PYTHONPATH=$PWD/python
export INTENT_OPT=$PWD/build/tools/intent-opt/intent-opt

# RTX 5090：C 顺序消融（两臂使用相同 L1 candidate universe）
python tools/benchmark_production_cuda.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/rtx-with-c.json --search-prior with-c \
  --b-search-mode contract-guided
python tools/benchmark_production_cuda.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/rtx-without-c.json --search-prior without-c \
  --b-search-mode contract-guided

# B-conditioned local 变换消融：固定 J=A+B、ContractView 与 C，只切换 L1/L0
python tools/benchmark_production_cuda.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/rtx-schedule-only.json --search-prior with-c \
  --b-search-mode schedule-only

# B/C 编译消融
python tools/audit_production_backend_search.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/b-c-audit.json --search-budget 3

# sm90a bundle（远端只执行）
INTENT_GPU_CHIP=sm_90a python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/sm90a-bundle.json

# RVV：本地 true-MLIR 编译；bundle 再传到 ssh rvv 执行
python tools/export_rvv_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/rvv-bundle.json
```

## 10. 当前边界

当前冻结实现已经达到 12/12 lowering/correctness；RTX 5090 与 H100 均有 6/12 达到或超过
Triton，整体 MLIR/Triton 几何平均分别为 1.013x 与 1.019x，但不是 12/12 全部超越。
剩余主要差距是小 shape LayerNorm/RMSNorm/fused-add
reduction 的固定指令/collective 开销。下一阶段先导入新增四个 J，再增加 source-neutral 的
producer/reduction/epilogue fusion、materialization/copy elimination，并把 RVV fusion 正式建模为
structural transformation trace。subgroup/workgroup/hierarchical reduction、FP32 FMA 与 target-supported
precision 作为 hard-safe concrete MLIR states 统一进入 Q/D filter，不新增 per-mechanism J 字段，也不按
kernel 名、Source recipe 或 Gate receipt 增加隐藏路径。RVV 当前证明 12/12 可编译、可执行、可搜索，
以及 fusion path 的内部机制收益；在
建立同机 Source baseline 前不声称端到端优于原实现。
