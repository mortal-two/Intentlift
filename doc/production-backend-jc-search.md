# Accepted J+C production backend：真实 MLIR lowering、B 合同与 C 搜索

> 状态：production catalog 已 append-only 扩展为 content-addressed 的 **16 个**闭合 J；2026-07-26。
> 文中 RTX 5090/H100 的历史 aggregate 仍只对应原 closed-12，不得重标为 16-kernel 结果。

本文记录已冻结 16 个 SCRC Gate1/Gate2 闭合 Kernel 的 production backend 实现。原 12 项性能与消融数值见
[production-backend-report.md](results/production-backend-closed-12/production-backend-report.md)。
本轮 16 项编译覆盖、CUDA Q/D smoke 与新增 RVV state 对照见
[backend-implementation-smoke-2026-07-26.md](results/production-backend-closed-16/backend-implementation-smoke-2026-07-26.md)。

新增四项已经通过同一个 catalog loader 导入；snapshot 只复制已接受 artifact/C/evidence，未重新 Lift、
未修改 J，也不把 `/tmp` 作为长期 truth：

| case | frozen snapshot digest | evidence kind |
|---|---|---|
| xFormers RMSNorm-add | `e8e5451fa837443024a435b34b125f5c9f2261bdaf178842f80898875cf727be` | zero-model Gate2 recheck |
| Liger Sparsemax | `ca974aed6b5954b197a063f0e96d62652d569b5728608818adcdbfbc3cb11fda` | zero-model Gate2 recheck |
| Liger KLDiv | `b39a3df6d64782be475b5b459f6b7dc01efdb3356fa85abe22a75334f296df7a` | certified run report |
| FlashAttention one-pass LayerNorm | `76ffde5e8eb4e53d0fca5afad8bb1d44ad4ba3f6c9df764dc67b21aa95402480` | zero-model Gate2 recheck |

当前 16 项均无 typed contraction；Flash `do_o_dot` 是 FP32 multiply + row reduction。因此当前性能
推进不把 TF32 误套到 row reduction；TF32/IEEE 精度状态留给后续 `tl.dot`/matmul/完整 Attention J。

## 1. 输入边界

core lowerer 的持久输入只有：

- accepted canonical `J=A+B`；
- 可选、可擦除的 `C`；
- target capability。

它不读取 Source、Replay、π、Gate verdict、Q、kernel 名或 provider 身份来决定
lowering。`corpus/production-jc-v1/` 保存 16 个 content-addressed
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

这里的 Q/D filter 是已经实现的唯一通用 `SearchAndCloseController`，不是每个 Kernel 的 Python
emitter 或 validator。core MLIR compiler 不读取 Q；controller 只通过 signature-driven opaque harness
执行 Source/Replay/RefJ/target，并使用冻结 comparator。`D_search` 对每个具体 post-pass MLIR state 做
direct target–RefJ；性能选择后的固定 winner 在输入 digest 不相交的 `D_cert` 上直接比较
target–Source、target–Replay、target–RefJ，禁止 tolerance 传递。缺 Replay 时可以留下 search evidence，
但不能产生 formal backend-closure receipt。

## 2. J、B 与 constexpr

`materialize_j()` 先对 canonical J 做 deterministic materialization。当前支持：

- functional boundary projection；
- dense predicate/fill materialization；
- exact value-dimension relations；
- full-axis logical step：包括 no-scan/null-init carry，也包括 scan 输入与 typed literal/reference
  carry init；该投影与 RefJ 的完整逻辑轴一步语义相同，不把 C tile 当语义；
- J-declared constexpr 对 typed `effect_if` 的精确分支专化；普通 scalar `value_if` 不伪装成
  constexpr，而在 M0 中保留为标准 `scf.if`；
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
注入未支持 `may_alias` 作为负对照：历史 closed-12 的 with-C 与 Erase_C 两臂都在 codegen 前以
`alias_relation` 拒绝。

Sparsemax 复用同一个 functional-boundary projection 消费 authored access-local guards；scan 上的
reduction relation 与 reduce 使用同一闭合合同。若 projection receipt 未覆盖某个 guard，ContractView
仍返回 `missing/unsupported`，不会仅因图中出现 select 而猜测它已被消费。

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
transition。实现已把它从 `RiscvSchedule` 拆成独立 `RiscvTransformationState`；materialized 与 fused
state 分别搜索 O2/O3、vectorize on/off，post-pass Linalg op count 与 topology digest 来自真实 MLIR
receipt，不是 pass 开关自报。

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

reduction topology 与 FP32 FMA 已进入 production lowering-state search。C++ pass 在结果 MLIR 上记录
实际 shuffle/barrier/all-reduce/FMA 数量与 transformation trace；请求 FMA 但实际未产生 FMA 的状态会
按 post-pass MLIR digest 去重。它们标记为 `observation-changing`，在 `D_search` 通过 target–RefJ 后
才可计时，固定 winner 再走 `D_cert` 三条 direct edges。现有 `reduce.assoc`、value-dimension equality、
predicate/output/alias projection 继续参与 hard admission 或 codegen，而不是数值权限开关。

已消费：Liger RMSNorm、fused-add RMSNorm、LayerNorm、Softmax、JSD、PolyNorm、
xFormers RMSNorm、xFormers RMSNorm-add、KLDiv、Sparsemax 与 Flash one-pass LayerNorm。KLDiv 的
scan/carry 在 J-only full-axis materializer 展开；Sparsemax 的 scan 与 logical `and` 有 generic/target
lowering；Flash 的普通 scalar `value_if` 保留为 `scf.if`，C++ row state 再对 pure arms 做 predication。
该 state 的实际 receipt 为 2 次 value-if predication、10 次 constexpr select folding，而非 Python 分支。

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
5. 在 `D_search` 上 direct target–RefJ filter；
6. 通过者的 device timing；
7. top-finalist robust tournament 与 fastest candidate selection；
8. 固定 winner 在独立 `D_cert` 上 direct target–Source/Replay/RefJ closure。

运行结果不用于放宽 memory/effect/ABI hard boundary。正确性测试与 performance selection 分离；
历史 frozen-12 winner 另外与 canonical J 比较；新 controller 已对 observation-changing state 执行通用
Q/D feasibility 与 direct-edge closure。baseline 是 SCRC 已使用的 exact declarative Triton
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

新增 J 的 2026-07-26 原生 RVV smoke 进一步验证了同一机制：KLDiv 的 8 个对象全部由真实
MLIR→LLVM→riscv64 链生成并通过 canonical-J correctness；fixed-state schedule-only best 为
14.1428 μs，加入 whole-graph fused state 后 best 为 9.1056 μs，即 **1.553x E3/E2 增量**。
最快 fused-vector state 含 120 条 RVV 指令、9.3754 μs；全局 winner 是更快的 fused-scalar O3，说明
搜索会选择实测 state，而不把“出现向量指令”当目标。Flash one-pass LayerNorm 当前两臂均为
36.9453 μs（1.000x），如实记录为该结构变换不适用/无增益。

RVV winner 不强制“必须含向量指令”：例如包含 `tanhf` 的 DYT 在该小 shape 上标量候选更快，
搜索会如实选择它；其他候选与 receipt 仍记录 RVV 指令数。当前没有与 RVV 上的 Source/Triton
实现进行配对，因此 1.157x 只能解释为 whole-graph fusion path 的内部因果消融，不能解释为超越
Source。KLDiv 的 1.553x 是同一目标、同一 J、相同 schedule domain 下 materialized 与 fused concrete
MLIR state 的内部因果消融；RVV 侧尚未接入 Source/Replay 的独立 D_cert，因此不冒充正式端到端
backend closure 或超越 Source 的证据。

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

# 正式 CUDA E2/E3/N0-N3：arm 由 machine-consumed spec 冻结；Replay manifest
# 必须给出 content hash，缺 Replay 时 --require-backend-closure 会 fail closed。
# 报告会绑定 canonical experiment spec digest；当前 v1 为 60ca646b...050a7。
python tools/run_production_backend_ablation.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/cuda-ablation.json --arms E2 E3 N0 N1 N2 N3 \
  --candidate-budget 96 --replay-manifest /path/to/replay-manifest.json \
  --require-backend-closure

# sm90a bundle（远端只执行）
INTENT_GPU_CHIP=sm_90a python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/sm90a-bundle.json

# RVV：本地 true-MLIR 编译；bundle 再传到 ssh rvv 执行
python tools/export_rvv_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/rvv-bundle.json
scp /tmp/rvv-bundle.json tools/run_rvv_benchmark_bundle.py rvv:/tmp/
ssh rvv python3 /tmp/run_rvv_benchmark_bundle.py \
  --bundle /tmp/rvv-bundle.json --out /tmp/rvv-report.json
```

## 10. 当前边界

当前冻结实现达到 **16/16 deterministic materialization、generic cubin compilation**，并且每项至少有
一个 C++ whole-graph specialized path 生成单 cubin。新增四项的真实消费者包括：inout/post-state、
access-local guard、scan/reduction relation、loop scan/carry、runtime value-if 与多输出 half-storage
epilogue。CUDA 的正式 Q/D smoke 已在 KLDiv 与 Flash one-pass LayerNorm 上直接通过 held-out
target–Source、target–Replay、target–RefJ；这证明 controller 与 opaque Replay 接口真实可用，但不是
16-kernel 的论文 aggregate。

RTX 5090/H100 的已发布 aggregate 仍是 closed-12：两机各有 6/12 达到或超过 Triton，整体
MLIR/Triton 几何平均分别为 1.013x 与 1.019x。新增 Flash LayerNorm 的 equal-budget E2/E3 当前约与
Triton 持平，E3 的 2 个实际 FMA 没有产生显著额外收益；不能为故事挑选性删除。下一步性能重点是
CUDA 上 producer/reduction/epilogue fusion、materialization/copy elimination，以及在更多适用 J 上跑满
machine spec 冻结的 E2/E3/E4、L0/L1、N0--N3。当前 16 项没有 typed contraction，因此 precision/TF32
axis 会如实记录 `unavailable/not_matched`，不制造候选。

RVV 已证明原 closed-12 全量可编译/执行/搜索，并在新增 KLDiv 上得到 1.553x 的真实 structural-state
增量。其 target-side runner 当前是 target–RefJ correctness + timing；在加入预计算 Source/Replay 的
独立 D_search/D_cert bundle 前，不声称 RVV 端到端 backend closure 或超越 Source。
