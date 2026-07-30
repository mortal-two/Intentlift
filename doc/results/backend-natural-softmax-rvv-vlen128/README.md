# Liger Softmax natural-shape RVV development result（VLEN=128）

> 状态：**历史诊断，不能解释为最终三臂实验。** 该回执是 `uncertified_backend_dev`、
> Target--RefJ-only、warm development measurement；旧 `frozen-C` 把未经 target role/axis 投影的
> C 排序结果当控制，旧 fused path 也没有 read-once multi-use protection，因此既不是 v3 C-Fixed，
> 也不是当前 IntentIR path。它只证明早期自然 J_dev、retile/object search 与预算边界可在 RVV 执行。

## Exact input 与 compiler receipts

- J_dev：`corpus/backend-j-dev-v1/liger_softmax_upstream_m4096_n4096_fp32.json`；
- shape/dtype：`M=4096,N=4096,FP32`；J_dev receipt
  `c028948a30a90d64ac3d11fba529bfc36f8a604ce891a3946986296e67595837`；
- target：native riscv64，VLEN=128，进程固定CPU0；
- precompiled bundle：48个不同native objects、4个object-digest duplicates、0 compile rejection；bundle
  SHA-256 `05f06c7f092e6389513b3ffaa0d22f7eae8b50e3e5855e3b4b7a968a0841d603`，不纳入Git；
- 每个tile state的`parallel/reduction_tile_requested`与`*_applied`、generated loops、structured MLIR/
  LLVM/assembly/object digest均来自真实compiler receipt，不是report参数自报；
- 原始远端报告：[report.json](report.json)，SHA-256
  `8319fab25162798544221bab314829361844c53526795b040d0315dc7e6e37a5`。

```bash
PYTHONPATH=python python tools/export_rvv_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 --kernels softmax \
  --j-dev-variant corpus/backend-j-dev-v1/liger_softmax_upstream_m4096_n4096_fp32.json \
  --dev-workload-config spec/lowering/backend-natural-workloads-v1/liger_softmax_upstream_m4096_n4096_fp32.json \
  --out /tmp/rvv_softmax_m4096_n4096_bundle.json

scp /tmp/rvv_softmax_m4096_n4096_bundle.json tools/run_rvv_benchmark_bundle.py rvv:/tmp/
ssh rvv python3 /tmp/run_rvv_benchmark_bundle.py \
  --bundle /tmp/rvv_softmax_m4096_n4096_bundle.json \
  --out /tmp/rvv_softmax_m4096_n4096_report_v4.json \
  --regime warm --search-budget 4 --warmup 1 --iterations 1 --batches 2
```

## Bounded-search outcome

以下数字全部保留为 v4 runner 的历史故障定位资产，不进入 `C-Fixed / Schedule-Only / IntentIR`
主图或 C prior 小表；当前实现与结果以 v3 bundle/v5 runner 的新回执为准。

runner v4把预算变成真实访问边界，而不是计时后再截断：

- C-prior E3前缀、C-prior/Erase-C schedule前缀、frozen/no-retile控制取并集，共访问10个object；
- 10/10通过Target--RefJ并计时；其余38个编译成功对象明确记`not_visited_due_budget`；
- frozen-C=`793.841 ms`；Schedule-Only best=`728.096 ms`，即schedule-search/frozen-C=`1.0903x`；
- 最佳非零retiled state=`771.878 ms`，相对frozen-C为`1.0285x`；但no-retile=`728.096 ms`，所以
  retiled-only/no-retile=`0.9433x`。这两个指标分开记录，不把搜索最终关闭tiling的收益冒充成retile收益；
- whole-graph fused最佳约`728.292 ms`，未超过no-retile Schedule-Only，故IntentIR/Schedule-Only=`1.0000x`；
- C-prior schedule前4最佳=`771.878 ms`，Erase-C前4最佳=`603.072 ms`，C-prior/Erase-C=`0.7813x`。

因此这个点同时证明两件事。第一，冻结的CUDA式C不能直接当RVV schedule；非零retile本身相对冻结
C约提升2.8%，而完整schedule search通过选择no-retile获得约9.0%收益。第二，当前C scorer在这个自然RVV shape上排序失配，且whole-graph fusion
并非自动胜出。C只排序、candidate universe与correctness不变，所以负C结果不会影响合法性；它为后续
target-aware C prior学习/消融提供了真实基线。
