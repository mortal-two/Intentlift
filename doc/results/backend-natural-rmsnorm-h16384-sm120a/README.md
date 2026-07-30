# Liger RMSNorm long-reduction development result（RTX 5090D）

> 状态：`uncertified_backend_dev` capability/performance evidence；不是新的 SCRC closure，计时为
> development warm CUDA-event，不进入最终 cold-data aggregate。

## Exact workload 与 provenance

- base snapshot：`63514c55f14ab5e26e16285f3a6c19588584a079ac2217834df6dc8558426406`；
- J_dev：`corpus/backend-j-dev-v1/liger_rms_norm_sweep_m2048_h16384_fp32.json`；
- shape/dtype：`M=2048,H=16384,FP32 boundary/accumulation`；H来自冻结实验协议的1K--32K
  power-of-two sweep，Liger `calculate_settings` 对该点使用 `BLOCK_SIZE=16384,num_warps=16`；
- J_dev receipt：`37940f4374f9809c844d79f82f758304fb77b42ef77e6daa51010065ba1dbb28`；
- workload config digest：`03a368c45f9ff2dcee30d058e158c7d2334b5f71fdd24d9154c4dfd360297dd8`；
- target：NVIDIA GeForce RTX 5090 D / `sm_120a`。

生成的 search bundle 不入 Git，SHA-256 为
`b6c4c929bdc43abf0d8cac8675847fd8de4441571a5162fad8bbd37dfbc55580`。原始 target report 为
[report.json](report.json)，文件 SHA-256 为
`2bb8b55d8897d70937103ceafdbeec1e4888c0ae0dcfc67587b665fd2a5e5ca5`。

```bash
PYTHONPATH=python python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/liger_rms_m2048_h16384_e3_b96.json \
  --kernels liger_rms_norm --search-budget 96 --arm E3 \
  --j-dev-variant corpus/backend-j-dev-v1/liger_rms_norm_sweep_m2048_h16384_fp32.json \
  --dev-workload-config spec/lowering/backend-natural-workloads-v1/liger_rms_norm_sweep_m2048_h16384_fp32.json
```

## Outcome

- 39个真实 states 全部通过 Target--RefDev并计时，Source--RefDev通过；
- single-stage winner：1024 threads、每线程16列、实际FP32 vec4、FMA，单 launch；
- paired-confirmation target=`176.0939 us`，Triton Source=`184.6030 us`，Source/Target=`1.0483x`；
- 同一回执 IntentIR/Schedule-Only=`1.0008x`，C-prior/Erase-C=`1.0000x`；
- 三 launch `multi_stage_partial_finalize` 确实执行，最佳为256 threads、parts=16、`264.6531 us`，
  比 single-stage 慢约50.3%。

因此这个点把“长规约是否需要多阶段”从设计问题变成了实测结论：single-stage仍能以16元素/thread保持
read-once并接近logical roof；multi-stage为partial pass读取X，又在输出epilogue重读X/W，两次额外launch
无法由并行度补回。该状态仍保留在搜索空间，因为更长规约、只有per-row输出的图或SplitK可能改变成本；
但不能把候选存在写成已获得结构加速。logical roof仍因rsqrt等special-function未建模而不支持硬件
memory-wall claim。
