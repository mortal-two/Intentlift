# Liger Softmax natural-shape development result（RTX 5090D）

> 状态：`uncertified_backend_dev` capability/performance evidence；不是新的 SCRC closure，计时为
> development warm CUDA-event，不进入最终 cold-data aggregate。

## Exact workload 与 provenance

- base snapshot：`2828eba9e064e190d5acd058aabb0cbe49e12ac62b78a67174d353680b1dbb06`；
- J_dev：`corpus/backend-j-dev-v1/liger_softmax_upstream_m4096_n4096_fp32.json`；
- shape/dtype：`M=4096,N=4096,FP32 boundary/accumulation`，与
  `python/tests/test_row_reduction_pk.py:94-120` 的真实 Liger PK shape一致；
- J_dev receipt：`c028948a30a90d64ac3d11fba529bfc36f8a604ce891a3946986296e67595837`；
- workload config digest：`0f1dca82342a1bffb2ee7d79c7d4b696499845cc274d1ea96f21cb81a5bc1064`；
- target：NVIDIA GeForce RTX 5090 D / `sm_120a`。

生成的 search bundle 不入 Git，SHA-256 为
`06773f8ca76d8da5491b644d8747625e9bf043dcf80c2a00abd4bbe19b7df123`。原始 target report 为
[report.json](report.json)，文件 SHA-256 为
`ade69b1fbe027ade09c8dfbd503c57d7abc2c27a30c938914ea663bfc5a38ded`。

```bash
PYTHONPATH=python python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/liger_softmax_m4096_n4096_e3_b64.json \
  --kernels softmax --search-budget 64 --arm E3 \
  --j-dev-variant corpus/backend-j-dev-v1/liger_softmax_upstream_m4096_n4096_fp32.json \
  --dev-workload-config spec/lowering/backend-natural-workloads-v1/liger_softmax_upstream_m4096_n4096_fp32.json
```

## Outcome

- 13 个真实 cubin states 全部通过 Target--RefDev 并计时，Source--RefDev 通过；
- winner 是单 launch `workgroup_hierarchical` reduction：256 threads、每线程16列、实际FP32 vec4；
- paired-confirmation target=`88.9134 us`，Triton Source=`90.8551 us`，Source/Target=`1.0218x`；
- 同一候选回执 IntentIR/Schedule-Only=`1.0000x`，C-prior/Erase-C=`1.0000x`。

J-derived logical boundary throughput 为约 `1.510 TB/s`，report 将其标成
`memory_roof_approaching`。但该值是 logical minimum traffic，不是硬件 DRAM transaction；并且 Softmax
含未建模 special-function ops，所以 `supports_memory_wall_claim=false`。可以说当前 fixed read-once state
已经接近逻辑带宽上限，不能说已经由 counter 证明“打满 DRAM memory wall”。本点证明成熟自然 shape
下的真实 MLIR/cubin/QD/roofline 能力和小幅 Source 优势，但不提供额外 E3/E2 正增益。
