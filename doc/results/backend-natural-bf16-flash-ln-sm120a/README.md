# Flash one-pass LayerNorm BF16 envelope development result（RTX 5090D）

> 状态：`uncertified_backend_dev` capability/performance evidence；不是新 SCRC closure，也不是最终主表。

## Exact workload

- base snapshot：`76ffde5e8eb4e53d0fca5afad8bb1d44ad4ba3f6c9df764dc67b21aa95402480`；
- J_dev：`corpus/backend-j-dev-v1/flash_layer_norm_1pass_m2048_n4096_bf16.json`；
- shape/dtype：`M=2048,N=4096,BF16 boundary,FP32 accumulation`；
- evidence boundary：upstream manifest声明`rows=128..4096,cols=256..8192,dtype={FP16,BF16}`；其
  archived concrete specialization仍是FP16 `[128,1024]`，本文不声称上游已实测BF16；
- target：NVIDIA GeForce RTX 5090 D / `sm_120a`；
- timing：development warm CUDA-event，尚未替代最终冻结协议。

## Reproducible receipts

- raw target report：[report.json](report.json)，SHA-256
  `fe006d0ecceb6f04756950d376be48bf3c5b763725e9fcd2b853c0aa3f04f080`；
- precompiled search bundle SHA-256
  `8510db946ed6b310f77caf2f1c457bce816d688ee78af36d0c8cd591581469ea`；bundle约151 MiB，
  不纳入Git，report保存所有winner/candidate/binary digest与correctness receipt；
- J_dev receipt digest：
  `f1b2890ba23b7c18a6323346305499d9f5c3309765243d95e31ee80ba106507b`；
- workload config digest：
  `8d24216a691d04f3c1c28de4f3ed79e4d0da12b48bf27056da26577c0d1bdb4b`。

导出命令：

```bash
PYTHONPATH=python python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/flash_ln_bf16_m2048_n4096_vec_e3_b64.json \
  --kernels flash_layer_norm_1pass --search-budget 64 --arm E3 \
  --j-dev-variant corpus/backend-j-dev-v1/flash_layer_norm_1pass_m2048_n4096_bf16.json \
  --dev-workload-config spec/lowering/backend-natural-workloads-v1/flash_layer_norm_1pass_m2048_n4096_bf16.json
```

## Outcome

- 20个transported concrete cubin states；19个通过Target--RefDev并计时；
- 1个generic materialized graph因host block-argument pointer transport缺口明确记录
  `runtime_or_transport_failed`，没有拖掉其余候选；
- Source--RefDev通过；BF16输出在NumPy/Torch/portable receipt中保持真实BF16 storage；
- winner实际schedule：256 threads、8 warps、BF16 vec4、one block/row、FP32 hierarchical reduction、
  reciprocal-approx state；
- winner `69.5268 us`，Source Triton `65.9875 us`，Source/Target=`0.9491x`；
- 同一回执 expanded/Schedule-Only=`1.0023x`。

因此本结果证明真实BF16消费、vec4 packetization、Q/D与候选隔离能力，但它是一个诚实的负性能结果：
当前后端仍慢约5.4%，且该图没有形成足够强的额外规约拓扑。后续结构收益应在Softmax、RMSNorm、
Sparsemax/scan等适用图上横向验证，不能把本结果挑成大E3/E2正例。
