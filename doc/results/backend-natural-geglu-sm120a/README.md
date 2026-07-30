# Liger GeGLU natural-shape development result（RTX 5090D）

> 状态：`uncertified_backend_dev` capability/performance evidence；不是新的 SCRC closure，计时为
> development warm CUDA-event，不进入最终 cold-data aggregate。

## Exact workload 与 provenance

- base snapshot：`42155b91a8e1a9e400d39be4d4928a1b10c92dd222353f11cf6db3be0da319c0`；
- J_dev：`corpus/backend-j-dev-v1/liger_geglu_upstream_m4096_n4096_fp32.json`；
- shape/dtype：`M=4096,N=4096,FP32 boundary`，与
  `python/tests/test_elementwise_pk.py:195-222` 的真实 Liger PK shape一致；
- J_dev receipt：`bba2029a753304fd587813614759bf04d4e3cbcb38d6a634dfb2a16cd957a230`；
- workload config digest：`7c709807fcd4dbc4e893494eee482cb3e3a41422026ef3f0adae1a1ae17f93ae`；
- target：NVIDIA GeForce RTX 5090 D / `sm_120a`。

生成的 search bundle 不入 Git，SHA-256 为
`50f8abeb6c9e5c13c2be90074ca948bbb5a228a3b5fd97c91b88ace1d2077266`。原始 target report 为
[report.json](report.json)，文件 SHA-256 为
`4f89e7ae3137417d411103f97b5945d962efd6b8f410173db379c4e89848bba4`。

```bash
PYTHONPATH=python python tools/export_cuda_benchmark_bundle.py \
  --snapshot-dir corpus/production-jc-v1 \
  --out /tmp/liger_geglu_m4096_n4096_e3_b64.json \
  --kernels geglu --search-budget 64 --arm E3 \
  --j-dev-variant corpus/backend-j-dev-v1/liger_geglu_upstream_m4096_n4096_fp32.json \
  --dev-workload-config spec/lowering/backend-natural-workloads-v1/liger_geglu_upstream_m4096_n4096_fp32.json
```

## Outcome

- 25 个真实 cubin states进入 Q/D；13 个 `stable-ex2` states 通过 Target--RefDev并计时；
- 12 个 `native-ptx` tanh states 被统一 Q/D 拒绝，典型最大绝对误差约 `2.2054e-6`；它们没有因更快
  而越过 correctness policy；
- winner 是单 launch row elementwise state：512 threads、`stable-ex2` tanh；
- paired-confirmation target=`131.2597 us`，Triton Source=`131.3138 us`，Source/Target=`1.0004x`；
- 同一候选回执 IntentIR/Schedule-Only=`1.0000x`，C-prior/Erase-C=`1.0000x`。

J-derived logical throughput约 `1.534 TB/s`，同样因 logical traffic 与未建模 tanh SFU 而不能声明硬件
DRAM memory wall。这个自然大 shape 说明：whole-graph read-once state 已把 generic materialization 的巨大
内部差距收敛到成熟 Triton parity；继续增加局部开关不会自动产生 E3/E2 增益。它同时给出一个重要的
负例：Q/D 会真实淘汰不满足当前 comparator 的数值候选，而不是把 tolerance 当作无条件许可。
