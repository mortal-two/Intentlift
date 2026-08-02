# FlashAttention forward constant-token shape sweep

这组结果固定 `B*S=8192`、`H=32`、`D=64`，依次测量
`(B,S)=(16,512),(8,1024),(4,2048)`。Q/K/V/Out 为 FP16，LSE/TMP 为
FP32。这样既覆盖模型级序列长度，也保持被认证 Source 的 128-key
online-softmax recurrence，不把 reduction-order 改变混进 shape sweep。

三柱定义保持不变：`G_schedule=C-Fixed/Schedule-Only`，
`G_full=Schedule-Only/Full`。所有新测的 512/2048 点使用 CUDA event
single-shot、cold-data、256 MiB L2 eviction（eviction 不计时）、3 轮候选测量和
7 批 Source/Full 交替配对确认。1024 点复用同日正式结果。

主要结论：

- RTX 5090 D 的 `G_full` 在三个序列长度上稳定为 `1.172–1.189×`；严格确认下
  Full 相对 Triton Source 分别为 `1.062× / 0.996× / 0.977×`，即从小幅领先到
  2.3% 差距。
- H100 的同轮三柱 `G_full` 为 `2.421–3.326×`，证明 register-WGMMA/async-KV
  路径确实不在 direct schedule spine 中；但七批确认仍比 Triton Source 慢
  `5.0–12.2%`，因此不能用搜索阶段的瞬时 winner 宣称击败 Source。
- 新增的 BM128 warpgroup+async-KV 候选真实生成一个 MLIR/CUDA kernel，并把
  grid 从 4096 减为 2048；但其动态共享内存升到 182272 bytes。S=512 上两个
  BM128 状态约为 `747.94/748.54 us`，明显慢于 BM64 register winner，故未获胜。

逐项可画图数据在 [summary.csv](summary.csv)。
