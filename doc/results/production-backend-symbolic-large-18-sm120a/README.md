# closed-18 symbolic-large / RTX 5090D

这组结果是 `symbolic-large-v1` 在 RTX 5090D (`sm_120a`) 上的全 18 横向
CUDA 主实验，不是从适用子集挑选的示例。E2/E3 均使用 total candidate budget 32；
每个可执行候选先通过 Target--RefJ，再在 25 ms untimed code/P-state warm-up 后做
L2-controlled single-shot CUDA-event 计时。每个正式样本前在 timed interval 外用
256 MiB working set 驱逐 96 MiB L2。

## 冻结身份

- compiler commit：`ac48fad6`（结果生成时 HEAD）；
- toolchain digest：`3a99d5518559e264be861d71b2d8bd1cf6525bee51fea8ca72f4d3ca53c9114b`；
- ablation spec digest：`28c2117f35e1360fa4f9b4c85f4af27edf343630d0d5be000eef0b3b4e9c8f9a`；
- measurement spec digest：`7797827a12b0563084dcc8c838c3b3017b84ae9381c9ad87e6379210b72c4fcf`；
- summary SHA-256：`d08214e2853fd3aac3ce7318e4df7aaee2b3aa59cf4e01f0c50f050b78b80985`；
- E2 compressed report SHA-256：`fd75b42a83b784c945da97936b255e70462dd851683b53fb21e7e9cb80db19d2`；
- E3 compressed report SHA-256：`5cfe0cca75e1eb6f126441ad28f262ea5d035e951ad9240ab3f226e1c22e8b99`。

## 结果

- 18/18 E2、18/18 E3 执行成功；17 项有 exact Source baseline，`masked_copy`
  保持 Target--RefJ-only；
- primary E3/E2：`1.0321337x`。它在同一 E3 candidate receipt 中比较
  fixed-structure schedule-spine best 与 expanded best；
- 独立 E2/E3 进程的 diagnostic cross-report ratio：`1.0062950x`；
- E2/Source：`1.0857808x`；E3/Source：`1.1061843x`；
- C-prior/Erase-C：`1.0036794x`；
- primary E3/E2 winner：7/18；结构状态扩展实际形成于 16/18；
- DYT primary E3/E2 `1.5687501x`，其同回执 schedule-spine/expanded best
  为 `4.016/2.560 us`；xFormers RMSNorm 为 `1.0434782x`；
- 相同 executable 或没有可测额外收益的格子记为 `1.0x`，不把两个进程间的
  微秒级时钟漂移算成 lowering-state gain。

`e2-e3-budget32-spine-paired-summary.json` 是便于审阅的全 18 压缩摘要；两个
`.json.gz` 文件保留完整候选、correctness、finalist、C-order、schedule-spine 与
Source/target paired-batch receipt。含 cubin/输入的 portable bundle 共约 1.7 GiB，未纳入 Git；
可按 `doc/production-backend-jc-search.md` 的命令从同一 commit 重新导出。
