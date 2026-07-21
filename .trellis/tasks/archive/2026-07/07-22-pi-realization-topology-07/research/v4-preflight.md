# Protocol-v4 Formal Preflight（0 LLM）

## Frozen identity

- manifest: `test/real_llm_batch_28_v4.json`
- manifest schema: `intentir.real-llm-batch-28.v4`
- manifest digest: `0da031b95049acf5884e1adb60126253ab5441d6f7de142a0373e2659ab98b18`
- preflight: `artifacts/real_llm_batch_28_v4/preflight.json`
- preflight SHA-256: `74862a1402776d68988f3e704b2ceb22d61e89a5849e6a8c1da6c278e585fa30`
- observed date: `2026-07-22`
- online LLM calls: `0`

## Result

```text
source pins verified       28/28
symbol pins verified       28/28
source observer ready      28/28
replay arm ready           28/28
Triton compiled            26/26
CuTe compiled                2/2
overall_ready              true
blocking_capability        null
```

该结果只证明冻结 source、frontend/compiler、observer、anonymous harness 和 provider
configuration 已准备好；它不声称尚未生成的 Replay 会闭合，也没有调用 provider。

Swiglu 使用二维 fake tensor specialization 后仍产生历史相同 source compiler view：

- TTIR: `92e14ee61cbc4adce23bc6bf6813853d3983ff167732d40b1e2a934ce514f5fb`
- TTGIR: `75bc6af49a8d8d5b9e219d0036e60a29304029227f2d867b74aa156a516f4a16`

因此下一步只允许一次 `liger_swiglu_forward` protocol-v4 recovery：1 Lift、1 normal
Replay；只有 normal strict Gate1 hard-close 后，runner 才按冻结协议追加唯一 masked-J。
