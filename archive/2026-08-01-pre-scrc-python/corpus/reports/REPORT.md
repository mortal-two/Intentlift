# IntentIR kernel corpus

Real high-performance-library Triton kernels lifted to IntentIR by the **live provider agent** (gpt-5.5, OpenAI responses API), then verified against the actual library kernel and lowered to linalg.

- **extract**: `intentir.extraction` calls the provider; JSON is NOT hand-authored (see each `provenance.json`).
- **verify**: `interpret(JSON)` vs the real `flag_gems.<op>` output, max relative error < 1e-3 (the dynamic lifting certificate).
- **lower**: emitted `intent` dialect MLIR through `-convert-intent-to-linalg`, zero residual `intent.*` ops.

| kernel | library | extract | verify | lower | rel err | #ops |
|---|---|---|---|---|---|---|
| softmax | flag_gems | ✅ | ✅ | ✅ | 5.98e-08 | 5 |
| log_softmax | flag_gems | ✅ | ✅ | ✅ | 9.84e-08 | 6 |
| silu | flag_gems | ✅ | ✅ | ✅ | 1.23e-07 | 2 |
| gelu_tanh | flag_gems | ✅ | ✅ | ✅ | 5.61e-08 | 13 |
| relu | flag_gems | ✅ | ✅ | ✅ | 0.00e+00 | 2 |
| rms_norm | flag_gems | ✅ | ✅ | ✅ | 1.48e-07 | 7 |
| matmul | flag_gems | ✅ | ✅ | ✅ | 2.04e-07 | 1 |
| sigmoid | flag_gems | ✅ | ✅ | ✅ | 1.22e-07 | 1 |
| tanh | flag_gems | ✅ | ✅ | ✅ | 1.19e-07 | 1 |
| exp | flag_gems | ✅ | ✅ | ✅ | 1.37e-07 | 1 |
| abs | flag_gems | ✅ | ✅ | ✅ | 0.00e+00 | 1 |
| log | flag_gems | ✅ | ✅ | ✅ | 8.60e-08 | 2 |
| add | flag_gems | ✅ | ✅ | ✅ | 0.00e+00 | 1 |
| mul | flag_gems | ✅ | ✅ | ✅ | 0.00e+00 | 1 |
| maximum | flag_gems | ✅ | ✅ | ✅ | 0.00e+00 | 1 |

**15/15 kernels** pass all three stages.
