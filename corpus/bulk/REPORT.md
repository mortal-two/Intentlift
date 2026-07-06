# C1 universe pilot — composability-measurement gate

- extraction success: **97.3%** (72/74)
- composability-failure rate (extract-ok but linalg-lower-fail): **0.0%** (0/72)
- composes (linalg leg + GPU-pass): **100.0%** (of which 4 ride a fused GPU pass: iota/scan/layout/gather/...)
- verify pass: **91.9%** (68/74)

| op | family | extract | lower | verify | rel | class |
|---|---|---|---|---|---|---|
| tanh | unary | ✅ | ✅ | ✅ | 1.19e-07 | ok |
| sigmoid | unary | ✅ | ✅ | ✅ | 1.22e-07 | ok |
| silu | unary | ✅ | ✅ | ✅ | 1.83e-07 | ok |
| gelu | unary | ✅ | ✅ | ✅ | 6.20e-08 | ok |
| relu | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| abs | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| neg | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| exp | unary | ✅ | ✅ | ✅ | 1.55e-07 | ok |
| sqrt | unary | ✅ | ✅ | ✅ | 9.73e-08 | ok |
| log | unary | ❌ | ❌ | ❌ | — | extraction_fail |
| reciprocal | unary | ✅ | ✅ | ✅ | 5.96e-08 | ok |
| rsqrt | unary | ❌ | ❌ | ❌ | — | extraction_fail |
| atan | unary | ✅ | ✅ | ✅ | 8.92e-08 | ok |
| tan | unary | ✅ | ✅ | ✅ | 3.67e-08 | ok |
| exp2 | unary | ✅ | ✅ | ✅ | 1.55e-07 | ok |
| celu | unary | ✅ | ✅ | ✅ | 4.16e-08 | ok |
| log_sigmoid | unary | ✅ | ✅ | ✅ | 5.98e-08 | ok |
| add | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| sub | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| mul | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| div | binary | ✅ | ✅ | ✅ | 7.24e-08 | ok |
| maximum | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| minimum | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| pow | binary | ✅ | ✅ | ✅ | 1.45e-07 | ok |
| ge | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| gt | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| lt | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| le | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| ne | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| sum | reduce | ✅ | ✅ | ✅ | 7.49e-07 | ok |
| mean | reduce | ✅ | ✅ | ✅ | 3.86e-07 | ok |
| prod | reduce | ✅ | ✅ | ✅ | 2.80e-33 | ok |
| max | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| min | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| mm | blas | ✅ | ✅ | ✅ | 1.62e-07 | ok |
| bmm | blas | ✅ | ✅ | ✅ | 2.26e-07 | ok |
| addmm | blas | ✅ | ✅ | ✅ | 1.71e-07 | ok |
| mv | blas | ✅ | ✅ | ✅ | 1.19e-07 | ok |
| outer | blas | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| zeros | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| ones | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| eye | init | ✅ | ❌ | ✅ | 0.00e+00 | gpu_pass_lower |
| linspace | init | ✅ | ❌ | ✅ | 5.96e-08 | gpu_pass_lower |
| full_like | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| geglu | fused | ✅ | ✅ | ❌ | — | verify_fail |
| glu | fused | ✅ | ✅ | ❌ | — | verify_fail |
| clamp | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| threshold | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| nan_to_num | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| isnan | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| isinf | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| isfinite | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| logical_not | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| fill | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| logical_and | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| logical_or | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| logical_xor | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| isclose | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| lerp | binary | ✅ | ✅ | ✅ | 8.85e-08 | ok |
| masked_fill | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| all | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| any | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| trace | reduce | ✅ | ✅ | ✅ | 2.52e-07 | ok |
| cumsum | scan | ✅ | ❌ | ✅ | 3.67e-07 | gpu_pass_lower |
| dot | blas | ✅ | ✅ | ✅ | 7.75e-08 | ok |
| vdot | blas | ✅ | ✅ | ✅ | 2.51e-06 | ok |
| baddbmm | blas | ✅ | ✅ | ✅ | 2.08e-07 | ok |
| addmv | blas | ✅ | ✅ | ✅ | 1.31e-07 | ok |
| addr | blas | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| rms_norm | norm | ✅ | ✅ | ✅ | 1.23e-07 | ok |
| layer_norm | norm | ✅ | ✅ | ❌ | — | verify_fail |
| zeros_like | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| ones_like | init | ✅ | ✅ | ❌ | — | verify_fail |
| logspace | init | ✅ | ❌ | ✅ | 9.54e-08 | gpu_pass_lower |

Gate: composability-fail 0.0% (< 20% ? **C2 CAN scale**)
