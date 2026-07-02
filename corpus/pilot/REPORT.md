# C1 universe pilot — composability-measurement gate

- extraction success: **91.7%** (22/24)
- composability-failure rate (extract-ok but lower-fail): **36.4%** (8/22)
- verify pass: **83.3%** (20/24)

| op | family | extract | lower | verify | rel | class |
|---|---|---|---|---|---|---|
| cos | unary | ✅ | ✅ | ✅ | 5.96e-08 | ok |
| sin | unary | ✅ | ✅ | ✅ | 5.96e-08 | ok |
| erf | unary | ✅ | ✅ | ✅ | 5.65e-08 | ok |
| elu | unary | ✅ | ✅ | ✅ | 3.17e-08 | ok |
| softplus | unary | ✅ | ✅ | ✅ | 6.64e-08 | ok |
| addcdiv | binary | ✅ | ✅ | ✅ | 9.92e-08 | ok |
| addcmul | binary | ✅ | ✅ | ✅ | 9.41e-08 | ok |
| mse_loss | binary | ✅ | ❌ | ✅ | 5.99e-08 | lowering_fail |
| eq | binary | ✅ | ❌ | ✅ | 0.00e+00 | lowering_fail |
| amax | reduce | ✅ | ❌ | ✅ | 0.00e+00 | lowering_fail |
| vector_norm | reduce | ✅ | ❌ | ✅ | 1.19e-07 | lowering_fail |
| std | reduce | ✅ | ❌ | ✅ | 0.00e+00 | lowering_fail |
| var_mean | reduce | ✅ | ❌ | ✅ | 6.11e-05 | lowering_fail |
| count_nonzero | reduce | ✅ | ❌ | ✅ | 0.00e+00 | lowering_fail |
| batch_norm | norm | ❌ | ❌ | ❌ | — | extraction_fail |
| weight_norm | norm | ✅ | ✅ | ❌ | 1.95e+00 | verify_fail |
| instance_norm | norm | ❌ | ❌ | ❌ | — | extraction_fail |
| softmax | softmax | ✅ | ✅ | ✅ | 7.67e-08 | ok |
| log_softmax | softmax | ✅ | ✅ | ✅ | 9.02e-08 | ok |
| arange | init | ✅ | ❌ | ✅ | 0.00e+00 | lowering_fail |
| full | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| silu_and_mul | fused | ✅ | ✅ | ✅ | 1.05e-07 | ok |
| gelu_and_mul | fused | ✅ | ✅ | ✅ | 6.35e-08 | ok |
| reglu | fused | ✅ | ✅ | ❌ | — | verify_fail |

Gate: composability-fail 36.4% (< 20% ? **C2 needs pass-boundary triage first**)
