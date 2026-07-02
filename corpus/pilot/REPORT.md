# C1 universe pilot — composability-measurement gate

- extraction success: **90.9%** (20/22)
- composability-failure rate (extract-ok but lower-fail): **0.0%** (0/20)
- verify pass: **81.8%** (18/22)

| op | family | extract | lower | verify | rel | class |
|---|---|---|---|---|---|---|
| cos | unary | ✅ | ✅ | ✅ | 5.96e-08 | ok |
| sin | unary | ✅ | ✅ | ✅ | 5.96e-08 | ok |
| erf | unary | ✅ | ✅ | ✅ | 5.76e-08 | ok |
| elu | unary | ✅ | ✅ | ✅ | 2.86e-08 | ok |
| softplus | unary | ✅ | ✅ | ✅ | 6.18e-08 | ok |
| addcdiv | binary | ✅ | ✅ | ✅ | 8.36e-08 | ok |
| addcmul | binary | ✅ | ✅ | ✅ | 1.02e-07 | ok |
| mse_loss | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| eq | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| amax | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| vector_norm | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| std | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| var_mean | reduce | ✅ | ✅ | ✅ | 6.09e-05 | ok |
| count_nonzero | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| batch_norm | norm | ❌ | ❌ | ❌ | — | extraction_fail |
| weight_norm | norm | ✅ | ✅ | ❌ | 2.23e+00 | verify_fail |
| instance_norm | norm | ❌ | ❌ | ❌ | — | extraction_fail |
| arange | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| full | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| silu_and_mul | fused | ✅ | ✅ | ✅ | 6.01e-08 | ok |
| gelu_and_mul | fused | ✅ | ✅ | ✅ | 7.59e-08 | ok |
| reglu | fused | ✅ | ✅ | ❌ | — | verify_fail |

Gate: composability-fail 0.0% (< 20% ? **C2 CAN scale**)
