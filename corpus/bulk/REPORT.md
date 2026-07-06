# C1 universe pilot — composability-measurement gate

- extraction success: **95.1%** (39/41)
- composability-failure rate (extract-ok but lower-fail): **5.1%** (2/39)
- verify pass: **95.1%** (39/41)

| op | family | extract | lower | verify | rel | class |
|---|---|---|---|---|---|---|
| tanh | unary | ✅ | ✅ | ✅ | 1.19e-07 | ok |
| sigmoid | unary | ✅ | ✅ | ✅ | 1.22e-07 | ok |
| silu | unary | ✅ | ❌ | ✅ | 1.33e-07 | lowering_fail |
| gelu | unary | ✅ | ✅ | ✅ | 5.94e-08 | ok |
| relu | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| abs | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| neg | unary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| exp | unary | ✅ | ✅ | ✅ | 1.38e-07 | ok |
| sqrt | unary | ✅ | ✅ | ✅ | 9.73e-08 | ok |
| log | unary | ❌ | ❌ | ❌ | — | extraction_fail |
| reciprocal | unary | ✅ | ✅ | ✅ | 5.96e-08 | ok |
| rsqrt | unary | ❌ | ❌ | ❌ | — | extraction_fail |
| atan | unary | ✅ | ❌ | ✅ | 8.93e-08 | lowering_fail |
| tan | unary | ✅ | ✅ | ✅ | 7.81e-08 | ok |
| exp2 | unary | ✅ | ✅ | ✅ | 6.65e-08 | ok |
| celu | unary | ✅ | ✅ | ✅ | 4.87e-08 | ok |
| log_sigmoid | unary | ✅ | ✅ | ✅ | 6.43e-08 | ok |
| add | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| sub | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| mul | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| div | binary | ✅ | ✅ | ✅ | 7.28e-08 | ok |
| maximum | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| minimum | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| pow | binary | ✅ | ✅ | ✅ | 1.44e-07 | ok |
| ge | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| gt | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| lt | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| le | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| ne | binary | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| sum | reduce | ✅ | ✅ | ✅ | 2.04e-07 | ok |
| mean | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| prod | reduce | ✅ | ✅ | ✅ | 2.80e-33 | ok |
| max | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| min | reduce | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| mm | blas | ✅ | ✅ | ✅ | 1.52e-07 | ok |
| bmm | blas | ✅ | ✅ | ✅ | 2.12e-07 | ok |
| addmm | blas | ✅ | ✅ | ✅ | 2.45e-07 | ok |
| mv | blas | ✅ | ✅ | ✅ | 1.21e-07 | ok |
| outer | blas | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| zeros | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |
| ones | init | ✅ | ✅ | ✅ | 0.00e+00 | ok |

Gate: composability-fail 5.1% (< 20% ? **C2 CAN scale**)
