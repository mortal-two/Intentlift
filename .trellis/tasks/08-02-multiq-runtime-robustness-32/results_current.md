# Task32 pre-registered multi-Q runtime robustness

- denominator: 31 same-origin pairs / 62 exact accepted artifacts
- Q schedule: one accepted Q plus four SHA256-derived Q seeds per artifact
- consumer compiles each exact J once before opening any of its five Q files
- Source / Replay / pi / frontend / case / seed are unavailable to multiq_consume.py

## Runtime closure

| Evidence | Result |
|---|---:|
| Compile-once receipts | 62/62 |
| Cubin sequence identical to Task31 | 62/62 |
| Accepted-Q agreement/graph/output receipt identical to Task31 | 62/62 |
| Artifacts passing all five Q | 62/62 |
| Pairs passing all ten frontend-Q evaluations | 31/31 |
| Artifact-Q evaluations | 310/310 |
| Output comparisons | 410/410 |
| Artifacts with five distinct logical-Q digests | 62/62 |

## By Q provenance

| Q set | Evaluations | Q pass | Outputs pass | Tolerance-output max abs error median / p95 / max |
|---|---:|---:|---:|---:|
| accepted | 62 | 62 | 82/82 | 2.60770321e-08 / 5.60283661e-07 / 1.90734863e-06 |
| robustness | 248 | 248 | 328/328 | 2.98023224e-08 / 7.74860382e-07 / 1.90734863e-06 |

## By frontend provenance

| Frontend | Artifacts | Q evaluations | Q pass | Output pass | Max observed abs error |
|---|---:|---:|---:|---:|---:|
| tilelang | 31 | 155 | 155 | 205/205 | 1.90734863e-06 |
| cuda | 31 | 155 | 155 | 205/205 | 1.90734863e-06 |

## By numeric policy

| Policy | Q evaluations | Outputs | Exact / tolerance | Pass | Max observed tolerance-output abs error |
|---|---:|---:|---:|---:|---:|
| `intentir.gate2.default.v1` | 260 | 360 | 40 / 320 | 360/360 | 1.90734863e-06 |
| `intentir.gate2.tf32-tensorcore.v1` | 50 | 50 | 0 / 50 | 50/50 | 4.76837158e-07 |

## J-correspondence strata

| Stratum | Pairs | Pairs all ten pass | Artifact-Q pass | Output pass |
|---|---:|---:|---:|---:|
| Semantic core exact | 26 | 26/26 | 260/260 | 360/360 |
| Source-grounded non-exact | 5 | 5/5 | 50/50 | 50/50 |

## Failure taxonomy

- none

This is fixed-extent value robustness, not shape robustness, Source/Replay execution, RVV runtime, or a performance experiment.
