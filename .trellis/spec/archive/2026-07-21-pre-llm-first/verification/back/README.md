# 旧验证规范归档

`legacy_v0/` 保存 SCRC 之前的 schema/obligation、FULL/PARTIAL/OOS 与 L0–L9
验证设计。它们保持可读，但不再是项目当前实现依据。

`gate2_production_lowering_v1/` 保存 2026-07-20 前的 source/replay/production-
lowering 三方 Gate2 规范。当前 Gate2 执行 source、final-J replay 与 scalar reference，
要求 `source↔replay`、`source↔scalar` 两份比较；该目录只作历史复核。中间存在过的
`source_replay_dynamic_equivalence_v1` 与 `final_j_replay_scalar_differential_v2` 也已
被 v3 取代，其旧报告不得进入现行统计。
