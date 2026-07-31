# 2026-07-31 J-only Target-Normal / 确定性 Composer 归档

本目录保存已停止的 Task 19 方法。该路线把 Replay 拆成 J-only Target-Normal Agent 与读取 π/MaskC
的确定性 target Composer；原型没有形成完整 Target-Normal IR，Composer仍直接消费隐藏的 `S_J`，
且 loop/carrier/control 等复杂 π profile 未闭合。

当前 Task 20 改为 J-only Replay；π 不进入 Replay，而只实例化冻结的 frontend/profile-specific
Gate1 observer canonicalization。本文仅供历史追溯，不是活动规范。
