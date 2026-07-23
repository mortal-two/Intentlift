# 旧实现退役与首步重构

> 状态：当前迁移规范。目标是让代码开始实现三份核心设计及其数学/claim 总述，不维持新旧双轨。

## 必须退役的协议设施

- artifact hash/digest/commitment/epoch 认证门与所谓冻结状态机；
- 独立 interface artifact、`Ids(interface)`、LogicalABI/LogicalInterface 与语义 BindingMap；
- deterministic composer、RuleCard/RuleRegistry、selected card、`C.strategy`；
- Replay patch、diff-guided Replay、跨候选复用 Replay；
- 仅自闭合的 `Parse(π)` 冒充 `TypeCheck(J,C,π)`；
- `masked hard-close -> LEAKAGE_REJECTED` 的旧主状态机；
- IntentProgram 或 program adapter 作为第二份 semantic truth；
- 在线 mutation/swap/dropout/family sweep 进入主认证。

通用文件校验、缓存 key 或 Git 对象使用哈希不在此退役范围；只有把 hash/digest 当作实验合法性、
信息隔离或语义认证依据的旧协议设施必须删除。

## 2026-07-22 已完成的清理与门禁核心重构

- formal batch v1–v7 的 runner、preflight、Lift wire lock、manifest、测试、Task10 与 artifacts 已
  整体迁入 `.trellis/archive/experiments/2026-07-22-formal-hash-protocol/`；活动代码不再 import。
- Gate2 已删除 replay/scalar artifact digest 对齐门，改为直接要求 RefEval request 与 comparator
  持有同一个 J 对象。
- Triton 与 CuTe Gate1 不再把 masked-J hard close 改写为 `LEAKAGE_REJECTED`；主 Gate1 closure
  与 `j_necessity_status` 已分开报告。
- logical signature 已收入 J；proof artifact wire 只保留 `(J,C,π)`，独立 interface namespace 与
  兼容 fallback 已从活动 schema/API 删除。
- dependent `TypeCheck(J,C,π)` 已成为 Replay 的生产前置条件，并 fail closed 地检查 refs、route、
  observable ownership 与 sensitive authorization。
- Replay packet 已收敛为 typed `J + π + MaskC(C) + U_f`；Source/Harness/C values/diff/旧对话等
  嵌套旁路由闭合 allowlist 拒绝。
- Gate2 已使用外部 Q 的对称 `Close_Q`，并保存 Source–Replay、Source–J、Replay–J 三条直接边；
  J wire 不再携带 tolerance 数值。
- 唯一 `SCRCGateRunner` 已统一完整 relift、fresh Replay factory、三轮 Gate1、terminal Gate2、同一
  `c0`、独立 masked-J 状态和真实调用预算；局部 frontend runner 不再拥有第二套候选生命周期。
- 2026-07-23 已加入唯一 `J_wire -> ElaborateJ -> CanonicalJ` 边界：structured ContractExpr/index、
  唯一 result type 与 identity output binding 在 Replay 前确定性归一；RefEval/production 共用
  `CanonicalJ -> Intent MLIR`，不再要求 Lift 猜 Gate2 consumer 的字符串拼写。

## 已执行的首步重构顺序

1. 以三份核心文档、数学/claim 总述和本 spec 替换旧运行口径，历史材料只留在 archive/source-notes。
2. 删除 formal runner 中的旧 hash/epoch/commitment gate 与对应 schema/tests。
3. 将 logical signature 收入 J，删除独立 interface 输入；需要时只计算 `InterfaceView(J)`。
4. 建立最小 `TypeCheck(J,C,π)`：refs、slot/type、observable J anchor、泄漏禁令、authorization。
5. 将 Replay packet 收紧为 `J + π + MaskC(C) + U_f`。
6. 重写 runner：每轮完整 relift、全新盲 Replay、最多三轮；Gate2 失败结束实验。
7. Gate2 接入外部 Q，并直接比较 Source–Replay、Source–J 与 Replay–J 三条 `Close_Q` 边；当前
   只覆盖 deterministic / contract-deterministic Kernel。
8. 拆分 `main_status`、`typecheck_status` 与 `j_necessity_status`，迁移 masked-J 行为。
9. 记录 observer profile/version、compiler/dump stage、canonicalization version 与真实搜索预算。
10. deterministic 本地 smoke 已验证门禁边界；真实 source-aware Lift/Replay canary 属于下一任务。

## 迁移纪律

- 不建立新旧双轨、兼容转换器或静默 fallback；
- 不删除与旧协议无关的通用基础设施；
- 历史 receipt/artifact 保持历史解释，不静默重判；
- 每个退役提交必须同时清理 imports、exports、tests、CLI 参数与文档引用；
- 遇到工作区已有实验结果或用户修改，先审计并保存，不用 destructive reset 清理。

## 首步完成条件

```text
proof artifact = (J,C,pi)
Replay inputs  = J + pi + MaskC(C) + U_f
pre-Replay     = TypeCheck(J,C,pi)
main verdict   = TypeCheck and Gate1 and Gate2
Gate2          = pairwise Close_Q(Source, Replay, RefEval(J)) under observed c0
```

上述首步完成条件已由活动代码、负例和本地 end-to-end smoke 覆盖。下一步只运行少量真实
source-aware Lift/Replay canary；若暴露 schema/observer 能力缺口，再按系统改进流程修改通用能力。
