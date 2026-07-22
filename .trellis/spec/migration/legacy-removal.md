# 旧实现退役与首步重构

> 状态：当前迁移规范。目标是让代码开始实现三份核心设计，不维持新旧双轨。

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

## 首步重构顺序

1. 以三份核心文档和本 spec 替换旧运行口径，历史材料只留在 archive/source-notes。
2. 删除 formal runner 中的旧 hash/epoch/commitment gate 与对应 schema/tests。
3. 将 logical signature 收入 J，删除独立 interface 输入；需要时只计算 `InterfaceView(J)`。
4. 建立最小 `TypeCheck(J,C,π)`：refs、slot/type、observable J anchor、泄漏禁令、authorization。
5. 将 Replay packet 收紧为 `J + π + MaskC(C) + U_f`。
6. 重写 runner：每轮完整 relift、全新盲 Replay、最多三轮；Gate2 失败结束实验。
7. 拆分 `main_status`、`typecheck_status` 与 `j_necessity_status`，迁移 masked-J 行为。
8. 用少量代表性 canary 验证边界，再决定下一步 schema/observer 能力改进。

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
```

活动代码中不再存在旧协议入口；尚未实现的新能力明确 fail closed，随后才运行代表性 canary。
