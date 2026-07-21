# 旧控制面的清除合同

活动实现必须移除以下旧架构：

```text
deterministic composer
RuleCard / RuleRegistry / selected cards
C.strategy / StrategyFamily
compose provenance / registry digest
IntentProgram 作为第二 wire IR
LogicalABI / LogicalInterface 语义对象
program_adapter
在线 semantic intervention / π ablation matrix
```

## 顺序

1. 声明新 artifact schema epoch，旧 cache/artifact 硬失败；
2. 建立 Lift LLM 直接输出最小 π 的通路；
3. Replay packet 改为 `J+π+纯C+全局前端上下文`；
4. 一次性删除 composer、RuleCard、C.strategy 和 selected-card channel；
5. 重写 masked-J，并将其他消融移出在线 core；
6. 删除 IntentProgram/LogicalABI/program adapter，lowering/scalar 直接消费完整 J；
7. 清理 imports、exports、tests、cache schema 和兼容 shim。

不得建立新旧双轨或兼容转换器。旧代码和 spec 只能保存在 archive，不得被活动模块
import。
