# run-003 证据审计

本文件记录只读离线诊断，不修改 accepted artifact、Q、Gate 结果或原始 `INFRASTRUCTURE_FAILED`
terminal verdict。

run-003 完整复现了 run-002 的 Gate2 builder 缺口，并保存了 partial artifact、TypeCheck、Gate1、
Lift/Replay response 与 receipt。accepted J 把 `HAVE_BETA` 表达为 rank-0 `i1` input，并把最终
浮点结果关系声明为 `exact`。

按通用实现逐步修复后，对该历史 candidate 的离线诊断为：

1. rank-0 binding 补齐后，RefEval 因 boolean domain guard 不支持而停止；
2. guard/logical-domain mask 支持补齐后，RefEval 因 singleton direct index `[0]` 不支持而停止；
3. 仅在对应 extent 确为 1 时允许 `[0]` 后，Source、Replay、J 三腿均可执行；
4. Source–Replay 是 exact；Source–J 与 Replay–J 的最大绝对差均约为 `2.9802322387695312e-08`；
5. candidate 选择的 comparison class 是 `exact`，所以两条 J 边失败，结论只能是
   `pairwise_diverged`，不能认证。

把该历史 artifact 的 result class 仅为诊断而改成 `tolerance` 时，固定 f32 Q
(`rtol=1e-5`, `atol=1e-6`) 的三条边都会通过；这不是本 run 的合法重判，只说明通用 Lift 提示必须
把 transcendental/reassociation 浮点关系归入封闭的 tolerance class，并继续由外部 Q 决定阈值。
