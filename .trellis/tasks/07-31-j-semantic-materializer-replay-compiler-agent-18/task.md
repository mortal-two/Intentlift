# Task 18：J SemanticMaterializer 与 π 约束 Replay Compiler Agent

> 状态：`DESIGN_READY_IMPLEMENTATION_PENDING`（2026-07-31）
> 接管已归档的 Task 16/17。Task 16 的 grouped composer 与 Task 17 的 hole/binding/structural
> renderer 都没有显著降低 Replay 编译负担；它们只作为历史证据保留。

## 0. 唯一目标

建立确定性的 `J -> read-only parameterized S_J`，让一个 fresh source-blind Replay Compiler Agent
直接消费 `S_J + π + MaskC(C) + U_f` 并生成完整目标源码；停止 fixed target skeleton、binding、hole、
fragment/route assembler 路线，在不改变 SCRC 外部数学接口的前提下完成 28-Kernel closure。

设计依据：

- `doc/method/IntentIR_J语义物化与pi约束ReplayCompilerAgent_2026-07-31.md`；
- 四份 2026-07-22 SCRC 权威文档；
- 后端分支已实现的 canonical-J、typed-boundary、contract/predicate materialization。

## 1. 基线与分母

- 严格功能基线：24/28 `SCRC_CERTIFIED`；
- active：Llama4 RoPE、SplitK forward、FlashAttention forward、FlashAttention full backward；
- TypeCheck、Replay compile、Gate1 接近通过均不计闭合；
- Task 17 Flash-forward file-native pilot：可编译，但 Gate1 dataflow 232 项差异、predicate/effect 2 项；
- Task 11 继续是不停止的 28-Kernel 总闭合主线，Task 18 负责新的 Replay 编译方法。

## 2. P0：归档与退役旧路径

- [x] Task 16/17 迁入 `.trellis/tasks/archive/2026-07/`；
- [x] 两份旧 Composer 方法文档迁入 `doc/archive/2026-07-31-replay-composer-prototypes/`；
- [x] Git 历史与归档 README 记录 grouped、binding、file-block 原型及真实失败证据；
- [ ] 默认 Codex Replay 移除 binding/block/structural-renderer 工具与 prompt；
- [ ] 退役未使用的 target-AST binding renderer，不保留活动第二实现；
- [ ] 保留 packet query、source-blind sandbox、compile/own-TTIR、checkpoint 与 submit；
- [ ] 不新增、恢复或生成 `python/tests`。

## 3. P1：Replay SemanticMaterializer Core

- [ ] 从后端分支选择性迁入 canonical-J、typed-boundary、contract/predicate 逻辑；
- [ ] Core 直接消费 exact canonical J，不建立跨仓库接口依赖；
- [ ] 生成参数化、只读、anchor-preserving `S_J`；
- [ ] 保留 symbolic extents，不读取 concrete Harness valuation；
- [ ] 不做 constexpr specialization、production structural search 或 `Attach_C`；
- [ ] 每个 node/domain/access/effect/output binding 可回指 exact J handle；
- [ ] structured loop/control 不得因 materialization 删除 π 所引用的 anchor；
- [ ] normal 与 MaskJ 分别 fresh materialize，禁止 full-J `S_J` 泄漏到 masked episode。

## 4. P2：完整 Replay Compiler Agent

- [ ] 每个 formal Gate1 round 只启动一个 fresh、persistent、source-blind Agent；
- [ ] 启动 prompt 只给文件路径、边界和提交条件，不内嵌大段 artifact/手册；
- [ ] Agent 读取完整 `S_J` 与 exact/queryable π，自由生成一份完整 target source；
- [ ] 不要求 per-node mapping、target binding、block manifest 或固定 source map；
- [ ] Agent 可自行创建 helper/笔记、编译、查看 own IR、checkpoint、rollback、局部 patch；
- [ ] 资源按 wall-time/formal Gate1 round 管理，不硬限制正常工具调用；
- [ ] exact submitted bytes 在 Agent 外重新 ingress/compile 后才进入 Gate1。

## 5. P3：π 直接消费与开放世界前端

- [ ] exact π 或严格无损 nested wire 作为只读输入；
- [ ] query 只做 region/instance/route/order/sharing/carrier 的无损局部索引；
- [ ] query 可将 semantic_ref 定位到 `S_J`，但不生成 target spelling 或 lowering plan；
- [ ] 删除 fixed `π relation -> source`、program/loop/carrier renderer；
- [ ] π 继续拒绝 operator、formula、predicate、fill、Source/API、C value 与 Kernel recipe；
- [ ] Triton 首先落地；CUDA/TileLang 只增加 `U_f`、编译工具与 Agent 能力，不增加完整 deterministic
      J→frontend compiler。

## 6. P4：形式边界与文档同步

- [ ] spec 明确允许 `S_J=M(J)` 的 deterministic、J-only、proof-irrelevant 派生；
- [ ] 原“禁止 deterministic composer”窄化为禁止 `(J,π)->完整 target source` 与 π executable plan；
- [ ] Replay 正式输入/输出、TypeCheck、Gate1、Gate2 三边和 production `J+C` 保持不变；
- [ ] production 不读取 π，Replay Core 不读取 C value/ν/target search；
- [ ] Gate1 diff 只反馈 persistent Lift owner，fresh Replay 不看 diff；
- [ ] 不用 hash/banner/receipt identity、mock tests 或全量无关审查替代真实闭合。

## 7. P5：真实纵切与能力修正

- [ ] 首核 FlashAttention forward：同一 accepted `(J,C,π)` 对比旧 file-block frontier；
- [ ] 记录 `S_J` 大小、首次完整源码时间、首次可编译时间、compile/tool 次数、Gate1 frontier、wall-time；
- [ ] 若失败，严格区分 SemanticMaterializer、J、π、Replay consumption、observer；
- [ ] SemanticMaterializer 缺口直接修 Core；J/π 缺口才反馈 Lift owner；
- [ ] 至少同时保持两个真实 Kernel 推进，winner-first，不等待无效败者；
- [ ] 需要时从 Terra 升级 Sol high/xhigh/max，但模型升级不替代能力修复。

## 8. P6：剩余四项与原闭合项

- [ ] Llama4 RoPE；
- [ ] SplitK forward；
- [ ] FlashAttention forward；
- [ ] FlashAttention full backward；
- [ ] 新路径稳定后，对原已闭合项 fresh Replay；正确且参数化的 J 不无条件 relift；
- [ ] Gate1 已闭合、Gate2 失败时先判断 J 与 RefEval/Gate2 消费边界；
- [ ] MaskJ 从 masked J fresh 生成 masked semantic view；
- [ ] J-authority mutation：改变闭合 J 后，fresh `S_J` 与 Replay 必须随 J 改变。

## 9. 完成条件

- [ ] 默认 Replay 不再使用 binding/hole/fixed target skeleton/route assembler；
- [ ] `S_J` 参数化、J-only、read-only、anchor-preserving，并通过真实复杂 Kernel 消费；
- [ ] Replay agent 独立生成完整 target source，π 不泄漏语义 recipe；
- [ ] active 四项闭合，随后达到 28/28 current-version `SCRC_CERTIFIED`；
- [ ] 28/28 fresh Replay、Gate1 与 Gate2 三条直接比较成立；
- [ ] spec、四份权威文档、方法文档和实际 runtime 一致；
- [ ] 无 tests、per-kernel recipe、历史 winner、production π consumer；
- [ ] 提交后工作区干净。

最终裁决：

> **SemanticMaterializer 确定性解释 closed J；Replay Compiler Agent 在 π 约束下承担完整开放世界
> target compilation；宿主只提供只读语义输入、查询、编译、隔离与提交，不再施工目标程序。**
