# Task 18：J SemanticMaterializer 与 π 约束 Replay Compiler Agent

> 状态：`IMPLEMENTATION_ACTIVE_REAL_GATE1_RUNNING`（2026-07-31）
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
- [x] 默认 Codex Replay 移除 binding/block/structural-renderer 工具与 prompt；
- [x] 退役未使用的 target-AST binding renderer，不保留活动第二实现；
- [x] 保留 packet query、source-blind sandbox、compile/own-TTIR、checkpoint 与 submit；
- [x] 不新增、恢复或生成 `python/tests`。

## 3. P1：Replay SemanticMaterializer Core

- [x] 对照后端分支选择 J-only canonical/typed-boundary 能力；主仓现有 `canonical_j` 已包含更新的
      dimension/broadcast 语义，直接复用并避免迁入 concrete-extents production materializer；
- [x] Core 直接消费 exact canonical J，不建立跨仓库接口依赖；
- [x] 生成参数化、只读、anchor-preserving `S_J`；
- [x] 保留 symbolic extents，不读取 concrete Harness valuation；
- [x] 不做 constexpr specialization、production structural search 或 `Attach_C`；
- [x] 每个 node/domain/access/effect/output binding 可回指 exact J handle；
- [x] structured loop/control 不得因 materialization 删除 π 所引用的 anchor；
- [x] normal 与 MaskJ 分别 fresh materialize，禁止 full-J `S_J` 泄漏到 masked episode。

## 4. P2：完整 Replay Compiler Agent

- [x] 每个 formal Gate1 round 只启动一个 fresh、persistent、source-blind Agent；
- [x] 启动 prompt 只给文件路径、边界和提交条件，不内嵌大段 artifact/手册；
- [x] Agent 读取完整 `S_J` 与 exact/queryable π，自由生成一份完整 target source；
- [x] 不要求 per-node mapping、target binding、block manifest 或固定 source map；
- [x] Agent 可自行创建 helper/笔记、编译、查看 own IR、checkpoint、rollback、局部 patch；
- [x] 资源按 wall-time/formal Gate1 round 管理，不硬限制正常工具调用；
- [x] exact submitted bytes 在 Agent 外重新 ingress/compile 后才进入 Gate1。

## 5. P3：π 直接消费与开放世界前端

- [x] exact π 或严格无损 nested wire 作为只读输入；
- [x] query 只做 region/instance/route/order/sharing/carrier 的无损局部索引；
- [x] query 可将 semantic_ref 定位到 `S_J`，但不生成 target spelling 或 lowering plan；
- [x] 删除 fixed `π relation -> source`、program/loop/carrier renderer；
- [x] π 继续拒绝 operator、formula、predicate、fill、Source/API、C value 与 Kernel recipe；
- [ ] Triton 首先落地；CUDA/TileLang 只增加 `U_f`、编译工具与 Agent 能力，不增加完整 deterministic
      J→frontend compiler。

## 6. P4：形式边界与文档同步

- [x] spec 明确允许 `S_J=M(J)` 的 deterministic、J-only、proof-irrelevant 派生；
- [x] 原“禁止 deterministic composer”窄化为禁止 `(J,π)->完整 target source` 与 π executable plan；
- [x] Replay 正式输入/输出、TypeCheck、Gate1、Gate2 三边和 production `J+C` 保持不变；
- [x] production 不读取 π，Replay Core 不读取 C value/ν/target search；
- [x] Gate1 diff 只反馈 persistent Lift owner，fresh Replay 不看 diff；
- [x] 启动 transport 已移除 packet digest/嵌入；不用 mock tests 或全量无关审查替代真实闭合。

## 7. P5：真实纵切与能力修正

- [x] 首核 FlashAttention forward：同一 accepted `(J,C,π)` 对比旧 file-block frontier；
- [x] 记录 `S_J` 大小、首次完整源码时间、首次可编译时间、compile/tool 次数、Gate1 frontier、wall-time；
- [x] 若失败，严格区分 SemanticMaterializer、J、π、Replay consumption、observer；
- [x] 首轮定位为 Replay 对 π placement 的消费缺口：`S_J` 已令 interface/op/reduction/effect 全闭合，
      dataflow 从 232 降到 162；Replay 将 program-region address base 错放入 loop，触发 8 vs 2
      `tt.addptr` 与 15 vs 8 captured values。新增完整逐关系 `pi.intent`，不增加 target recipe；
- [x] SemanticMaterializer 缺口直接修 Core；J/π 缺口才反馈 Lift owner；
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

## 10. 首个真实纵切记录（2026-07-31）

- exact parent artifact：Task17 Flash-forward `attempt-01`，TypeCheck PASS；
- 首版 `S_J`：17,578 bytes、38 nodes、53 typed values；Agent 一次 provider turn 在 600 秒内写出并
  多次修订约 3 KB 完整 Triton，ingress/compile PASS；
- 旧 file-block Gate1：总分 0.7327，dataflow 232 diffs，predicate/effect 2 diffs；
- Task18 首版 Gate1：总分 0.8142，dataflow 162 diffs，其余 hard categories 全 PASS；
- 根因不是 J/S_J value semantics 缺失，而是 60+ π instances/routes 的 minified 交叉引用没有被完整
  消费。当前新增 `pi.intent`（同 artifact 为 11,486 bytes、158 exact rows），逐行保留原 relation 与
  declaration order，不映射 target statement；第二次 real Gate1 recheck 正在运行。
