# Task 19：J-only Target-Normal IR Agent / π 约束确定性 Composer 与最终闭合

> 状态：`IMPLEMENTATION_ACTIVE`（2026-07-31）  
> 接管已归档 Task 18；Task 11 继续是不停止的 28-Kernel 总闭合主线。

## 0. 唯一目标

把 source-blind Replay 重构为：

```text
J -> S_J -> J-only Target-Normal IR Agent -> F_f
F_f + π + MaskC(C) -> deterministic Composer_f -> target source
```

在不新增认证 artifact、Gate、Source 泄漏、Kernel recipe 或 production π consumer 的前提下，先完成
可运行的 Triton 纵切，再闭合当前 24/28 后的四项，并 fresh Replay 其余已闭合项。

设计依据：

- `doc/method/IntentIR_J-only_Target-NormalIR与pi约束确定性Composer_2026-07-31.md`；
- 四份 2026-07-22 SCRC 权威文档；
- production backend 分支的 J-only canonical/typed-boundary materialization（只复制相关逻辑，不建立
  跨仓接口，也不迁入 C search/target recipe）。

## 1. 起点

- 严格功能基线：24/28 `SCRC_CERTIFIED`；
- active：Llama4 RoPE、SplitK forward、FlashAttention forward、FlashAttention full backward；
- Task 18 Flash-forward `B=3,K=3`：9 个 artifact TypeCheck PASS，全部 Gate1 dataflow/op no-closure；
- 最佳 Task 18 `S_J+π` 直接 Agent frontier 仍有 112 项 dataflow 差异；
- TypeCheck PASS、compile PASS、Gate1-only 均不计新增闭合。

## 2. P0：归档与边界

- [x] Task 18 与其方法文档物理归档；
- [x] 新建 Task 19 与活动方法文档；
- [ ] 更新 spec/四份 SCRC 文档中的 Replay 内部分解；
- [ ] 明确 `F_f` 是 proof-irrelevant episode evidence，不是第四 artifact；
- [ ] 不新增/恢复 `python/tests`，只用真实 closure workflow 验收。

## 3. P1：Target-Normal IR v1

- [ ] 定义 closed protocol、signature、typed value、structured body、origin refs 与 outputs；
- [ ] 每个 semantic node/value/access 必须由 J-origin 覆盖，缺失/重复/越权均 fail closed；
- [ ] 禁止 π/C/Source/Gate/target source/API/physical placement 字段；
- [ ] 支持 map/unary/binary/cast/select/reduce/contraction、structured control 与 loop/carrier 基础；
- [ ] parser/checker 只做 J/IR 自洽检查，不看 π、不补语义；
- [ ] normal/MaskJ 分别 fresh 生成，禁止复用 `F_f`。

## 4. P2：J-only Codex Semantic Agent

- [ ] 新 workspace 只挂载 exact J、`S_J`、`U_f` 和 Target-Normal contract；
- [ ] 不挂载 packet、π、MaskC、Source、Harness、Gate、旧 Replay；
- [ ] 一个 fresh persistent Codex session 编辑 `target-normal.json`；
- [ ] Agent 可调用 Target-Normal parser/checker，但不能调用 Composer 或查询 π；
- [ ] exact `F_f`、trajectory、usage、wall time 与 checker receipt 保存到 Replay evidence；
- [ ] 按 wall time 管理，不以极小工具调用数截断。

## 5. P3：Triton Deterministic Composer

- [ ] Composer 输入严格为 `F_f + canonical π + MaskC(C) + U_f`；
- [ ] generic signature/program-coordinate/address/load-store/control/loop/carrier emitter；
- [ ] π 只作为 constraint，不是 executable plan；
- [ ] generic physical template library 不含 Kernel 名、Source 路径、历史 winner 或 per-kernel算法；
- [ ] complete consumption receipt：F origin、π relation、C slot 与生成源码位置；
- [ ] 不支持/歧义/semantic replacement 一律 fail closed；
- [ ] exact generated source 经现有 ingress/compile 后进入正式 Gate1。

## 6. P4：真实纵切

- [ ] 先用一个现有已闭合、非特化 Kernel 验证 J-only Agent/Composer/Gate1/Gate2；
- [ ] 再用 Flash-forward exact parent 验证 program linearization、loop/carrier 与 dataflow；
- [ ] 同时记录 `F_f` bytes、Agent turns/tokens、Composer consumption、首次 compile、Gate1 frontier、wall time；
- [ ] failure 必须归入 J、Target-Normal、π、Composer、observer 或 Gate2 consumer，不盲目 relift；
- [ ] 真实缺口扩展 generic primitive/constraint，不加 Kernel prompt/模板。

## 7. P5：最终四项与横向复验

- [ ] Llama4 RoPE；
- [ ] SplitK forward；
- [ ] FlashAttention forward；
- [ ] FlashAttention full backward；
- [ ] 新路径稳定后 fresh Replay 原 24 项；正确、参数化 J 不无条件 relift；
- [ ] Gate1 已过、Gate2 失败先区分 J 与 RefEval/Gate2 消费边界；
- [ ] 至少两个 Kernel 横向推进，same-case `B>=3`，首个完整 winner 后协作停止。

## 8. P6：π compact vNext（纵切后）

- [ ] 先量化 full π 中纯机械重复与真实非唯一约束；
- [ ] 只有 closed typed endpoints 唯一决定的关系允许 lossless omission；
- [ ] schema/Decode/TypeCheck/Gate1/Composer 同步版本化；
- [ ] Lift 不读取 Composer/template，π 不携带 template ID；
- [ ] 受 schema 影响的 artifact fresh Lift，不静默迁移旧证书。

## 9. 完成条件

- [ ] 默认 Replay 走 J-only Agent → `F_f` → deterministic Composer；
- [ ] Agent episode 中不存在 π/C/Source/Gate 泄漏；
- [ ] Composer 不含 per-kernel recipe，且删除 J/`F_f` 后不能形成计算结果；
- [ ] active 四项闭合并达到 28/28 current-version `SCRC_CERTIFIED`；
- [ ] 28/28 fresh Replay 与 Gate2 三条直接比较成立；
- [ ] spec、四份权威文档、方法文档、runtime 与 evidence 一致；
- [ ] 不生成 tests、hash/banner gate 或生产 π consumer；
- [ ] 提交后工作区干净。

最终裁决：

> **Replay 的开放世界语义编译由 J-only Agent 承担；物理 realization 由 π 约束的通用确定性
> Composer 承担；两者谁都不得代替另一方发明事实。**
