# Address Realization 09：区分 Program Base 与 Lane Address

> 状态：进行中
> 优先级：P0
> 流程：普通 Markdown 工作项；只借用 Trellis 文件布局，不运行 Trellis workflow

## 为什么这是独立任务

Protocol-v5 已证明 π graph 的 ID/anchor/edge 合同可以由 Lift LLM 正确生成。合法 π 包含 runtime
stride、program instance、三路 access fanout、sharing 与 realization→transfer chain；Replay 也
使用了这些信息，但仍生成：

```text
program * stride + lane -> vector pointer add
```

source 的 strict topology 是：

```text
program/stride scalar materialization
  -> program-uniform scalar pointer bases
  -> lane-varying pointer addresses
  -> load/store transfer
```

现有所有中间 stage 都只能标成泛化 `realize`。Replay 无法知道哪个 instance 必须在 pointer
base/broadcast 之前，继续要求它从 route 图形猜测会把未定义的图案变成隐式规则卡。

Task09 只增加这一个已经由合法 Lift+Replay+Gate1 反例证明的最小类型信息。

## 最终目标

在不改变 π JSON shape 的前提下，为 `instances[*].mechanism` 增加两个 source-neutral tag：

```text
address_base = program-uniform pointer/base realization stage
address_lane = lane-varying final address realization stage
```

Lift LLM 直接依据匿名源码标注，Replay LLM 直接消费；固定代码只验证 enum 和 graph closure。
目标是让 source-blind Replay 能保留 scalar pointer-base → lane address → transfer 的严格拓扑，
同时 π 仍不能定义 logical index、predicate、value formula 或 operator。

## 架构不变量

- Intent 仍只有 `J(A+B) + pure C`；π 在 Intent 外，harness 只保存真实调用绑定。
- `address_base/address_lane` 只分类已有 compiler-observable realization event，不创建语义值；
  logical access/index/domain/predicate/fill/effect 仍全部由 J 决定。
- tag 不携带 offset formula、constant、integer width、cast operator、dtype、shape、vector length、
  frontend API、source spelling、kernel family 或自然语言策略。
- pointer-width conversion 由公共 frontend 实现 program-uniform address base 时自然产生，π 不记录
  `i32→i64` 或目标位宽。
- 不引入 composer、RuleCard/Registry、IntentProgram、LogicalABI、per-kernel prompt、source-to-π
  extractor、response repair 或自动 retry。
- Replay 看不到 source/location map/privileged binding/TTIR/fingerprint/diff/历史 candidate。
- Gate1 继续 strict normalized compiler topology，不合并 base/lane 或弱化 cast/addptr/dataflow。
- normal 不闭合立即停止；只有 normal hard-close 后才运行一次 masked-J。

## 阶段与提交节奏

### 0. 冻结 v5 反例并定义两个 tag（0 次 LLM）

- 固定 v5 source/replay TTIR、receipt/artifact/pi/replay hashes 与 4 个 op-count delta；
- 定义 `address_base`、`address_lane` 的必要/禁止信息以及与 `transfer_direct/async` 的正交关系；
- 用 source-neutral fixture 表达 `base -> lane -> transfer` 与 flat address，两者不能被同一 π 表示；
- 审计 copy/gather/fusion/atomic/async：address tags 只描述寻址 stage，不取代 transfer/update/sync。

完成后提交。

### 1. 最小 mechanism contract（0 次 LLM）

- enum 真相只维护一份，由 parser 与 Responses schema 投影；mechanism 的公共语义由 Lift/Replay
  共享，不建立 rule registry；
- Lift prompt 要求只在源码确有独立 compiler-visible stage 时使用 address tag，不为了凑 Gate1
  虚构 instance；
- Replay 必须先实现 `address_base`，再用其结果实现 `address_lane`，然后 transfer；不得把两者
  代数合并；具体前端写法仍由 LLM 选择；
- validator 拒绝未知 tag，不推导 cast、formula、width 或 API。

完成后提交。

### 2. Conformance、泄漏边界与全量回归（0 次 LLM）

- typed round-trip、schema/parser enum 同构、base/lane/flat 区分、route/order/share survival；
- 删除/交换 base/lane tag 必须改变 strict topology fixture；
- mask J 后，address π 仍不能恢复 value DAG、predicate、logical index map 或 output relation；
- direct/async transfer、atomic/privatized update 与 address tags 可正交组合，不添加 op-family 分支；
- v1--v5 evidence SHA 不变，全量回归通过。

完成后提交。

### 3. Protocol-v6 单一 Canary

- v6 manifest 的 28 cases/source invocation 与 v5 完全一致，只改变 Lift wire epoch；
- v6 recovery 精确绑定 v5 receipt SHA、manifest/preflight、一次 Lift+一次 normal Replay、strict
  `no_closure`、0 masked-J 与 v5→v4 lineage；
- 0-LLM preflight 必须 28/28 ready 后，fresh Swiglu 最多 1 Lift + 1 normal Replay；normal
  hard-close 后最多 1 masked-J；失败立即停止；
- 若 normal 闭合且 masked-J 不闭合，Task09 成功；否则冻结新边界，不在 v6 重试。

完成后提交、归档 Task09，并创建下一任务。

## 验收标准

1. 只新增两个全局 source-neutral mechanism tags，不增加字段、formula 或 kernel rule；
2. schema/parser/prompt/Replay 对 base/lane 唯一语义一致；
3. source-neutral fixtures 严格区分 base→lane→transfer 与 flat vector address；
4. address tags 与 copy/async/atomic/fusion 正交，J 仍唯一拥有 semantics；
5. full regression 通过，历史 evidence 不修改；
6. v6 28/28 preflight 为 0 LLM，唯一 canary provenance/预算/lineage 完整；
7. 结论由 strict Gate1 与条件性 masked-J 实际结果决定，不能预写成功。

## 当前第一步

阶段 0 已完成，详见 `research/stage0-address-mechanism-contract.md`。合同已明确：base/lane 只分类
已有 compiler-visible address stages，与 direct/async transfer、atomic/private update 正交；不携带
formula、width/cast、dtype/shape、mask、API 或 logical index。flat source 没有独立 base 时不得
虚构 instance。

下一步进入阶段 1：集中 mechanism enum/grammar 真相，投影 parser、Responses schema 与
Lift/Replay prompt，并用 source-neutral fixtures 验证 base→lane→transfer 与 flat graph；继续保持
0 次 LLM。
