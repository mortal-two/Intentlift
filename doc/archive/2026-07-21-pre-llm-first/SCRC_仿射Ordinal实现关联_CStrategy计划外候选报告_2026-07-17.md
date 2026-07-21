# SCRC 仿射 Ordinal 实现关联：C.strategy 计划外候选报告

> 状态：**等待用户审查，未实现。**
> 日期：2026-07-17
> 范围：`fg_arange` 暴露的 global-ordinal 与 program-base/lane 算术关联差异。
> 本报告不新增 Gate、IntentIR op、RuleCard 或 strategy family；用户批准前不得实施。

## 1. 为什么需要报告

当前 SCRC 计划规定：J 保存 schedule-invariant 算法语义，C 保存实现策略，π 根据
J+C+RuleRegistry 确定性组装。新增 C.strategy family 会扩大正式实现词汇表，因此
必须先报告必要性，而不能为了让一个 Kernel 通过 Gate1 直接加字段或专用规则。

本轮 schema-24 fresh `gpt-5.6-sol/xhigh` 运行已经排除“J 写错”和“replay code
需要修补”两个解释，但 Gate1 仍稳定不闭合，说明现有 C.strategy/RuleCard 可能缺少
一种可枚举的算术关联机制。

完整报告位于：

```text
/tmp/scrc_v8_schema24_arange_layer_contract/fg_arange.json
```

## 2. 已取得的真实证据

### 2.1 fresh 调用身份

- Lift response：`resp_0a2620bb19f67a1a016a5a04f55f80819bbdf2affe3e7ec1dc`；
- Relift responses：
  - `resp_0698d1a292d7ce92016a5a05a5356c81998b4f5dd10efc80da`；
  - `resp_029b7e3c58b89d65016a5a0687a5548198b61e07c713a354df`；
- 三轮 normal replay 均为独立 response、独立 generation id 和独立 code hash；
- writer 没有 prior replay 输入，Relifter 只收到 source-free `ReliftEvidence`。

第二次 Relift 的第一次 LiftCore 响应出现一次 schema 错误：普通 iota 节点误带
`carry`。它在 Lift 内重新生成完整 artifact 后通过；这属于 LiftArtifact schema
correction，不是 replay repair。

### 2.2 J 在三轮中保持正确且完全不变

三轮 J body 均为：

```text
iota(n="size", i32)
  -> cast(f32)
  -> mul(step)
  -> add(start)
  -> y
```

逐字段比较结果：

```text
initial J == relift-1 J == relift-2 J
initial C.strategy == relift-1 C.strategy == relift-2 C.strategy
initial C.schedule == relift-1 C.schedule == relift-2 C.schedule
initial π == relift-1 π == relift-2 π
interface 全部相同
```

旧运行曾把 `BLOCK_SIZE=128`、integer div、block-start、lane 重建写进 J；新的
Lift/Relift 分层合同已经阻止了该污染。当前 J 中不存在 `BLOCK/pid/lane` 算法节点，
runtime extent 也没有再被烘焙成样本值 `1003`。

### 2.3 Gate1 差异三轮稳定不变

三份 replay 均可编译，但 source/replay 硬指纹每轮保持同一类差异：

- source 比 replay 多 1 个 `arith.addf`；
- source 比 replay 多 1 个 `arith.mulf`；
- source 比 replay 多 1 个 `arith.sitofp`；
- replay 多 1 个 `arith.trunci`；
- source 比 replay 多 1 个 `tt.splat`；
- dataflow 差异对应同一套 affine-value association。

Relift 没有通过污染 J 人为缩小差异，预算耗尽后正确返回：

```text
Gate1 = NO_CLOSURE
```

### 2.4 Gate2 证明三方语义一致

Gate2 diagnostic 使用 3 个合法 profile：

- forward fractional；
- descending fractional；
- forward unit；

每个 profile 使用 3 个 seed，共 9 case。结果：

```text
9/9 GATE2_PASS
source <-> replay      9/9 PASS
source <-> Lower2C     9/9 PASS
replay <-> Lower2C     9/9 PASS
全部 max_abs_error = 0
```

因此当前观测属于合理的非包容四象限：

```text
Gate1 NO_CLOSURE + Gate2 PASS
```

Gate2 不能覆盖 Gate1，但它证明不能通过修改 J 来解释剩余结构差异。

## 3. 缺失的实现信息是什么

源实现把全局 ordinal 的 affine value 分开计算：

```text
lane_term = cast(lane) * step
base_term = cast(program_id * BLOCK_SIZE) * step
y = lane_term + base_term + start
```

当前 canonical replay 先合成全局 ordinal：

```text
global = program_id * BLOCK_SIZE + lane
y = cast(global) * step + start
```

两者算法相同，但 TTIR op multiset/dataflow 不同。这不是：

- J schema 缺少算法语义；
- π 缺少自由文本；
- replay code 应被 repair；
- 可以用固定 normalization 安全消掉的简单 alpha/ordering 差异。

它是“同一全局坐标的 affine 消费应合并还是按 program-base/lane 分配”的实现选择。
当前已有 family 只覆盖 program coordinate width、pointer association、mask staging 和
reduction call shape，没有表达 value arithmetic association。

## 4. 候选设计（仅供审查）

若批准，建议评估如下最小 family，而不是修改 J 或让 `realize_iota` 对所有 Kernel
固定采用源端分解：

```text
family: affine_ordinal_association
canonical absence: combined_global
candidate override: program_lane_distributed
target: 使用 cast(iota) 的 affine map_binary 节点
```

候选 override RuleCard 的职责仅是：

1. 从 J 确认模式为 `cast(iota(n)) * runtime_scalar`；
2. 从 B/C 已声明的同一 global coordinate 取得 program-base 与 lane component；
3. 分别 cast/mul 后相加，得到与 canonical J 节点相同的结果；
4. 不读取源码、不保存 BLOCK 数值、不改变 J、不影响不匹配该模式的节点。

仍需设计审查的关键问题：

- target 应绑定乘法节点还是一个确定性识别的 affine 子图；
- composer 如何保证 override 精确替换 canonical elementwise step，而不是重复施工；
- refs 如何只引用既有 opaque J/B handle；
- 多个 iota/多维 ordinal 时如何避免歧义；
- integer/float dtype promotion 如何保持 source frontend 的固定、可枚举结构。

## 5. 已排除的替代方案

### 5.1 修改 J 为 block/lane 算术

已由真实失败证明错误：它把 C.schedule 复制到 J，并让 Lower2C 遇到本不需要的
integer-div/float-div 问题。

### 5.2 让 `realize_iota` 永远分开计算

会把一个 source-specific association 变成所有 iota Kernel 的默认机制；对于源码先
合成 global ordinal 的 Kernel，会制造反方向 Gate1 差异。

### 5.3 在 TTIR comparator 中加入代数分配律 normalization

分配律、重结合与 cast 分配会快速扩展为高风险代数等价系统，无法保证只消除固定、
可枚举的 frontend 表达差异，也可能掩盖浮点舍入或 dtype promotion 的真实差异。

### 5.4 放宽 Gate1 或用 Gate2 PASS 覆盖

违反两道 Gate 独立裁决的既定设计。当前结果应保留为四象限证据。

## 6. 批准前还需要的证据

建议至少满足一项再实施：

1. 找到第二个真实 Kernel，具有相同 J 语义但 source 使用 distributed
   program-base/lane affine association；或
2. 用户明确批准把 `fg_arange` 作为首个 growth case 引入该 family。

批准后的最低验收集合应包括：

- canonical combined-global source 与 distributed source 各至少一个；
- 两者 J 完全相同，只允许 C.strategy/π 不同；
- direct π tamper 仍在 writer 前拒绝；
- random `size`、`size < BLOCK`、非整除尾块、正/负/小数 step；
- Gate1 hard diff、Gate2 三方矩阵、masked-J terminal attack；
- held-out 中不匹配该模式的 iota Kernel 不得发生行为变化。

## 7. 当前建议

在用户审查前保持现状：

```text
保留正确 J
保留 Gate1 NO_CLOSURE
保留 Gate2 9/9 PASS
不新增 family/RuleCard
不修改 comparator verdict
```

该结果比通过污染 J 强行闭合更符合 SCRC 主线。
