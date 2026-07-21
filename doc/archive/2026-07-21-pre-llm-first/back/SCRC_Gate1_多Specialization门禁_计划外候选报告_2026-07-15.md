# SCRC Gate1 多 Specialization：方案计划与实施边界（归档）

日期：2026-07-15；方案细化：2026-07-16

> 归档状态：fixed-C 第一阶段基础路径已实现，剩余 live/held-out 与 family-wide
> masked-J 接入已统一迁移到 `doc/SCRC_v8后续实施与规模验证流程.md`。本文仅保留
> 当时的决策过程与实验边界。
> `Gate1SpecializationFamily/Gate1FamilyRunner` 已支持共享 artifact 与单份 replay、
> 逐 profile source/replay compile 和严格 all-pass；跨 C family 与真实批量实验仍未
> 完成。全程未使用 Trellis task/workflow。
>
> **2026-07-17 masked-J 口径更新：** 本报告中的 diagnostic/fallback 与 ratio 聚合
> 属于历史候选口径。当前单 profile Gate1 已恢复 `masked J + 实际 Lift π` terminal
> attack；fixed-C family 仍需把同一 masked generation 在每个 profile 上编译并要求
> 逐 profile hard mismatch。完成该接入前，family runner 不能宣称已有 family-wide
> 反泄漏证据。

## 1. 当前决定

当前已新增独立的 fixed-C family 执行入口；原单 profile `Gate1Runner.run` 仍保留。
边界是：

```text
Gate1：单 profile 或显式 fixed-C runtime profile family，同一 C 下比较 TTIR
Gate2：可对已得到的 replay 执行严格 shape × value × seed suite
```

family verdict 采用全 profile hard all-pass；Gate2 matrix 仍是独立动态证据。

## 2. 为什么会出现这个候选需求

此前已实际运行的 11 个 FlagGems Kernel，其 normal replay 都只在一个具体
specialization 上闭合。新 Gate2 suite 已证明同一 replay 可以进一步覆盖：

- `fg_ones`：1、BLOCK−1、BLOCK、BLOCK+1、非整除、远多 tile；
- `fg_min`：固定 `BLOCK_MID=64` 下 3 个 shape × 4 个 value profile × 3 seeds，
  共 36 个三方 case 全部通过。
- `fg_vector_l1`：固定 `BLOCK_SIZE=1024` 下
  `M={1,1023,1024}`、runtime `ord={1.0,1.5,2.0}`、2 个 value profile ×
  3 seeds，共 18 个三方 case 全部通过。
- `fg_embedding`：静态 `indices[17]`、`weight[97,13]` J 下，2 个合法索引/权重
  value profile × 3 seeds，共 6 个三方 case 全部零误差通过；尝试把 K 改为
  `{1,15,16,17}` 时，matrix builder 因静态 J extent 不同而 fail-fast。

这些证据增强行为可信度，但不能反推“所有 specialization 的 source/replay TTIR
都已结构闭合”。constexpr、runtime scalar specialization 和不同 grid 可能改变
TTIR；若论文要把结论从“该 specialization 闭合”提升为“该 Kernel 闭合”，需要
额外定义 Gate1 的 profile 语义。

`fg_embedding` 还给出更具体的边界：其 compiler view 没有 runtime K/V/D scalar，
Lift 因而正确生成静态 tensor extent。Gate2 value matrix 可以验证数据依赖地址与
词表上下界，却不能仅靠替换 host tensor shape 把静态 J 变成动态 J；这类 shape
扩展必须先有不同 compiler profile 或重新定义 artifact/profile 关系。

## 3. 不能直接实现的三个未决条件

### 3.1 profile 是否允许改变 C

- 固定 C、只改变 runtime shape/scalar：仍接近当前同一 C 控制实验；
- 改变 BLOCK/num_warps/num_stages：会进入多 C，source/replay 的结构和 masked-J
  信息量都可能变化。

主方案只规定一次比较中 source/replay 使用同一 C，没有规定一个 Gate1 verdict
是否要跨多个 C 聚合。

### 3.2 normal 闭合如何聚合

最保守的候选规则是：同一 replay source 在所有 profile 上都必须编译且 hard
closure，任一失败则整个 Gate1 不闭合。不能用平均 score 或多数票。当前计划已将
恢复路径收紧为统一 relift，但实施前仍需验证：

- 一个 profile 失败时，逐 profile 差分如何合并为一次 source-aware Lift 输入；
- 不同 profile 的修复要求冲突时如何保留最优候选；
- 已通过 profile 如何成为不可回归约束。第一阶段要求共享一个 J/π 和一份 fresh
  replay，禁止 profile-local replay repair。

### 3.3 masked-J 如何聚合

当前 v8 已明确 masked-J 不是 fallback：normal opaque 与 paired intervention 都不能
替代 `masked J + 同一实际 Lift π` terminal attack。family-wide 聚合应复用一份 fresh
masked generation，在每个 normal-closed profile 上恢复 ABI、编译和提取硬指纹；
每个 profile 都必须出现合法候选 compile rejection 或 compiled hard mismatch。
任一 profile hard closure 为 `LEAKAGE_REJECTED`，任一 generation/ABI/fingerprint
不可评估为 family `NO_CLOSURE`。不使用 score/ratio、平均或多数票。

## 4. 若获批准，建议的最小实现范围

只建议先做“固定 C、多 runtime specialization”，不立即跨 C：

1. `Gate1SpecializationProfile`：runtime argument shape/value、grid、稳定 id；
2. Lift 和 normal replay generation 只执行一次；
3. 同一 replay source 对每个 profile 分别编译 source/replay 并提取 TTIR；
4. 保存逐 profile compile/comparison，不做平均；
5. normal verdict 采用 all-pass；
6. masked-J 聚合政策必须由审批明确后再编码；
7. batch identity 绑定完整 profile matrix；
8. 首个实验只用 `fg_ones` 的固定 `BLOCK_SIZE=1024` runtime-shape profiles。

该方案不需要新增 IntentIR op，也不恢复旧 verifier；它只扩展 Gate1 的实验控制
维度。但在第 3 节条件未获确认前，不应进入代码。

## 5. 本轮已完成但不越界的部分

- production Gate2 `Gate2Suite` 与严格 all-case 聚合；
- shape/value/seed metadata 和 batch identity；
- 首个真实 36-case live matrix；
- IEEE-special 输入触发并修复 Lower2C `maxnum/minnum` 语义；
- fixed-C family core runner 已实现 normal all-pass；paired intervention、π audit 与
  actual-π masked terminal attack 尚未完成逐 profile 接入，因此当前 family 结果只
  是 normal structure evidence，不是完整 family-wide Gate1 evidence。

## 6. 总体目标与非目标

目标不是让一个 Gate1 score 覆盖更多输入，而是把当前结论：

```text
TTIR_src(S0) == TTIR_replay(S0)
```

扩展为一个显式、有限、可复核的 specialization family：

```text
对 family 中每个 Si：
    TTIR_src(Si) == TTIR_replay(Si)
```

最终 normal verdict 采用严格 `all-pass`：任一 profile 生成、编译、接口或硬指纹
失败，当前 artifact family 就没有 Gate1 closure。禁止平均 score、多数票或“主 profile
通过即可”。score 仍只用于失败诊断和候选排序。

第一阶段的非目标：

- 不改变 IntentIR op；
- 不新增 verifier/Gate；
- 不把 Gate2 多输入执行冒充 Gate1 多 specialization；
- 不立即支持多 C；
- 不让每个 profile 独立生成 J/π 或 replay；
- 不把静态 shape J 强行解释成动态 J；
- 不改变 Gate1 与 Gate2 非包含关系。

## 7. 数据模型计划

### 7.1 `Gate1SpecializationProfile`

逻辑字段：

```text
profile_id
runtime arguments:
  - tensor shape/dtype
  - runtime scalar value/dtype
grid
schedule C
target/compiler identity（沿用 CompileResult/缓存现有字段）
```

实现时优先复用现有 `CompileCase`，不复制另一套参数模型。profile 只是带稳定 id 的
受审计 CompileCase 包装。

### 7.2 `Gate1SpecializationFamily`

```text
family_id
primary_profile_id
profiles: 非空、有序、profile_id 唯一
fixed_c: true                 # 第一阶段必须为 true
```

同一 family 必须共享：

- source；
- interface 参数数量、顺序、mode、dtype、rank；
- J/π/location_map；
- schedule slot 声明及其具体 C values；
- frontend/target；
- 一份 normal replay source。

允许变化：

- 由 J 动态 shape symbol 绑定的 tensor extent；
- 对应 runtime shape scalar；
- 非 constexpr runtime scalar；
- 与 profile 对应的 launch grid。

不允许把静态 J extent 改成另一个值。例如 J 声明 `indices[17]` 时，`17→16` 不是
同一 artifact 的 runtime specialization，而需要新的 Lift/artifact 或未来明确的
shape-polymorphic J 方案。

### 7.3 profile 与 artifact 的纯代码合法性检查

每个 profile 必须满足：

1. 参数集合、顺序、dtype、pointer/scalar kind 与 interface 一致；
2. tensor rank 不变；
3. J 中字符串 shape symbol 必须绑定到相应 runtime scalar，且各 tensor 使用一致；
4. J 中整数静态 extent 必须与 profile 完全相同；
5. constexpr 与 compiler slot 值必须等于 family 冻结 C；
6. grid rank 符合 interface grid contract；
7. profile id 和完整 CompileCase 序列进入 cache/report identity。

非法 profile 属于实验/输入合同错误，应在 LLM 调用前 fail-fast，不触发 relift。

## 8. 第一阶段：固定 C、多 runtime specialization

### 8.1 一轮完整流程

```text
source + primary compiler view
  → Lift 一次，得到共享 J/π/C/interface/location_map
  → 对全部 profile 做 artifact/profile 合法性检查
  → source-blind writer 生成一份 fresh replay source

for profile in family.profiles:
  compile source(profile)
  compile same replay source(profile)
  extract source/replay TTIR fingerprints
  exact Gate1 comparison

all profiles hard-pass
  → family normal closure
otherwise
  → 汇总逐 profile 确定性失败报告，统一回流 Lift
```

每次 relift 后必须重新：

- 校验完整新 artifact；
- 校验全部 profile；
- 编译全部 source profile；
- 生成一份全新 replay；
- 编译/比较全部 replay profile。

旧 replay 永远不能作为修复输入。

### 8.2 source 与 replay 的生成次数

- Lift：每个 artifact round 一次完整 source-aware Lift；
- replay writer：每个 artifact round 一次正常 source-blind generation；
- source/replay compile：每个 profile 各一次；
- 不能为失败 profile 单独修 replay，否则同一 family 不再共享一个 J/π/replay。

### 8.3 normal verdict

```text
REPLAY_CLOSED ⇔
  全部 profile source compile 成功
  ∧ 全部 profile replay compile 成功
  ∧ 全部 profile interface 硬相等
  ∧ 全部 profile A/B 硬指纹闭合
  ∧ 最终反作弊条件成立
```

任一 profile 失败，状态仍为现有 `NO_CLOSURE`；报告增加 termination/profile
信息，不新增 `PARTIAL_PASS` 门禁状态。部分通过只作为诊断事实。

## 9. 多 profile relift 与停止语义

### 9.1 聚合给 Lift 的证据

每轮反馈包含：

- family/profile id；
- 每个 profile 的 generation/compile/interface 状态；
- 每个 profile 的硬差异类别与确定性差分；
- 已通过 profile 的确认字段；
- profile 间共同差异和互相冲突的差异；
- 不包含 replay source body。

完整 Lift 仍只能修改 `J/C.strategy/location_map`，interface/C.schedule 冻结；π 在
同一次 Lift 内重新确定性生成。

### 9.2 防止修一处坏一处

如果新 artifact 让原来通过的 profile 退化，该 round 记录为 regression，不能因为
失败 profile 的 score 上升而采用为闭合结果。最终只认 all-pass。

### 9.3 预算与停滞

停止条件沿用用户已明确的主线：

1. 全 profile 闭合：停止并进入最终反作弊；
2. 达到 relift artifact-round 预算：`NO_CLOSURE`；
3. 连续预设轮数的逐 profile 硬差分签名完全不变：stagnated，提前
   `NO_CLOSURE`；
4. 不因停滞或预算耗尽把 Gate1 改判通过；best candidate 只进入诊断和方案允许的
   rejected-sample Gate2 对照。

停滞签名只使用确定性信息：profile id、失败阶段、硬 component、差分 kind/item/
expected/actual；不使用 score 变化率或模糊距离。

## 10. 反作弊在多 specialization 下的计划

### 10.1 paired J/B intervention

只有 normal family 已 all-pass 后才运行。每个注册 intervention：

1. J/B 只修改一个合法坐标；
2. π/C/interface/profile family 全部冻结；
3. mutant replay source 生成一次；
4. 对每个适用 profile 分别编译；
5. 每个适用 profile 都必须出现注册的精确 TTIR delta；
6. 不适用 profile 必须由纯代码 eligibility 明确排除，不能静默跳过；
7. 任何 profile 的 generation/compile/evaluation failure 都 fail-closed 当前
   intervention，不做 replay repair。

### 10.2 π audit

每个 ablation 只生成一份 replay source，再跨全部 profile 编译比较：

- J-only；
- plan step dropout；
- mechanism substitution。

消融仍是 diagnostic，不修改 normal Gate1 verdict、不触发 relift。报告逐 profile
区分 `DEGRADED / UNEXPECTED_CLOSURE / INCONCLUSIVE`，禁止把 generation failure
统计成 π 必要性。

### 10.3 masked-J

masked-J 始终是 terminal verdict 条件，与是否存在 paired coordinate 无关。复用同一
actual-π masked generation，对所有 normal-closed profile 分别编译：全部 profile
出现 compile rejection 或 hard mismatch 才通过；任一 hard closure 为
`LEAKAGE_REJECTED`，任一 generation/ABI/fingerprint 不可评估为 `NO_CLOSURE`。

## 11. profile 选择计划

profile 不能由 LLM 临时发明，而由测试/批处理 manifest 显式声明。第一阶段沿用
SCRC 9.3 的边界集合。

### 11.1 一维 elementwise/fill

固定 `BLOCK=B`：

```text
N ∈ {1, B-1, B, B+1, 2B+3, 远大于 B 的非整除值}
```

覆盖单 lane、尾块、整块、刚跨块、多 tile 和远距离 grid。

### 11.2 row reduction / softmax / norm

固定 `BLOCK_N=B`，至少覆盖：

```text
M ∈ {1, 多行}
N ∈ {1, B-1, B}
runtime scalar：至少两个非等价值（如 eps）
```

`N>B` 若当前源码/π 只实现单 tile reduction，不得伪装成同一支持范围；应保持
unsupported，直到已有计划内 structured/streaming reduction 真正支持。

### 11.3 matmul

固定 `BM/BN/BK`，profile 必须同时覆盖：

- M/N 的 `1、block-1、block、block+1、非方形 tail`；
- K 的单 tile、整 tile、`block+1`、多 K tile；
- 至少一个三个方向都非整除的 profile。

不能用完全笛卡尔积制造不可控成本；manifest 采用显式列举的轴隔离 + 组合压力
profile，每个 profile 都是可审计输入，不引入覆盖 score。

### 11.4 runtime stride/scalar

对 interface/J 已明确支持的 runtime scalar，至少包含：

- 当前 reviewed 值；
- 非中性值；
- 会改变 Triton scalar specialization 的值（例如 stride `1→2`）。

如果修改 scalar 会改变语义合同而 J 没有表达，则不是合法 profile。

## 12. Gate2 对齐

Gate1 family all-pass 后，Gate2 suite 按 profile 分组：

```text
profile
  × value distribution
  × seeds
  → source/replay/Lower2C 三方执行
```

每个 Gate2 case 记录所属 specialization profile。Gate2 仍使用逐 case all-pass；
不能用 Gate2 行为通过覆盖某个 Gate1 profile 的结构失败。

Gate1 rejected family 可按方案 9.1 抽样离线运行 Gate2，继续统计：

```text
P(Gate2 pass | Gate1 closed)
P(Gate2 pass | Gate1 rejected)
```

## 13. 缓存、报告与协议迁移

实施时必须升级：

- SCRC relift protocol；
- held-out batch identity/schema；
- Gate1 report schema。

identity 至少绑定：

- profile 有序清单及每个完整 CompileCase；
- primary profile；
- fixed-C 标志与 C values；
- source/compiler/target；
- model/effort、规则库、Lift/replay prompt；
- intervention/π audit policy；
- relift/stagnation预算。

旧单-specialization报告不得静默升级或 resume 为多-specialization结果。

报告建议新增：

```text
specialization_family
profile_results[]
all_profiles_closed
failed_profile_ids[]
termination_reason: closed | budget_exhausted | stagnated | infrastructure
relift_rounds[].profile_diffs[]
```

Gate1 顶层状态仍复用现有 `REPLAY_CLOSED/NO_CLOSURE/...`。

## 14. 分阶段实施顺序（获批后）

### S0：纯模型与单元测试，不改 live verdict

- 实现 profile/family 数据类和 artifact compatibility；
- 实现逐 profile compile/comparison record；
- 实现 all-pass 聚合、差分签名和停滞判断；
- 用 static writer/fake compiler 验证完整重跑与失败分支。

### S1：固定 C 的 Gate1Runner 接入

- Lift/replay 各一次；
- source/replay 跨 profiles 编译；
- 任一失败统一 relift；
- 现有单 profile API 作为长度为 1 的 family，避免建立第二套 Gate1。

### S2：反作弊跨 profiles

- paired intervention 一次生成、多 profile 精确 delta；
- π audit 一次生成、多 profile diagnostic；
- actual-π masked terminal attack 一次生成、多 profile hard comparison。

### S3：真实小规模证据

按顺序运行：

1. `fg_ones`：固定 `BLOCK_SIZE=1024` 的边界 N；
2. add：固定 BLOCK 的多 N/grid；
3. row-sum：M/N profile；
4. `fg_addr`：stride `1/2`；
5. matmul：多 M/N/K tail 与 K-loop profile。

全部必须 fresh `gpt-5.6-sol/xhigh` artifact 或与当前协议严格匹配的本轮 cache；旧
手写 J/π、旧 MLIR、旧单 profile terminal report 不计入。

### S4：held-out 与 Gate2 matrix 对齐

- ready real Kernel 的 manifest 增加 profile family；
- 正式 batch 默认多 profile；
- 输出 closed/rejected 条件 Gate2 成对统计；
- 继续按真实失败补规则/Lower2C，不预先加 op。

### S5：多 C（另行审批）

多 C 暂不并入上述实现。后续必须单独决定：

- 同一 J/π 是否跨 C 共享；
- C 改变导致机制变化时是否需要独立 artifact；
- per-C Gate1 结果如何组成 Kernel-level evidence bundle；
- π audit 与 masked-J 如何跨 C 聚合。

在这些条件明确前，不把 `BLOCK/warps/stages` 变化混入第一阶段。

## 15. 必须具备的正负测试

正向：

- 同一参数化 replay 在所有固定-C runtime profiles 硬闭合；
- relift 后全部 profile 从 source compile 起完整重跑；
- paired intervention 在所有适用 profile 得到精确 delta；
- Gate2 profile×value×seed 全部通过。

负向：

- replay 把 `N` 烘成 primary 值：primary 通过，其他 profile 必须失败；
- replay 把 K-loop trip count 写死：两 tile 通过，单/多 tile profile 失败；
- replay 只实现整块、不实现 tail：整除 profile 通过，`block±1` 失败；
- grid 写死：shape 改变后 interface/profile 失败；
- stride=1 被专门化正确但 stride=2 地址错误；
- relift 修复一个 profile却让已通过 profile退化；
- profile list/cache identity 改变后不得 resume 旧结果。

## 16. 已明确决定与剩余实施

1. 第一阶段已经采用：

   ```text
   固定 C + 多 runtime specialization + all-pass
   ```

2. masked-J 不再依赖 paired coordinate，也不是 fallback。正式 family 条件为：

   ```text
   所有 normal-closed profile 都必须 compile reject 或形成可比指纹并 hard mismatch
   ```

剩余工作是把该条件接入 `Gate1FamilyRunner` 与 live/held-out harness；不引入新的
Gate、score 或人工资格体系。
