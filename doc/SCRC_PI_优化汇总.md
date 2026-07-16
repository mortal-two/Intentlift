# SCRC π 优化汇总：严格重放计划

> 更新时间：2026-07-16  
> 当前协议：`relift_only_pi_audit_on_strict_opaque_refs_causal_v7`  
> 设计主线：`doc/IntentIR_SCRC_方案_重写版.md`  
> 范围：汇总 TianchenIR 中与 π 定义、生成、规则卡、重放隔离、因果验证和历史试验有关的全部已落地优化。

## 1. 结论

用户对上一版 π 定义的质疑是成立的。

上一版 `pi-v2/causal-v3` 把 π 改成了“稀疏、无序、只记录 source-specific override 的集合”。这不是单纯压缩表示，而是把 π 从原方案中的“严格施工说明”改成了“少数例外提示”。当前项目又不存在一个纯代码、确定性、版本化、可审计的 full-plan materializer，因此缺失步骤只能由规则库、prompt 和 LLM 隐式补齐。此时 π 本身已经不再是严格重放规则。

所以本轮没有为 sparse π 作事后合理化，而是将 π 恢复为：

> **由 Lift 生成的、完整的、按施工依赖排序的严格重放规则实例序列。**

“完整”指规则级施工计划完整，不是让 π 复制 J、Triton 源码或前端 API。每个步骤只选择一个规则卡，并通过 typed reference 从 J/B/C/接口读取其参数。

## 2. J、π、C 与规则卡的职责

| 通道 | 负责内容 | 明确禁止 |
|---|---|---|
| J/A | 算子、常量、operands、DAG、dtype/shape、归约与 contraction 语义 | 前端 API、来源相关施工选择 |
| J/B | domain、access、index、predicate、fill、effect、output mapping、局部归约合同 | C 值、前端 API |
| C | constexpr/compiler slot 及当前 source specialization 使用的值 | A/B 语义 |
| π | 完整列出规则实例、target、上下文引用和施工顺序 | 复制 J/B/C 的值、源码文本、Triton API |
| 规则卡 | 解释某个已选规则实例如何映射到前端机制 | 替 π 选择或补齐隐藏步骤 |

严格 π 的核心分工是：

```text
J/B/C/interface：保存“事实和值”
π：保存“执行哪些施工机制、作用于谁、先后顺序是什么”
rule card：保存“该机制在前端如何实现”
```

这里必须区分“规则定义”和“π 实例”：21 张 RuleCard 是跨 Kernel 共享、由系统维护
的规则定义，Lift 无权按每份源码创建新规则；Lift 只生成当前 artifact 的 π 实例，
即从全局规则库选择 rule id，并绑定到当前 J/B/interface/C 的 opaque references。

因此同一组 π mechanism/顺序可以绑定到不同 J，writer 会从各自 J 读取 operator、
operands、axis、predicate 和 fill，生成不同重放代码。反过来，如果两个源码具有
相同 J 但使用不同且会影响 TTIR 指纹的 realization（例如 pointer-staged 与
integer-offset、explicit-axis 与 flattened reduction），差异就必须体现在 π；若只
是 tile/warp 等调度差异，则体现在 C。对于 J、π、C、interface 全部相同的两个
输入，source-blind writer 没有信息依据按源码分别生成来源特定结构，也不应靠随机
性制造差别。

因此，完整 π 仍然不是第二份 J。例如：

```json
{"rule": "realize_elementwise", "node_ref": "n0"}
```

只说明对 `n0` 使用 elementwise 施工规则；`add/sub/mul`、operands 和 dtype 必须从 J 的 `n0` 读取，不能抄进 π。

## 3. 当前 π 的数据模型

持久化字段仍保持兼容名称 `plan.steps`。每一步只有：

- `rule`：注册规则卡 id；
- `node_ref`：首选 target；
- `refs`：没有 `node_ref` 时 `refs[0]` 是 target，其余项是有序上下文绑定。

示意性的完整 add 计划如下：

```json
{
  "steps": [
    {"rule": "bind_program_domain", "refs": ["d0"]},
    {"rule": "bind_runtime_scalar", "refs": ["p3"]},
    {"rule": "declare_schedule_slot", "refs": ["p4"]},
    {"rule": "declare_schedule_slot", "refs": ["c0"]},
    {"rule": "materialize_index_tile", "refs": ["a0"]},
    {"rule": "realize_masked_access", "refs": ["a0"]},
    {"rule": "materialize_index_tile", "refs": ["a1"]},
    {"rule": "realize_masked_access", "refs": ["a1"]},
    {"rule": "realize_elementwise", "node_ref": "n0"},
    {"rule": "materialize_index_tile", "refs": ["a2"]},
    {"rule": "realize_masked_access", "refs": ["a2"]},
    {"rule": "realize_output_mapping", "node_ref": "n0"}
  ]
}
```

这里没有源表达式、常量值、`tl.*` API 或 `a_load/reduce_max_0` 一类语义化名字。
`n/t/d/a/v/p/c + declaration index` 只是 source-neutral handle；例如 `p3` 通过
`reference_bindings` 指向 `interface.parameters[3]`，writer 再读取该对象中受信任的
ABI 名称。π 自身不保存这个名称。

## 4. 已落地的 π 优化

### 4.1 恢复完整性

非平凡 Kernel 必须显式列出所需规则实例。writer 不再获得一个可自行推导完整施工过程的隐藏 derived baseline。

当前完整性检查覆盖：

- 已注册 `node_id` 对应的 IntentIR op；
- 每个 boundary access 的地址构造和 masked/unmasked 访问；
- boundary broadcast；
- output mapping；
- runtime scalar；
- 每个 schedule slot；
- 与 grid rank 对应的 program domain binding。

已知 op 的规则覆盖关系为：

| IntentIR op | 必须选择的规则 |
|---|---|
| `map_unary` / `map_binary` | `realize_elementwise` |
| `const` | `realize_const` |
| `reduce` | `realize_reduce` 或 `realize_flattened_reduce` |
| `contraction` | `realize_k_tiled_dot` 或 `realize_outer_product` |
| `cast` | `realize_cast` |
| `select` | `realize_select` |
| `loop` | `realize_loop_carry` |
| `gather` | `realize_indexed_access` |

未知 op 当前 fail-closed，避免缺规则时让 writer 自行猜测。

### 4.2 恢复有序性

`ReplayPlan` 保留 Lift 产生的原始顺序：

- 不再按字典序规范排序；
- 不再把 π 解释成无序集合；
- writer 被明确要求按顺序执行；
- repair/relift 可以定位“π 第 N 步”；
- drop-one-step 消融重新具有确定含义。

顺序只能表达施工依赖，不能通过自由文本或复制语义值旁路编码 J。

### 4.3 Closed schema

π step 只允许 `rule/node_ref/refs`。`semantic_hint`、源码片段、API 片段、常量和任意私有字段都会被拒绝，避免自由文本逐渐把 π 变成第二份 IR。

### 4.4 Typed target 与命名空间

π 的 target/ref 只能引用 artifact 中已有标识：

- J node id 或登记路径；
- J tensor；
- boundary domain；
- boundary access；
- interface parameter；
- schedule slot。

未知引用在进入 writer 前失败。`node_ref` 已作为 target 时，不允许在 `refs` 中重复。

### 4.5 规则适用性校验

Gate1 会检查规则卡是否适用于 target 的实际类别。例如：

- `realize_const` 不能作用于 `reduce` node；
- address materialization 必须作用于 boundary access；
- runtime scalar binding 必须作用于接口参数或对应 J scalar input；
- output mapping 规则必须作用于已登记 mapping target。

这属于严格 π 的合同检查，不是恢复旧 JSON interpreter，也不是另建一套通用 verifier。

### 4.6 重复、冲突与互斥机制 fail-closed

当前拒绝：

- 完全重复的 rule/target/context 实例；
- 同一 alternative family、同一 target 的冲突选择；
- 同一 target 同时选择 native/wide program domain；
- 同一 access 同时选择 integer-offset/pointer-staged address；
- 同一 access 同时选择 masked/unmasked access；
- 同一 reduction 同时选择 explicit-axis/flattened reduction；
- 同一 contraction 同时选择 K-tiled dot/outer product。

不存在“后一条覆盖前一条”的隐式优先级。

### 4.7 语义值只引用、不复制

π 禁止复制：

- operator、operands、常量、dtype、shape；
- reduction fn/axis/identity/masked contribution；
- predicate、fill、store guard、effect、output value；
- C 的 concrete value；
- Triton API、原始源码或来源片段。

这项约束允许 paired intervention 在 π 冻结时只修改 J/B，并检验 replay 是否产生目标 TTIR 变化。

### 4.8 规则卡按 π 精确检索

writer payload 只包含：

```text
完整有序 π
π 中每一步的 selection trace
π 实际选择到的规则卡
registry digest
```

未被 π 选择的 canonical 卡不会作为隐藏 baseline 注入。规则库解释 π，但不替 π 决策。

### 4.9 空 π 语义收紧

空 π 不再表示“全部采用默认 lowering”。它只在 artifact 确实没有任何重放工作时合法。对于包含 grid、参数、access 或 J node 的普通 Kernel，空 π 会因完整性缺失被拒绝。

### 4.10 masked-J 也先验证原始完整 π

masked-J 会把 J/B 标识和内容变成 opaque view，因此不能从遮蔽后的 packet 重建完整性。当前实现先对未遮蔽 artifact 执行 strict-plan validation，再生成 masked packet，防止无效 π 只在 ablation 通道漏过检查。

### 4.11 Lift 与 replay prompt 同步收紧

Lift prompt 现在明确要求：

- 输出完整、按序计划；
- canonical 与 alternative 规则只要适用都必须显式实例化；
- 不允许假定 writer 会补默认步骤；
- 语义值只通过 refs 读取；
- native/wide、explicit/flattened、dot/outer-product 等选择必须明确。

Replay prompt 明确要求逐步执行 π，并禁止隐藏默认步骤。

### 4.12 协议与 cache identity 升级

当前协议为：

```text
relift_only_pi_audit_on_strict_opaque_refs_causal_v7
```

v4 恢复了完整严格 π，但 paired intervention 仍让模型从空白重新生成整个 mutant
Kernel，真实 softmax 因此出现与目标变异无关的 `constant/select` 重构漂移。v5
把 normal 的模型生成 replay 作为受控结构基线交给同一 source-blind writer：writer
仍只看到当前 J、同一完整 π、接口与 C 槽，不看到 source、mutation sidecar 或期望
TTIR delta；未变化的 π 步骤禁止被重新分组或“好心优化”。

v5 同时把 live harness 的已注册 J/B intervention 上限从 1 提高到 3，并在每条
记录中保存 `base_generation_id`。v6 不改变完整、有序 π 的语义，而是禁止
Lift-owned reference id 携带来源提示，并在正常 replay handoff 中统一重命名 J/B/π
引用。v7 进一步取消所有 normal/消融 replay-code repair，把正常失败统一回流 Lift，
并让正式 π audit 默认开启。held-out batch identity 当前升到 schema 17：除
intervention 与 generation 预算外，还绑定 TTIR score、masked-J fallback、
`opaque_replay_refs_v1`、relift-only 恢复、冻结 C 以及 full-π closed-control 前置
条件。历史 v3/v4/v5/v6 identity 不能静默当作当前结果。

### 4.13 空分量与 masked-J 裁决收紧

TTIR component 的双方空集合仍表示“没有差异”，但不再表示“获得满权重证据”。
当前 `active_evidence_weighted_v2` 将其序列化为 `applicable=false`、component
`score=null`，并从总分分子和分母同时排除；总分只在确有 source 或 replay 证据的
category 上归一化。normal Gate1 继续由硬分量精确相等裁决。

masked-J fallback 也不再使用 score ratio 阈值。当前
`exact_semantic_hard_degradation_v1` 只接受两类确定性证据：masked replay 无法形成
可编译且接口合法的指纹，或其 `op_multiset/dataflow/reduction/predicate_effect`
至少一个硬分量与 source 不同。ratio 仅保留为诊断字段。

### 4.14 refs/node_ref 正常通道不透明化

过去只有 masked-J 把 `reduce_max_0/a_load/row_domain` 一类标识符改成 opaque id，
normal replay 仍可能从名字获得来源提示。v6 增加两层约束：

- 真实 LLM Lift 输出的 node/domain/access id 必须严格为 `n0.../d0.../a0...`，
  plan 不接受 `a_load` 这类 Lift-owned reference；
- 无论持久化的是新 artifact 还是历史 artifact，交给 normal writer 前都确定性生成
  `opaque_replay_refs_v1` view：J tensor/value、node、domain、access、π plan、rule
  selection 使用同一组 opaque handles；ABI/C reference 使用 `p*/c*`，并只通过无
  来源名称的 `reference_bindings` JSON path 定位。

持久化 artifact 仍可保留历史诊断名称，便于 Relift 和人工定位；这些名称不再越过
Lift/replay 隔离边界。payload 断言会拒绝任何未登记或非 opaque 的 plan/selection
reference。

真实 `gpt-5.6-sol/xhigh` fresh add 验证中，首个 Lift response 因 plan 引用了非 opaque
tensor 名 `out` 被拒绝；一次修复后 artifact 使用 `n0/d0/a0..a2`。normal writer
实际看到的 target 序列为
`p4,c0,c1,p3,d0,a0,a0,a1,a1,n0,a2,a2,t2`，Gate1
`REPLAY_CLOSED`、3/3 paired intervention PASS、Gate2 PASS、0 relift。另将五个 v5
历史 artifact 重新构造 v6 packet，`acc_a/a_x_load/a_load/d_row` 等旧诊断名在
J/plan/rule-selection/reference-bindings 四个 writer channel 中均为零残留。

### 4.15 历史 v6：π 重放引导与干净消融复验（2026-07-16）

本节记录的是 v6 当时的实验口径，其中仍存在受限接口适配器；它不是当前 v7
实现合同。当前合同见下一节：所有 replay-side adapter/repair 均已删除。

正式复验前先消除了三类会伪造 π 结论的实验混杂：

1. 消融适配器收紧为 `interface_only_trailing_unused_constexpr_v2`。其函数参数只接收
   `KernelInterface`，从类型上无法读取 J、π、boundary、schedule 值或
   `reference_bindings`；只允许补齐接口已声明、位于参数末尾、且函数体完全未使用的
   constexpr。runtime 参数改名/重排、opaque handle body rewrite 均不代修。
2. 消融编译现在与完整 π 控制组使用同一份冻结 artifact C；此前 add J-only 的
   `missing BLOCK` 是消融驱动漏传 schedule 的实验错误，不是 π 必要性证据。
3. 只有完整 π 控制组已经 `REPLAY_CLOSED` 才允许启动消融。source 指纹缺失或
   full-π `NO_CLOSURE` 时记录 `pi_ablation_control.valid=false` 并跳过，不能用无效
   控制组计算 π kill。消融候选代码也随报告保存，便于复核生成/编译失败。

使用真实 `gpt-5.6-sol/xhigh`、相同 Lift/rewrite/relift 预算，对五个标准 probe 重新
执行当前 v6 packet。add 显式复用本轮已有的 fresh-v6 Lift artifact 后重新生成正常
控制与消融；row-sum、softmax、RMSNorm、matmul 均 fresh v6 Lift。没有把 v5 terminal
report 静默升级为 v6 结果。

| Kernel | Lift attempts | π steps | paired J/B | full-π Gate1 | normal Gate2 | relift |
|---|---:|---:|---:|---|---|---:|
| add | 2（复用 fresh-v6 artifact） | 13 | 3/3 PASS | `REPLAY_CLOSED` | PASS | 0 |
| row-sum | 2 | 12 | 1/1 PASS | `REPLAY_CLOSED` | PASS | 0 |
| softmax | 2 | 18 | 2/2 PASS | `REPLAY_CLOSED` | PASS | 0 |
| RMSNorm | 3 | 24 | 2/2 PASS | `REPLAY_CLOSED` | PASS | 0 |
| matmul `17×13×19` | 2 | 18 | 2/2 PASS | `REPLAY_CLOSED` | PASS | 0 |

完整 π 因而在五个标准 probe 上得到 5/5 source-blind TTIR 硬闭合、10/10 合法 paired
intervention 和 5/5 独立 normal Gate2 PASS。该结果说明当前 π 能稳定引导这些样本的
来源结构，但仍只是五个 probe，不能外推到数百 Kernel。

干净 π 消融结果如下；`DEGRADED` 只在消融分支成功生成、编译并得到 TTIR 后成立：

| Kernel | J-only | 单步 dropout | mechanism substitution |
|---|---|---|---|
| add | `UNEXPECTED_CLOSURE` | generation failed：删 `realize_elementwise@n0` 后中间值未定义 | `index→pointer@a0`，`DEGRADED` |
| row-sum | `DEGRADED` | generation failed：删 `pointer-tile@a0` 后 `column_offsets` 未定义 | `pointer→index@a0`，`DEGRADED` |
| softmax | `DEGRADED` | 删 `pointer-tile@a0`，`DEGRADED` | `pointer→index@a0`，`DEGRADED` |
| RMSNorm | `DEGRADED` | 删 `pointer-tile@a0`，`DEGRADED` | `pointer→index@a0`，`DEGRADED` |
| matmul | `DEGRADED` | 删 `pointer-tile@a0`，`DEGRADED` | `pointer→index@a0`，`DEGRADED` |

五个 J-only 分支全部一次生成并编译；4/5 硬退化，add 仍完全闭合。三个可编译的
step-drop 分支全部硬退化，另两例只能作为“π 对生成完整性有操作性作用”的证据，
不能计入结构消融分母。五个 mechanism substitution 均一次生成、一次编译并硬退化。
全部 13 个可编译消融分支都没有实际触发当时存在的确定性 adapter；v7 已删除该
adapter。其中 12 个退化分支的失败硬类别一致为 `op_multiset + dataflow`，reduction
与 predicate/effect 保持通过。这说明本轮
π 的主要可观测边际信息是来源相关的地址 association/施工结构，而不是把 reduction、
softmax、RMSNorm 或 dot 语义复制进 π。

额外边界样本 `matmul M=7,N=1003,K=19` 不计入上述五核标准表。其完整 π 在一次
semantic relift 后仍 `NO_CLOSURE`，最佳结构 score 为 `0.740510`，但独立 Gate2
diagnostic 中 source↔replay、source↔lowering、replay↔lowering 三对全部 PASS。
该结果再次直接证明 Gate1 与 Gate2 不是包含关系：这里是合理的“Gate2 通过、Gate1
不通过”。边界运行发生在 closed-control guard 落地之前，其 π 消融记录控制组无效，
全部排除，不进入任何 π 成功率或 kill 统计。

当前证据同时暴露一个尚未解决的上限：dropout 的确定性优先级通常选择首个 compute
或 address target，本轮复杂四核又都集中到 `a0`，所以它并未覆盖 π 的全部 step 或
所有 mechanism family。mechanism substitution 比 dropout 更稳定，但当前也只验证
首个可替换 address association，尚不能证明 21 张规则卡逐卡必要、π 已最小化或复杂
attention/atomic/scatter/loop 等特性已覆盖。

### 4.16 v7：取消 replay code repair，正式 π audit 默认常开

用户指出的偏航成立。原方案早期第六、七章确实允许“π/重放代码小修”，这是此前
实现 replay feedback/base-candidate repair 的来源；但实际代码修的是某个 replay
候选，并没有重新产出 π，因此既不能证明 Lift 的 J/π 已经正确，也会让闭合依赖旧
候选代码。v7 按当前主线取消这条内循环。

normal Gate1 现在只有 source-aware 外循环：

```text
source + 当前 J/π
  → fresh replay
  → generation / compile / interface / hard fingerprint
      ├─ 全部闭合：进入 paired/masked 终检与最终 π audit
      └─ 任一失败：提取失败类别或确定性 TTIR diff
                    → 与 source 一起回流 Lift
                    → 重产 J/π/location_map
                    → 从静态扫描、源编译、fresh replay 开始全流程重跑
```

writer API 已删除 compiler/comparator `RepairFeedback`，normal replay 永远不接收旧
失败候选；`Gate1Config.max_repairs` 与 live/held-out `--rewrite-repairs` 也已删除。
relift trigger 明确区分 `replay_generation_failed`、`replay_compile_failed`、
`replay_evaluation_failed` 与 `normal_no_closure`。Lift 内部对不合法 artifact JSON 的
修正仍允许，因为它始终看 source 且重新发出完整 J/π；provider 对同一请求的 HTTP
重试也只是传输容错，不属于 replay 语义修复。

编译失败不会只回传一个空泛的“失败”标志，也不会回传 replay body。compiler
diagnostic 会先确定性去除 host path、traceback、编译器引用的代码行与 caret，只保留
异常类别和原因摘要，再作为 Lift evidence。这样 Lift 能据此补足 J/π 合同，但无法
沿着旧 replay 做逐行补丁。

π audit 的“默认关闭”过去只是节省额外 LLM/编译成本的原型折中，不是正式语义。
正常 replay 从未关闭 π；关闭的是 J-only、step-drop、mechanism-substitution 这组三类
因果审计。当前 π/规则库仍在扩展，若正式运行默认不审计，就无法持续证明 writer
确实读取 π、π 是否冗余或是否出现新泄漏。因此 v7 的策略是：

- 正式 live 与 held-out：full-π `REPLAY_CLOSED` 后默认执行一次 π audit；
- quick smoke / 纯单元测试：允许显式 `--no-pi-ablations`；
- 中间 relift 轮次不做消融，只有最终有效控制组做一次；
- 消融只作诊断，不修改 Gate1 verdict，不触发 relift；
- 消融 replay 不做 LLM 小修或 AST/ABI/body adapter，生成/编译失败直接
  `inconclusive`。

“π 已成型”也不能永久取消消融。成熟后可以改为分层抽样，但每次 π schema、规则卡、
Lift/replay prompt 变更以及 held-out 新家族仍必须执行；否则只能证明完整 π 能闭合，
不能证明闭合因果上依赖 π。

### 4.17 fresh v7 实测：默认 audit 与无 replay repair 证据（2026-07-16）

使用真实 `gpt-5.6-sol/xhigh` fresh Lift 对 add 和边界 matmul
`M=7,N=1003,K=19` 复验。两例 normal writer 都只生成一个 fresh candidate，报告中
`base_generation_id=null`，均编译并通过 Gate1 硬指纹；Gate2 也均 PASS。两例 Lift
各发生一次 artifact-level schema/opaque-reference 修正：首个完整 artifact 的 π
引用了非 opaque 的 `out`，Lift 随后重新发出完整 J/π。这不是 replay 修复。

matmul 命令没有传 `--pi-ablations`，仍自动进入三档 audit，证明正式 live CLI 的
默认开启策略实际生效。两例每个消融分支均为 `attempts=1`，没有 LLM 小修、AST/ABI
适配或 relift：

| Kernel | J-only | 单步 dropout | mechanism substitution | full-π Gate1 | Gate2 |
|---|---|---|---|---|---|
| add | `UNEXPECTED_CLOSURE` | `GENERATION_FAILED`（inconclusive） | `DEGRADED` | `REPLAY_CLOSED` | PASS |
| matmul `7×1003×19` | `DEGRADED` | `DEGRADED` | `DEGRADED` | `REPLAY_CLOSED` | PASS |

该结果不能解读为“add 的 π 不合理”或“所有消融都必须退化”。add 属于低熵语义，J、B
和接口本身可能足以让 writer 复现结构，所以 J-only 闭合是合法观测；dropout 生成失败
只是不完备实验，不能算 π 有效；mechanism substitution 的硬退化才是该实例中明确的
π 机制因果证据。matmul 三项全部退化，则说明复杂实例对 π 的完整步骤和机制选择均
明显敏感。

这两个 fresh live 样本都在首轮 normal replay 闭合，因此 **没有** 产生真实 relift
轨迹，不能把它们写成 live relift 成功证据。relift-only 分支由确定性回归覆盖：
generation failure、compile failure 和 hard semantic mismatch 均触发 Lift，新一轮
重新编译 source 并生成 fresh replay，失败候选从不成为 writer 输入。

## 5. 当前规则库

规则库共有 21 张卡，其中实现字段仍使用历史名称 `selection_mode=derived|override`：

- `derived` 现在表示通常依据 J/B/C 选择的 canonical mechanism；
- `override` 表示来源相关的 alternative mechanism；
- 两类卡只要适用，都必须由 π 显式实例化；
- `derived` 不再表示系统会自动注入该步骤。

| 规则卡 | 类别 | 作用 |
|---|---|---|
| `bind_program_domain` | canonical | 原生宽度 program coordinate |
| `bind_wide_program_domain` | alternative | 显式宽化 program coordinate |
| `materialize_index_tile` | canonical | 先形成完整整数 offset |
| `materialize_pointer_tile` | alternative | 保留 pointer + scalar base + lane 的结合方式 |
| `realize_masked_access` | canonical | masked load/store |
| `materialize_mask_tile` | alternative/additional | 保留已暂存并复用的 mask component |
| `realize_elementwise` | canonical | unary/binary map |
| `realize_const` | canonical | 常量节点 |
| `realize_reduce` | canonical | 显式 local tile axis reduction |
| `realize_flattened_reduce` | alternative | source 省略 axis 的 flatten reduction |
| `realize_broadcast` | canonical | 语义 broadcast/local rank insertion |
| `realize_k_tiled_dot` | canonical | 有 contracted label 的 tiled contraction |
| `realize_outer_product` | canonical | 无 contracted label 的 outer product |
| `realize_indexed_access` | canonical | gather/data-dependent index |
| `bind_runtime_scalar` | canonical | extent 或 J scalar input 绑定 |
| `declare_schedule_slot` | canonical | constexpr/compiler slot 声明 |
| `realize_cast` | canonical | cast |
| `realize_select` | canonical | select |
| `realize_loop_carry` | canonical | structured loop carry |
| `realize_unmasked_access` | canonical | 无 predicate 的 load/store |
| `realize_output_mapping` | canonical | J value 到输出的映射 |

当前 4 张 `selection_mode=override` 卡及 family 是：

| alternative 卡 | family |
|---|---|
| `bind_wide_program_domain` | `program_coordinate_width` |
| `materialize_pointer_tile` | `address_association` |
| `materialize_mask_tile` | `mask_materialization` |
| `realize_flattened_reduce` | `reduction_call_shape` |

注意：`realize_k_tiled_dot` 与 `realize_outer_product` 都是 canonical 卡，由 J subscripts 决定选哪一个，不属于上述 4 个 override family。

## 6. 为什么撤回 sparse unordered π

历史 v2 的实际语义是：

```text
实际重放 = 稀疏 δπ + 通用规则卡 + prompt 隐含默认 + LLM 推断
```

它存在四个根本问题：

1. π 本身无法独立说明完整施工过程；
2. 缺失步骤和“默认步骤”无法区分；
3. 无序化丢失 domain → address → access → compute → store 的自然依赖；
4. repair、relift 和 drop-step 消融无法稳定定位一个明确的 π 步骤。

只有满足以下全部条件，sparse 表示才可能作为内部压缩编码合理存在：

1. 有纯代码、确定性的 baseline materializer；
2. baseline 显式版本化、序列化并可审计；
3. `π_full = materialize(B, J, C, interface, δπ)` 唯一确定；
4. writer 实际看到并执行的是 materialized full plan；
5. full plan 有独立 identity/hash；
6. 缺失或冲突会 fail-closed，而不是交给 LLM 猜测。

当前项目没有这样的 materializer，所以稀疏对象最多只能在未来叫 `δπ` 或 `π_delta`，不能直接作为正式 π。正式 artifact、Gate1 和 writer 使用的仍必须是完整 π。

## 7. 保留下来的历史优化

撤回 sparse 定义不等于回到会泄漏语义的旧式自由文本 π。以下优化全部保留：

- 三字段 closed schema；
- typed target/ref；
- 禁止源码和前端 API 泄漏；
- 禁止复制 J/B/C 语义值；
- registry digest 与精确卡片检索；
- duplicate/conflict fail-closed；
- source/API 静态泄漏扫描；
- paired J/B causal intervention；
- masked-J 降级为诊断或无干预坐标时的 fallback；
- relift 时 interface 与 C 冻结；
- Gate1 与 Gate2 维持独立证据关系。

## 8. Paired intervention 与 π 的关系

paired intervention 是 Gate1 的因果验证机制，不是 π 生成规则。

当前做法是固定完整 π、interface、C、规则库和 source，最多选择 3 个合法、互不
破坏合同的注册 J/B 坐标。每个 mutant replay 都复用同一个 normal 模型生成候选作为
结构控制，只修改当前 J 要求的部分，再要求 TTIR 出现精确、可枚举的目标差分。这能验证：

- writer 是否真正读取 J；
- π 是否偷偷复制了被干预的语义值；
- 目标变化是否伴随 off-target drift。

恢复完整 π 后该机制仍然成立，因为 π 只引用 J/B 值，不复制它们。

blank/masked-J 会混合测量接口可猜性、模型先验、规则卡信息和 π 泄漏，因此高相似度不能单独证明 π 错误。它不能成为削弱 π 定义的理由。

masked-fill intervention 还会沿 J DAG 检查是否流入带
`masked_contribution` 的 reduction。若 fill 经 elementwise 节点间接进入 reduction，
单改 fill 会迫使 writer 合法新增 select，因此该坐标必须排除；RMSNorm 的 `x` load
就是由真实失败沉淀出的这一规则。provider 的 408/503 仍 fail-closed，但每次 replay
generation 现在最多进行 3 次有界 transport 尝试。

## 9. 历史证据边界

以下协议与结果仅作为历史试验保留：

```text
π-v2 sparse unordered override
normal_no_closure_pi_override_causal_v3
```

历史 causal-v3 记录包括：

| Kernel | 当时的 sparse π | 历史 Gate1 结果 | 说明 |
|---|---|---|---|
| add | 空 | PASS | `add→sub` 精确 TTIR delta |
| row-sum | 少量 override | PASS | 合法复合 reduction 干预通过 |
| RMSNorm | 3 条 pointer association | PASS | `mul→add` 通过；blank-J 仍可完全猜出 |
| matmul | 少量 pointer association | PASS | masked fill 干预通过 |
| softmax | sparse override | FAIL | 目标变化发生，但出现 constant/select off-target drift |
| FlagGems ones | 1 条 wide-domain alternative | PASS | output constant 干预通过 |

这些结果证明 causal intervention 机制有价值，也暴露 sparse π 的行为，但不能宣称 strict-v5 已经通过真实 LLM 端到端验证。不得：

- 把 v3/v4 artifact 或 terminal report 静默升级为 v5；
- 复用旧 cache 作为 v5 正式结果；
- 只改协议字符串而沿用旧空 π/稀疏 π；
- 用历史 4/5 或 held-out 成功率冒充 v5 覆盖率。

### strict-controlled-causal-v5 五核证据（2026-07-15）

五个 Kernel 均使用真实 `gpt-5.6-sol/xhigh` strict π artifact。add 是完整 fresh v5
Lift；其余四例显式复用本轮刚生成并已通过 strict 校验的 v4 Lift artifact，只重跑
v5 causal/Gate2 阶段，报告保留 `reused_from`，没有静默复用：

| Kernel | π steps | 合法 interventions | Gate1 | Gate2 | masked-J |
|---|---:|---:|---|---|---|
| add | 13 | 3/3 PASS | `REPLAY_CLOSED` | PASS | hard-close, score 1.0 |
| row-sum | 12 | 1/1 PASS | `REPLAY_CLOSED` | PASS | hard-close, score 1.0 |
| softmax | 18 | 2/2 PASS | `REPLAY_CLOSED` | PASS | hard-close, score 1.0 |
| RMSNorm | 24 | 2/2 PASS | `REPLAY_CLOSED` | PASS | compile 后接口不闭合 |
| matmul | 18 | 2/2 PASS | `REPLAY_CLOSED` | PASS | score 0.5811 |

softmax 的两个坐标分别为 `sub→add` 与 `div→mul`，均只产生目标 operator/dataflow
delta。RMSNorm 首轮 strict Lift 曾因 π 引用未登记中间名 `inv_rms` 被 schema 拒绝，
第二次 response 修复后通过；其 `x` fill 假干预随后被 dataflow-aware eligibility
排除。五例均为 0 semantic relift。

这组结果证明多个已注册 J/B 坐标对 replay 有确定性因果控制。

### J-only、π step-drop 与 mechanism substitution 诊断

以下是 v5/v6 历史实验口径：当时三档 π 消融接入 live harness 但默认关闭，且只作 diagnostic，
不修改 Gate1 verdict。正常 artifact 必须先通过完整 strict π 校验；消融 packet 才
允许故意使用空 π、删一步 π，或在保持完整性的前提下替换一个互斥 mechanism。
每个消融当时最多小修一次。v7 已删除该修复。只有成功编译后丢失至少一个 Gate1 硬分量，才记为结构级
`supports_pi_necessity=true`；生成/编译失败保留为 inconclusive，不再被包装成成功。

五核真实 `gpt-5.6-sol/xhigh` 结果：

| Kernel | J-only | 确定性 step-drop | 删除内容 |
|---|---|---|---|
| add | `UNEXPECTED_CLOSURE` | generation failed，inconclusive | `realize_elementwise@n0` |
| row-sum | `DEGRADED` | generation failed，inconclusive | `materialize_pointer_tile@a0` |
| softmax | `DEGRADED` | `DEGRADED` | `materialize_pointer_tile@a_x_load` |
| RMSNorm | `DEGRADED` | `UNEXPECTED_CLOSURE` | `materialize_pointer_tile@a_x` |
| matmul | `DEGRADED` | `UNEXPECTED_CLOSURE` | `materialize_pointer_tile@a_load` |

受控 mechanism substitution 保持 π 完整、冻结 J/B/C，并以 normal 模型生成 replay
为结构基线，只替换首个 address association。五核均成功编译且硬退化：

| Kernel | 替换 | 结果 |
|---|---|---|
| add | `index→pointer @ acc_a` | `COMPILED + DEGRADED` |
| row-sum | `pointer→index @ a0` | `COMPILED + DEGRADED` |
| softmax | `pointer→index @ a_x_load` | `COMPILED + DEGRADED` |
| RMSNorm | `pointer→index @ a_x` | `COMPILED + DEGRADED` |
| matmul | `pointer→index @ a_load` | `COMPILED + DEGRADED` |

表中的 `a_x_load/a_load` 是 v5 历史报告为了人工定位保存的 source-side diagnostic
label，不是 v6 writer packet 内容；同一目标在 `opaque_replay_refs_v1` 中只显示为
`a0/a1/...`。

这说明：

- add 的 canonical J/B 加完整规则库足以唯一恢复源结构，J-only 不能作为所有 Kernel
  的硬拒绝条件；
- row-sum、softmax、RMSNorm、matmul 的 J-only 都丢失硬结构，π 对来源相关 realization
  有可观测价值；
- 单步 dropout 也不是普遍硬判据。同一 rule card 若仍被其他 target 的 π 实例选中，
  writer 可能把该机制推广回被删 target；RMSNorm/matmul 即出现这种情况；
- 受控 mechanism substitution 比 dropout 更稳定地证明 π 对来源相关 realization
  有因果控制；跨 Kernel 整份 π swap 因 typed target 不兼容，仍只适合作为后续
  held-out 研究。

这组结果与 paired J/B intervention 构成两个方向互补、但不能相互替代的因果证据：

- **固定 π、修改已登记的 J/B 语义坐标**：replay 产生精确且可枚举的目标 TTIR
  delta，证明 writer 确实读取 J/B，同时完整 π 没有复制并泄漏这些语义值；
- **固定 J/B/C/interface、修改 π 的 realization mechanism**：五核均可编译但出现
  Gate1 硬结构退化，证明 writer 确实服从 π 指定的来源相关施工选择，而不是仅凭
  J、规则库或模型先验恢复原结构。

前者检验语义通道，后者检验施工机制通道；两者共同支持当前 J/π 职责分离，但都
仍是 Gate1 内部的诊断证据，不能替代 Gate2 的真实执行与 lower 证据。

### π 合理性的当前证据等级

当前可以支持的是“π 的定义和五核初步机制合理”，而不是“π 已被普遍证明正确”：

1. **结构证据**：π 的 closed schema 只允许 `rule/node_ref/refs`；五核 artifact 均
   通过 strict completeness、typed target、冲突检查和静态泄漏扫描。π 不携带
   operator、operand、predicate、fill、C 值、源码片段或 Triton API。
2. **J 通道因果证据**：固定完整 π，修改已登记 J/B 坐标，五核共 10 个 paired
   intervention 全部产生精确注册 TTIR delta，没有 ignored-J 或 off-target 漂移。
3. **π 通道因果证据**：固定 J/B/C/interface，仅替换一个 address realization
   mechanism，五核均一次编译成功且 Gate1 硬结构退化，说明 writer 真实服从 π，
   而不是把 π 当成装饰。
4. **外部行为证据**：五核 normal replay 均 `REPLAY_CLOSED`，独立 Gate2 均 PASS。
   Gate2 不证明 π 唯一或最小，但排除了“结构闭合却在这批执行样例上行为错误”。
5. **反证与边界**：J-only 中 add 仍闭合，说明 canonical 简单 Kernel 的 π 可能没有
   额外区分信息；step-dropout 结果混合，不能宣称每一步均必要。当前 substitution
   每核只覆盖首个可用 address target，也尚未覆盖复杂 attention、atomic/scatter、
   block pointer、descriptor 或全部 alternative family。

因此当前合理结论是：π 作为“完整、有序、只引用 J/B/C 的来源相关施工计划”已有
双向因果与五核执行证据；π 的逐步最小性、复杂 Kernel 可扩展性和全规则覆盖仍是
待继续验证的命题。

## 10. 代码落点

| 内容 | 文件 |
|---|---|
| π 数据模型与顺序语义 | `python/intentir/verify/gate1/model.py` |
| 规则库、完整性、适用性、冲突检查 | `python/intentir/verify/gate1/rules.py` |
| source-blind replay payload/prompt | `python/intentir/verify/gate1/prompt.py` |
| Lift/Relift prompt 与当前 relift-only strict opaque-ref v7 协议 | `python/intentir/extraction/lifters/scrc.py` |
| LLM wire schema | `python/intentir/extraction/lifters/scrc_schema.py` |
| Gate1 主循环验证接入 | `python/intentir/verify/gate1/orchestrate.py` |
| 静态泄漏检查 | `python/intentir/verify/gate1/leakage.py` |
| paired intervention | `python/intentir/verify/gate1/interventions.py` |
| J-only / π step-drop / mechanism substitution diagnostic | `python/intentir/verify/gate1/pi_ablations.py` |
| 项目 Gate1 spec | `.trellis/spec/verification/scrc-gate1.md` |

## 11. 当前验证与能力边界

本轮测试覆盖了：

- 完整 π 的正常生成视图；
- construction order 保留；
- 非平凡 artifact 的空 π 拒绝；
- canonical/alternative 卡均可被完整 π 实例化；
- target 缺失、未知 target、错误适用类别拒绝；
- duplicate、family conflict 和互斥 mechanism 拒绝；
- masked-J 前先验证原始完整 π；
- paired intervention 固定 π 的行为；
- J-only、确定性单步 π dropout 与受控 mechanism substitution 的 source-blind
  单次生成、编译和硬结构比较；失败不修 replay；
- Lift compiler-view 接缝；
- Gate1/Gate2/SCRC 相关回归。

仍不能宣称：

- 21 张规则卡已覆盖全部 IntentIR op 或数百个真实 Kernel；
- 历史 v3/v4/v5/v6 的 5 核/FlagGems 结果自动成为当前 v7 结果；
- 每一张 π 卡已经完成逐卡必要性消融，或 π swap 已形成硬裁决；
- mechanism substitution 已覆盖每个 π step 或每个 alternative family；当前五核试验
  每个 Kernel 仅替换首个可用的 address mechanism target；
- structured loop、atomic/scatter、block pointer、descriptor、复杂 attention 等特性已充分覆盖。

下一步正式证据必须从真实 Kernel 重新走 `gpt-5.6-sol/xhigh` Lift，生成完整 π，再执行 source-blind replay、Gate1 paired intervention 和独立 Gate2。手写 J/π 只作为单元测试或机制探针，不计入规模化端到端覆盖。

## 12. 最终定义

当前 π 的正式定义为：

> π 是 Lift 生成的完整、有序、严格的重放规则实例序列。它通过 typed references 引用 J、boundary、interface 与 schedule，明确规定完成重放所需的全部施工步骤；它不复制 J/B/C 语义、不包含源码或前端 API，也不允许 writer 补充隐藏默认步骤。

历史 sparse unordered override 是一次有价值但改变了 π 属性的试验，因此已撤回为正式定义，只保留为历史对照。
