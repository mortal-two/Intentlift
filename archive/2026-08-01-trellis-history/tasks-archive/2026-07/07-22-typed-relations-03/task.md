# Typed Relations 03：闭合 logical bounds、update 与 allowed-result relation

> 状态：已完成并通过全量验证（629 passed, 4 skipped）
> 优先级：P0，IntentOp Robustness 02 的直接后继
> 流程：普通 Markdown 工作项；不运行 Trellis workflow

## 目标

在唯一 canonical `J=A+B` 中增加最小、闭集、可执行裁决的 typed relation，使 parser、
`E_J` preflight、Gate2 comparator、Intent MLIR emitter 与 direct CUDA analyzer 消费同一个
J，而不是从 harness、自由文本 proof 或第二 semantic wrapper 补语义。

本任务不引入 `LogicalABI`、`IntentProgram`、RuleCard、composer、kernel recognizer 或
source-specific deterministic policy。

## 范围 A：三类 typed relation

1. `IndexBoundsSpec`：把 indexed node、data、indices、axis 精确绑定为
   `0 <= indices < dim(data, axis)`；不得使用裸 `proven=true` 或 proof prose。
2. `LogicalUpdateSpec`：精确绑定 base/indices/values/axis，并类型化
   combiner、duplicate policy 与 observable order；J 中不得出现 atomic/CAS spelling。
3. `ResultRelationSpec`：每个 logical output 默认为 exact，也可显式声明 tolerance 或
   one-of-updates；容差必须来自 J，不能由 Gate2 case 覆盖。

## 范围 B：交叉验证与匿名化

- node/value/output 引用必须属于同一个 J；
- bounds/update 必须与目标 gather/scatter operands 和 axis 完全一致；
- combiner、duplicate policy、order 与 result-relation 组合必须合法；
- one-of-updates 首批只允许 update result 直接成为 observable output；
- strict JSON、SCRC Lift schema、opaque replay view 与 masked-J view 必须同步覆盖新字段；
- 缺 relation 的 canonical scatter 必须 hard fail，而不是留给 backend 猜测。

## 范围 C：消费者

1. emitter 仅由精确匹配的 typed bounds 生成 `intent.index_bounds_proven`；
2. `E_J` 接受并验证 bounds/result relation，按实际输入检查 bounds；CPU 尚未有正式
   scatter MLIR lowering时仍以具体 update capability reason fail closed；
3. Gate2 删除 harness tolerance 语义旁路：exact/tolerance 直接读 J；
4. one-of-updates 检查 source、replay 与 scalar witness 分别属于同一 `R_J(x)`，不要求
   两个合法 racing winner 数值相同；
5. Gate2 在执行三腿前保存 immutable input pre-state，并给各 runner 独立输入副本；
6. direct CUDA analyzer 对 typed update 给出具体 supported/unsupported consumption，不再只
   报“缺 relation”。

## 能力边界

- 本任务不把 `scatter` 加入 `CPU_LOWERABLE_OPS`，也不增加 Python interpreter fallback；
- one-of checker 首批只消费来自函数输入的 base/indices/values；predicate-bearing update
  等待 first-class masked index op，融合内部 SSA
  witness 在没有 J-only observation lowering 前精确 fail closed；
- masked-scatter、general alias/effect、general recurrence 仍按各 executor 真实能力处理；
- production CUDA 是否使用 atomic 是 backend realization，不回写 J，也不参与 Gate2。

## 真实锚点

- xFormers scaled-index-add：unique logical update；
- Liger embedding backward：duplicate add + atomic realization；
- 一个固定 digest 的 racing-overwrite Triton source fixture：one-of-updates；
- xFormers index-select：typed gather bounds。

## 验收标准

1. canonical J round-trip、严格 parser、Lift schema、normal/masked replay view 全部闭合。
2. bounds 与 op 任一 operand/axis 错配均在 J 构造时失败。
3. 缺 update/result relation 或非法 policy 组合不能形成 canonical scatter J。
4. Gate2 case 不再拥有 tolerance 字段；exact/tolerance 只从同一个 J 读取。
5. 两个不同但合法的 one-of winner 可以 PASS；非 contender 或改变 untouched base 必须失败。
6. CPU、CUDA 均在外部编译前逐字段消费或给出具体 unsupported reason。
7. 不新增重型控制面、第二 semantic IR 或 per-kernel 规则。
8. 全量测试通过，任务归档并独立提交。

## 完成后的下一项工作

依据本任务冻结的 relation schema，创建 28 个真实高性能 Kernel 的正式 LLM
Lift/Replay/Gate1/Gate2/backend-conformance/performance 批次任务；批次中暴露的 executor
缺口必须作为明确里程碑闭合，不能用历史 interpreter 或 hand-written MLIR 冒充。

## 实际完成结果

- Gate1 artifact epoch 升级到 `intentir.gate1.llm-direct.v2`；旧 Boundary 缺新闭集字段时
  strict persisted parser 硬失败；
- canonical J 新增 `IndexBoundsSpec`、`LogicalUpdateSpec`、`ResultRelationSpec`，并对
  node/operand/axis、policy/order、observable output 做统一交叉验证；
- canonical gather 必须有 bounds；canonical scatter 必须同时有 bounds、update 和每个
  output 的显式 result relation；
- emitter 只有收到由具体 J node/path 投影的 bounds 才生成
  `intent.index_bounds_proven`，同签名另一 gather 不会继承授权；
- Gate2 协议升级为 `source_replay_j_relation_v4`，删除 `Gate2Case.tolerances`；exact、
  tolerance、one-of-updates 全部由同一个 J 决定；
- one-of comparator 验证 untouched base 与 active contender membership，允许 source/replay
  选择不同合法 winner，并使用独立 committed input pre-state；
- CPU E_J 运行前检查 typed bounds，识别 typed update 后仍因缺正式 scatter MLIR lowering
  精确 unsupported，没有解释器 fallback；
- direct CUDA analyzer 消费已审计的 rank-2 unique overwrite/f32 add 与 unordered duplicate
  f32 add+tolerance，racing overwrite 继续 fail closed；
- 新增 fixed-digest Triton racing-overwrite source，真实 source inventory 升级到 v2 共 9 项；
- 新增完整 schema、opaque/masked view、runtime precondition、relation comparator、consumer
  regression；最终全量 `629 passed, 4 skipped`。
