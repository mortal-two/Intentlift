# LLM 提取

## 核心立场:LLM 直接提取到原子 IR

LLM 的工作是把任意 kernel **直接提取成张量级原子的 IntentIR JSON**(三层 A/B/C)。这个"理解 + 原子化分解"的动作**就是论文的科研核心价值**。

- LLM **是** 受约束的语义提案者 + 原子化分解者。
- LLM **不是** 正确性裁判(由验证裁决),也不是"直接生成目标 kernel 代码"的生成器。
- **禁止**把原子化分解外包给下游 compiler pass——那会把 LLM 降级成"打标签的",偏离论文卖点(见 [../project/constitution.md](../project/constitution.md))。

## 提取流程

1. 取 kernel 源码、同一 source compile case 的编译器视图摘要(C 绑定、
   compiler slots、alias contract),以及可选 evidence hint(见
   [evidence.md](evidence.md))。C 若只存在于 launch site，禁止要求模型从
   kernel 函数体猜值。
2. LLM 在 IntentIR 规范与 LiftCore wire JSON Schema 约束下，提出
   J(A+B)、接口、C.schedule/C.strategy 与定位表。LiftCore 禁止输出 `plan/π`；
   同一次 Lift 在结构化 J/C 构造后由 deterministic composer 生成 π 与逐叶
   provenance，再把完整 LiftArtifact 交给 Gate1。固定协议对象禁止未知键；
   op-specific `attrs` 仍由 IntentIR op vocabulary 管理，但 correctness-relevant B
   只能进入 program-boundary 的 closed records，active handoff 拒绝 legacy
   `op.layer_b`。J 内的 `layer_c` 必须为空，具体 C 只进入独立 schedule。
   最终由项目 strict JSON/schema parser 裁决并做有界的 source-aware Lift repair；
   typed lowering parser 不在这里形成额外 semantic admission。
3. SCRC Gate1 用源码隔离重放和双 TTIR 结构闭合裁决 J/π 是否装下源结构。
4. Gate1 闭合后，Gate2 用同一输入分别执行 source、最终 J replay 与 typed
   `IntentProgram` scalar lowering，比较 `source↔replay`、`source↔scalar`。
5. JSON/schema 构造失败只进入有界 Lift repair；形成合法 artifact 后，normal replay
   没有 inner repair。Gate1 hard diff 只回到 source-aware relifter 重新提出完整 J/C core；
   超过上限则记录未闭合，不强行接受。

## SCRC 下必须区分两种 LLM 角色

- **提升官/lifter**：可看源 Kernel，负责提出 J、接口、C 和定位表；π 由同一次
  Lift 的 deterministic finalizer 生成；
- **重写员/rewriter**：绝对不可看源码，只能接收 J、π、接口、C 槽和可信
  规则卡，生成用于 [../verification/scrc-gate1.md](../verification/scrc-gate1.md)
  compile-only/TTIR 闭合的重放 Kernel。

两者不能复用同一个包含源码的 conversation/context。LocationMap、源 TTIR、
C 绑定值与源侧差异明细只能留在提升侧；重写员不接收 prior replay、候选代码、
编译错误或 comparator feedback，也不存在 replay repair API。只有已经形成的 Gate1
结构化 hard diff/净化诊断能进入下一轮 source-aware relift。重写员生成代码是验证探针，不改变
“LLM 直接提取到原子 IntentIR”这一科研主线，也不能成为生产语义真值。

## Layer-B predicate 生成与 typed handoff

LiftCore 中 `boundary.domains` 与 access predicate 虽以字符串传输，但不是任意
source-realization 文本。prompt 要求它们可无损解析成同一个 target-neutral
`ContractExpr v1` AST，并递归证明根为 boolean：
算术只消费/产生 scalar，comparison 产生 boolean，`and/or/not` 只消费 boolean。当前没有
symbol type environment，因此 bare symbol 不得靠 Python truthiness 充当 predicate。

合法形式例如 `0 <= row < n_rows`；赋值、tuple、`where/with`、区间 prose、attribute、
任意 source helper 以及 `program_id/grid_axis/grid_coordinate` 等 realization call 都
不能进入 typed lowering。compiler 不抽取后缀、不补 `True`、不删除 predicate，也不
根据 Kernel 名改写。

`Gate1Artifact` 与 Gate1 replay/fingerprint 只保留字符串，不调用 typed predicate parser
提前拒绝。Gate1 闭合后的 handoff 若解析失败，Gate2 将 scalar leg 标为 unavailable，
但仍执行 source/replay；失败不反馈给 Lift/Relift，也不产生 Gate0/Gate1.5。

当前 Responses 路径允许调用侧显式选择模型与 reasoning effort，不把具体模型
硬编码进 lifter。2026-07-14 的首个真实闭环使用 `gpt-5.6-sol/xhigh`：add 的
完整 artifact 首轮通过严格 parser，随后由独立的 source-blind rewriter 进入
Gate1/Gate2；详见 `test/scrc_live_add_report.md`。

## re-roll 是提取的一部分

- 手写高性能 kernel 大量手工 unroll / strip-mine。LLM 要把展开的重复体**卷回(re-roll)**成紧凑的张量算子 / 循环。
- 识别重复模式是 LLM 强项(优于脆弱的规则匹配),这也强化了"LLM 做提取"的主线。
- re-roll 在**提取阶段**完成,不是下游 pass(见 [../ir/structured-loop.md](../ir/structured-loop.md))。

## 为什么用 LLM 而不是纯规则

优化 kernel 把同一个张量计算用不同前端抽象、不同 schedule 编码表达,纯规则提取器很脆。LLM 作为有泛化能力的识别器,能处理没见过的 kernel——前提是**不被 evidence 完备性卡死**(见 [evidence.md](evidence.md))。
