# 分层验证架构(L0–L9)

本文件给出 IntentIR 从**今天的真实状态**(两个 C++ 字符串-enum verifier + 一个 numpy 参考解释器)走向**成熟的带证书编译器**的分层验证架构。它不是教程,是 spec:每一层只回答 **检查什么 / 救活什么 / 用什么工具 / 给什么保证 / 当前状态**。

本文件的脊柱是**诚实的保证标签**:绝不把"在某个输入域上测过"夸大成"证明过"。这是宪法 [../project/constitution.md](../project/constitution.md) 的硬约束——certificate 是"测试输入域执行一致 + 静态义务"级别的保证,不是全域形式证明。本文件把这条约束**逐层落实**。

## 两个本质不同的正确性问题(必须分开)

整个架构围绕两个**不要混为一谈**的问题展开:

- **(A) Lift 正确性**:IntentIR 是否与**源 kernel** 同义?这**天生是经验性的**。源是已优化的 GPU 代码,**没有形式语义**;LLM 是受约束的提案者(见 [../project/constitution.md](../project/constitution.md))。因此 lift 只能靠**在采样输入域上执行一致**来验证(一个采样 oracle),**永远证不出**。这是测试域置信度,不是证明。这是 lift 正确性的**诚实天花板**。
- **(B) Lowering 正确性**:GPU 输出是否等于 IntentIR 的参考语义?这是经典的 **translation validation**,**两边都是我们自己定义的东西**,所以可以做得**强很多**(现在用 differential testing;将来用 SMT / translation validation)。

**贯穿两者的第三条轴** —— **DECIDABLE vs FP-REASSOCIATING**:
- **DECIDABLE(可判定片段)**:整数 shape / mask / bound / 索引来源 / 整数归约结合性。可静态推理,可上 SMT,逐输入可 **bit-exact** 比较。
- **FP-REASSOCIATING(浮点重结合片段)**:浮点归约。并发顺序影响数值结果,**只有 tolerance 级一致才有意义**(见 [dynamic-feedback.md](dynamic-feedback.md))。

这条轴**同时切过 A 和 B**:它改变的是**逐输入的比较模式**(精确 vs tolerance),**不改变保证的类别**。

## SCRC Gate1 在本架构中的位置

[scrc-gate1.md](scrc-gate1.md) 增加的是 lift 问题(A)进入动态 oracle 前的
**静态结构前置证据**，不修改“A 的最终保证仍是测试域一致”这一天花板：

```text
source → lifter(J + π + interface + C)
       → source-isolated rewriter
       → source/replay same-C compile
       → TTIR hard fingerprint Gate1
       → Gate2: source / replay / production-lowering dynamic differential
```

Gate1 回答“J+π 是否足以源码隔离地重构同一编译器可见结构”；L3/L4/Gate2
回答“这些结构在采样输入上是否真的同行为”。`REPLAY_CLOSED` 不是新的
FULL/PARTIAL/OOS 等级，也不能把 lift 等价宣称为证明。随机值、边界 shape 和
数值 tolerance 仍只属于动态层；它们不得反向参与 Gate1 的硬裁决。

- **状态**：**BUILT（Triton MVP）**。严格 artifact/泄漏模型、隔离编译、
  TTIR 指纹/纯代码 comparator、有界修复、稳定闭合、masked-J 消融及五核
  随机/边界能力探针已落地；正式三方 Gate2、TileLang/CUDA 指纹仍按 roadmap
  后续实现。

## 与三道检查 / 契约等级的对应(不另起炉灶)

[validation.md](validation.md) 是契约判定的权威,本文件的 L0–L9 是它的**细化展开**,不发明平行词汇:

| 三道检查([validation.md](validation.md)) | 对应层 |
|---|---|
| Schema / type 检查 | L1 / L1b |
| 静态 obligation 检查(PASS/FAIL/UNKNOWN) | L5 |
| 动态源码执行一致性 | L3 / L4(lift)、L8(lowering) |
| 契约等级 FULL / PARTIAL / OOS | L6(**就是** validation.md 的等级,不是新等级) |

"动态可救静态"(L4 救 L5 的 UNKNOWN → PARTIAL)直接沿用 [dynamic-feedback.md](dynamic-feedback.md);evidence 的角色(可选 hint + 事后标尺 + LLM 自报 claim,**不作硬门槛**)在 L6 交汇,见 [../extraction/evidence.md](../extraction/evidence.md)。

---

## L0–L9 分层架构

每层标注真实状态:**BUILT(已建)/ PARTIAL(部分)/ PLANNED(计划)**。

### L0 — 书面参考语义(真理的定义)

- **检查什么**:每个原子算子有 prose + math 的指称语义,落在 [../ir/atomic-ops.md](../ir/atomic-ops.md) 与 [../ir/op-signatures.md](../ir/op-signatures.md)。
- **救活什么**:它是**真理本身**。没有 L0,"正确"无从谈起——A、B 两个问题都需要一个"对的定义"作锚。
- **工具 / 文件**:`spec/ir/atomic-ops.md`、`spec/ir/op-signatures.md`。
- **保证**:**定义级**。L0 不"检查"任何东西,它**规定**算子的含义;下游所有层都对照它。
- **状态**:**PARTIAL**。逐元素 / 归约 / 收缩 / select / cast / transpose 已规范;`loop`(带状态携带)与间接寻址族(gather/scatter/segment_reduce)语义已列出但未全部钉死可执行细节。

### L1 — Well-formedness verifier(ODS / C++ `verify()`)

- **检查什么**:结构合法性。fn-name enum、归约结合性 enum、contraction subscript 解析 + rank/extent 一致性 + result-index 来源。
- **救活什么**:在更贵的检查之前**挡掉畸形 IR**,让下游层只面对结构合法的输入。
- **工具 / 文件**:`lib/Intent/IntentOps.cpp`,由 `intent-opt` 驱动。具体:`MapBinaryOp::verify`(fn ∈ {add,sub,mul,div,max,min,pow,cmp})、`ReduceOp::verify`(fn ∈ {sum,max,min,prod,any,all};assoc ∈ {associative, float_nondeterministic})、`ContractionOp::verify`(解析 `lhs,rhs->out`、按字符绑定一致 extent、result-index 必须出现在某输入中)。
- **保证**:**拒绝畸形 IR;对"含义"一字不说**。verify 通过 ≠ 语义对。
- **状态**:**BUILT**。三个 verifier 存在并过 lit 测试(见 L7)。

### L1b — 符号 shape 关系检查

- **检查什么**:符号维(Sq、Sk、D)上的 shape inference + 关系约束,覆盖动态维。
- **救活什么**:把 L1 的"静态 extent 一致"升级到**符号维一致**,使动态 shape 的 kernel 也能在结构层被检查。
- **工具 / 文件**:计划中,扩展 `lib/Intent/IntentOps.cpp` 的 verifier / shape-inference 接口。
- **保证**:**shape 健全性**(含动态维),仍属结构层,**不触及值语义**。
- **状态**:**PLANNED**。
- **关键设计规则(必须遵守)**:verifier 的 **result-shape 规则必须与参考解释器(L2)逐字一致**,否则 oracle 与 well-formedness 检查会**静默分歧**。具体到现有代码:`reduce` 解释器用 `keepdims=True`(`interpret.py:71`,使 rowmax/rowsum 保持可广播),verifier 推出的 result rank 必须匹配;`map_binary` 解释器依赖 numpy broadcast(`interpret.py:57`),verifier 的广播规则必须与之同源;contraction 的 extent 绑定逻辑已在 `IntentOps.cpp:76–107`,L1b 须与 einsum 解释器(`np.einsum`,`interpret.py:75`)对齐。

### L2 — 参考解释器(KEYSTONE)

- **检查什么**:Layer A 的 **numpy 可执行指称**。把每个 IntentIR 程序变成一个**具体的值函数**。
- **救活什么**:这是让"正确性"对 **A 和 B 两个问题都可定义**的那块基石。没有 L2,源 kernel 与 GPU 输出都"无可对照"——A、B 都退化成无法裁决。
- **工具 / 文件**:`python/intentir/verify/back/legacy_v0/interpret.py`(指称),`python/tests/test_interpret.py`(golden 测试)。
- **保证**:给每个 IntentIR 程序一个具体值函数。**浮点结果按 policy 做 tolerance 比较,绝不宣称 exact**:numpy `sum` 是 pairwise、`einsum` dispatch 到 BLAS,**即便自比也是 tolerance 级**(见 `interpret.py` 文件头注释)。
- **状态**:**BUILT**(`map_binary`/`map_unary`/`reduce`/`contraction`/`select`/`cast`/`transpose`)。已能执行**完整的非 online attention**(由原子算子组合),并在 `test_interpret.py` 中**同时匹配 numpy 参考与 torch SDPA**(`test_attention_matches_reference`、`test_attention_matches_torch_if_available`,后者 torch 缺失则跳过)。**`loop`(带状态携带)尚未实现**——多结果 op 当前直接 `raise InterpError`(`interpret.py:99`),是下一步(对应 roadmap M3)。

### L3 — 源码执行 harness

- **检查什么**:跑**原始 kernel**(Triton / CUDA)在采样输入上得到 golden 输出。
- **救活什么**:为 lift 问题(A)提供**对照真值**——没有它,L4 无从比较。
- **工具 / 文件**:Triton flash-attn golden harness。对应 [dynamic-feedback.md](dynamic-feedback.md) 的源码执行获取(local / remote / cached),记录 input shape/dtype/seed/tolerance/源版本。
- **保证**:**提供源 golden**(采样输入域),自身不下任何正确性结论。
- **状态**:**PARTIAL**(Triton flash-attn golden harness 已证可跑)。

### L4 — Differential agreement(问题 A,lift oracle)

- **检查什么**:`interpret(IntentIR)` vs 源 golden,在**采样输入域**上比较。
- **救活什么**:这是 **lift 问题的 oracle**。它能把一个静态 UNKNOWN(L5)的 kernel **救成 PARTIAL**——动态执行一致就接受(见 [dynamic-feedback.md](dynamic-feedback.md))。
- **工具 / 文件**:L2 解释器 + L3 harness,逐输入 diff;浮点用合理 tolerance,整数可 bit-exact。
- **保证**:**测试域一致,不是证明**。这是 **lift 正确性的诚实天花板**——源无形式语义,LLM 是提案者,**A 永远证不出**。
- **状态**:**PLANNED / PARTIAL**(解释器侧 L2 已 BUILT;源 golden 侧 L3 PARTIAL;端到端 diff 是 roadmap Track A 的 M2/M3 落点)。

### L5 — 静态 obligation + SMT(可判定分支)

- **检查什么**:discharge Layer-B 义务——mask 覆盖、bound、索引来源、**整数**归约结合性——经由 Z3(`z3-solver 4.15.4.0` 已在环境)。
- **救活什么**:把可判定片段从"测一测"升到"**证一证**",为 PARTIAL 之上的 FULL 提供静态依据;并明确把"证不出"标成 UNKNOWN 而非失败,留给 L4 动态兜底。
- **工具 / 文件**:静态 obligation checker + Z3,对应 [validation.md](validation.md) 的第二道检查,三态 PASS/FAIL/UNKNOWN。
- **保证**:**对可判定片段是真证明**;**UNKNOWN 必须如实报成 UNKNOWN(绝不报 pass)**。
- **状态**:**PLANNED**。
- **作用域红线(防止越界)**:L5 证的是 **IntentIR 上的 Layer-B 义务**(mask/bound/整数结合性),**不是 lift 等价于源**。L5 的"证明"**绝不能流进问题 A 的列**——A 没有形式语义,任何"证明 lift"的说法都与本文件的诚实前提自相矛盾。

### L6 — 契约决策

- **检查什么**:把 L1/L1b/L4/L5 综合成契约等级 **FULL / PARTIAL / OOS**。
- **救活什么**:**PARTIAL 是可发布的**(诚实的部分保证 + 明确作用域)。**动态执行一致(L4)可把一个静态 UNKNOWN(L5)的 kernel 救成 PARTIAL**——这正是 [dynamic-feedback.md](dynamic-feedback.md) 的"动态可救静态"。
- **工具 / 文件**:契约判定逻辑,沿用 [validation.md](validation.md) 的 FULL/PARTIAL/OOS 语义(**不是新等级**)。OOS 只留给**本质边界**(见 [../capability/boundaries.md](../capability/boundaries.md)),不是"静态证不出"的同义词。
- **evidence 的角色**:在 L6 交汇时,evidence = 可选生成 hint + 事后标尺 + LLM 自报 claim,**不是硬门槛**(见 [../extraction/evidence.md](../extraction/evidence.md))。规则提不出 evidence 只导致 obligation UNKNOWN,不导致提取中止。
- **保证**:**等级化的诚实保证**——FULL = 动态一致且必需义务全 PASS;PARTIAL = 动态一致但有 UNKNOWN / 记录假设;OOS = 本质不支持。
- **状态**:**PLANNED**(L1 侧已 BUILT;判定综合待 L4/L5 就位)。

### L7 — 验证基础设施

- **检查什么**:回归与语料维护。lit/FileCheck 回归、fuzzing + crash 约减、CI。
- **救活什么**:**保护既有正确性不被改回归**,并把崩溃用例最小化成可复现语料。
- **工具 / 文件**:`test/`(`map_binary.mlir`、`contraction-invalid.mlir` 等)、`check-intent` target、`mlir-reduce`/`llvm-reduce`。
- **保证**:**回归保护 + 语料最小化**。不对单个 kernel 的语义正确性下结论。
- **状态**:**BUILT**(round-trip 与 negative-diagnostic 套件 green;reduce 工具在环境中)。

### L8 — Lowering differential(问题 B,动态)

- **检查什么**:GPU cubin 输出 vs 解释器参考,在采样输入上比较。
- **救活什么**:为 lowering 问题(B)提供**经验置信度**——在形式 translation validation(L9)就位之前,这是 B 的主力验证。
- **工具 / 文件**:lowered kernel(`gpu→nvvm→cubin`)对照 L2 解释器的值,经由**共享的、seeded 的 `.npz` fixture**,使浮点非确定归约不会造成虚假跨侧不匹配。
- **保证**:**测试域一致(lowering 方向)**,不是证明。
- **状态**:**PLANNED**——这是 roadmap Track B 的 M5–M9 所演练的:每个 lowered kernel 都对照解释器的值做 diff。

### L9 — Lowering translation validation(问题 B,形式,aspirational 顶层)

- **检查什么**:SMT / translation validation,证 **IntentIR ≡ 目标语义**(对可判定片段)。
- **救活什么**:把 B 从"测试域一致"推到"**对可判定片段是证明**",这是架构的最高目标。
- **工具 / 文件**:计划中;对标 **Alive2**(LLVM IR peephole/InstCombine 的有界 SMT translation validation)、**CompCert**(Coq 证明的 C→asm 语义保持);**MLIR** 提供 op verifier + dialect conversion,但**没有内建的端到端形式语义**。
- **保证**:**对可判定片段是证明** —— 但带一条硬前提(见下"诚实纠正")。
- **状态**:**DEFERRED**(长周期)。

> **诚实纠正(经对抗式 fact-check 确认)**:上游 MLIR 目标 dialect(linalg / vector / scf / gpu)只发布 **prose 文档**,且有**已知语义缺口**(例如并发的 `tensor.insert_slice` 在 bufferization 之前被文档明确标为"未良定义"),且**没有生产级机制化形式语义**。因此"lowering 验证是形式的"在**今天是 aspirational**:任何形式 lowering-等价的主张,都必须**自带我们自己对 IntentIR 侧与目标侧两边的逻辑编码**——**MLIR 不白送**。这一点必须明说。(校准锚:Alive2 = 有界 SMT;CompCert = Coq 证明语义保持;MLIR = verifier + conversion 但无端到端形式语义。)

---

## 保证诚实性(汇总表)

把 {lift vs lowering} × {decidable vs fp-reassociating} 映射到**可达成的保证类别**,确保没有一层被夸大:

| | DECIDABLE(整数 shape/mask/bound) | FP-REASSOCIATING(浮点归约) |
|---|---|---|
| **(A) Lift**(IntentIR ≟ 源) | **测试域一致**(逐输入 bit-exact 比较) | **测试域一致**(逐输入 tolerance 比较) |
| **(B) Lowering**(GPU 输出 ≟ IntentIR) | **可证明**(L9,aspirational,须自带两侧编码);今天为测试域一致(L8) | **tolerance 一致**(bit-exact 在重结合下无意义) |

**两条不可逾越的红线**:
1. **A 整行都是"测试域一致",与可判定性无关**。源没有形式语义,lift 永远经验性;可判定性只改变**逐输入比较模式**(精确 vs tolerance),**不改变保证类别**。**A 列不存在"证明"。**
2. **唯一能"证明"的格子是 B × decidable**(L9),且要求**我们自己编码两侧**。L5 对可判定片段的"证明"是**IntentIR 上 Layer-B 义务的证明**,**不是 lift 等价于源的证明**——它属于 B/结构层,绝不渗入 A 列。

---

## 当前状态 vs 成熟编译器(gap 总结)

**今天已 BUILT**:
- **L1**:三个 well-formedness verifier(`lib/Intent/IntentOps.cpp`),过 lit 测试。
- **L2**:numpy 参考解释器(`python/intentir/verify/back/legacy_v0/interpret.py`),7 个原子算子 + 完整非 online attention,匹配 numpy 与 torch SDPA。
- **L7**:lit/FileCheck 回归 harness(`test/`、`check-intent`),round-trip 与 negative-diagnostic 套件 green。

**通向成熟的有序路径**:
1. **L3/L4(Track A,lift oracle)**:把源 golden harness 接到解释器 diff,得到第一批经验 lift 置信度。
2. **L2 补 `loop`**:实现带状态携带循环的解释(roadmap M3),解锁 FlashAttention online-softmax 的 lift 验证。
3. **L1b**:符号 shape 关系检查,且**与解释器 result-shape 逐字对齐**(防静默分歧)。
4. **L8(Track B,lowering differential)**:GPU cubin 对照解释器值,经共享 `.npz` fixture(roadmap M5–M9)。
5. **L5**:静态 obligation + Z3,可判定片段上真证明,UNKNOWN 如实上报。
6. **L6**:契约综合判定,落实"动态可救静态"。
7. **L9**:lowering translation validation(长周期,须自带两侧编码,对标 Alive2/CompCert)。

> **端到端载体**:FlashAttention 端到端 roadmap([../../tasks/06-25-flashattn-e2e/roadmap.md](../../tasks/06-25-flashattn-e2e/roadmap.md))是演练本架构的具体载体——**Track A** 演练 L2/L3/L4,**Track B** 演练 L8。
