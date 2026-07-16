# TianchenIR / IntentIR 只读审计报告

> 审计主题：Op 完整度、真实 LLM 提取与 Lowering、多次尝试与候选差异处理、验证与证书、当前工程成熟度
>
> 审计日期：2026-07-10
>
> 审计对象：`/home/kingdom/phdworks/TianchenIR` 当前 `master`
>
> 审计基线：`9a689f59fbca60f23365c2f3cfdedfdbf14509c2`
>
> 审计方式：读取当前代码、规范、Trellis 任务、Git 历史、已提交语料与报告，并执行只读构建、回放和测试。除新增本报告外，没有修改代码、Trellis 状态、已有语料或 Git 历史；没有重新调用付费/外部 LLM Provider。

---

## 1. 最终结论

先直接回答最关心的三个问题。

### 1.1 Op 是否已经足够完善？

答案不是简单的“够”或“不够”。

- 如果目标是支撑当前 FlagGems/Triton 研究语料、构建高性能 GPU Lowering、验证 TianchenIR 作为 Kernel Understanding/Compilation 研究平台：**Op 骨架已经比较强，不能再称为 toy dialect，研究型覆盖面基本够用。**
- 如果目标是形成一套稳定、统一、可由 LLM 可靠生成、可由验证器严格裁决、可自动进入各类后端的 IntentIR 规范：**还不够完善。主要问题不是 Op 数量，而是同一 Op 在规范、Prompt、Python 模型、Interpreter、MLIR emitter、ODS、通用 Lowering、专用 Lowering之间没有完全闭合。**

当前 C++ Dialect 中有 27 个 Intent Op、26 个自定义 verifier 方法和约 22 个转换 Pass。后端已经覆盖 elementwise、reduction、contraction、scan、sort、gather/scatter、动态压缩、layout、RNG、pool、upsample、conv、FlashAttention 等多类研究场景。

但当前仍存在可直接复现的跨层不一致，例如：

- `transpose` 被 Prompt 允许、Interpreter 能执行，但 emitter 和 ODS 没有对应闭环；候选能够通过提取期验证，随后在 emit MLIR 时失败。
- `rsqrt` 在 Prompt 和 C++ Dialect 中存在，但 Interpreter 的 unary vocabulary 没有直接支持。
- `iota` 在 Interpreter、emitter 和 ODS 中存在，但 Prompt 没有把它提供给 LLM；当前 `arange` artifact 退化为固定长度常量展开。
- JSON 可以声明多种 dtype，但 emitter 仍明显以 `f32` 为中心；非布尔 cast 在部分路径上只是 SSA 别名，已导致 `count_nonzero` 回放失败。

因此最准确的判断是：

> **现有 Op 广度足以继续做当前研究，但还不足以作为已经冻结、跨层一致、可审计的 LLM-facing 语义标准。下一阶段应优先“闭合已有 Op”，而不是继续无边界增加 Op 名称。**

### 1.2 是否真的由 LLM 提取后进行 Lowering？

**是，真实链路确实存在，但必须分层描述。**

仓库中存在真实 HTTP Provider 客户端、gitignored Provider 配置和大量 provenance。当前共发现 111 份已保存的 `intent.json`：其中 110 份 provenance 标记为 Provider/LLM 提取，1 份明确为 hand-authored。

官方 corpus 主链确实执行：

```text
真实 FlagGems 源文件
  → LLM Provider
  → Intent JSON
  → Python Interpreter
  → 与真实 FlagGems kernel 做动态数值比较
  → emit Intent dialect MLIR
  → intent-opt -convert-intent-to-linalg
```

所以“LLM 结果是否真的进入 C++ Lowering Pass”答案为 **是**。

但当前还不是统一的：

```text
任意源 Kernel
  → 实时 LLM
  → 严格 schema/type/obligation 验证
  → 正式 certificate
  → 自动选择 specialized Pass
  → NVVM/PTX/cubin
  → GPU correctness/performance
```

当前实际有三条相对独立的链：

```text
链 A：官方 LLM corpus 主链
source → LLM JSON → smoke/dynamic comparison → generic linalg

链 B：高性能后端主链
手写 Intent MLIR / Python 构造 Intent 图 → specialized GPU Pass → cubin → PK

链 C：少量真实连接
已保存 LLM JSON → 人工指定 specialized Pass → cubin → GPU PK
```

最成熟的是链 B。链 A 已真实到达 generic linalg。链 C 已在 softmax、RMSNorm 等少量 row-reduction artifact 上成立，但还没有扩展成自动、统一、全覆盖的产品主链。

### 1.3 多次尝试结果不同，当前如何处理？

当前只有两类 retry：

1. **网络 transport retry**：同一请求遇到 HTTP/网络问题时最多重发约 4 次。这不是多候选比较。
2. **语义 repair retry**：首次 LLM 回复失败后，最多再修复 3 次，即最多约 4 份语义回复。

当前选择规则非常简单：

```text
解析 JSON
  → 构造 dataclass
  → outputs 非空
  → 一次随机输入 Interpreter smoke
  → 输出 finite
  → 第一份通过者立即获胜
```

当前没有实现：

- 多个合法候选并行采样；
- 多数投票或 self-consistency；
- canonical graph fingerprint；
- 候选语义等价性比较；
- disagreement 指标；
- 基于 verifier、可 Lower 性或性能的排序；
- 保存全部候选、失败候选和候选 diff；
- 多模型 ensemble；
- 对独立重跑结果做稳定性审计。

因此，如果两个候选都合法但语义图不同，系统不会比较它们：**第一次通过轻量 smoke 的候选直接被接受，第二个候选根本不会生成。**

更重要的是，真实 source-vs-Intent 数值 mismatch、MLIR emitter 失败、C++ verifier 失败和 Lowering 失败，目前都不会自动回到 LLM repair loop。

---

## 2. 审计基线与当前项目状态

### 2.1 Git 与 Trellis 状态

审计开始时：

- 分支：`master`
- HEAD：`9a689f59fbca60f23365c2f3cfdedfdbf14509c2`
- 本地与 `origin/master` 对齐
- 工作树 clean
- Trellis developer：`kingdom`
- 当前任务：`.trellis/tasks/07-07-max-perf-systematic`
- 当前任务状态：`in_progress`

Git 历史和归档 Trellis 任务支持“此前 Claude 深度参与项目开发”这一判断。关键 live corpus 提交为：

```text
fb1d430 Live LLM extraction corpus: 15 high-perf-library kernels, extract->verify->lower
```

该提交记录中包含 Claude Opus 4.8 co-author 信息。近期多项性能与后端提交也延续了类似协作痕迹。

不过本报告的判断不依赖提交说明中的自述，而以当前 HEAD 的代码、artifact 回放和测试结果为准。

### 2.2 当前性能任务做到哪里

当前系统化性能任务规划了 N1–N8。根据当前提交和代码状态：

- N1–N6 已有对应实现或性能提交；
- N7 “autotuner everywhere”尚未形成完整调用与 cache 生命周期；
- N8 “schedule fork-collapse”尚未完成，各 specialized Pass 的 schedule/launch 决策仍有较多重复或硬编码；
- 最终统一性能报告还没有完全刷新和锁定。

所以后端是当前项目最强的子系统，但仍处于“系统性收尾和统一化”阶段，而不是已经完成的最终产品。

### 2.3 规范目标与当前实现的距离

项目 constitution 把 IntentIR 定义成 Tianchen AI 软件栈中的 Kernel Understanding/Extraction/Verification/Generation 规范，并强调：

```text
LLM proposes; verifier decides.
```

这一目标很清晰，但当前实现更接近：

```text
LLM proposes;
轻量 smoke、动态比较和 lowering 分别记录结果;
verified 字段尚未成为强制准入控制。
```

换句话说，“验证器裁决”目前主要是设计原则，不是已经贯穿所有执行入口的硬控制流。

---

## 3. 当前 Op 面到底有多完整

### 3.1 C++ Dialect 中的 Op

`include/Intent/IntentOps.td` 当前定义 27 个 Op，可按语义分成以下几类。

| 类别 | 主要 Op | 当前用途 |
|---|---|---|
| 基础逐元素 | `map_unary`、`map_binary`、`select`、`cast` | elementwise、比较、激活、类型转换 |
| 索引与生成 | `iota` | 索引生成、arange 类语义基础 |
| 归约与扫描 | `reduce`、`scan`、`arg_reduce` | row reduction、prefix scan、argmax/argmin |
| 张量代数 | `contraction` | matmul、dot、MMA 和 attention 子图 |
| 控制与区域 | `loop`、`yield` | 显式分块/循环式 Intent 表达 |
| 顺序与数据依赖 | `sort`、`gather`、`scatter` | 排序、间接访存、写冲突语义 |
| 动态压缩 | `nonzero`、`masked_select`、`masked_scatter` | 动态结果数量、mask compaction |
| 布局与变形 | `concat`、`pad`、`tile`、`flip`、`triu` | 布局拼接、padding、复制、翻转、三角 mask |
| 随机 | `rand` | RNG 语义 |
| 神经网络算子 | `pool`、`upsample`、`polar`、`conv` | 常见复合张量算子 |

`lib/Intent/IntentOps.cpp` 中为其中 26 个 Op 实现了自定义 verifier；`yield` 没有独立 verifier。

### 3.2 Lowering Pass 广度

`include/Intent/Conversion/Passes.td` 当前约有 22 个转换 Pass，包括：

- 通用 `Intent → Linalg`；
- row reduction；
- elementwise；
- gather/scatter；
- layout；
- RNG；
- arg-reduce；
- sort；
- pool/upsample/polar/conv；
- dynamic compact；
- contraction、MMA 和 split-K；
- FlashAttention forward/backward 等专用 GPU 路径。

从“有没有实际编译后端”看，答案是明确的：**有，而且后端不是纸面设计。**

### 3.3 为什么仍不能说 Op 已经完全成熟

一个 LLM-facing IR 是否成熟，不能只数 ODS 中有多少个 Op，还要检查下面八层是否一致：

```text
规范
  ↕
LLM Prompt vocabulary
  ↕
Python JSON/model
  ↕
Interpreter
  ↕
MLIR emitter
  ↕
ODS + C++ verifier
  ↕
generic lowering
  ↕
specialized lowering
```

当前这八层不是由同一个机器可读权威源生成，已经出现明确漂移。

### 3.4 代表性跨层不一致

| 语义/Op | 规范或 Prompt | Interpreter | Emitter | C++ ODS/Verifier | 结论 |
|---|---:|---:|---:|---:|---|
| `transpose` | 有 | 有 | 无 | 无独立 Op | 可通过提取 smoke，随后 emit 失败 |
| `rsqrt` | 有 | unary 表缺失直接项 | 可组合表达 | 有 | 模型若直接输出该 fn，可能被 smoke 拒绝 |
| `iota` | Prompt 缺失 | 有 | 有 | 有 | 后端有能力，但 LLM vocabulary 没暴露 |
| `const` | JSON 常用 | 有 | 转成 `arith.constant` | 无 Intent const | 可工作，但不是 JSON 与 Dialect 1:1 |
| `reshape/broadcast/slice` | 规范提及 | 局部或隐式 | 无统一闭环 | 无对应稳定 Op | 规范面大于落地面 |
| `segment_reduce` | 规范提及 | 无完整通路 | 无完整通路 | 无正式 Op | 尚未落地 |
| `cast` | 有 | 有 | 部分转换真实、部分 alias | 有 | dtype 闭环不完整 |

审计中构造了一个纯内存 `transpose` 候选：它通过当前 `_validate()`，但 `lower_to_linalg()` 在 emitter 阶段报：

```text
NotImplementedError: emit_mlir: unhandled body op transpose
```

这直接证明：

> **当前 extraction-valid 不等于 emitter-valid，也不等于 lowerable。**

### 3.5 Python model 不是严格 schema

`python/intentir/ir/model.py` 主要使用 dataclass 装载 JSON 字段。它不会形成严格、封闭、机器可验证的 schema，也不会全面拒绝：

- 未知 Op；
- 未知 dtype；
- 声明 output shape 与 body 实际结果 shape 不一致；
- 声明 output dtype 与实际 SSA dtype 不一致；
- Layer B obligation 无法解释或无法证明；
- 某些 SSA/type/shape 约束不闭合。

审计中构造了一个纯内存反例：

- input dtype 为不存在的 `"banana"`；
- 声明 output 为 `[999] i64`；
- body 实际产生 `[2,3] f32 abs(X)`。

结果是：

```text
_validate = PASS
lower_to_linalg = zero residual
```

这说明当前 `extracted=true` 或 `zero residual` 都不能单独证明 JSON 声明的 shape/type 正确。

### 3.6 C++ verifier 比 Python gate 强，但也不均匀

C++ 层已经有较多真实 verifier，这是项目的重要基础。但不同 Op 的约束强度并不一致。

例如某些 verifier 会检查：

- 函数枚举合法性；
- region/yield 结构；
- rank/轴范围；
- 结果是否必须为 `i1`；
- 某些 reduction 的结合性声明；
- 某些专用 Op 的静态 shape/layout 限制。

但部分基础 Op 仍主要检查枚举或少数属性，没有完整约束输入输出 shape、dtype、broadcast、轴归约后的结果类型等。

本轮还直接把三类畸形 MLIR 送入当前 `intent-opt`：

1. `tensor<2x3xf32>` 与 `tensor<4x3xf32>` 做所谓 broadcast add；
2. `f32` 与 `i32` operand 做普通 add，并声明 f32 result；
3. reduce axis 设置为 `99`，result shape 随意声明为 `tensor<7x9xf32>`。

这三类输入都返回 `rc=0`。这说明基础 verifier 目前更接近“枚举和局部结构检查”，还没有成为完整的 shape/type well-formedness gate。

因此 C++ verifier 是“已有真实基础”，但还不是完整的 IntentIR 语义证明器。

### 3.7 dtype 和 shape 是当前最明显的 Op 完整度短板

当前 emitter 默认 element type 仍明显偏向 `f32`。虽然输入 JSON 可以声明 `f16`、`i1`、`i64` 等类型，但类型传播和 cast lowering 尚未完全一致。

当前 `count_nonzero` 回放失败就是直接证据：

```text
f32 → i1 → i64 → reduce sum
```

其中 `f32 → i1` 生成了真实转换，但 `i1 → i64` 被当作 SSA alias，最终 reduce 的声明类型和实际 value 类型冲突。

因此下一阶段必须优先闭合：

- `f16/f32/bf16`；
- `i1/i8/i32/i64`；
- bool reduction；
- cast legality；
- input/output dtype；
- Interpreter dtype；
- emitter SSA 类型；
- ODS 类型约束；
- generic 和 specialized lowering 的类型转换。

### 3.8 `out` 还不是真正的 destination-passing

规范希望 JSON 中的 `out/in_place` 最终对应 Dialect 的 output buffer/`outs` operand，并显式表达 alias、ownership 和写入关系。

当前实现不是这样：

- 大多数 TableGen Op 仍是纯 SSA result，没有通用 `outs` operand；
- Python emitter 遇到 `op.out` 时，主要把 output 名称绑定到 result SSA；
- Interpreter 也主要把同一个 ndarray 绑定到另一个名称。

所以当前 `out` 更准确地说是“结果名称别名”，还不能表达真正的：

- in-place mutation；
- input/output alias contract；
- output buffer ownership；
- 通用 multi-output destination；
- read-after-write / write-after-write obligation；
- bufferization 前的显式 DPS 语义。

这不妨碍当前纯函数式 tensor 语义和部分 backend 实验，但如果 IntentIR 要成为 LLM-facing kernel contract，这一层必须正式定义。

### 3.9 `loop` 与动态 shape 仍是受限原型

当前 `intent.loop` 主要覆盖静态 scan recurrence：

- trip count 来自 scan input 的静态 extent；
- 依赖 `scan_axes` 和 tile；
- 没有 condition operand；
- 没有 while-before/after region；
- 没有数据依赖 termination、break 或 continue；
- C++ generic Lowering 要求静态 extent，并且当前要求 extent 能整除 tile。

因此它不能覆盖规范中更一般的“数据决定迭代次数/收敛条件”控制流。

此外，Interpreter 对最后一个短 tail slice 的容忍度高于当前 C++ loop Lowering，说明同一个 `loop` 在 reference semantics 和 Lowering legality 上仍有差异。

动态 compaction 也有类似问题：

- Interpreter 的 `nonzero` 返回实际 `[nnz, ndim]`；
- Interpreter 的 `masked_select` 返回实际 `[nnz]`；
- emitter 为了静态 MLIR 容量，通常把 leading dim 发射成输入元素总数的 upper bound；
- 有效长度再依赖 runtime count/host 侧解释。

这在工程上可以工作，但应该在 IR 中正式定义为：

```text
capacity tensor + runtime length
```

或真正的 dynamic tensor/descriptor。当前它还是实现约定，不是清晰的跨层语义契约。

### 3.10 Op 完整度分项判断

以下评分是基于当前代码和测试的工程判断，不是形式化度量。

| 维度 | 判断 | 约略成熟度 |
|---|---|---:|
| 研究 benchmark 的 Op 广度 | 已经较强 | 4/5 |
| specialized GPU 后端覆盖 | 有实质成果 | 4/5 |
| 规范、Prompt、Interpreter、Emitter、ODS 一致性 | 存在明显漂移 | 2.5/5 |
| dtype/shape 通用性 | 仍以固定/static/f32 场景为主 | 2/5 |
| LLM 可稳定生成性 | 有 corpus，但实际使用 Op 集中 | 2–2.5/5 |
| Layer B obligations/certificate | 规范多、实现少 | 1–1.5/5 |
| 自动 specialized Pass 映射 | 主要靠调用方指定 | 2/5 |

---

## 4. 真实 LLM 提取链的真实性与边界

### 4.1 真实 Provider 客户端确实存在

Provider 配置从 gitignored 的本地配置读取，代码支持：

- `base_url`；
- `api_key`；
- `model`；
- Responses wire API；
- reasoning effort。

`python/intentir/extraction/providers/client.py` 会：

1. 构造 `base_url + /responses`；
2. 发送包含 model、instructions、input、reasoning 的请求体；
3. 添加 Bearer Authorization；
4. 使用 `urllib.request.urlopen` 真正发送 HTTP 请求；
5. 对 transport error 做有界重试。

因此这不是 mock Provider。

安全边界说明：本次审计只确认配置存在和非敏感元数据，没有输出或复制 API key，也没有发起新的 live 请求。

### 4.2 artifact 数量与 provenance

当前共发现：

- 111 个 `intent.json`；
- 110 个 provenance 标记为 Provider/LLM 提取；
- 1 个 `layer_norm` 标记为 hand-authored。

在有 `attempts` 字段的 107 份 Provider provenance 中：

- 95 份一次通过；
- 10 份两次；
- 2 份四次；
- 另有 3 份 CUDA artifact 没有相同的 attempts 字段。

这足以支持“以前确实执行过真实 Provider 提取”，但 provenance 还不是完整法证链。

### 4.3 当前 provenance 能证明什么、不能证明什么

能够支持：

- 项目存在真实 Provider 调用实现；
- 仓库中保存了大量模型请求标签、attempts 和 repair 摘要；
- Git 历史中有明确的 live corpus 提交；
- 保存的 artifact 当前仍可被回放；
- 核心 15 个 artifact 当前仍能与真实 FlagGems 对比并进入 generic linalg。

不能严格证明：

- 每份 JSON 是服务端原始回复未经人工编辑的逐字保存；
- Provider 实际服务的模型一定等于客户端请求标签；
- 当前 source、prompt 和环境能逐字复现当时请求；
- 每一次失败候选是什么；
- 同一输入独立重跑时输出是否稳定。

原因包括：

- 没保存原始 Provider response；
- 没保存 response ID；
- 客户端记录的是请求配置中的 model，而不是服务端返回 model；
- 没保存 token usage、服务端时间戳和完整候选历史；
- 部分 source 指向仓库外绝对路径；
- 当前 prompt/source 重新计算的 hash 与已提交 provenance 不一致，说明环境或 prompt 已演进；
- artifact 重跑通常覆盖固定文件。

因此最准确的说法是：

> **仓库强力支持“此前通过真实 Provider 生成过这些 Intent JSON”，但还不能对每份 artifact 提供完全可复现、不可抵赖的原始模型调用证明。**

### 4.4 Prompt 中包含很强的语义提示

当前 `build_user_input()` 不只提供 source，还提供：

- kernel/function 名称；
- shape_hint；
- 很多情况下的归约轴；
- 很多情况下的完整数学公式。

例如：

- softmax hint 直接说明 row softmax 和 axis；
- GELU hint 给出完整 tanh approximation；
- RMSNorm hint 给出 `X * rsqrt(mean(X^2)+eps) * W`；
- matmul hint 直接说明 contraction；
- ELU、softplus、MSE、batch norm 等也提供接近答案的公式；
- arange hint 直接说明 `[0, 1, ..., END-1]`。

因此当前 corpus 证明的是：

> **LLM 能在完整 schema、算子名称、shape、归约信息和大量显式语义提示下，把真实源码转写为 IntentIR。**

它尚未充分证明：

> **模型仅依靠陌生、匿名、优化后的 kernel source，就能稳定恢复正确张量语义。**

建议未来必须拆成至少三组评测：

1. source + 必要 ABI/shape，不给语义公式；
2. source + 当前强语义 hint；
3. 只给 hint、不提供 source。

并加入匿名 kernel 名、unseen kernel 和重复采样，以量化“源码理解”本身的贡献。

---

## 5. 提取期验证器实际做了什么

### 5.1 当前 `_validate()` 的真实门槛

`extract_intent()` 中当前的候选 gate 主要是：

```text
从回复中截取 JSON
  → json.loads
  → IntentFunction.from_json
  → outputs 非空
  → 生成一次随机输入
  → Interpreter 执行不报错
  → 输出全部 finite
```

这不是严格的 schema/type/shape/obligation validator。

### 5.2 smoke 输入的局限

当前 smoke 大致采用：

- symbolic dim 默认替换为 6；
- 固定随机种子；
- 输入主要是标准正态 `float32`；
- 只检查执行和 finiteness。

这会遗漏：

- dtype-specific 语义；
- 大尺寸/边界尺寸；
- broadcast 与动态 shape；
- 稀疏/重复 index；
- NaN/Inf/符号零；
- 正负定义域；
- overflow/underflow；
- output 声明 shape/dtype；
- source semantic equivalence。

### 5.3 finiteness repair 已经诱导语义改变

仓库中有三个非常关键的 committed artifact：

#### `log`

smoke 因负数输入产生 non-finite 后，最终 artifact 实际表达：

```text
log(abs(X))
```

对负输入它与真实 `log(X)` 不等价。

#### `sqrt`

多次 non-finite repair 后，最终 artifact 实际表达：

```text
sqrt(max(X, 0))
```

对负输入，artifact 返回 0，而真实 sqrt 产生 NaN。

#### `pow`

最终 artifact 实际表达：

```text
pow(abs(A), B)
```

对 `A=-2, B=3`，artifact 得到 `+8`，真实结果是 `-8`。

需要公平说明：当前对应 harness 使用了正输入 domain，所以这些 artifact 在测试域上可以通过。

真正的问题是：

> **正输入前提没有可靠写进 Layer B/certificate。模型通过改变算子消除了 NaN，而不是保留原语义并声明 precondition。**

这说明当前 gate 更接近“在一组随机输入上能产生有限数”，而不是“忠实于源程序语义”。

### 5.4 Layer B 与正式 certificate 尚未落地

111 份 intent artifact 中，只有 2 份包含任何 `layer_b` 内容。

当前没有发现完整实现的：

- 机器可执行 obligation checker；
- PASS / FAIL / UNKNOWN 结果对象；
- FULL / PARTIAL / OUT-OF-SCOPE certificate；
- persisted certificate artifact；
- witness/claim checker；
- precondition solver；
- certificate 对 Lowering 的强制 gate。

`python/intentir/verify/` 当前主要是 Interpreter；`__init__.py` 中的描述不等于已经存在相应实现。

### 5.5 verified 当前是观测列，不是控制面

核心 corpus 流程在得到动态误差后设置 `verified`，但仍会继续 Lowering 并保存 artifact。

Universe 流程更明确地把 extraction、lowering 和 verification 作为三个独立阶段。

当前 bulk 记录中已经存在：

```text
lower = true
verify = false
```

的案例。

因此当前事实是：

> **项目有 verification 结果和统计，但“verifier decides whether lowering may proceed”还没有形成强制执行语义。**

---

## 6. 多次尝试、不同结果和 repair 的真实机制

### 6.1 网络重试

Provider client 对空回复、HTTP error、URL/timeout 等做有界 transport retry。

这类 retry 的特点是：

- payload 不变；
- 目的是恢复网络请求；
- 不比较语义结果；
- 不能算 self-consistency。

### 6.2 语义 repair

`extract_intent()` 默认首次调用后最多 3 次 repair，因此最多约 4 个 semantic attempts。

失败后主要把下面的内容反馈给模型：

```text
异常类型 + 异常文本
```

典型异常包括：

- JSON parse error；
- dataclass/字段错误；
- Interpreter exception；
- non-finite output。

随后再次发送原始 user prompt 加 validator diagnostic，并要求返回修正 JSON。

### 6.3 第一个通过者获胜

当前选择逻辑是：

```text
attempt 1 通过 → 立即返回
attempt 1 失败 → diagnostic → attempt 2
attempt 2 通过 → 立即返回
...
```

因此 attempts 统计表示“直到第一份 smoke-valid 候选的尝试次数”，不表示：

- 多候选一致性；
- 重跑稳定率；
- 候选之间的语义距离；
- verifier 置信度；
- 全输入域正确性。

### 6.4 真实数值 mismatch 和 Lowering 失败不会 repair

这是当前机制的核心断点。

`extract_intent()` 返回之后，caller 才执行：

- 与真实 FlagGems kernel 的数值比较；
- emit MLIR；
- generic Lowering；
- 某些后续 specialized backend。

所以以下失败不会回到当前 repair loop：

- source-vs-Intent mismatch；
- 具体 counterexample；
- output shape/dtype mismatch；
- emitter 不支持 Op；
- MLIR verifier diagnostic；
- generic Lowering failure；
- specialized Pass legality failure；
- cubin 编译失败；
- 性能明显劣化。

### 6.5 rerun 时没有候选历史治理

当前保存方式主要覆盖固定的：

```text
intent.json
provenance.json
```

没有系统保存：

- 每次 raw candidate；
- rejected candidate；
- old-vs-new diff；
- canonical semantic fingerprint；
- response ID；
- source snapshot；
- prompt snapshot；
- 验证 counterexample；
- candidate ranking reason。

所以当前面对 LLM 非确定性的真实策略是：

> **能复用旧 artifact 就不重跑；必须重跑时接受第一份通过轻量 gate 的候选；不同合法候选之间不比较。**

### 6.6 推荐的候选治理流程

建议把一次 extraction run 改成：

```text
生成 N 个候选
  → 严格 schema/type/shape 检查
  → canonicalize
  → graph fingerprint 去重
  → obligations 检查
  → source counterexample 测试
  → generic/specialized lowering feasibility
  → 语义等价分组
  → 成本/性能排序
  → 保存全部证据
  → 选中 candidate + disagreement 标记
```

如果多个候选都通过但不等价，应输出 `DISAGREEMENT` 或 `PARTIAL`，而不是静默选择第一份。

---

## 7. Lowering 到底做到什么地步

### 7.1 官方 LLM corpus：真实到 generic linalg

核心 corpus 的真实流程包括：

```text
live provider candidate
  → IntentFunction
  → real source comparison
  → emit Intent dialect
  → intent-opt -convert-intent-to-linalg
```

当前核心 15 个 committed Provider artifact 回放测试通过：

- 重新加载 JSON；
- 重新运行 Interpreter；
- 对比真实 FlagGems kernel；
- 重新执行 `-convert-intent-to-linalg`；
- 要求没有残余 `intent.*`。

这证明“真实历史 LLM artifact → 当前 generic Lowering”是成立的。

### 7.2 committed LLM artifact：部分真实到 CPU JIT

CPU backend 测试会加载 add、silu、softmax、matmul 等 Provider JSON，然后执行：

```text
Intent JSON
  → Intent dialect
  → Linalg
  → loops/LLVM
  → mlir-runner JIT
```

因此这部分不是只检查 MLIR 文本，而是真实执行。

### 7.3 committed LLM artifact：少量真实到 sm_120a cubin

`RowReductionKernel` 会读取 committed softmax/RMSNorm corpus artifact，再由调用方指定 row-reduction specialized Pass，继续生成 cubin 并运行 GPU PK。

本次审计实际确认 softmax LLM artifact 的 row-reduction GPU 路径通过。

此外，审计中还把 committed `add` 和 `softmax` artifact 手工串接到现有 GPU program runner：

- add：GPU 输出与参考一致，max absolute error 为 0；
- softmax：max absolute error 约 `4.47e-08`，max relative error 约 `2.43e-07`。

这说明现有组件在简单 `f32` 场景下确实能够连接到真实 cubin。

但这个连接尚未被封装成官方统一入口。

### 7.4 高性能路径多数从手写/固定 Intent IR 开始

当前最先进的：

- FlashAttention forward/backward；
- sort；
- matmul/MMA；
- split-K；
- scan；
- gather/scatter；
- dynamic compact；

大量使用：

- 手写 `.mlir` fixture；
- Python `_mlir()` 字符串构造器；
- 手工构图；
- 固定、缺少完整 live provenance 的 JSON fixture。

这证明 specialized backend 很真实、很强，但不能把这些 PK 全部算成“LLM 从原始 kernel 稳定提取出的结果”。

尤其旗舰 FlashAttention 路径的早期 Trellis 记录明确把关键 Intent JSON 描述为 hand-authored。

### 7.5 Flash 的 whole-function JSON → Dialect 还不是完整 1:1

当前 Flash recurrence 相关测试证明了重要能力：

- JSON 中的 loop body 能被 Interpreter 执行；
- loop body 能发射并 lower；
- host/Python 能驱动多 tile recurrence；
- GPU 结果能够与 reference 对比。

但 `emit_loop_module()` 当前主要发射 loop/carries，并没有把 loop 后完整 epilogue 一并变成同一个完整 Intent function。

对典型 Flash JSON，loop 后仍有：

```text
O = Oacc / l
```

而当前 loop module 路径主要返回 recurrence carries，例如 `m`、`l`、`Oacc`；最终归一化仍可能由 host/PyTorch 完成。

因此这些测试证明“recurrence body 和 backend 真实可用”，但还不能证明：

- 整个 Flash JSON 已 1:1 发射成单一完整 Intent function；
- loop + epilogue 已统一由 Dialect/Lowering 执行；
- 该高级 Flash JSON 来自 live LLM。

### 7.6 `gpu_pass_lower` 是分类，不等于当次真实 GPU Lowering

通用 `-convert-intent-to-linalg` 当前只为一部分 Op 注册 rewrite pattern，主要包括：

- `map_unary`；
- `map_binary`；
- `select`；
- `reduce`；
- `contraction`；
- `loop`；
- `cast`。

该 Pass 使用 greedy rewrite；它不是一个要求所有 `intent.*` 必须消失的完整 dialect conversion target。

Universe harness 对 iota、scan、gather/scatter、sort、arg-reduce、pool、upsample、conv、rand、layout、dynamic compaction 等 residual Op 使用白名单分类。如果 residual 名属于已知 specialized-Pass 类别，就可以记录为 `gpu_pass_lower`。

这应该解释为：

> 通用 Linalg 路径不能完全消除该 Op，但仓库中存在对应或相关 specialized backend 能力。

它不等同于：

> 该条 LLM artifact 在本次 Universe run 中已经真实调用 specialized Pass、生成 cubin 并执行。

因此 coverage 中的 “composes” 或 `gpu_pass_lower` 必须与真实 `specialized-GPU-lowered`、`cubin-executed` 分开统计。

### 7.7 没有通用 automatic specialized Pass dispatch

当前没有统一组件自动完成：

```text
从 Intent 图识别 kernel class
  → 选择 generic 或 specialized Pass
  → 选择 block/warp/stage
  → 选择 launch plan
  → 选择 autotune config
  → 编译和执行
```

例如 row reduction 需要调用者显式给出 `pass_name`。Universe harness 中的 `GPU_PASS_OPS` 主要用于分类，不是正式 dispatch。

所以当前准确结论是：

- 真实 LLM → generic Lowering：**已完成**；
- 保存的 LLM artifact → CPU JIT：**代表性案例已完成**；
- 保存的 LLM artifact → GPU cubin：**少量代表案例已完成**；
- 任意 source → live LLM → certificate → 自动 specialized Pass → cubin：**尚未完成**。

---

## 8. Corpus 与覆盖率应该怎样解读

### 8.1 核心 corpus

核心 manifest 记录 15/15：

- extracted；
- source dynamically verified；
- generic lowered。

当前回放仍通过，因此这是当前最可靠的 LLM corpus 证据。

但这 15 个案例大量使用强语义 hint，且主要覆盖常见 elementwise/reduction/contraction 组合。

### 8.2 pilot corpus

已提交 manifest 原先记录 22 total、20 ok、2 extraction failure。

当前回放发现 `count_nonzero` 已由原记录的 `ok` 变成 `lowering_fail`，所以不能继续无条件引用“20 个 pilot 当前全都 ok”。

### 8.3 bulk corpus

已提交记录大致为：

- 74 total；
- 72 extracted；
- 68 generic lowered；
- 4 classified as GPU-pass lowering；
- 68 verified；
- 64 ok；
- 2 extraction failure；
- 4 verify failure。

需要注意：

- bulk 没有在默认 Python 测试中逐项做完整当前 HEAD 回放；
- `gpu_pass_lower` 很多时候只是分类，不代表真正执行了 GPU Pass；
- manifest 可能因 emitter、依赖或 backend 演进而漂移。

### 8.4 Provider artifact 实际使用的 Op 很集中

对已保存 Provider JSON 的统计显示，主要使用：

| Op | 约使用次数 |
|---|---:|
| `map_binary` | 173 |
| `const` | 143 |
| `map_unary` | 62 |
| `reduce` | 30 |
| `select` | 13 |
| `contraction` | 12 |
| `cast` | 5 |
| `triu` | 2 |
| `scan` | 2 |
| `concat` | 1 |

这意味着真实 LLM corpus 主要实际验证了约 10 类 JSON body op。

高级 ODS Op，例如 sort、arg-reduce、gather/scatter、pool、upsample、conv、复杂 dynamic compact 和 attention specialized representation，其主要正确性证据仍来自手写 fixture 和 PK，而不是大量 LLM extraction。

### 8.5 `168/231 verified-e2e` 不能等同于 168 条完整 LLM 链

`corpus/COVERAGE.md` 当前给出：

- FlagGems including DSA：143 verified-e2e、45 extraction-only-proven、5 miss；
- Triton/reference：25 verified-e2e、10 extraction-only-proven、3 miss；
- 合计：168/231 verified-e2e、55 extraction-only、8 miss。

但 coverage generator 混合了：

- 核心/pilot/bulk LLM corpus；
- 手写 cluster proof-kernel；
- specialized backend 测试；
- 基于 Op class 的归类或外推。

因此 `168/231` 更准确的含义是“研究覆盖面”，不能解释为：

> 168 个独立源 kernel 全都经过 live LLM、正式 certificate、自动 GPU Lowering、cubin 执行和性能验证。

建议以后拆成以下互不混淆的计数：

1. `LLM-extracted`；
2. `schema-valid`；
3. `source-dynamically-verified`；
4. `generic-lowered`；
5. `specialized-GPU-lowered`；
6. `cubin-executed`；
7. `performance-measured`；
8. `hand-authored-only`；
9. `class-inferred-only`；
10. 完整 `source → live LLM → cubin`。

---

## 9. 当前测试结果与真实回归

### 9.1 C++/MLIR lit

执行：

```bash
rtk cmake --build build --target check-intent -j 4
```

结果：

```text
70 discovered
70 passed
100%
```

这说明当前 Dialect、parser/printer 和各类转换 Pass 的 lit 基线是稳定的。

### 9.2 完整 Python 测试

在显式关闭 live LLM 调用的默认模式下执行：

```bash
rtk proxy env -u IIR_LIVE_LLM PYTHONPATH=python pytest -q python/tests
```

结果：

```text
310 collected
304 passed
3 skipped
3 failed
60 warnings
```

3 个 skipped 都是需要显式设置 `IIR_LIVE_LLM=1` 的 live LLM 测试。因此默认完整测试回放 committed artifacts，不会重新调用 Provider，也不会测试跨多次 LLM 调用的稳定性。

### 9.3 三个失败如何定性

#### 两个 xformers/Triton 参考基线兼容问题

`python/tests/test_splitk_reduce_bwd_pk.py` 中两个失败来自当前 xformers/Triton baseline launcher：

```text
dynamic_func() got multiple values for argument BLOCK_SIZE
```

IntentIR 自己的 split-K backward 路径已经运行，失败发生在参考实现接口漂移。它不直接证明 IntentIR Lowering 错误，但意味着完整测试客观上不是全绿。

#### 一个真实 frontend/emitter regression

`python/tests/test_universe_pilot.py::test_saved_replay_reproduces_failure_class` 对应 `count_nonzero`。

提交 manifest 记录为 `ok`，当前 replay 变成 `lowering_fail`，错误核心为：

```text
use of value %is_nonzero_1 expects different type
```

原因是：

```text
f32 → i1   生成真实 cast
i1 → i64   被 emitter 当作 alias
reduce      声明/实际 SSA 类型不一致
```

这个失败非常重要，因为它同时证明：

- corpus manifest 会随代码演进漂移；
- output/type contract 没有完全闭合；
- “保存时 ok”不代表当前 HEAD 仍然 ok；
- coverage/report 必须基于当前 replay 生成。

### 9.4 本轮没有做 live LLM 重跑

本轮刻意没有设置 `IIR_LIVE_LLM=1`，原因是：

- 用户要求只读检查；
- live runner 可能产生外部费用；
- 当前脚本可能覆盖既有 artifact；
- 本轮目标是审计 Claude 留下的现状，而不是生成一批新候选改变证据现场。

所以本报告对“历史真实 LLM 调用”的判断来自当前 Provider 实现、provenance、Git 历史和 committed artifact 回放，而不是本轮重新调用 Provider。

---

## 10. 当前项目成熟度定性

### 10.1 可以安全对外声明的内容

以下表述有当前代码和测试支持：

- TianchenIR 已有真实、非 toy 的 Intent Dialect 和 specialized GPU backend。
- 当前 C++ Dialect 有 27 个 Op，转换层约有 22 个 Pass。
- 项目存在真实 HTTP LLM Provider 调用实现。
- 历史上确实生成并提交了 Provider/LLM Intent corpus。
- 核心 15 个 committed LLM artifact 当前仍能与真实 FlagGems 对比并 lower 到 linalg。
- 代表性 LLM artifact 能执行到 CPU LLVM JIT。
- softmax/RMSNorm 等少量 committed LLM artifact 能进入 specialized GPU Pass 并运行 cubin。
- 高性能 GPU 后端已有真实 correctness/performance PK。
- 项目当前是一个较强的 Kernel Understanding/Compilation 研究平台。

### 10.2 当前不应对外宣称的内容

以下表述会超过当前证据：

- “IntentIR Op 已经完整覆盖任意 GPU kernel。”
- “JSON schema、Interpreter、emitter、ODS 和所有 Lowering 已经严格一致。”
- “所有 168 个 verified-e2e 都来自独立 live LLM 提取。”
- “验证器已经强制决定 artifact 能否进入 Lowering。”
- “已经实现 FULL/PARTIAL/OOS certificate。”
- “多次 LLM 输出不同时会自动做一致性裁决。”
- “LLM 已经被证明能在不给语义 hint 的情况下仅从任意源码稳定理解 kernel。”
- “所有高级 FlashAttention/sort/matmul 性能路径都由 LLM 自动提取。”
- “当前完整测试全绿。”
- “已经有统一的一键 source-to-cubin 产品管线。”

### 10.3 分层成熟度

| 子系统 | 当前定性 |
|---|---|
| 项目定位/研究方向 | 清晰，Kernel Understanding 层的方向成立 |
| C++ Dialect / Op 骨架 | 较成熟的研究型实现 |
| generic Linalg Lowering | 对核心子集真实可用 |
| specialized sm_120a GPU backend | 当前最强子系统，有实质成果 |
| 真实 LLM Provider 与 corpus | 确实存在，属于研究原型闭环 |
| source-only 语义理解证据 | 不足，受强 hint 影响 |
| Python schema/type/shape gate | 较弱 |
| Layer B obligations | 很少实际产出，基本未验证 |
| certificate | 规范存在，执行实现基本未闭环 |
| 候选分歧治理 | 基本未实现 |
| 自动 specialized Pass 选择 | 未形成统一系统 |
| 全链 provenance/reproducibility | 不足 |
| 统一产品级 source-to-cubin | 尚未完成 |

### 10.4 一句话总定性

> **TianchenIR 已经是一个真实可编译、拥有较强 GPU 后端和真实 LLM corpus 的 Kernel Understanding/Compilation 研究平台；但当前仍是“LLM 提取原型、手写/固定 IntentIR 后端 PK、性能工程”三部分的拼接，尚未形成由严格证书门控、自动 Pass 选择和多候选一致性机制支撑的统一产品链。**

---

## 11. 建议的完善顺序

### P0：建立唯一正式端到端入口

目标应是：

```text
source snapshot
  → provider extraction
  → strict schema/type/shape
  → Layer B obligations
  → source consistency
  → certificate
  → pass selection
  → lowering
  → compile
  → execute
  → persisted evidence
```

不能再主要依赖多个测试脚本由人工知道“应该怎样串起来”。

验收标准：

- 一个命令或 API 能从 source 开始；
- 每阶段有结构化状态和 artifact；
- verification failure 默认 fail-close；
- 只有明确 override 才允许未验证候选进入后端实验；
- 最终证据包含 source、candidate、certificate、pass/config、binary 和执行结果。

### P0：建立唯一机器可读 Op schema

由一份权威 schema 自动生成或至少自动校验：

- Prompt vocabulary；
- Python parser/model；
- Interpreter dispatch；
- emitter dispatch；
- ODS Op 列表；
- C++ verifier coverage；
- generic Lowering coverage；
- specialized Pass coverage；
- obligation template；
- 文档中的 Op 表。

验收标准：

- Prompt 中不存在 emitter/ODS 不支持的 Op；
- ODS 新增 Op 时 CI 会发现 Interpreter/Prompt 缺失；
- 每个 Op 有明确 dtype、shape、broadcast、domain、side effect 和 Lowering contract；
- 自动生成一张跨层 coverage matrix。

### P0：修复 dtype/cast/output contract

首先修复 `count_nonzero`，随后系统覆盖：

- f16/f32/bf16；
- i1/i8/i32/i64；
- bool/int reduction；
- signed/unsigned cast；
- output shape/dtype 与 body SSA 的一致性；
- mixed precision；
- dynamic result count ABI。

验收标准：

- `"banana"` dtype 和 `[999] i64` 伪声明在 schema gate 被拒绝；
- `i1 → i64` 生成真实转换；
- Interpreter、emitter、ODS 和 Lowering 对同一 dtype case 一致；
- pilot manifest 可由当前 HEAD 全量回放重建。

### P0：把真实失败纳入 bounded repair

repair 不能只接收 JSON/Interpreter exception，还应接收：

- source-vs-Intent counterexample；
- output shape/dtype mismatch；
- obligation violation；
- MLIR verifier diagnostic；
- emitter unsupported op；
- generic/specialized Lowering diagnostic；
- 必要时的性能或成本反馈。

对于 `log/sqrt/pow`，系统应优先生成显式 precondition，而不是悄悄改写全域语义。

### P1：实现 candidate/difference management

每次 extraction 至少保存：

- run ID；
- source snapshot/hash；
- prompt snapshot/hash；
- provider/model/config；
- response ID；
- raw candidate；
- canonical candidate；
- graph fingerprint；
- schema/type result；
- obligations result；
- counterexample；
- Lowering result；
- performance/cost；
- 最终选择理由。

当多个候选均通过但不等价时，输出明确 `DISAGREEMENT`，不要静默 first-wins。

### P1：补 source-understanding 消融

至少建立：

- source + ABI/shape only；
- source + semantic hint；
- hint only；
- anonymized name；
- unseen kernel；
- 每例多次独立采样。

报告应分别给出 extraction、semantic verification、Lowering 和 candidate agreement 指标。

### P1：实现自动 Pass selection

把当前由调用方知道的经验结构化为：

```text
Intent graph
  → legality/capability analysis
  → generic vs specialized choice
  → schedule/autotune config
  → launch plan
```

并让 certificate、shape、dtype、architecture 和性能 profile 共同参与选择。

### P1：重做覆盖率口径和报告生成

覆盖率必须由当前 HEAD 回放证据生成，避免混合：

- LLM extraction；
- 手写 fixture；
- specialized PK；
- class inference。

每行应能追溯到 artifact、测试、commit、硬件/软件环境和最近 replay 时间。

### P2：完成当前 N7/N8 性能统一工作

- 让 autotune config 真正接入编译和 launch；
- 建立 cache lifecycle；
- 收敛 duplicated schedule/launch logic；
- 逐步退休 regex fallback；
- 重跑并统一刷新性能 JSON 与 Markdown；
- 把 headline 性能与 CI guard 的口径明确区分。

---

## 12. 建议的项目阶段定义

为了避免以后“已经完成”这个词含义混乱，建议采用以下五级定义。

| Level | 定义 | 当前状态 |
|---|---|---|
| L1 | 有 Op/JSON 设计和手写样例 | 已超过 |
| L2 | 有 Interpreter、emitter 和 generic Lowering | 已超过 |
| L3 | 真实 LLM 提取、动态 source 对比、committed corpus 回放 | 已达到，但验证强度有限 |
| L4 | strict schema + obligations + certificate gate + candidate governance + 自动 Pass selection | 未达到 |
| L5 | 大规模 source-only 泛化、全链可复现 provenance、稳定自动 source-to-cubin、性能闭环 | 未达到 |

如果只看 specialized sm_120a 后端，可认为接近独立的 L4 研究后端；如果看整个“LLM Kernel Understanding → certified lowering”产品链，当前更接近 L3。

---

## 13. 给项目负责人的直接回答

### “我们的 Op 足够完善了吗？”

**对继续做当前研究基本够；对冻结成稳定 LLM 规范还不够。**

当前最该做的不是再加十几个 Op，而是把已有核心 Op 的：

```text
schema → prompt → interpreter → emitter → ODS → verifier → lowering → certificate
```

闭合成同一套权威语义。

### “做到了什么地步？”

已经做到：

- 真实 Intent Dialect；
- 真实 C++ Lowering Pass；
- 真实 CPU JIT；
- 真实 sm_120a cubin；
- 大量 GPU PK；
- 真实 Provider 调用实现；
- 110 份标记为 Provider 提取的 artifact；
- 核心 committed LLM corpus 当前可回放到 real-kernel comparison 和 generic linalg；
- 少量 LLM artifact 可进入 specialized GPU cubin。

尚未做到：

- 严格 schema/type/shape；
- obligations/certificate；
- verifier 强制 gate；
- 多候选分歧处理；
- source mismatch/lowering failure 自动 repair；
- 自动 specialized Pass selection；
- 大规模 source-only 泛化证据；
- 统一的一键 source-to-cubin 产品链。

### “是真实 LLM 提取进行 Lower 吗？”

**是，但主 corpus 统一保证的是 LLM artifact 到 generic linalg；高级 GPU 性能主链多数仍从手写或固定 Intent IR 开始。少量 row-reduction LLM artifact 已经真实走到 cubin。**

### “多次尝试如果不同如何处理？”

**目前没有真正处理不同合法候选。网络失败会重发；候选失败会把异常字符串交给模型修复；第一份能解析、能 smoke 且 finite 的候选直接获胜。不会投票、不会比较等价性、不会保留所有候选，也不会把真实 source mismatch 或 Lowering failure 自动反馈给模型。**

### “现在能否称为 LLM 提议、验证器裁决？”

**作为设计目标可以；作为当前强制执行事实不可以。当前 verified 更像观测字段，未验证或验证失败的候选仍可能继续 Lowering。**

---

## 14. 关键证据索引

以下文件是本报告的主要证据入口。

### 项目定位与任务

- `.trellis/spec/project/constitution.md`
- `.trellis/spec/extraction/llm-extraction.md`
- `.trellis/spec/verification/validation.md`
- `.trellis/tasks/07-07-max-perf-systematic/task.json`
- `.trellis/tasks/07-07-max-perf-systematic/prd.md`
- `.trellis/tasks/archive/2026-06/06-26-llm-extraction-corpus/task.json`

### Dialect 与 Lowering

- `include/Intent/IntentOps.td`
- `lib/Intent/IntentOps.cpp`
- `include/Intent/Conversion/Passes.td`
- `lib/Intent/Conversion/ConvertIntentToLinalg.cpp`
- `lib/Intent/Conversion/Schedule.h`

### LLM 提取

- `python/intentir/extraction/providers/provider.py`
- `python/intentir/extraction/providers/client.py`
- `python/intentir/extraction/lifters/extract.py`
- `python/intentir/extraction/lifters/prompt.py`
- `python/intentir/ir/model.py`

### Interpreter 与 emitter

- `python/intentir/verify/interpret.py`
- `python/intentir/frontend/emit_mlir.py`
- `python/intentir/verify/__init__.py`

### Corpus 与覆盖率

- `python/intentir/extraction/data/corpus.py`
- `python/intentir/extraction/data/corpus_specs.py`
- `python/intentir/extraction/data/universe_specs.py`
- `python/intentir/extraction/data/run_universe.py`
- `corpus/kernels/manifest.json`
- `corpus/pilot/manifest.json`
- `corpus/bulk/REPORT.md`
- `corpus/COVERAGE.md`
- `python/intentir/extraction/data/gen_coverage.py`

### 代表性问题 artifact

- `corpus/kernels/log/intent.json`
- `corpus/kernels/log/provenance.json`
- `corpus/bulk/sqrt/intent.json`
- `corpus/bulk/sqrt/provenance.json`
- `corpus/bulk/pow/intent.json`
- `corpus/bulk/pow/provenance.json`
- `corpus/pilot/count_nonzero/intent.json`
- `corpus/pilot/manifest.json`
- `corpus/pilot/arange/intent.json`

### 后端真实执行

- `python/intentir/backend/cpu.py`
- `python/intentir/backend/autotune.py`
- `python/intentir/backend/launch_plan.py`
- `python/intentir/benchmarks/row_reduction.py`
- `python/tests/test_cpu_backend.py`
- `python/tests/test_row_reduction_pk.py`
- `python/tests/test_corpus.py`
- `python/tests/test_universe_pilot.py`
- `python/tests/test_splitk_reduce_bwd_pk.py`

---

## 15. 审计最终判断

当前项目的真实价值很明确：它已经把“LLM-facing kernel semantics”和“真实高性能编译后端”两个很难的方向都做出了实物，绝不是只有文档或演示。

但两者之间目前仍有一段关键断层：

```text
强提示下的 LLM 转写
  → 轻量 smoke
  → 动态结果记录
  → generic lowering

与

手写/固定 Intent IR
  → 专用高性能 GPU Pass
  → cubin/PK
```

还没有被 strict schema、obligations、certificate、候选治理和自动 Pass selection 统一为同一条可信主链。

因此，最务实的下一阶段不是继续扩大宣传口径，也不是继续单纯增加 Op，而是完成“语义闭合”：

1. 统一 Op 权威源；
2. 修复 dtype/shape/cast；
3. 实现 certificate gate；
4. 把真实 mismatch 和 Lowering failure 纳入 repair；
5. 实现候选分歧治理；
6. 自动选择 specialized Pass；
7. 用 source-only/hint ablation 证明真正的 Kernel Understanding；
8. 让所有覆盖率和性能数字都能从当前 HEAD 一键回放生成。

做到这些之后，TianchenIR 才能从“很强的研究平台”跨到“可信的 Tianchen AI Kernel Understanding 基础设施”。
