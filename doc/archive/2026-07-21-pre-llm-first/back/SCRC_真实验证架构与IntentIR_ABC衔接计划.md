# SCRC 真实验证架构与 IntentIR A/B/C 衔接计划（归档）

> 归档状态：本文的两门架构与 A/B/C 归因原则已经吸收到当前 Gate1/Gate2 spec、
> `doc/SCRC_Gate1_规则库与数百Kernel扩展计划.md` 和
> `doc/SCRC_v8后续实施与规模验证流程.md`，不再作为独立活动路线图。v8 中 π 由
> `DeterministicCompose(Canon(J), Interface, C, RuleRegistry)` 自动生成；本文出现的
> “J+π”应按“J/interface/C/规则库派生的完整 π”理解。旧解释器、手写 probe 和
> strict-v5-v7 报告已经归档，不计入当前规模验证结果。当前分阶段流程和完成定义见
> `doc/SCRC_v8后续实施与规模验证流程.md`。

## 1. 正确的问题不是“op 是否 verify-ready”

IntentIR/JSON 本身就是被 Gate1 和 Gate2 验证的对象，不需要再给每个 op 建一套
“verify-ready”资格体系。

项目要回答的是：

1. 提升官从真实源码提取的 J 是否能用现有 A/B/C 表达；
2. J+π 是否足以让源码隔离的重写员重构同一编译器可见结构；
3. 同一 J 经 production lowering 后，是否与源和重放同行为。

对应关系：

```text
A：计算图、数据流、归约、递推语义
B：mask、fill、index、output mapping、effect 等可移植正确性结构
C：tile/warps/stages 等非绑定调度信息

Gate1：验证 J+π 在同 C 下能否结构重放闭合
Gate2：验证 source / replay / J-lowering 是否在测试域同行为
```

## 2. 项目当前真实状态

### 2.1 LLM corpus 与手写路径必须区分

仓库当前共有 111 份 `provenance.json`：

- 110 份标记为 provider/LLM extraction；
- 1 份 `corpus/kernels/layer_norm/provenance.json` 明确标记 hand-authored。

因此不能说“现有 Intent 全是手写”。但用户指出的核心问题同样成立：高级 GPU
能力和性能 PK 中，大量入口是手写 `.mlir`、固定 Intent DAG 或专用 emitter。
例如：

- `python/tests/test_pool_interp_pk.py`；
- `python/tests/test_layout_pk.py`；
- `python/tests/test_gather_scatter_pk.py`；
- `python/tests/test_rng_pk.py`；
- `python/tests/test_dynamic_shape_pk.py`；
- `python/tests/test_elementwise_pk.py`；
- `python/intentir/emitters/`；
- 多个 `test/lower-*-to-gpu.mlir`。

这些可以证明某个 dialect op/pass/PK 能 lower 和执行，但不能证明：

```text
真实源码 → LLM 提取 J/π/C → Gate1 → Gate2
```

### 2.2 现有 production lowering

仓库已有两条真实执行后端：

- GPU：Intent MLIR → GPU/NVVM/PTX/cubin；
- CPU：Intent MLIR → linalg/loops → LLVM → mlir-runner。

Gate2 初版选择 CPU 腿，因为它直接复用通用 `convert-intent-to-linalg`，适合验证
J 的计算结果，不需要再造 JSON evaluator。

### 2.3 A/B/C 的现有范围

项目规范已经计划并实现了多类 A/B/C 结构，具体依据是：

- `.trellis/spec/ir/layers-abc.md`；
- `.trellis/spec/ir/op-signatures.md`；
- `include/Intent/IntentOps.td`；
- `python/intentir/frontend/emit_mlir.py`；
- `lib/Intent/Conversion/`。

后续不凭想象增加 op。只有真实 LLM 提取 Kernel 暴露以下情况时，才提出 A/B/C
修改报告：

1. 源语义无法用现有 A 表达；
2. 影响正确性的 mask/index/effect 无法放入现有 B；
3. 源实现的重要调度知识无法放入现有 C；
4. 现有 J 能表达，但 production lowering 尚未覆盖。

前三类才是 IntentIR/A/B/C 缺口；第四类是 lowering 缺口，不能反过来乱改 IR。

## 3. 新的唯一正式流水线

```text
真实 Triton/TileLang/CUDA Kernel
  │
  ├─ 完整 Lift（可看源码）
  │     ├─ source-aware LLM：J / C / 接口 / 定位表
  │     ├─ deterministic finalizer：π / provenance
  │     └─ 返回完整 J / π / C artifact
  │
  ├─ 源前端 compile-only → source fingerprint
  │
  └─ Gate1
        ├─ π 计划
        ├─ 最小规则卡检索
        ├─ 重写员 LLM（看不到源码）
        ├─ C 后注入
        ├─ replay compile-only
        └─ source/replay fingerprint 裁决
              │
              └─ REPLAY_CLOSED → Gate2
                    ├─ source 实跑
                    ├─ replay 实跑
                    └─ J production lowering 实跑
```

旧解释器、旧契约等级、手写 J 模式都不在这条正式流水线中。

## 4. 如何处理 IntentIR 不足

不能先列一张想象中的“复杂特性清单”然后扩 IR。正确顺序是：

1. 从项目规划的真实前端输入中选 Kernel；
2. 真实调用完整 Lift：LLM 提取 J/C，Lift 内确定性生成 π；
3. 跑 Gate1；
4. 闭合后跑 Gate2；
5. 对具体失败做三方归属；
6. 只有确认是 A/B/C 无法表达时，提交扩展报告，说明：
   - 哪个真实 Kernel；
   - 哪段源语义无法表达；
   - 为什么现有组合不够；
   - 建议落在 A、B 还是 C；
   - 对 Gate1 规则卡、TTIR 指纹、Gate2 lowering 的影响；
7. 用户批准后再改 schema/op。

这样既不会因为当前五核太简单而假设 IntentIR 已经完整，也不会在没有真实反例
时加入一堆计划外特性。

## 5. 实施顺序

### 阶段一：Triton 真实 LLM 样本

- 从现有真实 source/provenance 可追溯 Kernel 选 20～40 个；
- 全部重新通过 `extract_scrc_artifact` 在 Lift 内生成完整 J/π/C；
- 不复用手写 J 或手写 `.mlir` 作为提升结果；
- 跑 Gate1，按真实失败补规则卡或规范化规则。

### 阶段二：Gate2 扩到同一批样本

- Gate1 closed artifact 才进正式 Gate2；
- source/replay 分别实跑；
- J 走现有 production lowering；
- lower 不支持时记录具体 op/结构，不回退解释器。

### 阶段三：按 SCRC 文档推进复杂结构

只处理现有 A/B/C 与输入规划已经覆盖的结构，如 structured loop/scan、已有
gather/scatter/sort/RNG/attention 表达；每一项都从真实 Kernel 提取失败或
Gate1/Gate2 失败出发，不以手写 PK 成功替代 extraction 证据。

### 阶段四：数百 Kernel

- 批量运行真实 LLM extraction；
- Gate1 relift-only 外循环；
- Gate2 边界输入；
- 失败按 J/π/replay/lowering/A-B-C 缺口归档；
- 规则库继续由真实 diff 增长。

该阶段沿用 SCRC 方案已有指标，不新增额外“模式”或人为资格体系。
