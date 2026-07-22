# 科研定位与护城河

## 我们解决的问题(与传统编译器相反的方向)

传统张量编译器(Halide、TVM、Ansor、MLIR linalg)在**已有显式高层意图**时最有效——它们做的是"高层 → 优化实现"的 **lowering**。但真实世界里要复用的工件,往往**已经是**一个 schedule 特化死的 Triton/TileLang/CUDA kernel。

IntentIR 做的是**相反方向**:从优化死的实现里**反向恢复**出意图。这不是反编译(decompilation)——反编译恢复控制流 / 可读代码;我们恢复的是**张量级语义 + 使其可移植所需的约束**,并带可审计的证书。

## N×M → N+M

没有共享语义空间，支持 N 个源前端 × M 个目标后端容易退化为 N×M 套翻译。
IntentIR 让每个前端独立提升为 `J_i=(A_i,B_i)`，再由同一种中立 A+B+ABI backend
合同消费，接口变成 **N 个前端 adapter + M 个 backend consumer**。

这里的“统一”不要求 Triton、TileLang、CUDA 对同一个算子生成字节相同的 J；要求的
是它们落入同一语义空间，且 backend 不需要 frontend-specific source branch。

## 护城河:IntentIR = 给 LLM 用的 kernel 理解规范

本项目的差异化不在"又一个 IR",而在:**定义一套为 LLM 设计的、人可读、可验证、可 lower 的原子语义规范,让 LLM 用这套规范去理解 / 提取任意 kernel。**

- 对 LLM 友好:JSON 人可读、原子粒度适中(不爆炸长图、也不吞约束)、可自报 witness 供验证核对。
- 对验证友好：A/B 字段语义明确，可做 source-blind replay、结构闭合与动态比较。
- 对生态友好:以真正的 MLIR dialect 形态存在,能 lower 进 MLIR 已有构造(linalg/vector/scf/...),而不是重造轮子。

## LLM 的角色边界(不可越界)

- LLM **是**:受约束的语义提案者 + 原子化分解者(把纠缠 kernel 理解成三层原子表示)。
- LLM **不是**：正确性裁判，也不是 target backend selector；Gate 与独立 backend
  conformance 分别裁决 Lift 和 lowering。
- 论文的科研主张围绕"**LLM 提取 + 验证**得到可信意图表示",不是"LLM 直接写 kernel"。

## 与"另一条线"的关系

`TianchenForge` / `OptGuide` / `org` / `provider29` 是另一条科研思路(结构化优化资产引导的编译器搜索),**思想与本项目无关,不纳入**。两者唯一的交集是:本项目复用它整理过的高性能前端输入 kernel(见 [../extraction/frontend-inputs.md](../extraction/frontend-inputs.md))。
