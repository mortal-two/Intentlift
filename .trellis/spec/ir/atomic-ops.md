# 张量级原子算子集

## 粒度原则

原子算子是**张量级结构化原语**（linalg 风格），不是标量指令，也不是库级大算子。

- 不能把 softmax/attention 当黑盒，否则 logical index、mask、reduction 与数值合同不可见；
- 也不能把整个图摊成标量 add/mul，否则 LLM 生成、审计和验证失控；
- 正确粒度是 map/reduce/contraction/scan/view/index/effect 等有限结构化原语。

## 新增 Op 的完成定义

新增一个 IntentIR op 不能只改 TableGen。至少同时完成：

1. **A 值语义**：输入、输出、shape/dtype、logical axes/index 与 corner-case policy；
2. **B 合同**：bounds、mask/fill、destination/alias/effect、conflict/order 中适用项；
3. **C 边界**：哪些 source realization 只是可替换 schedule/strategy；
4. **JSON/Python model**：strict schema、round-trip 与错误拒绝；
5. **Intent MLIR**：TableGen、C++ verifier、assembly/type contract；
6. **Frontend**：shape/dtype inference 与确定性 MLIR emission；
7. **Reference/Gate 语义**：可执行 reference 或适用的结构 fingerprint；
8. **Backend 状态**：generic/specialized `consumed / checked / unsupported`，不能因
   暂无 lowering 而静默留下 residual op；
9. **测试**：JSON round-trip、verifier negative、lit lowering，以及适用的设备执行／
   conformance test。

Kernel-name recognizer、只对一个样本成立的手写 MLIR、解释器 fallback 都不算新增 op
完成。

## 语义家族

### Elementwise

- `map_unary`：exp/log/sqrt/abs/neg/recip/sigmoid/tanh/erf 等；
- `map_binary`：add/sub/mul/div/min/max/pow/cmp 等，logical broadcast 属 A；
- `select`、`cast`。

### Reduction / scan / ordering

- `reduce`：sum/max/min/prod/any/all 等；
- `arg_reduce`：argmax/argmin，tie policy 必须显式；
- `scan`：cumsum/cummax/cummin 或其他受控 recurrence；
- `sort`：排序/top-k 的 value/index 与 stability/tie policy 必须定义。

Reduction 的 global axes、result shape、accumulator/result dtype 属 A；identity、masked
contribution、reassociation/determinism 属 B；tree/tile/warp 属 C。

### Contraction / convolution

- `contraction`：einsum-style logical contraction；
- `conv`：window/stride/dilation/padding/group 的逻辑值语义。

MMA、tensor-core tile、shared staging 与 pipeline 都属于 C；backend 无对应 target
能力时必须显式 unsupported 或走正确 generic path。

### Logical view / layout

- `transpose`、`concat`、`pad`、`tile`、`flip`、`triu`；
- reshape/slice/broadcast/reverse 等必须由显式 A op 或等价 typed logical map 表达。

logical permutation 属 A；physical memory layout、local rank insertion 与 address staging
属 C/witness。

### Indirect indexing / compaction

- `iota`、`gather`、`scatter`；
- `nonzero`、`masked_select`、`masked_scatter`。

index tensor 与 logical mapping 属 A；OOB、duplicate/conflict、atomic、write coverage 属
B；coalescing/sorting/bucketing 属 C。

### RNG / window / structured values

- `rand`：algorithm/seed/counter 与 statistical contract 必须明确；
- `pool`、`upsample`：window/coordinate/padding/value policy；
- `polar`：多结果 dtype 与 output mapping。

### Structured control

- `loop` + `yield`：semantic recurrence 与 carry；
- schedule-only loop 必须 re-roll 到 tensor op，tile/unroll 留在 C；
- branch/while 等未有一等、闭合 region contract 时 fail closed。

## 实现清单的权威来源

当前 dialect 的确切 op 名与字段以
`include/Intent/IntentOps.td` 为权威来源。本文件定义家族与完成标准，不复制一个容易
漂移的“当前数量”作为长期真值。任何新增 TableGen op 都必须同步本文件所属家族或说明
为何需要新的语义家族。

## 数值与并发合同

浮点 reduction/atomic 不可仅写一个模糊的 `associative` 标签。至少要区分：

- 数学 reducer 与 identity；
- masked contribution；
- reassociation 是否允许；
- deterministic 是否要求；
- tolerance/NaN/tie/empty-domain policy。

这些是 A/B 合同；backend realization 与 schedule 在 C 中选择。

## 不做

- 不引入语义未定义、不可执行、不可验证的占位 op；
- 不用库级 named op 隐藏内部 A/B；未来若引入，必须给出固定原子展开等价定义；
- 不为某个 backend 暂时缺能力而扭曲 portable op 语义；
- 不把 source-local pointer arithmetic 或 Kernel 名写入 portable op schema。
