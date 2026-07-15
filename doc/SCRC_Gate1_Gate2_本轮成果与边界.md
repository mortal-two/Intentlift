# SCRC Gate1 / Gate2：本轮成果、两道门禁关系与原型边界

## 0. 当前整合状态

| 部分 | 当前状态 | 已有真实证据 | 主要剩余问题 |
|---|---|---|---|
| SCRC Lift | 已可执行 | `gpt-5.6-sol / xhigh` 能从真实 Triton 源码生成 J/π/C/接口/定位表 | 复杂 Kernel 仍会暴露 IntentIR 表达或规则卡覆盖缺口 |
| Gate1 内层小修 | 已可执行 | 源码隔离 replay、有界 repair、真实 Triton 编译与 TTIR 硬比较已跑通 | 需继续通过真实 diff 扩充可枚举规则，不能用模糊规范化吞掉差异 |
| Gate1 外层重提 | 已打通首个真实闭环 | `fg_ones` 的错误 π 经一次重提从 `0.558333` 修正到 normal `1.0` | 目前只有一个真实重提样本，还需在更复杂语义缺失上验证稳定性 |
| masked-J 终检 | 已按方案实现并 fail-closed | 五核与首批十一个 FlagGems 真实样本都有报告 | 低熵 Kernel 和可识别 π 拓扑会产生大量“Gate1 拒绝、离线 Gate2 通过”样本；同一 `fg_vector_l1` artifact 的两次独立 masked-J 还得到 `0.906140/1.0`，说明单次比值也含模型随机性；这不等于 Gate1 判错，但说明现有单次比值不能区分真泄漏、模型先验、机制拓扑可识别性与采样波动 |
| Gate2 三方执行 | 已是真实 Lower2C 原型 | source/replay 在 GPU 运行，J 经 production IntentIR→MLIR→LLVM CPU 链运行；已覆盖边界 shape、对抗值和多 seed | 当前 PASS 是 rejected-sample 离线对照或固定 J 回归，正式 Gate1→Gate2 正样本仍为 0；Lower2C 还未覆盖全部 A/B/C 特性，mixed-f16/f32 仍不是 dtype-faithful lowering |
| 规模化 | 已启动 | 24 个真实 Kernel 队列中 11 个已完成真实 Lift/normal/masked/离线 Gate2 | 剩余 13 个 queued，离数百 Kernel 的覆盖、缓存和失败分诊目标仍有距离 |

## 1. 这轮到底做了什么

这轮不是继续修补旧 `verify`，而是把项目验证主线改成了 SCRC 文档规定的
两个过程：

```text
真实 Kernel
  → LLM 提升官产出 J / π / C / 接口 / 定位表
  → Gate1：源码隔离重放 + source/replay 双 TTIR 结构闭合
  → Gate2：source / replay / production lowering 三方实际执行
```

具体落地如下。

### 1.1 旧实现归档

- 旧 NumPy JSON 解释器移动到
  `python/intentir/verify/back/legacy_v0/interpret.py`；
- `intentir.verify.interpret` 只保留兼容转发，已有调用仍能执行；
- 新 Gate1、Gate2 和新的 SCRC extraction 均不导入旧解释器；
- 旧 schema/obligation、FULL/PARTIAL/OOS、L0–L9 验证规范移动到
  `.trellis/spec/verification/back/legacy_v0/`；
- 当前 `verify` 规范只保留 Gate1 和 Gate2。

### 1.2 真实 LLM 提升入口

新增 `intentir.extraction.lifters.scrc.extract_scrc_artifact`。它把真实源码交给
提升官 LLM，要求直接产出完整的：

- J：IntentIR A+B；
- π：规则卡 id + J/B 引用；
- C：槽和值；
- 接口签名；
- 定位表。

该入口只解析 artifact，不用解释器“先验算一遍”来决定正确性。正确性由
后续 Gate1/Gate2 裁决。模型由现有 provider 配置选择，不在代码中绑定 5.5。

当时也做了真实 provider 审计，历史结果必须如实保留：

- 当时使用了不完整模型名 `gpt-5.6`，因此返回 `unknown provider for model`；
- 回退到配置中的 `gpt-5.5` 后，add 的完整 artifact 在有界修复内仍未解析闭合；
- 实际暴露的都是 wire/schema 漂移：`fill→fill_value`、`node_ids` 列表、以及
  access 字段被重复塞入 `effects`；
- 代码已按这些真实失败收紧 prompt，并只兼容这几个已观察到的序列化形态，
  但本轮没有继续无界调用来制造一个“成功样例”。

这段结论只描述当时那次调用。2026-07-14 后续已改用正确名称
`gpt-5.6-sol` 与 `xhigh`，并通过 wire schema + 可信 compiler view 打通真实
add 的 Lift、正常 replay 和三方执行。2026-07-15 补齐方案要求的不透明
ABI/C 消融后，最新 add 的 masked replay 也完全闭合，因 `L=1.0` 被 Gate1
正确拒绝；此前 Gate2 PASS 仍是 lowering 工程证据，但不再是当前 Gate1 接受
证据。详见 `test/scrc_live_add_report.md` 与
`doc/SCRC_Gate1_MaskedJ_低熵Kernel阻塞报告_2026-07-15.md`。

2026-07-15 同一入口又推进了 row-sum 和 softmax。两者的 normal
source-blind replay 都已达到 TTIR 硬闭合；softmax 还用真实 diff 补了
全局/局部归约轴契约、隐式 broadcast 和 staged pointer tile 规则。但
row-sum 的 masked score 为 `0.778676`，softmax 的 masked replay 又完整
闭合为 `1.0`，所以两者最终仍是 `LEAKAGE_REJECTED`。softmax 的
production lower2c 三方预检已 PASS，但正式 live Gate2 按门禁顺序没有
运行。详见 `test/scrc_live_softmax_report.md`。

RMSNorm 随后也沿用同一入口完成真实 Lift 和 normal replay。本轮补齐
runtime `eps` 到 J `shape=[]` 输入的 `tensor_ref` 连接、`mean` 的语义
除数、`sqrt+recip` 规范形态，并用真实失败收紧 staged pointer card 的
选择说明。normal replay 首候选 score `1.0`，masked replay 也为 `1.0`，
最终仍是 `LEAKAGE_REJECTED`。对这份被拒绝的真实 live artifact 按方案
9.1 离线执行 Gate2，三方 PASS，但不改变门禁结论。详见
`test/scrc_live_rmsnorm_report.md`。

matmul 最后沿同一入口完成五核探针。最终 J 只包含一个
`contraction("mk,kn->mn")`，源码 K-tile loop 被正确还原为 contraction 的
Layer-C/前端实现，而不是 J `loop/reduce`。本轮还根据真实失败补全 staged
二维 pointer association、向量 mask conjunction、K extent/tile/result dtype
规则，并把 source n-gram 扫描纠正为只约束 π，不再误伤依法保存源码边界的
J.B。normal replay 首候选 score `1.0`；masked replay 虽未硬闭合，仍以
`0.804518 > 0.5` 被 `LEAKAGE_REJECTED`。真实 live 被拒绝样本的离线 Gate2
三方 PASS。详见 `test/scrc_live_matmul_report.md`。

### 1.3 Gate1 收回 SCRC 主线

保留了方案要求的硬项：

- 重放能否编译；
- 接口签名；
- TTIR 算子多重集与数据流；
- 归约；
- mask/fill/store/effect；
- π 静态泄漏与 masked-J 消融；
- 同一 C 下比较。

移除了不属于 SCRC 语义裁决的硬条件：

- `do_not_specialize` 不再作为接口硬相等；
- compiler/target 的整份 metadata 不再逐字段相等后才能比较；
- source 不再因为调用者没打 `trusted=True` 标签而直接拒绝；
- 默认不再要求两个独立重写候选连续闭合，一次硬闭合即可进入方案规定的
  最终反作弊检查。

### 1.4 规则库完善

Triton 默认规则库由原来的 7 张扩到 19 张，覆盖当前 A/B/C 主线需要的：

- native/wide program domain；
- integer-offset index tile 与 staged pointer tile；
- masked access、staged mask tile 与 unmasked access；
- elementwise；
- 显式轴 reduce 与 flattened reduce；
- broadcast；
- K-tiled dot 与无 contracted label 的 outer product；
- runtime scalar；
- constexpr C 槽与仅在 launch 生效的 compiler slot；
- cast/select；
- loop carry；
- output mapping。

规则卡总数仍为 19。`fg_prod` 没有新增卡，只把已有显式轴 reduce 卡升级到
v4，补齐 J 已规划的 `fn=prod` 到纯 sibling `@triton.jit` 乘法 combiner 的实现；
安全策略同步允许经过完整 AST 校验且调用图无递归的 sibling helper，任意 Python
global/free name 仍拒绝。随后 `fg_vector_l1` 也没有新增卡，只把已有
elementwise 卡升级到 v4，补齐 `abs/pow` 到 `tl.abs/libdevice.pow`；AST 只允许
受信 `triton.language.extra.cuda.libdevice.pow`，其他 namespace 调用仍拒绝。
纯 replay policy/生成失败不再触发 source-aware relift；外层重提必须有已编译的
normal source/replay 硬差异。

prompt 不再只是塞卡片，而是显式记录：

```text
π step → 检索到的 rule id → 最小规则卡集合 → 重写
```

这才是方案中的“计划—检索—生成”。

### 1.5 两条真实 TTIR 规范化规则

根据上一轮 Gate1 比较器的实现侧误拒，只加入了两条可枚举规则：

- 冗余 `x + 0`；
- `1 / sqrt(x)` 与 `rsqrt(x)`。

没有顺手写一个通用代数化简器。新规则仍必须来自真实 diff。

### 1.6 Gate2 最小真实原型

新增 `python/intentir/verify/gate2/`：

- source Triton 与 replay Triton 分别加载、分别 launch；
- IntentIR 第三腿直接复用项目已有
  `IntentIR → intent MLIR → linalg → LLVM → mlir-runner`；
- 三路使用同一份输入；
- 按 tolerance 两两比较并机械归属；
- lower 不支持时明确返回 `LOWERING_UNSUPPORTED`，不会退回旧解释器伪装成功。

本轮又把输入值生成抽到 production Gate2 模块，显式支持 normal、uniform、
positive、near-one、large-dynamic、IEEE-special 和 integer-boundary 七种可复现
分布。真实 L2 final stage 使用 nonnegative partial-sum 域，prod 使用 near-one
域，避免“输入本身不符
中间 buffer 契约、三条腿一起 NaN/下溢”形成假通过；embedding 使用合法随机索引
和显式 `0/V-1` 边界索引。batch identity 绑定 seed、distribution、逐参数 scale
与 tolerance。

Gate2 现有 `InputCaseTemplate`、canonical boundary extent 和 `Gate2Suite`：suite
完整运行每个 shape/value/seed case，只有逐例全部 PASS 才通过。live/batch 的
`--gate2-matrix` 已打通，首个真实 `fg_min` 矩阵在固定 C 下覆盖
`3 shapes × 4 value profiles × 3 seeds = 36` 个三方 case，全部最大误差 0；
`fg_ones` 还持续回归 1、BLOCK±1、非整除和远多 tile 六类 shape。其余 ready
Kernel 和需要改变 C 的 specialization 仍未完成全量矩阵。

CPU Lower2C 还修复了一个真实 Gate2 观测接缝：RunnerUtils 的 f32 文本打印会把
`91.181175` 舍入成 `91.1812`，造成 max 的假 `LOWERING_DIVERGED`。runner 现在
通过可 bufferize 的逐元素 f32→i32 bitcast 打印精确位模式并原位恢复，没有放宽
容差。输入 dense literal 也改为精确十六进制 f32 位模式，支持 `-0.0/±inf/NaN`。
IEEE-special max 矩阵随后抓到通用 CPU reduction 使用传播 NaN 的
`maximum/minimum`，而 Triton 与 GPU lowering 使用 `maxnum/minnum`；修复后
max 的 12 个 value×seed case 与 min 的 36 个 live matrix case 全部通过。

`fg_embedding` 又暴露了 production CPU Lower2C 只在 GPU 特化路径支持 gather
的缺口。通用 `GatherLowering` 现在把 `take` 与 `along_axis` lowering 为
`linalg.generic`，并在 region 中用 `tensor.extract` 执行数据依赖寻址；MLIR
回归同时覆盖两种 mode。Gate2 orchestration 也不再在第三腿失败时丢弃已经成功
的 source↔replay 比较，而是保存所有双方均执行成功的 pairwise agreement。

`python/tests/test_gate2.py` 已在真实 CUDA/Triton 和真实 MLIR lowering 上跑通
add、collapsed row-sum、softmax、RMSNorm 与 matmul 的三方一致。RMSNorm 还用
`eps=1e-5/1e-2` 两个值证明 production lower2c 真正消费 runtime scalar。
matmul 使用 `M=17,N=13,K=19` 的 f16 输入/f32 输出和三个随机 seed 覆盖
M/N/K 尾块；错误 B 地址会被机械归属为 `REPLAY_DIVERGED`。
这些固定 J 测试是
production lowering 能力证据，不代替每个 live artifact 的 Gate1 接收。

matmul 当前 Lower2C 会消费已经量化为 f16 的输入数值并产生 f32 输出，但其
production MLIR 仍统一发射为 f32；因此这是数值语义第三腿，不是
mixed-f16/f32 dtype-faithful lowering 已完成的证据。

### 1.7 首批真实库 Kernel 扩展

新增 `test/scrc_heldout_batch.py`，不读取旧 corpus `intent.json`，而是从真实
FlagGems Triton source 重新执行：

```text
source-only compile preflight
  → gpt-5.6-sol/xhigh Lift
  → Gate1 normal replay
  → masked-J
  → 正式 Gate1 闭合时才进入正式 Gate2
```

批处理显式登记 24 个 Kernel；当前 `fg_dot`、`fg_addr`、`fg_ones`、
`fg_embedding`、final-stage
`fg_sum/fg_mean/fg_max/fg_min/fg_prod/fg_vector_l2` 与 first-stage
`fg_vector_l1` 已具备审核过的 compiler/execution contract 并真实运行，其余
13 个保持 queued。结果为：

| Kernel | normal | masked | 正式 Gate1 | rejected-sample 离线 Gate2 |
|---|---:|---:|---|---|
| FlagGems `dot_kernel` | `1.0` | `1.0` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `addr_kernel` | `1.0` | `0.763387` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `ones_kernel` | `1.0` | `0.925` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `embedding_kernel` | `1.0` | `1.0` | `LEAKAGE_REJECTED` | PASS，6/6 fixed-shape value matrix |
| FlagGems `sum_kernel_2` | `1.0` | `1.0` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `mean_kernel_2` | `1.0` | `1.0` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `max_kernel_2` | `1.0` | `0.507692` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `min_kernel_2` | `1.0` | `0.507692` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `prod_kernel_result` | `1.0` | `0.509470` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `l2_norm_kernel_2` | `1.0` | `0.917544` | `LEAKAGE_REJECTED` | PASS |
| FlagGems `l1_norm_kernel_1` | `1.0` | `0.906140`；独立重跑 `1.0` | `LEAKAGE_REJECTED` | PASS，18/18 matrix |

真实扩展还暴露并修复了三个工程接缝：FlagGems `LibEntry → JITFunction` wrapper、
rank-0 tensor pointer，以及 Triton 从 TTIR 签名中消去普通 shape/stride scalar
的 specialization。pointer/tensor 和通过 `tensor_ref` 连接 J 的语义 scalar
仍不得被消去。

`fg_dot`/`fg_addr` 的 diff 又触发 wide program domain、flattened reduce、
staged mask tile 与 outer product 四张通用规则卡。它们因此是五核之后的首批
真实规则增长样本；`fg_prod` 只完善已有 reduce 卡，
`fg_vector_l1` 只完善已有 elementwise 卡和受限 libdevice 安全路径，
`fg_ones/fg_embedding/fg_sum/fg_mean/fg_max/fg_min/fg_vector_l2` 没有新增卡并直接
复用当前 19 张卡；embedding 的数据依赖地址已经由 J/B 明示，不需要添加一个
把具体 gather 拓扑编码进规则卡的专用模板。但十一例规模仍不能当作总体
held-out 成功率。

`fg_vector_l1` 的 rejected-sample suite 覆盖 `M={1,1023,1024}`、
runtime `ord={1.0,1.5,2.0}`、normal/uniform 与 3 seeds，共 18 个三方 case，
18/18 PASS；source/replay 始终零误差，两者对 Lower2C 的最大绝对误差为
`0.005859375`。

`fg_embedding` 使用静态 `indices[17]`、`weight[97,13]`、`output[17,13]` J，
normal 与独立 masked-J 都为 `1.0`，所以正式仍是 `LEAKAGE_REJECTED`。固定形状
下的随机合法索引/normal 权重与整数边界索引/uniform 权重各跑 3 seeds，6/6
三方 PASS 且全部最大误差 0。曾尝试把 K 改为 `{1,15,16,17}`，但该 compiler view
没有 runtime K/V/D scalar，Lift 正确生成静态 J；matrix builder 因 J extent
`17 != 1` fail-fast。该尝试不能写成动态 shape 覆盖。

后三例还补齐两个真实 Lower2C 接缝：零输入 symbolic-shape const 的 dims 传递，
以及 rank-0 i32 semantic scalar 经真实 `intent.cast` 转 f32。`fg_mean` 首次
diagnostic 因 CPU wrapper 把输入硬编码为 f32 而 `EXECUTION_FAILED`；修复后严格
复用同一 Lift artifact，source/replay 最大误差 0，二者对 lowering 最大误差
`3.6089e-9`。这些修复属于既有 const/cast/reduce lowering，不新增 IntentIR op。
详细证据见 `doc/SCRC_首批24个真实Kernel执行报告_2026-07-15.md`。

### 1.8 第七章外层语义重提

Gate1 现在已经不再是“内层 replay 小修用尽后直接失败”。新增的
外层 relifter 与方案第七章一致：

- 重写员仍然看不到源码；
- 只有提升官能看源码、源 TTIR 指纹和经白名单编码的历史裁决；
- 重提只允许修改 J、π 和定位表；
- 接口与 C 必须与受信 artifact 完全相同，任何漂移都在本地拒绝；
- provenance 只追加不覆盖，可以区分初始 Lift 与每轮重提；
- live 与 held-out batch 默认总共最多 3 版 artifact，即初始 Lift
  加最多 2 次语义重提；
- provider/schema 失败会收敛为可审计的 `NO_CLOSURE`，不会让整个
  Gate1 runner 崩溃。

首个真实证据使用历史上真实错误的 `fg_ones` 初始 LLM artifact，
不是人工篡改 J 或 π：

```text
初始 artifact:
  3 个 normal replay 候选均可编译
  score = 0.558333，硬项失败
  错误机制 = bind_program_domain

第 1 次外层重提:
  gpt-5.6-sol / xhigh，1 次响应、0 次 schema repair
  关键机制从 bind_program_domain 改为 bind_wide_program_domain
  J 仍表达输出常量 1，只补齐了规范输出绑定
  interface 不变，schedule/C 不变，source hash 不变
  normal replay score = 1.0，所有硬项通过

最终 masked-J:
  重放成功编译，但生成的是常量 0 而非 J 中的常量 1
  dataflow 硬项失败，其他结构项全过
  masked score = L = 0.925 > 0.5
  Gate1 = LEAKAGE_REJECTED
```

本次重提的 response id 为
`resp_0223585c072743d5016a56cd508354819ba4a5158466eeecda`，prompt SHA-256 为
`83dcabc02b34b22fa4f612a81de409a3614fa390af84fae611d628148049fb70`；源码
SHA-256 仍为
`ef4549cb35849f36a14a98ce789f38ee47e4a0e09c55570ae078119a7c2e0028`，artifact
provenance 记录从 1 条追加到 2 条。闭合 normal replay 与 masked replay
的 response id 分别为
`resp_039eebb5b5953bd6016a56cd987624819ab392735bbdb0c60d` 和
`resp_07320e35aeac0a75016a56cda97f408198a1e4df5ba3c0f86b`。

这次实验把两个问题明确分开了：外层重提确实修正了 J/π 与源结构
的不一致；最后没有放行的原因只剩第八章 masked-J 终检。因为
正式 Gate1 未闭合，本次 live 流程按方案没有进入 Gate2；历史上对
`fg_ones` 的 rejected-sample 离线三方 PASS 只是“Gate1 拒绝、
Gate2 通过”象限的评估证据，
不会覆盖这次门禁结果。

## 2. Gate1 与 Gate2 不是包含关系

Gate1 和 Gate2 验证的不是同一个命题：

- Gate1 问的是：在重写员看不到源码的条件下，J+π 能否在同一 C 下
  重建源编译结构，并满足语义分层、信息隔离和反作弊契约。
- Gate2 问的是：在有限的 shape/value/seed 测试矩阵上，source、replay 和
  production lowering 的输出是否一致。

因此，Gate2 通过不推导 Gate1 应该通过，Gate1 通过也不推导
Gate2 必然通过。四个象限都有独立含义：

| Gate1 | Gate2 | 含义 |
|---|---|---|
| 通过 | 通过 | 结构/来源闭包证据和当前行为证据同时成立 |
| 通过 | 不通过 | Gate2 抓到了 Gate1 的理论盲区、replay/lowering 工程错误，或测试中才显现的数值/边界问题 |
| 不通过 | 通过 | 完全可以合理：例如 π 泄漏、masked-J 不可识别、结构不同但当前输入上行为相同，或 Gate2 矩阵尚未覆盖反例 |
| 不通过 | 不通过 | 两道门禁都找到了问题，但失败责任仍需三方关系分析 |

所以“false reject”必须指定相对于什么标签。只有在论文能力统计中，
人为把“Gate2 矩阵通过”定义成二分类的行为标签时，才可以把
“Gate1 拒绝、Gate2 通过”记为 `behavior-label false reject`。这只是
统计名称，不表示 Gate1 的 SCRC 裁决错了，也不表示 Gate2 是 Gate1 的
真值裁判器。项目文档默认应写成“Gate1-rejected / Gate2-pass 象限”。

例如 `fg_ones`：它的 normal replay 和离线 Gate2 都通过，但 masked-J
`L=0.925 > 0.5`。按当前方案，Gate1 必须拒绝；Gate2 PASS 只证明
当前行为矩阵一致，不能证明 J/π 的信息分区和反作弊契约成立。

目前已修复的 Gate1 实现侧误拒包括：

- TTIR `+0` 规范化缺口；
- `rsqrt ↔ 1/sqrt` 的已知局部等价形态；
- 把 `do_not_specialize` 原始 metadata 误当语义接口硬项；
- source n-gram 误扫合法的 J.B；
- Triton 专门化消去普通 shape/stride scalar 时的 ABI 子序列处理。

以下则不能称为 Gate1 判错：

- π 中出现被禁止的前端 API；
- masked-J 泄漏比高于方案阈值；
- Gate2 在有限样例上通过，但 Gate1 的结构或来源契约不成立。

### 2.1 `do_not_specialize` 到底是什么

当前环境使用 Triton 3.6.0。对非 `tl.constexpr` 的 runtime 参数，Triton
JIT 默认会根据实际值或对齐特征生成专门化缓存变体。在这一版中：

- integer 值为 1 时可被提升成编译期常量；
- integer 能被 16 整除，或 pointer 满足 16-byte 对齐时，可生成
  `tt.divisibility = 16` 专门化属性；
- 这些专门化信息参与 JIT cache key，因而可能产生多个编译变体。

```python
@triton.jit(do_not_specialize=["n"])
def kernel(x, n):
    ...
```

这里的 `do_not_specialize=["n"]` 表示：`n` 仍是 runtime scalar，不要因为
本次调用恰好是 1、16 或其他可专门化值，就把它提升为常量或
绑定对齐假设。参数可以按名称或位置索引列出。

它与 `tl.constexpr` 正好相反：

- `tl.constexpr` 明确要求该参数是编译期值；
- `do_not_specialize` 要求普通 runtime 参数保持动态；
- Triton 3.6.0 会拒绝将同一参数同时标成 `tl.constexpr` 和
  `do_not_specialize`。

它的工程取舍是：可以减少编译变体和 cache 膨胀，保持更稳定的
runtime ABI；但也可能失去常量折叠、对齐优化等性能机会。

在当前 TianchenIR Gate1 中：

1. 它仍保存在 artifact/编译 metadata 中，便于审计和兼容旧材料；
2. 不把这份 decorator 列表本身交给重写员，也不将其当成 A/B 语义接口硬项；
3. source 和 replay 仍在同一 `CompileCase` 下实际编译，它对 TTIR 参数、
   属性或数据流造成的真实差异仍由 Gate1 硬比较抓取。

换句话说，当前策略是“不比较 `do_not_specialize` 这串原始标签，
但比较它对实际编译结果造成的影响”。

## 3. 现在是不是“只是原型”

是研究原型，但不是纯占位或玩具。

已经真实发生的事情包括：

- 真实 `gpt-5.6-sol/xhigh` Lift artifact 与 normal replay 已在
  add、row-sum、softmax、RMSNorm、matmul、FlagGems dot 与 FlagGems addr
  以及 FlagGems ones/embedding/sum-final/mean-final/max-final/min-final/prod-final/
  L2-final 和 L1-first 上结构闭合；十六者最终接收目前都被
  模型先验和 π 结构可识别性阻塞；
- 真实 Triton frontend 编译；
- 真实 TTIR 提取和纯代码结构比较；
- 真实 GPU source/replay launch；
- 真实 IntentIR→MLIR→LLVM lowering 与执行；
- 随机、tail、block±1、大动态范围、多 K tile 等边界样例。
- 真实错误的 LLM artifact 已经能在外层重提中修正 J/π，且接口、C
  与 provenance 护栏有可重放证据。

但必须同时写清：当前 16 个真实 LLM artifact 虽然 normal TTIR 都已
硬闭合，但 16/16 都因 masked-J 的 `L > 0.5` 被正式拒绝。因此，
目前真实 LLM artifact 在严格正式流程中的 Gate1→Gate2 正样本数仍为 0；
现有 Gate2 PASS 来自方案 9.1 允许的 rejected-sample 离线对照与固定 J
Lower2C 回归，不得表述为“已有真实 artifact 正式通过两道门禁”。

所以核心机制已经不是“差得远到只剩接口”。差距主要在覆盖与工程闭环：

- 尚未把数百个真实 Kernel 全部重新走一遍新的 `extract_scrc_artifact`；
- live 五核的 normal 路径已收敛，但一次性 masked-J
  无法稳定区分 π 泄漏和模型先验猜测，且 softmax 已实测出 π
  有序拓扑结合仍公开的结构信息，可成为语义替代编码；matmul 还证明 masked
  无需完全硬闭合也可能超过当前泄漏比阈值；
- 规则卡已在五核与十一个真实外部 Kernel 上证过；除 `fg_dot/fg_addr` 的新卡、
  `fg_prod` 对已有 reduce 卡和 `fg_vector_l1` 对已有 elementwise 卡/安全路径的
  映射补齐外，其余真实样本直接复用 19 张卡；
- production lowering 不是所有现有 A/B/C 结构都能走 CPU 腿；
- TileLang/CUDA 是方案后续阶段；
- 已有 24 个真实 Kernel 的批处理队列、逐 case 报告、resume/Lift cache 与失败
  分诊，但只实际完成前十一例；还缺其余 13 个及后续数百 Kernel 的运行证据。

因此最准确的称呼是：

> Gate1 已是可执行的 Triton SCRC MVP；Gate2 已是可执行的严格三方矩阵原型；
> 整体还不是数百 Kernel 规模的完整工具。

## 4. 这轮没有做什么

- 没有建立第三套 verifier；
- 没有引入新的 FULL/PARTIAL/OOS 模式；
- 没有用人工/确定性生成 J 作为正式路径；
- 没有把手写 MLIR PK 当成 LLM lifting 的有效证据；
- 没有新增计划外的 IntentIR op；
- 没有把环境版本、缓存策略、整份编译器 metadata 变成新的门禁条件。
