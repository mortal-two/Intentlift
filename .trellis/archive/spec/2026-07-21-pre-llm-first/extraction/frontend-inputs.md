# 前端输入 Kernel 与证据合同

本项目面向真实、已优化的 Triton / TileLang / CUDA Kernel。来源清单会持续扩展，
因此稳定 spec 不承诺一个易漂移的固定数量；实际覆盖以 `corpus/`、测试清单与报告为准。

## 支持的前端类别

| 前端 | 需要的 adapter 证据 |
|---|---|
| Triton | Python source/AST、launch binding、constexpr/specialization、TTIR fingerprint、真实执行 |
| TileLang | source/launch contract、TIR/编译器中层 fingerprint、真实执行 |
| CUDA C++ | kernel signature、launch/grid、编译中层或受控结构 fingerprint、真实执行 |

新增前端不要求与已有前端生成相同 JSON；它必须映射到同一个 IntentIR semantic space，
并能被共享 backend contract 消费。

## 输入来源

典型来源包括：

- FlagGems、Liger-Kernel、xFormers、flash-attention 等真实高性能库；
- 项目自有手写 Kernel 与论文／benchmark Kernel；
- 为隔离机制而构造的最小 probe；
- held-out family 与 fresh、uncached campaign 输入。

第三方安装目录、外部 Dataset 与 refs 默认只读。进入仓库的 corpus artifact 必须记录
来源和 digest，不能复制一份失去 provenance 的源码后当作原样输入。

## Capture 与 provenance

每个可用于正式 Lift 的样本至少绑定：

- source identity/digest；
- Python/CUDA signature 与参数顺序；
- launch positional/keyword binding；
- compiler view 的 parameter order、kind、dtype、rank 与 specialization；
- grid/runtime shape contract；
- 适用的 frontend/compiler version；
- 合法输入域与 source correctness 前提。

source、launch、compiler view 任一不一致都属于 P0 capture failure：停止并修 capture，
不能要求 LLM 猜测，也不能归罪于 Gate1。

## 与能力矩阵的关系

前端语料常覆盖 recurrence、atomic、gather/scatter、in-place、多输出、复杂 layout 等
机制。是否纳入取决于 [capability-matrix.md](../capability/capability-matrix.md) 的
represent/extract/verify/lower 五维合同，而不是因为某个库名字“重要”就特判。

## Corpus 纪律

- `corpus/` 保存 artifact、provenance、覆盖和报告，不是 spec 真值；
- extraction-only、Gate-closed、backend-executed、performance-measured 必须分开标记；
- hand-written semantic subset 不能伪装成 exact source capture；
- source 自身不可信或环境不支持时，要记录 coverage gap，不能制造 PASS；
- 新数量和新结果更新机器可读 manifest/report，不在本文件追加日期流水账。
