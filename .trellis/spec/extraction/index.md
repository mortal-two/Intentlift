# Extraction 层:LLM 前端提取

定义"如何把任意 kernel 提取成 IntentIR JSON":LLM 的角色、evidence 的角色、以及前端输入清单。

## 已落地(2026-06-26):提取管线真由 provider agent 跑通

`JSON 是不是 provider agent 生成的?` —— **是**。`python/intentir/extraction/`:
- `client.py`:OpenAI `responses` 协议 HTTP 客户端(stdlib urllib,**绝不打印
  key**);`prompt.py`:内嵌 IntentIR 原子 op 词表 + 一个**合成的、不在评测集里**
  的 few-shot(L1-normalize),只给 schema 与形状,不泄漏目标 kernel 答案;
  `extract.py`:结构校验 + smoke 解释 + 有界 re-roll。模型 `gpt-5.5`。
- `corpus_specs.py` + `run_corpus.py`:**15 个真实 FlagGems Triton 内核**
  (softmax/log_softmax/silu/gelu_tanh/relu/rms_norm/matmul/sigmoid/tanh/exp/abs/
  log/add/mul/maximum)。每个:源码→LLM→JSON→`interpret(JSON)` vs **实际
  `flag_gems.<op>` 内核**(动态证书,rel ≤ 1.9e-7)→发射 intent dialect→
  `-convert-intent-to-linalg`(零残留 `intent.*`)。**15/15 全过**。产物committed:
  `corpus/<name>/{intent.json,provenance.json}`(provenance 记录 model / prompt
  sha / 源文件 / rel-err,可审计)。`saved` 模式对committed JSON 做确定性复验(不联网)。
- **多前端(语言无关)**:`cuda_frontend.py` 把**手写 CUDA C++** 内核
  (softmax/silu/rmsnorm,带 grid-stride / shared-memory / warp reduction)提到
  **同一份 IntentIR**(CUDA softmax 与 Triton softmax 提出的 op-graph 逐 op 相同),
  verify vs torch ref rel ~1e-7。证明提取器前端语言无关(nvcc 缺失,故此处对 torch
  参考验证 = 语言可插拔演示,区别于 Triton 语料的"跑实际内核")。

## 何时读

- 改动提取管线、LLM prompt、evidence 提取、契约前置数据之前。
- 新增 / 变更前端输入 kernel 来源之前。

## 文件

| 文件 | 内容 |
|---|---|
| [llm-extraction.md](llm-extraction.md) | LLM 直接提取到张量级原子 IR(科研核心),提取流程,re-roll,不是 pass 分解。 |
| [evidence.md](evidence.md) | evidence 的三个独立角色:生成 hint + 验证标尺 + LLM 自报 claim;化解"约束生成 vs 泛化"矛盾。 |
| [frontend-inputs.md](frontend-inputs.md) | 前端输入 kernel 清单(Triton / TileLang / CUDA)、Dataset 目录、FlagGems 入口获取。 |

## Pre-Development Checklist

- [ ] 原子化分解仍由 LLM 完成,没有被塞进 pass 吗?
- [ ] evidence 是作为 hint / 标尺,而不是"硬前置门槛"卡死泛化吗?
- [ ] 新 kernel(没现成 evidence)也能走通提取 + 动态验证吗?

## Quality Check

- 提取不依赖"规则能不能提出 evidence"来决定能不能做;LLM 读源码 + 动态验证是兜底。
- 提取产物是张量级原子 JSON,不是库级大算子标签、也不是标量长图。
