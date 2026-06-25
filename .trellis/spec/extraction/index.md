# Extraction 层:LLM 前端提取

定义"如何把任意 kernel 提取成 IntentIR JSON":LLM 的角色、evidence 的角色、以及前端输入清单。

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
