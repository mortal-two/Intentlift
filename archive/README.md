# 归档索引

本目录保存被取代的代码、数据与流程材料。所有内容**只用于追溯与取证**，不是活动规范，
也不得作为当前 runtime、实验能力或论文数据来源。

活动状态见：
- 当前任务 `.trellis/tasks/08-01-final-three-kernel-closure/task.md`
- 权威设计 [`doc/README.md`](../doc/README.md)
- 派生规范 [`.trellis/spec/index.md`](../.trellis/spec/index.md)

## 批次

| 批次 | 文件数 | 内容 |
|---|---:|---|
| [`2026-07-21-wrong-control-plane/`](2026-07-21-wrong-control-plane/) | 131 | 架构重置前的旧控制面：RuleCard/Composer 路线的代码、artifacts 与测试 |
| [`2026-07-29-python-tests/`](2026-07-29-python-tests/) | 444 | 原 `python/tests` 的 mock 单元/回归测试套件。见 [`AGENTS.md`](../AGENTS.md)：不得在活动树中恢复或新增测试文件；验证只走真实 Kernel closure |
| [`2026-07-31-j-only-target-normal-composer/`](2026-07-31-j-only-target-normal-composer/) | 3 | Task 19 停止路线的 Target-Normal IR / Composer 原型源码（设计文档留在 `doc/archive/` 同名批次） |
| [`2026-07-31-pi-consuming-replay/`](2026-07-31-pi-consuming-replay/) | 2 | Replay 消费 π 的旧通道；Task 20 改为 J-only Replay 后退役 |
| [`2026-08-01-pre-scrc-python/`](2026-08-01-pre-scrc-python/) | 297 | pre-SCRC 的 benchmarks / 覆盖率战役代码 / 旧 CUDA 执行栈 / `corpus/` 语料 |
| [`2026-08-01-trellis-history/`](2026-08-01-trellis-history/) | 1005 | `.trellis` 的已归档任务（793）、退役 workflow 与实验（208）、空 workspace |

## 另外两处归档树

归档材料按类型分处三地，各自带索引：

- [`doc/archive/`](../doc/archive/README.md) — 被取代的**设计文档**（6 批）
- [`archive/`](.) — 被取代的**代码与数据**（本目录）
- ~~`.trellis/archive/`~~ — 2026-08-01 起并入本目录的 `2026-08-01-trellis-history/trellis-archive/`

## 解释规则

历史 artifact、receipt 与 verdict 只按其**落盘时的协议版本**解释。后续 schema、TypeCheck、
observer profile 或 RefEval 升级不静默追认、不改写、也不重判旧结论。
