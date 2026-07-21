# 动态执行一致性反哺契约

## 旧系统的缺陷(本项目要补的)

旧系统契约等级**纯静态**算,完全不看动态执行结果——动态验证(`diff.ok`)只是旁边记着的审计信号,没有任何机制把"动态验过的"反映到契约里。结果是一大批 kernel 动态上明明对(执行一致),却被静态 obligation 压在 PARTIAL、或干脆被拒,且救不回来。

## 本项目的规则:动态可救静态

> **静态证不出(obligation UNKNOWN)+ 动态执行一致 = 可接受**,而不是烂在 PARTIAL 或被 OOS。

- 动态执行一致性是契约的**一等输入**,不是旁路审计信号。
- "运行时才能确定"的东西(gather/scatter 的数据依赖地址、数据依赖分段等)正是靠这条被正当接住:静态 UNKNOWN,动态过 → 接受。
- 只有**本质边界**(见 [../capability/boundaries.md](../capability/boundaries.md))才判 OOS;"静态保守"不等于 OOS。

## atomic / 浮点非结合的 tolerance 语义

- 并发归约里浮点加法不结合 → 并发顺序影响数值结果。这类归约在 IR 里标 `float_nondeterministic`(见 [../ir/atomic-ops.md](../ir/atomic-ops.md))。
- 对这类归约,动态执行一致性用**合理 tolerance**(源 kernel 本身就有顺序抖动,本不该要求 bit-exact),而不是 scope-out。
- 这把旧系统最大的盲区(atomic 一律 OOS)变成可验证的一等公民,也是潜在的 paper 亮点。

## 源码执行的获取方式

- **local**:源 kernel 在同机可跑。
- **remote**:在源能力设备上跑(如远程 GPU / RVV)。
- **cached**:复用之前记录的 golden 执行。
- 记录需含 input shape/dtype/seed/tolerance/源版本,保证可复现。
