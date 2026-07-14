# SCRC Gate1 随机／边界能力复验报告

日期：2026-07-14

## 1. 实验范围

本报告运行 `test/scrc_gate1_capability.py` 的 18 个定向样例：五个正确重放、
四个等价改写、七个行为错误变异、一个非语义 JIT 元数据变化和一个 π 泄漏
变异。

Gate1 只做 compile-only 与 TTIR 结构裁决。该脚本的动态部分仍是
source/replay 两方 GPU 代理；正式三方 Gate2 已在 `python/tests/test_gate2.py`
以 add 原型单独跑通，但尚未替代本脚本的五核离线矩阵。

执行命令：

```bash
PYTHONPATH=python python test/scrc_gate1_capability.py \
  --gate2-depth full \
  --cache-dir /tmp/scrc_gate1_capability_cache \
  --strict-expectations \
  --fail-on-false-accept
```

输入覆盖：

- seed `0/1/2`；
- add：`N={1,63,64,65,113}`，另含 signed-zero、`±inf`、NaN；
- row-sum/softmax/RMSNorm：`N={1,63,64,65}`；
- softmax：随机、全负、大动态范围；
- row-sum：大数抵消与 `±inf`；
- RMSNorm：全零、`1e10`、`1e-10`；
- matmul：M/N/K 覆盖 1、block−1、block、block+1、非方形 tail、多 K tile、
  零矩阵。

## 2. 结果

| 样例组 | Gate1 | 两方动态代理 |
|---|---:|---:|
| 五个正确重放 | 5/5 闭合 | 5/5 通过 |
| add 交换操作数 | 闭合 | 通过 |
| add 冗余 `+0` | 闭合 | 通过 |
| row-sum 反向谓词 | 闭合 | 通过 |
| RMSNorm `rsqrt→1/sqrt` | 闭合 | 通过 |
| 删除 `do_not_specialize` | 闭合 | 通过 |
| 七个行为错误变异 | 0/7 闭合 | 0/7 通过 |
| π 中出现 `tl.load` | 编译前拒绝 | 行为通过 |

混淆矩阵：

```text
true accept  = 10
false accept = 0
false reject = 1
true reject  = 7
```

关键结果：

```text
P(Gate2 proxy pass | Gate1 closed)    = 10 / 10 = 1.000
Gate1 coverage of behavior-pass cases = 10 / 11 = 0.909
P(Gate2 proxy pass | Gate1 rejected)  = 1 / 8   = 0.125
```

没有出现“Gate1 闭合、动态失败”的 false accept。七个行为错误变异仍全部被
Gate1 拒绝。

## 3. 本轮修复对应的根因

### 3.1 `+0`

真实 TTIR 使用与 masked load fill 共用的 dense-splat zero：

```text
%out   = arith.addf %a, %b
%out_1 = arith.addf %out, %zero_tensor
```

规则现在识别标量零和 dense-splat 零，在 SSA 层把第二个 add 折回 `%out`；若
zero 仍被真实 mask fill 使用，则保留该常量节点。

### 3.2 `1/sqrt ↔ rsqrt`

真实 TTIR 的两种形态分别是 `math.rsqrt` 与
`math.sqrt + arith.divf(1, root)`。规则只吸收这一固定局部模式。

### 3.3 `do_not_specialize`

它改变 Triton JIT 的特化策略，但不改变本次 SCRC 要比较的 callable signature、
J.A、J.B 或同一 CompileCase 下的 TTIR 结构。因此保留为编译 metadata，不再作为
硬接口项。

### 3.4 π API 泄漏

这是唯一剩余 raw false reject：候选代码行为正确，但 π 写入 `tl.load`，破坏了
“具体 API 只能来自规则卡、语义必须来自 J”的信息分区。它是 SCRC 有意拒绝，
不是继续放宽 TTIR 规范化的问题。

## 4. 仍然成立的能力边界

- 本实验是 18 个手工设计的 probe/mutant，不代表数百真实 Kernel；
- 五核 J/π 探针不能替代真实 `源码→LLM extract_scrc_artifact`；
- 本脚本动态部分仍是两方代理，不能发现 J 与 replay 自洽地错；
- production Gate2 目前只以 add 完成三方接线证明；
- 后续规则只能从真实 LLM 样本 diff 增长，不能据此扩成通用代数证明器。
