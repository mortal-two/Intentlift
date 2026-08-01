# Task 27 实验报告：add2d Gate1 mutation sensitivity

> 状态：`6/6 MAIN MUTATIONS REJECTED; 1 KNOWN COVERAGE PROBE MISSED`
>
> 机器证据：`mutations/add2d-v1/mutation_metrics.json`

## 1. 协议

本实验不重新调用 Lift，也不让 Replay agent生成候选。它从已通过Gate1/Gate2的TileLang/CUDA add2d
exact Replay各复制一份，做一个明确的局部mutation，再进入与正式运行相同的native
compiler → fingerprint → comparator路径。

`StaticReplayWriter`在这里只是exact bytes transport，model receipt被标为
`controlled-mutation-writer`；它不读取Source、不生成代码、不修复mutation，也不构成新的certificate。
每个变体重新编译Source与Replay。mutation一旦Gate1 hard拒绝就结束，整个实验`Gate2 calls=0`。

## 2. baseline控制

| frontend | parent accepted Replay | baseline Gate1 | elapsed |
|---|---|---:|---:|
| TileLang | `tilelang_add2d/run-002` | replay_closed | 9.56 s |
| CUDA | `cuda_tilelang_add2d/run-002` | replay_closed | 2.35 s |

因此后续拒绝来自受控变异，而不是current compiler/observer已经无法复现accepted Replay。

## 3. 六个主mutation

| frontend | mutation | 具体变化 | compile | Gate1 hard字段 | elapsed | Gate2 |
|---|---|---|---:|---|---:|---:|
| TileLang | wrong-op | `+`→`-` | PASS | `tir_structure` | 10.63 s | 0 |
| TileLang | wrong-index | 第二输入当前列→第0列 | PASS | `tir_structure` | 10.21 s | 0 |
| TileLang | wrong-effect | write→read-modify-write | PASS | `tir_structure` | 8.88 s | 0 |
| CUDA | wrong-op | `+`→`-` | PASS | `value_operations` | 2.19 s | 0 |
| CUDA | wrong-index | `threadIdx.x`→`threadIdx.y` | PASS | `coordinate_uses` | 2.48 s | 0 |
| CUDA | wrong-effect | write→read-modify-write | PASS | `memory_effects`, `value_operations` | 2.54 s | 0 |

六个mutation全部先编译成功，再被hard Gate拒绝；总拒绝时间36.92 s，平均6.15 s。这个结果支持一个很
有限但重要的结论：当前两个frontend profile能在执行Gate2前发现所选的operator、coordinate和effect
错误，而且拒绝依据与各自公开编译边界一致。

## 4. 不能隐藏的constant-offset probe

为了避免只挑容易成功的mutation，实验额外加入一个不计入主表的coverage probe：

```cpp
p1[index]  ->  p1[index + 1ULL]
```

NVCC在Replay PTX中明确生成`ld.global ... [%reg+4]`，Source是`[%reg]`；raw instruction topology、raw
memory和raw opcode diagnostics都不同。但semantic-effects v6的hard
`address_coordinate_folds`只追踪到memory address的cyclic/bitmask folds，当前不投影普通常量位移，所以
该probe仍为`replay_closed`。

这意味着论文不能把“主表6/6”外推成“Gate1检测所有wrong-index”。正确表述是：

- 选定的lane-coordinate rewire被hard coordinate projection捕获；
- 一个单元素constant displacement是已知false negative；
- 修复需要通用typed affine memory-access projection，不能粗暴把raw PTX文本设为hard，也不能加入
  针对`+4`或add2d的matcher。

该probe同样没有进入Gate2，因而本报告不声称三边执行对它作出了判定。

## 5. 论文最小用法

正文小表可以报告两个profile上的6个预定义coarse mutation均在Gate1被提前拒绝，平均6.15 s；紧邻表格
加一句限制：当前CUDA observer对常量地址位移仍有coverage gap。这样数据既能说明Gate1有实际价值，
又不会把有限观察器包装成完备验证器。
