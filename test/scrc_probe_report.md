# IntentIR SCRC Triton 探针报告

日期：2026-07-14
结论：**受控的第一阶段探针通过。五个 kernel 均在首次盲重放、零次修复下编译成功；源/重放 TTIR 的接口、算子多重集、归约结构、mask/effect 摘要全部一致。观察到的剩余差异是有限且可枚举的规范化噪声，没有发现发散型差异。**

## 1. 本次探针回答的范围

本次只验证 SCRC Gate1 的最小核心假设：

```text
J + π + 前端接口 + C 空位名 + 检索规则卡
                    ↓（重写员看不到源码和 C 数值）
               重放 Triton
                    ↓
       与源 Triton 用相同 specialization 编译
                    ↓
          人工比较两份 TTIR 的结构
```

本次**没有执行 Gate2**，也没有做遮盖 J 的反作弊消融。因此这里的“通过”是 replay-closure 可行性证据，不是全域语义等价证明，也不是完整 SCRC 认证结果。

## 2. 隔离与可复现控制

- 提升官手写的 J、π、接口和 8 张规则卡在 [scrc_probe_inputs.json](scrc_probe_inputs.json)。
- 源 kernel 在 [scrc_probe_sources.py](scrc_probe_sources.py)，没有进入重写请求。
- GPT 返回的盲重放代码原样保存在 [scrc_probe_replays.py](scrc_probe_replays.py)。
- π 静态检查：不含 `tl.*`/`triton.*` API；没有源文件路径；没有 BLOCK/warp/stage/CTA 的具体值。π 中的数字只来自 `J...nodes[n]` 一类结构引用。
- C 数值仅存在于 [scrc_probe_compile.py](scrc_probe_compile.py) 的注入/编译侧，重写员只知道空位名称。
- 源和重放在不同 Python 进程、不同 Triton cache/dump 目录中完成最终编译。
- 使用 `JITFunction.warmup()`，只编译、不启动 kernel；同时读取 `CompiledKernel.asm["ttir"]` 并启用 `TRITON_KERNEL_DUMP=1`。
- 设置 `TRITON_ALWAYS_COMPILE=1`，避免磁盘 cache hit 导致 dump 目录为空。

模型记录：

| 项 | 值 |
|---|---|
| 当时使用的不完整模型名 | `gpt-5.6` |
| 该模型名结果 | HTTP 502：`unknown provider for model gpt-5.6` |
| 实际请求/响应模型 | `gpt-5.5` / `gpt-5.5` |
| reasoning effort | `high` |
| response id | `resp_09484adc73d88790016a55f4e4fd188195868c5a7ef804a822` |
| 重写调用次数 | 1 |
| 编译修复次数 | 0 |
| 重写输入 SHA-256 | `b218d66338b5838291f3ce649e64983680f5ae1e3d50c697f56476c4e0d4da22` |

注：这张表是早期探针的历史记录，不表示 5.6 系列不可用。正确模型名
`gpt-5.6-sol` 已在后续 `test/scrc_live_add_report.md` 中以 `xhigh` 完成真实
Lift→Gate1→Gate2 闭环。

环境记录：Python 3.10.12、PyTorch 2.10.0+cu130、Triton 3.6.0、NVIDIA RTX 5090 D（sm_120）、驱动 580.95.05。

## 3. 五个固定 specialization

| kernel | dtype / shape | grid | 注入的 C |
|---|---|---|---|
| add | A/B/O：`f32[1000]` | `(4,)` | `BLOCK=256, warps=4, stages=2, ctas=1` |
| row-sum | X：`f32[37,1000]`，O：`f32[37]` | `(37,)` | `BLOCK_N=1024, warps=4, stages=2, ctas=1` |
| softmax | X/Y：`f32[37,1000]` | `(37,)` | `BLOCK_N=1024, warps=4, stages=2, ctas=1` |
| RMSNorm | X/Y：`f32[37,1000]`，W：`f32[1000]`，eps runtime | `(37,)` | `BLOCK_N=1024, warps=4, stages=2, ctas=1` |
| matmul | A：`f16[45,53]`，B：`f16[53,37]`，O：`f32[45,37]` | `(2,2)` | `BM=BN=BK=32, warps=4, stages=2, ctas=1` |

matmul 含动态 `ceil_div(K, BLOCK_K)` 的 `scf.for`，不是仅覆盖单 K tile 的退化版本；M/N/K 三个方向均包含 tail mask。

三个 row kernel 的 body 不读取 `M`，而是依赖 host 侧 grid=`(M,)` 保证“恰好一 program 对应一行”。因此 grid 映射是接口/launch contract 的硬字段，不能只比较 TTIR 函数参数。

## 4. 人工 TTIR diff 结果

[scrc_probe_compare.py](scrc_probe_compare.py) 只提供计数 sanity check；下面的结论来自对五组 raw unified diff 的逐段人工检查，包括每个 load/store 的地址、mask 依赖、归约 combiner 和数据流。

| kernel | 编译 | 接口 | 算子多重集 | 归约 | mask / effect | 人工数据流结论 |
|---|---|---|---|---|---|---|
| add | 首次成功 | 精确相同 | 精确相同 | 无 | 2 masked load、1 masked store；同一 `offset<L`；fill=0 | `load A/B → add → store` 相同 |
| row-sum | 首次成功 | 精确相同 | 精确相同 | 1× `(axis=0, addf)` | 1 masked load(fill=0)、1 unmasked store | `row load → sum → row store` 相同 |
| softmax | 首次成功 | 精确相同 | 精确相同 | `(axis=0,maxnumf)` + `(axis=0,addf)` | 1 masked load(fill=`-inf`)、1 masked store | `load→max→sub→exp→sum→div→store` 相同 |
| RMSNorm | 首次成功 | 精确相同 | 精确相同 | 1× `(axis=0, addf)` | 2 masked load(fill=0)、1 masked store | `x²→sum/N→+eps→rsqrt→×x→×w` 相同 |
| matmul | 首次成功 | 精确相同 | 精确相同 | 无 `tt.reduce`；1× `scf.for`、1× `tt.dot` | 每 K tile 2 masked load；1 masked store；6×`slt`、3×`and` | A=`m*K+k`、B=`k*N+n`、O=`m*N+n`，K-loop accumulator 数据流相同 |

Python 侧接口也逐项相同：

| kernel | runtime 参数顺序 | constexpr | do-not-specialize |
|---|---|---|---|
| add | 相同 | `BLOCK` | 无 |
| row-sum | 相同 | `BLOCK_N` | 无 |
| softmax | 相同 | `BLOCK_N` | 无 |
| RMSNorm | 相同 | `BLOCK_N` | `eps` |
| matmul | 相同 | `BLOCK_M/BLOCK_N/BLOCK_K` | 无 |

### 4.1 观察到的剩余差异

只有以下三类：

1. **调试位置噪声**：源/重放文件路径、行列号、`#loc`、`loc(callsite(...))`。
2. **非语义命名噪声**：Python 局部名导致的 SSA 名不同，如 `cols`/`col`、`out`/`y`、`square_sum`/`mean_square`。
3. **独立纯操作的拓扑输出顺序**：
   - add 中，源先发射 `pid*BLOCK` 再发射 `make_range`；重放顺序相反。
   - matmul 中，源先构造 pointer DAG 再构造 mask DAG；重放因表达式内联，先出现部分 mask DAG，再出现 pointer DAG。二者引用同一 m/n/k 坐标，最终接到相同 load/store。

RMSNorm 源码显式写了输入 `.to(tl.float32)`，重放没有写；因为输入本来就是 f32，Triton 前端将其消成 no-op，TTIR 中不存在差异。这说明比较层选择 TTIR 能自然吸收一部分前端写法噪声。

### 4.2 差异能否用固定规则规范化

本批次可以，且规则很小：

1. 删除 debug location/source path。
2. SSA 按定义/使用图做 canonical rename，不保留 Python 局部名。
3. 不按文本顺序比较互不依赖的纯 op；构造 DAG 后做稳定拓扑排序，或直接做带类型/属性的图同构比较。
4. 归约 region 单独规范化 block argument 名，保留 axis + combiner。
5. predicate DAG 对 `and` 做交换律排序，并统一 `a<b`/`b>a`（本次尚未出现反向比较，但应预置）。

没有观察到需要依赖 kernel-specific 特判的差异，也没有出现不可枚举的算法发散。

## 5. 对三个问题的直接回答

### ① 重放能编译吗？

**能，5/5 首次编译成功，零次修复。** 这比“一两次修复内”更强，但只代表本受控样本。

### ② 算子多重集 + 归约 + mask 结构对得上吗？

**对得上，5/5 全部一致。** 接口、常量/fill、load/store effect、归约 axis/combiner、softmax 稳定化数据流、matmul K-loop/dot/三向 tail mask 均一致。

### ③ 剩余差异是什么性质？

**是少量固定规则可规范化的差异。** 当前只见 debug loc、SSA/local name、独立纯 op 的线性输出顺序。未见发散、不可枚举或需要 LLM 裁决的差异。

## 6. 这对正式重构意味着什么

探针支持继续做 Triton SCRC MVP，但当前项目不能只“加一个 compare 函数”就完成新方案。至少需要调整：

1. **J.B 必须从自由字典变成严格 schema**：mask、load fill/neutral value、store guard、output mapping、effect、alias/sync 都要可验证。现有 corpus 的 softmax 等 artifact 主要只有 A 图，缺少重放所需的 B 信息。
2. **接口合同独立建模**：pointer/runtime scalar/constexpr、参数顺序、dtype/rank/stride、input/output/inout、alias、`do_not_specialize`，以及 TTIR 本身不携带的 grid/launch-domain 映射。
3. **π 独立 schema + 机制白名单 + 泄漏扫描**，不能继续塞进现有 Intent JSON 的自由字段。
4. **C 分成 slot 与 value**：重写员只见 slot；注入器和 cache key 持有实际 BLOCK/warp/stage/CTA。
5. **规则卡成为版本化系统资产**：至少按 frontend + version pin；Triton 3.6 的 API 约束必须进入规则卡。
6. **新增 replay prompt/代码提取/沙箱编译/两次小修循环**；现有 `extract_intent()` 是源码→JSON 提升器，不能复用为重放器。
7. **新增 TTIR graph fingerprint**：文本 regex 计数只能当 sanity；正式裁判应解析 op/type/attrs/SSA use-def/reduce region/predicate/effect。
8. **更新 verification spec 的措辞与层次**：SCRC Gate1 是动态 Gate2 之前的 replay-closure 结构证据，不推翻“lift 不是形式证明”的诚实上限；现有动态 L3/L4 应成为新方案的 Gate2，而不是被删除。

还发现两个需要在正式 schema/规范化策略中先钉死的具体问题：

- 当前没有 committed 的 row-sum corpus J；`corpus/bulk/sum` 是全张量两轴归约到标量，不能冒充本探针的 `[M,N]→[M]` 行归约。
- 当前 committed RMSNorm J 使用 `sqrt` 后 `recip`，本探针和 Triton TTIR 使用 `rsqrt`。重构前必须选择统一 canonical J，或明确注册 `rsqrt ↔ recip(sqrt)` 的受控规范化；不能让它变成临时特判。

另外，项目的 matmul 资料同时存在 `mk,kn->mn` 和权重按 `nk` 存放的 `mk,nk->mn` 约定。layout/stride 必须由接口/J.A 明确拥有，不能靠 kernel 名称猜。

## 7. 不能过度解读的地方

- 这是 controlled probe，规则卡有意给出稳定、版本正确的 Triton 构造；它证明“这条技术路线可跑”，还没有证明对任意真实优化 kernel 都能闭合。
- J.B 中给了明确地址表达式和 predicate，这在新方案里是合法且必要的信息，但后续必须通过遮盖消融证明语义没有从 π 泄漏。
- 尚未测试 autotune/heuristics、多输出、atomic、online softmax、非连续 stride、别名/inout、动态 control flow。
- row-sum/softmax/RMSNorm 都是单 row tile 版本，不覆盖 `N>BLOCK_N` 的流式归约或 online-softmax 状态携带。
- 下一轮最有价值的不是立刻扩到几百个 kernel，而是用**同一套规则卡**跑一组 held-out 的真实 FlagGems/Triton kernels，并加入 J 遮盖消融；否则容易把“规则卡复刻了这五个手写源”误当成泛化能力。

## 8. 复现命令

源侧：

```bash
rtk env \
  TRITON_KERNEL_DUMP=1 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_STORE_BINARY_ONLY=0 \
  TRITON_CACHE_DIR=/tmp/intentir-scrc-source-cache \
  TRITON_DUMP_DIR=/tmp/intentir-scrc-source-dump \
  python test/scrc_probe_compile.py \
  --side source --output-dir /tmp/intentir-scrc-probe
```

重放侧：

```bash
rtk env \
  TRITON_KERNEL_DUMP=1 \
  TRITON_ALWAYS_COMPILE=1 \
  TRITON_STORE_BINARY_ONLY=0 \
  TRITON_CACHE_DIR=/tmp/intentir-scrc-replay-cache \
  TRITON_DUMP_DIR=/tmp/intentir-scrc-replay-dump \
  python test/scrc_probe_compile.py \
  --side replay --output-dir /tmp/intentir-scrc-probe
```

结构摘要：

```bash
rtk python test/scrc_probe_compare.py /tmp/intentir-scrc-probe
```

`test/lit.cfg.py` 只收集 `.mlir`，这些 Python 探针不会进入现有 lit 测试集合。
