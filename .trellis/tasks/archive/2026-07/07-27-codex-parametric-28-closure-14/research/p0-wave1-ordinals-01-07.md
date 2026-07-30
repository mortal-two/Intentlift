# Task14 Wave 1 P0：registry ordinals 1--7

> 审查日期：2026-07-28  
> fresh-run 基线：`6980cf50`  
> 范围：只记录 Source/Harness 的事实合同；本文件不是 `(J,C,π)` 的组成部分，
> 不进入 Replay packet，也不预填 Intent op、result relation、access/control 或 π topology。

## 1. Source snapshot 规则

本波实际 resolver 选择的是当前 Python 环境中的 `module://liger_kernel.ops.*` Source。
每次 run 仍将 exact direct-symbol slice 保存为 `source.py`，并把 source projection 保存为
`source_projection.json`。下面的 SHA-256 是 P0 审查时的完整模块快照；若正式 run 前 resolver
位置或 digest 改变，该 case 必须重新 P0，不能静默落到不同 Dataset 版本。

| ordinal | case | direct symbol | selected module SHA-256 |
|---:|---|---|---|
| 1 | `high_performance:liger_swiglu` | `_swiglu_forward_kernel` | `5d55bd62ea4e9e226e1ae82704333e0663901292976c4fae12b6195a8ef785c0` |
| 2 | `high_performance:liger_rms_norm` | `_rms_norm_forward_kernel` | `1759600f3a4a7f164c233f3d0a83d7eaaadb45f9d168b0caa2824029507311e1` |
| 3 | `high_performance:liger_fused_add_rms_norm` | `_fused_add_rms_norm_forward_kernel` | `ce3d0031d6f9fff41a9638eb1bdae7d7eb544bd82e49d5b605880c4e50394457` |
| 4 | `high_performance:liger_rope` | `_triton_rope` | `bfeec91d1aeffb9226a113a4a4a7e5df45b0588e2a4dadad025ef341e0789a5e` |
| 5 | `high_performance:liger_cross_entropy` | `liger_cross_entropy_kernel` | `01276231c0f9f18d3ad6353b7c28b2f4426c408884515ee50934d89c10563958` |
| 6 | `high_performance:liger_geglu` | `_geglu_tanh_forward_kernel` | `fabb8c0020ec147d58b006a4e811af67492d2c622b044eb2185d1d03247cd945` |
| 7 | `high_performance:liger_layer_norm` | `_layer_norm_forward_kernel` | `7f1f97ab641dfdb0e3c5df5a4309807fcc7192936197a10ef3ab70cad6d857fd` |

特别地，ordinal 1 的 installed Source 有 runtime `gate_multiplier`，而 Dataset fallback 的旧
Source 没有该形参和缩放。两者是不同程序/ABI；本波只认证上表 snapshot，固定代码不得为旧
Source 补默认参数。

## 2. 逐 case P_cert 审查

### 2.1 SwiGLU — `PROFILE_READY`

- ordered logical buffers：`a,b` 为 `f32 [h0,h1]` input，`c` 为同 shape output；当前
  valuation 为 `[2,127]`；
- runtime：`stride=127:i64`、`gate_multiplier=1.0:f32`；semantic constexpr
  `n_cols=127` 与 `h1` 同一 Source-grounded extent；C 只有 `BLOCK_SIZE=128`；
- grid valuation 为 `[2]`，仅 realization 到 `h0`；observable output 为 `c`；
- evidence：`python/intentir/verify/workload_adapters.py:638` 及 selected installed
  `liger_kernel/ops/swiglu.py`。

Shape ledger：`h0=2` 是 tensor row identity（grid 只实现它）；`h1=127` 同时由 tensor axis
和 `n_cols` 绑定。两者均要求 Lift 作者化为 parameteric dimension，不能固化成 `2/127`。

### 2.2 RMSNorm — `PROFILE_READY`

- ordered buffers：`Y` output、`X/W` input、`RSTD` output，全部 boundary `f32`；`X/Y`
  为 `[h0,h1]=[2,127]`，`W=[h1]`，`RSTD=[h0]`；
- selected variant：`casting_mode=LLAMA`、affine enabled、`eps=1e-6`；`n_cols`
  绑定 `h1`，C 为 `BLOCK_SIZE=128`，grid 实现 `h0`；
- observable outputs：`Y,RSTD`；
- evidence：`python/intentir/verify/workload_adapters.py:732`、selected installed
  `liger_kernel/ops/rms_norm.py`、`python/tests/test_row_reduction_pk.py:54`。

Shape ledger：`h0` 是 row identity；`h1` 同时由 tensor axis 与 runtime `n_cols` 绑定，均可参数化。

### 2.3 Fused Add RMSNorm — `PROFILE_READY`

- ordered outputs 为 `Y,S,RSTD`；inputs 为 `X,R,W`；所有 boundary carrier 为 `f32`；
  `X/R/S/Y=[h0,h1]=[2,127]`、`W=[h1]`、`RSTD=[h0]`；
- selected variant：`casting_mode=-1`、`eps=1e-6`、`n_cols` 绑定 `h1`；C 为
  `BLOCK_SIZE=128`，grid 实现 `h0`；
- SCRC 的 Source runner、RefEval 与三边 Gate2 已逐 J output 观察 `Y,S,RSTD`。现有 CUDA
  lowering PK 只比较 `S,Y` 是独立的后端测试覆盖缺口，不允许据此从 P_cert 删除 `RSTD`；
- evidence：`python/intentir/verify/workload_adapters.py:1008`、
  `python/intentir/verify/gate2/orchestrate.py:192`、
  `python/intentir/verify/gate2/compare.py:279`。

Shape ledger：`h0` 是 row identity；`h1` 与 runtime `n_cols` 绑定；两者均可参数化。

### 2.4 Liger RoPE — `PROFILE_READY`

- selected program 是 `_triton_rope` 的 `BACKWARD_PASS=false` forward variant；`q,k`
  是 inout，`cos,sin` 是 input；
- Source-grounded identities：`q=[h0,h1,h2,h3]`、`k=[h0,h1,h4,h3]`、
  `cos/sin=[h5,h1,h3]`；`sl↔h1`，`bs↔h0`，`n_qh↔h2`，`n_kh↔h4`，
  `hd↔h3`，`cos_bs↔h5`。相等 valuation `h0=h5=1` 不代表两者是同一变量；
- tiny valuation `[1,4,2/1,16]` 能走完整 forward 路径；本波 boundary 使用与本项目
  upstream direct-kernel PK 一致的 `f32` primary specialization，四个 row stride 使用该调用的
  concrete `i32` specialization；
- grid `[4]` 只是 `bs*sl` realization，不向 J 泄漏乘积公式；
- evidence：`python/intentir/verify/workload_adapters.py:1585`、
  `python/tests/test_rope_pk.py:103`、selected installed `liger_kernel/ops/rope.py:7`。

### 2.5 CrossEntropy — `PROFILE_READY`

- selected direct-program variant：`HAS_GRADIENTS=false`、`HAS_WEIGHT=false`、
  `HAS_SOFTCAPPING=false`、三个 optional RETURN flag 均 false、`reduction=mean`；
- `X_ptr` 是 `f32 [h0,h1]=[2,64]` inout：即使关闭常规 gradient write，Source 的
  ignore-index early return 仍可能清零 X；`Y_ptr` 是 `i64 [h0]` input，符合 Liger upstream
  `torch.long` target；`loss_ptr` 是 `f32 [h0]` output；关闭的 weight/z-loss/accuracy/
  predicted-token carrier 保留为 inactive input，不伪造成 output；
- 当前 target valuation 全为 0、`ignore_index=-100`，两行均执行有效 CE，非退化；
  `n_cols↔h1`，grid 实现 `h0`。`n_non_ignore` 是逻辑值而非 shape identity；
- 项目 CUDA PK 的 `HAS_GRADIENTS=true + i32 target` 是另一个 concrete specialization，
  不能覆盖本 case；
- evidence：`python/intentir/verify/workload_adapters.py:1305`、selected installed
  `liger_kernel/ops/cross_entropy.py:27`、
  `/home/kingdom/phdworks/Dataset/Liger-Kernel/test/transformers/test_cross_entropy.py:89`。

### 2.6 GeGLU — `PROFILE_READY`

- current direct symbol 明确是 tanh approximation：`a,b` 为 `f32 [h0,h1]=[2,127]`
  input，`c` 为同 shape output；`stride=127:i64`，semantic `n_cols↔h1`，C 为
  `BLOCK_SIZE=128`，grid 实现 `h0`；
- 历史 `corpus/bulk/geglu/intent.json` 来自 FlagGems 且是 erf GELU，只能保留为历史资料，
  不得复用为本 Liger tanh Source 的 J；旧 65536x256 capture 也不是本次 valuation；
- evidence：`python/intentir/verify/workload_adapters.py:1377`、selected installed
  `liger_kernel/ops/geglu.py:23`。

Shape ledger：`h0` 是 row identity；`h1` 与 semantic `n_cols` 绑定，均可参数化。

### 2.7 LayerNorm — `PROFILE_READY`

- ordered buffers：`Y` output、`X/W/B` input、`Mean/RSTD` output，boundary 均为
  `f32`；`X/Y=[h0,h1]=[2,127]`、`W/B=[h1]`、`Mean/RSTD=[h0]`；
- runtime row strides、`n_cols↔h1`、`eps=1e-6`；C 为 `BLOCK_SIZE=128`，grid 实现
  `h0`；observable outputs 为 `Y,Mean,RSTD`；
- evidence：`python/intentir/verify/workload_adapters.py:1637`、selected installed
  `liger_kernel/ops/layer_norm.py:26`、`python/tests/test_layer_norm_pk.py`。

Shape ledger：`h0` 是 row identity；`h1` 与 runtime `n_cols` 绑定，均可参数化。

## 3. 本波准入裁决

Ordinals 1--7 均可进入 fresh SCRC。RoPE 的 primary specialization 修正只改变 Harness 的
concrete boundary dtype/stride，没有新增 J relation、Kernel-specific prompt 或 Replay recipe。
任何 Gate1/Gate2 失败均在 P0 之后按 Task14 failure tree 归责，不回头用历史 J、
backend recipe 或 sibling candidate 补答案。
