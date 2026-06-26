The epic already exists with the same framing. The synthesis task wants me to produce the structured markdown deliverable. I have everything verified. Let me produce the final synthesis directly.

# FlashAttention on RTX 5090 D — Synthesis: Critical Path, Tracks, Task Tree, Risks

## What is already de-risked (verified across the four lenses, do not re-litigate)

- **Triton baseline runs on sm_120 today:** `flash_attn_triton.py` forward = correct (max abs err 2.0e-4 vs fp32 SDPA golden), 0.308 ms / 13.9 TFLOP/s at B2/H8/S1024/D64. The vs-Triton baseline is **live**. (Caveat: it is one fixed non-autotuned config — BLOCK=128, num_stages=1, num_warps=4 — so all ratios are "vs this fixed config," stated honestly.)
- **sm_120 codegen + runtime is PROVEN:** an MLIR-20 GPU kernel was lowered to `.target sm_120a` PTX and JIT-launched via the CUDA Driver API (`cuModuleLoadData`/`cuLaunchKernel` through `cuda-python`) to a **numerically correct result on the 5090 D**. No sm_90a fallback needed. `libmlir_cuda_runtime.so` is absent but **deliberately sidestepped** (driver-API launch on torch pointers), not a wall.
- **A reference design exists:** `/home/aii-works/5090D/intentir_remote_20260302/` already shipped an executable flash-attn PTX (`.version 8.7 / .target sm_120a`) through a full intent→…→PTX pipeline. Port, don't invent.
- **M1 front-end artifact exists and round-trips:** `/home/kingdom/phdworks/TianchenIR/test/flashattention_fwd.intent.json` (scale-reconciled, 16 ops, validates through `model.py`).

So the long-pole risk was never the toolchain. **The long pole is the authoring work between the JSON contract and an executable cubin: the missing dialect ops (`loop`+carry, `map_unary`, broadcast, DPS outs), the `intent→linalg` lowering patterns, and the reference interpreter.**

---

## 1. THE CRITICAL PATH (single ordered sequence of executable checkpoints)

Each milestone is a runnable checkpoint. **[SPIKE]** = throwaway de-risking probe; **[BUILD]** = durable product code. The path is ordered to reach "something correct executes on the 5090 D" as fast as possible (M5), then thickens toward fused flash-attn.

| # | Milestone | Type | What runs | What proves it (DoD check) | Status |
|---|---|---|---|---|---|
| **M0a** | Triton baseline + golden | [SPIKE→keep numbers] | `flash_attn_triton.py` fwd + fp32 SDPA golden; save Q,K,V,O to durable `.npz` | assert Triton ≈ fp32-golden within fp16 tol; npz exists; latency/TFLOP-s logged | **DONE** |
| **M0b** | sm_120 toolchain spike | [SPIKE] | hand-written MLIR GPU kernel → sm_120a PTX → `cuModuleLoadData`/`cuLaunchKernel` | correct result on `cap (12,0)` 5090 D | **DONE** |
| **M1** | Hand-authored IntentIR JSON | [BUILD] | `model.py` round-trip of `flashattention_fwd.intent.json` | JSON↔model lossless; scale op present; carry `[m,l,Oacc]` | **DONE** |
| **M2** | numpy reference interpreter (NO loop) | [BUILD] | walk naive-attention IntentIR graph (`contraction/reduce/map_unary/map_binary`) vs M0a npz in `python/intentir/verify/` | interpreter-O ≈ golden-O within fp16 tol (fp32 accum) | TODO |
| **M3** | numpy interpreter, **fused loop form** | [BUILD] | run the online-softmax recurrence (`loop`+carry, Kj/Vj slicing, tail tile) over M1 JSON vs npz | fused-O ≈ golden-O AND ≈ M2-O (triangulation); tail-tile case passes | TODO |
| **M4** | Port runtime+pipeline scaffolding | [BUILD] | port `cuModuleLoadData` launcher + `downstream_cuda_std_llvm` pass string from prior art into `tianchenir/` | launcher runs the M0b PTX from inside tianchenir | TODO |
| **M5** | **FIRST INTENT→CUBIN→5090D** (elementwise add) | [BUILD] | author `map_binary→linalg.generic`; intent-opt(to-linalg)→bufferize→linalg-to-parallel-loops→gpu→nvvm→PTX→launch | add tensor from **real intent IR** correct vs numpy on 5090 D | **← minimum first executable (§5)** |
| **M6** | Single matmul from intent | [BUILD] | author `contraction→linalg.matmul` (einsum-subscript parser) | matmul matches torch on 5090 D (naive/global-mem acceptable) | TODO |
| **M7** | Softmax from intent | [BUILD] | `reduce→linalg.reduce` + `map_unary→exp` via `__nv_exp2f` rewrite + row-broadcast `map_binary` | softmax matches torch on 5090 D; exp-intrinsic path exercised | TODO |
| **M8** | Add `loop`/`yield`/broadcast/DPS to **dialect** (TableGen) | [BUILD] | M3 graph round-trips through real `intent` MLIR (not just python model) | `intent-opt` parses/verifies loop+carry; contraction verifier+shape-inference live | TODO |
| **M9** | **UNFUSED flash-attn on 5090 D** | [BUILD] | host-side loop over Sk launching per-tile kernels (QK^T→rowmax/exp/rowsum→P@V→rescale-accumulate) | O matches M0a Triton golden within atol/rtol 1e-2 on 5090 D | **← certified-executable target** |
| **M10** | Retune grid (Layer C) | [BUILD] | freeze A/B; sweep `tile∈{32,64,128}²ₓ`, `num_warps∈{2,4,8}`, `num_stages∈{1,2,3}` via Layer-C override→recompile→time | retuned-best beats freeze-tiling, each gated correct vs fp32; report ratios | TODO |
| **M11** | vs-Triton report | [BUILD] | same-process, same-tensors, shape sweep (S∈{512,1k,2k,4k}, D∈{64,128}, causal on/off) | table: latency, TFLOP-s, IntentIR/Triton ratio, freeze/retune speedup | TODO |
| **M12** | **FUSED flash-attn** (stretch) | [BUILD, research-grade] | fuse inner sequence into one `gpu.func`, shared-mem tiles from Layer C | one fused kernel matches golden; approaches Triton TFLOP-s | STRETCH |

**The cut line:** M9 (unfused) + M10/M11 (retune + vs-Triton) is a **complete, defensible deliverable** even if M12 (fusion auto-codegen) slips. Plan M9 as the deliverable, M12 as stretch.

---

## 2. THE TWO PARALLEL TRACKS + RENDEZVOUS

```
TRACK A — Front-end / IR / Interpreter (the ORACLE)        TRACK B — Back-end / Lowering / GPU (the EXECUTABLE)
─────────────────────────────────────────────             ──────────────────────────────────────────────────
M0a golden harness + .npz  ──────────────┐                 M0b sm_120 PTX spike (DONE)
M1 hand-authored JSON (DONE) ────────────┤                 M4 port runtime + pipeline scaffolding
M2 numpy interp, naive (no loop)         │                 M5 intent map_binary → cubin  (add)
M3 numpy interp, fused loop+carry        │                 M6 intent contraction → cubin (matmul)
                                         │                 M7 intent reduce+map_unary → cubin (softmax)
M8 dialect: loop/yield/broadcast/DPS  ◄──┴── shared spec   M8 (dialect work serves BOTH tracks)
   contraction verifier + shape-inf                        M9 unfused flash-attn cubin on 5090 D
                                                           M10 retune  ·  M11 vs-Triton  ·  M12 fused
```

- **Track A produces the contract + the oracle.** The JSON is the single source; the numpy interpreter (M2/M3) is the correctness oracle that exists **before any GPU code**, closing the lift loop (source ≡ IntentIR) with zero back-end.
- **Track B consumes the JSON contract** and produces the executable (M5→M9), diffing every kernel against Track A's `.npz` golden.
- **RENDEZVOUS 1 (spec, ~M3/M8):** the dialect ops Track A's loop form requires (`loop`+carry, `map_unary`, broadcast attr, DPS `outs`, contraction verifier/shape-inference) are exactly the ops Track B must lower. **M8 is a shared TableGen deliverable** — author it once, both tracks consume it. This is the critical-path coupling: Track B's M9 cannot start until these ops exist.
- **RENDEZVOUS 2 (the oracle, M9):** the reference interpreter is the oracle for **both** directions — lift (source ≡ IntentIR, Track A) and lowering (IntentIR ≡ GPU output, Track B). The `.npz` is the shared fixture so fp-nondeterministic reductions never cause spurious cross-track mismatch.

**Parallelism opportunity:** Track A (M2→M3) and Track B's spikes/scaffolding (M4→M7, which only need today's 3 ops) run **concurrently**. They serialize only at M8 (shared dialect work) → M9.

---

## 3. CONCRETE TRELLIS TASK TREE

Epic `06-25-flashattn-e2e` **already exists** (verified, same framing). Create the milestone subtasks under it via `task.py create … --parent .trellis/tasks/06-25-flashattn-e2e`. Names below are ready for `--slug`.

**Epic:** `flashattn-e2e` — *FlashAttention executes on 5090 D + retune + vs-Triton.* DoD: M9 + M10 + M11 all green.

| Slug | One-line PRD intent | Definition-of-done (executable check) |
|---|---|---|
| `golden-harness` | Seeded Triton fwd + fp32 SDPA golden, save Q/K/V/O to durable npz | assert Triton≈golden fp16-tol; `flashattn_golden.npz` exists | **DONE** |
| `sm120-toolchain-spike` | Prove MLIR-20 GPU kernel → sm_120a PTX → driver-API launch on 5090 D | hand kernel returns correct result on cap (12,0) | **DONE** |
| `handauthor-intentir-json` | Hand-author scale-reconciled FlashAttention IntentIR JSON | round-trips through `model.py`; carry `[m,l,Oacc]` | **DONE** |
| `numpy-interp-naive` | numpy reference interpreter for naive-attention graph (no loop) | interp-O ≈ npz-golden fp16-tol, fp32 accum |
| `numpy-interp-fusedloop` | Extend interpreter to loop+carry online-softmax; Kj/Vj slice; tail tile | fused-O ≈ golden AND ≈ naive-O; tail-tile passes |
| `port-cuda-runtime` | Port `cuModuleLoadData` launcher + `downstream_cuda_std_llvm` pass string into tianchenir | launcher runs M0b PTX from tianchenir |
| `intent-add-to-cubin` | Author `map_binary→linalg.generic`; full pipeline to cubin | add tensor from real intent IR correct on 5090 D |
| `intent-matmul-to-cubin` | Author `contraction→linalg.matmul` + einsum-subscript parser | matmul matches torch on 5090 D |
| `intent-softmax-to-cubin` | `reduce→linalg.reduce` + `map_unary→exp(__nv_exp2f)` + row-broadcast | softmax matches torch on 5090 D |
| `dialect-loop-broadcast-dps` | Add `loop`/`yield`/broadcast attr/DPS outs + contraction verifier+shape-inf to TableGen | M3 graph round-trips through real `intent` MLIR; verifiers fire |
| `unfused-flashattn-5090d` | Host-loop per-tile kernels for online-softmax flash-attn | O matches Triton golden atol/rtol 1e-2 on 5090 D |
| `retune-layerc` | Freeze A/B; grid tile×warps×stages via Layer-C override→recompile→time | retuned-best beats freeze, all correctness-gated |
| `vs-triton-report` | Same-process shape sweep, IntentIR vs Triton vs fp32 golden | table: latency/TFLOP-s/ratio/freeze-vs-retune speedup |
| `fused-flashattn` | Fuse inner sequence into one gpu.func, shared-mem tiles from Layer C | fused kernel matches golden; approaches Triton (STRETCH) |

Example creation command:
```
python .trellis/scripts/task.py create "numpy interpreter — fused loop+carry online-softmax" \
  --slug numpy-interp-fusedloop --parent .trellis/tasks/06-25-flashattn-e2e --priority P1
```

---

## 4. TOP RISKS / UNKNOWNS + cheapest spike to retire each

| # | Risk | Status | Cheapest spike to retire |
|---|---|---|---|
| R1 | **sm_120 ptxas/NVPTX codegen + GPU runtime** | **RETIRED** | M0b already ran intent-style MLIR → sm_120a PTX → `cuModuleLoadData` → correct on 5090 D. No sm_90a fallback. |
| R2 | **MLIR CUDA runtime lib missing** (`libmlir_cuda_runtime.so` absent) | **RETIRED (path choice)** | Don't build it. Launch cubin/PTX via CUDA Driver API on torch pointers (proven in M0b). Also gives apples-to-apples timing. |
| R3 | **loop-with-carry lowering** (the genuinely new dialect+lowering work) | OPEN, on critical path | Two-stage spike: (a) numpy interp loop (M3) proves *semantics* with zero MLIR; (b) hand-write the target `scf.for` + `iter_args` MLIR for one tiny shape, lower to PTX, run — proves the lowering *shape* before authoring the auto-pattern. |
| R4 | **fused-vs-unfused auto-codegen** (linalg→gpu won't fuse flash by itself) | OPEN, accepted | Don't spike fusion first. M9 unfused (host loop + per-tile kernels) is the accepted deliverable; M12 fusion is stretch with hand-authored shared-mem. Cheapest check: confirm M9's per-tile sequence is correct before any fusion attempt. |
| R5 | **MLIR 20 vs bundled 21.1.8 pass-set delta** — prior art's flash PTX may be 21-emitted | OPEN | Spike: run the *full* flash lowering pass string on system MLIR-20 against a tiny shape (M9). Keep `INTENTIR_MLIR_TOOLCHAIN_ROOT` escape-hatch to bundled 21.1.8 if a 20-specific pass gap appears. |
| R6 | **Kj/Vj scanned-input binding undefined** + **missing √D scale** in worked-example JSON | Scale FIXED in M1; Kj/Vj OPEN spec hole | Resolve in M3: add explicit scanned-input loop binding (`{name:Kj,from:K,axis:Sk}`) or a `slice` op in the body. Scale already reconciled in `flashattention_fwd.intent.json`. |
| R7 | **bare-ptr memref ABI mismatch** (silent segfault on kernel-arg marshaling) | KNOWN, mitigated | Lock `convert-gpu-to-nvvm="use-bare-ptr-memref-call-conv"` from M5; the M0b probe already hit and fixed this class. |
| R8 | **fp16 tolerance calibration** (online-softmax fp16 + nondeterministic rowsum) | OPEN | fp32-accumulate in reference + interpreter; compare O in fp16 at atol 2e-2/rtol 1e-2; pin the seeded `.npz` so all consumers share tensors. |
| R9 | **Config drift** (seqlen not ×128 / causal / bias silently change the kernel's function) | KNOWN | Pin maskless-path config in the harness: non-causal, no-bias, fp16, d=64, seqlen×128 (forces EVEN_M/N/HEADDIM). |

---

## 5. THE MINIMUM FIRST EXECUTABLE ON GPU

**`intent-add-to-cubin` (M5): an elementwise tensor add, sourced from real `intent.map_binary` IR, executing on the 5090 D.**

This is the smallest thing that proves the *whole spine* `intent → linalg → gpu → nvvm → cubin → 5090 D` with **product code**, not a spike:

1. Author the single conversion pattern `intent.map_binary {fn: add} → linalg.generic` (elementwise) — the smallest piece of the only new lowering machinery.
2. Pipeline (all stock MLIR-20, proven present): `intent-opt`(to-linalg) → `one-shot-bufferize` → `convert-linalg-to-parallel-loops` → `gpu-map-parallel-loops` + `convert-parallel-loops-to-gpu` → `gpu-kernel-outlining` → the proven NVVM pass string (`convert-gpu-to-nvvm=use-bare-ptr-memref-call-conv …`) → `mlir-translate` → `llc -mcpu=sm_120a` → PTX.
3. Launch via the ported `cuModuleLoadData` driver-API launcher (M4) on torch tensor pointers.
4. **DoD:** output tensor equals numpy `a+b` on the 5090 D.

Why this and not matmul/softmax first: it uses **only today's existing dialect op** (`map_binary`), needs **no einsum parser, no reduce, no exp-intrinsic, no loop** — isolating the one unproven thing (the intent→linalg→gpu→cubin authoring + ABI marshaling) from every other variable. Everything past M5 (matmul, softmax, loop) reuses the identical back-end tail; only the front conversion pattern changes. It is the literal "hello world" that converts the de-risked toolchain into a working product pipeline.

---

### Pointers (absolute)
- Contract artifact (DONE): `/home/kingdom/phdworks/TianchenIR/test/flashattention_fwd.intent.json`
- Interpreter stub to build (M2/M3): `/home/kingdom/phdworks/TianchenIR/python/intentir/verify/__init__.py`
- Dialect to extend (M8): `/home/kingdom/phdworks/TianchenIR/include/Intent/IntentOps.td`, `/home/kingdom/phdworks/TianchenIR/lib/Intent/IntentOps.cpp`
- Reference design to port (M4–M9): `/home/aii-works/5090D/intentir_remote_20260302/intent_ir/mlir/pipelines/downstream_cuda_std_llvm.yaml`, `.../backends/cuda/runtime.py`, `.../frontends/cuda/runtime.py`, `.../scripts/cuda_backend_smoke.py`; proven artifact `.../artifacts/full_pipeline_verify/flash_attention2d.intentir.intentdialect.downstream_cuda_std_llvm.kernel.ptx`
- Triton baseline (import by file path): `/home/kingdom/phdworks/Dataset/flash-attention/flash_attn/flash_attn_triton.py` (fwd `_flash_attn_forward` ~L815)
- Epic + task.py: `/home/kingdom/phdworks/TianchenIR/.trellis/tasks/06-25-flashattn-e2e/`, `/home/kingdom/phdworks/TianchenIR/.trellis/scripts/task.py`
- Toolchain: `/usr/lib/llvm-20/bin/` (GPU passes confirmed); ptxas `/usr/local/cuda-12.8/bin/ptxas` (export `PATH=/usr/local/cuda-12.8/bin:$PATH`).