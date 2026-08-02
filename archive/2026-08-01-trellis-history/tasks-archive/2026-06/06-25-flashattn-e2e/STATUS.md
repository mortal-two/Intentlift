# FlashAttention e2e — session status (2026-06-25)

## Proven end-to-end this session

**An intent-IR-sourced kernel executes on the RTX 5090 D, bit-exact (M5 DONE).**
The full Track-B spine is real: `intent` dialect → linalg → gpu → nvvm → cubin
(MLIR 20.1.8 + ptxas 12.8, `sm_120a`) → CUDA Driver-API launch on torch pointers
→ correct output (`out == a+b`, max abs err 0.0). No MLIR-21 escape hatch, no
`libmlir_cuda_runtime`.

## Milestone ledger

| M | What | Status |
|---|------|--------|
| M0a | Triton flash golden on 5090D | DONE (prior probe) |
| M0b | sm_120 toolchain spike | DONE (prior probe) |
| M1 | Hand-authored flash IntentIR JSON | DONE (+canonicalized: `map_binary`/`map_unary`, int axes, `scan` bindings) |
| M2 | numpy interpreter, naive attention | DONE — `interpret.py`; matches numpy + torch SDPA |
| M3 | numpy interpreter, fused loop+carry online softmax | DONE — tile-invariant, tail-tile, vs torch SDPA (`test_flash_interpret.py`) |
| R5 | MLIR-20 reaches sm_120a PTX for flash class | RETIRED — MLIR-20 viable; pass string captured |
| R3 | `scf.for`+`iter_args` on 5090D | GREEN — runtime loop, max abs err 0.0 |
| M5 | intent.map_binary{add} → cubin → 5090D | DONE — `compile.py`+`launch.py`; bit-exact (err 0.0) |
| M6 | intent.contraction (matmul) → reduction kernel → 5090D | DONE — executes bit-exact (err 0.0); batched lowering too |
| M7a | intent.map_unary{exp} → ex2.approx → 5090D | DONE — softmax primitive; `LowerGpuMathToNVVM` C++ pass; rel err 1.7e-7 (NO libdevice) |
| M7b | reduce (keepdims/expand_shape) + broadcast map_binary | DONE — `ReduceLowering`+broadcast `MapBinaryLowering`; lit + softmax on 5090D (rel 2e-7) |
| M9o | self-orchestrating multi-kernel launcher | DONE — `orchestrate.py::run_program`; add/matmul/softmax/attention on 5090D |
| **MA** | **naive (unfused) attention op-graph → 11 cubins → 5090D** | **DONE — `test/attention.mlir`; vs Layer-A ref rel 1.4e-6, vs torch SDPA, executes** |
| **MT** | **vs Triton flash (correctness)** | **DONE — IntentIR attn == Triton flash to rel 3.5e-7 == torch; `bench/triton_flash.py`+`bench/compare.py`** |
| **MF** | **FUSED FlashAttention (online softmax, ONE kernel) → 5090D** | **DONE — `test/flash_fused.mlir`; scf.parallel(query)×scf.for+scalar-carries(m,l,acc); rescale alpha=exp(m_old-m_new); ex2.approx exp; vs exact softmax rel 1.1e-7 across seeds/scales** |
| **M10** | **retune (schedule-invariance) on fused flash** | **DONE — `bench/flash_emit.py`; BK∈{1,2,4,8,16} bit-identical output (max disagree 1.8e-7 <1e-4) = non-binding-schedule property; `retune_sweep` sweeps+selects, all candidates correct. Perf deltas ~1% (one-thread/query) — mechanism not tuned-perf** |
| **M11** | **vs Triton (fused, multiple shapes)** | **DONE — fused == Triton flash rel 2.5e-5 (8,16,16); naive == Triton 3.5e-7; `bench/compare.py`** |
| L1 | ContractionOp + MapUnaryOp verifiers + lit | DONE |
| L7 | lit harness (`check-intent`, 4 tests) | DONE |
| spec | verification architecture (L0–L9) | DONE — `spec/verification/architecture.md` |
| **MO1** | **CERTIFIED-LIFTING check: interpret(JSON IR) == execute(intent-DIALECT MLIR)** | **DONE — `interpret(flashattention_fwd.json)` vs `run_program("attention.mlir")`, rel 1.6e-7. The executed side runs `convert-intent-to-linalg` on a real intent op-graph (contraction/reduce/map_binary/map_unary), so this certifies the COMPILER'S LOWERING preserved IR semantics (not just vs-torch). `test_lifting_certificate_intent_dialect`.** |
| **MO2** | **Cross-impl + tile-invariance: interpret(JSON) == fused emitter** | **DONE (but NOT a lowering certificate) — `interpret` vs `run_fused_flash`, rel 3.9e-7 across shapes incl. GENUINE tail tiles (10%4, 12%5). The emitter emits scf+memref directly, bypassing the intent dialect, so this checks the spec is realizable + the online-softmax recurrence is tile-invariant, NOT the dialect lowering (that path is M8). `test_fused_emitter_matches_interpreter`.** |
| **M8** | **FRONT-END: `intent.loop`+`intent.yield` C++ op + JSON→intent-MLIR emitter + `intent.loop`→`scf.for` lowering** | **DONE — real TableGen ops (verifiers fire: `test/loop-invalid.mlir`); `frontend/emit_mlir.py` lifts `flashattention_fwd.json` to `intent.loop` MLIR; `LoopLowering` → `scf.for`+`extract_slice`+linalg (`test/lower-loop.mlir`). Front+back CONNECTED: the EMITTED body executes on the 5090D (host-driven trip loop, carries in CUDA buffers) and matches `interpret(JSON)` rel 1.3e-7 (`test_frontend_emit.py`). orchestrate now returns multi-output (3 carries).** |
| **spec3.3** | **Phase 3.3 spec update (the "spec" half of 两个都做)** | **DONE — `spec/lowering/index.md` (后端已打通 + 血泪规则), `spec/verification/validation.md` (动态一致性已跑通), `spec/ir/atomic-ops.md` (v1 子集端到端验证). Task statuses reconciled: 10 completed, `vs-triton-report`+`flashattn-e2e` in_progress, `dialect-loop-broadcast-dps` (M8 front-end op) still planning.** |

**Every primitive the attention graph needs now executes on the 5090D through the
real C++ dialect + compile + orchestrate stack:** map_binary (add/sub/mul/div/
max/min), contraction (QK/PV reductions), map_unary{exp} (softmax, via native
ex2.approx — no libdevice), scf.for+carry (R3, online recurrence), and generic
multi-kernel orchestration. The last unproven link (exp launchability) is retired.

## Key verified recipe (reuse)

- **Compile (durable):** `python/intentir/backend/compile.py` — `compile_intent_to_cubin()`.
- **Launch (durable):** `python/intentir/backend/launch.py` — driver API, bare-ptr.
- **ABI rule (R3):** bare-ptr conv needs **statically-shaped** memrefs; drive sizes
  by runtime scalar args. `kernel-bare-ptr-calling-convention=true`.
- **Integrated 1-pass lowering (R3):** `mlir-opt --gpu-lower-to-nvvm-pipeline=`
  `"cubin-chip=sm_120a cubin-format=isa kernel-bare-ptr-calling-convention=true"`.
- **exp gotcha (R3):** `math.exp` → `__nv_expf` extern → unlaunchable (rc 218).
  Needs the prior-art `ex2.approx`/`__nv_exp2f` rewrite. Blocks M7 softmax, not loops.
- Prior-art to port from (read-only): `/home/aii-works/5090D/intentir_remote_20260302/`.

## Done since: naive attention end-to-end + vs-Triton correctness

`intent` attention op-graph (`test/attention.mlir`) → 7 linalg.generics → 11
`sm_120a` cubins → `run_program` → executes on the 5090 D. Validated three ways:
vs Layer-A reference `softmax(Q·Kᵀ)·V` (== `interpret.py`) rel **1.4e-6**, vs
**torch SDPA**, and vs an independent **Triton FlashAttention** kernel rel
**3.5e-7**. All agree — the vs-Triton *correctness* proof (lift→lower→execute
preserved the algorithm). `bench/triton_flash.py` + `bench/compare.py`.

**Perf NOT measured / NOT claimed (honest):** bring-up kernels are naive
one-thread-per-element; and `run_program` re-parses the host module +
`cuModuleLoadData`s all 11 modules per call, so its ~140 ms wall time is
orchestration overhead, not kernel time. A real perf number needs a cached launch
plan + the fused/tiled kernel — the retune work below.

## FRONT-END now BUILT — JSON → intent dialect → executes on 5090D (M8 DONE)

The front-end leg `JSON → intent MLIR dialect` is now real and connected to the
back-end on hardware:

- **C++ dialect ops** `intent.loop` + `intent.yield` (TableGen + verifiers):
  variadic scan inputs + carry inits, a body region, `SingleBlockImplicitTerminator`.
  Verifiers fire (`test/loop-invalid.mlir`). The flash JSON round-trips through it.
- **JSON→intent-MLIR emitter** `python/intentir/frontend/emit_mlir.py`: walks the
  IntentFunction op-list generically (einsum/broadcast/reduce/keepdims shape
  inference matching `interpret.py`), folds the single-head batch dims, and emits
  two artifacts from the SAME JSON — the full `intent.loop` module and the loop
  body as an executable `func.func` (scalar `scale` as a rank-0 `tensor<f32>`).
- **`intent.loop`→`scf.for` lowering** (`LoopLowering` in ConvertIntentToLinalg):
  trip `scf.for` with tensor `iter_args`, scan inputs → `tensor.extract_slice`,
  body intent ops → linalg. `test/lower-loop.mlir`. (This is the natural tensor-
  level lowering; the *fused* device kernel is a separate, already-proven schedule.)
- **Execution = host-driven trip loop:** the EMITTED body is launched once per
  K-tile (`orchestrate.run_program`, now multi-output for the 3 carries), carries
  live in persistent CUDA buffers fed forward; final `O = Oacc/l`. This is exactly
  how a NON-fused tiled flash runs — real kernels on the 5090D, host loop control.
  Matches `interpret(JSON)` at rel **1.3e-7** (`test_frontend_emit.py`).

So "前端后端都包括" holds end-to-end: **JSON → intent-dialect MLIR (emitter) →
convert-intent-to-linalg → sm_120a cubins → execute on the 5090 D → == interpreter.**

Residual (named optimization, NOT a hole in front/back): auto-emitting the *fused*
single device kernel directly from `intent.loop` (the proven `flash_fused.mlir`
schedule) — today the fused kernel is emitted by `bench/flash_emit.py`, not by the
loop lowering. Plus the perf write-up (cached launch plan + warp-tiled schedule),
tracked under `vs-triton-report`.
