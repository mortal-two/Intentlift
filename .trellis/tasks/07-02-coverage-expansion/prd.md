# Coverage expansion: 191 FlagGems + 38 Triton ops via automated extraction

## Goal

Take IntentIR from "29 high-perf kernels lowered" to **broad coverage of the whole op universe** —
**191 FlagGems op modules + 38 hand-written Triton kernels** (the 29 high-perf are already 100% lowered).
The certified lifting loop (LLM → atomic IntentIR → C++ lowering → sm_120a → verify vs the real kernel)
is proven on 29+15; this task **scales it to the universe** and turns the "≈150 ops lower for free" hypothesis
into MEASURED coverage, isolating the genuinely-new lowering mechanisms from the pure-extraction bulk.

## What I already know (from the 2026-07-02 audit+coverage workflow)

- **The dominant residual is EXTRACTION, not lowering.** Of ~176 remaining FlagGems ops: **~120 need extraction only**
  (the op class + an existing pass already lower the graph shape — the only work is the per-kernel LLM lift,
  sometimes + a one-line scalar-fn/NVVM entry), **~56 need a new lowering mechanism**. Of 38 Triton refs: ~30 extraction-only, ~8 net-new.
- **FlagGems universe = 191 distinct op modules** (169 `ops/` + 22 `fused/`; 315 public entrypoints; 1652 `@triton.jit` variants). 15 already extracted (the shipped corpus).
- **The ~56 new-mechanism ops collapse into 9 clusters** (ranked by leverage; top 3 = ~40 ops):
  1. gather/scatter/data-dependent addressing (~15): gather, scatter, scatter_add_, index_*, masked_scatter/select, embedding, nllloss, reshape_and_cache, concat_and_cache_mla, moe_sum, apply_repetition_penalties. **No op, no pass today.**
  2. structural layout concat/pad/tile (~14): cat, stack, hstack/vstack, pad, repeat, tile, repeat_interleave, diag(_embed), diagonal, triu, flip. (Only broadcast/reshape implicit today.)
  3. RNG/philox (~11): dropout, rand(n)(_like), normal, uniform, multinomial, randperm, exponential_. (Stateful seeded RNG → seed/tolerance verification.)
  4. order-stats sort/topk/arg-reduce (~9): sort, topk, argmax/argmin, quantile, unique, nonzero, cummax/cummin, topk_softmax. (reduce lacks arg*; scan lacks max/min; no sort primitive.)
  5. convolution/im2col (4); 6. complex dtype (4); 7. interpolation/resize (3); 8. pooling/windowed-reduce (2); 9. dynamic-shape output (nonzero/unique/masked_select — data-dependent extent).
- **Near-free unblocker**: `cast`/`to` dtype-convert has NO fn today (add to map_unary); plus ~22 one-line scalar-fn/NVVM entries (cos/sin/tan/atan/erf/exp2/celu/elu/softplus/log_sigmoid/clamp/threshold/isnan/logical_not/bitwise_not/nan_to_num…) unlock ~22 unary ops.
- **Existing base**: `python/intentir/extraction/` pipeline (client.py OpenAI responses gpt-5.5 / prompt.py op-vocab+synthetic few-shot / extract.py validate+smoke-interpret+bounded-repair / data/{corpus_specs,run_corpus}); 15-kernel corpus with provenance; 11 C++ lowering passes (row-reduction, elementwise, contraction wmma+mma.sync, flash fwd/bwd, splitK ×2, loop, + LowerGpuMathToNVVM); `interpret.py` reference.
- **Structural facts to respect**: `ConvertIntentToLinalg` registers only {MapUnary,MapBinary,Reduce,Contraction,Loop} — Select/Iota/Scan lower ONLY inside the fused GPU passes (no generic CPU fallback). The extraction verify path today = `interpret(JSON)` vs the real `flag_gems.<op>` output (rel ≤ 1.9e-7), GPU-dependent.

## Assumptions (temporary — validate)

- The "extraction-only ≈150" count assumes existing-pass composability, empirically proven ONLY for the 15 corpus shapes (pure map / pure row-reduce / pure contraction) + 29 hand-authored `.mlir` compositions. **A novel op-graph boundary (e.g. elementwise→scan→elementwise) may need a small new dispatch, not just extraction.** ← the #1 thing to measure.
- LLM extraction is stable enough at batch scale (the bounded-repair loop fired only once across the 15-corpus; untested at 150-op scale).

## Open Questions (Blocking / Preference only)

- MVP scope boundary: automated-extraction-pipeline + measurement gate (defer the 9 new-mechanism clusters), or also front-load the top cluster (gather/scatter)?
- Verification standard at scale: numeric vs the real `flag_gems.<op>` (as today) for all, and how to handle RNG (seed) + dynamic-shape outputs?
- Definition of "supported": extracted+verified (interpret==real kernel) enough, or must it also lower to sm_120a cubin (the full loop)?

## Requirements

- [ ] **Automated extraction+lowering+verify harness**: op module → LLM extract IntentIR → run the C++ pass pipeline → sm_120a cubin → **per-category verify** vs the real kernel. Batchable + resumable over the full universe; emits a coverage manifest {op, family, extracted?, lowered?, verified?, rel/stat, failure-class}.
- [ ] **Per-category verification** (user decision): deterministic → numeric rel<1e-3 vs the real `flag_gems.<op>`/triton kernel; RNG → matched-seed/philox bit-exact where feasible, else statistical (mean/var/moments + distribution); dynamic-shape (nonzero/unique/masked_select) → value-set / sorted comparison.
- [ ] **Failure triage**: distinguish extraction-failure (bad/incomplete lift) from lowering-failure (pass can't handle the op-graph = a composability gap needing new dispatch).
- [ ] **All 9 new-mechanism clusters** delivered end-to-end (new IntentIR op + verifier + emit_mlir + interpret + the GPU lowering pass + verify), in leverage order.
- [ ] Full universe (191 FlagGems + 38 Triton) extracted + lowered + verified; coverage manifest ≈100% (or every miss explicitly classified + tracked).

## Acceptance Criteria

- [ ] **C0** unblockers: `cast`/`to` fn + the ~22 scalar-fn/NVVM entries landed, each lit/interp-checked; the ~22 unary ops verify.
- [ ] **C1** harness + stratified pilot (20–30 ops across families) run; composability-failure rate MEASURED (extraction-fail vs lowering-fail split); manifest schema committed.
- [ ] **C2** the ~150 extraction-only bulk extracted+lowered+verified; the "≈150 lower-for-free" hypothesis confirmed/corrected with real numbers.
- [ ] **C3–C8** each new-mechanism cluster: the op/pass built, its ops verified per-category, manifest updated. (C3 gather/scatter · C4 layout · C5 RNG · C6 sort/topk/arg · C7 conv/complex/interp/pool · C8 dynamic-shape.)
- [ ] Final: coverage manifest committed; 191+38 all accounted (verified or explicitly-classified-miss); no per-kernel recognizer in lowering; Dataset/site-packages/refs stay read-only.

## Milestone plan (C0 → C8, difficulty-ordered, M1→M9-style)

| C | new capability | ops unlocked | verify |
|---|---|---|---|
| **C0** | `cast`/`to` fn + ~22 scalar-fn/NVVM unary entries (near-free) | ~23 unary | numeric |
| **C1** | automated extract→lower→verify harness + coverage manifest + failure triage; stratified pilot + measure composability-fail rate | infra (20-30 pilot) | numeric |
| **C2** | scale the harness over the extraction-only bulk | ~150 (elementwise/reduce/norm/softmax/blas/init/fused) | numeric |
| **C3** | gather/scatter/data-dependent addressing (new op+pass) | ~15 (gather,scatter,index_*,embedding,nllloss,kv-cache,moe_sum) | numeric |
| **C4** | structural layout concat/pad/tile (new op+pass) | ~14 (cat,stack,pad,repeat,tile,diag,triu,flip) | numeric |
| **C5** | RNG/philox (new op+pass, stateful seed) | ~11 (dropout,rand(n),normal,uniform,multinomial) | matched-seed / statistical |
| **C6** | order-stats sort/topk/arg-reduce (arg* reduce, scan max/min, sort) | ~9 (sort,topk,argmax/min,quantile,cummax/min) | numeric / value-set |
| **C7** | conv/im2col + complex dtype + interpolation + pooling | ~13 | numeric |
| **C8** | dynamic-shape output | ~3 (nonzero,unique,masked_select) | value-set |

## Definition of Done

- Tests added (extraction + verify per op batch; lit for new fns); lint/CI green; coverage manifest + provenance committed; no per-kernel recognizer in lowering; Dataset/site-packages stay read-only.

## Decision (ADR-lite) — MVP scope

**Context**: the ~150 extraction-only vs ~56 new-mechanism split offered a phased "measure-first, defer new-mechanism" MVP.
**Decision (user, 2026-07-02)**: **全都做,端到端** — full scope, NOT phased-defer. The task delivers the WHOLE universe
(191 FlagGems + 38 Triton), end-to-end (extract → lower → sm_120a cubin → verify), INCLUDING all 9 new-mechanism clusters.
Focus is **coverage breadth** ("只要做覆盖面的增加"), so per-op **correctness/bit-faithful is the gate, NOT 实测 ≥ perf**
(compute-bound perf is already at the established sm_120a hardware ceiling; new memory-bound ops ride the proven parity path).
**Consequences**: this is a big multi-phase task (like the 29-kernel one). Structure it phased-by-difficulty like M1→M9:
Sprint 0 (cast fn + ~22 scalar-fn/NVVM entries) → automated extraction pipeline over the extraction-only bulk (~150) →
the 9 new-mechanism clusters in leverage order (gather/scatter → layout → RNG → sort/topk → conv → complex → interp →
pooling → dynamic-shape). Each phase: new op/mechanism (if any) + batch-extract + verify + coverage-manifest update.

## Out of Scope (explicit)

- Compute-bound perf-closing (established as an sm_120a hardware ceiling last task; this task is coverage/correctness, not perf).
- The LLM→JSON lift authorship for loop/flash kernels beyond what the pipeline auto-produces.
- Non-GPU backends / multi-GPU.

## Technical Notes

- Audit+coverage evidence: `/tmp/claude-1006/.../tasks/ws3g2nwog.output` (perf HONEST; coverage 191 flaggems + 38 triton; 9 clusters).
- FlagGems: `~/.local/lib/python3.10/site-packages/flag_gems/{ops,fused}/`. Triton refs: `refs/intentir/kernels/triton/ops/` (38 modules, read-only).
- Extraction recipe + gotchas: memory `tianchenir-extraction-corpus`, `tianchenir-backend-recipe`.
