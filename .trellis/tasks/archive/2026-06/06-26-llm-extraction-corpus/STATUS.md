# LLM extraction corpus + multi-backend/frontend — status (2026-06-26)

## The headline: the JSON is now provider-agent-generated, end to end

`我们的 json 是利用 provider agent 生成的吗?` — **是的, now real.** Before today the
IntentIR JSON was hand-authored. Now `python/intentir/extraction/` calls the live
provider (gpt-5.5, OpenAI responses API) and lifts real library kernels.

## Pipeline proven for every kernel: extract → IR → verify → lower

```
FlagGems/CUDA source --(gpt-5.5)--> IntentIR JSON --validate-->
  interpret(JSON) == the ACTUAL kernel  [dynamic certificate, rel <= 1.9e-7]
  --emit--> intent dialect MLIR --convert-intent-to-linalg--> linalg  [no residual intent.*]
  --(GPU: cubin@sm_120a)  AND  (CPU: LLVM JIT via mlir-runner)
```

## Deliverables

| item | status |
|---|---|
| Extraction client/prompt/driver (live provider, key never logged) | DONE |
| **15 FlagGems Triton kernels** extract+verify+lower | DONE 15/15 (`corpus/REPORT.md`) |
| Provenance per kernel (model, prompt sha, source, rel-err) | DONE (`corpus/<k>/provenance.json`) |
| Deterministic re-verify of committed JSON (no network) | DONE (`run_corpus saved`, `test_corpus.py`) |
| Dialect+interpreter vocab: tanh/sigmoid/pow/mean + const op | DONE |
| **vs-Triton performance** (latency/TFLOP-s) | DONE (`corpus/PERF.md`, `bench/perf.py`) |
| **Multi-backend**: same IR JIT-executes on CPU (6 kernels) | DONE (`backend/cpu.py`, `test_cpu_backend.py`) |
| **Multi-frontend**: CUDA C++ kernels lift to same IR (3) | DONE (`extraction/cuda_frontend.py`, `corpus/cuda/`) |

## Honest scope

- The verified kernels run f32; `cast` is treated as a no-op (precision casts out
  of scope). gelu uses the tanh approximation (erf has no primitive yet).
- The CUDA frontend demo verifies lifted math vs torch (nvcc absent), so it proves
  language handling, not a live CUDA run — distinct from the Triton corpus.
- Perf: our kernels are correctness-first; Triton is ~570–980× faster on flash.
  Parity needs a warp-tiled Layer-C retune; NOT claimed.
- RVV backend is documented as the next `linalg→vector→llvm` tail, NOT built.

## Tests
41 → 49 pytest (+2 gated live-LLM skips), 17 lit. All green.
