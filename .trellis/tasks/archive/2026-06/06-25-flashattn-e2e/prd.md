# Epic: FlashAttention end-to-end on 5090D

## Goal

Walk the full FlashAttention pipeline end to end and make the kernel actually
execute on the local **RTX 5090 D**, then **retune** and compare **vs Triton**:

```
source flash_attn_triton.py  --[front-end: LLM extract / hand-author]-->  IntentIR JSON
  --[1:1]-->  intent MLIR dialect
  --[verify: interpreter + differential vs source golden]-->  contract
  --[back-end: lower intent -> linalg/scf/vector -> gpu -> nvvm -> cubin]-->  GPU kernel
  --[execute on 5090D]-->  output  ==(tolerance)==  Triton output
  --[retune Layer C schedule]-->  faster  --vs-->  Triton latency
```

## Confirmed environment

- RTX 5090 D (driver 580.95.05, 32GB, sm_120 Blackwell); torch 2.10+cu130, triton 3.6.0.
- CUDA 12.8 toolkit at /usr/local/cuda-12.8 (ptxas). MLIR 20.1.8 full GPU lowering
  (`gpu-lower-to-nvvm-pipeline`, `gpu-module-to-binary`, NVPTX target).
- Source: /home/kingdom/phdworks/Dataset/flash-attention/flash_attn_triton.py.

## Two parallel tracks

- **Track A (front-end / IR / interpreter)**: ops (loop+carry, contraction, reduce, map),
  verifiers, reference interpreter, FlashAttention IntentIR (hand-authored from worked-example +
  LLM extraction), differential vs source golden. → subtask `interp-keystone` (first slice).
- **Track B (back-end / lowering / GPU)**: intent → linalg/scf/vector → gpu → nvvm → cubin →
  execute on 5090D; de-risk with a trivial op first (sm_120/ptxas/runtime).
- **Rendezvous**: the reference interpreter is the oracle for BOTH lift (source≡IntentIR) and
  lowering (IntentIR≡GPU output) correctness.

## Definition of done

1. A FlashAttention IntentIR lowers to a GPU kernel that **executes on the 5090D**.
2. Its output matches Triton's flash-attn within tolerance (correctness).
3. **Retune** over Layer C schedule params measurably changes/improves latency.
4. A **vs-Triton** report (latency / TFLOP-s ratio) on matched shapes.

> Intermediate executable milestones are expected and acceptable (correct-but-unfused before fused).
> Detailed critical path + milestone task tree: filled from the roadmap workflow, then split into
> subtasks under this epic.

## Subtasks

- `06-25-verif-arch` — layered verification architecture into spec/verification.
- `06-25-interp-keystone` — reference interpreter + ContractionOp verifier + lit harness.
- (more milestone subtasks added from the roadmap.)
