# Scalar Gate2 shape/runtime audit

## Current bottleneck

The scalar path is semantically correct (`IntentProgram -> Intent MLIR ->
linalg/loops -> LLVM CPU`) and has no CUDA/NumPy interpreter fallback, but the
current CPU wrapper embeds every concrete input as an MLIR `dense` constant.

GeGLU capture shape is `[65536,256]`: 16,777,216 fp32 elements or 64 MiB per
input before textual IR expansion and compiler copies. Two random inputs already
represent 128 MiB raw; each seed changes module text and prevents compile reuse.

Multi-output readback currently invokes a result wrapper per output, which is
correct but not scalable for TVD/LayerNorm/KLDiv campaigns.

## Recommended base capability

Implement a runtime memref scalar ABI:

```text
compile once: IntentProgram/spec/static specialization -> host executable
run many:     runtime input/output memref descriptors + scalar dimensions
```

Requirements:

- concrete values are runtime buffers, never textual dense constants;
- compile cache key binds scalar spec/program/dtype/rank/static specialization,
  not seed/input bytes;
- one invocation writes every output/inout buffer;
- replay and scalar receive isolated instances derived from one canonical input
  bundle with exact dtype/shape/stride/byte/alias digest;
- execution remains real MLIR/LLVM CPU scalar code;
- no target backend fallback or interpreter.

## Shape profiles

Small profiles are a later budget mechanism, not a substitute for runtime ABI.
They must be declarative, kernel-name independent, ABI-valid, and digest-bound
to a specialized J/replay/scalar spec. Fixed literal shapes cannot be silently
shrunk after Gate1; if parameterization is not proven, a smaller launch requires
its own Lift/Gate1 artifact.

Recommended order:

```text
runtime memref ABI
  -> one-call multi-output
  -> compile cache
  -> canonical input/alias bundle
  -> declarative digest-bound shape profiles
  -> full-shape offline campaign
```

