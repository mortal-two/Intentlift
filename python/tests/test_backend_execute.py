"""M5: execute an intent-IR-sourced kernel on the GPU (the back-end spine end to
end). Compiles intent.map_binary{add} to an sm_120a cubin, launches it via the
CUDA Driver API on torch device pointers, and checks bit-exact agreement with
a+b. Skips cleanly without a GPU / CUDA / build.
"""

from __future__ import annotations

import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT

_TEST = pathlib.Path(__file__).resolve().parents[2] / "test"
_ADD = _TEST / "add.mlir"
_MATMUL = _TEST / "matmul.mlir"

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def test_intent_add_executes_on_gpu() -> None:
    from intentir.backend.compile import compile_intent_to_cubin
    from intentir.backend.launch import I64Arg, PtrArg, launch, load_module

    k = compile_intent_to_cubin(_ADD)
    func = load_module(k.cubin).function(k.kernel_name)

    torch.manual_seed(0)
    a = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    b = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    out = torch.empty(8, 16, dtype=torch.float32, device="cuda").contiguous()

    # bare-ptr ABI: (index step=1, index lb=0, ptr a, ptr b, ptr out).
    # The lowering maps one block per element -> grid = output shape, block = 1.
    launch(func, (8, 16, 1), (1, 1, 1),
           [I64Arg(1), I64Arg(0), PtrArg(a.data_ptr()),
            PtrArg(b.data_ptr()), PtrArg(out.data_ptr())])

    assert (out - (a + b)).abs().max().item() == 0.0


def test_intent_exp_executes_on_gpu() -> None:
    """map_unary{exp} is the softmax primitive. The NVVM peephole rewrites
    math.exp to the native ex2.approx PTX instruction (no libdevice extern),
    proving the last unproven link on the attention path runs on hardware."""
    from intentir.backend.orchestrate import run_program

    torch.manual_seed(0)
    a = torch.randn(256, dtype=torch.float32, device="cuda").contiguous()
    out = run_program(str(_TEST / "exp.mlir"), {"a": a})
    res = next(iter(out.values()))
    ref = a.exp()
    rel = ((res - ref).abs() / (ref.abs() + 1e-6)).max().item()
    assert rel < 1e-2, rel


def test_intent_attention_executes_on_gpu() -> None:
    """Naive (unfused) scaled-dot-product attention as an intent op-graph,
    lowered to ~11 kernels and orchestrated on the 5090D. Validated against the
    Layer-A reference O = softmax(Q·Kᵀ)·V (what interpret.py computes) at a tight
    1e-3 rtol — the LIFT-vs-LOWERING translation check. This is attention
    bring-up, NOT the fused FlashAttention algorithm."""
    from intentir.backend.orchestrate import run_program

    torch.manual_seed(0)
    q = torch.randn(4, 16, dtype=torch.float32, device="cuda").contiguous()
    k = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    v = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()

    out = run_program(str(_TEST / "attention.mlir"), {"q": q, "k": k, "v": v})
    res = next(iter(out.values()))

    # Layer-A reference (== interpret.py naive attention) and torch SDPA sanity.
    s = q @ k.T
    ref = torch.softmax(s, dim=1) @ v
    rel = ((res - ref).abs() / (ref.abs() + 1e-6)).max().item()
    assert rel < 1e-3, f"vs Layer-A reference: rel={rel}"

    sdpa = torch.nn.functional.scaled_dot_product_attention(
        q[None, None], k[None, None], v[None, None], scale=1.0)[0, 0]
    assert (res - sdpa).abs().max().item() < 1e-2, "vs torch SDPA"


def test_fused_flash_executes_on_gpu() -> None:
    """THE FlashAttention milestone: the fused online-softmax forward as ONE
    kernel. scf.parallel(query) -> GPU grid; inner scf.for over keys carries the
    running max m, running sum l, and accumulator (scalar iter_args) and does the
    rescale alpha=exp(m_old-m_new), acc=alpha*acc + p*v_j — the FA recurrence.
    math.exp -> native ex2.approx. Validated vs the exact-softmax reference (==
    interpret.py Layer-A semantics) across seeds/scales: the BK=1 per-key online
    rescale agreeing with the one-shot reference is the tile-invariance property
    across the two extreme groupings."""
    import struct

    from intentir.backend.compile import lower_memref_module_to_kernels
    from intentir.backend.launch import load_module
    from intentir.backend.orchestrate import _F32Arg, _I64Arg, _PtrArg, _launch

    flash = lower_memref_module_to_kernels(_TEST / "flash_fused.mlir")[0]
    func = load_module(flash.cubin).function("flash_kernel")
    ninf = struct.unpack(">f", bytes.fromhex("FF800000"))[0]

    for seed, scale in [(0, 0.5), (1, 1.0), (2, 0.125)]:
        torch.manual_seed(seed)
        q = torch.randn(2, 2, device="cuda").contiguous()
        k = torch.randn(4, 2, device="cuda").contiguous()
        v = torch.randn(4, 2, device="cuda").contiguous()
        o = torch.empty(2, 2, device="cuda").contiguous()
        # args: [step=1, lb=0, Q, K, scale, V, Sk=4, m0=-inf, l0/acc0=0.0, O]
        _launch(func, (2, 1, 1), (1, 1, 1),
                [_I64Arg(1), _I64Arg(0), _PtrArg(q.data_ptr()), _PtrArg(k.data_ptr()),
                 _F32Arg(scale), _PtrArg(v.data_ptr()), _I64Arg(4),
                 _F32Arg(ninf), _F32Arg(0.0), _PtrArg(o.data_ptr())])
        torch.cuda.synchronize()
        ref = torch.softmax((q @ k.T) * scale, dim=1) @ v
        rel = ((o - ref).abs() / (ref.abs() + 1e-6)).max().item()
        assert rel < 1e-3, f"seed={seed} scale={scale}: rel={rel}"


def test_intent_matmul_executes_on_gpu() -> None:
    """A contraction lowers to a reduction kernel (internal scf.for over the
    contracted dim) that runs bit-exact on the GPU — flash's QK/PV step."""
    from intentir.backend.compile import compile_all_kernels
    from intentir.backend.launch import I64Arg, PtrArg, launch, load_module

    # [fill, matmul] in launch order; the matmul is last. Pre-zeroing the output
    # lets us skip the fill kernel and launch the reduction kernel directly.
    matmul = compile_all_kernels(_MATMUL)[-1]
    func = load_module(matmul.cubin).function(matmul.kernel_name)

    torch.manual_seed(0)
    a = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    b = torch.randn(8, 16, dtype=torch.float32, device="cuda").contiguous()
    out = torch.zeros(8, 8, dtype=torch.float32, device="cuda").contiguous()

    # ABI: (step=1, lb=0, ptr a, ptr b, ptr out, index K=16); grid = output shape.
    launch(func, (8, 8, 1), (1, 1, 1),
           [I64Arg(1), I64Arg(0), PtrArg(a.data_ptr()),
            PtrArg(b.data_ptr()), PtrArg(out.data_ptr()), I64Arg(16)])

    assert (out - a @ b.T).abs().max().item() < 1e-4
