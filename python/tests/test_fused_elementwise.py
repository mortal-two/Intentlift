"""The fused-elementwise 'beat Triton via fusion' path.

Strict, deterministic checks: a pointwise chain fuses to EXACTLY one GPU kernel,
and the fused kernel matches torch (rel < 1e-3). Lenient perf check: the fused
kernel beats the unfused Triton composition (physics gives a 1.5x traffic ceiling
at HBM-bound sizes; we assert a conservative >1.2x). GPU-gated.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

_HAS_GPU = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="needs CUDA GPU")


@pytest.mark.parametrize("chain_name", ["add_mul", "silu_mul_add", "gelu_mul_add"])
def test_chain_fuses_to_one_kernel_and_matches(chain_name, tmp_path):
    from intentir.backend.compile import compile_all_kernels_fused
    from intentir.benchmarks.fused_elementwise import _chains, _build_plan, _launch_nosync

    chain = next(c for c in _chains() if c.name == chain_name)
    rows, cols = 4096, 512
    path = str(tmp_path / f"{chain_name}.mlir")
    with open(path, "w") as f:
        f.write(chain.intent_mlir(rows, cols))

    # exactly one fused kernel (else there is no traffic advantage)
    ks = compile_all_kernels_fused(path, (1, 256))
    assert len(ks) == 1, f"{chain_name} did not fuse to one kernel: {len(ks)}"

    numel = rows * cols
    plan = _build_plan(path, (1, 256), numel)
    ts = [torch.randn(rows, cols, device="cuda").contiguous() for _ in chain.inputs]
    out = torch.empty(numel, device="cuda").contiguous()
    _launch_nosync(plan, out, ts)
    torch.cuda.synchronize()
    ref = chain.ref(*ts)
    rel = (out.view(rows, cols) - ref).abs().max().item() / (ref.abs().max().item() + 1e-12)
    assert rel < 1e-3, f"{chain_name} fused kernel wrong: rel={rel}"


def test_fused_beats_unfused_triton():
    """End-to-end: the auto-fused kernel beats the unfused Triton composition."""
    from intentir.benchmarks.fused_elementwise import _chains, bench_chain

    chain = next(c for c in _chains() if c.name == "add_mul")
    # HBM-bound shape (working set ~268MB > 101MB L2); short windows for test speed.
    r = bench_chain(chain, rows=32768, cols=512, warmup=10, iters=50)
    assert r["certified"], f"not certified: rel_ref={r['rel_ref']}"
    assert r["speedup_vs_unfused"] > 1.2, (
        f"expected >1.2x over unfused Triton, got {r['speedup_vs_unfused']:.2f}x"
    )
