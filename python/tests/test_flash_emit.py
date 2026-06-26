"""Tests for the parameterized fused-FlashAttention emitter (``bench/flash_emit``).

Validates four things end to end on the 5090D:

* CORRECTNESS — ``run_fused_flash`` matches exact ``softmax((Q@K.T)*scale)@V`` for
  a spread of (sq, sk, d, bk), including a non-divisible TAIL shape (3,10,4,4).
* TILE-INVARIANCE (retune safety) — for a fixed shape, every ``bk`` schedule gives
  the identical Layer-A output (the schedule is non-binding).
* vs TRITON — agrees with the independent ``bench/triton_flash`` golden.
* RETUNE MECHANISM — ``retune_sweep`` times each ``bk`` (compile-once) and selects a
  ``bk`` whose output is correct (no timing assertion — one-thread-per-query makes
  the absolute latency deltas small; this is the mechanism, not a perf claim).

Skip guards mirror ``test_backend_execute.py`` (torch / CUDA / build / mlir-opt).
Run: ``export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \\
python3 -m pytest python/tests/test_flash_emit.py -q``.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")

# Make ptxas reachable from inside the test process (the device pipeline shells
# out to ptxas via compile.py, which prepends CUDA_BIN to PATH itself, but we set
# it here too so a bare `pytest` invocation without the export still finds it).
os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")

pytestmark = [
    pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                       reason="intent-opt not built"),
    pytest.mark.skipif(not pathlib.Path(MLIR_OPT).exists(),
                       reason="mlir-opt not present"),
    pytest.mark.skipif(not pathlib.Path(CUDA_BIN).exists(),
                       reason="CUDA toolkit not present"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU"),
]


def _rand(sq: int, sk: int, d: int, seed: int = 0):
    torch.manual_seed(seed)
    q = torch.randn(sq, d, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(sk, d, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(sk, d, device="cuda", dtype=torch.float32).contiguous()
    return q, k, v


def _exact(q, k, v, scale):
    return torch.softmax((q @ k.T) * scale, dim=1) @ v


def test_bk1_structurally_matches_flash_fused() -> None:
    """The bk=1 emission must reproduce the hand-written ``flash_fused.mlir`` result
    (same single-key online rescale) — the structural-equivalence check, done
    numerically: same inputs, same output to f32 round-off."""
    from intentir.emitters.flash_emit import run_fused_flash

    for seed, scale in [(0, 0.5), (1, 1.0), (2, 0.125)]:
        q, k, v = _rand(2, 4, 2, seed)
        o = run_fused_flash(q, k, v, scale, 1)
        ref = _exact(q, k, v, scale)
        rel = ((o - ref).abs() / (ref.abs() + 1e-6)).max().item()
        assert rel < 1e-3, f"seed={seed} scale={scale}: rel={rel}"


@pytest.mark.parametrize(
    "sq,sk,d,bk",
    [(2, 4, 2, 1), (4, 8, 16, 4), (8, 16, 16, 16), (3, 10, 4, 4)],
)
def test_correctness(sq: int, sk: int, d: int, bk: int) -> None:
    """run_fused_flash vs exact softmax across divisible and TAIL (3,10,4,4) shapes."""
    from intentir.emitters.flash_emit import run_fused_flash

    q, k, v = _rand(sq, sk, d, seed=7)
    scale = 1.0 / (d ** 0.5)
    o = run_fused_flash(q, k, v, scale, bk)
    ref = _exact(q, k, v, scale)
    rel = ((o - ref).abs() / (ref.abs() + 1e-6)).max().item()
    assert rel < 1e-3, f"(sq={sq},sk={sk},d={d},bk={bk}): rel={rel}"


def test_tile_invariance() -> None:
    """Retune safety: for a fixed (4,16,16) shape, all bk schedules agree to <1e-4.

    Different K-tile groupings (different schedules) -> the identical Layer-A
    output: the schedule is non-binding, so the retuner is free to pick any bk."""
    from intentir.emitters.flash_emit import run_fused_flash

    sq, sk, d = 4, 16, 16
    q, k, v = _rand(sq, sk, d, seed=3)
    scale = 1.0 / (d ** 0.5)
    outs = {bk: run_fused_flash(q, k, v, scale, bk) for bk in (1, 2, 4, 8, 16)}
    base = outs[1]
    maxdis = max((outs[bk] - base).abs().max().item() for bk in (2, 4, 8, 16))
    assert maxdis < 1e-4, f"tile-invariance max disagreement = {maxdis}"


def test_vs_triton() -> None:
    """The fused kernel agrees with the independent Triton flash golden (<1e-3)."""
    triton_flash = pytest.importorskip("intentir.benchmarks.triton_flash")
    if not triton_flash.HAVE_TRITON:
        pytest.skip("triton not available")
    from intentir.emitters.flash_emit import run_fused_flash

    sq, sk, d = 8, 16, 16
    q, k, v = _rand(sq, sk, d, seed=5)
    scale = 1.0 / (d ** 0.5)
    o_fused = run_fused_flash(q, k, v, scale, 4)
    o_tri = triton_flash.triton_flash_attention(q, k, v, scale, block_n=4)
    rel = ((o_fused - o_tri).abs() / (o_tri.abs() + 1e-6)).max().item()
    assert rel < 1e-3, f"fused vs triton: rel={rel}"


def test_retune_sweep_selects_correct_bk() -> None:
    """retune_sweep on (64,128,16) returns per-bk latencies and a best bk whose
    output is correct vs the exact reference. No timing assertion: one-thread-per-
    query makes the deltas small — this asserts the *mechanism* (sweep + correctness-
    preserving selection), not a tuned-perf claim."""
    from intentir.emitters.flash_emit import retune_sweep, run_fused_flash

    sq, sk, d = 64, 128, 16
    res = retune_sweep(sq, sk, d)
    assert set(res["latencies_ms"]) == {1, 2, 4, 8, 16}
    assert all(v < float("inf") for v in res["latencies_ms"].values()), \
        f"a candidate failed to compile/launch: {res['latencies_ms']}"
    best = res["best_bk"]
    assert best in (1, 2, 4, 8, 16)

    # best_bk is selected by (noisy, ~1% spread) timing, so WHICH bk wins varies
    # run to run. The correctness-preserving property the retuner relies on is that
    # EVERY candidate (hence whichever one is selected) is correct vs the reference,
    # so assert all of them — this both pins "the best is correct" (best is a subset)
    # and removes the nondeterminism from what the test validates.
    q, k, v = _rand(sq, sk, d, seed=0)
    scale = 1.0 / (d ** 0.5)
    ref = _exact(q, k, v, scale)
    for bk in (1, 2, 4, 8, 16):
        o = run_fused_flash(q, k, v, scale, bk)
        rel = ((o - ref).abs() / (ref.abs() + 1e-6)).max().item()
        assert rel < 1e-3, f"bk={bk} correctness: rel={rel}"
