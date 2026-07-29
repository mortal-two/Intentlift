"""PK the M4a cross-row AtomicOut lowering (kl_div, jsd) vs the REAL liger kernels.

kl_div (batchmean) and jsd (beta=0) are lowered from atomic IntentIR through the
extended ``-convert-intent-row-reduction-to-gpu`` pass to a read-once block-
cooperative sm_120a cubin whose per-row block reduce is folded ACROSS blocks into
a rank-0 scalar via a thread-0 ``memref.atomic_rmw addf`` (the AtomicOut fold).
This fuses the host ``.sum()`` the library launches separately — the named beat.

Correctness: rel < 1e-3 vs fp32 torch ref AND vs the real liger op. Timing:
kernel-only no-sync ``do_bench``; the liger closure brackets its post-launch host
``.sum()`` (kl_div) / ``torch.sum`` over the [BT,V] loss buffer (jsd), because
that host work IS what the fused atomicAdd removes.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_loss_divergence_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

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


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _num_warps(BLOCK_SIZE):
    if BLOCK_SIZE >= 32768:
        return 32
    if BLOCK_SIZE >= 8192:
        return 16
    if BLOCK_SIZE >= 2048:
        return 8
    return 4


def _kl_div_mlir(M, N):
    return f"""
func.func @kl_div(%y: tensor<{M}x{N}xf32>, %gt: tensor<{M}x{N}xf32>,
                  %eps: tensor<f32>, %invBT: tensor<f32>) -> tensor<f32> {{
  %me = intent.map_binary %gt, %eps {{fn = "max"}} : (tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %lg = intent.map_unary %me {{fn = "log"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %d  = intent.map_binary %lg, %y {{fn = "sub"}} : (tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %le = intent.map_binary %gt, %d {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %rs = intent.reduce %le {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xf32>
  %tot = intent.reduce %rs {{fn = "sum", axes = [0], assoc = "float_nondeterministic"}} : (tensor<{M}xf32>) -> tensor<f32>
  %loss = intent.map_binary %tot, %invBT {{fn = "mul"}} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %loss : tensor<f32>
}}
"""


def _jsd_mlir(M, N):
    return f"""
func.func @jsd(%X: tensor<{M}x{N}xf32>, %Y: tensor<{M}x{N}xf32>,
               %scale: tensor<f32>) -> (tensor<f32>, tensor<{M}x{N}xf32>) {{
  %Yp = intent.map_unary %Y {{fn = "exp"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %ymx = intent.map_binary %Y, %X {{fn = "sub"}} : (tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %le = intent.map_binary %Yp, %ymx {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %les = intent.map_binary %le, %scale {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  %rs = intent.reduce %les {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}xf32>
  %tot = intent.reduce %rs {{fn = "sum", axes = [0], assoc = "float_nondeterministic"}} : (tensor<{M}xf32>) -> tensor<f32>
  %nYp = intent.map_unary %Yp {{fn = "neg"}} : (tensor<{M}x{N}xf32>) -> tensor<{M}x{N}xf32>
  %dX = intent.map_binary %nYp, %scale {{fn = "mul"}} : (tensor<{M}x{N}xf32>, tensor<f32>) -> tensor<{M}x{N}xf32>
  return %tot, %dX : tensor<f32>, tensor<{M}x{N}xf32>
}}
"""


@pytest.mark.parametrize("M,N", [(4096, 4096), (2048, 8192)])
def test_kl_div_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.kl_div import kldiv_forward_triton

    T = _num_warps(N) * 32
    eps = 1e-10
    torch.manual_seed(0)
    y_true = torch.softmax(torch.randn(M, N, device="cuda"), dim=1).contiguous()
    y_pred = torch.log_softmax(torch.randn(M, N, device="cuda"), dim=1).contiguous()

    k = RowReductionKernel("kl_div", {"M": M, "N": N}, T, mlir_text=_kl_div_mlir(M, N))
    epsd = torch.tensor(eps, device="cuda")
    invBT = torch.tensor(1.0 / M, device="cuda")
    loss = torch.zeros(1, device="cuda")
    go = k.thunk([y_pred, y_true, epsd, invBT, loss])  # sig order: y, gt, eps, invBT, loss
    loss.zero_(); go(); torch.cuda.synchronize()

    ref = (y_true * (torch.log(torch.clamp(y_true, min=eps)) - y_pred)).sum() / M
    rel_ref = float(abs(loss[0] - ref) / abs(ref))
    ligv = kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="batchmean", eps=eps)
    rel_lig = float(abs(loss[0] - ligv) / abs(ligv))

    def liger():
        kldiv_forward_triton(y_pred, y_true, log_target=False, reduction="batchmean", eps=eps)

    tm, tl = _bench(go), _bench(liger)
    print(f"\nkl_div ({M},{N}) T={T}: mine {tm*1e3:.1f}us liger {tl*1e3:.1f}us "
          f"mine/liger {tm/tl:.2f}  rel_ref {rel_ref:.2e} rel_liger {rel_lig:.2e}")
    assert rel_ref < 1e-3 and rel_lig < 1e-3
    assert tm <= tl * 1.5


@pytest.mark.parametrize("M,N", [(4096, 4096), (2048, 8192)])
def test_jsd_pk_vs_liger(M, N):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    from liger_kernel.ops.jsd import jsd_forward

    T = _num_warps(N) * 32
    torch.manual_seed(0)
    X = torch.log_softmax(torch.randn(M, N, device="cuda"), dim=1).contiguous()
    Y = torch.log_softmax(torch.randn(M, N, device="cuda"), dim=1).contiguous()

    k = RowReductionKernel("jsd", {"M": M, "N": N}, T, mlir_text=_jsd_mlir(M, N))
    scale = torch.tensor(1.0 / M, device="cuda")
    loss = torch.zeros(1, device="cuda")
    dX = torch.empty(M, N, device="cuda")
    go = k.thunk([X, Y, scale, loss, dX])  # sig order: X, Y, scale, loss, dX
    loss.zero_(); go(); torch.cuda.synchronize()

    Yp = torch.exp(Y)
    loss_ref = (Yp * (Y - X)).sum() / M
    dX_ref = -Yp / M
    rel_loss = float(abs(loss[0] - loss_ref) / abs(loss_ref))
    rel_dx = _rel(dX, dX_ref)
    lig_loss, lig_dX = jsd_forward(X, Y, None, 0.0, -100, False)
    rel_lig = float(abs(loss[0] - lig_loss) / abs(lig_loss))

    def liger():
        jsd_forward(X, Y, None, 0.0, -100, False)

    tm, tl = _bench(go), _bench(liger)
    print(f"\njsd ({M},{N}) T={T}: mine {tm*1e3:.1f}us liger {tl*1e3:.1f}us "
          f"mine/liger {tm/tl:.2f}  rel_loss {rel_loss:.2e} rel_dx {rel_dx:.2e} rel_liger {rel_lig:.2e}")
    assert rel_loss < 1e-3 and rel_dx < 1e-3 and rel_lig < 1e-3
    assert tm <= tl * 1.5
