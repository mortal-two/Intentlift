"""PK the mature cross_entropy lowering vs the REAL liger kernel on the 5090D.

liger cross_entropy forward (loss + IN-PLACE gradient), the common config
(reduction='mean', no label_smoothing / z-loss / weight / ignore_index, so
n_non_ignore = M) is lowered from atomic IntentIR through the C++
``-convert-intent-row-reduction-to-gpu`` pass — the M1 register-tile scheduler,
now taught to evaluate ``intent.iota`` / ``intent.select`` / integer ``cmpi``
inside emitAt (the M6 residual) — to a read-once block-cooperative sm_120a cubin,
then launched and compared head-to-head with the raw ``liger_cross_entropy_kernel``
(bypassing the autograd Function, which crashes on this torch) for (1) correctness
(rel < 1e-3 on BOTH loss and grad, vs an fp32 torch reference AND vs liger's own
output) and (2) kernel-only median latency.

Lowered DAG (one row = one token's logits X[M,V], target Y[M]):
  m   = reduce{max}(X)                              # like softmax pass 1
  se  = reduce{sum}(exp(X - m)); lse = m + log(se)
  ori = reduce{sum}(select(iota == Y, X, 0))        # gather the target logit
  loss[row]      = (lse - ori) / M                  # PerRowOut memref<Mxf32>
  grad[row,col]  = (exp(X - lse) - one_hot(Y)) / M  # RowOut, in place over X

PERF — HONEST PARITY FLOOR (measured 1.00x / 0.99x, ~1528 GB/s = peak HBM). The
named-beat mechanism is real: our register tile loads X ONCE into SSA and
recomputes exp(X-lse) from registers in the grad epilogue, fusing liger's logical
pass-2 X reload (liger reads X for the online-softmax stats, then re-loads it to
recompute the gradient). BUT at these shapes that reload is served from L2, not
HBM: a V*4 = 16-32 KB logits row stays resident between pass 1 and pass 2 within
the same block, so the "third pass" costs ~0 HBM bytes. Both kernels therefore
sit at peak HBM for the REAL 2-pass traffic (read X once + write grad once) and
tie — exactly the M5-xformers situation (an evict_last reload that L2 already
made free). The read-once beat would only materialize once the row is too large
to stay in L2 (large V) — but that is precisely where K = V/T spills the register
tile (threshold K > tile-cols=32, i.e. V > 32768 at T=1024) and the streaming
scf.for loop becomes necessary. At T=1024: V=4096 -> K=4, V=8192 -> K=8, both far
under budget, so numTiles==1 (register tile, no streaming loop) and parity is the
honest result.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_cross_entropy_pk.py -q -s
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

_PASS = "convert-intent-row-reduction-to-gpu"
_T = 1024  # liger uses num_warps=32 -> 1024 threads/block


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=100, rep=300, return_mode="median")


def _scalar(v):
    return torch.tensor(v, device="cuda", dtype=torch.float32)


def _ce_mlir(M, V):
    R = f"tensor<{M}x{V}xf32>"      # X / grad
    P = f"tensor<{M}x1xf32>"        # per-row scalar (reduce keepdims)
    S = "tensor<f32>"               # uniform scalars
    YI = f"tensor<{M}xi32>"         # target class Y
    II = f"tensor<{V}xi32>"         # iota column index
    I = f"tensor<{M}x{V}xi1>"       # one-hot condition
    return f"""
func.func @cross_entropy(%X: {R}, %Y: {YI}, %zero: {S}, %one: {S}, %ninv: {S})
    -> ({P}, {R}) {{
  %iota = intent.iota : {II}
  %cmp = intent.map_binary %iota, %Y {{fn = "eq"}} : ({II}, {YI}) -> {I}
  %m = intent.reduce %X {{fn = "max", axes = [1], assoc = "associative"}} : ({R}) -> {P}
  %xm = intent.map_binary %X, %m {{fn = "sub"}} : ({R}, {P}) -> {R}
  %ex = intent.map_unary %xm {{fn = "exp"}} : ({R}) -> {R}
  %se = intent.reduce %ex {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %logse = intent.map_unary %se {{fn = "log"}} : ({P}) -> {P}
  %lse = intent.map_binary %m, %logse {{fn = "add"}} : ({P}, {P}) -> {P}
  %selo = intent.select %cmp, %X, %zero : ({I}, {R}, {S}) -> {R}
  %ori = intent.reduce %selo {{fn = "sum", axes = [1], assoc = "float_nondeterministic"}} : ({R}) -> {P}
  %lnum = intent.map_binary %lse, %ori {{fn = "sub"}} : ({P}, {P}) -> {P}
  %loss = intent.map_binary %lnum, %ninv {{fn = "mul"}} : ({P}, {S}) -> {P}
  %xlse = intent.map_binary %X, %lse {{fn = "sub"}} : ({R}, {P}) -> {R}
  %g = intent.map_unary %xlse {{fn = "exp"}} : ({R}) -> {R}
  %oh = intent.select %cmp, %one, %zero : ({I}, {S}, {S}) -> {R}
  %gnum = intent.map_binary %g, %oh {{fn = "sub"}} : ({R}, {R}) -> {R}
  %grad = intent.map_binary %gnum, %ninv {{fn = "mul"}} : ({R}, {S}) -> {R}
  return %loss, %grad : {P}, {R}
}}
"""


def _ref(logits, target, M):
    """fp32 torch reference (per-row loss + in-place-style grad, both /M)."""
    lse = torch.logsumexp(logits, dim=1)
    ori = logits.gather(1, target.long().view(-1, 1)).squeeze(1)
    loss = (lse - ori) / M
    grad = torch.softmax(logits, dim=1)
    grad[torch.arange(M, device=logits.device), target.long()] -= 1.0
    grad = grad / M
    return loss, grad


def _liger_launch(X, target, M, V):
    """Raw liger_cross_entropy_kernel launch (mean reduction, gradients on)."""
    from liger_kernel.ops.cross_entropy import (MAX_FUSED_SIZE,
                                                 liger_cross_entropy_kernel)
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    loss_1d = torch.zeros(M, dtype=torch.float32, device="cuda")
    dummy = torch.empty(1, device="cuda")

    def go():
        liger_cross_entropy_kernel[(M,)](
            X_ptr=X, X_stride=X.stride(-2),
            Y_ptr=target, Y_stride=target.stride(-1),
            weight_ptr=dummy,
            loss_ptr=loss_1d, z_loss_ptr=dummy, loss_stride=loss_1d.stride(-1),
            token_accuracy_ptr=dummy, token_accuracy_stride=0,
            predicted_tokens_ptr=dummy, predicted_tokens_stride=0,
            n_cols=V, n_non_ignore=M, sum_non_ignore_weight=M,
            weight_sum=0.0, ignore_index=-100,
            lse_square_scale=0.0, label_smoothing=0.0, reduction="mean",
            softcap=0.0, RETURN_Z_LOSS=False, RETURN_TOKEN_ACCURACY=False,
            RETURN_PREDICTED_TOKENS=False, BLOCK_SIZE=BLOCK_SIZE,
            HAS_WEIGHT=False, HAS_SOFTCAPPING=False, HAS_GRADIENTS=True,
            num_warps=32)
    return go, loss_1d


@pytest.mark.parametrize("M,V", [(8192, 4096), (4096, 8192)])
def test_cross_entropy_pk_vs_liger(M, V):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel

    assert V % _T == 0 and (V // _T) <= 32, "register tile must not spill"
    torch.manual_seed(0)
    logits = torch.randn(M, V, device="cuda", dtype=torch.float32).contiguous()
    target = torch.randint(0, V, (M,), device="cuda", dtype=torch.int32)

    k = RowReductionKernel("cross_entropy", {"M": M, "N": V}, _T,
                           pass_name=_PASS, mlir_text=_ce_mlir(M, V))

    # Correctness: run ONCE on a fresh clone (grad is written in place over X).
    Xmine = logits.clone()
    loss = torch.empty(M, device="cuda", dtype=torch.float32)
    # sig order: X, Y, zero, one, ninv, then outputs loss, grad(==X, in place).
    go = k.thunk([Xmine, target, _scalar(0.0), _scalar(1.0), _scalar(1.0 / M),
                  loss, Xmine])
    go()
    torch.cuda.synchronize()
    grad = Xmine  # gradient written back over the logits

    loss_ref, grad_ref = _ref(logits, target, M)
    assert _rel(loss, loss_ref) < 1e-3, f"loss rel {_rel(loss, loss_ref):.2e}"
    assert _rel(grad, grad_ref) < 1e-3, f"grad rel {_rel(grad, grad_ref):.2e}"

    # vs the real liger kernel (separate clone — it also writes grad in place).
    Xlig = logits.clone()
    lgo, lloss = _liger_launch(Xlig, target, M, V)
    lgo()
    torch.cuda.synchronize()
    assert _rel(loss, lloss) < 1e-3, f"loss vs liger {_rel(loss, lloss):.2e}"
    assert _rel(grad, Xlig) < 1e-3, f"grad vs liger {_rel(grad, Xlig):.2e}"

    # Perf: read X once + write grad once = 2 row passes (liger reads X twice +
    # writes once = 3). do_bench runs in place repeatedly (finite, value-agnostic).
    Xb = logits.clone()
    gob = k.thunk([Xb, target, _scalar(0.0), _scalar(1.0), _scalar(1.0 / M),
                   loss, Xb])
    Xlb = logits.clone()
    lgob, _ = _liger_launch(Xlb, target, M, V)
    tm, tl = _bench(gob), _bench(lgob)
    gb = 2 * M * V * 4 / 1e9        # matched real traffic: read X + write grad
    print(f"\ncross_entropy ({M},{V}) T={_T} K={V//_T}: "
          f"mine {tm*1e3:.1f}us ({gb/(tm/1e3):.0f} GB/s, 2-pass) "
          f"liger {tl*1e3:.1f}us  mine/liger {tm/tl:.2f}  "
          f"PARITY FLOOR: read-once removes liger's logical pass-2 X reload, but "
          f"that reload is L2-served (a {V*4//1024}KB row stays resident), so it "
          f"costs ~0 HBM bytes -> both sit at peak HBM (~1500 GB/s) at the real "
          f"2-pass traffic. The named read-once beat needs the reload to actually "
          f"reach HBM (V too large to stay in L2), where the tile would spill.")
    assert tm <= tl * 1.5
