"""PK the general OP-GRAPH DISPATCH — multi-op programs that cross the single-op
cluster boundary — vs torch AND the real flag_gems kernels on the 5090D.

The keystone the coverage task deferred. Each cluster pass (sort/gather/...) only
lowers a func whose body is ONE cluster op; a program mixing a cluster op with a
fused map/reduce chain cannot lower as one func. ``intentir.backend.cuda.opgraph``
SEQUENCES such a program into outlined single-op / single-chain stages, each
lowered by an existing pass unchanged, wired by intermediate device buffers:

  quantile     = sort(asc) -> gather([lo,hi]) -> weighted-sum lerp   (3 stages;
                 sort C6b + gather C3 + reduce). Linear interpolation, exact.
  topk_softmax = softmax(experts) -> topk(desc)                      (2 stages;
                 reduce + sort). The vLLM fused-MoE gating op: routing weights +
                 expert indices. Matches flag_gems.topk_softmax exactly.
  cross_entropy= log_softmax + nll fused to ONE row-reduction kernel (1 stage;
                 the gather of the target logit is a masked row-sum). The
                 degenerate single-stage op-graph.

Verify (correctness-only, deterministic): rel < 1e-3 vs an fp32 torch reference
AND vs the real flag_gems kernel for every target.

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_opgraph_pk.py -q -s
"""

from __future__ import annotations

import os
import pathlib

import pytest

from intentir.backend.cuda.compile import CUDA_BIN, INTENT_OPT, MLIR_OPT

torch = pytest.importorskip("torch")
flag_gems = pytest.importorskip("flag_gems")

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


def _scalar(v):
    return torch.tensor(v, device="cuda", dtype=torch.float32)


# --------------------------------------------------------------------------- #
# quantile = sort -> gather -> lerp   (sort C6b + gather C3 + reduce)          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q", [0.1, 0.37, 0.5, 0.95])
@pytest.mark.parametrize("M,N", [(256, 512), (128, 1024)])
def test_quantile_opgraph_pk(M, N, q):
    from intentir.backend.cuda.opgraph import quantile_plan

    torch.manual_seed(0)
    x = torch.randn(M, N, device="cuda").contiguous()

    prog, c, lo, hi, frac = quantile_plan(M, N, q)
    qindex = torch.tensor([[lo, hi]] * M, device="cuda", dtype=torch.int64).contiguous()
    qweight = torch.tensor([[1 - frac, frac]] * M,
                           device="cuda", dtype=torch.float32).contiguous()
    out = prog.run({"x": x, "qindex": qindex, "qweight": qweight})["out"].squeeze(1)

    ref = torch.quantile(x, q, dim=1, interpolation="linear")
    rt = _rel(out, ref)
    # flag_gems.quantile is broken for interpolated ranks on this arch (its own
    # triton launch raises "function takes 22 args (19 given)"; the FlagGems
    # "Unsupported GPU arch" specialization). PK vs it where it works (the
    # exact-rank path), else report the flag_gems bug and keep the torch assert.
    try:
        fg = flag_gems.quantile(x, q, dim=1, interpolation="linear")
        rg = _rel(out, fg)
        gems = f"rel_gems {rg:.2e}"
    except Exception as e:  # noqa: BLE001 - flag_gems-internal triton incompat
        rg, gems = None, f"gems_SKIP({type(e).__name__})"
    print(f"\nquantile ({M},{N}) q={q} [lo={lo},hi={hi},frac={frac:.3f}]: "
          f"rel_torch {rt:.2e} {gems} (3-stage sort->gather->lerp)")
    assert rt < 1e-3
    if rg is not None:
        assert rg < 1e-3


# --------------------------------------------------------------------------- #
# topk_softmax = softmax -> topk   (reduce + sort; the fused-MoE gating op)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k", [2, 8])
@pytest.mark.parametrize("tokens,experts", [(256, 32), (128, 64)])
def test_topk_softmax_opgraph_pk(tokens, experts, k):
    from intentir.backend.cuda.opgraph import topk_softmax_plan

    torch.manual_seed(0)
    gating = torch.randn(tokens, experts, device="cuda", dtype=torch.float32).contiguous()

    prog = topk_softmax_plan(tokens, experts, k)
    env = prog.run({"gating": gating})
    w, idx = env["weights"], env["indices"]

    # torch reference: softmax over experts, then top-k.
    probs = torch.softmax(gating, dim=1)
    rw, ri = torch.topk(probs, k, dim=1)
    rt = _rel(w, rw)

    # real flag_gems.topk_softmax (vLLM fused-MoE gating, fills in place).
    tw = torch.empty(tokens, k, device="cuda", dtype=torch.float32)
    ti = torch.empty(tokens, k, device="cuda", dtype=torch.int32)
    tei = torch.empty(tokens, k, device="cuda", dtype=torch.int32)
    flag_gems.topk_softmax(tw, ti, tei, gating)
    torch.cuda.synchronize()
    rg = _rel(w, tw)
    idx_ok = bool(torch.equal(idx, ti))
    print(f"\ntopk_softmax tokens={tokens} experts={experts} k={k}: "
          f"weights rel_torch {rt:.2e} rel_gems {rg:.2e} "
          f"indices==gems {idx_ok} (2-stage softmax->topk)")
    assert rt < 1e-3
    assert rg < 1e-3
    assert idx_ok


# --------------------------------------------------------------------------- #
# cross_entropy = log_softmax + nll, fused to ONE kernel (1-stage op-graph)    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("M,V,T", [(256, 1024, 256), (128, 2048, 256)])
def test_cross_entropy_opgraph_pk(M, V, T):
    from intentir.backend.cuda.opgraph import cross_entropy_plan

    assert V % T == 0 and (V // T) <= 32, "register tile must not spill"
    torch.manual_seed(0)
    X = torch.randn(M, V, device="cuda", dtype=torch.float32).contiguous()
    Y = torch.randint(0, V, (M,), device="cuda", dtype=torch.int32)

    prog = cross_entropy_plan(M, V, T)
    env = prog.run({"X": X.clone(), "Y": Y, "zero": _scalar(0.0),
                    "one": _scalar(1.0), "ninv": _scalar(1.0 / M)})
    loss = env["loss"].squeeze(1)
    grad = env["grad"]

    # fp32 torch reference (per-row loss + softmax-minus-onehot gradient, both /M).
    lse = torch.logsumexp(X, dim=1)
    ori = X.gather(1, Y.long().view(-1, 1)).squeeze(1)
    loss_ref = (lse - ori) / M
    grad_ref = torch.softmax(X, dim=1)
    grad_ref[torch.arange(M, device=X.device), Y.long()] -= 1.0
    grad_ref = grad_ref / M
    rt_loss, rt_grad = _rel(loss, loss_ref), _rel(grad, grad_ref)

    # real flag_gems.cross_entropy_loss (mean reduction, scalar). Our per-row loss
    # sums to the mean (each row divided by M).
    fg = flag_gems.cross_entropy_loss(X, Y.long(), reduction="mean")
    rg = abs(float(loss.sum()) - float(fg)) / abs(float(fg))
    print(f"\ncross_entropy ({M},{V}) T={T}: loss rel_torch {rt_loss:.2e} "
          f"grad rel_torch {rt_grad:.2e} mean-loss rel_gems {rg:.2e} (1-stage fused)")
    assert rt_loss < 1e-3
    assert rt_grad < 1e-3
    assert rg < 1e-3
