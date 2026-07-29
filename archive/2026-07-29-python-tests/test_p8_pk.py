"""PK the P8 "last deferred" ops vs the REAL flag_gems kernels on the 5090D.

The three ops the llm-serving / opgraph clusters explicitly named-deferred, each a
composition of mechanisms IntentIR already lowers — NOT a new pass:

  per_token_group_quant_fp8 = grouped abs-max scale + clamp/divide, then an e4m3
        grid cast. ONE row-reduction kernel computes the scales + clamped q (the
        reduce -> broadcast -> map shape); the fp8 rounding is the arch's
        float8_e4m3fn cast (emitting an f8E4M3FN store through NVVM is a separate
        large mechanism, deferred). Verify the SCALES exactly + the DEQUANTIZED
        values (q_fp8*scale) vs the real flag_gems dequant.
  reshape_and_cache (non-flash) = the x-vectorized cache-layout C3 scatter (the
        extra reshape bookkeeping the flash/MLA row-scatter deferred). One
        intent.scatter into the flattened cache with the vectorized destination
        coordinate. Byte-exact vs flag_gems.
  moe_align_block_size = by-expert sort (C6b) + block-padded scan (C6) + placement
        scatter (C3). EXACT vs flag_gems (sorted_ids, expert_ids, num post-pad).

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_p8_pk.py -q -s
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
    b = b.float()
    n = b.norm()
    a = a.float()
    return float((a - b).norm() / n) if n > 0 else float((a - b).abs().max())


def _scalar(v):
    return torch.tensor(v, device="cuda", dtype=torch.float32)


# --------------------------------------------------------------------------- #
# per_token_group_quant_fp8 — grouped abs-max scale + clamp/divide + e4m3 cast  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rows,cols,group_size", [(64, 512, 128), (128, 256, 256)])
def test_per_token_group_quant_fp8_pk(rows, cols, group_size):
    from intentir.backend.cuda.opgraph import per_token_group_quant_fp8_plan

    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32).contiguous()
    eps = 1e-10

    # real flag_gems: returns x_q (float8_e4m3fn) + x_s (f32 scales).
    xq_fg, xs_fg = flag_gems.per_token_group_quant_fp8(x, group_size, eps=eps)
    torch.cuda.synchronize()
    fp8_dtype = xq_fg.dtype
    fp8_max = float(torch.finfo(fp8_dtype).max)
    fp8_min = float(torch.finfo(fp8_dtype).min)

    G = x.numel() // group_size
    xg = x.reshape(G, group_size).contiguous()
    prog = per_token_group_quant_fp8_plan(G, group_size)
    env = prog.run({"x": xg, "eps": _scalar(eps), "invmax": _scalar(1.0 / fp8_max),
                    "fmin": _scalar(fp8_min), "fmax": _scalar(fp8_max)})
    scale = env["scale"]           # [G,1]
    q = env["q"]                   # [G,group_size] pre-cast clamp
    q_fp8 = q.to(fp8_dtype)        # the e4m3 grid cast (the dtype step)
    dequant = q_fp8.float() * scale

    # references.
    scale_fg = xs_fg.reshape(G, 1)
    dequant_fg = xq_fg.float().reshape(G, group_size) * scale_fg
    rs = _rel(scale, scale_fg)
    rd = _rel(dequant, dequant_fg)
    # torch reference dequant.
    amax = xg.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    scale_ref = amax / fp8_max
    q_ref = (xg / scale_ref).clamp(fp8_min, fp8_max).to(fp8_dtype).float()
    dequant_ref = q_ref * scale_ref
    rt = _rel(dequant, dequant_ref)
    print(f"\nper_token_group_quant_fp8 (rows={rows},cols={cols},gs={group_size}) "
          f"dtype={fp8_dtype}: scale rel_gems {rs:.2e} dequant rel_gems {rd:.2e} "
          f"dequant rel_torch {rt:.2e}")
    assert rs < 1e-6              # scales are exact (f32)
    assert rd < 1e-3
    assert rt < 1e-3


# --------------------------------------------------------------------------- #
# reshape_and_cache (non-flash) — x-vectorized cache-layout C3 scatter          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_blocks,block_size,num_heads,head_size,x,Nt",
                         [(64, 16, 8, 64, 8, 200), (32, 32, 4, 128, 16, 300)])
def test_reshape_and_cache_pk(num_blocks, block_size, num_heads, head_size, x, Nt):
    from intentir.backend.cuda.opgraph import reshape_and_cache_run

    torch.manual_seed(0)
    key = torch.randn(Nt, num_heads, head_size, device="cuda", dtype=torch.float32)
    value = torch.randn(Nt, num_heads, head_size, device="cuda", dtype=torch.float32)
    Rc = num_blocks * block_size
    slot = torch.randperm(Rc, device="cuda")[:Nt].to(torch.int32).contiguous()

    # real flag_gems (the caches after the write are the reference).
    kc = torch.zeros(num_blocks, num_heads, head_size // x, block_size, x, device="cuda")
    vc = torch.zeros(num_blocks, num_heads, head_size, block_size, device="cuda")
    flag_gems.reshape_and_cache(key, value, kc, vc, slot, "auto",
                                _scalar(1.0), _scalar(1.0))
    torch.cuda.synchronize()

    my_kc = torch.zeros_like(kc)
    my_vc = torch.zeros_like(vc)
    reshape_and_cache_run(key, value, my_kc, my_vc, slot, block_size, x)

    rk = _rel(my_kc, kc)
    rv = _rel(my_vc, vc)
    print(f"\nreshape_and_cache (blocks={num_blocks},bs={block_size},H={num_heads},"
          f"hs={head_size},x={x},Nt={Nt}): key rel_gems {rk:.2e} value rel_gems {rv:.2e}")
    assert torch.equal(my_kc, kc)   # byte-exact key cache
    assert torch.equal(my_vc, vc)   # byte-exact value cache


# --------------------------------------------------------------------------- #
# moe_align_block_size — by-expert sort + block-padded scan + placement scatter #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_tokens,top_k,num_experts,block_size",
                         [(256, 2, 8, 16), (128, 4, 16, 32), (512, 2, 8, 64)])
def test_moe_align_block_size_pk(num_tokens, top_k, num_experts, block_size):
    from intentir.backend.cuda.opgraph import moe_align_block_size_run

    torch.manual_seed(0)
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k),
                             device="cuda", dtype=torch.int32)

    # real flag_gems.
    s_fg, e_fg, n_fg = flag_gems.moe_align_block_size(topk_ids, block_size, num_experts)
    torch.cuda.synchronize()

    s_my, e_my, n_my = moe_align_block_size_run(topk_ids, block_size, num_experts)

    ns = int(n_fg.item())
    npp_ok = int(n_my.item()) == ns
    # sorted_ids: compare the meaningful padded prefix (both fill unused with numel).
    sorted_ok = torch.equal(s_my, s_fg)
    expert_ok = torch.equal(e_my, e_fg)
    print(f"\nmoe_align_block_size (tokens={num_tokens},top_k={top_k},"
          f"E={num_experts},bs={block_size}): num_post_pad {int(n_my.item())} vs "
          f"{ns} ({npp_ok}) sorted_ids=={sorted_ok} expert_ids=={expert_ok}")
    assert npp_ok
    assert sorted_ok
    assert expert_ok
