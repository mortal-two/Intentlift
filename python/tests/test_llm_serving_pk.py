"""PK the LLM-serving compositions (rotary / KV-cache / MoE / repetition-penalty)
vs torch AND the REAL flag_gems kernels on the 5090D.

These are the deferred "stateful serving" cluster — every one is a COMPOSITION of
mechanisms IntentIR already lowers (paired map/select for rotary, C3 scatter for
the paged KV-cache write, reduce for the MoE expert sum), NOT a new pass:

  apply_rotary_pos_emb  = paired rotation out_a = x_a*cos - x_b*sin,
                          out_b = x_a*sin + x_b*cos over the half-split
                          (GPT-NeoX) OR the interleaved even/odd (GPT-J) layout.
                          Lowered by -convert-intent-elementwise-to-gpu with the
                          partner coupling expressed as strided half / even-odd
                          VIEWS of one [M,hd] buffer (intent.strides arg-attr) +
                          the cos/sin per-token broadcast row. Matches
                          flag_gems.apply_rotary_pos_emb's math exactly
                          (rotary_interleaved False and True).
  reshape_and_cache_flash / concat_and_cache_mla
                        = scatter key/value rows into a paged KV-cache tensor at
                          the slot-mapping addresses. The flash / MLA cache
                          layouts flatten to [num_blocks*block_size, D] and the
                          write is cache[slot[t], :] = kv[t, :] — a C3
                          intent.scatter along axis 0 (the slot index is the
                          data-dependent row address), lowered by
                          -convert-intent-gather-to-gpu. Matches the flag_gems
                          cache tensor after the write, byte-for-byte.
  moe_sum               = sum the top-k expert outputs per token = a reduce over
                          the expert axis, expressed as an add-chain over the k
                          strided expert VIEWS of the [T,K,H] buffer
                          (-convert-intent-elementwise-to-gpu). Matches
                          flag_gems.moe_sum.
  apply_repetition_penalties
                        = scale logits at previously-seen token ids by a per-seq
                          penalty (divide if >0, multiply if <=0) = a masked
                          select/map over the seen-token mask
                          (-convert-intent-elementwise-to-gpu). Matches the
                          flag_gems logits tensor after the in-place update.

Correctness-only (deterministic): rel < 1e-3 vs an fp32 torch reference AND vs the
real flag_gems kernel for every target.

DEFERRED (named, not landed here): reshape_and_cache (the non-flash key cache's
`x`-vectorized [num_blocks, num_heads, head_size/x, block_size, x] reshape is a
genuine index-bookkeeping remap, not a row scatter); moe_align_block_size
(sort+scan+scatter integer routing); rwkv_* (recurrent carried state).

Run: export PATH=/usr/local/cuda-12.8/bin:$PATH; PYTHONPATH=python \
     python3 -m pytest python/tests/test_llm_serving_pk.py -q -s
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

_ELT = "convert-intent-elementwise-to-gpu"
_SCAT = "convert-intent-gather-to-gpu"


def _rel(a, b):
    b = b.float()
    n = b.norm()
    return float((a.float() - b).norm() / n) if n > 0 else float((a.float() - b).abs().max())


def _T(N):
    t = 1
    while t * 2 <= N and t * 2 <= 256:
        t *= 2
    return t


def _div_threads(cols, cap=256):
    t = min(cap, cols)
    while t > 1 and cols % t != 0:
        t >>= 1
    return t


def _kernel(name, M, N, T, pass_name, mlir_text):
    from intentir.backend.cuda.kernel_driver import RowReductionKernel
    return RowReductionKernel(name, {"M": M, "N": N}, T,
                              pass_name=pass_name, mlir_text=mlir_text)


def _scalar(v):
    return torch.tensor(v, device="cuda", dtype=torch.float32)


# --------------------------------------------------------------------------- #
# apply_rotary_pos_emb — the flagship. Half-split (GPT-NeoX) + interleaved      #
# (GPT-J). Paired rotation over strided views; cos/sin per-token broadcast row. #
# --------------------------------------------------------------------------- #
def _rope_half_mlir(M, hd, Ntok):
    N = hd // 2
    R = f"tensor<{M}x{N}xf32>"
    C = f"tensor<{Ntok}x{N}xf32>"
    sq = f"array<i64: {hd}, 1>"       # q half-views: row stride hd, col 1
    sc = f"array<i64: {N}, 1>"        # cos/sin contiguous [Ntok, hd/2]
    return f"""
func.func @rope_half(%q_lo: {R} {{intent.strides = {sq}}},
                     %q_hi: {R} {{intent.strides = {sq}}},
                     %cos: {C} {{intent.strides = {sc}}},
                     %sin: {C} {{intent.strides = {sc}}})
    -> ({R}, {R}) attributes {{intent.inplace = array<i64: 0, 1>}} {{
  %a = intent.map_binary %q_lo, %cos {{fn = "mul"}} : ({R}, {C}) -> {R}
  %b = intent.map_binary %q_hi, %sin {{fn = "mul"}} : ({R}, {C}) -> {R}
  %ylo = intent.map_binary %a, %b {{fn = "sub"}} : ({R}, {R}) -> {R}
  %c = intent.map_binary %q_hi, %cos {{fn = "mul"}} : ({R}, {C}) -> {R}
  %d = intent.map_binary %q_lo, %sin {{fn = "mul"}} : ({R}, {C}) -> {R}
  %yhi = intent.map_binary %c, %d {{fn = "add"}} : ({R}, {R}) -> {R}
  return %ylo, %yhi : {R}, {R}
}}
"""


def _rope_il_mlir(M, hd, Ntok):
    N = hd // 2
    R = f"tensor<{M}x{N}xf32>"
    C = f"tensor<{Ntok}x{N}xf32>"
    sq = f"array<i64: {hd}, 2>"       # q even/odd views: row stride hd, col 2
    sc = f"array<i64: {N}, 1>"        # cos/sin contiguous [Ntok, hd/2]
    return f"""
func.func @rope_il(%qe: {R} {{intent.strides = {sq}}},
                   %qo: {R} {{intent.strides = {sq}}},
                   %cos: {C} {{intent.strides = {sc}}},
                   %sin: {C} {{intent.strides = {sc}}})
    -> ({R}, {R}) attributes {{intent.inplace = array<i64: 0, 1>}} {{
  %a = intent.map_binary %qe, %cos {{fn = "mul"}} : ({R}, {C}) -> {R}
  %b = intent.map_binary %qo, %sin {{fn = "mul"}} : ({R}, {C}) -> {R}
  %ye = intent.map_binary %a, %b {{fn = "sub"}} : ({R}, {R}) -> {R}
  %c = intent.map_binary %qe, %sin {{fn = "mul"}} : ({R}, {C}) -> {R}
  %d = intent.map_binary %qo, %cos {{fn = "mul"}} : ({R}, {C}) -> {R}
  %yo = intent.map_binary %c, %d {{fn = "add"}} : ({R}, {R}) -> {R}
  return %ye, %yo : {R}, {R}
}}
"""


def _rope_ref(q, cos, sin, interleaved):
    hd2 = q.shape[-1] // 2
    c = cos[:, None, :]
    s = sin[:, None, :]
    if interleaved:
        x1, x2 = q[..., 0::2], q[..., 1::2]
        out = torch.empty_like(q)
        out[..., 0::2] = x1 * c - x2 * s
        out[..., 1::2] = x1 * s + x2 * c
        return out
    x1, x2 = q[..., :hd2], q[..., hd2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("Ntok,nh,hd", [(512, 8, 128), (256, 16, 64)])
def test_apply_rotary_pos_emb_pk(Ntok, nh, hd, interleaved):
    M = Ntok * nh
    hd2 = hd // 2
    torch.manual_seed(0)
    q = torch.randn(Ntok, nh, hd, device="cuda", dtype=torch.float32)
    k = torch.randn(Ntok, nh, hd, device="cuda", dtype=torch.float32)
    cos = torch.randn(Ntok, hd2, device="cuda", dtype=torch.float32).contiguous()
    sin = torch.randn(Ntok, hd2, device="cuda", dtype=torch.float32).contiguous()
    pos = torch.arange(Ntok, device="cuda", dtype=torch.int32)

    # real flag_gems (out-of-place); its non-interleaved is the half-split, its
    # interleaved is the even/odd pairing — matched exactly by our two kernels.
    qf, kf = flag_gems.apply_rotary_pos_emb(
        q, k, cos, sin, position_ids=pos, rotary_interleaved=interleaved,
        inplace=False)
    torch.cuda.synchronize()

    mlir = _rope_il_mlir(M, hd, Ntok) if interleaved else _rope_half_mlir(M, hd, Ntok)
    kern = _kernel("rope", M, hd2, _T(hd2), _ELT, mlir)

    qm = q.reshape(M, hd).clone().contiguous()
    km = k.reshape(M, hd).clone().contiguous()
    if interleaved:
        vq = (qm[:, 0::2], qm[:, 1::2])
        vk = (km[:, 0::2], km[:, 1::2])
    else:
        vq = (qm[:, :hd2], qm[:, hd2:])
        vk = (km[:, :hd2], km[:, hd2:])
    kern.thunk([vq[0], vq[1], cos, sin])()   # in-place rotate q
    kern.thunk([vk[0], vk[1], cos, sin])()   # in-place rotate k
    torch.cuda.synchronize()

    ref_q = _rope_ref(q, cos, sin, interleaved).reshape(M, hd)
    rt = _rel(qm, ref_q)
    rg_q = _rel(qm, qf.reshape(M, hd))
    rg_k = _rel(km, kf.reshape(M, hd))
    print(f"\napply_rotary_pos_emb interleaved={interleaved} "
          f"(Ntok={Ntok},nh={nh},hd={hd}): q rel_torch {rt:.2e} "
          f"q rel_gems {rg_q:.2e} k rel_gems {rg_k:.2e}")
    assert rt < 1e-3
    assert rg_q < 1e-3
    assert rg_k < 1e-3


# --------------------------------------------------------------------------- #
# reshape_and_cache_flash — C3 scatter of key/value rows into the paged cache   #
# at slot-mapping addresses. Cache flattens to [num_blocks*block_size, D].      #
# --------------------------------------------------------------------------- #
def _cache_scatter_mlir(name, Rc, D, Nt):
    Dc = f"tensor<{Rc}x{D}xf32>"
    S = f"tensor<{Nt}x{D}xf32>"
    I = f"tensor<{Nt}x{D}xi64>"
    return f"""
func.func @{name}(%data: {Dc}, %index: {I}, %src: {S}) -> {Dc} {{
  %o = intent.scatter %data, %index, %src {{axis = 0 : i64, mode = "along_axis", reduce = "none"}} : ({Dc}, {I}, {S}) -> {Dc}
  return %o : {Dc}
}}
"""


def _slot_index(Nt, Rc, D):
    slot = torch.randperm(Rc, device="cuda")[:Nt].to(torch.int32).contiguous()
    index = slot.view(-1, 1).expand(-1, D).to(torch.int64).contiguous()
    return slot, index


@pytest.mark.parametrize("num_blocks,block_size,num_heads,head_size,Nt",
                         [(64, 16, 8, 64, 200), (32, 32, 4, 128, 300)])
def test_reshape_and_cache_flash_pk(num_blocks, block_size, num_heads, head_size, Nt):
    D = num_heads * head_size
    Rc = num_blocks * block_size
    torch.manual_seed(0)
    key = torch.randn(Nt, num_heads, head_size, device="cuda", dtype=torch.float32)
    value = torch.randn(Nt, num_heads, head_size, device="cuda", dtype=torch.float32)
    slot, index = _slot_index(Nt, Rc, D)

    # real flag_gems (cache tensor after the write is the reference).
    kc = torch.zeros(num_blocks, block_size, num_heads, head_size, device="cuda")
    vc = torch.zeros(num_blocks, block_size, num_heads, head_size, device="cuda")
    flag_gems.reshape_and_cache_flash(
        key, value, kc, vc, slot, "auto", _scalar(1.0), _scalar(1.0))
    torch.cuda.synchronize()

    T = _div_threads(D)
    kern = _kernel("kv_flash", Nt, D, T, _SCAT, _cache_scatter_mlir("kv_flash", Rc, D, Nt))
    my_kc = torch.zeros(Rc, D, device="cuda").contiguous()
    my_vc = torch.zeros(Rc, D, device="cuda").contiguous()
    kern.thunk([my_kc, index, key.reshape(Nt, D)])()
    kern.thunk([my_vc, index, value.reshape(Nt, D)])()
    torch.cuda.synchronize()

    # torch reference: row scatter into the flat cache.
    ref_kc = torch.zeros(Rc, D, device="cuda")
    ref_kc[slot.long()] = key.reshape(Nt, D)
    rk = _rel(my_kc, kc.reshape(Rc, D))
    rv = _rel(my_vc, vc.reshape(Rc, D))
    rt = _rel(my_kc, ref_kc)
    print(f"\nreshape_and_cache_flash (blocks={num_blocks},bs={block_size},"
          f"H={num_heads},hs={head_size},Nt={Nt}): key rel_gems {rk:.2e} "
          f"value rel_gems {rv:.2e} key rel_torch {rt:.2e}")
    assert rt < 1e-3
    assert rk < 1e-3
    assert rv < 1e-3
    assert torch.equal(my_kc, kc.reshape(Rc, D))  # exact scatter


# --------------------------------------------------------------------------- #
# concat_and_cache_mla — concat kv_c + k_pe, then row-scatter into the MLA cache #
# [num_blocks, block_size, kv_lora_rank + pe_dim].                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_blocks,block_size,kv_lora,pe,Nt",
                         [(32, 16, 512, 64, 200), (16, 32, 256, 128, 150)])
def test_concat_and_cache_mla_pk(num_blocks, block_size, kv_lora, pe, Nt):
    D = kv_lora + pe
    Rc = num_blocks * block_size
    torch.manual_seed(0)
    kv_c = torch.randn(Nt, kv_lora, device="cuda", dtype=torch.float32)
    k_pe = torch.randn(Nt, pe, device="cuda", dtype=torch.float32)
    slot, index = _slot_index(Nt, Rc, D)

    # real flag_gems (kv_cache after the write is the reference).
    kv_cache = torch.zeros(num_blocks, block_size, D, device="cuda")
    flag_gems.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot, "auto", _scalar(1.0))
    torch.cuda.synchronize()

    T = _div_threads(D)
    kern = _kernel("mla", Nt, D, T, _SCAT, _cache_scatter_mlir("mla", Rc, D, Nt))
    src = torch.cat([kv_c, k_pe], dim=1).contiguous()   # concat then scatter
    my_cache = torch.zeros(Rc, D, device="cuda").contiguous()
    kern.thunk([my_cache, index, src])()
    torch.cuda.synchronize()

    rg = _rel(my_cache, kv_cache.reshape(Rc, D))
    print(f"\nconcat_and_cache_mla (blocks={num_blocks},bs={block_size},"
          f"lora={kv_lora},pe={pe},Nt={Nt}): rel_gems {rg:.2e}")
    assert rg < 1e-3
    assert torch.equal(my_cache, kv_cache.reshape(Rc, D))  # exact concat+scatter


# --------------------------------------------------------------------------- #
# moe_sum — sum the top-k expert outputs per token = reduce over the expert     #
# axis, expressed as an add-chain over the k strided expert views of [T,K,H].   #
# --------------------------------------------------------------------------- #
def _moe_sum_mlir(M, H, K, stride0):
    R = f"tensor<{M}x{H}xf32>"
    s = f"array<i64: {stride0}, 1>"
    args = ",\n                      ".join(
        f"%a{e}: {R} {{intent.strides = {s}}}" for e in range(K))
    lines = []
    prev = "%a0"
    for e in range(1, K):
        lines.append(f'  %s{e} = intent.map_binary {prev}, %a{e} {{fn = "add"}}'
                     f" : ({R}, {R}) -> {R}")
        prev = f"%s{e}"
    body = "\n".join(lines)
    return f"""
func.func @moe_sum({args}) -> {R} {{
{body}
  return {prev} : {R}
}}
"""


@pytest.mark.parametrize("topk", [2, 4, 8])
@pytest.mark.parametrize("tokens,hidden", [(256, 512), (128, 1024)])
def test_moe_sum_pk(tokens, hidden, topk):
    torch.manual_seed(0)
    inp = torch.randn(tokens, topk, hidden, device="cuda", dtype=torch.float32).contiguous()

    # real flag_gems (fills the output tensor).
    out_fg = torch.empty(tokens, hidden, device="cuda", dtype=torch.float32)
    flag_gems.moe_sum(inp, out_fg)
    torch.cuda.synchronize()

    kern = _kernel("moe_sum", tokens, hidden, _T(hidden), _ELT,
                   _moe_sum_mlir(tokens, hidden, topk, topk * hidden))
    experts = [inp[:, e, :] for e in range(topk)]   # strided expert views
    my_out = torch.empty(tokens, hidden, device="cuda", dtype=torch.float32)
    kern.thunk(experts + [my_out])()
    torch.cuda.synchronize()

    ref = inp.sum(dim=1)
    rt = _rel(my_out, ref)
    rg = _rel(my_out, out_fg)
    print(f"\nmoe_sum tokens={tokens} topk={topk} hidden={hidden}: "
          f"rel_torch {rt:.2e} rel_gems {rg:.2e}")
    assert rt < 1e-3
    assert rg < 1e-3


# --------------------------------------------------------------------------- #
# apply_repetition_penalties — scale logits at seen-token ids by a per-seq      #
# penalty: divide if logit>0, multiply if logit<=0. Masked select/map.          #
# --------------------------------------------------------------------------- #
def _rep_penalty_mlir(S, V):
    R = f"tensor<{S}x{V}xf32>"
    I = f"tensor<{S}x{V}xi1>"
    F = "tensor<f32>"
    return f"""
func.func @rep_pen(%logits: {R}, %pmask: {R}, %omask: {R}, %pen: {R}, %zero: {F})
    -> {R} {{
  %rep = intent.map_binary %pmask, %omask {{fn = "max"}} : ({R}, {R}) -> {R}
  %isrep = intent.map_binary %rep, %zero {{fn = "gt"}} : ({R}, {F}) -> {I}
  %gt0 = intent.map_binary %logits, %zero {{fn = "gt"}} : ({R}, {F}) -> {I}
  %pos = intent.map_binary %logits, %pen {{fn = "div"}} : ({R}, {R}) -> {R}
  %neg = intent.map_binary %logits, %pen {{fn = "mul"}} : ({R}, {R}) -> {R}
  %pick = intent.select %gt0, %pos, %neg : ({I}, {R}, {R}) -> {R}
  %out = intent.select %isrep, %pick, %logits : ({I}, {R}, {R}) -> {R}
  return %out : {R}
}}
"""


@pytest.mark.parametrize("S,V", [(64, 1024), (32, 2048)])
def test_apply_repetition_penalties_pk(S, V):
    torch.manual_seed(0)
    logits = torch.randn(S, V, device="cuda", dtype=torch.float32).contiguous()
    pmask = (torch.rand(S, V, device="cuda") < 0.1).contiguous()
    omask = (torch.rand(S, V, device="cuda") < 0.1).contiguous()
    penalties = (1.0 + torch.rand(S, device="cuda")).contiguous()   # in [1,2)

    # real flag_gems (mutates a clone of logits in place).
    fg_logits = logits.clone().contiguous()
    flag_gems.apply_repetition_penalties(fg_logits, pmask, omask, penalties)
    torch.cuda.synchronize()

    kern = _kernel("rep_pen", S, V, _T(V), _ELT, _rep_penalty_mlir(S, V))
    pen_bc = penalties.view(S, 1).expand(S, V).contiguous()
    my_out = torch.empty(S, V, device="cuda", dtype=torch.float32)
    kern.thunk([logits, pmask.float(), omask.float(), pen_bc, _scalar(0.0), my_out])()
    torch.cuda.synchronize()

    # torch reference.
    is_rep = pmask | omask
    penb = penalties.view(-1, 1)
    scaled = torch.where(logits > 0, logits / penb, logits * penb)
    ref = torch.where(is_rep, scaled, logits)
    rt = _rel(my_out, ref)
    rg = _rel(my_out, fg_logits)
    print(f"\napply_repetition_penalties (S={S},V={V}): rel_torch {rt:.2e} "
          f"rel_gems {rg:.2e}")
    assert rt < 1e-3
    assert rg < 1e-3
