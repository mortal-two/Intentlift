"""M3: execute the hand-authored FlashAttention IntentIR (loop+carry online
softmax) through the reference interpreter and triangulate it against a numpy
batched-attention reference and torch SDPA.

This proves the *fused* form's semantics with zero GPU code — the Track-A oracle
that the back-end (M5-M9) will diff every lowered kernel against. Tested at
several tile sizes (full / clean-dividing / tail) because online softmax must be
tile-invariant up to float tolerance.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.interpret import interpret

_JSON = pathlib.Path(__file__).resolve().parents[2] / "test" / "flashattention_fwd.intent.json"


def _load() -> IntentFunction:
    return IntentFunction.from_json(json.loads(_JSON.read_text()))


def _ref_attention(Q, K, V, scale):
    s = scale * np.einsum("bhqd,bhkd->bhqk", Q, K)
    s = s - s.max(axis=-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, V)


def _inputs(B=2, H=3, Sq=8, Sk=64, D=16, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, Sq, D)).astype(np.float32)
    K = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
    V = rng.standard_normal((B, H, Sk, D)).astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(D))
    return Q, K, V, scale


def test_flash_json_round_trips() -> None:
    fn = _load()
    assert fn.name == "flash_attention_fwd"
    loop = fn.body[0]
    assert loop.op == "loop"
    assert [c.state_id for c in loop.carry] == ["state0", "state1", "state2"]
    assert [c.body_argument for c in loop.carry] == ["m_iter", "l_iter", "Oacc_iter"]
    assert [c.result_value for c in loop.carry] == ["m", "l", "Oacc"]
    assert [s["name"] for s in loop.scan] == ["Kj", "Vj"]
    # JSON -> model -> JSON is lossless (scan/carry/yield survive).
    assert IntentFunction.from_json(fn.to_json()).to_json() == fn.to_json()


@pytest.mark.parametrize("tile", [None, 16, 32, 64])
def test_flash_matches_reference(tile) -> None:
    fn = _load()
    Q, K, V, scale = _inputs()
    params = {} if tile is None else {"BK": tile}
    out = interpret(fn, {"Q": Q, "K": K, "V": V, "scale": scale}, params)["O"]
    np.testing.assert_allclose(out, _ref_attention(Q, K, V, scale),
                               rtol=1e-4, atol=1e-4)


def test_flash_tail_tile() -> None:
    # Sk not divisible by BK exercises the tail iteration (R6 / tail-tile case).
    fn = _load()
    Q, K, V, scale = _inputs(Sk=70)
    out = interpret(fn, {"Q": Q, "K": K, "V": V, "scale": scale}, {"BK": 16})["O"]
    np.testing.assert_allclose(out, _ref_attention(Q, K, V, scale),
                               rtol=1e-4, atol=1e-4)


def test_flash_tile_invariance() -> None:
    # Triangulation: every tiling agrees with the single-iteration (naive) form.
    fn = _load()
    Q, K, V, scale = _inputs(Sk=48)
    inp = {"Q": Q, "K": K, "V": V, "scale": scale}
    naive = interpret(fn, inp, {})["O"]
    for bk in (8, 16, 24):
        tiled = interpret(fn, inp, {"BK": bk})["O"]
        np.testing.assert_allclose(tiled, naive, rtol=1e-5, atol=1e-5)


def test_flash_matches_torch_if_available() -> None:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return
    fn = _load()
    Q, K, V, scale = _inputs(seed=3)
    out = interpret(fn, {"Q": Q, "K": K, "V": V, "scale": scale}, {"BK": 16})["O"]
    ref = F.scaled_dot_product_attention(
        torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V),
        scale=float(scale),
    ).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
