"""M3: execute the hand-authored FlashAttention IntentIR (loop+carry online
softmax) through the reference interpreter and triangulate it against a numpy
batched-attention reference and torch SDPA.

This preserves a historical NumPy witness for the structured recurrence.  The
canonical J no longer accepts a tile hint: source/target tiling is supplied by C
to lowering, while this interpreter executes the complete semantic trip axis.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from intentir.ir.model import IntentFunction
from intentir.verify.back.legacy_v0.interpret import interpret

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


@pytest.mark.parametrize("irrelevant_schedule_value", [None, 16, 32, 64])
def test_flash_matches_reference(irrelevant_schedule_value) -> None:
    fn = _load()
    Q, K, V, scale = _inputs()
    params = (
        {} if irrelevant_schedule_value is None
        else {"BK": irrelevant_schedule_value}
    )
    out = interpret(fn, {"Q": Q, "K": K, "V": V, "scale": scale}, params)["O"]
    np.testing.assert_allclose(out, _ref_attention(Q, K, V, scale),
                               rtol=5e-4, atol=5e-4)


def test_flash_non_multiple_extent_is_semantic_not_a_j_tile() -> None:
    fn = _load()
    Q, K, V, scale = _inputs(Sk=70)
    out = interpret(fn, {"Q": Q, "K": K, "V": V, "scale": scale}, {"BK": 16})["O"]
    np.testing.assert_allclose(out, _ref_attention(Q, K, V, scale),
                               rtol=5e-4, atol=5e-4)


def test_legacy_interpreter_does_not_consume_schedule_values() -> None:
    fn = _load()
    Q, K, V, scale = _inputs(Sk=48)
    inp = {"Q": Q, "K": K, "V": V, "scale": scale}
    naive = interpret(fn, inp, {})["O"]
    for bk in (8, 16, 24):
        tiled = interpret(fn, inp, {"BK": bk})["O"]
        np.testing.assert_array_equal(tiled, naive)


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
