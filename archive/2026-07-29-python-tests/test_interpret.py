"""Golden tests for the IntentIR reference interpreter.

These pin the executable semantics of the atomic ops against independent numpy
references, and compose them into a full (non-online) attention to prove the
interpreter can execute the FlashAttention *math* before the online-softmax loop
op exists. Float comparisons are tolerance-based by policy (see interpret.py).
"""

from __future__ import annotations

import numpy as np

from intentir.ir.model import IntentFunction
from intentir.verify.back.legacy_v0.interpret import interpret


def _run(fn_json: dict, **inputs: np.ndarray) -> dict[str, np.ndarray]:
    return interpret(IntentFunction.from_json(fn_json), inputs)


def test_map_binary_add_broadcast() -> None:
    fn = {
        "name": "addrow",
        "inputs": [
            {"name": "x", "dtype": "f32", "shape": [4, 8]},
            {"name": "b", "dtype": "f32", "shape": [4, 1]},
        ],
        "outputs": [{"name": "y", "dtype": "f32", "shape": [4, 8]}],
        "body": [
            {"op": "map_binary", "result": "y", "operands": ["x", "b"],
             "attrs": {"fn": "add"}},
        ],
    }
    x = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(4, 1).astype(np.float32)
    out = _run(fn, x=x, b=b)["y"]
    np.testing.assert_allclose(out, x + b, rtol=1e-6, atol=1e-6)


def test_reduce_max_keepdims() -> None:
    fn = {
        "name": "rowmax",
        "inputs": [{"name": "x", "dtype": "f32", "shape": [4, 8]}],
        "outputs": [{"name": "m", "dtype": "f32", "shape": [4, 1]}],
        "body": [
            {"op": "reduce", "result": "m", "operands": ["x"],
             "attrs": {"fn": "max", "axes": [1], "assoc": "associative"}},
        ],
    }
    x = np.random.randn(4, 8).astype(np.float32)
    out = _run(fn, x=x)["m"]
    assert out.shape == (4, 1)
    np.testing.assert_allclose(out, x.max(axis=1, keepdims=True), rtol=0, atol=0)


def test_map_unary_rsqrt() -> None:
    fn = {
        "name": "rsqrt",
        "inputs": [{"name": "x", "dtype": "f32", "shape": [8]}],
        "outputs": [{"name": "y", "dtype": "f32", "shape": [8]}],
        "body": [
            {"op": "map_unary", "result": "y", "operands": ["x"],
             "attrs": {"fn": "rsqrt"}},
        ],
    }
    x = np.linspace(0.25, 4.0, 8, dtype=np.float32)
    out = _run(fn, x=x)["y"]
    np.testing.assert_allclose(out, 1.0 / np.sqrt(x), rtol=1e-6, atol=1e-6)


def test_contraction_qk() -> None:
    fn = {
        "name": "qk",
        "inputs": [
            {"name": "q", "dtype": "f32", "shape": [4, 16]},
            {"name": "k", "dtype": "f32", "shape": [4, 16]},
        ],
        "outputs": [{"name": "s", "dtype": "f32", "shape": [4, 4]}],
        "body": [
            {"op": "contraction", "result": "s", "operands": ["q", "k"],
             "attrs": {"subscripts": "qd,kd->qk"}},
        ],
    }
    q = np.random.randn(4, 16).astype(np.float32)
    k = np.random.randn(4, 16).astype(np.float32)
    out = _run(fn, q=q, k=k)["s"]
    np.testing.assert_allclose(out, q @ k.T, rtol=1e-5, atol=1e-5)


# --- full (non-online) attention composed from atomic ops --------------------

_ATTENTION = {
    "name": "attention",
    "inputs": [
        {"name": "Q", "dtype": "f32", "shape": ["S", "D"]},
        {"name": "K", "dtype": "f32", "shape": ["S", "D"]},
        {"name": "V", "dtype": "f32", "shape": ["S", "D"]},
    ],
    "outputs": [{"name": "O", "dtype": "f32", "shape": ["S", "D"]}],
    "body": [
        {"op": "contraction", "result": "qk", "operands": ["Q", "K"],
         "attrs": {"subscripts": "qd,kd->qk"}},
        {"op": "reduce", "result": "m", "operands": ["qk"],
         "attrs": {"fn": "max", "axes": [1], "assoc": "associative"}},
        {"op": "map_binary", "result": "sc", "operands": ["qk", "m"],
         "attrs": {"fn": "sub", "broadcast": "row"}},
        {"op": "map_unary", "result": "p", "operands": ["sc"],
         "attrs": {"fn": "exp"}},
        {"op": "reduce", "result": "l", "operands": ["p"],
         "attrs": {"fn": "sum", "axes": [1], "assoc": "float_nondeterministic"}},
        {"op": "map_binary", "result": "pn", "operands": ["p", "l"],
         "attrs": {"fn": "div", "broadcast": "row"}},
        {"op": "contraction", "result": "O", "operands": ["pn", "V"],
         "attrs": {"subscripts": "qk,kd->qd"}},
    ],
}


def _softmax_attention_ref(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    s = Q @ K.T
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(axis=1, keepdims=True)
    return p @ V


def test_attention_matches_reference() -> None:
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((8, 16)).astype(np.float32)
    K = rng.standard_normal((8, 16)).astype(np.float32)
    V = rng.standard_normal((8, 16)).astype(np.float32)
    out = _run(_ATTENTION, Q=Q, K=K, V=V)["O"]
    np.testing.assert_allclose(out, _softmax_attention_ref(Q, K, V),
                               rtol=1e-5, atol=1e-5)


def test_attention_matches_torch_if_available() -> None:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return  # torch optional for the keystone unit test
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((8, 16)).astype(np.float32)
    K = rng.standard_normal((8, 16)).astype(np.float32)
    V = rng.standard_normal((8, 16)).astype(np.float32)
    out = _run(_ATTENTION, Q=Q, K=K, V=V)["O"]
    ref = F.scaled_dot_product_attention(
        torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V), scale=1.0
    ).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
