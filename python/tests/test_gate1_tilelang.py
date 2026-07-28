from __future__ import annotations

import importlib.util

import pytest

from intentir.verify.gate1._tilelang_worker import _load as load_tilelang_source

from direct_fixtures import make_copy_artifact
from intentir.verify.gate1.model import Frontend
from intentir.verify.gate1.cuda_compiler import FrontendParameter
from intentir.verify.gate1.replay import StaticReplayWriter
from intentir.verify.gate1.tilelang import (
    compare_tilelang_fingerprints,
    fingerprint_tilelang,
    normalize_tilelang_tir_ops,
    normalize_tilelang_tir_structure,
)
from intentir.verify.gate1.tilelang_codegen import (
    UnsafeTileLangReplayCodeError,
    validate_tilelang_module,
)
from intentir.verify.gate1.tilelang_compiler import (
    TileLangCompileCase,
    TileLangCompiler,
    TileLangKernelSource,
)


_TIR = """@T.prim_func
def main(A: T.Buffer((32,), "float32"), B: T.Buffer((32,), "float32")):
    T.func_attr({"global_symbol": "main"})
    with T.Kernel(1, threads=32) as (bx,):
        v = T.alloc_fragment((32,), "float32")
        T.copy(A[0], v)
        T.copy(v, B[0])
"""

_PTX = """
.visible .entry main(.param .u64 main_param_0, .param .u64 main_param_1) {
  .reg .b64 %rd<3>;
  ld.param.u64 %rd1, [main_param_0];
  ld.global.f32 %f1, [%rd1];
  st.global.f32 [%rd1], %f1;
  ret;
}
"""

_REPLAY = """import tilelang.language as T

N = 32

@T.prim_func
def scrc_k0(p0: T.Tensor((N,), "float32"), p1: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=32) as (v0,):
        for v1 in T.Parallel(N):
            p1[v1] = p0[v1]
"""


def test_tilelang_replay_policy_accepts_public_primfunc_and_rejects_host_io() -> None:
    validation = validate_tilelang_module(_REPLAY, required_function="scrc_k0")
    assert validation.function_name == "scrc_k0"
    with pytest.raises(UnsafeTileLangReplayCodeError, match="forbidden"):
        validate_tilelang_module(
            _REPLAY.replace(
                "p1[v1] = p0[v1]",
                "p1[v1] = open('private.txt')",
            ),
            required_function="scrc_k0",
        )


@pytest.mark.parametrize(
    "unsafe_module",
    [
        _REPLAY.replace(
            "        for v1 in T.Parallel(N):",
            "        import os\n        for v1 in T.Parallel(N):",
        ),
        _REPLAY.replace("N = 32", "N: __import__('os') = 32"),
        _REPLAY.replace(
            "def scrc_k0(",
            "def scrc_k0(unsafe=open('private.txt'), ",
        ),
    ],
)
def test_tilelang_replay_policy_rejects_host_execution_surfaces(
    unsafe_module: str,
) -> None:
    with pytest.raises(UnsafeTileLangReplayCodeError):
        validate_tilelang_module(
            unsafe_module,
            required_function="scrc_k0",
        )


def test_tilelang_observer_combines_tir_topology_and_ptx_semantics() -> None:
    source = fingerprint_tilelang(_TIR, _PTX, interface={"arity": 2})
    replay = fingerprint_tilelang(
        _TIR.replace("def main", "def replay").replace('"main"', '"replay"'),
        _PTX.replace("main", "replay"),
        interface={"arity": 2},
    )
    assert compare_tilelang_fingerprints(source, replay).hard_pass

    different_tir = _TIR.replace(
        "T.copy(v, B[0])", "T.reduce_sum(v, B, dim=0)"
    )
    comparison = compare_tilelang_fingerprints(
        source,
        fingerprint_tilelang(different_tir, _PTX, interface={"arity": 2}),
    )
    assert not comparison.hard_passes["tir_structure"]
    assert not comparison.hard_passes["tir_topology"]


def test_tilelang_observer_normalizes_serial_and_parallel_loop_spellings() -> None:
    source = fingerprint_tilelang(_REPLAY, _PTX, interface={"arity": 2})
    replay = fingerprint_tilelang(
        _REPLAY.replace("T.Parallel", "T.serial"),
        _PTX,
        interface={"arity": 2},
    )

    comparison = compare_tilelang_fingerprints(source, replay)

    assert comparison.hard_pass
    assert comparison.primary_difference_category is None

    changed_bound = fingerprint_tilelang(
        _REPLAY.replace("T.Parallel(N)", "T.serial(N - 1)"),
        _PTX,
        interface={"arity": 2},
    )
    assert not compare_tilelang_fingerprints(
        source, changed_bound
    ).hard_passes["tir_structure"]


@pytest.mark.parametrize(
    "iterator",
    (
        "T.parallel",
        "T.serial",
        "T.vectorized",
        "T.unroll",
        "T.pipelined",
        "T.thread_binding",
    ),
)
def test_tilelang_observer_normalizes_only_public_t_schedule_iterators(
    iterator: str,
) -> None:
    source = fingerprint_tilelang(_REPLAY, _PTX, interface={"arity": 2})
    replay = fingerprint_tilelang(
        _REPLAY.replace("T.Parallel", iterator),
        _PTX,
        interface={"arity": 2},
    )

    assert compare_tilelang_fingerprints(source, replay).hard_pass

    non_t_iterator = fingerprint_tilelang(
        _REPLAY.replace("T.Parallel", f"helper.{iterator.rsplit('.', 1)[-1]}"),
        _PTX,
        interface={"arity": 2},
    )
    assert not compare_tilelang_fingerprints(
        source, non_t_iterator
    ).hard_passes["tir_structure"]


@pytest.mark.parametrize(
    "statement",
    (
        "p1[v1] = p0[v1] + 1",
        "p1[(v1 + 1) % N] = p0[v1]",
        "p0[v1] = p1[v1]",
    ),
)
def test_tilelang_schedule_normalization_keeps_body_indices_and_effects_hard(
    statement: str,
) -> None:
    source = fingerprint_tilelang(_REPLAY, _PTX, interface={"arity": 2})
    changed = fingerprint_tilelang(
        _REPLAY.replace("p1[v1] = p0[v1]", statement),
        _PTX,
        interface={"arity": 2},
    )

    assert not compare_tilelang_fingerprints(
        source, changed
    ).hard_passes["tir_structure"]


def test_tilelang_alpha_normalization_respects_function_scope_shadowing() -> None:
    source = _REPLAY.replace(
        "for v1 in T.Parallel(N):\n            p1[v1] = p0[v1]",
        "for N in T.Parallel(4):\n            p1[N] = p0[N]",
    )
    renamed = _REPLAY.replace(
        "for v1 in T.Parallel(N):\n            p1[v1] = p0[v1]",
        "for local_index in T.Parallel(4):\n            p1[local_index] = p0[local_index]",
    )

    assert normalize_tilelang_tir_structure(source) == normalize_tilelang_tir_structure(
        renamed
    )


def test_tilelang_structural_tir_retains_bounds_beyond_operation_names() -> None:
    source = fingerprint_tilelang(_TIR, _PTX, interface={"arity": 2})
    changed_bound = fingerprint_tilelang(
        _TIR.replace("threads=32", "threads=16"),
        _PTX,
        interface={"arity": 2},
    )

    comparison = compare_tilelang_fingerprints(source, changed_bound)

    assert comparison.hard_passes["tir_topology"]
    assert comparison.hard_passes["tir_operations"]
    assert not comparison.hard_passes["tir_structure"]
    assert not comparison.hard_pass


def test_tilelang_ptx_topology_is_diagnostic_below_equal_tir() -> None:
    source = fingerprint_tilelang(_TIR, _PTX, interface={"arity": 2})
    changed_ptx = fingerprint_tilelang(
        _TIR,
        _PTX.replace("ld.global.f32", "ld.global.ca.f32"),
        interface={"arity": 2},
    )

    comparison = compare_tilelang_fingerprints(source, changed_ptx)

    assert comparison.hard_pass
    assert comparison.primary_difference_category is None
    assert not comparison.diagnostic_passes["ptx.raw_instruction_topology"]
    assert any(
        item.category == "ptx.raw_instruction_topology"
        for item in comparison.differences
    )


def test_static_tilelang_writer_validates_frontend_specific_module() -> None:
    writer = StaticReplayWriter(_REPLAY, frontend=Frontend.TILELANG)
    generation = writer.generate(make_copy_artifact())
    assert generation.function_name == "scrc_k0"
    assert normalize_tilelang_tir_ops(_TIR).count("T.copy") == 2


def test_tilelang_factory_source_consumes_anonymous_schedule_globals(tmp_path) -> None:
    source = tmp_path / "kernel.py"
    source.write_text(
        "def make_kernel(*, bias=0):\n    return c0 + bias\n",
        encoding="utf-8",
    )
    request = {
        "source": {
            "kind": "factory_file",
            "path": str(source),
            "function": "make_kernel",
            "factory_kwargs": {"bias": 3},
        },
        "case": {"schedule": {"c0": 16}},
    }

    assert load_tilelang_source(request) == 19


@pytest.mark.skipif(
    importlib.util.find_spec("tilelang") is None,
    reason="TileLang unavailable",
)
def test_tilelang_real_compile_emits_tir_and_ptx_without_launching() -> None:
    case = TileLangCompileCase(
        (
            FrontendParameter("pointer", "f32", 1, "input"),
            FrontendParameter("pointer", "f32", 1, "output"),
        ),
        block=(32, 1, 1),
        schedule={"c0": 32},
    )
    compiled = TileLangCompiler(timeout=300).compile(
        TileLangKernelSource.from_replay(
            _REPLAY.replace("threads=32", "threads=c0"), "scrc_k0"
        ),
        case,
    )
    assert compiled.ok, compiled.diagnostic
    fingerprint = fingerprint_tilelang(
        compiled.tir or "",
        compiled.ptx or "",
        interface=case.interface,
        target={"name": "cuda"},
    )
    assert fingerprint.ptx.memory_effects
