from __future__ import annotations

import pytest

from intentir.verify.gate1.cute import (
    compare_cute_fingerprints,
    fingerprint_cute,
    normalize_cute_mlir,
)
from intentir.verify.gate1.cute_codegen import (
    UnsafeCuteReplayCodeError,
    validate_cute_module,
)


def _mlir(*, symbol: str, argument: str, result: str, location: str) -> str:
    return f"""
#loc = loc(unknown)
#loc1 = loc(\"{location}\":1:1)
module attributes {{gpu.container_module}} {{
  gpu.module @{symbol} {{
    cuda.kernel @{symbol}_kernel({argument}: !llvm.ptr<1> loc(#loc1)) attributes {{gpu.kernel}} {{
      {result} = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc1)
      nvvm.mbarrier.init.shared {argument}, {result} : !llvm.ptr<1>, i32 loc(#loc1)
      cuda.return loc(#loc1)
    }} loc(#loc1)
  }} loc(#loc1)
}} loc(#loc1)
"""


def test_cute_observer_alpha_normalizes_names_and_locations() -> None:
    left = _mlir(symbol="source", argument="%arg0", result="%tid", location="a.py")
    right = _mlir(symbol="replay", argument="%input", result="%lane", location="b.py")
    assert normalize_cute_mlir(left) == normalize_cute_mlir(right)
    ptx = """
.version 8.8
.target sm_120a
    mbarrier.init.shared.b64 [%r1], 1;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r2], [%rd1], 16, [%r1];
    bar.sync 0;
"""
    source = fingerprint_cute(left, ptx, python_interface={"arity": 1})
    replay = fingerprint_cute(right, ptx, python_interface={"arity": 1})
    comparison = compare_cute_fingerprints(source, replay)
    assert source.digest == replay.digest
    assert comparison.hard_pass


def test_cute_observer_keeps_raw_bulk_and_tma_distinct() -> None:
    mlir = _mlir(symbol="module", argument="%arg0", result="%tid", location="x.py")
    raw = fingerprint_cute(
        mlir,
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1], 16, [%r2];",
    )
    tensor = fingerprint_cute(
        mlir,
        "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r1], [%rd1], {%r2}, [%r3];",
    )
    comparison = compare_cute_fingerprints(raw, tensor)
    assert not comparison.hard_pass
    assert {item.category for item in comparison.differences} >= {
        "ptx_topology",
        "async_operations",
    }


_SAFE_REPLAY = """import cutlass
import cutlass.cute as cute

@cute.kernel
def scrc_k0(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    dst[tidx] = src[tidx]

@cute.jit
def scrc_entry(src, dst, stream):
    scrc_k0(src, dst).launch(
        grid=[1, 1, 1], block=[32, 1, 1], smem=0, stream=stream
    )
"""


def test_cute_replay_policy_accepts_public_kernel_and_rejects_host_io() -> None:
    validation = validate_cute_module(
        _SAFE_REPLAY,
        required_kernel="scrc_k0",
        required_entry="scrc_entry",
    )
    assert validation.kernel_name == "scrc_k0"
    unsafe = _SAFE_REPLAY.replace(
        "tidx, _, _ = cute.arch.thread_idx()", "open('answer.txt').read()"
    )
    with pytest.raises(UnsafeCuteReplayCodeError, match="host-side"):
        validate_cute_module(
            unsafe,
            required_kernel="scrc_k0",
            required_entry="scrc_entry",
        )
