"""Compile + launch the read-once row-reduction kernels, for PK vs the library.

The mature ``-convert-intent-row-reduction-to-gpu`` pass lowers a row-norm intent
func (rms_norm / softmax / ...) to ONE read-once block-cooperative ``gpu.launch``.
This module compiles that to an sm_120a cubin and launches it, resolving the
bare-ptr launch plan from the lowered host module (the outliner's capture order
is NOT the func-arg order, so we parse ``gpu.launch_func ... args(...)`` and map
each aligned-ptr back to its input index — same machinery the flash emitter uses).

MEASUREMENT (cost real debug time — encode here, do not relearn):
  * Time with ``triton.testing.do_bench`` over a NO-SYNC launch (``thunk``), the
    same way a Triton baseline is timed. do_bench manages clocks + an L2 flush
    and brackets just the kernel via CUDA events.
  * Two traps that fabricate a 4-38x phantom slowdown:
      (1) a ``torch.cuda.synchronize()`` INSIDE the timed launch (what
          ``backend.launch.launch`` does) breaks do_bench -> ~6800us for a 178us
          kernel. Use the no-sync ``thunk`` here.
      (2) a hand-rolled CUDA-event loop floors at ~730us for EVERY kernel
          (incl. torch.copy_ and liger) -> a fake "366 GB/s ceiling". It is the
          timer, not the kernel.
    Measured correctly, rms_norm/softmax sit at parity with liger (~1507 GB/s,
    peak HBM, bit-exact) — parity is the honest ceiling for these min-traffic
    library kernels (read-once, write-once: no bytes left to take out).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

from intentir.backend.compile import (INTENT_OPT, _run,
                                       lower_memref_module_to_kernels)
from intentir.backend.launch import (I64Arg, PtrArg, _driver, _marshal,
                                      load_module)
from intentir.backend.orchestrate import (_host_func_text,
                                           _map_input_aligned_ptrs,
                                           _parse_constants, _parse_launches,
                                           _parse_signature_blockargs,
                                           _resolve_const_index)
from intentir.frontend.emit_mlir import emit_module
from intentir.ir.model import IntentFunction

_REPO = Path(__file__).resolve().parents[3]  # benchmarks/intentir/python/<root>


def _intent_mlir(name: str, dims: dict) -> str:
    spec = json.loads(
        (_REPO / "corpus" / "kernels" / name / "intent.json").read_text()
    )["intent_function"]
    return emit_module(IntentFunction.from_json(spec), dims, fold=0)


class RowReductionKernel:
    """A compiled row-reduction kernel + its resolved bare-ptr launch plan.

    ``thunk(tensors)`` returns a zero-arg, NO-SYNC launch closure ready for
    ``do_bench`` (or a single call + manual ``torch.cuda.synchronize()`` for a
    correctness check). ``tensors`` are the inputs in func-signature order
    followed by the output Y (all contiguous CUDA f32; a rank-0 scalar like eps
    is a 0-d / 1-elem tensor).
    """

    def __init__(self, name: str, dims: dict, block_threads: int,
                 pass_name: str = "convert-intent-row-reduction-to-gpu",
                 mlir_text: str | None = None):
        """Lower + compile a read-once kernel for PK.

        ``pass_name`` selects the C++ lowering pass to drive (default the
        row-reduction pass; pass ``"convert-intent-elementwise-to-gpu"`` for the
        pure-elementwise M3 activations). ``mlir_text`` is a raw intent-MLIR
        module string (M3 hand-writes the intent the same shape as the lit test,
        bypassing corpus extraction); when ``None`` the corpus ``intent.json``
        for ``name`` is emitted instead.
        """
        src = mlir_text if mlir_text is not None else _intent_mlir(name, dims)
        rr = _run(
            [INTENT_OPT, f"-{pass_name}=block-threads={block_threads}"],
            stdin=src,
        )
        kernels = lower_memref_module_to_kernels(rr)
        if len(kernels) != 1:
            raise RuntimeError(
                f"row_reduction: expected one kernel for {name}, got {len(kernels)}"
            )
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(
                f"row_reduction: expected one gpu.launch_func, got {len(sites)}"
            )
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        # M5 tiled+smem path: a workgroup (dynamic shared) row cache. Resolve the
        # dynamic_shared_memory_size constant and, when it exceeds the 48 KiB
        # default static cap, opt the kernel into a larger dynamic-shared budget.
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            # Opt the kernel into the larger dynamic-shared budget: the 48 KiB
            # default cap is for static shared, and the gpu.all_reduce already
            # uses a little static shared, so even a 48 KiB cache needs the
            # opt-in (max-dynamic-shared up to the sm_120 ~100 KiB ceiling).
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)
        # plan: per launch arg, ("ptr", input_index) | ("i64", literal value).
        # An i64 arg appears when -canonicalize hoists a constant out of the
        # launch region so the outliner captures it (e.g. a folded `1`).
        self.plan = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            elif op.ty in ("i64", "index"):
                self.plan.append(("i64", int(consts[op.ssa][0])))
            else:
                raise RuntimeError(f"row_reduction: unhandled launch-arg {op.ty}")
        self._cuda = _driver()

    def thunk(self, tensors):
        """A zero-arg NO-SYNC launch closure binding `tensors` (kept alive)."""
        import torch

        args = [PtrArg(tensors[i].data_ptr()) if kind == "ptr" else I64Arg(i)
                for kind, i in self.plan]
        keep, argp = _marshal(args)
        cuda, func, grid, block = self._cuda, self.func, self.grid, self.block
        shared = self.shared
        addr = int(ctypes.addressof(argp))
        stream = int(torch.cuda.current_stream().cuda_stream)

        def go(_keep=keep, _argp=argp):  # keep the marshaled buffer alive
            cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                block[0], block[1], block[2],
                                shared, stream, addr, 0)

        return go


class MatmulKernel:
    """A compiled tensor-core matmul kernel + its resolved bare-ptr launch plan.

    Drives ``-convert-intent-contraction-to-gpu`` on a raw intent-MLIR module
    (one ``intent.contraction{"mk,nk->mn"}``, A/W f16 -> Y f32) and compiles the
    resulting tiled-K tensor-core ``gpu.launch`` to an sm_120a cubin. Unlike the
    row-reduction pass this pass has no ``block-threads`` option (the WMMA tile
    geometry is fixed: BM=BN=64, BK=16, 4 warps), and it needs ``scf_to_cf=True``
    so the gpu->nvvm pipeline can legalize the ``scf.for`` carrying
    ``!gpu.mma_matrix`` iter_args (the K-loop accumulator fragments).

    ``thunk(tensors)`` returns a zero-arg NO-SYNC launch closure ready for
    ``do_bench``; ``tensors`` are [A, W, Y] (A[M,K] f16, W[N,K] f16, Y[M,N] f32,
    all contiguous CUDA).
    """

    def __init__(self, mlir_text: str):
        rr = _run([INTENT_OPT, "-convert-intent-contraction-to-gpu"],
                  stdin=mlir_text)
        kernels = lower_memref_module_to_kernels(rr, scf_to_cf=True)
        if len(kernels) != 1:
            raise RuntimeError(
                f"matmul: expected one kernel, got {len(kernels)}")
        ck = kernels[0]
        self.func = load_module(ck.cubin).function(ck.kernel_name)

        _n, ftext = _host_func_text(ck.binary_mlir)
        _s, _o, ptr_blockargs = _parse_signature_blockargs(ftext)
        aligned = _map_input_aligned_ptrs(ptr_blockargs)
        consts = _parse_constants(ftext)
        sites = _parse_launches(ftext)
        if len(sites) != 1:
            raise RuntimeError(
                f"matmul: expected one gpu.launch_func, got {len(sites)}")
        site = sites[0]

        class _Shim:
            constants = consts

        self.grid = tuple(_resolve_const_index(_Shim, s) for s in site.grid_ssa)
        self.block = tuple(_resolve_const_index(_Shim, s) for s in site.block_ssa)
        self.shared = 0
        if site.dynamic_shared_ssa is not None:
            self.shared = int(consts[site.dynamic_shared_ssa][0])
            cuda = _driver()
            attr = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
            cuda.cuFuncSetAttribute(self.func, attr, self.shared)
        self.plan = []
        for op in site.args:
            if op.ty.startswith("!llvm.ptr"):
                self.plan.append(("ptr", aligned[op.ssa]))
            elif op.ty in ("i64", "index"):
                self.plan.append(("i64", int(consts[op.ssa][0])))
            else:
                raise RuntimeError(f"matmul: unhandled launch-arg {op.ty}")
        self._cuda = _driver()

    def thunk(self, tensors):
        """A zero-arg NO-SYNC launch closure binding `tensors` (kept alive)."""
        import torch

        args = [PtrArg(tensors[i].data_ptr()) if kind == "ptr" else I64Arg(i)
                for kind, i in self.plan]
        keep, argp = _marshal(args)
        cuda, func, grid, block = self._cuda, self.func, self.grid, self.block
        shared = self.shared
        addr = int(ctypes.addressof(argp))
        stream = int(torch.cuda.current_stream().cuda_stream)

        def go(_keep=keep, _argp=argp):  # keep the marshaled buffer alive
            cuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],
                                block[0], block[1], block[2],
                                shared, stream, addr, 0)

        return go
