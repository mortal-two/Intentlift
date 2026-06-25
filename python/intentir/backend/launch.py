"""CUDA Driver-API launch for intent-compiled kernels (M4).

Loads a cubin/PTX module and launches a kernel on torch device pointers, with no
dependency on libmlir_cuda_runtime. Kernels are compiled with bare-ptr calling
convention, so each tensor arg marshals to a single device pointer and the memref
must be statically shaped (R3 finding); index/scalar kernel args pass by value.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Sequence


def _driver():
    try:
        from cuda import cuda  # type: ignore
        return cuda
    except Exception:
        from cuda.bindings import driver as cuda  # type: ignore
        return cuda


def _check(ret, what: str):
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0:
        raise RuntimeError(f"{what}: CUDA driver error {int(err)}")
    return ret[1:] if isinstance(ret, tuple) and len(ret) > 1 else None


class PtrArg(int):
    """A kernel argument that is a raw device pointer (u64)."""


class I64Arg(int):
    """A kernel argument passed by value as a 64-bit index/integer."""


class I32Arg(int):
    """A kernel argument passed by value as a 32-bit integer."""


@dataclass
class Module:
    handle: object

    def function(self, name: str):
        cuda = _driver()
        (func,) = _check(cuda.cuModuleGetFunction(self.handle, name.encode()),
                         "cuModuleGetFunction")
        return func


_ctx_ready = False


def ensure_context() -> None:
    """Make a CUDA context current (torch owns it) and init the driver once."""
    global _ctx_ready
    if _ctx_ready:
        return
    import torch
    torch.empty((1,), device="cuda")  # forces context creation
    cuda = _driver()
    _check(cuda.cuInit(0), "cuInit")
    _ctx_ready = True


def load_module(blob: bytes) -> Module:
    ensure_context()
    cuda = _driver()
    (mod,) = _check(cuda.cuModuleLoadData(blob), "cuModuleLoadData")
    return Module(mod)


def _marshal(args: Sequence[int]):
    cvals = []
    for a in args:
        if isinstance(a, I32Arg):
            cvals.append(ctypes.c_int32(int(a)))
        elif isinstance(a, I64Arg):
            cvals.append(ctypes.c_int64(int(a)))
        else:  # PtrArg or plain int -> device pointer
            cvals.append(ctypes.c_void_p(int(a)))
    arg_ptrs = (ctypes.c_void_p * len(cvals))()
    for i, cv in enumerate(cvals):
        arg_ptrs[i] = ctypes.cast(ctypes.byref(cv), ctypes.c_void_p)
    return cvals, arg_ptrs  # keep cvals alive


def launch(func, grid: tuple[int, int, int], block: tuple[int, int, int],
           args: Sequence[int], shared: int = 0) -> None:
    """Launch and synchronize on torch's current stream."""
    import torch
    cuda = _driver()
    _keep, arg_ptrs = _marshal(args)
    stream = int(torch.cuda.current_stream().cuda_stream)
    ret = cuda.cuLaunchKernel(
        func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
        shared, stream, int(ctypes.addressof(arg_ptrs)), 0,
    )
    _check(ret, "cuLaunchKernel")
    torch.cuda.synchronize()
