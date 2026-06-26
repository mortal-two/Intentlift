"""Generic multi-kernel launcher for the intent IR back-end (M9 orchestration).

Today's launches are hand-specified per fixture: ``test_backend_execute.py`` types
the grid/block/args of each kernel by hand. This module makes the *whole* lowered
op-graph self-orchestrating by parsing the lowered host module that
``compile_all_kernels`` already returns, then executing every ``gpu.launch_func``
in order with the right grid/block/args resolved from the IR.

Public entry point::

    run_program(intent_mlir_path, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]

What the lowered host module actually looks like
------------------------------------------------
The task brief describes the *front-pass* module (memref + arith + linalg). But
``compile_all_kernels`` runs the full device pipeline, so ``binary_mlir`` is the
**fully LLVM-dialect lowered** module. Concretely (see ``lower_to_binary_mlir``)::

    llvm.func @add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, ... ) -> !llvm.struct<...> {
      ...
      %10 = llvm.call @malloc(%9) : (i64) -> !llvm.ptr        # an output/intermediate buffer
      ...
      %16 = llvm.inttoptr %15 : i64 to !llvm.ptr              # its aligned pointer
      ...
      gpu.launch_func @add_kernel::@add_kernel
          blocks in (%4, %3, %5) threads in (%5, %5, %5) : i64
          args(%5 : i64, %6 : i64, %arg1 : !llvm.ptr, %arg8 : !llvm.ptr, %16 : !llvm.ptr)
      llvm.return %23 : !llvm.struct<...>                     # descriptor whose aligned ptr is %16
    }
    gpu.binary @add_kernel [#gpu.object<#nvvm.target<...>, bin = "...">]

So we parse, against this LLVM-dialect form:

* the **host llvm.func** signature -> input tensor names (from the *intent*
  source) and the flattened memref-descriptor block-arg layout (per memref:
  allocated-ptr, *aligned-ptr*, offset, sizes..., strides...; kernels receive the
  **aligned** = 2nd pointer of each descriptor group);
* every ``llvm.mlir.constant(V : ty) : i64`` -> a literal value (grid/block/scalar);
* every ``llvm.call @malloc`` -> one buffer, identified in launch args by its
  ``llvm.inttoptr`` aligned-pointer result, shape from the descriptor's
  ``insertvalue ...[3, i]`` size fields;
* every ``gpu.launch_func`` -> a kernel invocation, classifying each ``args(...)``
  operand by its explicit type annotation (``i64``/``i32`` scalar-by-value,
  ``f32`` 4-byte-float-by-value, ``!llvm.ptr`` a memref pointer that is either an
  input block-arg or a malloc buffer);
* ``llvm.return`` traced through the ``insertvalue ...[1]`` (aligned-ptr) chain ->
  the buffer that is the program output.

Kernel<->cubin matching: in a multi-kernel program the *gpu.func* name can repeat
(matmul: both launches target ``@mm_kernel``); only the ``gpu.binary`` *module*
name (``mm_kernel`` vs ``mm_kernel_0``) is unique. We match each launch site by the
token before ``::`` against ``CompiledKernel.binary_name``.

f32 scalar args: ``launch.py``'s ``_marshal`` only handles i32/i64/pointer; a float
would silently fall through to ``c_void_p`` (a null-ish pointer) and corrupt the
result with no error. The matmul fixture's fill kernel takes an ``f32`` value, so
this module reimplements the marshal + ``cuLaunchKernel`` locally with a
``ctypes.c_float`` branch (we may NOT edit launch.py). We still reuse launch.py's
``load_module`` / ``ensure_context`` (loading is unaffected).
"""

from __future__ import annotations

import ctypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from intentir.backend.compile import CompiledKernel, compile_all_kernels
from intentir.backend.launch import ensure_context, load_module

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch


# --------------------------------------------------------------------------- #
# Scalar kernel-argument wrappers (orchestrate-local; launch.py is untouched). #
# --------------------------------------------------------------------------- #
class _PtrArg(int):
    """Kernel arg: a raw device pointer (u64)."""


class _I64Arg(int):
    """Kernel arg passed by value as a 64-bit index/integer."""


class _I32Arg(int):
    """Kernel arg passed by value as a 32-bit integer."""


class _F32Arg(float):
    """Kernel arg passed by value as a 32-bit float (4 raw bytes)."""


def _parse_f32(s: str) -> float:
    """Parse an MLIR f32 literal: decimal (``1.0``, ``0.0e+00``) or the raw
    IEEE-754 hex form MLIR prints for special values (``0xFF800000`` = -inf)."""
    s = s.strip()
    if s.lower().startswith("0x"):
        import struct
        return struct.unpack(">f", bytes.fromhex(s[2:].rjust(8, "0")))[0]
    return float(s)


def _marshal(args):
    """Build the ctypes arg-pointer array for ``cuLaunchKernel``.

    Mirrors launch.py's ``_marshal`` but adds the f32 branch it lacks.
    """
    cvals = []
    for a in args:
        if isinstance(a, _F32Arg):
            cvals.append(ctypes.c_float(float(a)))
        elif isinstance(a, _I32Arg):
            cvals.append(ctypes.c_int32(int(a)))
        elif isinstance(a, _I64Arg):
            cvals.append(ctypes.c_int64(int(a)))
        else:  # _PtrArg or plain int -> device pointer
            cvals.append(ctypes.c_void_p(int(a)))
    arg_ptrs = (ctypes.c_void_p * len(cvals))()
    for i, cv in enumerate(cvals):
        arg_ptrs[i] = ctypes.cast(ctypes.byref(cv), ctypes.c_void_p)
    return cvals, arg_ptrs  # keep cvals alive


def _launch(func, grid, block, args, shared: int = 0) -> None:
    """Launch + synchronize on torch's current stream (local copy with f32)."""
    import torch

    from intentir.backend.launch import _driver  # type: ignore  # internal helper, read-only

    cuda = _driver()
    _keep, arg_ptrs = _marshal(args)
    stream = int(torch.cuda.current_stream().cuda_stream)
    ret = cuda.cuLaunchKernel(
        func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
        shared, stream, int(ctypes.addressof(arg_ptrs)), 0,
    )
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0:
        raise RuntimeError(f"cuLaunchKernel: CUDA driver error {int(err)}")
    torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# MLIR type -> torch dtype.                                                    #
# --------------------------------------------------------------------------- #
def _torch_dtype(elem: str):
    import torch

    table = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "f64": torch.float64,
        "i64": torch.int64,
        "i32": torch.int32,
        "i16": torch.int16,
        "i8": torch.int8,
        "i1": torch.bool,
    }
    if elem not in table:
        raise NotImplementedError(
            f"orchestrate: unhandled element type {elem!r} (extend _torch_dtype)"
        )
    return table[elem]


# --------------------------------------------------------------------------- #
# Intent-source signature parsing (input names / shapes / dtypes).            #
# --------------------------------------------------------------------------- #
@dataclass
class _IntentInput:
    name: str
    shape: tuple[int, ...]
    dtype_str: str


def _parse_intent_signature(intent_src: str) -> list[_IntentInput]:
    """Parse ``func.func @name(%a: tensor<8x16xf32>, ...) -> ...`` from intent IR.

    Returns inputs in positional order. Used only to recover *names* (so callers
    can key the ``inputs`` dict by name) and to sanity-check shapes/dtypes.
    """
    m = re.search(r"func\.func\s+@\w+\s*\(([^)]*)\)", intent_src)
    if not m:
        raise ValueError("orchestrate: could not find a func.func signature in intent IR")
    arglist = m.group(1).strip()
    inputs: list[_IntentInput] = []
    if not arglist:
        return inputs
    # Split on top-level commas (tensor types here carry no nested commas).
    for piece in arglist.split(","):
        piece = piece.strip()
        if not piece:
            continue
        am = re.match(
            r"%(\w+)\s*:\s*tensor<([0-9x]*)x?([a-z0-9]+)>", piece
        )
        if not am:
            am2 = re.match(r"%(\w+)\s*:\s*tensor<([a-z0-9]+)>", piece)
            if am2:  # 0-d tensor
                inputs.append(_IntentInput(am2.group(1), (), am2.group(2)))
                continue
            raise NotImplementedError(
                f"orchestrate: unhandled intent arg type in {piece!r}"
            )
        name = am.group(1)
        dims_str = am.group(2)
        elem = am.group(3)
        dims = tuple(int(d) for d in dims_str.split("x") if d != "")
        inputs.append(_IntentInput(name, dims, elem))
    return inputs


# --------------------------------------------------------------------------- #
# Host LLVM-func parsing.                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class _LaunchOperand:
    ssa: str          # e.g. "%5", "%arg1", "%16"
    ty: str           # e.g. "i64", "f32", "!llvm.ptr"


@dataclass
class _LaunchSite:
    binary_module: str            # token before "::"  -> matches CompiledKernel.binary_name
    func_symbol: str              # token after  "::"  -> the gpu.func name
    grid_ssa: tuple[str, str, str]
    block_ssa: tuple[str, str, str]
    args: list[_LaunchOperand]
    dynamic_shared_ssa: str | None = None  # dynamic_shared_memory_size operand ssa


@dataclass
class _HostModule:
    func_name: str
    sig_args: list[str]                       # ordered block-arg SSA names ("%arg0", ...)
    constants: dict[str, tuple[str, str]]     # ssa -> (raw_value, type)
    aligned_ptr_blockargs: dict[str, int]     # aligned-ptr block-arg ssa -> input index
    malloc_buffers: dict[str, "_Buffer"]      # aligned-ptr ssa (inttoptr result) -> buffer
    loose_buffers: dict[str, "_LooseBuffer"]  # inttoptr ssa -> loose intermediate buffer
    launches: list[_LaunchSite]
    return_aligned_ptrs: list[str]            # aligned-ptr ssa per result, in order
    body_text: str = field(default="", repr=False)


@dataclass
class _Buffer:
    aligned_ssa: str
    shape: tuple[int, ...]
    elem: str


@dataclass
class _LooseBuffer:
    """A malloc'd intermediate buffer passed to kernels as the raw aligned
    ``llvm.inttoptr`` pointer, never packed into a memref-descriptor struct.

    Such buffers carry no struct shape in the host IR, only an element count
    (from the ``getelementptr`` that computes the malloc byte size). We allocate
    a flat 1-D tensor of ``n_elems`` of ``elem`` dtype — enough storage for the
    kernel's bare-ptr writes; torch's allocator is >=256-byte aligned, which
    satisfies the 64-byte alignment the host computes for the kernel pointer.
    """
    aligned_ssa: str        # the inttoptr result (the kernel-arg pointer)
    malloc_ssa: str         # the llvm.call @malloc result
    n_elems: int            # element count (from getelementptr %null[N])
    elem: str               # element type (from the same getelementptr)


def _host_func_text(binary_mlir: str) -> tuple[str, str]:
    """Return (func_name, body-up-to-launches) for the single host ``llvm.func``.

    The host function is the one containing ``gpu.launch_func`` (``@malloc`` decl
    has no body). We slice from its ``{`` to the matching closing of the gpu
    launches region via a simple brace counter, which is robust to the nested
    ``!llvm.struct<...>`` angle brackets (we only count ``{`` / ``}``).
    """
    func_starts = [(mm.start(), mm.group(1), mm.end())
                   for mm in re.finditer(r"llvm\.func\s+@(\w+)\s*\(", binary_mlir)]
    for k, (start, name, end) in enumerate(func_starts):
        # find the body opening brace after the signature/attrs
        open_idx = binary_mlir.find("{", end)
        if open_idx == -1:
            continue
        # A declaration-only func (e.g. @malloc) has no body brace before the
        # next llvm.func; reject a brace that belongs to a *later* function.
        next_start = func_starts[k + 1][0] if k + 1 < len(func_starts) else len(binary_mlir)
        if open_idx >= next_start:
            continue
        depth = 0
        i = open_idx
        while i < len(binary_mlir):
            c = binary_mlir[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    body = binary_mlir[open_idx + 1 : i]
                    if "gpu.launch_func" in body:
                        sig = binary_mlir[start:open_idx]
                        return name, sig + "{" + body + "}"
                    break
            i += 1
    raise ValueError("orchestrate: no host llvm.func containing gpu.launch_func found")


def _parse_signature_blockargs(func_text: str) -> tuple[str, list[str], list[str]]:
    """Return (func_name, ordered_blockarg_ssa, ptr_blockarg_ssa).

    ptr_blockarg_ssa is the subset typed ``!llvm.ptr`` in declaration order.
    """
    m = re.search(r"llvm\.func\s+@(\w+)\s*\((.*?)\)\s*(?:->|attributes|\{)",
                  func_text, re.DOTALL)
    if not m:
        raise ValueError("orchestrate: could not parse host llvm.func signature")
    name = m.group(1)
    sig = m.group(2)
    ordered: list[str] = []
    ptrs: list[str] = []
    for am in re.finditer(r"(%\w+)\s*:\s*([^,]+?)(?:,|$)", sig):
        ssa = am.group(1)
        ty = am.group(2).strip()
        ordered.append(ssa)
        if ty.startswith("!llvm.ptr"):
            ptrs.append(ssa)
    return name, ordered, ptrs


def _map_input_aligned_ptrs(ptr_blockargs: list[str]) -> dict[str, int]:
    """Map each input's *aligned* (2nd) pointer block-arg -> input index.

    Bare-ptr-disabled host ABI flattens every memref descriptor as
    (allocated-ptr, aligned-ptr, offset, sizes..., strides...). Within the
    block-arg list the pointers appear in consecutive pairs (allocated, aligned)
    per memref, because offset/sizes/strides are i64 and interleave *between*
    descriptors, not between the two pointers of one descriptor. So the pointer
    block-args, taken two at a time, are (alloc0, align0, alloc1, align1, ...);
    every odd-indexed pointer is an aligned pointer = the value the kernel
    receives for that input.
    """
    if len(ptr_blockargs) % 2 != 0:
        raise NotImplementedError(
            "orchestrate: host signature has an odd number of !llvm.ptr block-args "
            f"({len(ptr_blockargs)}); expected (allocated, aligned) pairs per memref"
        )
    aligned: dict[str, int] = {}
    for idx in range(len(ptr_blockargs) // 2):
        aligned[ptr_blockargs[2 * idx + 1]] = idx
    return aligned


def _parse_constants(func_text: str) -> dict[str, tuple[str, str]]:
    """ssa -> (raw_value, mlir_type) for every ``llvm.mlir.constant``."""
    consts: dict[str, tuple[str, str]] = {}
    for m in re.finditer(
        r"(%\w+)\s*=\s*llvm\.mlir\.constant\(([^:)]+?)\s*:\s*[^)]+\)\s*:\s*(\S+)",
        func_text,
    ):
        consts[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return consts


def _parse_malloc_buffers(func_text: str) -> dict[str, _Buffer]:
    """Identify each malloc'd buffer by its aligned-pointer ``inttoptr`` result.

    A buffer descriptor is built by a chain of ``insertvalue``s into an
    ``llvm.struct``. The aligned pointer (struct field [1]) is the
    ``llvm.inttoptr`` result; the sizes are inserted at ``[3, i]``. We group
    insertvalues by their struct-SSA lineage to recover each buffer's shape.
    """
    # Map each undef-struct lineage: chase insertvalue dst<-src to a root undef.
    # value_inserted[ssa_result] = (src_ssa, inserted_ssa, position)
    inserts: list[tuple[str, str, str, str]] = []  # (result, src, inserted, pos)
    for m in re.finditer(
        r"(%\w+)\s*=\s*llvm\.insertvalue\s+(%\w+),\s*(%\w+)\[([0-9, ]+)\]",
        func_text,
    ):
        inserts.append((m.group(1), m.group(3), m.group(2), m.group(4).strip()))

    # Build forward chains rooted at llvm.mlir.undef structs.
    undefs = set(re.findall(r"(%\w+)\s*=\s*llvm\.mlir\.undef", func_text))
    # For each insertvalue, follow src back to find the originating undef root.
    src_to_result: dict[str, str] = {r: r for (r, _s, _i, _p) in inserts}
    # parent map: result -> src
    parent = {r: s for (r, s, _i, _p) in inserts}

    def root(ssa: str) -> str:
        seen = set()
        cur = ssa
        while cur in parent and cur not in seen:
            seen.add(cur)
            cur = parent[cur]
        return cur

    # Collect, per root, the aligned ptr ([1]) and size fields ([3, i]).
    by_root: dict[str, dict] = {}
    for (result, src, inserted, pos) in inserts:
        r = root(result)
        d = by_root.setdefault(r, {"aligned": None, "sizes": {}})
        if pos == "1":
            d["aligned"] = inserted
        else:
            sm = re.match(r"3,\s*(\d+)", pos)
            if sm:
                d["sizes"][int(sm.group(1))] = inserted

    # Map size-field SSA values to integers via the constant table.
    consts = _parse_constants(func_text)

    # Element type of malloc'd buffers: from getelementptr "[N] : (...) -> ..., <elem>".
    gep = re.search(r"llvm\.getelementptr\s+%\w+\[\d+\][^,]*,\s*([a-z0-9]+)", func_text)
    elem = gep.group(1) if gep else "f32"

    buffers: dict[str, _Buffer] = {}
    # Only roots that descend from an undef AND whose aligned ptr is an inttoptr
    # (a malloc buffer) are buffers; the input descriptors are not rebuilt here.
    inttoptr_results = set(
        re.findall(r"(%\w+)\s*=\s*llvm\.inttoptr", func_text)
    )
    for r, d in by_root.items():
        if r not in undefs:
            continue
        aligned = d["aligned"]
        if aligned is None or aligned not in inttoptr_results:
            continue
        sizes: list[int] = []
        if d["sizes"]:
            for i in range(max(d["sizes"]) + 1):
                ssa = d["sizes"].get(i)
                if ssa is None or ssa not in consts:
                    raise NotImplementedError(
                        "orchestrate: could not resolve buffer size field "
                        f"[3,{i}] (ssa={ssa}) to a constant"
                    )
                sizes.append(int(consts[ssa][0]))
        buffers[aligned] = _Buffer(aligned_ssa=aligned, shape=tuple(sizes), elem=elem)
    return buffers


def _parse_loose_buffers(func_text: str) -> dict[str, _LooseBuffer]:
    """Identify malloc'd *intermediate* buffers passed as loose aligned pointers.

    Unlike the returned output (which is packed into an ``insertvalue`` struct),
    intermediate buffers between kernels are handed to ``gpu.launch_func`` as the
    bare ``llvm.inttoptr`` aligned pointer straight off ``malloc``. Each such
    buffer is recovered as::

        %m = llvm.call @malloc(%s)          # %s = byte size
        ... %a = ptrtoint %m ; alignment arithmetic ...
        %p = llvm.inttoptr %q : i64 to !llvm.ptr   # the kernel-arg pointer

    Returns ``{inttoptr_ssa -> _LooseBuffer}`` keyed by the pointer the kernel
    actually receives.

    Size is taken by *tracing the malloc's size operand*, NOT by source-order
    proximity to a getelementptr: the size geps are CSE-shared across mallocs
    (e.g. two [4]-element buffers reference the same gep), so the nearest
    preceding gep is unreliable and can mis-size. We chase
    ``malloc(%s)`` -> ``%s = add %p, 64`` -> ``%p = ptrtoint %g``
    -> ``%g = getelementptr %null[N] ..., <elem>`` to read N and elem per buffer.
    The malloc->inttoptr pairing, by contrast, *is* source-order safe: the
    alignment chain is emitted fresh per malloc.
    """
    # defs[ssa] = the verbatim RHS of its single defining op (SSA is unique).
    defs: dict[str, str] = {}
    for m in re.finditer(r"(%\w+)\s*=\s*([^\n]+)", func_text):
        defs.setdefault(m.group(1), m.group(2).strip())

    # getelementptr def -> (N, elem). Matches "llvm.getelementptr %x[N] : (...) , elem".
    def _gep_info(rhs: str):
        gm = re.match(
            r"llvm\.getelementptr\s+%\w+\[(\d+)\][^,]*,\s*([A-Za-z0-9_!.<>]+)", rhs
        )
        if not gm:
            return None
        return int(gm.group(1)), gm.group(2).strip()

    def _trace_size(size_ssa: str):
        """size_ssa -> (n_elems, elem) by walking add/ptrtoint/getelementptr."""
        seen = set()
        cur = size_ssa
        while cur is not None and cur not in seen and cur in defs:
            seen.add(cur)
            rhs = defs[cur]
            g = _gep_info(rhs)
            if g is not None:
                return g
            # llvm.add %p, %c   -> follow the operand that is NOT the 64-const
            am = re.match(r"llvm\.add\s+(%\w+),\s*(%\w+)", rhs)
            if am:
                a, b = am.group(1), am.group(2)
                # prefer the ptrtoint-bearing side; fall back to first.
                cur = a
                # if a resolves to a constant and b doesn't, swap
                if a in defs and defs[a].startswith("llvm.mlir.constant") and \
                        b in defs and not defs[b].startswith("llvm.mlir.constant"):
                    cur = b
                continue
            # llvm.ptrtoint %g -> follow %g
            pm = re.match(r"llvm\.ptrtoint\s+(%\w+)", rhs)
            if pm:
                cur = pm.group(1)
                continue
            break
        return None

    # Mallocs and inttoptrs in source order.
    mallocs = [(m.start(), m.group(1), m.group(2))
               for m in re.finditer(
                   r"(%\w+)\s*=\s*llvm\.call\s+@malloc\((%\w+)\)", func_text)]
    inttoptrs = [(m.start(), m.group(1))
                 for m in re.finditer(r"(%\w+)\s*=\s*llvm\.inttoptr", func_text)]

    loose: dict[str, _LooseBuffer] = {}
    for _pos, malloc_ssa, size_ssa in mallocs:
        # pair with the first inttoptr that appears after this malloc in source.
        aligned = None
        for ipos, issa in inttoptrs:
            if ipos > _pos:
                aligned = issa
                break
        if aligned is None:
            raise NotImplementedError(
                f"orchestrate: malloc {malloc_ssa} has no following llvm.inttoptr "
                "(cannot recover its aligned device pointer)"
            )
        info = _trace_size(size_ssa)
        if info is None:
            raise NotImplementedError(
                f"orchestrate: could not trace malloc {malloc_ssa} size operand "
                f"{size_ssa} to a getelementptr element count"
            )
        n_elems, elem = info
        loose[aligned] = _LooseBuffer(
            aligned_ssa=aligned, malloc_ssa=malloc_ssa, n_elems=n_elems, elem=elem
        )
    return loose


def _parse_launches(func_text: str) -> list[_LaunchSite]:
    sites: list[_LaunchSite] = []
    pat = re.compile(
        r"gpu\.launch_func\s+@(\w+)::@(\w+)\s+"
        r"blocks\s+in\s+\((%\w+),\s*(%\w+),\s*(%\w+)\)\s+"
        r"threads\s+in\s+\((%\w+),\s*(%\w+),\s*(%\w+)\)\s*"
        r"(?::\s*\S+\s*)?"
        r"(?:dynamic_shared_memory_size\s+(%\w+)\s*)?"  # M5 tiled smem cache
        r"args\(([^)]*)\)"
    )
    for m in pat.finditer(func_text):
        binmod, fsym = m.group(1), m.group(2)
        grid = (m.group(3), m.group(4), m.group(5))
        block = (m.group(6), m.group(7), m.group(8))
        dyn_shared = m.group(9)
        argstr = m.group(10)
        operands: list[_LaunchOperand] = []
        for piece in _split_top_level(argstr):
            piece = piece.strip()
            if not piece:
                continue
            om = re.match(r"(%\w+)\s*:\s*(.+)$", piece)
            if not om:
                raise NotImplementedError(
                    f"orchestrate: unparsable launch operand {piece!r}"
                )
            operands.append(_LaunchOperand(om.group(1), om.group(2).strip()))
        sites.append(_LaunchSite(binmod, fsym, grid, block, operands, dyn_shared))
    return sites


def _split_top_level(s: str) -> list[str]:
    """Split on commas not nested inside <>, (), [] (defensive for arg lists)."""
    out: list[str] = []
    depth = 0
    cur = []
    for ch in s:
        if ch in "<([":
            depth += 1
        elif ch in ">)]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


def _parse_return_aligned_ptrs(func_text: str) -> list[str]:
    """Trace ``llvm.return %X`` to each output's aligned pointer, in result order.

    A single-tensor result returns one flat memref descriptor
    ``struct<(ptr, ptr, i64, ...)>`` whose field [1] is the aligned ptr. A
    multi-tensor result (e.g. a loop body yielding ``m_new, l_new, O_new``)
    returns a struct-of-structs ``struct<(struct<...>, struct<...>, ...)>``: each
    top-level field [i] is itself a memref descriptor, inserted by an
    ``insertvalue ...[i]`` whose inserted value is another built struct (i.e. it
    is itself an insertvalue result), and each inner descriptor's [1] is its
    aligned ptr. Returns ``[]`` if no return ptr can be recovered.
    """
    rm = re.search(r"llvm\.return\s+(%\w+)", func_text)
    if not rm:
        return []
    ret_ssa = rm.group(1)
    # Build insertvalue maps: parent (dst<-src), and per-result (pos, inserted).
    parent: dict[str, str] = {}
    pos_of: dict[str, str] = {}
    inserted_of: dict[str, str] = {}
    for m in re.finditer(
        r"(%\w+)\s*=\s*llvm\.insertvalue\s+(%\w+),\s*(%\w+)\[([0-9, ]+)\]",
        func_text,
    ):
        result, inserted, src, pos = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        parent[result] = src
        pos_of[result] = pos
        inserted_of[result] = inserted

    def aligned_of_struct(struct_ssa: str) -> str | None:
        """First [1] insert walking up a single descriptor's chain = aligned ptr."""
        seen: set[str] = set()
        cur = struct_ssa
        while cur is not None and cur not in seen:
            seen.add(cur)
            if pos_of.get(cur) == "1":
                return inserted_of[cur]
            cur = parent.get(cur)
        return None

    # Top-level field inserts in the returned struct's chain: pos is a single int
    # AND the inserted value is itself a built struct (an insertvalue result).
    fields: dict[int, str] = {}
    seen: set[str] = set()
    cur: str | None = ret_ssa
    while cur is not None and cur not in seen:
        seen.add(cur)
        p = pos_of.get(cur)
        if p is not None and p.isdigit() and inserted_of[cur] in pos_of:
            fields.setdefault(int(p), inserted_of[cur])
        cur = parent.get(cur)

    if fields:  # multi-output: one aligned ptr per top-level field, in order
        out: list[str] = []
        for i in sorted(fields):
            a = aligned_of_struct(fields[i])
            if a is not None:
                out.append(a)
        return out
    a = aligned_of_struct(ret_ssa)
    return [a] if a is not None else []


def _parse_host_module(binary_mlir: str, intent_inputs: list[_IntentInput]) -> _HostModule:
    _func_name, func_text = _host_func_text(binary_mlir)
    sig_name, ordered, ptr_blockargs = _parse_signature_blockargs(func_text)
    aligned_map = _map_input_aligned_ptrs(ptr_blockargs)
    if intent_inputs and len(aligned_map) != len(intent_inputs):
        raise NotImplementedError(
            "orchestrate: host has "
            f"{len(aligned_map)} input memrefs but intent signature has "
            f"{len(intent_inputs)} tensor args; cannot map inputs unambiguously"
        )
    struct_buffers = _parse_malloc_buffers(func_text)
    loose_buffers = _parse_loose_buffers(func_text)
    # The struct-packed (returned) buffer wins: it carries real shape+dtype, so
    # drop any loose entry that the struct path already shaped correctly.
    for ssa in struct_buffers:
        loose_buffers.pop(ssa, None)
    return _HostModule(
        func_name=sig_name,
        sig_args=ordered,
        constants=_parse_constants(func_text),
        aligned_ptr_blockargs=aligned_map,
        malloc_buffers=struct_buffers,
        loose_buffers=loose_buffers,
        launches=_parse_launches(func_text),
        return_aligned_ptrs=_parse_return_aligned_ptrs(func_text),
        body_text=func_text,
    )


# --------------------------------------------------------------------------- #
# Execution.                                                                   #
# --------------------------------------------------------------------------- #
def _resolve_const_index(host: _HostModule, ssa: str) -> int:
    if ssa not in host.constants:
        raise NotImplementedError(
            f"orchestrate: grid/block extent {ssa} is not a literal constant"
        )
    return int(host.constants[ssa][0])


def run_program(intent_mlir_path, inputs, compile_all=compile_all_kernels):
    """Compile *intent_mlir_path* and execute its whole kernel graph on the GPU.

    Parameters
    ----------
    intent_mlir_path : str | PathLike
        Path to an intent-dialect ``.mlir`` module (single- or multi-kernel).
    inputs : dict[str, torch.Tensor]
        Maps each intent function input *name* (from the ``func.func`` signature,
        e.g. ``"a"``, ``"b"``) to a CUDA torch tensor. If a name is missing, the
        positional key ``"arg{i}"`` (0-based) is accepted as a fallback.
    compile_all : callable, optional
        The ``intent_mlir_path -> list[CompiledKernel]`` compiler. Defaults to the
        per-op pipeline; pass ``compile_all_kernels_fused`` (the fused+tiled
        pipeline) to execute a pointwise chain as one fused kernel. The host module
        is parsed generically, so a fused single-kernel graph runs unchanged.

    Returns
    -------
    dict[str, torch.Tensor]
        The program outputs. Keyed by ``"out{i}"`` (0-based, in return order).
    """
    import torch

    ensure_context()

    intent_src = Path(intent_mlir_path).read_text()
    intent_inputs = _parse_intent_signature(intent_src)

    kernels = compile_all(intent_mlir_path)
    if not kernels:
        raise RuntimeError("orchestrate: compile_all_kernels produced no kernels")
    binary_mlir = kernels[0].binary_mlir
    by_binary: dict[str, CompiledKernel] = {k.binary_name: k for k in kernels}

    host = _parse_host_module(binary_mlir, intent_inputs)

    # --- bind input tensors by aligned-ptr block-arg -> input index -------- #
    # aligned_ptr_blockargs: %argK (aligned ptr) -> input index i
    input_tensor_by_index: dict[int, "torch.Tensor"] = {}
    for blockarg_ssa, idx in host.aligned_ptr_blockargs.items():
        # Determine the user key for this input index.
        key = None
        if idx < len(intent_inputs):
            name = intent_inputs[idx].name
            if name in inputs:
                key = name
        if key is None:
            fallback = f"arg{idx}"
            if fallback in inputs:
                key = fallback
        if key is None:
            raise KeyError(
                f"orchestrate: no tensor supplied for input index {idx} "
                f"(tried name "
                f"{intent_inputs[idx].name!r} and 'arg{idx}')"
                if idx < len(intent_inputs)
                else f"orchestrate: no tensor supplied for input index {idx} (tried 'arg{idx}')"
            )
        t = inputs[key]
        if not t.is_cuda:
            raise ValueError(f"orchestrate: input {key!r} must be a CUDA tensor")
        input_tensor_by_index[idx] = t.contiguous()

    # --- allocate every malloc buffer ------------------------------------- #
    # Struct-packed buffers (the returned output) get their real shape; loose
    # intermediates get a flat 1-D buffer of the right element count. One
    # SSA->tensor entry per buffer means a pointer reused across kernels
    # (e.g. softmax's exp buffer) resolves to the same storage automatically.
    buffer_tensor_by_ssa: dict[str, "torch.Tensor"] = {}
    for ssa, buf in host.malloc_buffers.items():
        dtype = _torch_dtype(buf.elem)
        buffer_tensor_by_ssa[ssa] = torch.zeros(
            *buf.shape, dtype=dtype, device="cuda"
        ).contiguous()
    for ssa, lbuf in host.loose_buffers.items():
        dtype = _torch_dtype(lbuf.elem)
        # torch.empty is fine: every reduce buffer is overwritten by its explicit
        # identity-fill kernel (-inf for max, 0 for sum) before first read.
        buffer_tensor_by_ssa[ssa] = torch.empty(
            (lbuf.n_elems,), dtype=dtype, device="cuda"
        ).contiguous()

    # --- map each launch operand to a concrete launch arg ------------------ #
    def _resolve_ptr(ssa: str) -> "torch.Tensor":
        if ssa in host.aligned_ptr_blockargs:
            return input_tensor_by_index[host.aligned_ptr_blockargs[ssa]]
        if ssa in buffer_tensor_by_ssa:
            return buffer_tensor_by_ssa[ssa]
        raise NotImplementedError(
            f"orchestrate: pointer operand {ssa} is neither an input aligned-ptr, "
            f"a struct-packed buffer, nor a loose malloc buffer "
            f"(unhandled host construct)"
        )

    # --- execute each launch in order -------------------------------------- #
    for site in host.launches:
        ck = by_binary.get(site.binary_module)
        if ck is None:
            raise RuntimeError(
                f"orchestrate: no compiled cubin for gpu.binary @{site.binary_module} "
                f"(have {sorted(by_binary)})"
            )
        func = load_module(ck.cubin).function(site.func_symbol)
        # NB: ck.kernel_name == site.func_symbol; both fixtures agree, but we
        # match the *binary* module name (unique) not the func name (may repeat).

        grid = tuple(_resolve_const_index(host, s) for s in site.grid_ssa)
        block = tuple(_resolve_const_index(host, s) for s in site.block_ssa)

        call_args = []
        for op in site.args:
            ty = op.ty
            if ty.startswith("!llvm.ptr"):
                t = _resolve_ptr(op.ssa)
                call_args.append(_PtrArg(t.data_ptr()))
            elif ty in ("i64", "index"):
                if op.ssa not in host.constants:
                    raise NotImplementedError(
                        f"orchestrate: integer scalar arg {op.ssa} is not a constant"
                    )
                call_args.append(_I64Arg(int(host.constants[op.ssa][0])))
            elif ty == "i32":
                if op.ssa not in host.constants:
                    raise NotImplementedError(
                        f"orchestrate: i32 scalar arg {op.ssa} is not a constant"
                    )
                call_args.append(_I32Arg(int(host.constants[op.ssa][0])))
            elif ty == "f32":
                if op.ssa not in host.constants:
                    raise NotImplementedError(
                        f"orchestrate: f32 scalar arg {op.ssa} is not a constant"
                    )
                call_args.append(_F32Arg(_parse_f32(host.constants[op.ssa][0])))
            else:
                raise NotImplementedError(
                    f"orchestrate: unhandled launch-arg type {ty!r} (operand {op.ssa})"
                )

        _launch(func, grid, block, call_args)

    # --- collect outputs --------------------------------------------------- #
    # One ``out{i}`` per function result, in return order (multi-result programs,
    # e.g. a loop body yielding several carries, return them all).
    outputs: dict[str, "torch.Tensor"] = {}
    if host.return_aligned_ptrs:
        for i, ssa in enumerate(host.return_aligned_ptrs):
            if ssa in buffer_tensor_by_ssa:
                outputs[f"out{i}"] = buffer_tensor_by_ssa[ssa]
            elif ssa in host.aligned_ptr_blockargs:
                # an input is returned as-is (identity program)
                outputs[f"out{i}"] = input_tensor_by_index[host.aligned_ptr_blockargs[ssa]]
            else:
                raise NotImplementedError(
                    f"orchestrate: returned aligned ptr {ssa} is not a known buffer/input"
                )
    else:
        # Fallback: if exactly one malloc buffer exists, treat it as the output.
        if len(buffer_tensor_by_ssa) == 1:
            outputs["out0"] = next(iter(buffer_tensor_by_ssa.values()))
        else:
            raise NotImplementedError(
                "orchestrate: could not determine program output(s) from llvm.return "
                f"(found {len(buffer_tensor_by_ssa)} malloc buffers)"
            )
    return outputs


__all__ = ["run_program"]
