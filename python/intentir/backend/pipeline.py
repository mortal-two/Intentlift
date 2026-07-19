"""Verification-neutral IntentIR -> CUDA lowering entry points.

Gate, paper, command-line, and future target frontends may share this module,
but none of their verdicts participate in lowering authorization.  Callers must
first construct an ``IntentFunction`` (or Intent dialect MLIR) whose A/B/C
contract they accept.  This module emits/probes/compiles that exact program and
returns executable cubins plus a neutral lowering record.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import intentir.backend.compile as backend_compile
from intentir.backend.selection import (
    LoweringTarget,
    VariantSelection,
    read_intent_module,
    residual_intent_ops,
    select_variant,
)
from intentir.frontend import emit_module
from intentir.ir.model import IntentFunction


@dataclass(frozen=True)
class LoweredProgram:
    """One compiled CUDA program and the structural decision that produced it."""

    selection: VariantSelection
    kernels: tuple[backend_compile.CompiledKernel, ...]
    source_mlir_sha256: str
    specialized_mlir_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError("lowering must produce at least one CUDA kernel")
        for kernel in self.kernels:
            if not kernel.cubin.startswith(b"\x7fELF"):
                raise ValueError(
                    f"compiled kernel {kernel.kernel_name!r} is not an ELF cubin"
                )
            residuals = residual_intent_ops(kernel.binary_mlir)
            if residuals:
                raise ValueError(
                    f"compiled kernel {kernel.kernel_name!r} retains Intent ops: "
                    f"{list(residuals)}"
                )
        if self.selection.input_sha256 != self.source_mlir_sha256:
            raise ValueError("selection and compiled source hashes disagree")
        if self.selection.pipeline_kind == "specialized":
            if self.specialized_mlir_sha256 != self.selection.probe_output_sha256:
                raise ValueError(
                    "compiled specialized MLIR differs from its structural probe"
                )
        elif self.specialized_mlir_sha256 is not None:
            raise ValueError("generic lowering cannot carry specialized MLIR")

    def to_record(self) -> dict[str, object]:
        """Return JSON-safe compiler provenance without embedding binary bytes."""

        return {
            "selection": self.selection.to_dict(),
            "source_mlir_sha256": self.source_mlir_sha256,
            "specialized_mlir_sha256": self.specialized_mlir_sha256,
            "kernel_count": len(self.kernels),
            "kernels": [
                {
                    "binary_name": kernel.binary_name,
                    "kernel_name": kernel.kernel_name,
                    "cubin_size_bytes": len(kernel.cubin),
                    "cubin_sha256": hashlib.sha256(kernel.cubin).hexdigest(),
                    "cubin_is_elf": kernel.cubin.startswith(b"\x7fELF"),
                    "binary_mlir_sha256": hashlib.sha256(
                        kernel.binary_mlir.encode("utf-8")
                    ).hexdigest(),
                }
                for kernel in self.kernels
            ],
        }


def _existing_path(intent_module: str | os.PathLike[str]) -> Path | None:
    try:
        path = Path(intent_module)
        return path if path.is_file() else None
    except OSError:
        return None


def _compile_generic(
    intent_module: str | os.PathLike[str], source: str
) -> tuple[backend_compile.CompiledKernel, ...]:
    path = _existing_path(intent_module)
    if path is not None:
        return tuple(backend_compile.compile_all_kernels(path))
    with tempfile.TemporaryDirectory(prefix="intentir-lowering-") as directory:
        source_path = Path(directory) / "input.mlir"
        source_path.write_text(source, encoding="utf-8")
        return tuple(backend_compile.compile_all_kernels(source_path))


def lower_to_cuda(
    intent_module: str | os.PathLike[str],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    allow_specialized: bool = True,
) -> LoweredProgram:
    """Compile accepted Intent dialect MLIR through generic or direct CUDA.

    ``only_fast_path`` is useful for an explicit tuning/ablation candidate.  It
    does not force acceptance: the named C++ pass must still consume the whole
    graph, otherwise this function uses the generic fallback.
    """

    source = read_intent_module(intent_module)
    source_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    selected_target = target or LoweringTarget()
    if selected_target.chip != backend_compile.CHIP:
        raise ValueError(
            "lowering target must match the configured CUDA backend: "
            f"requested {selected_target.chip!r}, configured "
            f"{backend_compile.CHIP!r}"
        )
    selection = select_variant(
        source,
        target=selected_target,
        pass_overrides=pass_overrides,
        only_fast_path=only_fast_path,
        allow_specialized=allow_specialized,
    )
    if selection.input_sha256 != source_digest:
        raise RuntimeError("selector did not probe the exact lowering input")

    specialized_digest: str | None = None
    if selection.pipeline_kind == "specialized":
        assert selection.pass_argument is not None
        specialized = backend_compile._run(
            [backend_compile.INTENT_OPT, selection.pass_argument], stdin=source
        )
        specialized_digest = hashlib.sha256(
            specialized.encode("utf-8")
        ).hexdigest()
        if specialized_digest != selection.probe_output_sha256:
            raise RuntimeError(
                "specialized pass output changed between selection and compilation"
            )
        kernels = tuple(
            backend_compile.lower_memref_module_to_kernels(
                specialized, scf_to_cf=selection.scf_to_cf
            )
        )
    else:
        kernels = _compile_generic(intent_module, source)

    return LoweredProgram(
        selection=selection,
        kernels=kernels,
        source_mlir_sha256=source_digest,
        specialized_mlir_sha256=specialized_digest,
    )


def lower_function_to_cuda(
    function: IntentFunction,
    concrete_extents: Mapping[str, int],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    allow_specialized: bool = True,
    fold: int = 0,
) -> LoweredProgram:
    """Emit an accepted flat ``IntentFunction`` and compile it to CUDA.

    Validation policy intentionally remains outside this function.  Gate1/2,
    another verifier, or a trusted handwritten source can all use the same
    backend once they have accepted the function and supplied concrete extents.
    """

    mlir = emit_module(
        function,
        dict(concrete_extents),
        fold=fold,
    )
    return lower_to_cuda(
        mlir,
        target=target,
        pass_overrides=pass_overrides,
        only_fast_path=only_fast_path,
        allow_specialized=allow_specialized,
    )


__all__ = ["LoweredProgram", "lower_function_to_cuda", "lower_to_cuda"]
