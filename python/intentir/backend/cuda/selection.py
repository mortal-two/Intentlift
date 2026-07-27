"""Target-aware structural selection for the real CUDA lowering backend.

This module is deliberately independent of every verification pipeline.  Its
input is already-accepted Intent dialect MLIR; selection probes the registered
C++ conversion passes and chooses a specialized path only when one pass
consumes the complete graph into one or more compiler-emitted ``gpu.launch``
operations. Otherwise the generic Intent -> Linalg -> CUDA path remains the
mandatory fallback.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import intentir.backend.cuda.compile as backend_compile


_INTENT_OP_NAMES = (
    "arg_reduce",
    "cast",
    "concat",
    "const",
    "contraction",
    "conv",
    "flip",
    "gather",
    "iota",
    "loop",
    "map",
    "map_binary",
    "map_unary",
    "masked_scatter",
    "masked_select",
    "nonzero",
    "pad",
    "polar",
    "pool",
    "rand",
    "reduce",
    "scan",
    "scatter",
    "select",
    "sort",
    "tile",
    "transpose",
    "triu",
    "upsample",
    "yield",
)
_INTENT_OP_ALTERNATION = "|".join(
    re.escape(name)
    for name in sorted(_INTENT_OP_NAMES, key=lambda item: (-len(item), item))
)
_INTENT_OPERATION = re.compile(
    rf'(?<![@A-Za-z0-9_.$])"?(intent\.(?:{_INTENT_OP_ALTERNATION}))\b"?'
)
_GPU_LAUNCH = re.compile(
    r"(?<![A-Za-z0-9_.$])gpu\.launch(?![A-Za-z0-9_.$])"
)
_PORTABLE_BLOCK_CUDA_CHIPS = ("sm_90a", "sm_120a")


def residual_intent_ops(mlir: str) -> tuple[str, ...]:
    """Return executable Intent operations that remain in an MLIR module."""

    return tuple(sorted(set(_INTENT_OPERATION.findall(mlir))))


def gpu_launch_count(mlir: str) -> int:
    return len(_GPU_LAUNCH.findall(mlir))


@dataclass(frozen=True)
class LoweringTarget:
    """Backend target facts that may participate in variant selection."""

    backend: str = "cuda"
    chip: str = field(default_factory=lambda: backend_compile.CHIP)
    features: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.backend != "cuda":
            raise ValueError(f"unsupported lowering backend {self.backend!r}")
        if not isinstance(self.chip, str) or not self.chip.strip():
            raise ValueError("lowering target chip must be a non-empty string")
        normalized = frozenset(
            item.strip()
            for item in self.features
            if isinstance(item, str) and item.strip()
        )
        if len(normalized) != len(self.features):
            raise ValueError("lowering target features must be non-empty strings")
        object.__setattr__(self, "chip", self.chip.strip())
        object.__setattr__(self, "features", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "chip": self.chip,
            "features": sorted(self.features),
        }


@dataclass(frozen=True)
class PassDescriptor:
    fast_path_id: str
    pass_argument: str
    supported_chips: tuple[str, ...]
    description: str
    scf_to_cf: bool = False
    required_markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProbeAttempt:
    fast_path_id: str
    pass_argument: str
    target_match: bool
    status: str
    direct_hit: bool
    residual_ops: tuple[str, ...] = ()
    gpu_launch_count: int = 0
    probe_output_sha256: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "target_match": self.target_match,
            "status": self.status,
            "direct_hit": self.direct_hit,
            "residual_ops": list(self.residual_ops),
            "gpu_launch_count": self.gpu_launch_count,
            "probe_output_sha256": self.probe_output_sha256,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class VariantSelection:
    pipeline_kind: str
    target: LoweringTarget
    input_sha256: str
    fast_path_id: str | None
    pass_argument: str | None
    scf_to_cf: bool
    probe_output_sha256: str | None
    attempts: tuple[ProbeAttempt, ...]
    generic_fallback: bool = True

    def __post_init__(self) -> None:
        if self.pipeline_kind not in {"generic", "specialized"}:
            raise ValueError(f"invalid pipeline kind {self.pipeline_kind!r}")
        if self.pipeline_kind == "specialized":
            if not self.fast_path_id or not self.pass_argument:
                raise ValueError("specialized selection requires an id and pass")
            if not self.probe_output_sha256:
                raise ValueError("specialized selection requires a probe hash")
        elif self.fast_path_id is not None or self.pass_argument is not None:
            raise ValueError("generic selection cannot carry a specialized pass")

    def to_dict(self) -> dict[str, object]:
        return {
            "pipeline_kind": self.pipeline_kind,
            "target": self.target.to_dict(),
            "input_sha256": self.input_sha256,
            "fast_path_id": self.fast_path_id,
            "pass_argument": self.pass_argument,
            "scf_to_cf": self.scf_to_cf,
            "probe_output_sha256": self.probe_output_sha256,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "generic_fallback": self.generic_fallback,
        }


DEFAULT_PASSES: tuple[PassDescriptor, ...] = (
    PassDescriptor(
        fast_path_id="cuda.xformers.splitk-forward.flat-rank5.mma.v2",
        pass_argument=(
            "-convert-intent-loop-contraction-to-gpu="
            "num-stages=1 captured-flat-only=true block-m=16 block-n=32 "
            "fast-f32-lse-log=true"
        ),
        supported_chips=("sm_120a",),
        description="flat rank-5 xFormers split-K forward MMA",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_forward.flat_rank4.mma.v1",
        pass_argument="-convert-intent-flash-mma-to-gpu=num-stages=2",
        supported_chips=("sm_120a",),
        description="flat rank-4 FlashAttention forward MMA",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.flash_backward.flat_rank4.mma.v1",
        pass_argument=(
            "-convert-intent-attention-backward-to-gpu="
            "flat-shared-d-pad=8 flat-shared-pds-pad=8 flat-dq-init=prelaunch"
        ),
        supported_chips=("sm_120a",),
        description="flat rank-4 FlashAttention backward MMA",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.loop_contraction.online_softmax.v1",
        pass_argument="-convert-intent-loop-contraction-to-gpu",
        supported_chips=("sm_120a",),
        description="general fused loop-contraction/online-softmax graph",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.splitk.reduce-varargs.v1",
        pass_argument="-convert-intent-splitk-reduce-varargs-to-gpu",
        supported_chips=("sm_120a",),
        description="xFormers split-K varargs reduction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.splitk.reduce-forward.v1",
        pass_argument="-convert-intent-splitk-reduce-to-gpu",
        supported_chips=("sm_120a",),
        description="split-K attention forward reduction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.splitk.reduce-backward.v1",
        pass_argument="-convert-intent-splitk-reduce-backward-to-gpu",
        supported_chips=("sm_120a",),
        description="split-K attention backward reduction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        # Keep the original public ID for schedule/CLI compatibility.  The
        # implementation now accepts matching f16 or bf16 storage.
        fast_path_id="cuda.linear.f16_f32.bias_saved.mma_sync.v1",
        pass_argument=(
            "-convert-intent-contraction-to-gpu-mma=num-stages=5 "
            "block-m=16 block-n=32 block-k=32 smem-pad-k=8"
        ),
        supported_chips=("sm_120a",),
        description="tuned f16/bf16 tensor-core contraction with f32 accumulate",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.contraction.mma-warp-specialized.v1",
        pass_argument="-convert-intent-contraction-to-gpu-mma-ws",
        supported_chips=("sm_120a",),
        description="warp-specialized tensor-core contraction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.contraction.mma-sync.v1",
        pass_argument="-convert-intent-contraction-to-gpu-mma",
        supported_chips=("sm_120a",),
        description="tensor-core contraction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.contraction.tiled.v1",
        pass_argument="-convert-intent-contraction-to-gpu",
        supported_chips=("sm_120a",),
        description="tiled-K contraction",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.conv.implicit-gemm-mma.v1",
        pass_argument="-convert-intent-conv-to-gpu-mma",
        supported_chips=("sm_120a",),
        description="conv2d implicit-GEMM tensor-core lowering",
        scf_to_cf=True,
    ),
    PassDescriptor(
        fast_path_id="cuda.conv.direct-window.v1",
        pass_argument="-convert-intent-conv-to-gpu",
        supported_chips=("sm_120a",),
        description="direct windowed conv1d/2d/3d lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.row_scan.sparsemax.sorted_prefix.v1",
        pass_argument="-convert-intent-row-reduction-to-gpu=block-threads=256",
        supported_chips=("sm_120a",),
        description="sorted-prefix Sparsemax row scan",
        scf_to_cf=True,
        required_markers=(
            "intent_sorted_descending = true",
            "intent_host_preprocessing_in_kernel_timing = false",
        ),
    ),
    PassDescriptor(
        fast_path_id="cuda.padded_row_reduction.masked_short_dot.v1",
        pass_argument=(
            "-convert-intent-padded-row-reduction-to-gpu="
            "block-threads=64 vectorize-full-features=false "
            "promote-rank0-scalars=true reduction-topology=serial-row"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="masked short row dot with authored zero padding",
    ),
    PassDescriptor(
        fast_path_id="cuda.elementwise.row.vectorized.v1",
        pass_argument=(
            "-convert-intent-elementwise-to-gpu=block-threads=64 "
            "tanh-approximation=stable-ex2"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="complete static row-elementwise graph",
    ),
    PassDescriptor(
        fast_path_id="cuda.row_reduction.normalization_loss.read_once.v1",
        pass_argument=(
            "-convert-intent-row-reduction-to-gpu=block-threads=1024 "
            "promote-rank0-scalars=true rows-per-block=1"
        ),
        supported_chips=_PORTABLE_BLOCK_CUDA_CHIPS,
        description="row normalization/loss reduction graph",
    ),
    PassDescriptor(
        fast_path_id="cuda.gather-scatter.block.v1",
        pass_argument="-convert-intent-gather-to-gpu",
        supported_chips=("sm_120a",),
        description="data-dependent gather/scatter lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.layout.block.v1",
        pass_argument="-convert-intent-layout-to-gpu",
        supported_chips=("sm_120a",),
        description="concat/pad/tile/flip/triu data-movement lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.compact.prefix-sum.v1",
        pass_argument="-convert-intent-compact-to-gpu",
        supported_chips=("sm_120a",),
        description="nonzero/masked-select/masked-scatter compaction lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.rng.philox.v1",
        pass_argument="-convert-intent-rng-to-gpu",
        supported_chips=("sm_120a",),
        description="counter-based Philox RNG lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.arg-reduce.block.v1",
        pass_argument="-convert-intent-arg-reduce-to-gpu",
        supported_chips=("sm_120a",),
        description="argmax/argmin pair-reduction lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.sort.bitonic.v1",
        pass_argument="-convert-intent-sort-to-gpu",
        supported_chips=("sm_120a",),
        description="block bitonic sort/top-k lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.pool.window.v1",
        pass_argument="-convert-intent-pool-to-gpu",
        supported_chips=("sm_120a",),
        description="windowed average/max pooling lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.upsample.nearest.v1",
        pass_argument="-convert-intent-upsample-to-gpu",
        supported_chips=("sm_120a",),
        description="nearest-neighbor upsample lowering",
    ),
    PassDescriptor(
        fast_path_id="cuda.polar.elementwise.v1",
        pass_argument="-convert-intent-polar-to-gpu",
        supported_chips=("sm_120a",),
        description="polar real/imag pair lowering",
    ),
)


def read_intent_module(intent_module: str | os.PathLike[str]) -> str:
    path = Path(intent_module)
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            text = str(intent_module)
    except OSError:
        text = str(intent_module)
    if not text.strip() or ("module" not in text and "func.func" not in text):
        raise ValueError("selection requires non-empty Intent dialect MLIR")
    return text


def _passes(
    only_fast_path: str | None,
    passes: Sequence[PassDescriptor],
) -> tuple[PassDescriptor, ...]:
    selected = tuple(passes)
    if only_fast_path is None:
        return selected
    filtered = tuple(
        descriptor
        for descriptor in selected
        if descriptor.fast_path_id == only_fast_path
    )
    if not filtered:
        raise ValueError(f"unknown fast path {only_fast_path!r}")
    return filtered


def select_variant(
    intent_module: str | os.PathLike[str],
    *,
    target: LoweringTarget | None = None,
    pass_overrides: Mapping[str, str] | None = None,
    only_fast_path: str | None = None,
    allow_specialized: bool = True,
    passes: Sequence[PassDescriptor] = DEFAULT_PASSES,
) -> VariantSelection:
    """Select a CUDA lowering variant by target capability and C++ probe.

    No receipt, verifier result, provider response, dataset identity, function
    name, or source path is accepted by this API.  ``pass_overrides`` is the
    explicit schedule/tuning seam; every override is still required to pass the
    same whole-graph structural probe before it can be selected.
    """

    text = read_intent_module(intent_module)
    selected_target = target or LoweringTarget()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not allow_specialized:
        return VariantSelection(
            pipeline_kind="generic",
            target=selected_target,
            input_sha256=digest,
            fast_path_id=None,
            pass_argument=None,
            scf_to_cf=False,
            probe_output_sha256=None,
            attempts=(),
        )

    overrides = dict(pass_overrides or {})
    known_ids = {descriptor.fast_path_id for descriptor in passes}
    unknown_overrides = sorted(set(overrides) - known_ids)
    if unknown_overrides:
        raise ValueError(f"pass overrides reference unknown fast paths: {unknown_overrides}")
    for fast_path_id, pass_argument in overrides.items():
        if not isinstance(pass_argument, str) or not pass_argument.strip():
            raise ValueError(f"pass override for {fast_path_id!r} is empty")

    attempts: list[ProbeAttempt] = []
    for descriptor in _passes(only_fast_path, passes):
        pass_argument = overrides.get(
            descriptor.fast_path_id, descriptor.pass_argument
        )
        if selected_target.chip not in descriptor.supported_chips:
            attempts.append(
                ProbeAttempt(
                    fast_path_id=descriptor.fast_path_id,
                    pass_argument=pass_argument,
                    target_match=False,
                    status="rejected_target",
                    direct_hit=False,
                    reason=f"target {selected_target.chip!r} is unsupported",
                )
            )
            continue
        missing_markers = tuple(
            marker for marker in descriptor.required_markers if marker not in text
        )
        if missing_markers:
            attempts.append(
                ProbeAttempt(
                    fast_path_id=descriptor.fast_path_id,
                    pass_argument=pass_argument,
                    target_match=True,
                    status="rejected_contract",
                    direct_hit=False,
                    reason="required typed MLIR markers are absent: "
                    + ", ".join(missing_markers),
                )
            )
            continue
        try:
            probe = backend_compile._run(
                [backend_compile.INTENT_OPT, pass_argument], stdin=text
            )
        except Exception as error:
            attempts.append(
                ProbeAttempt(
                    fast_path_id=descriptor.fast_path_id,
                    pass_argument=pass_argument,
                    target_match=True,
                    status="probe_error",
                    direct_hit=False,
                    reason=f"{type(error).__name__}: {error}",
                )
            )
            continue
        residuals = residual_intent_ops(probe)
        launch_count = gpu_launch_count(probe)
        direct_hit = not residuals and launch_count >= 1
        probe_digest = hashlib.sha256(probe.encode("utf-8")).hexdigest()
        attempt = ProbeAttempt(
            fast_path_id=descriptor.fast_path_id,
            pass_argument=pass_argument,
            target_match=True,
            status="selected" if direct_hit else "rejected_structure",
            direct_hit=direct_hit,
            residual_ops=residuals,
            gpu_launch_count=launch_count,
            probe_output_sha256=probe_digest,
            reason=(
                "compiler-launch-graph whole-graph structural match"
                if direct_hit
                else "specialized pass did not consume the complete graph"
            ),
        )
        attempts.append(attempt)
        if direct_hit:
            return VariantSelection(
                pipeline_kind="specialized",
                target=selected_target,
                input_sha256=digest,
                fast_path_id=descriptor.fast_path_id,
                pass_argument=pass_argument,
                scf_to_cf=descriptor.scf_to_cf,
                probe_output_sha256=probe_digest,
                attempts=tuple(attempts),
            )

    return VariantSelection(
        pipeline_kind="generic",
        target=selected_target,
        input_sha256=digest,
        fast_path_id=None,
        pass_argument=None,
        scf_to_cf=False,
        probe_output_sha256=None,
        attempts=tuple(attempts),
    )


__all__ = [
    "DEFAULT_PASSES",
    "LoweringTarget",
    "PassDescriptor",
    "ProbeAttempt",
    "VariantSelection",
    "gpu_launch_count",
    "read_intent_module",
    "residual_intent_ops",
    "select_variant",
]
