#!/usr/bin/env python3
"""Export strict LLVM, Expert, ggml and optional Triton-CPU RVV baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intentir.backend.riscv.native_baselines import (
    EXTERNAL_RVV_BASELINE_SIDECAR_PROTOCOL,
    external_rvv_baseline_records,
)
from intentir.backend.riscv.ggml_baselines import maybe_ggml_rvv_baseline_records
from intentir.backend.riscv.triton_cpu_baselines import (
    dyt_triton_cpu_rvv_record,
    flash_preprocess_triton_cpu_rvv_record,
    geglu_triton_cpu_rvv_record,
    kldiv_triton_cpu_rvv_record,
    layer_norm_triton_cpu_rvv_record,
    rms_norm_triton_cpu_rvv_record,
    softmax_triton_cpu_rvv_record,
    swiglu_triton_cpu_rvv_record,
)


DEFAULT_KERNELS = (
    "dyt",
    "flash_bwd_preprocess",
    "flash_layer_norm_1pass",
    "fused_add_rms_norm",
    "geglu",
    "kl_div",
    "layer_norm",
    "liger_rms_norm",
    "poly_norm",
    "softmax",
    "sparsemax",
    "splitk_reduce_varargs",
    "swiglu",
    "xformers_rms_norm_add",
    "cross_entropy",
    "flash_linear",
    "group_norm",
    "jsd",
    "liger_rope",
    "qwen2vl_mrope",
    "tvd",
    "xformers_rms_norm",
    "gather2d",
    "rowmask_where2d",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*", choices=DEFAULT_KERNELS)
    parser.add_argument("--triton-cpu-swiglu-library", type=Path)
    parser.add_argument("--triton-cpu-swiglu-columns", type=int, default=11008)
    parser.add_argument("--triton-cpu-swiglu-source", type=str)
    parser.add_argument("--triton-cpu-softmax-library", type=Path)
    parser.add_argument("--triton-cpu-softmax-columns", type=int, default=4096)
    parser.add_argument("--triton-cpu-softmax-source", type=str)
    parser.add_argument("--triton-cpu-kldiv-library", type=Path)
    parser.add_argument("--triton-cpu-kldiv-columns", type=int, default=4096)
    parser.add_argument("--triton-cpu-kldiv-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--triton-cpu-kldiv-source", type=str)
    parser.add_argument("--triton-cpu-flash-preprocess-library", type=Path)
    parser.add_argument("--triton-cpu-flash-preprocess-block-m", type=int, default=128)
    parser.add_argument("--triton-cpu-flash-preprocess-block-d", type=int, default=128)
    parser.add_argument("--triton-cpu-flash-preprocess-source", type=str)
    parser.add_argument("--triton-cpu-geglu-library", type=Path)
    parser.add_argument("--triton-cpu-geglu-columns", type=int, default=11008)
    parser.add_argument("--triton-cpu-geglu-source", type=str)
    parser.add_argument("--triton-cpu-dyt-library", type=Path)
    parser.add_argument("--triton-cpu-dyt-columns", type=int, default=4096)
    parser.add_argument("--triton-cpu-dyt-block-size", type=int, default=128)
    parser.add_argument("--triton-cpu-dyt-source", type=str)
    parser.add_argument("--triton-cpu-rms-norm-library", type=Path)
    parser.add_argument("--triton-cpu-rms-norm-columns", type=int, default=4096)
    parser.add_argument("--triton-cpu-rms-norm-source", type=str)
    parser.add_argument("--triton-cpu-layer-norm-library", type=Path)
    parser.add_argument("--triton-cpu-layer-norm-columns", type=int, default=4096)
    parser.add_argument("--triton-cpu-layer-norm-source", type=str)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kernels = tuple(args.kernels or DEFAULT_KERNELS)

    def records(kernel: str) -> list[dict[str, object]]:
        result = [
            *external_rvv_baseline_records(kernel),
            *maybe_ggml_rvv_baseline_records(kernel),
        ]
        if kernel == "swiglu" and args.triton_cpu_swiglu_library is not None:
            if not args.triton_cpu_swiglu_source:
                raise ValueError(
                    "--triton-cpu-swiglu-library requires "
                    "--triton-cpu-swiglu-source"
                )
            result.append(
                swiglu_triton_cpu_rvv_record(
                    args.triton_cpu_swiglu_library,
                    columns=args.triton_cpu_swiglu_columns,
                    source_identity=args.triton_cpu_swiglu_source,
                )
            )
        if kernel == "softmax" and args.triton_cpu_softmax_library is not None:
            if not args.triton_cpu_softmax_source:
                raise ValueError(
                    "--triton-cpu-softmax-library requires "
                    "--triton-cpu-softmax-source"
                )
            result.append(
                softmax_triton_cpu_rvv_record(
                    args.triton_cpu_softmax_library,
                    columns=args.triton_cpu_softmax_columns,
                    source_identity=args.triton_cpu_softmax_source,
                )
            )
        if kernel == "kl_div" and args.triton_cpu_kldiv_library is not None:
            if not args.triton_cpu_kldiv_source:
                raise ValueError(
                    "--triton-cpu-kldiv-library requires "
                    "--triton-cpu-kldiv-source"
                )
            result.append(
                kldiv_triton_cpu_rvv_record(
                    args.triton_cpu_kldiv_library,
                    columns=args.triton_cpu_kldiv_columns,
                    epsilon=args.triton_cpu_kldiv_epsilon,
                    source_identity=args.triton_cpu_kldiv_source,
                )
            )
        if (
            kernel == "flash_bwd_preprocess"
            and args.triton_cpu_flash_preprocess_library is not None
        ):
            if not args.triton_cpu_flash_preprocess_source:
                raise ValueError(
                    "--triton-cpu-flash-preprocess-library requires "
                    "--triton-cpu-flash-preprocess-source"
                )
            result.append(
                flash_preprocess_triton_cpu_rvv_record(
                    args.triton_cpu_flash_preprocess_library,
                    block_m=args.triton_cpu_flash_preprocess_block_m,
                    block_head_dim=args.triton_cpu_flash_preprocess_block_d,
                    source_identity=args.triton_cpu_flash_preprocess_source,
                )
            )
        if kernel == "geglu" and args.triton_cpu_geglu_library is not None:
            if not args.triton_cpu_geglu_source:
                raise ValueError(
                    "--triton-cpu-geglu-library requires --triton-cpu-geglu-source"
                )
            result.append(
                geglu_triton_cpu_rvv_record(
                    args.triton_cpu_geglu_library,
                    columns=args.triton_cpu_geglu_columns,
                    source_identity=args.triton_cpu_geglu_source,
                )
            )
        if kernel == "dyt" and args.triton_cpu_dyt_library is not None:
            if not args.triton_cpu_dyt_source:
                raise ValueError(
                    "--triton-cpu-dyt-library requires --triton-cpu-dyt-source"
                )
            result.append(
                dyt_triton_cpu_rvv_record(
                    args.triton_cpu_dyt_library,
                    columns=args.triton_cpu_dyt_columns,
                    block_size=args.triton_cpu_dyt_block_size,
                    source_identity=args.triton_cpu_dyt_source,
                )
            )
        if (
            kernel == "liger_rms_norm"
            and args.triton_cpu_rms_norm_library is not None
        ):
            if not args.triton_cpu_rms_norm_source:
                raise ValueError(
                    "--triton-cpu-rms-norm-library requires "
                    "--triton-cpu-rms-norm-source"
                )
            result.append(
                rms_norm_triton_cpu_rvv_record(
                    args.triton_cpu_rms_norm_library,
                    columns=args.triton_cpu_rms_norm_columns,
                    source_identity=args.triton_cpu_rms_norm_source,
                )
            )
        if (
            kernel == "layer_norm"
            and args.triton_cpu_layer_norm_library is not None
        ):
            if not args.triton_cpu_layer_norm_source:
                raise ValueError(
                    "--triton-cpu-layer-norm-library requires "
                    "--triton-cpu-layer-norm-source"
                )
            result.append(
                layer_norm_triton_cpu_rvv_record(
                    args.triton_cpu_layer_norm_library,
                    columns=args.triton_cpu_layer_norm_columns,
                    source_identity=args.triton_cpu_layer_norm_source,
                )
            )
        return result

    payload = {
        "protocol": EXTERNAL_RVV_BASELINE_SIDECAR_PROTOCOL,
        "baselines": {
            kernel: records(kernel)
            for kernel in kernels
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    baseline_count = sum(len(items) for items in payload["baselines"].values())
    print(
        f"wrote {baseline_count} external RVV baselines for "
        f"{len(kernels)} kernels to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
