#!/usr/bin/env python3
"""Export independent LLVM-RVV C and Expert RVV baseline source maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intentir.backend.riscv.native_baselines import (
    EXTERNAL_RVV_BASELINE_SIDECAR_PROTOCOL,
    external_rvv_baseline_records,
)
from intentir.backend.riscv.ggml_baselines import maybe_ggml_rvv_baseline_records


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kernels = tuple(args.kernels or DEFAULT_KERNELS)
    payload = {
        "protocol": EXTERNAL_RVV_BASELINE_SIDECAR_PROTOCOL,
        "baselines": {
            kernel: [
                *external_rvv_baseline_records(kernel),
                *maybe_ggml_rvv_baseline_records(kernel),
            ]
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
