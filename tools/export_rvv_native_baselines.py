#!/usr/bin/env python3
"""Export the small independent Native baseline source map for RVV runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intentir.backend.riscv.native_baselines import (
    NATIVE_RVV_BASELINE_SIDECAR_PROTOCOL,
    native_rvv_baseline_record,
)


FRESH14 = (
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
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*", choices=FRESH14)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kernels = tuple(args.kernels or FRESH14)
    payload = {
        "protocol": NATIVE_RVV_BASELINE_SIDECAR_PROTOCOL,
        "baselines": {
            kernel: native_rvv_baseline_record(kernel) for kernel in kernels
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(kernels)} Native RVV baselines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
