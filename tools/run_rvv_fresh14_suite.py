#!/usr/bin/env python3
"""Run the Fresh14 RVV bundles through the target-side production runner."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


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
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--native-baseline-sidecar", type=Path, required=True)
    parser.add_argument("--kernels", nargs="*", choices=FRESH14)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--threads", type=int, choices=(1, 16), required=True)
    parser.add_argument("--regime", choices=("warm", "streaming", "cold-data"), default="warm")
    parser.add_argument("--working-set-mib", type=int, default=128)
    parser.add_argument("--search-budget", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--measure-c-oracle", action="store_true")
    return parser


def _bundle_name(kernel: str) -> str:
    suffix = ".closed.v4.json" if kernel == "splitk_reduce_varargs" else ".v4.json"
    return kernel + suffix


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    kernels = tuple(args.kernels or FRESH14)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for kernel in kernels:
        command = [
            sys.executable,
            str(args.runner),
            "--bundle",
            str(args.bundle_dir / _bundle_name(kernel)),
            "--native-baseline-sidecar",
            str(args.native_baseline_sidecar),
            "--out",
            str(args.report_dir / f"{kernel}.t{args.threads}.json"),
            "--kernels",
            kernel,
            "--cpu",
            str(args.cpu),
            "--threads",
            str(args.threads),
            "--regime",
            args.regime,
            "--working-set-mib",
            str(args.working_set_mib),
            "--search-budget",
            str(args.search_budget),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--batches",
            str(args.batches),
        ]
        if args.measure_c_oracle:
            command.append("--measure-c-oracle")
        print(f"[{kernel}] starting", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures.append(kernel)
            print(f"[{kernel}] failed", flush=True)
        else:
            print(f"[{kernel}] complete", flush=True)
    if failures:
        print("failed kernels: " + ", ".join(failures), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
