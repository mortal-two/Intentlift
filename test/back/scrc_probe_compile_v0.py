"""Compile the archived source/replay sides of the initial Triton SCRC probe.

The script uses ``JITFunction.warmup``: compilation happens, kernels are not
executed.  ``CompiledKernel.asm['ttir']`` is persisted independently of the
environment-driven Triton dump so cache behaviour cannot hide an artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import torch


TEST_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cases(module: ModuleType):
    device = "cuda"
    return {
        "add": {
            "kernel": module.add_kernel,
            "args": (
                torch.empty(1000, device=device, dtype=torch.float32),
                torch.empty(1000, device=device, dtype=torch.float32),
                torch.empty(1000, device=device, dtype=torch.float32),
                1000,
            ),
            "grid": (4,),
            "meta": {
                "BLOCK": 256,
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
            },
        },
        "row_sum": {
            "kernel": module.row_sum_kernel,
            "args": (
                torch.empty((37, 1000), device=device, dtype=torch.float32),
                torch.empty(37, device=device, dtype=torch.float32),
                37,
                1000,
            ),
            "grid": (37,),
            "meta": {
                "BLOCK_N": 1024,
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
            },
        },
        "softmax": {
            "kernel": module.softmax_kernel,
            "args": (
                torch.empty((37, 1000), device=device, dtype=torch.float32),
                torch.empty((37, 1000), device=device, dtype=torch.float32),
                37,
                1000,
            ),
            "grid": (37,),
            "meta": {
                "BLOCK_N": 1024,
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
            },
        },
        "rmsnorm": {
            "kernel": module.rmsnorm_kernel,
            "args": (
                torch.empty((37, 1000), device=device, dtype=torch.float32),
                torch.empty(1000, device=device, dtype=torch.float32),
                torch.empty((37, 1000), device=device, dtype=torch.float32),
                37,
                1000,
                1.0e-5,
            ),
            "grid": (37,),
            "meta": {
                "BLOCK_N": 1024,
                "num_warps": 4,
                "num_stages": 2,
                "num_ctas": 1,
            },
        },
        "matmul": {
            "kernel": module.matmul_kernel,
            "args": (
                torch.empty((45, 53), device=device, dtype=torch.float16),
                torch.empty((53, 37), device=device, dtype=torch.float16),
                torch.empty((45, 37), device=device, dtype=torch.float32),
                45,
                37,
                53,
            ),
            "grid": (2, 2),
            "meta": {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
                "BLOCK_K": 32,
                "num_ctas": 1,
                "num_warps": 4,
                "num_stages": 2,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("source", "replay"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    module_path = (
        TEST_ROOT / "scrc_probe_sources.py"
        if args.side == "source"
        else TEST_ROOT / "back" / "scrc_probe_replays_v0.py"
    )
    module = _load(module_path, f"scrc_probe_{args.side}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, object]] = {}
    for name, case in _cases(module).items():
        compiled = case["kernel"].warmup(
            *case["args"], grid=case["grid"], **case["meta"]
        )
        ttir = str(compiled.asm["ttir"])
        out = args.output_dir / f"{name}.{args.side}.ttir"
        out.write_text(ttir)
        summary[name] = {
            "ttir": str(out),
            "ttir_bytes": len(ttir.encode()),
            "asm_keys": sorted(compiled.asm.keys()),
            "python_interface": {
                "arg_names": list(case["kernel"].arg_names),
                "constexprs": [
                    case["kernel"].arg_names[i] for i in case["kernel"].constexprs
                ],
                "do_not_specialize": list(case["kernel"].do_not_specialize),
            },
            "grid": list(case["grid"]),
            "meta": case["meta"],
        }

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
