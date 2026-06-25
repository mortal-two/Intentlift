"""Track-B codegen: compile intent IR to an sm_120a cubin (M5 device half).

Validates the whole intent -> linalg -> gpu -> nvvm -> cubin pipeline produces a
launchable Blackwell kernel, WITHOUT needing a GPU (ptxas + cuobjdump are CPU).
Skips cleanly if the C++ build or CUDA tools are absent.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

from intentir.backend.compile import CUDA_BIN, INTENT_OPT, compile_intent_to_cubin

_ADD = pathlib.Path(__file__).resolve().parents[2] / "test" / "add.mlir"
_CUOBJDUMP = pathlib.Path(CUDA_BIN) / "cuobjdump"


@pytest.mark.skipif(not pathlib.Path(INTENT_OPT).exists(),
                    reason="intent-opt not built")
@pytest.mark.skipif(not _CUOBJDUMP.exists(), reason="cuobjdump (CUDA) not present")
@pytest.mark.skipif(not shutil.which("mlir-opt", path="/usr/lib/llvm-20/bin"),
                    reason="mlir-opt 20 not present")
def test_intent_add_compiles_to_sm120_cubin(tmp_path) -> None:
    k = compile_intent_to_cubin(_ADD)
    assert k.kernel_name == "add_kernel"
    assert k.cubin[:4] == b"\x7fELF", "expected an ELF cubin"
    cubin = tmp_path / "add.cubin"
    cubin.write_bytes(k.cubin)
    # cuobjdump validates the ELF and lists the launchable kernel entry.
    # (sm_120 execution itself is proven by test_backend_execute.)
    out = subprocess.run(
        [str(_CUOBJDUMP), "--dump-elf-symbols", str(cubin)],
        capture_output=True, text=True,
    ).stdout
    assert "add_kernel" in out, out
