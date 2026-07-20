"""Private subprocess entry for one scalar CPU native call."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .runtime import CPU_RUNTIME_PROTOCOL, _close_memmap, _native_call


def main() -> int:
    inputs: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    try:
        request = json.loads(sys.stdin.read())
        if request.get("protocol") != CPU_RUNTIME_PROTOCOL:
            raise ValueError("worker request protocol mismatch")
        shared_object = Path(request["shared_object"]).resolve(strict=True)
        inputs = [
            np.load(path, mmap_mode="r+", allow_pickle=False)
            for path in request["inputs"]
        ]
        outputs = [
            np.load(path, mmap_mode="r+", allow_pickle=False)
            for path in request["outputs"]
        ]
        _native_call(shared_object, inputs, outputs)
        for value in outputs:
            value.flush()
        print(json.dumps({"native_call_count": 1, "ok": True}, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "ok": False},
                sort_keys=True,
            )
        )
        return 1
    finally:
        for value in [*inputs, *outputs]:
            _close_memmap(value)


if __name__ == "__main__":
    raise SystemExit(main())
