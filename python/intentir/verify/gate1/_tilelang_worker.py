"""Fresh-process TileLang compile worker.  Internal to :mod:`tilelang_compiler`."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout


def _load(request: dict):
    source = request["source"]
    if source["kind"] == "code":
        # TileLang's eager decorator uses ``inspect.getsourcelines``.  Import a
        # validated temporary module so the public frontend can recover source.
        with tempfile.TemporaryDirectory(prefix="intentir-tilelang-replay-") as raw:
            path = Path(raw) / "replay.py"
            path.write_text(source["code"], encoding="utf-8")
            spec = importlib.util.spec_from_file_location(
                "_intentir_tilelang_replay", path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("cannot create TileLang Replay module")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, source["function"])
    path = Path(source["path"]).resolve()
    spec = importlib.util.spec_from_file_location("_intentir_tilelang_source", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TileLang source {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, source["function"])
    return factory(**dict(source.get("factory_kwargs") or {}))


def _interface(prim_func) -> dict:
    import tilelang  # noqa: F401
    from tvm import tir

    reads: set[str] = set()
    writes: set[str] = set()
    copy_destinations: set[str] = set()

    def visit(node) -> None:
        if isinstance(node, tir.BufferLoad):
            reads.add(str(node.buffer.name))
        elif isinstance(node, tir.BufferStore):
            writes.add(str(node.buffer.name))
        elif isinstance(node, tir.Call):
            operation = str(getattr(getattr(node, "op", None), "name", ""))
            if operation == "tl.tileop.copy" and len(node.args) >= 2:
                destination = node.args[1]
                if (
                    isinstance(destination, tir.Call)
                    and str(getattr(destination.op, "name", "")) == "tl.tileop.region"
                    and destination.args
                    and isinstance(destination.args[0], tir.BufferLoad)
                ):
                    name = str(destination.args[0].buffer.name)
                    writes.add(name)
                    copy_destinations.add(name)

    tir.stmt_functor.post_order_visit(prim_func.body, visit)

    def dimension(value):
        if isinstance(value, tir.IntImm):
            return int(value)
        return str(value)

    parameters: list[dict] = []
    for parameter in list(prim_func.params):
        if parameter in prim_func.buffer_map:
            buffer = prim_func.buffer_map[parameter]
            name = str(buffer.name)
            # TileLang represents a copy destination region with BufferLoad,
            # even though it is not a semantic read of the global buffer.
            read = name in reads and name not in copy_destinations
            write = name in writes
            mode = "inout" if read and write else "output" if write else "input"
            parameters.append(
                {
                    "name": name,
                    "kind": "pointer",
                    "dtype": str(buffer.dtype),
                    "rank": len(buffer.shape),
                    "shape": [dimension(item) for item in buffer.shape],
                    "mode": mode,
                }
            )
        else:
            dtype = str(getattr(parameter, "dtype", "handle"))
            parameters.append(
                {
                    "name": str(getattr(parameter, "name", parameter)),
                    "kind": "scalar",
                    "dtype": dtype,
                    "rank": 0,
                    "mode": "value",
                }
            )
    return {"parameters": parameters}


def main() -> int:
    started = time.perf_counter()
    try:
        request = json.loads(sys.stdin.read())
        # Executing this file directly places ``gate1/`` on sys.path, where the
        # observer module ``tilelang.py`` would shadow the external package.
        worker_directory = Path(__file__).resolve().parent
        sys.path = [
            item
            for item in sys.path
            if Path(item or ".").resolve() != worker_directory
        ]
        # TileLang/TVM may print build diagnostics.  Keep stdout a single JSON
        # object so the parent can fail closed on worker protocol corruption.
        with redirect_stdout(sys.stderr):
            import tilelang

            prim_func = _load(request)
            tir_text = str(prim_func.script(show_meta=False))
            kernel = tilelang.compile(prim_func, target=request["case"]["target"])
            cuda = str(kernel.get_kernel_source())
            ptx = str(kernel._get_ptx())
        payload = {
            "ok": True,
            "tir": tir_text,
            "cuda": cuda,
            "ptx": ptx,
            "interface": _interface(prim_func),
            "compiler": {
                "name": "tilelang",
                "version": str(getattr(tilelang, "__version__", "unknown")),
            },
            "target": {"name": request["case"]["target"]},
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    except Exception as exc:
        payload = {
            "ok": False,
            "diagnostic": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
