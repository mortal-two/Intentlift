"""Discover extension components and execute their checked compilation pipeline."""
import importlib.util
import json
from pathlib import Path
from .errors import require
from .registry import Registry


def load_registry(extension_root):
    root = Path(extension_root).resolve()
    registry = Registry()
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        require(path.resolve().parent == root and not path.is_symlink(), "extension symlink forbidden", "extension_invalid")
        spec = importlib.util.spec_from_file_location("lift_component_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        registry.owner = path.name
        module.register(registry)
    registry.owner = None
    pipeline = json.loads((root / "pipeline.json").read_text())
    require(isinstance(pipeline, list) and pipeline and all(isinstance(x, str) for x in pipeline),
            "pipeline must be a nonempty stage list", "extension_invalid")
    registry.plan(pipeline)
    return registry, pipeline


def compile_request(request, extension_root):
    registry, pipeline = load_registry(extension_root)
    return registry.execute(pipeline, request, request)
