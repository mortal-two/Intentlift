"""Discover extension components and execute their checked compilation pipeline."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
from uuid import uuid4
from .errors import require
from .registry import Registry


def load_registry(extension_root):
    root = Path(extension_root).resolve()
    # A candidate imports its own helper modules, never the official copies.
    namespace = "_intentlift_components_" + uuid4().hex
    package = ModuleType(namespace)
    package.__path__ = [str(root)]
    sys.modules[namespace] = package
    registry = Registry()
    for path in sorted(root.glob("*.py")):
        require(path.resolve().parent == root and not path.is_symlink(), "extension symlink forbidden", "extension_invalid")
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = importlib.import_module("." + path.stem, namespace)
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
