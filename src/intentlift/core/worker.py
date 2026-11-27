"""JSON-only compiler subprocess; executed inside a read-only mount namespace."""
import json
import sys
from .compiler import compile_request, load_registry
from .errors import LiftError


def main():
    try:
        request = json.load(sys.stdin)
        if request.get("action") == "inspect":
            registry, pipeline = load_registry(request["extension_root"])
            result = {"ok": True, **registry.describe(), "pipeline": pipeline}
        else:
            result = {"ok": True, "dsl": compile_request(request, request["extension_root"])}
    except LiftError as exc:
        result = exc.record()
    except Exception as exc:
        result = {"ok": False, "kind": "extension_invalid", "message": f"{type(exc).__name__}: {exc}"}
    json.dump(result, sys.stdout, allow_nan=False)


if __name__ == "__main__":
    main()
