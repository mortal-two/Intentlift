"""Static JSON-Schema check available inside the untrusted Lift sandbox."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import jsonschema


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_candidate.py SCHEMA CANDIDATE", file=sys.stderr)
        return 2
    schema = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    candidate = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    validator = jsonschema.validators.validator_for(schema)(schema)
    errors = sorted(
        validator.iter_errors(candidate),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if not errors:
        print("static schema: PASS")
        return 0
    for error in errors[:20]:
        path = "/".join(str(item) for item in error.absolute_path) or "<root>"
        print(f"{path}: {error.message}")
    if len(errors) > 20:
        print(f"... {len(errors) - 20} more errors")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
