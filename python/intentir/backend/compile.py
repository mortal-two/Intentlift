"""Compatibility alias for :mod:`intentir.backend.cuda.compile`.

New code must import the target-specific module.  Replacing this module entry
with the canonical implementation preserves private helper access and
monkeypatch behavior for existing callers during migration.
"""

from __future__ import annotations

import sys

from intentir.backend.cuda import compile as _implementation

sys.modules[__name__] = _implementation
