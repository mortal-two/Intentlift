"""Compatibility alias for :mod:`intentir.backend.cuda.selection`."""

from __future__ import annotations

import sys

from intentir.backend.cuda import selection as _implementation

sys.modules[__name__] = _implementation
