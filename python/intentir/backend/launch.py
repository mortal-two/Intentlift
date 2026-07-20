"""Compatibility alias for :mod:`intentir.backend.cuda.launch`."""

from __future__ import annotations

import sys

from intentir.backend.cuda import launch as _implementation

sys.modules[__name__] = _implementation
