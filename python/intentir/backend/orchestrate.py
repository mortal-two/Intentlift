"""Compatibility alias for :mod:`intentir.backend.cuda.orchestrate`."""

from __future__ import annotations

import sys

from intentir.backend.cuda import orchestrate as _implementation

sys.modules[__name__] = _implementation
