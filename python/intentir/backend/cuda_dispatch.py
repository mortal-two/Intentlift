"""Compatibility alias for :mod:`intentir.backend.cuda.dispatch`."""

from __future__ import annotations

import sys

from intentir.backend.cuda import dispatch as _implementation

sys.modules[__name__] = _implementation
