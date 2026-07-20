"""Compatibility alias for :mod:`intentir.backend.cuda.schedules`."""

from __future__ import annotations

import sys

from intentir.backend.cuda import schedules as _implementation

sys.modules[__name__] = _implementation
