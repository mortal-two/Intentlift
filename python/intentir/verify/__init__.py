"""SCRC verification entry point.

The active verification design consists only of Gate1 (source-isolated replay
closure) and Gate2 (source/replay dynamic execution).  Historical helpers are archived under
``intentir.verify.back`` and are intentionally not re-exported here.
"""

from . import gate1
from . import gate2

__all__ = ["gate1", "gate2"]
