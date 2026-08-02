"""Legacy verification helpers excluded from the active Gate1/Gate2 path.

Historical Gate1 result and masked-ratio readers intentionally live in their
own modules so importing the old interpreter does not load active Gate1.
"""

from .interpret import InterpError, interpret

__all__ = ["InterpError", "interpret"]
