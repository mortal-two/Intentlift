"""Compatibility import for the archived IntentIR NumPy interpreter.

The active verification pipeline is :mod:`intentir.verify.gate1` followed by
:mod:`intentir.verify.gate2`.  The interpreter remains executable for legacy
tests and historical callers, but new verification code must import neither
this module nor :mod:`intentir.verify.back`.
"""

from .back.legacy_v0.interpret import InterpError, _philox_uniform, interpret

__all__ = ["InterpError", "interpret"]
