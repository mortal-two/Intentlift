"""Archived prompt augmentation for the removed inner replay-repair loop.

The active writer never receives prior replay source.  This helper is retained
only to reproduce historical prompts and must not be imported by Gate1/Gate2.
"""

from __future__ import annotations


def augment_replay_repair_prompt(prompt: str, base_code: str) -> str:
    """Render the historical replay-code continuation prompt."""

    return (
        prompt
        + "\n\nCURRENT_REPLAY_CANDIDATE (model-generated, not source):\n"
        + base_code
        + "\n\nCONTROLLED_REPLAY_UPDATE:\n"
        + "Use the prior model-generated replay only as a structural starting "
        + "point for the current replay packet. It is not authoritative. "
        + "Re-read the current J and complete pi, preserve every construction "
        + "choice that remains consistent with them, and change only code "
        + "strictly required by the current packet. Re-emit the complete module."
    )


__all__ = ["augment_replay_repair_prompt"]
