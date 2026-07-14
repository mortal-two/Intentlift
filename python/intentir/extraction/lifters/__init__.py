"""LLM lifters from real kernel source to IntentIR/SCRC artifacts."""

from .scrc import SCRCExtraction, SCRCProvenance, extract_scrc_artifact

__all__ = ["SCRCExtraction", "SCRCProvenance", "extract_scrc_artifact"]
