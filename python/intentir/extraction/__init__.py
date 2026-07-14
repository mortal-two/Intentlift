"""LLM-based extraction: kernel source -> IntentIR JSON.

The LLM directly extracts to tensor-level atomic IntentIR (this is the research
core; see .trellis/spec/extraction/llm-extraction.md). This package owns the
provider client, prompts, re-roll, and evidence-as-hint plumbing.
"""

from .providers.client import LLMReply, ProviderError, call_responses
from .lifters.extract import Extraction, Provenance, extract_intent, smoke_interpret
from .lifters.scrc import (
    SCRCCompilerView,
    SCRCExtraction,
    SCRCLiftAttempt,
    SCRCLiftError,
    SCRCProvenance,
    extract_scrc_artifact,
)
from .lifters.scrc_schema import (
    scrc_artifact_json_schema,
    scrc_artifact_text_format,
)
from .providers.provider import ProviderConfig, get_provider, load_providers

__all__ = [
    "ProviderConfig",
    "get_provider",
    "load_providers",
    "call_responses",
    "LLMReply",
    "ProviderError",
    "extract_intent",
    "Extraction",
    "Provenance",
    "smoke_interpret",
    "SCRCExtraction",
    "SCRCCompilerView",
    "SCRCLiftAttempt",
    "SCRCLiftError",
    "SCRCProvenance",
    "extract_scrc_artifact",
    "scrc_artifact_json_schema",
    "scrc_artifact_text_format",
]
