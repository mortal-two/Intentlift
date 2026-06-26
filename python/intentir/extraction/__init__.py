"""LLM-based extraction: kernel source -> IntentIR JSON.

The LLM directly extracts to tensor-level atomic IntentIR (this is the research
core; see .trellis/spec/extraction/llm-extraction.md). This package owns the
provider client, prompts, re-roll, and evidence-as-hint plumbing.
"""

from .providers.client import LLMReply, ProviderError, call_responses
from .lifters.extract import Extraction, Provenance, extract_intent, smoke_interpret
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
]
