"""LLM-based extraction: kernel source -> IntentIR JSON.

The LLM directly extracts to tensor-level atomic IntentIR (this is the research
core; see .trellis/spec/extraction/llm-extraction.md). This package owns the
provider client, prompts, re-roll, and evidence-as-hint plumbing.
"""

from .provider import ProviderConfig, get_provider, load_providers

__all__ = ["ProviderConfig", "get_provider", "load_providers"]
