"""LLM-based extraction: kernel source -> IntentIR JSON.

The LLM directly extracts to tensor-level atomic IntentIR (this is the research
core; see .trellis/spec/extraction/llm-extraction.md). This package owns the
provider client, prompts, re-roll, and evidence-as-hint plumbing.
"""

from .providers.client import LLMReply, ProviderError, call_responses
from .lifters.extract import Extraction, Provenance, extract_intent, smoke_interpret
from .lifters.scrc import (
    ResponsesSCRCRelifter,
    SCRC_RELIFT_PROTOCOL,
    SCRCCompilerArgument,
    SCRCCompilerView,
    SCRCExtraction,
    SCRCLiftAttempt,
    SCRCLiftError,
    SCRCProvenance,
    SCRCReliftContext,
    SCRCReliftRecord,
    extract_scrc_artifact,
    scrc_lift_prompt_sha256,
    scrc_relift_prompt_sha256,
)
from .lifters.scrc_schema import (
    scrc_lift_core_json_schema,
    scrc_lift_core_text_format,
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
    "ResponsesSCRCRelifter",
    "SCRC_RELIFT_PROTOCOL",
    "SCRCExtraction",
    "SCRCCompilerArgument",
    "SCRCCompilerView",
    "SCRCLiftAttempt",
    "SCRCLiftError",
    "SCRCProvenance",
    "SCRCReliftContext",
    "SCRCReliftRecord",
    "extract_scrc_artifact",
    "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_text_format",
]
