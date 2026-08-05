"""LLM-based extraction: kernel source -> IntentIR JSON.

The LLM directly extracts to tensor-level atomic IntentIR.  The current Lift,
Replay, and closure contract is documented in ``docs/architecture.md`` and
``docs/workflows.md``.  This package owns the provider client, prompts, re-roll,
and evidence-as-hint plumbing; superseded campaign material is kept only in the
ignored local archive.
"""

from .providers.client import LLMReply, ProviderError, call_responses
from .lifters.miniswe import (
    MINISWE_LIFT_BACKEND,
    MINISWE_LIFT_KNOWLEDGE,
    MiniSWELiftAdapter,
)
from .lifters.scrc import (
    ResponsesSCRCLiftAdapter,
    ResponsesSCRCRelifter,
    SCRCCandidateCheckContext,
    SCRC_DECODE_PROTOCOL,
    SCRC_RELIFT_PROTOCOL,
    SCRCCompilerArgument,
    SCRCCompilerView,
    SCRCExtraction,
    SCRCLiftAttempt,
    SCRCLiftError,
    SCRCProvenance,
    SCRCReliftContext,
    SCRCReliftRecord,
    decode_scrc_lift_artifact,
    extract_scrc_artifact,
    scrc_candidate_check_context,
    scrc_lift_prompt_sha256,
    scrc_relift_prompt_sha256,
)
from .lifters.scrc_schema import (
    scrc_lift_core_json_schema,
    scrc_lift_core_text_format,
)
from .providers.provider import ProviderConfig, get_provider, load_providers

__all__ = [
    "MINISWE_LIFT_BACKEND",
    "MINISWE_LIFT_KNOWLEDGE",
    "MiniSWELiftAdapter",
    "ProviderConfig",
    "get_provider",
    "load_providers",
    "call_responses",
    "LLMReply",
    "ProviderError",
    "ResponsesSCRCRelifter",
    "ResponsesSCRCLiftAdapter",
    "SCRCCandidateCheckContext",
    "SCRC_DECODE_PROTOCOL",
    "SCRC_RELIFT_PROTOCOL",
    "SCRCExtraction",
    "SCRCCompilerArgument",
    "SCRCCompilerView",
    "SCRCLiftAttempt",
    "SCRCLiftError",
    "SCRCProvenance",
    "SCRCReliftContext",
    "SCRCReliftRecord",
    "decode_scrc_lift_artifact",
    "extract_scrc_artifact",
    "scrc_candidate_check_context",
    "scrc_lift_prompt_sha256",
    "scrc_relift_prompt_sha256",
    "scrc_lift_core_json_schema",
    "scrc_lift_core_text_format",
]
