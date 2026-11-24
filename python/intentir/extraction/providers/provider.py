"""LLM provider configuration loading.

Reads ``llm_providers.local.json`` (gitignored; holds API keys) from the repo
root and exposes a typed config. Actual HTTP calling lives in a client built on
top of this; kept dependency-free here so config can load without a network or
extra packages.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# provider.py -> extraction -> intentir -> python -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROVIDERS_PATH = REPO_ROOT / "llm_providers.local.json"
PROVIDERS_PATH_ENV = "INTENTIR_PROVIDER_CONFIG"
PROVIDER_NAME_ENV = "INTENTIR_PROVIDER_NAME"
PROVIDER_MODEL_ENV = "INTENTIR_PROVIDER_MODEL"
OPENAI_BASE_URL_ENV = "OPENAI_BASE_URL"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_MODEL_ENV = "OPENAI_MODEL"

# This is only the label for an explicitly environment-backed provider.  A
# provider file remains authoritative whenever one is supplied.
DEFAULT_ENV_PROVIDER_NAME = "openai"
DISABLED_PROVIDER_NAMES = frozenset({"cislc"})
# The environment-backed provider is used when a managed runner injects only
# OPENAI_BASE_URL and OPENAI_API_KEY.  The previous fallback model is no
# longer exposed by that transport.  gpt-6.1-sol has completed repeated real
# function-call SSE probes and is the current stable environment fallback;
# callers may still explicitly select a different listed model through
# OPENAI_MODEL.
DEFAULT_ENV_PROVIDER_MODEL = "gpt-6.1-sol"


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    wire_api: str = "responses"
    reasoning_effort: str = "high"

    def __post_init__(self) -> None:
        if self.name.casefold() in DISABLED_PROVIDER_NAMES:
            raise ValueError(f"provider {self.name!r} is disabled")


def _environment_provider() -> ProviderConfig | None:
    """Build an in-process provider without persisting credential material.

    The normal source of provider configuration is the ignored JSON file.  A
    caller may instead inject an OpenAI-compatible endpoint and key through
    the process environment, which is useful for ephemeral runners and keeps
    the key out of the repository and all evidence artifacts.
    """
    base_url = os.environ.get(OPENAI_BASE_URL_ENV, "").strip()
    api_key = os.environ.get(OPENAI_API_KEY_ENV, "").strip()
    if not base_url or not api_key:
        return None
    name = (
        os.environ.get(PROVIDER_NAME_ENV)
        or os.environ.get("OPENAI_PROVIDER_NAME")
        or DEFAULT_ENV_PROVIDER_NAME
    ).strip()
    model = (
        os.environ.get(PROVIDER_MODEL_ENV)
        or os.environ.get(OPENAI_MODEL_ENV)
        or DEFAULT_ENV_PROVIDER_MODEL
    ).strip()
    if not name or not model:
        return None
    if name.casefold() in DISABLED_PROVIDER_NAMES:
        raise ValueError(f"provider {name!r} is disabled")
    return ProviderConfig(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        wire_api="responses",
        reasoning_effort="high",
    )


def load_providers(path: os.PathLike | str | None = None) -> dict[str, ProviderConfig]:
    configured = os.environ.get(PROVIDERS_PATH_ENV)
    if path is not None:
        p = Path(path)
    elif configured:
        p = Path(configured).expanduser()
    else:
        p = DEFAULT_PROVIDERS_PATH
    if not p.exists():
        # An explicitly supplied path must remain authoritative and fail
        # closed.  Only the ordinary default lookup can use ephemeral process
        # credentials as a fallback.
        if path is None and not configured:
            env_provider = _environment_provider()
            if env_provider is not None:
                return {env_provider.name: env_provider}
        raise FileNotFoundError(f"LLM provider config not found: {p}")
    raw = json.loads(p.read_text())
    out: dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():
        if name.casefold() in DISABLED_PROVIDER_NAMES:
            continue
        out[name] = ProviderConfig(
            name=name,
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg.get("model", name),
            wire_api=cfg.get("wire_api", "responses"),
            reasoning_effort=cfg.get("reasoning_effort", "high"),
        )
    # A repository-local provider file remains the default and is authoritative
    # when a caller supplied it explicitly.  For the ordinary default lookup,
    # also expose an environment-injected provider under its own name.  This
    # lets a runner explicitly select a healthy temporary/managed endpoint via
    # INTENTIR_PROVIDER_NAME without serializing its credentials into a local
    # JSON file or silently replacing the configured default transport.
    if path is None and not configured:
        env_provider = _environment_provider()
        if env_provider is not None:
            out.setdefault(env_provider.name, env_provider)
    return out


def get_provider(
    name: str | None = None, path: os.PathLike | str | None = None
) -> ProviderConfig:
    """Return one enabled provider, defaulting to environment ``openai``.

    An omitted provider never inherits the first entry of a local credentials
    file, because file ordering cannot choose the active transport.
    """
    if name is not None and name.casefold() in DISABLED_PROVIDER_NAMES:
        raise ValueError(f"provider {name!r} is disabled")
    providers = load_providers(path)
    if not providers:
        raise ValueError("no providers configured")
    if name is None:
        name = (
            os.environ.get(PROVIDER_NAME_ENV, "").strip()
            or DEFAULT_ENV_PROVIDER_NAME
        )
    if name.casefold() in DISABLED_PROVIDER_NAMES:
        raise ValueError(f"provider {name!r} is disabled")
    if name not in providers:
        raise KeyError(f"provider {name!r} not in {list(providers)}")
    return providers[name]
