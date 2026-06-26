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


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    wire_api: str = "responses"
    reasoning_effort: str = "high"


def load_providers(path: os.PathLike | str | None = None) -> dict[str, ProviderConfig]:
    p = Path(path) if path is not None else DEFAULT_PROVIDERS_PATH
    if not p.exists():
        raise FileNotFoundError(f"LLM provider config not found: {p}")
    raw = json.loads(p.read_text())
    out: dict[str, ProviderConfig] = {}
    for name, cfg in raw.items():
        out[name] = ProviderConfig(
            name=name,
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg.get("model", name),
            wire_api=cfg.get("wire_api", "responses"),
            reasoning_effort=cfg.get("reasoning_effort", "high"),
        )
    return out


def get_provider(
    name: str | None = None, path: os.PathLike | str | None = None
) -> ProviderConfig:
    """Return one provider config; defaults to the first configured provider."""
    providers = load_providers(path)
    if not providers:
        raise ValueError("no providers configured")
    if name is None:
        name = next(iter(providers))
    if name not in providers:
        raise KeyError(f"provider {name!r} not in {list(providers)}")
    return providers[name]
