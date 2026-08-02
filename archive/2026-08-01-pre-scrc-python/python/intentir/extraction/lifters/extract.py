"""Extraction driver: kernel source -> IntentIR proposal.

The extractor asks the provider for JSON and parses it into the project model.
Correctness is not decided here: the proposal must subsequently pass SCRC
Gate1 and Gate2.  The old NumPy smoke interpreter remains available only as a
legacy compatibility helper and is no longer on the extraction path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from intentir.ir.model import IntentFunction
from . import prompt as _prompt
from intentir.extraction.providers.client import call_responses
from intentir.extraction.providers.provider import ProviderConfig


@dataclass
class Provenance:
    model: str
    prompt_sha256: str
    attempts: int
    repairs: list[str] = field(default_factory=list)


@dataclass
class Extraction:
    fn: IntentFunction
    json_obj: dict
    provenance: Provenance


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_to_json(text: str) -> str:
    """Pull the JSON object out of a model reply (tolerate fences / prose)."""
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s:
        raise ValueError("no JSON object found in reply")
    return text[s : e + 1]


def _concretize_dims(fn: IntentFunction, default: int = 6) -> dict[str, int]:
    dims: dict[str, int] = {}
    for t in fn.inputs + fn.outputs:
        for d in t.shape:
            if isinstance(d, str):
                dims.setdefault(d, default)
    return dims


def smoke_interpret(fn: IntentFunction) -> dict[str, Any]:
    """Legacy smoke helper, kept off the active extraction/verification path."""

    import numpy as np

    from intentir.verify.back.legacy_v0.interpret import interpret

    dims = _concretize_dims(fn)
    rng = np.random.default_rng(0)
    inputs: dict[str, np.ndarray] = {}
    for t in fn.inputs:
        shape = tuple(dims.get(d, d) if isinstance(d, str) else d for d in t.shape)
        arr = rng.standard_normal(shape).astype(np.float32)
        if "i" in t.dtype and t.dtype != "f16":  # crude: integer-ish stays small
            pass
        inputs[t.name] = arr if shape else np.float32(rng.standard_normal())
    params = {v: 3 for v in dims}  # any tile hints get a concrete small value
    out = interpret(fn, inputs, params)
    for name, arr in out.items():
        if not np.all(np.isfinite(np.asarray(arr, dtype=np.float64))):
            # NaN/Inf is allowed for log(0) etc. only if the kernel truly does it;
            # for a smoke test on N(0,1) inputs it signals a lifting bug.
            raise ValueError(f"output {name!r} not finite on random inputs")
    return out


def _parse_candidate(raw_text: str) -> tuple[IntentFunction, dict]:
    js = _strip_to_json(raw_text)
    obj = json.loads(js)  # JSONDecodeError -> repair
    fn = IntentFunction.from_json(obj)  # KeyError/TypeError -> repair
    if not fn.outputs:
        raise ValueError("no outputs declared")
    return fn, obj


def extract_intent(
    source: str,
    name_hint: str,
    *,
    shape_hint: str = "",
    provider: ProviderConfig | None = None,
    max_repairs: int = 3,
    effort: str | None = None,
) -> Extraction:
    """Lift ``source`` to validated IntentIR via the live provider."""
    instructions = _prompt.build_instructions()
    user = _prompt.build_user_input(source, name_hint, shape_hint)
    prompt_sha = hashlib.sha256((instructions + "\x00" + user).encode()).hexdigest()
    repairs: list[str] = []

    convo = user
    for attempt in range(max_repairs + 1):
        reply = call_responses(instructions, convo, provider=provider, effort=effort)
        try:
            fn, obj = _parse_candidate(reply.text)
            return Extraction(
                fn=fn,
                json_obj=obj,
                provenance=Provenance(
                    model=reply.model,
                    prompt_sha256=prompt_sha,
                    attempts=attempt + 1,
                    repairs=repairs,
                ),
            )
        except Exception as e:  # structured repair feedback, bounded
            diag = f"{type(e).__name__}: {e}"
            repairs.append(diag)
            if attempt == max_repairs:
                raise RuntimeError(
                    f"extraction failed after {attempt + 1} tries; last: {diag}"
                ) from e
            convo = (
                user
                + "\n\nYour previous answer was rejected by the validator with:\n"
                + diag
                + "\nRe-emit corrected IntentIR JSON ONLY (no prose)."
            )
