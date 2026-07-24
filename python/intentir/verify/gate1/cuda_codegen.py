"""Extraction and fail-closed policy for model-generated CUDA Replay code.

CUDA Replay is compiled to PTX but never linked or launched.  The policy still
keeps the translation unit deliberately small: one public ``__global__`` entry,
an optional set of ``__device__`` helpers, and a fixed public-header allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


class UnsafeCudaReplayCodeError(ValueError):
    """The CUDA translation unit is outside the reviewed Replay subset."""


@dataclass(frozen=True)
class CudaModuleValidation:
    kernel_name: str
    global_functions: tuple[str, ...]
    device_helpers: tuple[str, ...]


_FENCE_RE = re.compile(
    r"```[ \t]*(?P<label>[A-Za-z0-9_+.-]*)[ \t]*\r?\n(?P<body>.*?)```",
    re.DOTALL,
)
_CUDA_LABELS = {"", "c", "cc", "cpp", "cu", "cuda", "c++"}
_ALLOWED_HEADERS = {
    "cuda_bf16.h",
    "cuda_fp16.h",
    "cuda_runtime.h",
    "math.h",
    "stdint.h",
}
_FUNCTION_RE = re.compile(
    r"(?P<prefix>(?:(?:extern\s+\"C\"|__global__|__device__|__forceinline__|"
    r"static|inline)\s+)*)"
    r"(?P<return>[A-Za-z_][A-Za-z0-9_:<>]*)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
_FORBIDDEN_TOKENS = re.compile(
    r"\b(?:asm|__asm__|dlopen|exec|execl|fork|fopen|freopen|main|open|popen|"
    r"printf|puts|remove|rename|socket|system|unlink)\b"
)


def extract_cuda_code(response: str) -> str:
    """Extract one CUDA translation unit from a raw response or one fence."""

    if not isinstance(response, str) or not response.strip():
        raise UnsafeCudaReplayCodeError("CUDA Replay response is empty")
    fences = list(_FENCE_RE.finditer(response))
    if not fences:
        if "```" in response:
            raise UnsafeCudaReplayCodeError("CUDA Replay contains an unterminated fence")
        return response.strip() + "\n"
    if len(fences) != 1 or response[: fences[0].start()].strip() or response[fences[0].end() :].strip():
        raise UnsafeCudaReplayCodeError(
            "CUDA Replay must be raw source or exactly one source fence"
        )
    label = fences[0].group("label").lower()
    if label not in _CUDA_LABELS:
        raise UnsafeCudaReplayCodeError(f"unsupported CUDA fence label {label!r}")
    return fences[0].group("body").strip() + "\n"


def _without_comments_and_literals(code: str) -> str:
    """Erase comments/string contents while retaining enough syntax to inspect."""

    token = re.compile(
        r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.DOTALL,
    )
    return token.sub(lambda match: "\n" * match.group(0).count("\n"), code)


def validate_cuda_module(
    code: str,
    *,
    required_kernel: str,
) -> CudaModuleValidation:
    """Validate a compile-only CUDA Replay translation unit."""

    if not required_kernel.isidentifier() or "__" in required_kernel:
        raise ValueError("invalid canonical CUDA Replay function")
    if not isinstance(code, str) or not code.strip():
        raise UnsafeCudaReplayCodeError("CUDA Replay module is empty")

    for line_number, raw in enumerate(code.splitlines(), start=1):
        line = raw.strip()
        if not line.startswith("#"):
            continue
        include = re.fullmatch(r"#\s*include\s*<([^>]+)>\s*", line)
        if include is None or include.group(1) not in _ALLOWED_HEADERS:
            raise UnsafeCudaReplayCodeError(
                f"line {line_number}: unsupported CUDA preprocessor directive"
            )

    inspected = _without_comments_and_literals(code)
    forbidden = _FORBIDDEN_TOKENS.search(inspected)
    if forbidden is not None:
        raise UnsafeCudaReplayCodeError(
            f"forbidden host-side CUDA token {forbidden.group(0)!r}"
        )

    globals_: list[str] = []
    helpers: list[str] = []
    for match in _FUNCTION_RE.finditer(inspected):
        prefix = match.group("prefix")
        name = match.group("name")
        if "__" in name and name != required_kernel:
            raise UnsafeCudaReplayCodeError("dunder CUDA function names are forbidden")
        if "__global__" in prefix:
            globals_.append(name)
        elif "__device__" in prefix:
            helpers.append(name)
        else:
            raise UnsafeCudaReplayCodeError(
                f"host-side function {name!r} is forbidden in CUDA Replay"
            )
    if globals_ != [required_kernel]:
        raise UnsafeCudaReplayCodeError(
            "CUDA Replay must define exactly one canonical __global__ kernel"
        )
    return CudaModuleValidation(required_kernel, tuple(globals_), tuple(helpers))


def extract_cuda_module(response: str, *, required_kernel: str) -> str:
    code = extract_cuda_code(response)
    validate_cuda_module(code, required_kernel=required_kernel)
    return code


__all__ = [
    "CudaModuleValidation",
    "UnsafeCudaReplayCodeError",
    "extract_cuda_code",
    "extract_cuda_module",
    "validate_cuda_module",
]
