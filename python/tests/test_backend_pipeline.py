from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import intentir.backend.compile as backend_compile
import intentir.backend.pipeline as backend_pipeline
from intentir.backend.pipeline import LoweredProgram, lower_to_cuda
from intentir.backend.selection import LoweringTarget, ProbeAttempt, VariantSelection


_SOURCE = "module { func.func @f() { return } }\n"
_SPECIALIZED = "module { gpu.launch blocks(%c1, %c1, %c1) threads(%c1, %c1, %c1) }\n"


def _kernel(*, cubin: bytes = b"\x7fELFfake") -> backend_compile.CompiledKernel:
    return backend_compile.CompiledKernel(
        cubin=cubin,
        kernel_name="kernel",
        binary_name="kernel_module",
        binary_mlir="module { gpu.binary @kernel_module [] }\n",
    )


def _selection(kind: str) -> VariantSelection:
    source_digest = hashlib.sha256(_SOURCE.encode("utf-8")).hexdigest()
    if kind == "generic":
        return VariantSelection(
            pipeline_kind="generic",
            target=LoweringTarget(),
            input_sha256=source_digest,
            fast_path_id=None,
            pass_argument=None,
            scf_to_cf=False,
            probe_output_sha256=None,
            attempts=(),
        )
    specialized_digest = hashlib.sha256(_SPECIALIZED.encode("utf-8")).hexdigest()
    attempt = ProbeAttempt(
        fast_path_id="cuda.test.v1",
        pass_argument="-convert-intent-test-to-gpu",
        target_match=True,
        status="selected",
        direct_hit=True,
        gpu_launch_count=1,
        probe_output_sha256=specialized_digest,
    )
    return VariantSelection(
        pipeline_kind="specialized",
        target=LoweringTarget(),
        input_sha256=source_digest,
        fast_path_id="cuda.test.v1",
        pass_argument="-convert-intent-test-to-gpu",
        scf_to_cf=True,
        probe_output_sha256=specialized_digest,
        attempts=(attempt,),
    )


def test_specialized_pipeline_replays_the_exact_probe_before_codegen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("specialized")
    observed: dict[str, object] = {}
    monkeypatch.setattr(backend_pipeline, "select_variant", lambda *a, **k: selection)
    monkeypatch.setattr(
        backend_compile, "_run", lambda command, stdin=None: _SPECIALIZED
    )

    def fake_lower(module: str, *, scf_to_cf: bool = False):
        observed["module"] = module
        observed["scf_to_cf"] = scf_to_cf
        return [_kernel()]

    monkeypatch.setattr(
        backend_compile, "lower_memref_module_to_kernels", fake_lower
    )

    result = lower_to_cuda(_SOURCE)

    assert result.selection.pipeline_kind == "specialized"
    assert observed == {"module": _SPECIALIZED, "scf_to_cf": True}
    assert result.specialized_mlir_sha256 == selection.probe_output_sha256
    assert result.to_record()["kernels"][0]["cubin_is_elf"] is True


def test_generic_pipeline_uses_the_real_generic_compiler_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("generic")
    observed: list[Path] = []
    monkeypatch.setattr(backend_pipeline, "select_variant", lambda *a, **k: selection)

    def fake_compile(path: Path):
        path = Path(path)
        assert path.read_text(encoding="utf-8") == _SOURCE
        observed.append(path)
        return [_kernel()]

    monkeypatch.setattr(backend_compile, "compile_all_kernels", fake_compile)

    result = lower_to_cuda(_SOURCE)

    assert result.selection.pipeline_kind == "generic"
    assert len(result.kernels) == 1
    assert len(observed) == 1


def test_pipeline_rejects_a_target_different_from_the_configured_backend() -> None:
    with pytest.raises(ValueError, match="configured CUDA backend"):
        lower_to_cuda(_SOURCE, target=LoweringTarget(chip="sm_90"))


def test_lowered_program_rejects_non_elf_device_artifacts() -> None:
    selection = _selection("generic")
    with pytest.raises(ValueError, match="not an ELF cubin"):
        LoweredProgram(
            selection=selection,
            kernels=(_kernel(cubin=b"not-elf"),),
            source_mlir_sha256=selection.input_sha256,
        )


def test_specialized_pipeline_fails_if_replay_differs_from_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection("specialized")
    monkeypatch.setattr(backend_pipeline, "select_variant", lambda *a, **k: selection)
    monkeypatch.setattr(
        backend_compile,
        "_run",
        lambda command, stdin=None: _SPECIALIZED + "// changed\n",
    )

    with pytest.raises(RuntimeError, match="changed between selection"):
        lower_to_cuda(_SOURCE)


@pytest.mark.skipif(
    not Path(backend_compile.INTENT_OPT).exists(), reason="intent-opt not built"
)
@pytest.mark.skipif(
    not (Path(backend_compile.CUDA_BIN) / "ptxas").exists(),
    reason="ptxas not available",
)
def test_neutral_bridge_really_compiles_generic_and_specialized_cubins() -> None:
    add_module = Path(__file__).resolve().parents[2] / "test" / "add.mlir"
    generic = lower_to_cuda(add_module, allow_specialized=False)
    fast_path_id = "cuda.elementwise.row.vectorized.v1"
    specialized = lower_to_cuda(
        add_module,
        only_fast_path=fast_path_id,
        pass_overrides={
            fast_path_id: "-convert-intent-elementwise-to-gpu=block-threads=16"
        },
    )

    assert generic.selection.pipeline_kind == "generic"
    assert specialized.selection.pipeline_kind == "specialized"
    assert generic.kernels[0].cubin.startswith(b"\x7fELF")
    assert specialized.kernels[0].cubin.startswith(b"\x7fELF")
