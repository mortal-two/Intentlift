from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import intentir.backend.cuda.compile as backend_compile
import intentir.backend.cuda.program as cuda_program
from direct_fixtures import make_copy_artifact
from intentir.backend.protocol import LoweringRequest
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


class _FakeCudaArtifact:
    def __init__(self, source: str) -> None:
        self.source = source


def _add_artifact():
    artifact = make_copy_artifact()
    operation = artifact.j.intent.body[0]
    operation.op = "map_binary"
    operation.operands = ["t0", "t0"]
    operation.attrs["fn"] = "add"
    operation.type = None
    return artifact


def _fake_lowering(monkeypatch: pytest.MonkeyPatch):
    artifact = _add_artifact()

    def fake_lower(source, *, target, allow_specialized):
        del target, allow_specialized
        return _FakeCudaArtifact(source)

    monkeypatch.setattr(cuda_program, "lower_to_cuda", fake_lower)
    return cuda_program.lower_j_to_cuda(
        LoweringRequest(
            artifact.j,
            cuda_program.cuda_target(backend_compile.CHIP),
            "generic",
            artifact.c,
        ),
        {"x0": 33},
    )


def test_execute_j_cuda_uses_logical_input_and_output_handles(monkeypatch) -> None:
    lowering = _fake_lowering(monkeypatch)
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_execute(program, mlir, inputs):
        observed.update(program=program, mlir=mlir, inputs=inputs)
        return {"out0": sentinel}

    monkeypatch.setattr(cuda_program, "execute_cuda_artifact", fake_execute)

    outputs = cuda_program.execute_j_cuda(lowering, {"t0": "runtime-value"})

    assert outputs == {"t1": sentinel}
    assert observed == {
        "program": lowering.artifact,
        "mlir": lowering.intent_mlir,
        "inputs": {"t0": "runtime-value"},
    }


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        ({}, "missing=t0"),
        ({"t0": "value", "arg0": "positional"}, "unexpected=arg0"),
        ({"arg0": "positional"}, "missing=t0; unexpected=arg0"),
        ({0: "positional"}, "input keys must be J logical handle strings"),
    ],
)
def test_execute_j_cuda_rejects_non_j_input_sets_before_runtime(
    monkeypatch, inputs, expected
) -> None:
    lowering = _fake_lowering(monkeypatch)
    called = False

    def forbidden_execute(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("runtime must not observe an invalid J input set")

    monkeypatch.setattr(cuda_program, "execute_cuda_artifact", forbidden_execute)

    with pytest.raises(cuda_program.CudaJExecutionError, match=expected):
        cuda_program.execute_j_cuda(lowering, inputs)
    assert called is False


def test_execute_j_cuda_rejects_runtime_output_arity_drift(monkeypatch) -> None:
    lowering = _fake_lowering(monkeypatch)
    monkeypatch.setattr(
        cuda_program,
        "execute_cuda_artifact",
        lambda *args, **kwargs: {"out0": "value", "out1": "extra"},
    )

    with pytest.raises(
        cuda_program.CudaJExecutionError,
        match="runtime outputs differ from J logical outputs",
    ):
        cuda_program.execute_j_cuda(lowering, {"t0": "value"})


def _cuda_execution_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(
        Path(backend_compile.INTENT_OPT).exists()
        and (Path(backend_compile.CUDA_BIN) / "ptxas").exists()
        and torch.cuda.is_available()
    )


@pytest.mark.skipif(
    not _cuda_execution_available(),
    reason="Intent/CUDA production execution toolchain is unavailable",
)
@pytest.mark.parametrize("extent", [32, 33])
def test_real_j_cuda_execution_matches_independent_refeval(extent: int) -> None:
    import torch

    artifact = _add_artifact()
    request = LoweringRequest(
        artifact.j,
        cuda_program.cuda_target(backend_compile.CHIP),
        "generic",
        artifact.c,
    )
    lowering = cuda_program.lower_j_to_cuda(request, {"x0": extent})
    host_input = np.linspace(-3.25, 5.75, extent, dtype=np.float32)
    device_input = torch.from_numpy(host_input.copy()).to("cuda")

    actual = cuda_program.execute_j_cuda(lowering, {"t0": device_input})
    reference_spec = ScalarLoweringSpec.create(
        lowering.request.j,
        dims={"x0": extent},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    reference = ScalarLoweringRunner().run(
        reference_spec,
        {"input": host_input},
    )

    assert reference.ok, reference.error
    assert lowering.artifact.kernels
    assert all(kernel.cubin.startswith(b"\x7fELF") for kernel in lowering.artifact.kernels)
    assert set(actual) == {"t1"}
    np.testing.assert_array_equal(
        actual["t1"].cpu().numpy(),
        reference.outputs["output"],
    )

    with pytest.raises(ValueError, match="shape .* does not match declared"):
        cuda_program.execute_j_cuda(
            lowering,
            {
                "t0": torch.zeros(
                    extent - 1,
                    dtype=torch.float32,
                    device="cuda",
                )
            },
        )
    with pytest.raises(ValueError, match="dtype .* does not match declared"):
        cuda_program.execute_j_cuda(
            lowering,
            {
                "t0": torch.zeros(
                    extent,
                    dtype=torch.float16,
                    device="cuda",
                )
            },
        )
