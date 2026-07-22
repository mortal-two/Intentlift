from __future__ import annotations

import hashlib

import pytest

import intentir.backend.cuda.program as cuda_program
from direct_fixtures import make_reduce_artifact
from intentir.backend.protocol import (
    BackendProtocolError,
    BackendTarget,
    ComparisonEvidence,
    ComparisonKind,
    LoweringRequest,
)
from intentir.verify.gate1.model import ImplementationConfig


class _FakeCudaArtifact:
    def __init__(self, source: str) -> None:
        self.source_mlir_sha256 = hashlib.sha256(source.encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {"source_mlir_sha256": self.source_mlir_sha256, "kernel_count": 1}


def test_cuda_lowering_consumes_the_same_j_and_separate_c(monkeypatch) -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    seen: dict[str, object] = {}

    def fake_lower(source, *, target, allow_specialized):
        seen.update(
            source=source,
            target=target,
            allow_specialized=allow_specialized,
        )
        return _FakeCudaArtifact(source)

    monkeypatch.setattr(cuda_program, "lower_to_cuda", fake_lower)
    request = LoweringRequest(
        artifact.j,
        cuda_program.cuda_target("sm_120a"),
        "generic",
        artifact.c,
    )
    lowered = cuda_program.lower_j_to_cuda(request, {"x0": 2, "x1": 4})

    assert request.j is artifact.j
    assert lowered.boundary_consumption["supported"] is True
    assert "reduction:n1" in lowered.boundary_consumption["consumed"]
    assert "intent.reduce" in str(seen["source"])
    assert seen["target"].chip == "sm_120a"
    assert seen["allow_specialized"] is False
    with pytest.raises(BackendProtocolError, match="comparison evidence"):
        lowered.conformance_record(test_domain={"x0": 2, "x1": 4}, comparisons=())
    record = lowered.conformance_record(
        test_domain={"x0": 2, "x1": 4},
        comparisons=(
            ComparisonEvidence(
                ComparisonKind.INTENT_TO_LOWERING,
                True,
                "1" * 64,
            ),
        ),
    )
    record.assert_conformant()
    assert record.j_digest == request.j_digest
    assert record.c_digest == request.c_digest


def test_cuda_fails_before_codegen_when_b_is_not_implemented(monkeypatch) -> None:
    artifact = make_reduce_artifact()
    called = False

    def forbidden_lower(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("codegen must not run")

    monkeypatch.setattr(cuda_program, "lower_to_cuda", forbidden_lower)
    request = LoweringRequest(
        artifact.j,
        cuda_program.cuda_target("sm_120a"),
        "generic",
        artifact.c,
    )
    with pytest.raises(cuda_program.CudaJLoweringError, match="explicit_access_relation"):
        cuda_program.lower_j_to_cuda(request, {"x0": 2, "x1": 4})
    assert called is False


def test_c_can_change_without_changing_j_identity() -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    changed_c = ImplementationConfig.from_json(
        {
            "slots": [
                {"name": "c0", "kind": "constexpr", "role": "tile_extent"}
            ],
            "values": {"c0": 256},
        }
    )
    left = LoweringRequest(
        artifact.j, BackendTarget("cuda", "sm_120a"), "generic", artifact.c
    )
    right = LoweringRequest(
        artifact.j, BackendTarget("cuda", "sm_120a"), "generic", changed_c
    )
    assert left.j is right.j
    assert left.j_digest == right.j_digest
    assert left.c_digest != right.c_digest
