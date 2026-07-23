from __future__ import annotations

import hashlib

import numpy as np
import pytest

import intentir.backend.cuda.program as cuda_program
from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.model import IntentReturn
from intentir.backend.protocol import (
    BackendProtocolError,
    BackendTarget,
    ComparisonEvidence,
    ComparisonKind,
    LoweringRequest,
)
from intentir.verify.gate1.model import (
    AccessKind,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    ImplementationConfig,
    OutputMapping,
    ParameterMode,
)
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


class _FakeCudaArtifact:
    def __init__(self, source: str) -> None:
        self.source_mlir_sha256 = hashlib.sha256(source.encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {"source_mlir_sha256": self.source_mlir_sha256, "kernel_count": 1}


def _attach_dense_copy_boundary(artifact, index_expression: str = "i0") -> None:
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "t0", node_ref="n0",
            index_expression=index_expression,
        ),
        BoundaryAccess(
            "a1", AccessKind.STORE, "t1", node_ref="n0",
            index_expression=index_expression,
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
    ]


def _make_masked_pointwise_artifact():
    artifact = make_copy_artifact()
    artifact.j.intent.body[0].attrs["fn"] = "tanh"
    _attach_dense_copy_boundary(artifact, "i0")
    artifact.j.boundary.domains = {"d0": "0 <= i0 and i0 < x0"}
    artifact.j.boundary.accesses[0].predicate = "i0 < x0"
    artifact.j.boundary.accesses[0].fill_value = 0.0
    artifact.j.boundary.accesses[1].predicate = "i0 < x0"
    return artifact


def _make_masked_sum_artifact():
    artifact = make_reduce_artifact(explicit_accesses=False)
    artifact.j.intent.body[0].type = {
        "dtype": "f32",
        "shape": ["x0", "x1"],
    }
    artifact.j.intent.body[1].result = "v1"
    artifact.j.intent.body[1].type = {"dtype": "f32", "shape": ["x0"]}
    artifact.j.intent.returns = [IntentReturn("t1", "v1")]
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < x0",
        "d1": "0 <= i1 and i1 < x1",
    }
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0", AccessKind.LOAD, "t0", predicate="i1 < x1",
            fill_value=0.0, node_ref="n0", index_expression="i0,i1",
        ),
        BoundaryAccess(
            "a1", AccessKind.STORE, "t1", node_ref="n1",
            index_expression="i0",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n1", "e1", "a1"),
    ]
    artifact.j.boundary.output_mappings = [
        OutputMapping("t1", "v1", "n1")
    ]
    artifact.j.boundary.reductions[0].tile_axis = 0
    return artifact


def test_cuda_lowering_consumes_the_same_j_and_separate_c(monkeypatch) -> None:
    artifact = canonicalize_gate1_artifact(
        make_reduce_artifact(explicit_accesses=False)
    )
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


def test_cuda_and_refeval_consume_the_same_explicit_boundary_projection(
    monkeypatch,
) -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.body[0].attrs["fn"] = "tanh"
    _attach_dense_copy_boundary(artifact, "[i0]")

    def fake_lower(source, *, target, allow_specialized):
        del target, allow_specialized
        return _FakeCudaArtifact(source)

    monkeypatch.setattr(cuda_program, "lower_to_cuda", fake_lower)
    request = LoweringRequest(
        artifact.j,
        cuda_program.cuda_target("sm_120a"),
        "generic",
        artifact.c,
    )
    lowered = cuda_program.lower_j_to_cuda(request, {"x0": 8})
    spec = ScalarLoweringSpec.create(
        request.j,
        dims={"x0": 8},
        input_bindings={"t0": "input"},
        output_bindings={"t1": "output"},
    )
    reference = ScalarLoweringRunner().run(
        spec, {"input": np.arange(8, dtype=np.float32)}
    )

    assert reference.ok, reference.error
    assert lowered.boundary_consumption["supported"] is True
    assert lowered.boundary_consumption["boundary_projection"] == (
        reference.metadata["boundary_projection"]
    )
    assert "access:a0" in lowered.boundary_consumption["consumed"]
    assert "effect:e1" in lowered.boundary_consumption["consumed"]


@pytest.mark.parametrize(
    ("factory", "dims", "profile", "operation"),
    [
        (
            _make_masked_pointwise_artifact,
            {"x0": 8},
            "logical_domain_mask_erasure_v1",
            "intent.map_unary",
        ),
        (
            _make_masked_sum_artifact,
            {"x0": 2, "x1": 4},
            "logical_domain_reduction_padding_v1",
            "intent.reduce",
        ),
    ],
)
def test_cuda_projects_supported_mask_profiles_before_codegen(
    monkeypatch, factory, dims, profile, operation
) -> None:
    artifact = factory()

    def fake_lower(source, *, target, allow_specialized):
        del target, allow_specialized
        return _FakeCudaArtifact(source)

    monkeypatch.setattr(cuda_program, "lower_to_cuda", fake_lower)
    lowered = cuda_program.lower_j_to_cuda(
        LoweringRequest(
            artifact.j,
            cuda_program.cuda_target("sm_120a"),
            "generic",
            artifact.c,
        ),
        dims,
    )

    projection = lowered.boundary_consumption["boundary_projection"]
    assert lowered.boundary_consumption["supported"] is True
    assert projection["profile"] == profile
    assert operation in lowered.intent_mlir


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("affine", "direct ordered axis vector"),
        ("effect_mismatch", "disagrees with its effect"),
        ("inout", "does not implement inout/alias ABI"),
    ],
)
def test_cuda_fails_before_codegen_for_non_projectable_boundary(
    monkeypatch, mutation, expected
) -> None:
    artifact = make_copy_artifact()
    _attach_dense_copy_boundary(artifact)
    if mutation == "affine":
        artifact.j.boundary.accesses[0].index_expression = "i0 + 1"
    elif mutation == "effect_mismatch":
        artifact.j.boundary.effects[1].target = "t0"
    elif mutation == "inout":
        artifact.j.signature.parameters[0].mode = ParameterMode.INOUT
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(mutation)
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
    with pytest.raises(
        cuda_program.CudaJLoweringError,
        match=expected,
    ):
        cuda_program.lower_j_to_cuda(request, {"x0": 8})
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
