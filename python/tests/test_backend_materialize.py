from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from direct_fixtures import make_reduce_artifact
from intentir.backend.materialize import (
    MaterializationError,
    attach_c,
    materialize_j,
)
from intentir.backend.protocol import BackendTarget
from intentir.ir.canonical_j import canonicalize_gate1_artifact
from intentir.ir.production_snapshot import load_production_snapshot
from intentir.verify.gate2.scalar import ScalarLoweringRunner, ScalarLoweringSpec


_ROOT = Path(__file__).resolve().parents[2]


def test_materialization_is_deterministic_and_does_not_read_c() -> None:
    artifact = canonicalize_gate1_artifact(
        make_reduce_artifact(explicit_accesses=False)
    )
    left = materialize_j(artifact.j, {"x0": 2, "x1": 4})
    artifact.c.values["c0"] = 1024
    right = materialize_j(artifact.j, {"x0": 2, "x1": 4})

    assert left.intent_mlir == right.intent_mlir
    assert left.mlir_digest == right.mlir_digest
    assert left.j_digest == right.j_digest


def test_attach_c_keeps_base_ir_and_only_records_erasable_priors() -> None:
    artifact = canonicalize_gate1_artifact(
        make_reduce_artifact(explicit_accesses=False)
    )
    base = materialize_j(artifact.j, {"x0": 2, "x1": 4})
    state = attach_c(base, artifact.c, BackendTarget("cuda", "sm_120a"))

    assert state.base_ir == base.intent_mlir
    assert state.erase_c() == base.intent_mlir
    assert state.to_record()["erase_c_matches_base"] is True
    assert "tile_hint" not in base.intent_mlir


def test_attach_none_c_has_the_same_reachable_base() -> None:
    artifact = canonicalize_gate1_artifact(
        make_reduce_artifact(explicit_accesses=False)
    )
    base = materialize_j(artifact.j, {"x0": 2, "x1": 4})
    state = attach_c(base, None, BackendTarget("cuda", "sm_120a"))

    assert state.priors == ()
    assert state.c_digest is None
    assert state.base_ir == base.intent_mlir


def test_only_j_declared_constexpr_input_can_be_specialized() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "dyt",
    )
    generic = materialize_j(snapshot.j, {"x0": 127})
    specialized = materialize_j(
        snapshot.j,
        {"x0": 127},
        constexpr_bindings={"t4": True},
    )

    assert "t4" in generic.runtime_input_names
    assert "t4" not in specialized.runtime_input_names
    assert "dense<true>" in specialized.intent_mlir
    assert specialized.j_digest == generic.j_digest
    assert specialized.mlir_digest != generic.mlir_digest

    with pytest.raises(MaterializationError, match="not authorized"):
        materialize_j(
            snapshot.j,
            {"x0": 127},
            constexpr_bindings={"t0": 1.0},
        )


def test_constexpr_specialization_preserves_value_dimension_b_relation() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "xformers_rms_norm",
    )
    specialized = materialize_j(
        snapshot.j,
        {"x0": 127},
        constexpr_bindings={"t3": 127, "t4": True},
    )

    assert "t3" not in specialized.runtime_input_names
    assert "t4" not in specialized.runtime_input_names
    with pytest.raises(MaterializationError, match="value/dimension equality"):
        materialize_j(
            snapshot.j,
            {"x0": 127},
            constexpr_bindings={"t3": 126, "t4": True},
        )


def test_tvd_materializes_closed_loop_and_constexpr_effect_control() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "tvd",
    )

    materialized = materialize_j(
        snapshot.j,
        {"x0": 2, "x1": 64},
        constexpr_bindings={"t3": -100, "t5": False, "t6": 0},
    )

    assert materialized.form == "intentir.materialize.logical-full-axis-step.v1"
    assert materialized.consumed_structures == (
        "loop:n0:logical_full_axis_step",
        "effect_if:n10:constexpr_false",
    )
    assert materialized.runtime_input_names == ("t0", "t1", "t2", "t4", "t7")
    assert "effect_if" not in materialized.intent_mlir
    assert "intent.loop" not in materialized.intent_mlir
    assert "tensor<2xi64>" in materialized.intent_mlir


def test_tvd_structured_reference_consumes_loop_capture_and_effect_if() -> None:
    snapshot, _path = load_production_snapshot(
        _ROOT / "corpus" / "production-jc-v1",
        "tvd",
    )
    spec = ScalarLoweringSpec.create(
        snapshot.j,
        dims={"x0": 2, "x1": 64},
        input_bindings={item.name: item.name for item in snapshot.j.intent.inputs},
        output_bindings={"t8": "loss", "t9": "grads"},
    )
    p = np.linspace(-1, 1, 128, dtype=np.float32).reshape(2, 64)
    q = np.flip(p, axis=1).copy()
    scale = np.asarray(0.75, dtype=np.float32)
    values = {
        "t0": p,
        "t1": q,
        "t2": np.asarray([1, 2], dtype=np.int64),
        "t3": np.asarray(-100, dtype=np.int64),
        "t4": scale,
        "t5": np.asarray(False, dtype=np.bool_),
        "t6": np.asarray(0, dtype=np.int32),
        "t7": np.asarray(64, dtype=np.int32),
    }

    result = ScalarLoweringRunner().run(spec, values)

    assert result.ok, result.error
    np.testing.assert_allclose(result.outputs["loss"], 0.5 * np.abs(p - q))
    np.testing.assert_allclose(
        result.outputs["grads"],
        np.where(p > q, 0.5 * scale, -0.5 * scale),
    )
    evidence = result.metadata["execution_evidence"]
    assert evidence["loop_count"] == 1
    assert evidence["effect_if_count"] == 1
