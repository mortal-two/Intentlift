from __future__ import annotations

import copy
from pathlib import Path

import pytest

from intentir.backend.dev_j_variant import (
    J_DEV_ALLOWED_CLAIMS,
    J_DEV_CERTIFICATION_STATUS,
    JDevEdit,
    JDevVariant,
    JDevVariantError,
    create_j_dev_variant,
    load_j_dev_variant,
)
from intentir.backend.dev_workload import DevWorkloadConfig, prepare_dev_workload
from intentir.backend.materialize import materialize_j
from intentir.ir.production_snapshot import content_digest, load_production_snapshot
from intentir.verify.gate1.model import ImplementationConfig
from intentir.verify.gate2.scalar import ScalarLoweringRunner
from intentir.verify.scrc_workload_run import build_workload_jc_suite
from intentir.verify.workload_adapters import TRITON_WORKLOAD_ADAPTERS


_ROOT = Path(__file__).resolve().parents[2]
_CATALOG = _ROOT / "corpus" / "production-jc-v1"
_EVIDENCE = (
    "Dataset/Liger-Kernel/benchmark/scripts/benchmark_rms_norm.py:"
    "Llama-3-8B-M2048-H4096",
)


def _base():
    return load_production_snapshot(_CATALOG, "liger_rms_norm")[0]


def _row_symbol_edits() -> tuple[JDevEdit, ...]:
    return (
        JDevEdit(
            "/j/intent/intent_function/inputs/0/shape/0",
            2,
            "x1",
            _EVIDENCE,
        ),
        JDevEdit(
            "/j/intent/intent_function/outputs/0/shape/0",
            2,
            "x1",
            _EVIDENCE,
        ),
        JDevEdit(
            "/j/intent/intent_function/outputs/1/shape/0",
            2,
            "x1",
            _EVIDENCE,
        ),
        JDevEdit(
            "/j/boundary/domains/d0",
            "(0 <= k0) and (k0 < 2)",
            "(0 <= k0) and (k0 < x1)",
            _EVIDENCE,
        ),
    )


def _variant():
    return create_j_dev_variant(
        _base(),
        source_kernel_id="high_performance:liger_rms_norm",
        shape_config_id="llama3_8b_m2048_h4096",
        dtype_config_id="fp32_boundary_fp32_accum",
        edits=_row_symbol_edits(),
    )


def test_real_rmsnorm_dev_variant_is_detached_canonical_and_materializable() -> None:
    base = _base()
    base_j = copy.deepcopy(base.j.to_json())
    variant = _variant()

    assert base.j.to_json() == base_j
    assert variant.base_j_digest == base.j_digest
    assert variant.variant_j_digest != base.j_digest
    assert variant.j.intent.inputs[0].shape == ["x1", "x0"]
    assert variant.j.intent.outputs[0].shape == ["x1", "x0"]
    assert variant.j.intent.outputs[1].shape == ["x1"]
    assert variant.j.intent.body[16].type["shape"] == ["x1", "x0"]
    assert variant.j.boundary.domains["d0"] == "(0 <= k0) and (k0 < x1)"

    materialized = materialize_j(
        variant.j,
        {"x0": 4096, "x1": 2048},
        constexpr_bindings={"t5": 1, "t6": False},
    )
    assert "tensor<2048x4096xf32>" in materialized.intent_mlir


def test_dev_variant_receipt_round_trip_is_explicitly_uncertified() -> None:
    variant = _variant()
    payload = variant.to_dict()
    restored = JDevVariant.from_json(variant.to_json(indent=2))

    assert restored.to_dict() == payload
    assert payload["certification_status"] == J_DEV_CERTIFICATION_STATUS
    assert payload["allowed_claims"] == list(J_DEV_ALLOWED_CLAIMS)
    assert payload["j_unchanged"] is False
    assert payload["receipt_digest"] == variant.receipt_digest
    restored.validate_against_base(_base())


def test_dev_variant_rejects_stale_old_value_and_half_updated_shape() -> None:
    base = _base()
    stale = JDevEdit(
        "/j/intent/intent_function/inputs/0/shape/0",
        3,
        "x1",
        _EVIDENCE,
    )
    with pytest.raises(JDevVariantError, match="expected 3, observed 2"):
        create_j_dev_variant(
            base,
            source_kernel_id="high_performance:liger_rms_norm",
            shape_config_id="bad",
            dtype_config_id="fp32",
            edits=(stale,),
        )

    with pytest.raises(JDevVariantError, match="self-consistent"):
        create_j_dev_variant(
            base,
            source_kernel_id="high_performance:liger_rms_norm",
            shape_config_id="half_updated",
            dtype_config_id="fp32",
            edits=(_row_symbol_edits()[0],),
        )


def test_dev_variant_rejects_forbidden_identity_edit_and_receipt_tampering() -> None:
    with pytest.raises(JDevVariantError, match="limited to typed"):
        JDevEdit("/j/kernel_id", "scrc_k0", "other", _EVIDENCE)

    payload = _variant().to_dict()
    payload["certification_status"] = "SCRC_CERTIFIED"
    with pytest.raises(JDevVariantError, match="cannot claim certification"):
        JDevVariant.from_dict(payload)

    original = _variant()
    c_payload = original.c.to_json()
    c_payload["values"]["c0"] = 2048
    tampered = JDevVariant(
        base_snapshot_digest=original.base_snapshot_digest,
        base_j_digest=original.base_j_digest,
        base_c_digest=original.base_c_digest,
        base_certification_status=original.base_certification_status,
        source_kernel_id=original.source_kernel_id,
        shape_config_id=original.shape_config_id,
        dtype_config_id=original.dtype_config_id,
        edits=original.edits,
        j=original.j,
        c=ImplementationConfig.from_json(c_payload),
    )
    with pytest.raises(JDevVariantError, match="development C payload"):
        tampered.validate_against_base(_base())


def test_dev_variant_rejects_duplicate_paths() -> None:
    edit = _row_symbol_edits()[0]
    with pytest.raises(JDevVariantError, match="duplicate edit paths"):
        create_j_dev_variant(
            _base(),
            source_kernel_id="high_performance:liger_rms_norm",
            shape_config_id="duplicate",
            dtype_config_id="fp32",
            edits=(edit, edit),
        )


def test_real_flash_layernorm_bf16_variant_reelaborates_control_flow() -> None:
    base, _path = load_production_snapshot(_CATALOG, "flash_layer_norm_1pass")
    variant = load_j_dev_variant(
        _ROOT
        / "corpus"
        / "backend-j-dev-v1"
        / "flash_layer_norm_1pass_m2048_n4096_bf16.json"
    )

    variant.validate_against_base(base)
    assert variant.dtype_config_id == "bf16_boundary_fp32_accum"
    assert variant.c is not None and variant.c.values["c0"] == 4096
    assert [item.dtype for item in variant.j.intent.inputs[:7]] == ["bf16"] * 7
    assert [item.dtype for item in variant.j.intent.outputs[:3]] == ["bf16"] * 3
    assert variant.j.intent.body[21].op == "value_if"
    assert variant.j.intent.body[21].type == {"dtype": "f32", "shape": ["x1"]}
    assert variant.j.intent.body[33].type == {"dtype": "bf16", "shape": ["x0", "x1"]}


def test_real_softmax_natural_shape_variant_reelaborates_reductions() -> None:
    base, _path = load_production_snapshot(_CATALOG, "softmax")
    variant = load_j_dev_variant(
        _ROOT
        / "corpus"
        / "backend-j-dev-v1"
        / "liger_softmax_upstream_m4096_n4096_fp32.json"
    )

    variant.validate_against_base(base)
    assert variant.shape_config_id == "upstream_m4096_n4096_fp32"
    assert variant.j.intent.inputs[0].shape == [4096, "x0"]
    assert variant.j.intent.outputs[0].shape == [4096, "x0"]
    assert variant.j.intent.body[0].type == {"dtype": "f32", "shape": [4096, 1]}
    assert variant.j.intent.body[4].type == {
        "dtype": "f32",
        "shape": [4096, "x0"],
    }
    assert variant.j.boundary.domains["d0"] == "(0 <= k1) and (k1 < 4096)"
    assert variant.c is not None and variant.c.values["c0"] == 4096


def test_real_geglu_natural_shape_variant_reelaborates_fused_graph() -> None:
    base, _path = load_production_snapshot(_CATALOG, "geglu")
    variant = load_j_dev_variant(
        _ROOT
        / "corpus"
        / "backend-j-dev-v1"
        / "liger_geglu_upstream_m4096_n4096_fp32.json"
    )

    variant.validate_against_base(base)
    assert [item.shape for item in variant.j.intent.inputs] == [
        [4096, "x0"],
        [4096, "x0"],
    ]
    assert variant.j.intent.outputs[0].shape == [4096, "x0"]
    assert variant.j.intent.body[13].type == {
        "dtype": "f32",
        "shape": [4096, "x0"],
    }
    assert variant.j.boundary.domains["d0"] == "(0 <= k0) and (k0 < 4096)"
    assert variant.c is not None and variant.c.values["c0"] == 4096


def test_real_flash_layernorm_bf16_refdev_preserves_boundary_dtype() -> None:
    base, _path = load_production_snapshot(_CATALOG, "flash_layer_norm_1pass")
    variant = load_j_dev_variant(
        _ROOT
        / "corpus"
        / "backend-j-dev-v1"
        / "flash_layer_norm_1pass_m2048_n4096_bf16.json"
    )
    case_id = variant.source_kernel_id
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    half_buffers = {
        name: "bf16"
        for name in (
            "X",
            "Y",
            "W",
            "B",
            "RESIDUAL",
            "X1",
            "W1",
            "B1",
            "Y1",
            "RESIDUAL_OUT",
        )
    }
    prepared = prepare_dev_workload(
        DevWorkloadConfig(
            case_id=case_id,
            shape_config_id="closed_shape_bf16_reference_smoke",
            dtype_config_id="bf16_boundary_fp32_accum",
            base_adapter_digest=content_digest(adapter.to_json()),
            concrete_extents={"x0": 2, "x1": 127},
            buffer_shapes={},
            buffer_dtypes=half_buffers,
            scalar_values={},
            semantic_constexprs={},
            c_values={},
            grid=(2,),
            grid_contract=("M",),
            upstream_evidence=("BF16 dtype envelope reference smoke",),
        )
    )
    assert base.c is not None
    suite = build_workload_jc_suite(prepared, variant.j, base.c)
    result = ScalarLoweringRunner().run(
        suite.cases[0].refeval,
        suite.cases[0].values,
    )

    assert result.ok, result.error
    assert result.metadata["path"] == "canonical J -> typed structured CPU reference"
    assert result.outputs["y"].dtype.name == "bfloat16"
    assert result.outputs["y1"].dtype.name == "bfloat16"
    assert result.outputs["residual_out"].dtype.name == "bfloat16"
    assert result.outputs["mean"].dtype.name == "float32"
    assert result.outputs["rstd"].dtype.name == "float32"
