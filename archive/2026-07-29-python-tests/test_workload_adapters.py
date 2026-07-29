from __future__ import annotations

import pytest

from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.triton_source_slice import (
    requires_triton_vararg_expansion,
)
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferRole,
    NoneBinding,
    ScalarBinding,
    TRITON_WORKLOAD_ADAPTERS,
    TritonWorkloadAdapter,
    prepare_triton_workload,
)
from intentir.verify.workload_manifest import build_workload_manifest


_FLASH_FWD = "high_performance:flash_attn.flash_attn_triton._fwd_kernel"
_FLASH_BWD = "high_performance:flash_attn.flash_attn_triton._bwd_kernel"
_LINEAR = "high_performance:flash_attn.ops.triton.linear.kernel_fwd"
_FUSED_ADD_RMS = "high_performance:liger_fused_add_rms_norm"
_SPARSEMAX = "high_performance:liger_sparsemax"
_FLASH_BWD_ONE_COL = (
    "high_performance:flash_attn.flash_attn_triton._bwd_kernel_one_col_block"
)
_QWEN2VL_MROPE = "high_performance:liger_qwen2vl_mrope"
_GROUP_NORM = "high_performance:liger_group_norm"
_JSD = "high_performance:liger_jsd"
_CROSS_ENTROPY = "high_performance:liger_cross_entropy"
_GEGLU = "high_performance:liger_geglu"
_POLY_NORM = "high_performance:liger_poly_norm"
_LLAMA4_ROPE = "high_performance:liger_llama4_rope"
_FLASH_LAYER_NORM = (
    "high_performance:flash_attn.ops.triton.layer_norm."
    "_layer_norm_fwd_1pass_kernel"
)
_ROPE = "high_performance:liger_rope"
_LAYER_NORM = "high_performance:liger_layer_norm"
_KL_DIV = "high_performance:liger_kl_div"
_TVD = "high_performance:liger_tvd"
_XFORMERS_RMS = (
    "high_performance:xformers.ops._triton.rmsnorm_kernels._rms_norm_kernel"
)
_XFORMERS_RMS_ADD = (
    "high_performance:xformers.ops._triton.rmsnorm_kernels."
    "_rms_norm_add_kernel"
)
_XFORMERS_SPLITK = (
    "high_performance:xformers.ops.fmha._triton.splitk_kernels."
    "_fwd_kernel_splitK"
)
_XFORMERS_SPLITK_REDUCE = (
    "high_performance:xformers.ops.fmha._triton.splitk_kernels."
    "_splitK_reduce_varargs"
)
_XFORMERS_SPLITK_REDUCE_BWD = (
    "high_performance:xformers.ops.fmha._triton.splitk_kernels."
    "_splitK_reduce_varargs_backward"
)


def test_adapter_map_and_manifest_readiness_are_exactly_aligned() -> None:
    manifest = build_workload_manifest()
    representatives = manifest.select(tier="representative")

    assert list(TRITON_WORKLOAD_ADAPTERS)[:6] == [
        entry.case_id for entry in representatives
    ]
    assert list(TRITON_WORKLOAD_ADAPTERS)[6:] == [
        _FLASH_FWD,
        _FLASH_BWD,
        _FUSED_ADD_RMS,
        _SPARSEMAX,
        _FLASH_BWD_ONE_COL,
        _QWEN2VL_MROPE,
        _GROUP_NORM,
        _JSD,
        _CROSS_ENTROPY,
        _GEGLU,
        _POLY_NORM,
        _LLAMA4_ROPE,
        _FLASH_LAYER_NORM,
        _ROPE,
        _LAYER_NORM,
        _KL_DIV,
        _TVD,
        _XFORMERS_RMS,
        _XFORMERS_RMS_ADD,
        _XFORMERS_SPLITK,
        _XFORMERS_SPLITK_REDUCE,
        _XFORMERS_SPLITK_REDUCE_BWD,
    ]
    assert len(TRITON_WORKLOAD_ADAPTERS) == 28
    assert {
        entry.case_id
        for entry in manifest.entries
        if entry.adapter_readiness == "declared"
    } == set(TRITON_WORKLOAD_ADAPTERS)


def test_buffer_initializer_is_closed_and_output_only_rejects_it() -> None:
    default = BufferBinding("X", BufferRole.INPUT, "f32", (8,))
    zeros = BufferBinding(
        "Y",
        BufferRole.INOUT,
        "f32",
        (8,),
        "y",
        BufferInitializer.ZEROS,
    )

    assert default.initializer is BufferInitializer.RANDOM
    assert zeros.initializer is BufferInitializer.ZEROS
    assert (
        BufferBinding("Z", BufferRole.OUTPUT, "f32", (8,)).initializer
        is None
    )
    with pytest.raises(ValueError):
        BufferBinding("Z", BufferRole.INPUT, "f32", (8,), initializer="ones")
    with pytest.raises(ValueError, match="output-only"):
        BufferBinding(
            "Z",
            BufferRole.OUTPUT,
            "f32",
            (8,),
            initializer=BufferInitializer.ZEROS,
        )


def test_shape_contract_separates_parametric_axes_from_current_valuation() -> None:
    buffer = BufferBinding(
        "X",
        BufferRole.INPUT,
        "f32",
        (2, 127),
        shape_contract=("h0", "h1"),
    )

    assert buffer.shape == (2, 127)
    assert buffer.shape_contract == ("h0", "h1")
    with pytest.raises(ValueError, match="rank differs"):
        BufferBinding(
            "X",
            BufferRole.INPUT,
            "f32",
            (2, 127),
            shape_contract=("h0",),
        )
    with pytest.raises(ValueError, match="static buffer shape contract"):
        BufferBinding(
            "X",
            BufferRole.INPUT,
            "f32",
            (2, 127),
            shape_contract=(3, "h0"),
        )
    with pytest.raises(ValueError, match="anonymous h"):
        BufferBinding(
            "X",
            BufferRole.INPUT,
            "f32",
            (2, 127),
            shape_contract=("rows", "cols"),
        )

    scalar = ScalarBinding("N", "i32", 127, shape_coordinate="h1")
    assert scalar.shape_coordinate == "h1"
    with pytest.raises(ValueError, match="integer valuation"):
        ScalarBinding("N", "f32", 127.0, shape_coordinate="h1")
    with pytest.raises(ValueError, match="integer dtype"):
        ScalarBinding("N", "f32", 127, shape_coordinate="h1")


def test_explicit_grid_shape_coordinate_is_checked_without_parsing_grid_prose() -> None:
    common = {
        "case_id": "high_performance:test",
        "buffers": (
            BufferBinding(
                "Y",
                BufferRole.OUTPUT,
                "f32",
                (2,),
                "y",
                shape_contract=("h0",),
            ),
        ),
        "scalars": (),
        "semantic_constexprs": (),
        "c_values": (),
        "grid": (2,),
        "grid_contract": ("human_rows_label",),
        "seed": 1,
    }
    prose_only = TritonWorkloadAdapter(**common)
    assert prose_only.grid_shape_coordinates == ()

    explicit = TritonWorkloadAdapter(
        **common, grid_shape_coordinates=("h0",)
    )
    assert explicit.grid_shape_coordinates == ("h0",)

    with pytest.raises(ValueError, match="explicit shape-coordinate valuation"):
        TritonWorkloadAdapter(
            **{**common, "grid": (3,)},
            grid_shape_coordinates=("h0",),
        )


def test_rowwise_and_capacity_adapters_expose_anonymous_shape_coordinates() -> None:
    dyt = prepare_triton_workload(
        TRITON_WORKLOAD_ADAPTERS["high_performance:liger_dyt"]
    )
    softmax = prepare_triton_workload(
        TRITON_WORKLOAD_ADAPTERS["high_performance:liger_softmax"]
    )
    preprocess = prepare_triton_workload(
        TRITON_WORKLOAD_ADAPTERS[
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_preprocess_do_o_dot"
        ]
    )

    assert dyt.compiler_view.bound_constexpr_shape_coordinates == {"N": "h1"}
    assert dyt.compiler_view.to_json()[
        "bound_constexpr_shape_coordinates"
    ] == {"N": "h1"}
    assert softmax.compiler_view.shape_coordinates == {"h0": 2, "h1": 127}
    assert softmax.compiler_view.grid_shape_coordinates == ("h0",)
    softmax_buffers = {
        item.formal: item for item in softmax.adapter.buffers
    }
    assert softmax_buffers["X_ptr"].shape_contract == ("h0", "h1")
    softmax_scalars = {
        item.formal: item for item in softmax.adapter.scalars
    }
    assert softmax_scalars["n_cols"].shape_coordinate == "h1"
    assert softmax.adapter.to_json()["buffers"][0]["shape"] == [2, 127]
    assert softmax.adapter.to_json()["buffers"][0]["shape_contract"] == [
        "h0",
        "h1",
    ]

    assert preprocess.compiler_view.shape_coordinates == {
        "h0": 1,
        "h1": 2,
        "h2": 33,
        "h3": 16,
        "h4": 64,
    }
    assert preprocess.compiler_view.grid_shape_coordinates == ()
    preprocess_scalars = {
        item.formal: item for item in preprocess.adapter.scalars
    }
    assert {
        name: preprocess_scalars[name].shape_coordinate
        for name in ("nheads", "seqlen_q", "seqlen_q_rounded", "headdim")
    } == {
        "nheads": "h1",
        "seqlen_q": "h2",
        "seqlen_q_rounded": "h4",
        "headdim": "h3",
    }

    flash_fwd = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_FLASH_FWD])
    flash_buffers = {item.formal: item for item in flash_fwd.adapter.buffers}
    assert flash_buffers["TMP"].role is BufferRole.INOUT


def test_runtime_attention_and_matrix_extents_are_parametric_not_valuations() -> None:
    flash_fwd = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_FLASH_FWD])
    flash_bwd = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_FLASH_BWD])
    one_col = prepare_triton_workload(
        TRITON_WORKLOAD_ADAPTERS[_FLASH_BWD_ONE_COL]
    )
    linear = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_LINEAR])
    group_norm = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_GROUP_NORM])
    layer_norm = prepare_triton_workload(
        TRITON_WORKLOAD_ADAPTERS[_FLASH_LAYER_NORM]
    )

    expected_flash = {
        "h0": 1,      # batch
        "h1": 128,    # active query sequence
        "h2": 1,      # heads
        "h3": 64,     # head dimension
        "h4": 128,    # active key sequence
        "h5": 128,    # rounded query capacity
    }
    assert flash_fwd.compiler_view.shape_coordinates == expected_flash
    assert flash_bwd.compiler_view.shape_coordinates == expected_flash
    for prepared in (flash_fwd, flash_bwd):
        scalars = {item.formal: item for item in prepared.adapter.scalars}
        assert {
            name: scalars[name].shape_coordinate
            for name in (
                "nheads",
                "seqlen_q",
                "seqlen_k",
                "seqlen_q_rounded",
                "headdim",
            )
        } == {
            "nheads": "h2",
            "seqlen_q": "h1",
            "seqlen_k": "h4",
            "seqlen_q_rounded": "h5",
            "headdim": "h3",
        }
    # Equal current values never collapse distinct Source-grounded identities.
    assert len({"h1", "h4", "h5"}) == 3

    assert one_col.compiler_view.shape_coordinates == {
        "h0": 128,
        "h1": 64,
        "h2": 64,
    }
    assert linear.compiler_view.shape_coordinates == {
        "h0": 32,
        "h1": 32,
        "h2": 32,
    }
    assert group_norm.compiler_view.shape_coordinates == {
        "h0": 2,
        "h1": 2,
        "h2": 32,
        "h3": 4,
    }
    assert group_norm.compiler_view.grid_shape_coordinates == ("h0", "h1")
    assert layer_norm.compiler_view.shape_coordinates == {"h0": 2, "h1": 127}
    assert layer_norm.compiler_view.grid_shape_coordinates == ("h0",)


def test_rope_family_exposes_only_source_grounded_parametric_identities() -> None:
    qwen = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_QWEN2VL_MROPE])
    rope = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_ROPE])
    llama = prepare_triton_workload(TRITON_WORKLOAD_ADAPTERS[_LLAMA4_ROPE])

    assert qwen.compiler_view.shape_coordinates == {
        "h0": 1,
        "h1": 4,
        "h2": 2,
        "h3": 16,
        "h4": 1,
    }
    assert qwen.compiler_view.bound_constexpr_shape_coordinates == {
        "bs": "h0",
        "n_qh": "h2",
        "n_kh": "h4",
        "hd": "h3",
    }
    assert {item.dtype for item in qwen.adapter.buffers} == {"f32"}
    assert {item.formal: item.shape_coordinate for item in qwen.adapter.scalars}[
        "sl"
    ] == "h1"

    assert rope.compiler_view.shape_coordinates == {
        "h0": 1,
        "h1": 4,
        "h2": 2,
        "h3": 16,
        "h4": 1,
        "h5": 1,
    }
    assert rope.compiler_view.bound_constexpr_shape_coordinates == {
        "bs": "h0",
        "cos_bs": "h5",
        "n_qh": "h2",
        "n_kh": "h4",
        "hd": "h3",
    }
    # Equal current values do not collapse tensor batch and broadcast batch.
    assert rope.compiler_view.shape_coordinates["h0"] == 1
    assert rope.compiler_view.shape_coordinates["h5"] == 1

    assert llama.compiler_view.shape_coordinates == {
        "h0": 1,
        "h1": 4,
        "h2": 2,
        "h3": 1,
        "h4": 16,
    }
    assert llama.compiler_view.bound_constexpr_shape_coordinates == {
        "head_dim_half": "h4",
        "n_q_heads": "h2",
        "n_k_heads": "h3",
    }
    llama_buffers = {item.formal: item for item in llama.adapter.buffers}
    assert {item.dtype for item in llama_buffers.values()} == {"f32"}
    assert llama_buffers["q_ptr"].shape_contract == ("h0", "h1", "h2", 32)
    assert llama_buffers["freqs_complex_ptr"].shape_contract == ("h1", "h4", 2)


def test_none_binding_and_vararg_arity_are_closed_adapter_facts() -> None:
    assert NoneBinding("optional").dtype == "none"
    with pytest.raises(ValueError, match="exactly 'none'"):
        NoneBinding("optional", "opaque")
    with pytest.raises(ValueError, match="vararg arity"):
        TritonWorkloadAdapter(
            "high_performance:test",
            buffers=(
                BufferBinding("Y", BufferRole.OUTPUT, "f32", (1,), "y"),
            ),
            scalars=(),
            semantic_constexprs=(),
            c_values=(),
            grid=(1,),
            grid_contract=("1",),
            seed=1,
            vararg_arity=0,
        )


@pytest.mark.parametrize("adapter", TRITON_WORKLOAD_ADAPTERS.values())
def test_real_adapter_preparation_validates_source_abi_and_anonymizes(adapter) -> None:
    prepared = prepare_triton_workload(adapter)
    request = prepared.request
    code, _origin = request.source.gate_source.read_code()

    assert prepared.entry.case_id == adapter.case_id
    assert prepared.source_location.startswith(("module://", "dataset://"))
    assert set(prepared.source_slice.parameters) <= set(
        prepared.source_to_opaque
    )
    assert "def scrc_k0(" in code
    assert prepared.entry.symbol not in code
    assert request.binding.values == {
        prepared.source_to_opaque[name]: value
        for name, value in adapter.c_values
    }
    assert set(request.binding.gate1_case.arguments) == {
        prepared.source_to_opaque[item.formal]
        for item in (
            *adapter.buffers,
            *adapter.scalars,
            *adapter.none_arguments,
        )
    }
    assert any(item.role is not BufferRole.INPUT for item in adapter.buffers)


def test_adapter_json_contains_only_harness_bindings_not_semantic_formulas() -> None:
    for adapter in TRITON_WORKLOAD_ADAPTERS.values():
        payload = adapter.to_json()
        encoded = str(payload).casefold()
        for forbidden in (
            "source_code",
            "replay",
            "operator",
            "predicate",
            "formula",
            "gate1_diff",
        ):
            assert forbidden not in encoded


@pytest.mark.parametrize(
    ("case_id", "parameter_count", "included_functions", "removed"),
    (
        (_FLASH_FWD, 38, ("_fwd_kernel",), ("triton.heuristics",)),
        (
            _FLASH_BWD,
            51,
            (
                "_bwd_store_dk_dv",
                "_bwd_kernel_one_col_block",
                "_bwd_kernel",
            ),
            ("triton.autotune", "triton.heuristics"),
        ),
    ),
)
def test_complex_flash_adapter_is_exact_fixed_source_projection(
    case_id, parameter_count, included_functions, removed
) -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)

    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.included_functions == included_functions
    assert prepared.source_slice.removed_decorators == removed
    assert adapter.grid == (1, 1)
    assert dict(adapter.semantic_constexprs) == {
        "BIAS_TYPE": "none",
        "IS_CAUSAL": False,
    }
    expected_c = {
        "BLOCK_HEADDIM": 64,
        "EVEN_M": True,
        "EVEN_N": True,
        "EVEN_HEADDIM": True,
        "BLOCK_M": 128,
        "BLOCK_N": 128,
    }
    if case_id == _FLASH_BWD:
        expected_c["SEQUENCE_PARALLEL"] = False
    assert dict(adapter.c_values) == expected_c
    scalars = {item.formal: item.value for item in adapter.scalars}
    assert scalars["softmax_scale"] == 0.125
    assert scalars["nheads"] == 1
    assert scalars["seqlen_q"] == scalars["seqlen_k"] == 128
    assert scalars["seqlen_q_rounded"] == 128
    assert scalars["headdim"] == 64
    assert scalars["CACHE_KEY_SEQLEN_Q"] == 4
    assert scalars["CACHE_KEY_SEQLEN_K"] == 4
    assert (scalars["stride_qb"], scalars["stride_qh"], scalars["stride_qm"]) == (
        8192,
        64,
        64,
    )
    roles = {item.formal: item.role for item in adapter.buffers}
    if case_id == _FLASH_FWD:
        assert roles["Out"] is BufferRole.OUTPUT
        assert roles["Lse"] is BufferRole.OUTPUT
        assert roles["TMP"] is BufferRole.INOUT
    else:
        assert roles["DQ"] is BufferRole.INOUT
        assert roles["DK"] is BufferRole.OUTPUT
        assert roles["DV"] is BufferRole.OUTPUT


def test_flash_backward_fixed_invocation_materializes_removed_zero_pre_hook() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_FLASH_BWD]
    buffers = {item.formal: item for item in adapter.buffers}

    assert buffers["DQ"].role is BufferRole.INOUT
    assert buffers["DQ"].initializer is BufferInitializer.ZEROS
    assert dict(adapter.c_values)["SEQUENCE_PARALLEL"] is False
    assert adapter.grid_contract == ("1", "batch*nheads")


@pytest.mark.parametrize(
    (
        "case_id",
        "parameter_count",
        "included_functions",
        "semantic_constexprs",
        "c_values",
        "grid",
    ),
    (
        (
            _FUSED_ADD_RMS,
            17,
            ("_fused_add_rms_norm_forward_kernel",),
            {"casting_mode": -1},
            {"BLOCK_SIZE": 128},
            (2,),
        ),
        (
            _SPARSEMAX,
            9,
            ("_sparsemax_forward_kernel",),
            {},
            {"BLOCK_SIZE": 128, "num_warps": 4},
            (2,),
        ),
        (
            _FLASH_BWD_ONE_COL,
            32,
            ("_bwd_store_dk_dv", "_bwd_kernel_one_col_block"),
            {"BIAS_TYPE": "none", "IS_CAUSAL": False},
            {
                "ATOMIC_ADD": False,
                "BLOCK_HEADDIM": 64,
                "EVEN_M": True,
                "EVEN_N": True,
                "EVEN_HEADDIM": True,
                "BLOCK_M": 64,
                "BLOCK_N": 64,
            },
            (1,),
        ),
    ),
)
def test_additional_fixed_c_adapters_match_exact_source_abi(
    case_id,
    parameter_count,
    included_functions,
    semantic_constexprs,
    c_values,
    grid,
) -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)

    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.included_functions == included_functions
    assert prepared.source_slice.removed_decorators == ()
    assert dict(adapter.semantic_constexprs) == semantic_constexprs
    assert dict(adapter.c_values) == c_values
    assert adapter.grid == grid


def test_one_col_fixed_invocation_matches_exact_pk_fixture() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_FLASH_BWD_ONE_COL]
    buffers = {item.formal: item for item in adapter.buffers}
    scalars = {item.formal: item.value for item in adapter.scalars}

    assert buffers["Q"].shape == (128, 64)
    assert buffers["K"].shape == buffers["V"].shape == (64, 64)
    assert buffers["DQ"].role is BufferRole.INOUT
    assert buffers["DQ"].initializer is BufferInitializer.ZEROS
    assert buffers["DK"].role is buffers["DV"].role is BufferRole.OUTPUT
    assert scalars["start_n"] == 0
    assert scalars["seqlen_q"] == 128
    assert scalars["seqlen_k"] == scalars["headdim"] == 64
    assert scalars["softmax_scale"] == 0.125


@pytest.mark.parametrize(
    ("case_id", "parameter_count", "semantic_constexprs", "c_values", "grid"),
    (
        (
            _QWEN2VL_MROPE,
            16,
            {
                "bs": 1,
                "n_qh": 2,
                "n_kh": 1,
                "hd": 16,
                "mrope_section_t": 3,
                "mrope_section_h": 3,
                "BACKWARD_PASS": False,
            },
            {
                "pad_n_qh": 2,
                "pad_n_kh": 1,
                "pad_hd": 16,
                "BLOCK_SIZE": 2,
            },
            (4,),
        ),
        (_GROUP_NORM, 18, {}, {"BLOCK_SIZE": 32}, (2, 2)),
        (
            _JSD,
            15,
            {"beta": 0.5, "ignore_index": -100, "HAS_LABEL": False},
            {"BLOCK_SIZE": 64},
            (2,),
        ),
        (
            _CROSS_ENTROPY,
            28,
            {
                "lse_square_scale": 0.0,
                "label_smoothing": 0.0,
                "reduction": "mean",
                "RETURN_Z_LOSS": False,
                "RETURN_TOKEN_ACCURACY": False,
                "RETURN_PREDICTED_TOKENS": False,
                "HAS_WEIGHT": False,
                "HAS_SOFTCAPPING": False,
                "HAS_GRADIENTS": False,
            },
            {"BLOCK_SIZE": 64},
            (2,),
        ),
    ),
)
def test_wave2_fixed_c_adapters_match_exact_source_abi(
    case_id, parameter_count, semantic_constexprs, c_values, grid
) -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)

    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.included_functions == (prepared.entry.symbol,)
    assert prepared.source_slice.removed_decorators == ()
    assert dict(adapter.semantic_constexprs) == semantic_constexprs
    assert dict(adapter.c_values) == c_values
    assert adapter.grid == grid


def test_wave2_inout_and_disabled_carriers_are_explicit() -> None:
    mrope = {
        item.formal: item for item in TRITON_WORKLOAD_ADAPTERS[_QWEN2VL_MROPE].buffers
    }
    cross_entropy = {
        item.formal: item
        for item in TRITON_WORKLOAD_ADAPTERS[_CROSS_ENTROPY].buffers
    }

    assert mrope["q_ptr"].role is mrope["k_ptr"].role is BufferRole.INOUT
    assert cross_entropy["Y_ptr"].initializer is BufferInitializer.ZEROS
    assert cross_entropy["X_ptr"].role is BufferRole.INOUT
    assert cross_entropy["X_ptr"].output_key == "x"
    assert cross_entropy["z_loss_ptr"].role is BufferRole.INPUT
    assert cross_entropy["token_accuracy_ptr"].role is BufferRole.INPUT
    assert cross_entropy["predicted_tokens_ptr"].role is BufferRole.INPUT


@pytest.mark.parametrize(
    (
        "case_id",
        "parameter_count",
        "removed_decorators",
        "semantic_constexprs",
        "c_values",
        "grid",
    ),
    (
        (
            _GEGLU,
            6,
            (),
            {"n_cols": 127},
            {"BLOCK_SIZE": 128},
            (2,),
        ),
        (
            _POLY_NORM,
            11,
            (),
            {},
            {"BLOCK_SIZE": 128},
            (2,),
        ),
        (
            _LLAMA4_ROPE,
            15,
            (),
            {"head_dim_half": 16, "n_q_heads": 2, "n_k_heads": 1},
            {"BLOCK_SIZE": 16},
            (4, 2),
        ),
        (
            _FLASH_LAYER_NORM,
            38,
            ("triton.autotune",),
            {
                "IS_RMS_NORM": False,
                "HAS_RESIDUAL": True,
                "STORE_RESIDUAL_OUT": True,
                "HAS_BIAS": True,
                "HAS_DROPOUT": False,
                "STORE_DROPOUT_MASK": False,
                "HAS_ROWSCALE": False,
                "HAS_X1": True,
                "HAS_W1": True,
                "HAS_B1": True,
            },
            {"BLOCK_N": 128},
            (2,),
        ),
    ),
)
def test_wave3_fixed_c_adapters_match_exact_source_abi(
    case_id,
    parameter_count,
    removed_decorators,
    semantic_constexprs,
    c_values,
    grid,
) -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)

    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.included_functions == (prepared.entry.symbol,)
    assert prepared.source_slice.removed_decorators == removed_decorators
    assert dict(adapter.semantic_constexprs) == semantic_constexprs
    assert dict(adapter.c_values) == c_values
    assert adapter.grid == grid


def test_wave3_inout_and_optional_branch_roles_are_explicit() -> None:
    rope = {
        item.formal: item for item in TRITON_WORKLOAD_ADAPTERS[_LLAMA4_ROPE].buffers
    }
    layer_norm = {
        item.formal: item
        for item in TRITON_WORKLOAD_ADAPTERS[_FLASH_LAYER_NORM].buffers
    }

    assert rope["q_ptr"].role is rope["k_ptr"].role is BufferRole.INOUT
    assert layer_norm["Y"].role is BufferRole.OUTPUT
    assert layer_norm["Y1"].role is BufferRole.OUTPUT
    assert layer_norm["RESIDUAL_OUT"].role is BufferRole.OUTPUT
    assert layer_norm["Mean"].role is layer_norm["Rstd"].role is BufferRole.OUTPUT
    assert layer_norm["ROWSCALE"].role is BufferRole.INPUT
    assert layer_norm["SEEDS"].role is BufferRole.INPUT
    assert layer_norm["DROPOUT_MASK"].role is BufferRole.INPUT
    assert layer_norm["DROPOUT_MASK1"].role is BufferRole.INPUT


@pytest.mark.parametrize(
    ("case_id", "parameter_count", "semantic_constexprs", "c_values", "grid"),
    (
        (
            _ROPE,
            19,
            {
                "bs": 1,
                "cos_bs": 1,
                "n_qh": 2,
                "n_kh": 1,
                "hd": 16,
                "BACKWARD_PASS": False,
            },
            {
                "pad_n_qh": 2,
                "pad_n_kh": 1,
                "pad_hd": 16,
                "BLOCK_SIZE": 2,
            },
            (4,),
        ),
        (_LAYER_NORM, 15, {}, {"BLOCK_SIZE": 128}, (2,)),
        (
            _KL_DIV,
            11,
            {"log_target": True, "reduction": 3},
            {"BLOCK_SIZE": 64},
            (2,),
        ),
        (
            _TVD,
            15,
            {"ignore_index": -100, "HAS_LABEL": False, "reduction": 0},
            {"BLOCK_SIZE": 64},
            (2,),
        ),
    ),
)
def test_wave4_liger_adapters_match_exact_source_abi(
    case_id, parameter_count, semantic_constexprs, c_values, grid
) -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)

    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.included_functions == (prepared.entry.symbol,)
    assert prepared.source_slice.removed_decorators == ()
    assert dict(adapter.semantic_constexprs) == semantic_constexprs
    assert dict(adapter.c_values) == c_values
    assert adapter.grid == grid


def test_wave4_output_roles_and_safe_kl_branch_are_explicit() -> None:
    rope = {item.formal: item for item in TRITON_WORKLOAD_ADAPTERS[_ROPE].buffers}
    rope_scalars = {
        item.formal: item for item in TRITON_WORKLOAD_ADAPTERS[_ROPE].scalars
    }
    kl_div = TRITON_WORKLOAD_ADAPTERS[_KL_DIV]
    tvd = {item.formal: item for item in TRITON_WORKLOAD_ADAPTERS[_TVD].buffers}

    assert rope["q_ptr"].role is rope["k_ptr"].role is BufferRole.INOUT
    assert {item.dtype for item in rope.values()} == {"f32"}
    assert {
        rope_scalars[name].dtype
        for name in (
            "q_row_stride",
            "k_row_stride",
            "cos_row_stride",
            "sin_row_stride",
        )
    } == {"i32"}
    assert dict(kl_div.semantic_constexprs)["log_target"] is True
    assert tvd["loss_ptr"].shape == tvd["grads_ptr"].shape == (2, 64)
    assert tvd["label_ptr"].role is BufferRole.INPUT


@pytest.mark.parametrize(
    ("case_id", "parameter_count", "inout"),
    ((_XFORMERS_RMS, 8, False), (_XFORMERS_RMS_ADD, 9, True)),
)
def test_xformers_rms_plain_resolver_closure_and_exact_abi(
    case_id, parameter_count, inout
) -> None:
    manifest = build_workload_manifest()
    entry = manifest.by_id[case_id]
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)
    anonymous, _origin = prepared.request.source.gate_source.read_code()

    assert entry.adapter_readiness == "declared"
    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.protocol == (
        "intentir.triton.fixed-c-source-slice.v3"
    )
    assert prepared.source_slice.included_functions == (
        "libdevice_find",
        entry.symbol,
    )
    assert prepared.source_slice.inlined_imports == (
        "xformers.triton.importing:libdevice_find",
    )
    assert "from xformers" not in prepared.source_slice.code
    assert "import importlib.util" in prepared.source_slice.code
    assert "def libdevice_find" in prepared.source_slice.code
    assert "from xformers" not in anonymous
    assert "scrc_k1('rsqrt')" in anonymous
    assert dict(adapter.semantic_constexprs) == {
        "N_COLS": 127,
        "INCLUDE_WEIGHT": True,
    }
    assert dict(adapter.c_values) == {"BLOCK_SIZE": 128}
    roles = {item.formal: item.role for item in adapter.buffers}
    assert (roles["x_ptr"] is BufferRole.INOUT) is inout


@pytest.mark.parametrize(
    ("case_id", "parameter_count", "arity", "none_count"),
    (
        (_XFORMERS_SPLITK, 82, 1, 8),
        (_XFORMERS_SPLITK_REDUCE, 38, 2, 0),
        (_XFORMERS_SPLITK_REDUCE_BWD, 51, 2, 0),
    ),
)
def test_splitk_closed_vararg_and_none_abi_is_exact(
    case_id, parameter_count, arity, none_count
) -> None:
    entry = build_workload_manifest().by_id[case_id]
    adapter = TRITON_WORKLOAD_ADAPTERS[case_id]
    prepared = prepare_triton_workload(adapter)

    assert entry.adapter_readiness == "declared"
    assert len(prepared.source_slice.parameters) == parameter_count
    assert prepared.source_slice.protocol == (
        "intentir.triton.fixed-c-source-slice.v3"
    )
    assert prepared.source_slice.vararg_arity == arity
    assert not requires_triton_vararg_expansion(prepared.source_slice)
    assert adapter.vararg_arity == arity
    assert len(adapter.none_arguments) == none_count
    assert sum(
        item.kind == "none" for item in prepared.compiler_view.arguments
    ) == none_count


def test_splitk_forward_uses_real_none_and_non_degenerate_split_grid() -> None:
    adapter = TRITON_WORKLOAD_ADAPTERS[_XFORMERS_SPLITK]
    prepared = prepare_triton_workload(adapter)

    assert adapter.grid == (1, 1, 2)
    assert dict(adapter.semantic_constexprs)["N_GROUPS"] == 1
    assert dict(adapter.semantic_constexprs)["IS_SPLITK"] is True
    assert dict(adapter.c_values)["NUM_PROGRAMS_DIM2_CONST"] == 2
    assert {item.formal for item in adapter.none_arguments} == {
        "block_tables",
        "Seq_len",
        "Seq_starts_k",
        "Seq_starts_q",
        "Seq_starts_q_multiplier",
        "additive_bias",
        "K_fp8_scale_shift",
        "V_fp8_scale_shift",
    }
    runtime = prepared.request.binding.gate1_case.arguments
    assert all(
        runtime[prepared.source_to_opaque[item.formal]].kind == "none"
        for item in adapter.none_arguments
    )


def test_splitk_varargs_merged_buffers_follow_source_bghmk_axis_order() -> None:
    reduce_buffers = {
        item.formal: item
        for item in TRITON_WORKLOAD_ADAPTERS[_XFORMERS_SPLITK_REDUCE].buffers
    }
    backward_buffers = {
        item.formal: item
        for item in TRITON_WORKLOAD_ADAPTERS[_XFORMERS_SPLITK_REDUCE_BWD].buffers
    }

    # Source contract: merged attention tensors are [B, G, H, M, K].
    assert reduce_buffers["Out"].shape == (1, 1, 1, 2, 16)
    assert backward_buffers["Out"].shape == (1, 1, 1, 2, 16)
    assert backward_buffers["DOut"].shape == (1, 1, 1, 2, 16)


@pytest.mark.parametrize("adapter", TRITON_WORKLOAD_ADAPTERS.values())
def test_real_adapter_source_reaches_triton_frontend(adapter) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires Triton CUDA")
    prepared = prepare_triton_workload(adapter)

    result = TritonCompiler(timeout=240).compile(
        prepared.request.source.gate_source,
        prepared.request.binding.gate1_case,
    )

    assert result.ok, result.diagnostic
    assert result.ttir
