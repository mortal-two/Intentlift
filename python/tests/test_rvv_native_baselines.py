from __future__ import annotations

from intentir.backend.riscv.native_baselines import (
    EXTERNAL_RVV_BASELINE_PROTOCOL,
    NATIVE_RVV_BASELINE_PROTOCOL,
    NATIVE_RVV_SYMBOL,
    external_rvv_baseline_records,
    native_rvv_baseline_record,
)
from intentir.backend.riscv.ggml_baselines import (
    ggml_rvv_baseline_record,
    maybe_ggml_rvv_baseline_records,
)
from intentir.backend.riscv.triton_cpu_baselines import (
    swiglu_triton_cpu_rvv_record,
)


FRESH14 = {
    "dyt",
    "flash_bwd_preprocess",
    "flash_layer_norm_1pass",
    "fused_add_rms_norm",
    "geglu",
    "kl_div",
    "layer_norm",
    "liger_rms_norm",
    "poly_norm",
    "softmax",
    "sparsemax",
    "splitk_reduce_varargs",
    "swiglu",
    "xformers_rms_norm_add",
}

STAGE2 = {
    "cross_entropy",
    "flash_linear",
    "group_norm",
    "jsd",
    "liger_rope",
    "qwen2vl_mrope",
    "tvd",
    "xformers_rms_norm",
}


def test_fresh14_has_independent_native_rvv_sources() -> None:
    for kernel in sorted(FRESH14 | STAGE2):
        record = native_rvv_baseline_record(kernel)
        assert record["protocol"] == NATIVE_RVV_BASELINE_PROTOCOL
        assert record["symbol"] == NATIVE_RVV_SYMBOL
        assert f"void {NATIVE_RVV_SYMBOL}(" in record["source"]
        assert "#include <riscv_vector.h>" in record["source"]
        assert "intentir.backend" not in record["source"]
        assert record["implementation"] == (
            "independent-case-authored-c-llvm-rvv-autovectorized"
        )


def test_external_baselines_distinguish_portable_c_from_explicit_intrinsics() -> None:
    portable, expert = external_rvv_baseline_records("gather2d")

    assert portable["protocol"] == EXTERNAL_RVV_BASELINE_PROTOCOL
    assert portable["baseline_id"] == "llvm_rvv_c"
    assert portable["kind"] == "portable_compiler"
    assert "__riscv_vluxei32" not in portable["source"]
    assert expert["baseline_id"] == "expert_rvv"
    assert expert["kind"] == "expert_intrinsic"
    assert "__riscv_vsetvl_e32m1" in expert["source"]
    assert "__riscv_vluxei32_v_f32m1" in expert["source"]


def test_external_baseline_subset_keeps_full_llvm_rvv_coverage() -> None:
    records = external_rvv_baseline_records("softmax")

    assert [record["baseline_id"] for record in records] == ["llvm_rvv_c"]


def test_expert_tvd_fuses_both_outputs_in_one_rvv_traversal() -> None:
    portable, expert = external_rvv_baseline_records("tvd")

    assert portable["baseline_id"] == "llvm_rvv_c"
    assert "__riscv_vmflt_vv_f32m1_b32" in expert["source"]
    assert "__riscv_vmerge_vvm_f32m1" in expert["source"]
    assert expert["source"].count("while(k<total)") == 1
    assert "for(k=0;k<total" not in expert["source"]


def test_expert_reduction_baselines_use_dynamic_vl_and_widening() -> None:
    _portable, layer_norm = external_rvv_baseline_records("layer_norm")
    _portable, flash = external_rvv_baseline_records("flash_bwd_preprocess")

    assert layer_norm["source"].count("__riscv_vfredosum") == 2
    assert "__riscv_vsetvl_e32m1" in layer_norm["source"]
    assert "__riscv_vfwcvt_f_f_v_f32m2" in flash["source"]
    assert "__riscv_vfredosum_vs_f32m2_f32m1" in flash["source"]


def test_ggml_baseline_exists_only_for_strict_graph_boundaries() -> None:
    softmax = ggml_rvv_baseline_record("softmax")

    assert softmax["kind"] == "mature_library"
    assert softmax["display_name"] == "ggml CPU/RVV"
    assert softmax["build_profile"] == "ggml-rvv-strict"
    assert "ggml_soft_max" in softmax["source"]
    assert "ggml_graph_plan" in softmax["source"]
    assert maybe_ggml_rvv_baseline_records("gather2d") == ()


def test_ggml_elementwise_graphs_preserve_the_public_j_boundary() -> None:
    dyt = ggml_rvv_baseline_record("dyt")
    swiglu = ggml_rvv_baseline_record("swiglu")

    assert "ggml_tanh" in dyt["source"]
    assert "M1 *alpha_d, M1 *gamma_d, M1 *beta_d" in dyt["source"]
    assert "ggml_silu" in swiglu["source"]
    assert "M0 *multiplier_d" in swiglu["source"]
    assert all(
        record["strict_match"] is True
        for record in (dyt, swiglu)
    )
    assert maybe_ggml_rvv_baseline_records("geglu") == ()


def test_triton_cpu_swiglu_adapter_only_bridges_the_raw_kernel_abi() -> None:
    record = swiglu_triton_cpu_rvv_record(
        "/tmp/triton-swiglu.so",
        columns=11008,
        source_identity="accepted/source.py::_swiglu_forward_kernel",
    )

    assert record["baseline_id"] == "triton_cpu_rvv"
    assert record["kind"] == "source_portability"
    assert record["build_profile"] == "triton-cpu-rvv-prebuilt-strict"
    assert record["prebuilt_library"] == "/tmp/triton-swiglu.so"
    assert "extern void _swiglu_forward_kernel" in record["source"]
    assert "expf(" not in record["source"]
    assert "tanhf(" not in record["source"]
    assert "cols != 11008" in record["source"]
