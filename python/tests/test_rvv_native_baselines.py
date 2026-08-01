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
    dyt_triton_cpu_rvv_record,
    flash_preprocess_triton_cpu_rvv_record,
    geglu_triton_cpu_rvv_record,
    kldiv_triton_cpu_rvv_record,
    layer_norm_triton_cpu_rvv_record,
    rms_norm_triton_cpu_rvv_record,
    softmax_triton_cpu_rvv_record,
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
    portable, expert = external_rvv_baseline_records("tvd")

    assert portable["protocol"] == EXTERNAL_RVV_BASELINE_PROTOCOL
    assert portable["baseline_id"] == "llvm_rvv_c"
    assert portable["kind"] == "portable_compiler"
    assert "__riscv_vmflt_vv_f32m1_b32" not in portable["source"]
    assert expert["baseline_id"] == "expert_rvv"
    assert expert["kind"] == "expert_intrinsic"
    assert "__riscv_vsetvl_e32m1" in expert["source"]
    assert "__riscv_vmflt_vv_f32m1_b32" in expert["source"]


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
    assert maybe_ggml_rvv_baseline_records("kl_div") == ()


def test_ggml_elementwise_graphs_preserve_the_public_j_boundary() -> None:
    dyt = ggml_rvv_baseline_record("dyt")
    geglu = ggml_rvv_baseline_record("geglu")
    swiglu = ggml_rvv_baseline_record("swiglu")

    assert "ggml_tanh" in dyt["source"]
    assert "M1 *alpha_d, M1 *gamma_d, M1 *beta_d" in dyt["source"]
    assert "ggml_silu" in swiglu["source"]
    assert "M0 *multiplier_d" in swiglu["source"]
    assert "ggml_sqr(context, a_tensor)" in geglu["source"]
    assert "ggml_tanh(context, tanh_arg)" in geglu["source"]
    assert "ggml_geglu_split" not in geglu["source"]
    assert all(
        record["strict_match"] is True
        for record in (dyt, geglu, swiglu)
    )
    assert maybe_ggml_rvv_baseline_records("kl_div") == ()


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


def test_triton_cpu_reduction_adapters_do_not_reimplement_the_math() -> None:
    softmax = softmax_triton_cpu_rvv_record(
        "/tmp/softmax.so", columns=4096, source_identity="softmax.py::kernel"
    )
    kldiv = kldiv_triton_cpu_rvv_record(
        "/tmp/kldiv.so",
        columns=4096,
        epsilon=1.0e-6,
        source_identity="kldiv.py::kernel",
    )
    flash = flash_preprocess_triton_cpu_rvv_record(
        "/tmp/flash.so",
        block_m=128,
        block_head_dim=128,
        source_identity="flash.py::kernel",
    )

    assert "extern void _softmax_single_block_forward_kernel" in softmax["source"]
    assert "extern void _kldiv_kernel_forward" in kldiv["source"]
    assert "extern void _bwd_preprocess_do_o_dot" in flash["source"]
    for record in (softmax, kldiv, flash):
        assert record["baseline_id"] == "triton_cpu_rvv"
        assert "expf(" not in record["source"]
        assert "logf(" not in record["source"]
        assert record["specialization"]


def test_triton_cpu_extended_adapters_preserve_all_public_outputs() -> None:
    geglu = geglu_triton_cpu_rvv_record(
        "/tmp/geglu.so", columns=11008, source_identity="geglu.py::kernel"
    )
    dyt = dyt_triton_cpu_rvv_record(
        "/tmp/dyt.so",
        columns=4096,
        block_size=128,
        source_identity="dyt.py::kernel",
    )
    rms = rms_norm_triton_cpu_rvv_record(
        "/tmp/rms.so", columns=4096, source_identity="rms.py::kernel"
    )
    layer = layer_norm_triton_cpu_rvv_record(
        "/tmp/layer.so", columns=4096, source_identity="layer.py::kernel"
    )

    assert "extern void _geglu_tanh_forward_kernel" in geglu["source"]
    assert "extern void _dyt_fwd_kernel" in dyt["source"]
    assert "M2 *y_d, M1 *rstd_d" in rms["source"]
    assert "M2 *y_d, M1 *mean_d, M1 *rstd_d" in layer["source"]
    for record in (geglu, dyt, rms, layer):
        assert record["baseline_id"] == "triton_cpu_rvv"
        assert "sqrtf(" not in record["source"]
        assert "tanhf(" not in record["source"]
