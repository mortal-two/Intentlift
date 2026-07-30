from __future__ import annotations

from intentir.backend.riscv.native_baselines import (
    NATIVE_RVV_BASELINE_PROTOCOL,
    NATIVE_RVV_SYMBOL,
    native_rvv_baseline_record,
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
        assert record["implementation"] == "independent-case-authored-c-rvv-openmp"
