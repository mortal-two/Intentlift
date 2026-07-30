  ## 已通过的 22 个 Kernel

     #    Kernel                            正式证据 JSON
  ━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     1    SwiGLU                            /tmp/intentir-task14-wave1-ready-b-20260728/liger_swiglu/fanout-001/candidate-03/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     2    RMSNorm                           /tmp/intentir-task14-wave1-ready-a-20260728/liger_rms_norm/fanout-001/candidate-01/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     3    Fused Add RMSNorm                 /tmp/intentir-task14-wave1-ready-b-20260728/liger_fused_add_rms_norm/fanout-001/candidate-01/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     4    GeGLU                             /tmp/intentir-task14-wave1-ready-b-20260728/liger_geglu/fanout-001/candidate-03/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     5    LayerNorm                         /tmp/intentir-task14-wave1-ready-a-20260728/liger_layer_norm/fanout-001/candidate-01/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     6    Softmax                           /tmp/intentir-task14-wave2-ord08-14-terra-high-20260728/liger_softmax/fanout-001/candidate-03/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     7    DYT                               /tmp/intentir-task14-wave2-ord08-14-terra-high-20260728/liger_dyt/fanout-001/candidate-01/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     8    Sparsemax                         /tmp/intentir-task14-wave2-ord08-14-terra-high-20260728/liger_sparsemax/fanout-001/candidate-02/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     9    KLDiv                             /tmp/intentir-task14-wave2-ord08-14-terra-high-20260728/liger_kl_div/fanout-001/candidate-03/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    10    PolyNorm                          /tmp/intentir-task14-wave3-ord15-21-terra-high-20260728/liger_poly_norm/fanout-001/candidate-01/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    11    JSD                               /tmp/intentir-task14-wave7-remaining14-codex-sol-high-20260728/liger_jsd/fanout-001/candidate-03/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    12    TVD                               /tmp/intentir-task14-wave7-remaining14-codex-sol-high-20260728/liger_tvd/fanout-001/candidate-02/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    13    xFormers RMSNorm                  /tmp/intentir-task14-wave7-remaining14-codex-sol-high-20260728/xformers_ops_triton_rmsnorm_kernels_rms_norm_kernel/
                                            fanout-001/candidate-02/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    14    xFormers RMSNorm-add              /tmp/scrc_task11_v14_rmsnorm_add_canonical_consumers_recheck_20260725_01/
                                            xformers_ops_triton_rmsnorm_kernels_rms_norm_add_kernel/gate2-recheck-001/recheck.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    15    Flash backward preprocess         /tmp/intentir-task14-wave4-ord22-28-terra-high-20260728/flash_attn_flash_attn_triton_bwd_preprocess_do_o_dot/fanout-
                                            001/candidate-01/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    16    CrossEntropy                      /tmp/intentir-task14-wave31-final-consumer-rechecks-20260728/liger_cross_entropy/gate2-recheck-001/recheck.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    17    GroupNorm                         /tmp/intentir-task14-wave31-final-consumer-rechecks-20260728/liger_group_norm/gate2-recheck-001/recheck.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    18    Flash one-pass LayerNorm          /tmp/scrc_task11_v73_layernorm_gate2_live_input_fix_20260725_01/
                                            flash_attn_ops_triton_layer_norm_layer_norm_fwd_1pass_kernel/gate2-recheck-001/recheck.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    19    Liger RoPE                        /tmp/intentir-task15-semantic-probe-qwen-liger-v29-sol-xhigh-b3k5-20260730-01/liger_rope/fanout-001/candidate-01/
                                            report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    20    Qwen2VL MRoPE                     /tmp/intentir-task15-qwen-contract-indexed-gate2-recheck-20260730-01/liger_qwen2vl_mrope/gate2-recheck-001/
                                            recheck.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    21    Flash Linear                      /tmp/intentir-task15-v12-owned-realization-linear-splitkfwd-sol-high-b3k5-20260730-01/
                                            flash_attn_ops_triton_linear_kernel_fwd/fanout-001/candidate-02/report.json
  ─────  ────────────────────────────────  ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    22    SplitK reduce，split=4/arity=4    /tmp/intentir-task15-v12-splitk-reduce-exact-current-gate1-gate2-20260730-01/
                                            xformers_ops_fmha_triton_splitk_kernels_splitk_reduce_varargs/gate2-recheck-001/recheck.json

  Linear 有一份容易混淆的旧 artifact recheck，它是 gate2_failed，且明确写着 changes_parent_verdict=false，不能拿它代替上表的正式 winner：/tmp/intentir-task15-
  v12-linear-exact-current-gate1-gate2-20260730-01/flash_attn_ops_triton_linear_kernel_fwd/gate2-recheck-001/recheck.json。
