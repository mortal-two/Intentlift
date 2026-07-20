"""Run the first real held-out SCRC batch without the legacy corpus verifier.

Every ready case points at an installed, unmodified FlagGems Triton source
function and is executed through the shared live Lift -> Gate1 -> conditional
Gate2 path.  The larger manifest is the reviewed 20--40-kernel queue; cases are
promoted to ``ready`` only after their concrete compiler/execution contract is
recorded here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from intentir.extraction.lifters.scrc import (
    SCRC_RELIFT_PROTOCOL,
    SCRCCompilerArgument,
    scrc_lift_prompt_sha256,
)
from intentir.verify.gate1.compiler import TritonCompiler
from intentir.verify.gate1.compare import TTIR_SCORE_SEMANTICS
from intentir.verify.gate1.composer import COMPOSER_VERSION
from intentir.verify.gate1.model import Gate1Artifact
from intentir.verify.gate1.orchestrate import (
    GATE1_FAMILY_ANTI_LEAKAGE_POLICY,
    GATE1_STAGNATION_POLICY,
    GATE1_TERMINATION_POLICY,
    MASKED_J_FAMILY_POLICY,
    MASKED_J_POLICY,
    RELIFT_GUIDANCE_POLICY,
    RELIFT_STRUCTURAL_GUIDANCE_POLICY,
)
from intentir.verify.gate1.interventions import (
    PAIRED_INTERVENTION_FAMILY_POLICY,
)
from intentir.verify.gate1.specialization import (
    GATE1_FAMILY_RELIFT_GUIDANCE_POLICY,
)
from intentir.verify.gate1.rule_release import load_rule_release
from intentir.verify.gate1.rules import default_triton_rule_registry
from intentir.verify.gate1.ttir import TTIR_FINGERPRINT_SCHEMA
from intentir.verify.gate2 import (
    GATE2_COMPARISON_PROTOCOL,
    GATE2_EXECUTION_PROTOCOL,
    InputDistribution,
    Tolerance,
    boundary_1d_extents,
    extract_gate1_gate2_observation,
    summarize_gate1_gate2,
)

from scrc_live_e2e import (
    Gate2ShapeProfile,
    Gate2ValueProfile,
    LiveKernelCase,
    _compile_case,
    _gate1_specialization_policy_json,
    _gate2_matrix_policy_json,
    _positive_int,
    _selected_gate2_seeds,
    _write_report,
    run_case,
)


@dataclass(frozen=True)
class HeldoutCandidate:
    name: str
    module: str
    function: str
    family: str
    readiness: str = "queued"


CANDIDATES = (
    HeldoutCandidate(
        "fg_dot", "flag_gems.ops.dot", "dot_kernel", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_addr",
        "flag_gems.ops.addr",
        "addr_kernel",
        "broadcast",
        "ready",
    ),
    HeldoutCandidate(
        "fg_ones", "flag_gems.ops.ones", "ones_kernel", "fill", "ready"
    ),
    HeldoutCandidate(
        "fg_zeros", "flag_gems.ops.zeros", "zeros_kernel", "fill", "ready"
    ),
    HeldoutCandidate(
        "fg_arange",
        "flag_gems.ops.arange",
        "arange_func",
        "iota",
        "ready",
    ),
    HeldoutCandidate(
        "fg_eye", "flag_gems.ops.eye_m", "eye_kernel", "select", "ready"
    ),
    HeldoutCandidate(
        "fg_trace",
        "flag_gems.ops.trace",
        "trace_kernel",
        "gather_reduce",
        "ready",
    ),
    HeldoutCandidate(
        "fg_embedding",
        "flag_gems.ops.embedding",
        "embedding_kernel",
        "gather",
        "ready",
    ),
    HeldoutCandidate(
        "fg_index_select",
        "flag_gems.ops.index_select",
        "index_select_kernel",
        "indirect_indexed_gather",
        "ready",
    ),
    HeldoutCandidate(
        "fg_diag_extract",
        "flag_gems.ops.diag",
        "diag_2d_to_1d_kernel",
        "strided_gather",
        "ready",
    ),
    HeldoutCandidate(
        "fg_count_nonzero",
        "flag_gems.ops.count_nonzero",
        "count_nonzero_kernel_1",
        "atomic_reduce",
    ),
    HeldoutCandidate(
        "fg_amax", "flag_gems.ops.amax", "amax_kernel_2", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_sum", "flag_gems.ops.sum", "sum_kernel_2", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_mean", "flag_gems.ops.mean", "mean_kernel_2", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_max", "flag_gems.ops.max", "max_kernel_2", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_min", "flag_gems.ops.min", "min_kernel_2", "reduce", "ready"
    ),
    HeldoutCandidate(
        "fg_prod",
        "flag_gems.ops.prod",
        "prod_kernel_result",
        "reduce",
        "ready",
    ),
    HeldoutCandidate(
        "fg_vector_l2",
        "flag_gems.ops.vector_norm",
        "l2_norm_kernel_2",
        "reduce",
        "ready",
    ),
    HeldoutCandidate(
        "fg_vector_l1",
        "flag_gems.ops.vector_norm",
        "l1_norm_kernel_1",
        "reduce",
        "ready",
    ),
    HeldoutCandidate("fg_sort", "flag_gems.ops.sort", "sort_kernel", "sort"),
    HeldoutCandidate(
        "fg_log_softmax",
        "flag_gems.ops.log_softmax",
        "log_softmax_kernel",
        "structured_loop",
        "ready",
    ),
    HeldoutCandidate(
        "fg_fused_add_rms_norm",
        "flag_gems.fused.fused_add_rms_norm",
        "fused_add_rms_norm_kernel",
        "norm",
    ),
    HeldoutCandidate(
        "fg_skip_layer_norm",
        "flag_gems.fused.skip_layernorm",
        "skip_layer_norm_kernel",
        "norm",
        "ready",
    ),
    HeldoutCandidate(
        "fg_masked_select",
        "flag_gems.ops.masked_select",
        "masked_select_single_pass_kernel",
        "scan",
    ),
    HeldoutCandidate(
        "fg_topk_stage1",
        "flag_gems.ops.topk",
        "topk_stage1_kernel",
        "topk",
    ),
    HeldoutCandidate(
        "fg_rotary_embedding",
        "flag_gems.fused.rotary_embedding",
        "apply_rotary_pos_emb_kernel",
        "indexed_elementwise",
    ),
    HeldoutCandidate(
        "fg_cumsum_part",
        "flag_gems.ops.cumsum",
        "scan_part_sum_kernel",
        "scan",
    ),
)

_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}
READY_CASES = tuple(
    candidate.name for candidate in CANDIDATES if candidate.readiness == "ready"
)


def _function(candidate: HeldoutCandidate) -> Any:
    module = importlib.import_module(candidate.module)
    return getattr(module, candidate.function)


def build_heldout_case(name: str) -> LiveKernelCase:
    candidate = _BY_NAME[name]
    if candidate.readiness != "ready":
        raise ValueError(f"held-out case {name!r} has no reviewed compile contract")

    if name == "fg_dot":
        n = 1003
        block = 1024
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument("x_ptr", "tensor", "f32", (n,)),
                SCRCCompilerArgument("y_ptr", "tensor", "f32", (n,)),
                SCRCCompilerArgument("out_ptr", "tensor", "f32", ()),
                SCRCCompilerArgument("N", "scalar", "i32", value=n),
            ),
            (1,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1",),
            ("BLOCK_SIZE",),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )

    if name == "fg_addr":
        m, n = 17, 13
        block_m = block_n = 32
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument("input_ptr", "tensor", "f32", (m, n)),
                SCRCCompilerArgument("vec1_ptr", "tensor", "f32", (m,)),
                SCRCCompilerArgument("vec2_ptr", "tensor", "f32", (n,)),
                SCRCCompilerArgument("output_ptr", "tensor", "f32", (m, n)),
                SCRCCompilerArgument("beta", "scalar", "f32", value=0.75),
                SCRCCompilerArgument("alpha", "scalar", "f32", value=-1.25),
                SCRCCompilerArgument("M", "scalar", "i32", value=m),
                SCRCCompilerArgument("N", "scalar", "i32", value=n),
                SCRCCompilerArgument("stride_input_m", "scalar", "i32", value=n),
                SCRCCompilerArgument("stride_input_n", "scalar", "i32", value=1),
                SCRCCompilerArgument("stride_vec1", "scalar", "i32", value=1),
                SCRCCompilerArgument("stride_vec2", "scalar", "i32", value=1),
                SCRCCompilerArgument("stride_output_m", "scalar", "i32", value=n),
                SCRCCompilerArgument("stride_output_n", "scalar", "i32", value=1),
            ),
            (1, 1),
            {
                "BLOCK_SIZE_M": block_m,
                "BLOCK_SIZE_N": block_n,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            (
                "ceil_div(M, BLOCK_SIZE_M)",
                "ceil_div(N, BLOCK_SIZE_N)",
            ),
            ("BLOCK_SIZE_M", "BLOCK_SIZE_N"),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )

    if name == "fg_ones":
        n = 2051
        block = 1024
        boundary_profiles = tuple(
            Gate2ShapeProfile(
                boundary.label,
                {"output_ptr": (boundary.extent,)},
                {"n_elements": boundary.extent},
                ((boundary.extent + block - 1) // block,),
            )
            for boundary in boundary_1d_extents(block)
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "output_ptr", "tensor", "f32", (n,)
                ),
                SCRCCompilerArgument(
                    "n_elements", "scalar", "i32", value=n
                ),
            ),
            ((n + block - 1) // block,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(n_elements, BLOCK_SIZE)",),
            ("BLOCK_SIZE",),
            1.0,
            Tolerance(rtol=0.0, atol=0.0),
            gate2_shape_profiles=boundary_profiles,
            gate1_specialization_profiles=boundary_profiles,
            gate1_primary_profile_id="non_divisible",
        )

    if name == "fg_zeros":
        n = 2051
        block = 1024
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "output_ptr", "tensor", "f32", (n,)
                ),
                SCRCCompilerArgument(
                    "n_elements", "scalar", "i32", value=n
                ),
            ),
            ((n + block - 1) // block,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(n_elements, BLOCK_SIZE)",),
            ("BLOCK_SIZE",),
            1.0,
            Tolerance(rtol=0.0, atol=0.0),
            gate2_shape_profiles=tuple(
                Gate2ShapeProfile(
                    boundary.label,
                    {"output_ptr": (boundary.extent,)},
                    {"n_elements": boundary.extent},
                    ((boundary.extent + block - 1) // block,),
                )
                for boundary in boundary_1d_extents(block)
            ),
        )

    if name == "fg_arange":
        size = 1003
        block = 128
        start = -3.25
        step = 0.125
        end = start + step * size
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "y_ptr", "tensor", "f32", (size,)
                ),
                SCRCCompilerArgument(
                    "start", "scalar", "f32", value=start
                ),
                SCRCCompilerArgument(
                    "end", "scalar", "f32", value=end
                ),
                SCRCCompilerArgument(
                    "step", "scalar", "f32", value=step
                ),
                SCRCCompilerArgument(
                    "size", "scalar", "i32", value=size
                ),
            ),
            ((size + block - 1) // block,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(size, BLOCK_SIZE)",),
            ("BLOCK_SIZE",),
            1.0,
            Tolerance(rtol=1.0e-6, atol=1.0e-6),
            gate2_shape_profiles=(
                Gate2ShapeProfile(
                    "forward_fractional",
                    {},
                    {
                        "start": start,
                        "end": end,
                        "step": step,
                        "size": size,
                    },
                ),
                Gate2ShapeProfile(
                    "descending_fractional",
                    {},
                    {
                        "start": 10.0,
                        "end": 10.0 - 0.5 * size,
                        "step": -0.5,
                        "size": size,
                    },
                ),
                Gate2ShapeProfile(
                    "forward_unit",
                    {},
                    {
                        "start": 0.0,
                        "end": float(size),
                        "step": 1.0,
                        "size": size,
                    },
                ),
            ),
        )

    if name == "fg_eye":
        n, m = 37, 29
        block_i = block_j = 32
        shape_profiles = (
            Gate2ShapeProfile(
                "one_by_one",
                {"out_ptr": (1, 1)},
                {"N": 1, "M": 1},
                (1, 1),
            ),
            Gate2ShapeProfile(
                "block_edge",
                {"out_ptr": (block_i, block_j)},
                {"N": block_i, "M": block_j},
                (1, 1),
            ),
            Gate2ShapeProfile(
                "both_axes_tail",
                {"out_ptr": (block_i + 1, block_j + 3)},
                {"N": block_i + 1, "M": block_j + 3},
                (2, 2),
            ),
            Gate2ShapeProfile(
                "rectangular_nondivisible",
                {"out_ptr": (n, m)},
                {"N": n, "M": m},
                (2, 1),
            ),
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "out_ptr", "tensor", "f32", (n, m)
                ),
                SCRCCompilerArgument("N", "scalar", "i32", value=n),
                SCRCCompilerArgument("M", "scalar", "i32", value=m),
            ),
            (2, 1),
            {
                "BLOCK_i": block_i,
                "BLOCK_j": block_j,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            (
                "ceil_div(N, BLOCK_i)",
                "ceil_div(M, BLOCK_j)",
            ),
            ("BLOCK_i", "BLOCK_j"),
            1.0,
            Tolerance(rtol=0.0, atol=0.0),
            gate2_shape_profiles=shape_profiles,
            gate1_specialization_profiles=shape_profiles,
            gate1_primary_profile_id="rectangular_nondivisible",
        )

    if name == "fg_trace":
        rows, cols = 1031, 1025
        num_diag = min(rows, cols)
        block = 1024
        shape_profiles = (
            Gate2ShapeProfile(
                "short_trip",
                {"inp_ptr": (rows, cols), "out_ptr": ()},
                {
                    "num_diag": 1,
                    "stride0": cols,
                    "stride1": 1,
                },
                (1,),
            ),
            Gate2ShapeProfile(
                "exact_block_trip",
                {"inp_ptr": (rows, cols), "out_ptr": ()},
                {
                    "num_diag": block,
                    "stride0": cols,
                    "stride1": 1,
                },
                (1,),
            ),
            Gate2ShapeProfile(
                "block_plus_one_trip",
                {"inp_ptr": (rows, cols), "out_ptr": ()},
                {
                    "num_diag": num_diag,
                    "stride0": cols,
                    "stride1": 1,
                },
                (1,),
            ),
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "inp_ptr", "tensor", "f32", (rows, cols)
                ),
                SCRCCompilerArgument("out_ptr", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "num_diag", "scalar", "i32", value=num_diag
                ),
                SCRCCompilerArgument(
                    "stride0", "scalar", "i32", value=cols
                ),
                SCRCCompilerArgument(
                    "stride1", "scalar", "i32", value=1
                ),
            ),
            (1,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1",),
            ("BLOCK_SIZE",),
            0.25,
            Tolerance(rtol=4.0e-4, atol=4.0e-4),
            gate2_shape_profiles=shape_profiles,
            gate1_specialization_profiles=shape_profiles,
            gate1_primary_profile_id="block_plus_one_trip",
        )

    if name == "fg_log_softmax":
        m, n, k = 5, 513, 3
        block_m, block_n = 8, 256
        shape_profiles = (
            Gate2ShapeProfile(
                "single_n_tile",
                {"output_ptr": (3, 127, 2), "input_ptr": (3, 127, 2)},
                {"M": 3, "N": 127, "K": 2},
                (1, 2),
            ),
            Gate2ShapeProfile(
                "exact_tiles",
                {
                    "output_ptr": (block_m, block_n, 1),
                    "input_ptr": (block_m, block_n, 1),
                },
                {"M": block_m, "N": block_n, "K": 1},
                (1, 1),
            ),
            Gate2ShapeProfile(
                "multi_n_tail",
                {"output_ptr": (m, n, k), "input_ptr": (m, n, k)},
                {"M": m, "N": n, "K": k},
                (1, k),
            ),
            Gate2ShapeProfile(
                "multi_m_and_n_tail",
                {"output_ptr": (9, 257, 2), "input_ptr": (9, 257, 2)},
                {"M": 9, "N": 257, "K": 2},
                (2, 2),
            ),
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "output_ptr", "tensor", "f32", (m, n, k)
                ),
                SCRCCompilerArgument(
                    "input_ptr", "tensor", "f32", (m, n, k)
                ),
                SCRCCompilerArgument("M", "scalar", "i32", value=m),
                SCRCCompilerArgument("N", "scalar", "i32", value=n),
                SCRCCompilerArgument("K", "scalar", "i32", value=k),
            ),
            (1, k),
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "num_warps": 8,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(M, BLOCK_M)", "K"),
            ("BLOCK_M", "BLOCK_N"),
            3.0,
            Tolerance(rtol=5.0e-4, atol=5.0e-5),
            gate2_shape_profiles=shape_profiles,
            gate1_specialization_profiles=shape_profiles,
            gate1_primary_profile_id="multi_n_tail",
        )

    if name == "fg_sum":
        n = 1003
        block = 1024
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument("mid", "tensor", "f32", (n,)),
                SCRCCompilerArgument("out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "mid_size", "scalar", "i32", value=n
                ),
            ),
            (1, 1, 1),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1", "1", "1"),
            ("BLOCK_MID",),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-4),
        )

    if name == "fg_amax":
        mid_size = 63
        block = 64
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "mid", "tensor", "f32", (mid_size,)
                ),
                SCRCCompilerArgument("out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "mid_size", "scalar", "i32", value=mid_size
                ),
            ),
            (1, 1),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1", "1"),
            ("BLOCK_MID",),
            32.0,
            Tolerance(rtol=0.0, atol=0.0),
            gate2_shape_profiles=(
                Gate2ShapeProfile(
                    "one", {"mid": (1,)}, {"mid_size": 1}
                ),
                Gate2ShapeProfile(
                    "block_minus_one",
                    {"mid": (block - 1,)},
                    {"mid_size": block - 1},
                ),
                Gate2ShapeProfile(
                    "block", {"mid": (block,)}, {"mid_size": block}
                ),
            ),
            gate2_value_profiles=(
                Gate2ValueProfile(
                    "normal", {"mid": InputDistribution.NORMAL}, 32.0
                ),
                Gate2ValueProfile(
                    "uniform", {"mid": InputDistribution.UNIFORM}, 32.0
                ),
                Gate2ValueProfile(
                    "large_dynamic",
                    {"mid": InputDistribution.LARGE_DYNAMIC},
                    1.0,
                ),
            ),
        )

    if name == "fg_mean":
        total = 1003
        mid_size = 37
        block = 64
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "mid", "tensor", "f32", (mid_size,)
                ),
                SCRCCompilerArgument("out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "M", "scalar", "i32", value=total
                ),
                SCRCCompilerArgument(
                    "MID_SIZE", "scalar", "i32", value=mid_size
                ),
            ),
            (1, 1, 1),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1", "1", "1"),
            ("BLOCK_MID",),
            0.25,
            Tolerance(rtol=2.0e-4, atol=2.0e-6),
        )

    if name == "fg_max":
        mid_size = 63
        block = 64
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "mid", "tensor", "f32", (mid_size,)
                ),
                SCRCCompilerArgument("out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "mid_size", "scalar", "i32", value=mid_size
                ),
            ),
            (1, 1, 1),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1", "1", "1"),
            ("BLOCK_MID",),
            32.0,
            Tolerance(rtol=0.0, atol=0.0),
        )

    if name == "fg_min":
        mid_size = 64
        block = 64
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "mid", "tensor", "f32", (mid_size,)
                ),
                SCRCCompilerArgument("out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "mid_size", "scalar", "i32", value=mid_size
                ),
            ),
            (1, 1, 1),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1", "1", "1"),
            ("BLOCK_MID",),
            32.0,
            Tolerance(rtol=0.0, atol=0.0),
            gate2_shape_profiles=(
                Gate2ShapeProfile(
                    "one", {"mid": (1,)}, {"mid_size": 1}
                ),
                Gate2ShapeProfile(
                    "block_minus_one",
                    {"mid": (block - 1,)},
                    {"mid_size": block - 1},
                ),
                Gate2ShapeProfile(
                    "block", {"mid": (block,)}, {"mid_size": block}
                ),
            ),
            gate2_value_profiles=(
                Gate2ValueProfile(
                    "normal", {"mid": InputDistribution.NORMAL}, 32.0
                ),
                Gate2ValueProfile(
                    "uniform", {"mid": InputDistribution.UNIFORM}, 32.0
                ),
                Gate2ValueProfile(
                    "large_dynamic",
                    {"mid": InputDistribution.LARGE_DYNAMIC},
                    1.0,
                ),
                Gate2ValueProfile(
                    "ieee_special",
                    {"mid": InputDistribution.IEEE_SPECIAL},
                    1.0,
                ),
            ),
        )

    if name == "fg_prod":
        mid_size = 17
        block = 32
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "mid", "tensor", "f32", (mid_size,)
                ),
                SCRCCompilerArgument("out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "mid_size", "scalar", "i32", value=mid_size
                ),
            ),
            (1, 1, 1),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1", "1", "1"),
            ("BLOCK_MID",),
            0.125,
            Tolerance(rtol=2.0e-4, atol=2.0e-6),
            {"mid": InputDistribution.NEAR_ONE},
        )

    if name == "fg_vector_l2":
        mid_size = 37
        block = 64
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "Mid", "tensor", "f32", (mid_size,)
                ),
                SCRCCompilerArgument("Out", "tensor", "f32", ()),
                SCRCCompilerArgument(
                    "MID_SIZE", "scalar", "i32", value=mid_size
                ),
            ),
            (1,),
            {
                "BLOCK_MID": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("1",),
            ("BLOCK_MID",),
            4.0,
            Tolerance(rtol=2.0e-4, atol=2.0e-6),
            {"Mid": InputDistribution.POSITIVE},
        )

    if name == "fg_vector_l1":
        n = 1003
        block = 1024
        order = 1.5
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument("X", "tensor", "f32", (n,)),
                SCRCCompilerArgument("Mid", "tensor", "f32", (1,)),
                SCRCCompilerArgument(
                    "ord", "scalar", "f32", value=order
                ),
                SCRCCompilerArgument("M", "scalar", "i32", value=n),
            ),
            (1,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(M, BLOCK_SIZE)",),
            ("BLOCK_SIZE",),
            2.0,
            Tolerance(rtol=3.0e-4, atol=3.0e-4),
            gate2_shape_profiles=(
                Gate2ShapeProfile(
                    "one_ord1",
                    {"X": (1,), "Mid": (1,)},
                    {"ord": 1.0, "M": 1},
                    (1,),
                ),
                Gate2ShapeProfile(
                    "block_minus_one_ord15",
                    {"X": (block - 1,), "Mid": (1,)},
                    {"ord": 1.5, "M": block - 1},
                    (1,),
                ),
                Gate2ShapeProfile(
                    "block_ord2",
                    {"X": (block,), "Mid": (1,)},
                    {"ord": 2.0, "M": block},
                    (1,),
                ),
            ),
            gate2_value_profiles=(
                Gate2ValueProfile(
                    "normal", {"X": InputDistribution.NORMAL}, 2.0
                ),
                Gate2ValueProfile(
                    "uniform", {"X": InputDistribution.UNIFORM}, 2.0
                ),
            ),
        )

    if name == "fg_skip_layer_norm":
        rows, width = 7, 257
        block = 512
        profiles = tuple(
            Gate2ShapeProfile(
                profile_id,
                {
                    "Y": (rows, profile_width),
                    "X": (rows, profile_width),
                    "R": (rows, profile_width),
                    "W": (profile_width,),
                    "B": (profile_width,),
                },
                {
                    "y_stride_r": profile_width,
                    "y_stride_c": 1,
                    "x_stride_r": profile_width,
                    "x_stride_c": 1,
                    "r_stride_r": profile_width,
                    "r_stride_c": 1,
                    "N": profile_width,
                },
                (rows,),
            )
            for profile_id, profile_width in (
                ("single_width", 1),
                ("rectangular_tail", width),
                ("block_minus_one", block - 1),
                ("block", block),
            )
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "Y", "tensor", "f32", (rows, width)
                ),
                SCRCCompilerArgument(
                    "X", "tensor", "f32", (rows, width)
                ),
                SCRCCompilerArgument(
                    "R", "tensor", "f32", (rows, width)
                ),
                SCRCCompilerArgument("W", "tensor", "f32", (width,)),
                SCRCCompilerArgument("B", "tensor", "f32", (width,)),
                SCRCCompilerArgument(
                    "y_stride_r", "scalar", "i32", value=width
                ),
                SCRCCompilerArgument(
                    "y_stride_c", "scalar", "i32", value=1
                ),
                SCRCCompilerArgument(
                    "x_stride_r", "scalar", "i32", value=width
                ),
                SCRCCompilerArgument(
                    "x_stride_c", "scalar", "i32", value=1
                ),
                SCRCCompilerArgument(
                    "r_stride_r", "scalar", "i32", value=width
                ),
                SCRCCompilerArgument(
                    "r_stride_c", "scalar", "i32", value=1
                ),
                SCRCCompilerArgument(
                    "N", "scalar", "i32", value=width
                ),
                SCRCCompilerArgument(
                    "eps", "scalar", "f32", value=1.0e-5
                ),
            ),
            (rows,),
            {
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            (str(rows),),
            ("BLOCK_SIZE",),
            2.0,
            Tolerance(rtol=5.0e-4, atol=5.0e-5),
            gate2_shape_profiles=profiles,
            gate2_value_profiles=(
                Gate2ValueProfile(
                    "normal",
                    {
                        "X": InputDistribution.NORMAL,
                        "R": InputDistribution.NORMAL,
                        "W": InputDistribution.NORMAL,
                        "B": InputDistribution.NORMAL,
                    },
                    2.0,
                ),
                Gate2ValueProfile(
                    "large_dynamic",
                    {
                        "X": InputDistribution.LARGE_DYNAMIC,
                        "R": InputDistribution.LARGE_DYNAMIC,
                        "W": InputDistribution.NEAR_ONE,
                        "B": InputDistribution.UNIFORM,
                    },
                    1.0,
                ),
                Gate2ValueProfile(
                    "near_constant",
                    {
                        "X": InputDistribution.NEAR_ONE,
                        "R": InputDistribution.NEAR_ONE,
                        "W": InputDistribution.NEAR_ONE,
                        "B": InputDistribution.UNIFORM,
                    },
                    1.0e-3,
                ),
            ),
            gate1_specialization_profiles=profiles,
            gate1_primary_profile_id="rectangular_tail",
        )

    if name == "fg_index_select":
        rows, cols, index_len = 19, 23, 17
        block_m, block_n = 8, 32
        profiles = (
            Gate2ShapeProfile(
                "single",
                {
                    "inp": (1, cols),
                    "out": (1, 1),
                    "index": (1,),
                },
                {"M": 1, "N": cols, "index_len": 1},
                (1, 1),
            ),
            Gate2ShapeProfile(
                "rectangular_tail",
                {
                    "inp": (rows, cols),
                    "out": (rows, index_len),
                    "index": (index_len,),
                },
                {"M": rows, "N": cols, "index_len": index_len},
                (3, 1),
            ),
            Gate2ShapeProfile(
                "two_axis_multitile",
                {
                    "inp": (17, 37),
                    "out": (17, 65),
                    "index": (65,),
                },
                {"M": 17, "N": 37, "index_len": 65},
                (3, 3),
            ),
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "inp", "tensor", "f32", (rows, cols)
                ),
                SCRCCompilerArgument(
                    "out", "tensor", "f32", (rows, index_len)
                ),
                SCRCCompilerArgument(
                    "M", "scalar", "i32", value=rows
                ),
                SCRCCompilerArgument(
                    "N", "scalar", "i32", value=cols
                ),
                SCRCCompilerArgument(
                    "index", "tensor", "i64", (index_len,)
                ),
                SCRCCompilerArgument(
                    "index_len", "scalar", "i32", value=index_len
                ),
            ),
            (3, 1),
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            (
                "ceil_div(M, BLOCK_M)",
                "ceil_div(index_len, BLOCK_N)",
            ),
            ("BLOCK_M", "BLOCK_N"),
            2.0,
            Tolerance(rtol=0.0, atol=0.0),
            {
                "inp": InputDistribution.NORMAL,
                "index": InputDistribution.POSITIVE,
            },
            {"inp": 2.0, "index": float(cols - 1)},
            gate2_shape_profiles=profiles,
            gate2_value_profiles=(
                Gate2ValueProfile(
                    "random_index_normal_input",
                    {
                        "inp": InputDistribution.NORMAL,
                        "index": InputDistribution.POSITIVE,
                    },
                ),
                Gate2ValueProfile(
                    "boundary_index_uniform_input",
                    {
                        "inp": InputDistribution.UNIFORM,
                        "index": InputDistribution.INTEGER_BOUNDARY,
                    },
                ),
            ),
            gate1_specialization_profiles=profiles,
            gate1_primary_profile_id="rectangular_tail",
        )

    if name == "fg_diag_extract":
        rows, cols, diagonal = 11, 13, 2
        block = 128
        profiles = (
            Gate2ShapeProfile(
                "single",
                {"data_ptr": (1, 3), "output_ptr": (1,)},
                {"N": 1, "M": 3, "stride0": 3, "stride1": 1},
                (1,),
            ),
            Gate2ShapeProfile(
                "rectangular_tail",
                {
                    "data_ptr": (rows, cols),
                    "output_ptr": (rows,),
                },
                {
                    "N": rows,
                    "M": cols,
                    "stride0": cols,
                    "stride1": 1,
                },
                (1,),
            ),
            Gate2ShapeProfile(
                "multi_block",
                {"data_ptr": (129, 131), "output_ptr": (129,)},
                {"N": 129, "M": 131, "stride0": 131, "stride1": 1},
                (2,),
            ),
        )
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "data_ptr", "tensor", "f32", (rows, cols)
                ),
                SCRCCompilerArgument(
                    "output_ptr", "tensor", "f32", (rows,)
                ),
                SCRCCompilerArgument(
                    "N", "scalar", "i32", value=rows
                ),
                SCRCCompilerArgument(
                    "M", "scalar", "i32", value=cols
                ),
                SCRCCompilerArgument(
                    "stride0", "scalar", "i32", value=cols
                ),
                SCRCCompilerArgument(
                    "stride1", "scalar", "i32", value=1
                ),
            ),
            (1,),
            {
                "diagonal": diagonal,
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("ceil_div(N, BLOCK_SIZE)",),
            ("diagonal", "BLOCK_SIZE"),
            2.0,
            Tolerance(rtol=0.0, atol=0.0),
            gate2_shape_profiles=profiles,
            gate1_specialization_profiles=profiles,
            gate1_primary_profile_id="rectangular_tail",
        )

    if name == "fg_embedding":
        vocab, count, width = 97, 17, 13
        block = 16
        return LiveKernelCase(
            name,
            _function(candidate),
            (
                SCRCCompilerArgument(
                    "out_ptr", "tensor", "f32", (count, width)
                ),
                SCRCCompilerArgument(
                    "in_ptr", "tensor", "i64", (count,)
                ),
                SCRCCompilerArgument(
                    "weight_ptr", "tensor", "f32", (vocab, width)
                ),
            ),
            (count,),
            {
                "N": width,
                "BLOCK_SIZE": block,
                "num_warps": 4,
                "num_stages": 1,
            },
            ("num_warps", "num_stages"),
            ("K",),
            ("N", "BLOCK_SIZE"),
            2.0,
            Tolerance(rtol=0.0, atol=0.0),
            {
                "in_ptr": InputDistribution.POSITIVE,
                "weight_ptr": InputDistribution.NORMAL,
            },
            {"in_ptr": float(vocab - 1), "weight_ptr": 2.0},
            gate2_value_profiles=(
                Gate2ValueProfile(
                    "random_index_normal_weight",
                    {
                        "in_ptr": InputDistribution.POSITIVE,
                        "weight_ptr": InputDistribution.NORMAL,
                    },
                    2.0,
                ),
                Gate2ValueProfile(
                    "boundary_index_uniform_weight",
                    {
                        "in_ptr": InputDistribution.INTEGER_BOUNDARY,
                        "weight_ptr": InputDistribution.UNIFORM,
                    },
                    2.0,
                ),
            ),
        )

    raise AssertionError(name)


def _case_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.model,
        effort=args.effort,
        seed=args.seed,
        lift_repairs=args.lift_repairs,
        relift_rounds=args.relift_rounds,
        semantic_interventions=args.semantic_interventions,
        gate2_matrix=args.gate2_matrix,
        gate2_seeds=args.gate2_seeds,
        rule_release=getattr(args, "rule_release", None),
    )


def _rule_registry_from_args(args: argparse.Namespace):
    registry = default_triton_rule_registry()
    release_path = getattr(args, "rule_release", None)
    if release_path is not None:
        registry = load_rule_release(Path(release_path)).apply(registry)
    return registry.frozen_snapshot()


def _batch_identity(
    case: LiveKernelCase, args: argparse.Namespace
) -> dict[str, Any]:
    registry = _rule_registry_from_args(args)
    source_sha = hashlib.sha256(case.source_text.encode("utf-8")).hexdigest()
    return {
        "schema": 36,
        "performance_schema": "scrc_case_performance_v1",
        "gate1_gate2_metrics_schema": "scrc_gate1_gate2_conditional_v4",
        "model": args.model,
        "effort": args.effort,
        "seed": args.seed,
        "lift_repairs": args.lift_repairs,
        "relift_rounds": args.relift_rounds,
        "normal_recovery": "relift_only_restart_full_pipeline",
        "replay_generation_policy": "fresh_only_no_prior_replay_source",
        "gate1_stagnation_policy": {
            "policy": GATE1_STAGNATION_POLICY,
            "required_consecutive_observations": 2,
            "basis": "canonical_artifact_and_exact_hard_diff",
        },
        "ttir_fingerprint_schema": TTIR_FINGERPRINT_SCHEMA,
        "composer_version": COMPOSER_VERSION,
        "relift_guidance_policy": {
            "policy": RELIFT_GUIDANCE_POLICY,
            "structural_policy": RELIFT_STRUCTURAL_GUIDANCE_POLICY,
            "source": (
                "first_completed_gate1_or_fixed_c_family_hard_comparison"
            ),
            "family_policy": (
                GATE1_FAMILY_RELIFT_GUIDANCE_POLICY
            ),
            "transport_failure_relift": "disabled_without_hard_diff",
            "replay_source_input": "forbidden",
        },
        "gate1_termination_policy": {
            "policy": GATE1_TERMINATION_POLICY,
            "no_closure_requires_structured_reason": True,
            "gate2_verdict_independence": True,
        },
        "semantic_interventions": args.semantic_interventions,
        "fixed_c_family_closure_policy": (
            GATE1_FAMILY_ANTI_LEAKAGE_POLICY
        ),
        "paired_intervention_family_policy": (
            PAIRED_INTERVENTION_FAMILY_POLICY
        ),
        "paired_intervention_scope": (
            "full_fixed_c_family"
            if case.gate1_specialization_profiles
            else "primary_profile"
        ),
        "pi_ablations": True,
        "pi_ablation_coverage": "all_rulecard_causal_steps",
        "pi_ablation_verdict_role": "gate1_hard_condition",
        "pi_ablation_requires_closed_control": True,
        "pi_ablation_opaque_reference_discipline": (
            "resolve_handles_never_emit_v1"
        ),
        "maximum_provider_attempts_per_generation": 3,
        "ttir_score_semantics": TTIR_SCORE_SEMANTICS,
        "masked_j_policy": MASKED_J_POLICY,
        "masked_j_family_policy": MASKED_J_FAMILY_POLICY,
        "masked_j_scope": (
            "full_fixed_c_family"
            if case.gate1_specialization_profiles
            else "primary_profile"
        ),
        "masked_j_required_outcome": (
            "all_profiles_legal_comparable_hard_mismatch"
            if case.gate1_specialization_profiles
            else "compile_or_hard_nonclosure"
        ),
        "masked_j_score_role": "not_used",
        "replay_reference_naming": "opaque_replay_refs_v1",
        "relift_protocol": SCRC_RELIFT_PROTOCOL,
        "gate1_specialization_policy": _gate1_specialization_policy_json(
            case
        ),
        "gate2_input_policy": {
            "default_distribution": InputDistribution.NORMAL.value,
            "argument_distributions": {
                name: InputDistribution(distribution).value
                for name, distribution in sorted(
                    case.input_distributions.items()
                )
            },
            "scale": case.input_scale,
            "argument_scales": dict(sorted(case.input_scales.items())),
            "tolerance": asdict(case.tolerance),
        },
        "gate2_execution_protocol": GATE2_EXECUTION_PROTOCOL,
        "gate2_execution_legs": ["replay", "scalar"],
        "gate2_comparison_protocol": GATE2_COMPARISON_PROTOCOL,
        "gate2_matrix_policy": _gate2_matrix_policy_json(
            case,
            enabled=bool(args.gate2_matrix),
            seeds=_selected_gate2_seeds(args),
        ),
        "source_sha256": source_sha,
        "compiler_view": case.compiler_view().to_json(),
        "rule_registry_digest": registry.digest,
        "lift_prompt_sha256": scrc_lift_prompt_sha256(
            case.source_text,
            case.function_name,
            registry=registry,
            compiler_view=case.compiler_view(),
        ),
    }


def _resume_row(
    saved: dict[str, Any], identity: dict[str, Any]
) -> dict[str, Any] | None:
    batch = saved.get("batch")
    if not isinstance(batch, dict) or not batch.get("terminal"):
        return None
    if batch.get("identity") != identity:
        return None
    row = batch.get("row")
    return row if isinstance(row, dict) else None


def _result_status(report: dict[str, Any]) -> str:
    if report.get("preflight", {}).get("ok") is False:
        return "source_compile_failed"
    if not report.get("lift", {}).get("ok"):
        return "lift_failed"
    gate1 = report.get("gate1", {}).get("status")
    if gate1 != "replay_closed":
        return gate1 or "gate1_missing"
    suite = report.get("gate2_suite")
    if isinstance(suite, dict):
        return "gate2_pass" if suite.get("passed") else "gate2_suite_failed"
    return report.get("gate2", {}).get("status") or "gate2_missing"


def _response_ids(report: dict[str, Any]) -> tuple[list[str], str | None]:
    gate1 = report.get("gate1")
    if not isinstance(gate1, dict):
        return [], None
    normal: list[str] = []
    for item in gate1.get("candidates", []):
        generation = item.get("generation") if isinstance(item, dict) else None
        if isinstance(generation, dict) and generation.get("response_id"):
            normal.append(str(generation["response_id"]))
    masked = gate1.get("masked_generation")
    masked_id = (
        str(masked["response_id"])
        if isinstance(masked, dict) and masked.get("response_id")
        else None
    )
    return normal, masked_id


def _relift_response_ids(report: dict[str, Any]) -> list[str]:
    relifts = report.get("relifts")
    if not isinstance(relifts, list):
        return []
    return [
        str(item["response_id"])
        for item in relifts
        if isinstance(item, dict) and item.get("response_id")
    ]


def _attach_batch_performance(
    report: dict[str, Any],
    *,
    case_started: float,
    preflight_ms: float | None,
) -> dict[str, Any]:
    performance = report.get("performance")
    if not isinstance(performance, dict):
        performance = {
            "schema": "scrc_case_performance_v1",
            "protocol": "scrc_wall_clock_observation_v1",
            "clock": "time.perf_counter",
            "verdict_role": "observation_only",
            "timing_ms": {},
            "recorded_invocations": {},
            "gate1_observation": {},
            "limitations": [],
        }
        report["performance"] = performance
    timing = performance.get("timing_ms")
    if not isinstance(timing, dict):
        timing = {}
        performance["timing_ms"] = timing
    timing["preflight_compile"] = (
        None if preflight_ms is None else round(float(preflight_ms), 3)
    )
    timing["heldout_total"] = round(
        (time.perf_counter() - case_started) * 1000.0, 3
    )
    return performance


def _nearest_rank(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _timing_distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total_ms": None,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    total = sum(values)
    return {
        "count": len(values),
        "total_ms": round(total, 3),
        "mean_ms": round(total / len(values), 3),
        "p50_ms": round(float(_nearest_rank(values, 0.50)), 3),
        "p95_ms": round(float(_nearest_rank(values, 0.95)), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def _summarize_performance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    timing_keys = (
        "preflight_compile",
        "initial_lift",
        "gate1_loop",
        "specialization_family",
        "pi_ablation",
        "gate2",
        "gate1_core",
        "total",
        "heldout_total",
    )
    timings: dict[str, list[float]] = {key: [] for key in timing_keys}
    invocation_totals: Counter[str] = Counter()
    gate1_statuses: Counter[str] = Counter()
    termination_reasons: Counter[str] = Counter()
    masked_outcomes: Counter[str] = Counter()
    measured_cases = 0
    replay_compile = 0
    replay_compile_ok = 0
    relift_attempts = 0
    cases_with_relift = 0
    per_case: list[dict[str, Any]] = []

    for row in rows:
        performance = row.get("performance")
        if not isinstance(performance, dict):
            continue
        measured_cases += 1
        timing = performance.get("timing_ms")
        timing = timing if isinstance(timing, dict) else {}
        for key in timing_keys:
            value = timing.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timings[key].append(float(value))
        invocations = performance.get("recorded_invocations")
        invocations = invocations if isinstance(invocations, dict) else {}
        for key, value in invocations.items():
            if isinstance(value, int) and not isinstance(value, bool):
                invocation_totals[key] += value
        replay_compile += int(invocations.get("normal_replay_compile", 0))
        replay_compile_ok += int(
            invocations.get("normal_replay_compile_ok", 0)
        )
        observation = performance.get("gate1_observation")
        observation = observation if isinstance(observation, dict) else {}
        status = observation.get("status")
        if isinstance(status, str):
            gate1_statuses[status] += 1
        termination = observation.get("outer_loop_termination")
        if isinstance(termination, str):
            termination_reasons[termination] += 1
        masked = observation.get("masked_j_outcome")
        if isinstance(masked, str):
            masked_outcomes[masked] += 1
        relifts = observation.get("relift_attempts", 0)
        if isinstance(relifts, int) and not isinstance(relifts, bool):
            relift_attempts += relifts
            cases_with_relift += int(relifts > 0)
        per_case.append(
            {
                "case": row.get("case"),
                "status": row.get("status"),
                "gate1_status": status,
                "outer_loop_termination": termination,
                "timing_ms": {
                    key: timing.get(key) for key in timing_keys
                },
                "relift_attempts": relifts,
            }
        )

    return {
        "protocol": "scrc_batch_performance_v1",
        "verdict_role": "observation_only",
        "selected_cases": len(rows),
        "measured_cases": measured_cases,
        "missing_measurement_cases": len(rows) - measured_cases,
        "timing": {
            key: _timing_distribution(values)
            for key, values in timings.items()
        },
        "recorded_invocation_totals": dict(sorted(invocation_totals.items())),
        "gate1": {
            "status_counts": dict(sorted(gate1_statuses.items())),
            "outer_loop_termination_counts": dict(
                sorted(termination_reasons.items())
            ),
            "replay_compile_attempts": replay_compile,
            "replay_compile_ok": replay_compile_ok,
            "replay_compile_rate": (
                None
                if replay_compile == 0
                else round(replay_compile_ok / replay_compile, 6)
            ),
            "cases_with_relift": cases_with_relift,
            "relift_attempts": relift_attempts,
            "masked_j_outcomes": dict(sorted(masked_outcomes.items())),
        },
        "per_case": per_case,
    }


def _load_cached_lift(
    path: Path,
    case: LiveKernelCase,
    args: argparse.Namespace,
) -> tuple[Gate1Artifact, dict[str, Any]]:
    saved = json.loads(path.read_text(encoding="utf-8"))
    source_sha = hashlib.sha256(case.source_text.encode("utf-8")).hexdigest()
    registry = _rule_registry_from_args(args)
    if saved.get("model") != args.model or saved.get("effort") != args.effort:
        raise ValueError("cached lift model/effort differs from this batch")
    if saved.get("source", {}).get("sha256") != source_sha:
        raise ValueError("cached lift source hash differs from current source")
    if saved.get("compiler_view") != case.compiler_view().to_json():
        raise ValueError("cached lift compiler view differs from current case")
    lift = saved.get("lift")
    if not isinstance(lift, dict) or not lift.get("ok"):
        raise ValueError("cached report does not contain a successful lift")
    if lift.get("rule_registry_digest") != registry.digest:
        raise ValueError("cached lift rule registry differs from current registry")
    expected_prompt_sha = scrc_lift_prompt_sha256(
        case.source_text,
        case.function_name,
        registry=registry,
        compiler_view=case.compiler_view(),
    )
    if lift.get("prompt_sha256") != expected_prompt_sha:
        raise ValueError("cached lift prompt differs from current Lift prompt")
    artifact_payload = dict(lift["artifact"])
    artifact_payload.pop("source_sha256", None)
    artifact_payload.pop("source_bytes", None)
    artifact_payload["source"] = case.source_text
    return Gate1Artifact.from_json(artifact_payload), lift


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=READY_CASES)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--rule-release",
        type=Path,
        help=(
            "explicit approved Gate1 RuleRegistry release; one frozen snapshot "
            "is used by the entire held-out batch"
        ),
    )
    parser.add_argument("--gate2-matrix", action="store_true")
    parser.add_argument(
        "--gate2-seed", dest="gate2_seeds", action="append", type=int
    )
    parser.add_argument("--lift-repairs", type=int, default=2)
    parser.add_argument(
        "--semantic-interventions",
        type=_positive_int,
        default=3,
    )
    parser.add_argument(
        "--relift-rounds",
        type=_positive_int,
        default=3,
        help="total Gate1 artifact rounds, including the initial Lift",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/scrc_heldout_batch"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--lift-cache-dir", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.list:
        print(json.dumps([asdict(item) for item in CANDIDATES], indent=2))
        return 0

    selected = tuple(args.case or READY_CASES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    infrastructure_failed = False
    for name in selected:
        path = args.out_dir / f"{name}.json"
        print(json.dumps({"case": name, "stage": "start"}), flush=True)
        case_started = time.perf_counter()
        preflight_ms: float | None = None
        try:
            case = build_heldout_case(name)
            identity = _batch_identity(case, args)
            if args.resume and path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                resumed = _resume_row(saved, identity)
                if resumed is not None:
                    rows.append(resumed)
                    print(json.dumps({"case": name, "stage": "resumed"}))
                    continue
                print(
                    json.dumps({"case": name, "stage": "resume_stale"}),
                    flush=True,
                )
            compile_case = replace(
                _compile_case(case), schedule=case.schedule_values
            )
            preflight = TritonCompiler().compile(
                case.kernel_source, compile_case
            )
            preflight_ms = preflight.elapsed_ms
            if not preflight.ok:
                code = 1
                report = {
                    "case": name,
                    "model": args.model,
                    "effort": args.effort,
                    "source": {
                        "path": str(case.source_path),
                        "function": case.function_name,
                    },
                    "preflight": {
                        "ok": False,
                        "diagnostic": preflight.diagnostic,
                    },
                }
            else:
                cached_artifact = None
                cached_lift = None
                cached_origin = None
                if args.lift_cache_dir is not None:
                    cached_path = args.lift_cache_dir / f"{name}.json"
                    if cached_path.exists():
                        cached_artifact, cached_lift = _load_cached_lift(
                            cached_path, case, args
                        )
                        cached_origin = str(cached_path)
                code, report = run_case(
                    case,
                    _case_args(args),
                    artifact=cached_artifact,
                    lift_evidence=cached_lift,
                    lift_origin=cached_origin,
                )
                report["preflight"] = {
                    "ok": True,
                    "cache_hit": preflight.cache_hit,
                    "elapsed_ms": preflight.elapsed_ms,
                }
            status = _result_status(report)
            normal_ids, masked_id = _response_ids(report)
            relift_ids = _relift_response_ids(report)
            gate1_gate2 = extract_gate1_gate2_observation(report)
            row = {
                "case": name,
                "status": status,
                "individual_exit_code": code,
                "report": str(path),
                "lift_response_id": report.get("lift", {}).get("response_id"),
                "relift_response_ids": relift_ids,
                "normal_replay_response_ids": normal_ids,
                "masked_response_id": masked_id,
                "gate1_gate2": gate1_gate2,
            }
            report["batch"] = {
                "terminal": True,
                "identity": identity,
                "row": row,
            }
        except Exception as exc:  # batch records infrastructure failure and continues
            infrastructure_failed = True
            row = {
                "case": name,
                "status": "harness_error",
                "individual_exit_code": 5,
                "report": str(path),
                "error": f"{type(exc).__name__}: {exc}",
            }
            report = {
                "case": name,
                "model": args.model,
                "effort": args.effort,
                "batch": {"terminal": True, "row": row},
            }
        performance = _attach_batch_performance(
            report,
            case_started=case_started,
            preflight_ms=preflight_ms,
        )
        row["performance"] = performance
        _write_report(str(path), report)
        rows.append(row)
        print(
            json.dumps(
                {"case": name, "stage": "complete", "status": row["status"]}
            ),
            flush=True,
        )

    counts = Counter(row["status"] for row in rows)
    summary = {
        "model": args.model,
        "effort": args.effort,
        "selected": list(selected),
        "counts": dict(sorted(counts.items())),
        "gate1_gate2_metrics": summarize_gate1_gate2(rows),
        "performance_summary": _summarize_performance(rows),
        "results": rows,
    }
    _write_report(str(args.out_dir / "summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if infrastructure_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
