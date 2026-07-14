from __future__ import annotations

import pytest

from intentir.verify.gate1.compare import compare_fingerprints, compare_ttir
from intentir.verify.gate1.ttir import (
    TTIRParseError,
    fingerprint_ttir,
    runtime_argument_names,
)


EQUIVALENT_SOURCE = r'''
#loc = loc("/source/kernel.py":3:0)
#loc1 = loc("/source/kernel.py":4:8)
module {
  tt.func public @source_kernel(%x: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("x"(#loc)), %out: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("out"(#loc)), %n: i32 loc("n"(#loc))) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32> loc(#loc1)
    %one = arith.constant dense<1.000000e+00> : tensor<8xf32> loc(#loc1)
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> loc(#loc1)
    %limit = tt.splat %n : i32 -> tensor<8xi32> loc(#loc1)
    %mask = arith.cmpi slt, %offsets, %limit : tensor<8xi32> loc(#loc1)
    %xbase = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc1)
    %xptr = tt.addptr %xbase, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc1)
    %value = tt.load %xptr, %mask, %zero : tensor<8x!tt.ptr<f32>> loc(#loc1)
    %sum = arith.addf %value, %one : tensor<8xf32> loc(#loc1)
    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc1)
    %optr = tt.addptr %obase, %offsets : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc1)
    tt.store %optr, %sum, %mask : tensor<8x!tt.ptr<f32>> loc(#loc1)
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
'''


EQUIVALENT_REPLAY = r'''
#loc = loc("/different/replay.py":90:0)
#loc73 = loc(callsite(loc("library.py":1:2) at #loc))
module {
  tt.func public @blind_replay(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("renamed"(#loc)), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("also_renamed"(#loc)), %arg2: i32 loc("size"(#loc))) attributes {noinline = false} {
    %unit = arith.constant dense<1.000000e+00> : tensor<8xf32> loc(#loc73)
    %zfill = arith.constant dense<0.000000e+00> : tensor<8xf32> loc(#loc73)
    %srcs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc73)
    %lanes = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> loc(#loc73)
    %bound = tt.splat %arg2 : i32 -> tensor<8xi32> loc(#loc73)
    %guard = arith.cmpi sgt, %bound, %lanes : tensor<8xi32> loc(#loc73)
    %addresses = tt.addptr %srcs, %lanes : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc73)
    %loaded = tt.load %addresses, %guard, %zfill : tensor<8x!tt.ptr<f32>> loc(#loc73)
    %answer = arith.addf %unit, %loaded : tensor<8xf32> loc(#loc73)
    %dsts = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc73)
    %dst_addresses = tt.addptr %dsts, %lanes : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc73)
    tt.store %dst_addresses, %answer, %guard : tensor<8x!tt.ptr<f32>> loc(#loc73)
    tt.return loc(#loc73)
  } loc(#loc)
} loc(#loc)
'''


REDUCE_TTIR = r'''
module {
  tt.func public @row_sum(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %lanes = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %limit = tt.splat %n : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %lanes, %limit : tensor<8xi32>
    %base = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %ptrs = tt.addptr %base, %lanes : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %values = tt.load %ptrs, %mask, %zero : tensor<8x!tt.ptr<f32>>
    %total = "tt.reduce"(%values) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<8xf32>) -> f32
    tt.store %out, %total : !tt.ptr<f32>
    tt.return
  }
}
'''


DATAFLOW_TTIR = r'''
module {
  tt.func public @dataflow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %one = arith.constant 1.000000e+00 : f32
    %two = arith.constant 2.000000e+00 : f32
    %value = tt.load %x : !tt.ptr<f32>
    %left = arith.addf %value, %one : f32
    %right = arith.mulf %value, %two : f32
    %answer = arith.subf %left, %right : f32
    tt.store %out, %answer : !tt.ptr<f32>
    tt.return
  }
}
'''


LOOP_DOT_TTIR = r'''
module {
  tt.func public @matmul(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %out: !tt.ptr<f32>, %k: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<8x8xf32>
    %acc = scf.for %iv = %c0 to %k step %c1 iter_args(%carry = %zero) -> (tensor<8x8xf32>) : i32 {
      %av = tt.load %a : !tt.ptr<f16>
      %bv = tt.load %b : !tt.ptr<f16>
      %next = tt.dot %av, %bv, %carry, inputPrecision = tf32 : tensor<8x8xf16> * tensor<8x8xf16> -> tensor<8x8xf32>
      scf.yield %next : tensor<8x8xf32>
    }
    tt.store %out, %acc : !tt.ptr<f32>
    tt.return
  }
}
'''


def test_runtime_argument_names_preserve_surviving_specialized_subsequence() -> None:
    text = r'''
module {
  tt.func public @specialized(%x: !tt.ptr<f32>, %alpha: f32, %stride: i32) {
    tt.return
  }
}
'''
    assert runtime_argument_names(text) == ("x", "alpha", "stride")


def test_locations_ssa_names_independent_order_and_bounded_rewrites_are_ignored():
    source = fingerprint_ttir(EQUIVALENT_SOURCE)
    replay = fingerprint_ttir(EQUIVALENT_REPLAY)

    assert source == replay
    assert source.digest == replay.digest
    verdict = compare_fingerprints(source, replay)
    assert verdict.hard_pass
    assert verdict.score == pytest.approx(1.0)
    assert all(verdict.hard_passes.values())
    assert verdict.differences == ()

    assert [argument.access for argument in source.interface.arguments] == [
        "input",
        "output",
        "value",
    ]


def test_schedule_difference_is_reported_but_not_a_hard_failure():
    source = fingerprint_ttir(EQUIVALENT_SOURCE, schedule={"BLOCK": 8, "num_warps": 4})
    replay = fingerprint_ttir(EQUIVALENT_REPLAY, schedule={"BLOCK": 16, "num_warps": 4})
    verdict = compare_fingerprints(source, replay)

    assert verdict.hard_pass
    assert verdict.score == pytest.approx(1.0)
    assert not verdict.component("schedule-only").passed
    assert verdict.primary_difference_category == "schedule-only"


def test_runtime_interface_dtype_change_is_hard_failure():
    replay = EQUIVALENT_REPLAY.replace("%arg0: !tt.ptr<f32>", "%arg0: !tt.ptr<f16>")
    verdict = compare_ttir(EQUIVALENT_SOURCE, replay)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["interface"]
    assert verdict.primary_difference_category == "interface"
    assert 0.0 <= verdict.score < 1.0


def test_deleted_reduce_is_detected_by_op_graph_and_reduction_list():
    without_reduce = REDUCE_TTIR.replace(
        '''    %total = "tt.reduce"(%values) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<8xf32>) -> f32''',
        "    %total = arith.constant 0.000000e+00 : f32",
    )
    verdict = compare_ttir(REDUCE_TTIR, without_reduce)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["op_multiset"]
    assert not verdict.hard_passes["dataflow"]
    assert not verdict.hard_passes["reduction"]
    assert verdict.primary_difference_category == "reduction"


def test_changed_reduce_combiner_is_detected():
    max_reduce = REDUCE_TTIR.replace("arith.addf %lhs, %rhs", "arith.maxnumf %lhs, %rhs")
    verdict = compare_ttir(REDUCE_TTIR, max_reduce)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["reduction"]
    reduction_diffs = verdict.component("reduction").differences
    assert reduction_diffs
    assert verdict.primary_difference_category == "reduction"


def test_changed_other_fill_is_predicate_effect_failure():
    negative_infinity_fill = REDUCE_TTIR.replace(
        "dense<0.000000e+00>", "dense<0xFF800000>", 1
    )
    verdict = compare_ttir(REDUCE_TTIR, negative_infinity_fill)

    assert not verdict.hard_pass
    assert verdict.hard_passes["op_multiset"]
    assert verdict.hard_passes["reduction"]
    assert not verdict.hard_passes["predicate_effect"]
    assert verdict.primary_difference_category == "predicate_effect"


def test_changed_mask_boundary_is_predicate_effect_failure():
    off_by_one = REDUCE_TTIR.replace("arith.cmpi slt", "arith.cmpi sle")
    verdict = compare_ttir(REDUCE_TTIR, off_by_one)

    assert not verdict.hard_pass
    assert verdict.hard_passes["op_multiset"]
    assert not verdict.hard_passes["predicate_effect"]
    assert verdict.primary_difference_category == "predicate_effect"


def test_same_op_multiset_but_rewired_noncommutative_dataflow_is_rejected():
    rewired = DATAFLOW_TTIR.replace(
        "arith.subf %left, %right", "arith.subf %right, %left"
    )
    verdict = compare_ttir(DATAFLOW_TTIR, rewired)

    assert not verdict.hard_pass
    assert verdict.hard_passes["interface"]
    assert verdict.hard_passes["op_multiset"]
    assert not verdict.hard_passes["dataflow"]
    assert verdict.hard_passes["reduction"]
    assert verdict.hard_passes["predicate_effect"]
    assert verdict.primary_difference_category == "dataflow"


def test_same_op_graph_but_changed_computation_constant_is_rejected():
    changed_constant = DATAFLOW_TTIR.replace(
        "arith.constant 2.000000e+00", "arith.constant 3.000000e+00"
    )
    verdict = compare_ttir(DATAFLOW_TTIR, changed_constant)

    assert not verdict.hard_pass
    assert verdict.hard_passes["op_multiset"]
    assert not verdict.hard_passes["dataflow"]
    assert verdict.hard_passes["predicate_effect"]
    assert verdict.primary_difference_category == "dataflow"


def test_scf_for_dot_and_effects_are_fingerprinted():
    fingerprint = fingerprint_ttir(LOOP_DOT_TTIR)

    assert dict(fingerprint.op_multiset)["scf.for"] == 1
    assert dict(fingerprint.op_multiset)["tt.dot"] == 1
    assert dict(fingerprint.effect_multiset) == {
        "control.scf.for": 1,
        "read": 2,
        "write": 1,
    }
    assert len(fingerprint.dots) == 1
    assert fingerprint.dots[0].operand_count == 3
    assert len(fingerprint.loops) == 1
    assert fingerprint.loops[0].body_effects == (("read", 2),)


def test_unresolved_ssa_is_rejected_instead_of_weakening_fingerprint():
    malformed = DATAFLOW_TTIR.replace("%left, %right", "%missing, %right")
    with pytest.raises(TTIRParseError, match="unresolved SSA"):
        fingerprint_ttir(malformed)


def test_observed_redundant_zero_rule_is_normalized() -> None:
    redundant = DATAFLOW_TTIR.replace(
        "    tt.store %out, %answer : !tt.ptr<f32>",
        "    %zero = arith.constant 0.000000e+00 : f32\n"
        "    %same = arith.addf %answer, %zero : f32\n"
        "    tt.store %out, %same : !tt.ptr<f32>",
    )
    assert compare_ttir(DATAFLOW_TTIR, redundant).hard_pass

    dense_shared_zero = EQUIVALENT_SOURCE.replace(
        "    tt.store %optr, %sum, %mask : tensor<8x!tt.ptr<f32>>",
        "    %same = arith.addf %sum, %zero : tensor<8xf32>\n"
        "    tt.store %optr, %same, %mask : tensor<8x!tt.ptr<f32>>",
    )
    assert compare_ttir(EQUIVALENT_SOURCE, dense_shared_zero).hard_pass


def test_observed_rsqrt_reciprocal_sqrt_rule_is_normalized() -> None:
    rsqrt = DATAFLOW_TTIR.replace(
        "    %answer = arith.subf %left, %right : f32",
        "    %answer = math.rsqrt %left : f32",
    )
    reciprocal_sqrt = DATAFLOW_TTIR.replace(
        "    %answer = arith.subf %left, %right : f32",
        "    %unit = arith.constant 1.000000e+00 : f32\n"
        "    %root = math.sqrt %left : f32\n"
        "    %answer = arith.divf %unit, %root : f32",
    )
    assert compare_ttir(rsqrt, reciprocal_sqrt).hard_pass


def test_rank_tuple_prefers_a_closer_failed_candidate():
    fill_only = compare_ttir(
        REDUCE_TTIR,
        REDUCE_TTIR.replace("dense<0.000000e+00>", "dense<0xFF800000>", 1),
    )
    delete_reduce_text = REDUCE_TTIR.replace(
        '''    %total = "tt.reduce"(%values) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<8xf32>) -> f32''',
        "    %total = arith.constant 0.000000e+00 : f32",
    )
    deleted_reduce = compare_ttir(REDUCE_TTIR, delete_reduce_text)

    assert fill_only.rank_tuple > deleted_reduce.rank_tuple
    assert fill_only.to_dict()["rank_tuple"] == list(fill_only.rank_tuple)
