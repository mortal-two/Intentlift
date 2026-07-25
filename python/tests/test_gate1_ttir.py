from __future__ import annotations

import json
import re
from pathlib import Path

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


def test_cache_policy_hints_are_global_presentation_noise() -> None:
    with_cache_hints = EQUIVALENT_SOURCE.replace(
        "tt.load %xptr, %mask, %zero :",
        "tt.load %xptr, %mask, %zero cacheModifier = ca :",
    ).replace(
        "tt.store %optr, %sum, %mask :",
        "tt.store %optr, %sum, %mask cacheModifier = cs :",
    )

    result = compare_ttir(EQUIVALENT_SOURCE, with_cache_hints)
    unknown_hint = EQUIVALENT_SOURCE.replace(
        "tt.load %xptr, %mask, %zero :",
        "tt.load %xptr, %mask, %zero cacheModifier = future_policy :",
    )
    wrong_direction_hint = EQUIVALENT_SOURCE.replace(
        "tt.load %xptr, %mask, %zero :",
        "tt.load %xptr, %mask, %zero cacheModifier = cs :",
    )
    eviction_hint = EQUIVALENT_SOURCE.replace(
        "tt.load %xptr, %mask, %zero :",
        "tt.load %xptr, %mask, %zero evictionPolicy = evict_first :",
    )
    unknown_eviction = EQUIVALENT_SOURCE.replace(
        "tt.load %xptr, %mask, %zero :",
        "tt.load %xptr, %mask, %zero evictionPolicy = future_policy :",
    )
    volatile_change = EQUIVALENT_SOURCE.replace(
        "tt.load %xptr, %mask, %zero :",
        "tt.load %xptr, %mask, %zero isVolatile = true :",
    )

    assert result.passed, result.to_dict()
    assert not compare_ttir(EQUIVALENT_SOURCE, unknown_hint).passed
    assert not compare_ttir(EQUIVALENT_SOURCE, wrong_direction_hint).passed
    assert compare_ttir(EQUIVALENT_SOURCE, eviction_hint).passed
    assert not compare_ttir(EQUIVALENT_SOURCE, unknown_eviction).passed
    assert not compare_ttir(EQUIVALENT_SOURCE, volatile_change).passed


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


SCAN_TTIR = r'''
module {
  tt.func public @prefix(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      tt.scan.return %lhs : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %result : tensor<2x4xf32>
  }
}
'''


MULTI_SCAN_TTIR = r'''
module {
  tt.func public @tuple_prefix(%values: tensor<2x4xf32>, %indices: tensor<2x4xi32>) -> (tensor<2x4xf32>, tensor<2x4xi32>) {
    %result:2 = "tt.scan"(%values, %indices) <{axis = 1 : i32, reverse = true}> ({
    ^bb0(%lhs_value: f32, %lhs_index: i32, %rhs_value: f32, %rhs_index: i32):
      %next_value = arith.addf %lhs_value, %rhs_value : f32
      %next_index = arith.maxsi %lhs_index, %rhs_index : i32
      tt.scan.return %next_value, %next_index : f32, i32
    }) : (tensor<2x4xf32>, tensor<2x4xi32>) -> (tensor<2x4xf32>, tensor<2x4xi32>)
    tt.return %result#0, %result#1 : tensor<2x4xf32>, tensor<2x4xi32>
  }
}
'''


SAME_TYPE_MULTI_SCAN_TTIR = r'''
module {
  tt.func public @paired_prefix(%first: tensor<2x4xf32>, %second: tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>) {
    %result:2 = "tt.scan"(%first, %second) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs_first: f32, %lhs_second: f32, %rhs_first: f32, %rhs_second: f32):
      tt.scan.return %lhs_first, %lhs_second : f32, f32
    }) : (tensor<2x4xf32>, tensor<2x4xf32>) -> (tensor<2x4xf32>, tensor<2x4xf32>)
    tt.return %result#0, %result#1 : tensor<2x4xf32>, tensor<2x4xf32>
  }
}
'''


OUTLINED_SCAN_COMBINER_TTIR = r'''
module {
  tt.func public @outlined_prefix(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %next = tt.call @combine(%lhs, %rhs) : (f32, f32) -> f32
      tt.scan.return %next : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %result : tensor<2x4xf32>
  }
  tt.func private @combine(%left: f32, %right: f32) -> f32 {
    %sum = arith.addf %left, %right : f32
    tt.return %sum : f32
  }
}
'''


INLINED_SCAN_COMBINER_TTIR = r'''
module {
  tt.func public @inline_prefix(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %output = "tt.scan"(%input) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%left: f32, %right: f32):
      %sum = arith.addf %left, %right : f32
      tt.scan.return %sum : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %output : tensor<2x4xf32>
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


LOOP_CARRIED_POINTER_ACCESS_TTIR = r'''
module {
  tt.func public @loop_pointer_access(%source: !tt.ptr<f16>, %destination: !tt.ptr<f16>, %inout: !tt.ptr<f16>, %unused: !tt.ptr<f16>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %step = tt.splat %c1 : i32 -> tensor<8xi32>
    %source_tile = tt.splat %source : !tt.ptr<f16> -> tensor<8x!tt.ptr<f16>>
    %destination_tile = tt.splat %destination : !tt.ptr<f16> -> tensor<8x!tt.ptr<f16>>
    %inout_tile = tt.splat %inout : !tt.ptr<f16> -> tensor<8x!tt.ptr<f16>>
    %unused_tile = tt.splat %unused : !tt.ptr<f16> -> tensor<8x!tt.ptr<f16>>
    %final:3 = scf.for %iv = %c0 to %n step %c1 iter_args(%read_ptr = %source_tile, %write_ptr = %destination_tile, %inout_ptr = %inout_tile) -> (tensor<8x!tt.ptr<f16>>, tensor<8x!tt.ptr<f16>>, tensor<8x!tt.ptr<f16>>) : i32 {
      %value = tt.load %read_ptr : tensor<8x!tt.ptr<f16>>
      tt.store %write_ptr, %value : tensor<8x!tt.ptr<f16>>
      %prior = tt.load %inout_ptr : tensor<8x!tt.ptr<f16>>
      tt.store %inout_ptr, %prior : tensor<8x!tt.ptr<f16>>
      %next_read = tt.addptr %read_ptr, %step : tensor<8x!tt.ptr<f16>>, tensor<8xi32>
      %next_write = tt.addptr %write_ptr, %step : tensor<8x!tt.ptr<f16>>, tensor<8xi32>
      %next_inout = tt.addptr %inout_ptr, %step : tensor<8x!tt.ptr<f16>>, tensor<8xi32>
      scf.yield %next_read, %next_write, %next_inout : tensor<8x!tt.ptr<f16>>, tensor<8x!tt.ptr<f16>>, tensor<8x!tt.ptr<f16>>
    }
    %final_value = tt.load %final#0 : tensor<8x!tt.ptr<f16>>
    tt.store %final#1, %final_value : tensor<8x!tt.ptr<f16>>
    tt.return
  }
}
'''


LOOP_CARRIED_POINTER_REPLACEMENT_TTIR = r'''
module {
  tt.func public @loop_pointer_replacement(%initial: !tt.ptr<f16>, %replacement: !tt.ptr<f16>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %final = scf.for %iv = %c0 to %n step %c1 iter_args(%ptr = %initial) -> (!tt.ptr<f16>) : i32 {
      %value = tt.load %ptr : !tt.ptr<f16>
      scf.yield %replacement : !tt.ptr<f16>
    }
    tt.return
  }
}
'''


NON_LOOP_REGION_POINTER_LEAF_TTIR = r'''
module {
  tt.func public @non_loop_region(%seed: tensor<8x!tt.ptr<f16>>, %out: !tt.ptr<f16>) {
    %zero = arith.constant 0 : i32
    %result = "tt.reduce"(%seed) <{axis = 0 : i32}> ({
    ^bb0(%lhs: !tt.ptr<f16>, %rhs: !tt.ptr<f16>):
      %value = tt.load %lhs : !tt.ptr<f16>
      %next = tt.addptr %lhs, %zero : !tt.ptr<f16>, i32
      tt.reduce.return %next : !tt.ptr<f16>
    }) : (tensor<8x!tt.ptr<f16>>) -> !tt.ptr<f16>
    %stored = tt.load %result : !tt.ptr<f16>
    tt.store %out, %stored : !tt.ptr<f16>
    tt.return
  }
}
'''


SIBLING_REGION_SSA_REUSE_TTIR = r'''
module {
  tt.func public @sibling_reuse(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant 1.000000e+00 : f32
    %value = tt.load %x : !tt.ptr<f32>
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %value = arith.addf %value, %one : f32
      scf.yield
    }
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %value = arith.addf %value, %one : f32
      scf.yield
    }
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''


RENAMED_SIBLING_REGION_SSA_TTIR = r'''
module {
  tt.func public @sibling_reuse(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %lower = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %unit = arith.constant 1.000000e+00 : f32
    %outer = tt.load %x : !tt.ptr<f32>
    scf.for %first_iv = %lower to %n step %step : i32 {
      %first_local = arith.addf %outer, %unit : f32
      scf.yield
    }
    scf.for %second_iv = %lower to %n step %step : i32 {
      %second_local = arith.addf %outer, %unit : f32
      scf.yield
    }
    tt.store %out, %outer : !tt.ptr<f32>
    tt.return
  }
}
'''


SIBLING_SCOPE_NORMALIZATION_REUSE_TTIR = r'''
module {
  tt.func public @sibling_normalize(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant 0.000000e+00 : f32
    %value = tt.load %x : !tt.ptr<f32>
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %same = arith.addf %value, %zero : f32
      scf.yield
    }
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %same = arith.addf %value, %zero : f32
      scf.yield
    }
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''


RENAMED_SIBLING_SCOPE_NORMALIZATION_TTIR = r'''
module {
  tt.func public @sibling_normalize(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %lower = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %neutral = arith.constant 0.000000e+00 : f32
    %outer = tt.load %x : !tt.ptr<f32>
    scf.for %first_iv = %lower to %n step %step : i32 {
      %first_same = arith.addf %outer, %neutral : f32
      scf.yield
    }
    scf.for %second_iv = %lower to %n step %step : i32 {
      %second_same = arith.addf %outer, %neutral : f32
      scf.yield
    }
    tt.store %out, %outer : !tt.ptr<f32>
    tt.return
  }
}
'''


NESTED_SCOPE_SHADOW_TTIR = r'''
module {
  tt.func public @nested_shadow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %one = arith.constant 1.000000e+00 : f32
    %value = tt.load %x : !tt.ptr<f32>
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %local = arith.addf %value, %one : f32
      scf.for %iv = %c0 to %n step %c1 : i32 {
        %local = arith.mulf %local, %one : f32
        scf.yield
      }
      %after = arith.subf %local, %one : f32
      scf.yield
    }
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''


RENAMED_NESTED_SCOPE_SHADOW_TTIR = r'''
module {
  tt.func public @nested_shadow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %lower = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %unit = arith.constant 1.000000e+00 : f32
    %outer = tt.load %x : !tt.ptr<f32>
    scf.for %outer_iv = %lower to %n step %step : i32 {
      %outer_local = arith.addf %outer, %unit : f32
      scf.for %inner_iv = %lower to %n step %step : i32 {
        %inner_local = arith.mulf %outer_local, %unit : f32
        scf.yield
      }
      %outer_after = arith.subf %outer_local, %unit : f32
      scf.yield
    }
    tt.store %out, %outer : !tt.ptr<f32>
    tt.return
  }
}
'''


SIBLING_REDUCE_BLOCK_ARG_REUSE_TTIR = r'''
module {
  tt.func public @two_reduces(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %pointers = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %values = tt.load %pointers : tensor<8x!tt.ptr<f32>>
    %left_total = "tt.reduce"(%values) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<8xf32>) -> f32
    %right_total = "tt.reduce"(%values) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %combined = arith.addf %lhs, %rhs : f32
      tt.reduce.return %combined : f32
    }) : (tensor<8xf32>) -> f32
    %answer = arith.addf %left_total, %right_total : f32
    tt.store %out, %answer : !tt.ptr<f32>
    tt.return
  }
}
'''


RENAMED_SIBLING_REDUCE_BLOCK_ARG_TTIR = r'''
module {
  tt.func public @two_reduces(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %addresses = tt.splat %x : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %loaded = tt.load %addresses : tensor<8x!tt.ptr<f32>>
    %first_total = "tt.reduce"(%loaded) <{axis = 0 : i32}> ({
    ^bb0(%first_lhs: f32, %first_rhs: f32):
      %first_combined = arith.addf %first_lhs, %first_rhs : f32
      tt.reduce.return %first_combined : f32
    }) : (tensor<8xf32>) -> f32
    %second_total = "tt.reduce"(%loaded) <{axis = 0 : i32}> ({
    ^bb0(%second_lhs: f32, %second_rhs: f32):
      %second_combined = arith.addf %second_lhs, %second_rhs : f32
      tt.reduce.return %second_combined : f32
    }) : (tensor<8xf32>) -> f32
    %result = arith.addf %first_total, %second_total : f32
    tt.store %out, %result : !tt.ptr<f32>
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


def test_empty_hard_component_is_equal_but_not_similarity_evidence() -> None:
    verdict = compare_ttir(DATAFLOW_TTIR, DATAFLOW_TTIR)
    reduction = verdict.component("reduction")

    assert reduction.passed
    assert not reduction.applicable
    assert reduction.score is None
    assert verdict.inactive_categories == ("reduction",)
    assert verdict.active_categories == (
        "interface",
        "op_multiset",
        "dataflow",
        "predicate_effect",
    )
    assert verdict.active_weight == pytest.approx(0.8)
    assert verdict.score == pytest.approx(1.0)
    serialized = verdict.to_dict()
    assert serialized["inactive_categories"] == ["reduction"]
    assert serialized["active_weight"] == pytest.approx(0.8)
    assert next(
        component
        for component in serialized["components"]
        if component["category"] == "reduction"
    )["applicable"] is False


def test_score_renormalizes_only_categories_with_observed_evidence() -> None:
    changed = DATAFLOW_TTIR.replace(
        "arith.constant 2.000000e+00", "arith.constant 3.000000e+00"
    )
    verdict = compare_ttir(DATAFLOW_TTIR, changed)
    weights = {
        "interface": 0.15,
        "op_multiset": 0.20,
        "dataflow": 0.25,
        "predicate_effect": 0.20,
    }
    weighted_active = sum(
        weights[category] * verdict.component(category).score
        for category in verdict.active_categories
    )

    assert verdict.inactive_categories == ("reduction",)
    assert verdict.score == pytest.approx(weighted_active / 0.8)
    # The historical fixed-weight formula awarded the empty reduction category
    # a free 0.20, which strictly inflated every imperfect comparison here.
    assert weighted_active + 0.20 > verdict.score


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


def test_typed_scan_and_heterogeneous_multi_scan_are_observed() -> None:
    single = fingerprint_ttir(SCAN_TTIR)
    multiple = fingerprint_ttir(MULTI_SCAN_TTIR)

    assert single.schema_version == "triton-ttir-3.6-gate1-v10"
    assert dict(single.op_multiset)["tt.scan"] == 1
    assert dict(multiple.op_multiset)["tt.scan"] == 1
    assert compare_ttir(SCAN_TTIR, SCAN_TTIR.replace("%lhs", "%renamed")).hard_pass


def test_scan_input_type_is_proved_through_typed_arith_select() -> None:
    selected = SCAN_TTIR.replace(
        "%x: tensor<2x4xf32>",
        "%x: tensor<2x4xf32>, %mask: tensor<2x4xi1>",
        1,
    ).replace(
        "    %result = \"tt.scan\"(%x)",
        "    %zero = arith.constant dense<0.000000e+00> : tensor<2x4xf32>\n"
        "    %selected = arith.select %mask, %x, %zero : "
        "tensor<2x4xi1>, tensor<2x4xf32>\n"
        "    %result = \"tt.scan\"(%selected)",
        1,
    )

    fingerprint = fingerprint_ttir(selected)
    assert dict(fingerprint.op_multiset)["tt.scan"] == 1

    lying_select = selected.replace(
        "tensor<2x4xi1>, tensor<2x4xf32>\n"
        "    %result = \"tt.scan\"",
        "tensor<2x4xi1>, tensor<2x4xf64>\n"
        "    %result = \"tt.scan\"",
        1,
    )
    with pytest.raises(TTIRParseError, match="input SSA type cannot be established"):
        fingerprint_ttir(lying_select)


def test_scan_axis_and_reverse_are_hard_dataflow_evidence() -> None:
    changed_axis = SCAN_TTIR.replace("axis = 1", "axis = 0", 1)
    changed_reverse = SCAN_TTIR.replace("reverse = false", "reverse = true", 1)

    for changed in (changed_axis, changed_reverse):
        verdict = compare_ttir(SCAN_TTIR, changed)
        assert not verdict.hard_pass
        assert verdict.hard_passes["op_multiset"]
        assert not verdict.hard_passes["dataflow"]


def test_scan_noncommutative_argument_order_is_not_normalized_away() -> None:
    # Returning the first argument is associative but not commutative.  The
    # observer must retain lhs/rhs orientation even though both programs have
    # exactly the same operation multiset.
    returns_rhs = SCAN_TTIR.replace(
        "tt.scan.return %lhs : f32", "tt.scan.return %rhs : f32", 1
    )
    verdict = compare_ttir(SCAN_TTIR, returns_rhs)

    assert not verdict.hard_pass
    assert verdict.hard_passes["op_multiset"]
    assert not verdict.hard_passes["dataflow"]


def test_multi_scan_return_slot_order_is_hard_evidence() -> None:
    swapped = SAME_TYPE_MULTI_SCAN_TTIR.replace(
        "tt.scan.return %lhs_first, %lhs_second : f32, f32",
        "tt.scan.return %lhs_second, %lhs_first : f32, f32",
        1,
    )
    verdict = compare_ttir(SAME_TYPE_MULTI_SCAN_TTIR, swapped)

    assert not verdict.hard_pass
    assert verdict.hard_passes["op_multiset"]
    assert not verdict.hard_passes["dataflow"]


def test_outlined_pure_scan_combiner_matches_equivalent_inline_region() -> None:
    verdict = compare_ttir(OUTLINED_SCAN_COMBINER_TTIR, INLINED_SCAN_COMBINER_TTIR)

    assert verdict.hard_pass, verdict.to_dict()


def test_scan_inside_outlined_helper_matches_equivalent_inline_scan() -> None:
    outlined = r'''
module {
  tt.func public @caller(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %result = tt.call @prefix(%x) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %result : tensor<2x4xf32>
  }
  tt.func private @prefix(%input: tensor<2x4xf32>) -> tensor<2x4xf32> {
    %output = "tt.scan"(%input) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      tt.scan.return %lhs : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %output : tensor<2x4xf32>
  }
}
'''

    assert compare_ttir(outlined, SCAN_TTIR).hard_pass


def test_scan_participates_in_recursive_region_topology_digest() -> None:
    nested_loop = r'''
module {
  tt.func public @nested(%x: tensor<2x4xf32>, %n: i32) -> tensor<2x4xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      scf.for %iv = %c0 to %n step %c1 : i32 {
        scf.yield
      }
      tt.scan.return %lhs : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %result : tensor<2x4xf32>
  }
}
'''
    sibling_loop = r'''
module {
  tt.func public @sibling(%x: tensor<2x4xf32>, %n: i32) -> tensor<2x4xf32> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %n step %c1 : i32 {
      scf.yield
    }
    %result = "tt.scan"(%x) <{axis = 1 : i32, reverse = false}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      tt.scan.return %lhs : f32
    }) : (tensor<2x4xf32>) -> tensor<2x4xf32>
    tt.return %result : tensor<2x4xf32>
  }
}
'''
    left = fingerprint_ttir(nested_loop)
    right = fingerprint_ttir(sibling_loop)
    verdict = compare_fingerprints(left, right)

    assert dict(left.op_multiset) == dict(right.op_multiset)
    assert not verdict.hard_pass
    assert not verdict.hard_passes["dataflow"]


@pytest.mark.parametrize(
    "malformed, message",
    [
        (
            SCAN_TTIR.replace('"tt.scan"(%x)', "tt.scan(%x)", 1),
            "quoted-generic syntax",
        ),
        (
            SCAN_TTIR.replace(
                "<{axis = 1 : i32, reverse = false}>",
                "<{reverse = false}>",
                1,
            ),
            "exactly one axis and one reverse",
        ),
        (
            SCAN_TTIR.replace(
                "<{axis = 1 : i32, reverse = false}>",
                "<{axis = 1 : i32}>",
                1,
            ),
            "exactly one axis and one reverse",
        ),
        (
            SCAN_TTIR.replace(
                "<{axis = 1 : i32, reverse = false}>",
                "<{axis = 1 : i32, axis = 0 : i32, reverse = false}>",
                1,
            ),
            "exactly one axis and one reverse",
        ),
        (
            SCAN_TTIR.replace("reverse = false", "reverse = 0 : i1", 1),
            "reverse must be an explicit boolean",
        ),
        (
            SCAN_TTIR.replace("axis = 1", "axis = 2", 1),
            "outside rank 2",
        ),
        (
            SCAN_TTIR.replace("axis = 1", "axis = -1", 1),
            "outside rank 2",
        ),
        (
            SCAN_TTIR.replace(
                '%result = "tt.scan"(%x)', '"tt.scan"()', 1
            ),
            "at least one input",
        ),
        (
            MULTI_SCAN_TTIR.replace("%result:2 =", "%result =", 1),
            "same number of SSA results as inputs",
        ),
        (
            SCAN_TTIR.replace("tensor<2x4xf32>", "f32"),
            "ranked tensors",
        ),
        (
            MULTI_SCAN_TTIR.replace(
                "tensor<2x4xi32>", "tensor<2x8xi32>"
            ),
            "one common shape and encoding",
        ),
        (
            SCAN_TTIR.replace(
                "}) : (tensor<2x4xf32>) -> tensor<2x4xf32>",
                "}) : (tensor<2x8xf32>) -> tensor<2x4xf32>",
                1,
            ),
            "explicit input types do not match actual SSA types",
        ),
        (
            SCAN_TTIR.replace(
                "}) : (tensor<2x4xf32>) -> tensor<2x4xf32>",
                "}) : (tensor<2x4xf32>) -> tensor<2x4xf64>",
                1,
            ),
            "result type must match",
        ),
        (
            SCAN_TTIR.replace(
                "}) : (tensor<2x4xf32>) -> tensor<2x4xf32>",
                "}) : tensor<2x4xf32> -> tensor<2x4xf32>",
                1,
            ),
            "function type inputs must use an explicit tuple",
        ),
        (
            SCAN_TTIR.replace(
                "tensor<2x4xf32>", "tensor<2x4xf32, #layout>",
            ),
            "tensor encodings are unsupported",
        ),
        (
            SCAN_TTIR.replace(
                "^bb0(%lhs: f32, %rhs: f32):",
                "^bb0(%lhs: f32):",
                1,
            ),
            "exactly 2N ordered arguments",
        ),
        (
            MULTI_SCAN_TTIR.replace(
                "%lhs_value: f32, %lhs_index: i32, %rhs_value: f32, %rhs_index: i32",
                "%lhs_value: f32, %lhs_index: f32, %rhs_value: i32, %rhs_index: i32",
                1,
            ),
            "ordered lhs",
        ),
        (
            MULTI_SCAN_TTIR.replace(
                "tt.scan.return %next_value, %next_index : f32, i32",
                "tt.scan.return %next_value : f32",
                1,
            ),
            "one value per scan input",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs : i32",
                1,
            ),
            "explicit types do not match scan element types",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "%wrong = arith.constant 1 : i32\n"
                "      tt.scan.return %wrong : f32",
                1,
            ),
            "actual SSA types do not match scan element types",
        ),
        (
            SCAN_TTIR.replace(
                "      tt.scan.return %lhs : f32",
                "    ^bb1(%other_lhs: f32, %other_rhs: f32):\n"
                "      tt.scan.return %other_lhs : f32",
                1,
            ),
            "more than one block",
        ),
        (
            SCAN_TTIR.replace("      tt.scan.return %lhs : f32\n", "", 1),
            "exactly one tt.scan.return",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs : f32\n"
                "      tt.scan.return %lhs : f32",
                1,
            ),
            "operation after tt.scan terminator",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs : f32\n"
                "      %late = arith.addf %lhs, %rhs : f32",
                1,
            ),
            "operation after tt.scan terminator",
        ),
        (
            SCAN_TTIR.replace("tt.scan.return", "tt.reduce.return", 1),
            "requires tt.scan.return",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                '"tt.scan.return"(%lhs) : (f32) -> ()',
                1,
            ),
            "custom terminator syntax",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs, : f32",
                1,
            ),
            "non-empty comma-separated operand list",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return ??? %lhs : f32",
                1,
            ),
            "is not one SSA value",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs nonsense : f32",
                1,
            ),
            "is not one SSA value",
        ),
    ],
)
def test_typed_scan_contract_fails_closed(
    malformed: str, message: str
) -> None:
    with pytest.raises(TTIRParseError, match=message):
        fingerprint_ttir(malformed)


def test_direct_effect_inside_scan_fails_closed() -> None:
    effectful = SCAN_TTIR.replace(
        "%x: tensor<2x4xf32>",
        "%x: tensor<2x4xf32>, %side: !tt.ptr<f32>",
        1,
    ).replace(
        "      tt.scan.return %lhs : f32",
        "      %observed = tt.load %side : !tt.ptr<f32>\n"
        "      %next = arith.addf %lhs, %observed : f32\n"
        "      tt.scan.return %next : f32",
        1,
    )

    with pytest.raises(TTIRParseError, match="tt.load inside tt.scan"):
        fingerprint_ttir(effectful)


def _with_unknown_effect_pair(
    text: str,
    *,
    anchor: str,
    operand: str,
    reverse: bool,
) -> str:
    names = ("future.effect_b", "future.effect_a") if reverse else (
        "future.effect_a",
        "future.effect_b",
    )
    pair = "".join(
        f'      "{name}"({operand}) : (f32) -> ()\n' for name in names
    )
    return text.replace(anchor, pair + anchor, 1)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "text, anchor, operand",
    [
        (DATAFLOW_TTIR, "    tt.store %out", "%answer"),
        (SIBLING_REGION_SSA_REUSE_TTIR, "      scf.yield", "%value"),
        (SCAN_TTIR, "      tt.scan.return", "%lhs"),
        (REDUCE_TTIR, "      tt.reduce.return", "%lhs"),
    ],
    ids=["root", "scf-for", "scan", "reduce"],
)
def test_unknown_operations_are_globally_observer_unsupported(
    text: str,
    anchor: str,
    operand: str,
    reverse: bool,
) -> None:
    unknown = _with_unknown_effect_pair(
        text,
        anchor=anchor,
        operand=operand,
        reverse=reverse,
    )

    with pytest.raises(TTIRParseError, match="closed operation/effect profile"):
        fingerprint_ttir(unknown)


@pytest.mark.parametrize(
    "name",
    ["arith.future_effect", "future.barrier", "tt.atomic_future"],
)
def test_dialect_or_effect_like_name_does_not_bypass_closed_classification(
    name: str,
) -> None:
    unknown = DATAFLOW_TTIR.replace(
        "    tt.store %out, %answer",
        f'    "{name}"(%answer) : (f32) -> ()\n'
        "    tt.store %out, %answer",
        1,
    )

    with pytest.raises(TTIRParseError, match="closed operation/effect profile"):
        fingerprint_ttir(unknown)


def test_malformed_quoted_generic_operation_is_not_accepted_as_opaque() -> None:
    malformed = DATAFLOW_TTIR.replace(
        "    tt.store %out, %answer",
        '    "future.effect"(%answer) totally-malformed-tail\n'
        "    tt.store %out, %answer",
        1,
    )

    with pytest.raises(TTIRParseError, match="missing its explicit function type"):
        fingerprint_ttir(malformed)


@pytest.mark.parametrize(
    "malformed",
    [
        DATAFLOW_TTIR.replace(
            "    tt.store %out, %answer",
            "    this is not a TTIR operation\n    tt.store %out, %answer",
            1,
        ),
        SCAN_TTIR.replace(
            "      tt.scan.return %lhs : f32",
            "      %broken =\n      tt.scan.return %lhs : f32",
            1,
        ),
    ],
    ids=["root-garbage", "scan-empty-assignment"],
)
def test_nonempty_unparsed_body_lines_fail_closed(malformed: str) -> None:
    with pytest.raises(TTIRParseError, match="cannot parse non-empty TTIR body line"):
        fingerprint_ttir(malformed)


@pytest.mark.parametrize("payload", ["one", "two"])
def test_scan_rejects_unresolved_tensor_encoding_aliases(payload: str) -> None:
    encoded = SCAN_TTIR.replace(
        "module {",
        f"#layout = #future.layout<{payload}>\nmodule {{",
        1,
    ).replace("tensor<2x4xf32>", "tensor<2x4xf32, #layout>")

    with pytest.raises(TTIRParseError, match="tensor encodings are unsupported"):
        fingerprint_ttir(encoded)


def test_nested_effect_inside_scan_fails_closed() -> None:
    effectful = SCAN_TTIR.replace(
        "%x: tensor<2x4xf32>",
        "%x: tensor<2x4xf32>, %side: !tt.ptr<f32>, %n: i32",
        1,
    ).replace(
        "    %result = \"tt.scan\"",
        "    %c0 = arith.constant 0 : i32\n"
        "    %c1 = arith.constant 1 : i32\n"
        "    %result = \"tt.scan\"",
        1,
    ).replace(
        "      tt.scan.return %lhs : f32",
        "      scf.for %iv = %c0 to %n step %c1 : i32 {\n"
        "        tt.store %side, %lhs : !tt.ptr<f32>\n"
        "        scf.yield\n"
        "      }\n"
        "      tt.scan.return %lhs : f32",
        1,
    )

    with pytest.raises(TTIRParseError, match="tt.store inside tt.scan"):
        fingerprint_ttir(effectful)


def test_effect_in_outlined_scan_combiner_fails_closed_after_expansion() -> None:
    effectful = OUTLINED_SCAN_COMBINER_TTIR.replace(
        "%x: tensor<2x4xf32>",
        "%x: tensor<2x4xf32>, %side: !tt.ptr<f32>",
        1,
    ).replace(
        "@combine(%lhs, %rhs) : (f32, f32) -> f32",
        "@combine(%lhs, %rhs, %side) : (f32, f32, !tt.ptr<f32>) -> f32",
        1,
    ).replace(
        "@combine(%left: f32, %right: f32) -> f32",
        "@combine(%left: f32, %right: f32, %side: !tt.ptr<f32>) -> f32",
        1,
    ).replace(
        "    tt.return %sum : f32",
        "    tt.store %side, %sum : !tt.ptr<f32>\n"
        "    tt.return %sum : f32",
        1,
    )

    with pytest.raises(TTIRParseError, match="tt.store inside tt.scan"):
        fingerprint_ttir(effectful)


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
    assert fingerprint.loops[0].block_argument_count == 2
    assert fingerprint.loops[0].induction_use_count == 0
    assert fingerprint.loops[0].carried_argument_use_counts == (1,)
    assert len(fingerprint.loops[0].captured_values) == 2
    assert dict(fingerprint.loops[0].body_op_multiset) == {
        "tt.dot": 1,
        "tt.load": 2,
    }


def test_scf_for_carried_pointer_provenance_drives_exact_interface_modes() -> None:
    fingerprint = fingerprint_ttir(LOOP_CARRIED_POINTER_ACCESS_TTIR)

    assert [argument.access for argument in fingerprint.interface.arguments] == [
        "input",
        "output",
        "inout",
        "unused",
        "value",
    ]
    assert fingerprint.loops[0].block_argument_count == 4
    assert fingerprint.loops[0].carried_argument_use_counts == (2, 2, 3)


def test_scf_for_yield_can_introduce_another_pointer_dependency() -> None:
    fingerprint = fingerprint_ttir(LOOP_CARRIED_POINTER_REPLACEMENT_TTIR)

    assert [argument.access for argument in fingerprint.interface.arguments] == [
        "input",
        "input",
        "value",
    ]


@pytest.mark.parametrize(
    "malformed",
    [
        LOOP_CARRIED_POINTER_ACCESS_TTIR.replace(
            "%final:3 = scf.for", "%final:2 = scf.for", 1
        ),
        LOOP_CARRIED_POINTER_ACCESS_TTIR.replace(
            "scf.yield %next_read, %next_write, %next_inout :",
            "scf.yield %next_read, %next_write :",
            1,
        ),
    ],
)
def test_scf_for_carried_pointer_arity_mismatch_fails_closed(
    malformed: str,
) -> None:
    with pytest.raises(TTIRParseError, match="arity"):
        fingerprint_ttir(malformed)


def test_side_effecting_reduce_region_fails_closed() -> None:
    with pytest.raises(TTIRParseError, match="tt.load inside tt.reduce"):
        fingerprint_ttir(NON_LOOP_REGION_POINTER_LEAF_TTIR)


def test_sibling_regions_may_reuse_ssa_names_without_cross_scope_capture() -> None:
    reused = fingerprint_ttir(SIBLING_REGION_SSA_REUSE_TTIR)
    renamed = fingerprint_ttir(RENAMED_SIBLING_REGION_SSA_TTIR)

    assert reused == renamed
    assert dict(reused.op_multiset)["scf.for"] == 2
    assert dict(reused.op_multiset)["arith.addf"] == 2
    assert len(reused.loops) == 2
    assert [argument.access for argument in reused.interface.arguments] == [
        "input",
        "output",
        "value",
    ]


def test_bounded_normalization_is_independent_of_scope_local_ssa_reuse() -> None:
    reused = fingerprint_ttir(SIBLING_SCOPE_NORMALIZATION_REUSE_TTIR)
    renamed = fingerprint_ttir(RENAMED_SIBLING_SCOPE_NORMALIZATION_TTIR)

    assert reused == renamed
    assert "arith.addf" not in dict(reused.op_multiset)
    assert dict(reused.op_multiset)["scf.for"] == 2


def test_nested_region_shadowing_resolves_rhs_capture_before_local_definition() -> None:
    shadowed = fingerprint_ttir(NESTED_SCOPE_SHADOW_TTIR)
    renamed = fingerprint_ttir(RENAMED_NESTED_SCOPE_SHADOW_TTIR)

    assert shadowed == renamed
    assert len(shadowed.loops) == 2
    assert dict(shadowed.op_multiset)["arith.addf"] == 1
    assert dict(shadowed.op_multiset)["arith.mulf"] == 1
    assert dict(shadowed.op_multiset)["arith.subf"] == 1


def test_sibling_reduce_regions_may_reuse_block_arguments_and_locals() -> None:
    reused = fingerprint_ttir(SIBLING_REDUCE_BLOCK_ARG_REUSE_TTIR)
    renamed = fingerprint_ttir(RENAMED_SIBLING_REDUCE_BLOCK_ARG_TTIR)

    assert reused == renamed
    assert len(reused.reductions) == 2
    assert dict(reused.op_multiset)["tt.reduce"] == 2


def test_value_defined_in_one_sibling_region_is_not_visible_in_another() -> None:
    malformed = SIBLING_REGION_SSA_REUSE_TTIR.replace(
        "      %value = arith.addf %value, %one : f32\n      scf.yield\n    }\n"
        "    scf.for %iv = %c0 to %n step %c1 : i32 {\n"
        "      %value = arith.addf %value, %one : f32",
        "      %first_only = arith.addf %value, %one : f32\n      scf.yield\n    }\n"
        "    scf.for %iv = %c0 to %n step %c1 : i32 {\n"
        "      %value = arith.addf %first_only, %one : f32",
        1,
    )

    with pytest.raises(TTIRParseError, match="unresolved SSA value .*first_only"):
        fingerprint_ttir(malformed)


def test_single_result_pure_scf_if_is_observed() -> None:
    value_if = r'''
module {
  tt.func public @branch(%x: f32, %cond: i1) -> f32 {
    %result = scf.if %cond -> (f32) {
      scf.yield %x : f32
    } else {
      scf.yield %x : f32
    }
    tt.return %result : f32
  }
}
'''

    fingerprint = fingerprint_ttir(value_if)
    assert len(fingerprint.structured_ifs) == 1
    conditional = fingerprint.structured_ifs[0]
    assert conditional.result_type == "f32"
    assert conditional.positive_arm.yield_type == "f32"
    assert conditional.negative_arm.yield_type == "f32"
    assert conditional.positive_arm.terminal == "scf.yield:value"
    assert conditional.negative_arm.terminal == "scf.yield:value"


VALUE_SCF_IF = r'''
module {
  tt.func public @value_if(
      %x: f32, %y: f32, %flag0: i1, %flag1: i1) -> f32 {
    %result = scf.if %flag0 -> (f32) {
      %positive = arith.addf %x, %y : f32
      scf.yield %positive : f32
    } else {
      scf.yield %x : f32
    }
    %answer = arith.mulf %result, %y : f32
    tt.return %answer : f32
  }
}
'''


TWO_VALUE_SCF_IFS = r'''
module {
  tt.func public @two_value_ifs(
      %x: f32, %y: f32, %flag0: i1, %flag1: i1) -> f32 {
    %first = scf.if %flag0 -> (f32) {
      %positive0 = arith.addf %x, %y : f32
      scf.yield %positive0 : f32
    } else {
      scf.yield %x : f32
    }
    %second = scf.if %flag1 -> (f32) {
      %positive1 = arith.mulf %first, %y : f32
      scf.yield %positive1 : f32
    } else {
      scf.yield %first : f32
    }
    tt.return %second : f32
  }
}
'''


def test_value_scf_if_records_arm_dataflow_yields_and_result_use() -> None:
    fingerprint = fingerprint_ttir(VALUE_SCF_IF)

    assert fingerprint.schema_version == "triton-ttir-3.6-gate1-v10"
    assert dict(fingerprint.op_multiset)["control.scf.if"] == 1
    conditional = fingerprint.structured_ifs[0]
    assert conditional.ordinal == 0
    assert conditional.result_type == "f32"
    assert conditional.continuation == "parent"
    assert conditional.positive_arm.yielded_value is not None
    assert conditional.negative_arm.yielded_value == "arg[0]:f32"
    assert len(conditional.positive_arm.members) == 1
    assert not conditional.negative_arm.members
    roles = {
        role
        for _, _, role in fingerprint.dataflow_edges
        if role in {"condition", "positive_yield", "negative_yield"}
    }
    assert {"condition", "positive_yield", "negative_yield"}.issubset(roles)
    assert any(
        source.startswith("scf.if[")
        and target.startswith("arith.mulf[")
        for source, target, _ in fingerprint.dataflow_edges
    )


def test_value_scf_if_alpha_renaming_is_presentation_only() -> None:
    renamed = VALUE_SCF_IF.replace("%result", "%chosen").replace(
        "%positive", "%arm_value"
    )

    assert compare_ttir(VALUE_SCF_IF, renamed).hard_pass


@pytest.mark.parametrize(
    "mutated",
    [
        VALUE_SCF_IF.replace("arith.addf %x, %y", "arith.subf %x, %y"),
        VALUE_SCF_IF.replace(
            "scf.yield %positive : f32", "scf.yield %x : f32", 1
        ),
        VALUE_SCF_IF.replace("scf.if %flag0", "scf.if %flag1", 1),
        VALUE_SCF_IF.replace(
            "    %result = scf.if %flag0 -> (f32) {\n"
            "      %positive = arith.addf %x, %y : f32\n"
            "      scf.yield %positive : f32\n"
            "    } else {\n"
            "      scf.yield %x : f32\n"
            "    }",
            "    %result = arith.select %flag0, %x, %x : i1, f32",
        ),
    ],
)
def test_value_scf_if_predicate_arm_graph_yield_and_control_kind_are_hard(
    mutated: str,
) -> None:
    assert not compare_ttir(VALUE_SCF_IF, mutated).hard_pass


def test_multiple_value_scf_ifs_preserve_same_parent_ordinals() -> None:
    fingerprint = fingerprint_ttir(TWO_VALUE_SCF_IFS)

    assert [item.ordinal for item in fingerprint.structured_ifs] == [0, 1]
    assert all(item.result_type == "f32" for item in fingerprint.structured_ifs)


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (
            VALUE_SCF_IF.replace(
                "%result = scf.if %flag0 -> (f32)",
                "%result:2 = scf.if %flag0 -> (f32, f32)",
                1,
            ),
            "requires exactly one SSA result",
        ),
        (
            VALUE_SCF_IF.replace("    } else {\n", "    }\n", 1),
            "requires explicit positive and negative arms",
        ),
        (
            VALUE_SCF_IF.replace(
                "      scf.yield %positive : f32\n    } else {",
                "    } else {",
                1,
            ),
            "each value-producing scf.if arm must end",
        ),
        (
            VALUE_SCF_IF.replace(
                "      scf.yield %x : f32\n    }",
                "    }",
                1,
            ),
            "each value-producing scf.if arm must end",
        ),
        (
            VALUE_SCF_IF.replace(
                "scf.yield %positive : f32",
                "scf.yield %positive, %x : f32, f32",
                1,
            ),
            "exactly one typed yielded SSA value",
        ),
        (
            VALUE_SCF_IF.replace(
                "scf.yield %positive : f32",
                "scf.yield %positive : f64",
                1,
            ),
            "positive yield type must match",
        ),
        (
            VALUE_SCF_IF.replace(
                "scf.yield %positive : f32",
                "scf.yield %flag0 : f32",
                1,
            ),
            "positive yielded SSA type must match",
        ),
        (
            VALUE_SCF_IF.replace(
                "      %positive = arith.addf %x, %y : f32",
                "      %positive = future.unknown %x : f32",
                1,
            ),
            "unclassified operation future.unknown",
        ),
        (
            VALUE_SCF_IF.replace(
                "      %positive = arith.addf %x, %y : f32",
                "      %c0 = arith.constant 0 : i32\n"
                "      %c1 = arith.constant 1 : i32\n"
                "      scf.for %iv = %c0 to %c1 step %c1 : i32 {\n"
                "        scf.yield\n"
                "      }\n"
                "      %positive = arith.addf %x, %y : f32",
                1,
            ),
            "cannot contain a nested region",
        ),
        (
            VALUE_SCF_IF.replace(
                "      %positive = arith.addf %x, %y : f32",
                "      ^bb0:\n"
                "      %positive = arith.addf %x, %y : f32",
                1,
            ),
            "implicit single block",
        ),
        (
            VALUE_SCF_IF.replace(
                "%x: f32, %y: f32, %flag0: i1",
                "%x: f32, %y: f32, %flag0: tensor<1xi1>",
                1,
            ),
            "predicate must be a scalar i1",
        ),
        (
            VALUE_SCF_IF.replace(
                "      %positive = arith.addf %x, %y : f32",
                "      %positive = arith.addf %result, %y : f32",
                1,
            ),
            "result does not dominate",
        ),
    ],
)
def test_malformed_value_scf_if_fails_closed(
    malformed: str,
    message: str,
) -> None:
    with pytest.raises(TTIRParseError, match=message):
        fingerprint_ttir(malformed)


def test_value_scf_if_rejects_arm_side_effects() -> None:
    effectful = VALUE_SCF_IF.replace(
        "%x: f32, %y: f32, %flag0: i1, %flag1: i1",
        "%x: f32, %y: f32, %flag0: i1, %flag1: i1, "
        "%out: !tt.ptr<f32>",
        1,
    ).replace(
        "      scf.yield %positive : f32",
        "      tt.store %out, %positive : !tt.ptr<f32>\n"
        "      scf.yield %positive : f32",
        1,
    )

    with pytest.raises(TTIRParseError, match="must be side-effect free.*tt.store"):
        fingerprint_ttir(effectful)


EFFECT_ONLY_SCF_IF = r'''
module {
  tt.func public @continuing_if(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %flag0: i1, %flag1: i1) {
    scf.if %flag0 {
      %one = arith.constant 1.000000e+00 : f32
      %two = arith.constant 2.000000e+00 : f32
      %value = tt.load %input : !tt.ptr<f32>
      %scaled = arith.mulf %value, %two : f32
      %sum = arith.addf %scaled, %one : f32
      tt.store %output, %sum : !tt.ptr<f32>
      tt.store %output, %value : !tt.ptr<f32>
    }
    %after = arith.constant 3.000000e+00 : f32
    tt.store %output, %after : !tt.ptr<f32>
    tt.return
  }
}
'''


TWO_EFFECT_ONLY_SCF_IFS = r'''
module {
  tt.func public @two_continuing_ifs(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %flag0: i1, %flag1: i1) {
    scf.if %flag0 {
      %first = tt.load %input : !tt.ptr<f32>
      tt.store %output, %first : !tt.ptr<f32>
    }
    scf.if %flag1 {
      %second = tt.load %input : !tt.ptr<f32>
      tt.store %output, %second : !tt.ptr<f32>
    }
    tt.return
  }
}
'''


SCF_IF_IN_LOOP = r'''
module {
  tt.func public @loop_continuing_if(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %flag: i1, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %n step %c1 : i32 {
      scf.if %flag {
        %value = tt.load %input : !tt.ptr<f32>
        tt.store %output, %value : !tt.ptr<f32>
      }
      scf.yield
    }
    tt.return
  }
}
'''


SCF_IF_IN_TERMINAL_FORK = r'''
module {
  tt.func public @fork_then_continuing_if(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %terminal_flag: i1, %continuing_flag: i1) {
    cf.cond_br %terminal_flag, ^bb_true, ^bb_false
  ^bb_true:
    tt.return
  ^bb_false:
    scf.if %continuing_flag {
      %value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %value : !tt.ptr<f32>
    }
    scf.if %continuing_flag {
      %again = tt.load %input : !tt.ptr<f32>
      tt.store %output, %again : !tt.ptr<f32>
    }
    tt.return
  }
}
'''


TERMINAL_CONTROL_WITH_MATCHING_ABI = r'''
module {
  tt.func public @terminal_only(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %flag0: i1, %flag1: i1) {
    cf.cond_br %flag0, ^bb_true, ^bb_false
  ^bb_true:
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  ^bb_false:
    tt.return
  }
}
'''


def test_effect_only_scf_if_is_a_strong_continuing_control_fingerprint() -> None:
    fingerprint = fingerprint_ttir(EFFECT_ONLY_SCF_IF)

    assert fingerprint.schema_version == "triton-ttir-3.6-gate1-v10"
    assert dict(fingerprint.op_multiset)["control.scf.if"] == 1
    assert len(fingerprint.structured_ifs) == 1
    conditional = fingerprint.structured_ifs[0]
    assert conditional.ordinal == 0
    assert conditional.continuation == "parent"
    assert conditional.enclosing_terminal_arm is None
    assert conditional.positive_arm.role == "positive"
    assert conditional.negative_arm.role == "negative"
    assert conditional.positive_arm.terminal == "scf.yield:void"
    assert conditional.negative_arm.terminal == "scf.yield:void"
    assert not conditional.negative_arm.members
    assert [
        item.split(":", 1)[0]
        for item in conditional.positive_arm.effect_order
    ] == ["tt.load", "tt.store", "tt.store"]


def test_implicit_and_explicit_empty_scf_if_yields_are_identical() -> None:
    explicit = EFFECT_ONLY_SCF_IF.replace(
        "      tt.store %output, %value : !tt.ptr<f32>\n    }",
        "      tt.store %output, %value : !tt.ptr<f32>\n"
        "      scf.yield\n"
        "    } else {\n"
        "      scf.yield\n"
        "    }",
    )

    assert compare_ttir(EFFECT_ONLY_SCF_IF, explicit).hard_pass


def test_scf_if_sorts_pure_members_but_preserves_arm_effect_order() -> None:
    pure_reordered = EFFECT_ONLY_SCF_IF.replace(
        "      %one = arith.constant 1.000000e+00 : f32\n"
        "      %two = arith.constant 2.000000e+00 : f32",
        "      %two = arith.constant 2.000000e+00 : f32\n"
        "      %one = arith.constant 1.000000e+00 : f32",
    )
    effects_reordered = EFFECT_ONLY_SCF_IF.replace(
        "      tt.store %output, %sum : !tt.ptr<f32>\n"
        "      tt.store %output, %value : !tt.ptr<f32>",
        "      tt.store %output, %value : !tt.ptr<f32>\n"
        "      tt.store %output, %sum : !tt.ptr<f32>",
    )

    assert compare_ttir(EFFECT_ONLY_SCF_IF, pure_reordered).hard_pass
    assert not compare_ttir(EFFECT_ONLY_SCF_IF, effects_reordered).hard_pass


def test_scf_if_polarity_and_predicate_are_hard_evidence() -> None:
    moved_to_negative = EFFECT_ONLY_SCF_IF.replace(
        "    scf.if %flag0 {",
        "    scf.if %flag0 {\n"
        "      scf.yield\n"
        "    } else {",
        1,
    )
    changed_predicate = EFFECT_ONLY_SCF_IF.replace(
        "scf.if %flag0", "scf.if %flag1", 1
    )

    assert not compare_ttir(EFFECT_ONLY_SCF_IF, moved_to_negative).hard_pass
    assert not compare_ttir(EFFECT_ONLY_SCF_IF, changed_predicate).hard_pass


def test_multiple_scf_ifs_preserve_parent_local_sequence_ordinals() -> None:
    fingerprint = fingerprint_ttir(TWO_EFFECT_ONLY_SCF_IFS)

    assert [item.ordinal for item in fingerprint.structured_ifs] == [0, 1]
    swapped = TWO_EFFECT_ONLY_SCF_IFS.replace(
        "scf.if %flag0", "scf.if %temporary", 1
    ).replace(
        "scf.if %flag1", "scf.if %flag0", 1
    ).replace(
        "scf.if %temporary", "scf.if %flag1", 1
    )
    assert not compare_ttir(TWO_EFFECT_ONLY_SCF_IFS, swapped).hard_pass


def test_scf_if_preserves_structured_and_terminal_arm_parent_contexts() -> None:
    in_loop = fingerprint_ttir(SCF_IF_IN_LOOP)
    in_terminal_arm = fingerprint_ttir(SCF_IF_IN_TERMINAL_FORK)

    assert len(in_loop.structured_ifs[0].region_context) == 1
    assert in_loop.structured_ifs[0].region_context[0].startswith("scf.for[")
    assert dict(in_loop.loops[0].body_op_multiset) == {
        "control.scf.if": 1,
    }
    assert in_loop.loops[0].body_effects == (("control.scf.if", 1),)
    assert [item.ordinal for item in in_terminal_arm.structured_ifs] == [0, 1]
    assert all(
        item.enclosing_terminal_arm == "false"
        for item in in_terminal_arm.structured_ifs
    )
    assert len(in_terminal_arm.terminal_forks) == 1
    assert any(
        item.startswith("region[scf.if:")
        for item in in_terminal_arm.terminal_forks[0].false_arm.structured_regions
    )


def test_continuing_scf_if_cannot_equal_a_terminal_fork() -> None:
    continuing = r'''
module {
  tt.func public @continuing(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>,
      %flag0: i1, %flag1: i1) {
    scf.if %flag0 {
      %value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %value : !tt.ptr<f32>
    }
    tt.return
  }
}
'''
    continuing_fingerprint = fingerprint_ttir(continuing)
    terminal_fingerprint = fingerprint_ttir(TERMINAL_CONTROL_WITH_MATCHING_ABI)

    assert continuing_fingerprint.structured_ifs
    assert not continuing_fingerprint.terminal_forks
    assert terminal_fingerprint.terminal_forks
    assert not terminal_fingerprint.structured_ifs
    assert not compare_fingerprints(
        continuing_fingerprint, terminal_fingerprint
    ).hard_pass


def test_scf_if_arm_scopes_allow_same_local_ssa_spelling() -> None:
    both_arms = r'''
module {
  tt.func public @two_arms(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flag: i1) {
    scf.if %flag {
      %value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %value : !tt.ptr<f32>
      scf.yield
    } else {
      %value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %value : !tt.ptr<f32>
      scf.yield
    }
    tt.return
  }
}
'''
    fingerprint = fingerprint_ttir(both_arms)

    assert len(fingerprint.structured_ifs[0].positive_arm.effect_order) == 2
    assert len(fingerprint.structured_ifs[0].negative_arm.effect_order) == 2


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (
            EFFECT_ONLY_SCF_IF.replace(
                "%flag0: i1", "%flag0: tensor<1xi1>", 1
            ),
            "predicate must be a scalar i1",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "    scf.if %flag0 {",
                "    scf.if %flag0 {\n      ^bb0:",
                1,
            ),
            "implicit single block",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "    scf.if %flag0 {",
                "    scf.if %flag0 {\n      scf.if %flag1 {\n      }",
                1,
            ),
            "nested scf.if is unsupported",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "      %one = arith.constant 1.000000e+00 : f32",
                "      scf.yield %flag0 : i1\n"
                "      %one = arith.constant 1.000000e+00 : f32",
                1,
            ),
            "zero-operand scf.yield",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "      %one = arith.constant 1.000000e+00 : f32",
                "      scf.yield\n"
                "      %one = arith.constant 1.000000e+00 : f32",
                1,
            ),
            "operation after scf.if positive arm scf.yield",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "    scf.if %flag0 {",
                "    scf.if %flag0 attributes {test = true} {",
                1,
            ),
            "resultless scf.if must use exactly one scalar predicate",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "    %after = arith.constant 3.000000e+00 : f32",
                "    %after = arith.addf %value, %value : f32",
                1,
            ),
            "unresolved SSA value .*value",
        ),
        (
            EFFECT_ONLY_SCF_IF.replace(
                "      %one = arith.constant 1.000000e+00 : f32",
                "      scf.while %flag1 {\n      }\n"
                "      %one = arith.constant 1.000000e+00 : f32",
                1,
            ),
            "unsupported region-bearing TTIR operation 'scf.while'",
        ),
    ],
)
def test_malformed_or_out_of_profile_scf_if_fails_closed(
    malformed: str,
    message: str,
) -> None:
    with pytest.raises(TTIRParseError, match=message):
        fingerprint_ttir(malformed)


def test_duplicate_ssa_definition_in_the_same_region_is_still_rejected() -> None:
    malformed = SIBLING_REGION_SSA_REUSE_TTIR.replace(
        "    %one = arith.constant 1.000000e+00 : f32",
        "    %one = arith.constant 1.000000e+00 : f32\n"
        "    %one = arith.constant 2.000000e+00 : f32",
    )

    with pytest.raises(TTIRParseError, match="duplicate SSA definition %one"):
        fingerprint_ttir(malformed)


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


def test_exact_libdevice_rsqrt_extern_is_normalized() -> None:
    rsqrt = DATAFLOW_TTIR.replace(
        "    %answer = arith.subf %left, %right : f32",
        "    %answer = math.rsqrt %left : f32",
    )
    libdevice = DATAFLOW_TTIR.replace(
        "    %answer = arith.subf %left, %right : f32",
        '    %answer = tt.extern_elementwise %left '
        '{libname = "", libpath = "", pure = true, '
        'symbol = "__nv_rsqrtf"} : (f32) -> f32',
    )
    other_extern = libdevice.replace("__nv_rsqrtf", "__nv_sinf")
    wrong_input_type = libdevice.replace(": (f32) -> f32", ": (f64) -> f32")

    assert compare_ttir(rsqrt, libdevice).hard_pass
    assert not compare_ttir(rsqrt, other_extern).hard_pass
    assert not compare_ttir(rsqrt, wrong_input_type).hard_pass


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


OUTLINED_VOID_HELPER_IN_LOOP = r'''
module {
  tt.func public @outlined(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %n step %c1 : i32 {
      tt.call @copy_one(%x, %out, %iv) : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
      scf.yield
    }
    tt.return
  }
  tt.func private @copy_one(%source: !tt.ptr<f32>, %destination: !tt.ptr<f32>, %index: i32) {
    %source_at = tt.addptr %source, %index : !tt.ptr<f32>, i32
    %value = tt.load %source_at : !tt.ptr<f32>
    %destination_at = tt.addptr %destination, %index : !tt.ptr<f32>, i32
    tt.store %destination_at, %value : !tt.ptr<f32>
    tt.return
  }
}
'''


INLINED_VOID_HELPER_IN_LOOP = r'''
module {
  tt.func public @inlined(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %lower = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    scf.for %iteration = %lower to %arg2 step %step : i32 {
      %read_at = tt.addptr %arg0, %iteration : !tt.ptr<f32>, i32
      %loaded = tt.load %read_at : !tt.ptr<f32>
      %write_at = tt.addptr %arg1, %iteration : !tt.ptr<f32>, i32
      tt.store %write_at, %loaded : !tt.ptr<f32>
      scf.yield
    }
    tt.return
  }
}
'''


OUTLINED_NESTED_VALUE_HELPERS = r'''
module {
  tt.func public @outlined_value(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %value = tt.load %x : !tt.ptr<f32>
    %answer = tt.call @add_after_scale(%value) : (f32) -> f32
    tt.store %out, %answer : !tt.ptr<f32>
    tt.return
  }
  tt.func private @add_after_scale(%value: f32) -> f32 {
    %scaled = tt.call @scale(%value) : (f32) -> f32
    %one = arith.constant 1.000000e+00 : f32
    %answer = arith.addf %scaled, %one : f32
    tt.return %answer : f32
  }
  tt.func private @scale(%value: f32) -> f32 {
    %two = arith.constant 2.000000e+00 : f32
    %answer = arith.mulf %value, %two : f32
    tt.return %answer : f32
  }
}
'''


INLINED_NESTED_VALUE_HELPERS = r'''
module {
  tt.func public @inlined_value(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %input = tt.load %arg0 : !tt.ptr<f32>
    %factor = arith.constant 2.000000e+00 : f32
    %scaled = arith.mulf %input, %factor : f32
    %unit = arith.constant 1.000000e+00 : f32
    %result = arith.addf %scaled, %unit : f32
    tt.store %arg1, %result : !tt.ptr<f32>
    tt.return
  }
}
'''


def test_private_void_helper_and_equivalent_inline_loop_hard_pass() -> None:
    verdict = compare_ttir(
        OUTLINED_VOID_HELPER_IN_LOOP,
        INLINED_VOID_HELPER_IN_LOOP,
    )

    assert verdict.hard_pass, verdict.to_dict()
    fingerprint = fingerprint_ttir(OUTLINED_VOID_HELPER_IN_LOOP)
    assert fingerprint.schema_version == "triton-ttir-3.6-gate1-v10"
    assert "tt.call" not in dict(fingerprint.op_multiset)
    assert dict(fingerprint.loops[0].body_op_multiset) == {
        "tt.addptr": 2,
        "tt.load": 1,
        "tt.store": 1,
    }
    assert fingerprint.loops[0].body_effects == (("read", 1), ("write", 1))


def test_nested_private_ssa_returns_match_equivalent_inline_graph() -> None:
    verdict = compare_ttir(
        OUTLINED_NESTED_VALUE_HELPERS,
        INLINED_NESTED_VALUE_HELPERS,
    )

    assert verdict.hard_pass, verdict.to_dict()
    assert "tt.call" not in dict(
        fingerprint_ttir(OUTLINED_NESTED_VALUE_HELPERS).op_multiset
    )


def test_reachable_private_helper_body_change_changes_hard_fingerprint() -> None:
    changed = OUTLINED_NESTED_VALUE_HELPERS.replace(
        "%answer = arith.mulf %value, %two : f32",
        "%answer = arith.subf %value, %two : f32",
    )

    assert not compare_ttir(OUTLINED_NESTED_VALUE_HELPERS, changed).hard_pass


def test_unreachable_private_helpers_do_not_affect_fingerprint() -> None:
    unused = r'''
  tt.func private @never_called(%value: f32) -> f32 {
    %huge = arith.constant 999.000000e+00 : f32
    %changed = arith.mulf %value, %huge : f32
    tt.return %changed : f32
  }
'''
    with_unused = INLINED_NESTED_VALUE_HELPERS.replace("\n}", unused + "\n}")

    assert fingerprint_ttir(with_unused) == fingerprint_ttir(
        INLINED_NESTED_VALUE_HELPERS
    )


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (
            OUTLINED_NESTED_VALUE_HELPERS.replace(
                "@add_after_scale(%value)", "@missing(%value)", 1
            ),
            "unknown callee @missing",
        ),
        (
            OUTLINED_NESTED_VALUE_HELPERS.replace(
                "%scaled = tt.call @scale(%value) : (f32) -> f32",
                "%scaled = tt.call @add_after_scale(%value) : (f32) -> f32",
            ),
            "recursive tt.call graph",
        ),
        (
            OUTLINED_NESTED_VALUE_HELPERS.replace(
                "@add_after_scale(%value) : (f32) -> f32",
                "@add_after_scale(%value) : (i32) -> f32",
                1,
            ),
            "actual SSA type f32.*explicit call type i32",
        ),
        (
            OUTLINED_NESTED_VALUE_HELPERS.replace(
                "@add_after_scale(%value) : (f32) -> f32",
                "@add_after_scale() : () -> f32",
                1,
            ),
            "argument arity does not match",
        ),
        (
            OUTLINED_NESTED_VALUE_HELPERS.replace(
                "@add_after_scale(%value) : (f32) -> f32",
                "@add_after_scale(%value) : (f32) -> i32",
                1,
            ),
            "result types do not match",
        ),
        (
            OUTLINED_NESTED_VALUE_HELPERS.replace(
                "@add_after_scale(%value) : (f32) -> f32",
                "@add_after_scale(%value) : (f32) -> ()",
                1,
            ),
            "result SSA/type arity mismatch",
        ),
    ],
)
def test_reachable_call_contract_errors_fail_closed(
    malformed: str,
    message: str,
) -> None:
    with pytest.raises(TTIRParseError, match=message):
        fingerprint_ttir(malformed)


def test_reachable_helper_with_pure_value_if_is_expanded() -> None:
    with_value_if = OUTLINED_NESTED_VALUE_HELPERS.replace(
        "    %two = arith.constant 2.000000e+00 : f32\n"
        "    %answer = arith.mulf %value, %two : f32",
        "    %condition = arith.constant true : i1\n"
        "    %answer = scf.if %condition -> (f32) {\n"
        "      scf.yield %value : f32\n"
        "    } else {\n"
        "      scf.yield %value : f32\n"
        "    }",
    )

    fingerprint = fingerprint_ttir(with_value_if)

    assert "tt.call" not in dict(fingerprint.op_multiset)
    assert dict(fingerprint.op_multiset)["control.scf.if"] == 1
    assert len(fingerprint.structured_ifs) == 1
    assert fingerprint.structured_ifs[0].result_type == "f32"


TERMINAL_BINARY_FORK = r'''
module {
  tt.func public @terminal_fork(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flag: i32) {
    %zero_i32 = arith.constant 0 : i32
    %condition = arith.cmpi eq, %flag, %zero_i32 : i32
    cf.cond_br %condition, ^bb_true, ^bb_false
  ^bb_true:
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  ^bb_false:
    %zero_f32 = arith.constant 0.000000e+00 : f32
    tt.store %output, %zero_f32 : !tt.ptr<f32>
    tt.return
  }
}
'''


OUTLINED_TERMINAL_FORK_IN_LOOP = r'''
module {
  tt.func public @caller(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %n step %c1 : i32 {
      tt.call @terminal_helper(%input, %output, %iv) : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
      scf.yield
    }
    tt.return
  }
  tt.func private @terminal_helper(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %iv: i32) {
    %c0 = arith.constant 0 : i32
    %condition = arith.cmpi eq, %iv, %c0 : i32
    cf.cond_br %condition, ^bb_empty, ^bb_copy
  ^bb_empty:
    %zero = arith.constant 0.000000e+00 : f32
    tt.store %output, %zero : !tt.ptr<f32>
    tt.return
  ^bb_copy:
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  }
}
'''


OUTLINED_TERMINAL_FORK_AT_ROOT = r'''
module {
  tt.func public @caller(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flag: i32) {
    tt.call @terminal_helper(%input, %output, %flag) : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
    tt.return
  }
  tt.func private @terminal_helper(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flag: i32) {
    %zero_i32 = arith.constant 0 : i32
    %condition = arith.cmpi eq, %flag, %zero_i32 : i32
    cf.cond_br %condition, ^bb_true, ^bb_false
  ^bb_true:
    %value = tt.load %input : !tt.ptr<f32>
    tt.store %output, %value : !tt.ptr<f32>
    tt.return
  ^bb_false:
    %zero_f32 = arith.constant 0.000000e+00 : f32
    tt.store %output, %zero_f32 : !tt.ptr<f32>
    tt.return
  }
}
'''


TERMINAL_FORK_REPEATED_LOAD_CONSUMER = r'''
module {
  tt.func public @repeated_load_consumer(
      %input: !tt.ptr<f32>, %scratch: !tt.ptr<f32>,
      %output: !tt.ptr<f32>, %flag: i1) {
    cf.cond_br %flag, ^bb_true, ^bb_false
  ^bb_true:
    %first = tt.load %input : !tt.ptr<f32>
    %zero = arith.constant 0.000000e+00 : f32
    tt.store %scratch, %zero : !tt.ptr<f32>
    %second = tt.load %input : !tt.ptr<f32>
    tt.store %output, %first : !tt.ptr<f32>
    tt.return
  ^bb_false:
    tt.return
  }
}
'''


TERMINAL_FORK_REPEATED_LOAD_PREDICATE = r'''
module {
  tt.func public @repeated_load_predicate(
      %predicate: !tt.ptr<i1>, %scratch: !tt.ptr<i1>) {
    %first = tt.load %predicate : !tt.ptr<i1>
    %false = arith.constant false : i1
    tt.store %scratch, %false : !tt.ptr<i1>
    %second = tt.load %predicate : !tt.ptr<i1>
    cf.cond_br %first, ^bb_true, ^bb_false
  ^bb_true:
    tt.return
  ^bb_false:
    tt.return
  }
}
'''


OUTLINED_REPEATED_LOAD_RETURN = r'''
module {
  tt.func public @caller(
      %input: !tt.ptr<f32>, %scratch: !tt.ptr<f32>,
      %output: !tt.ptr<f32>) {
    %chosen = tt.call @choose(%input, %scratch) : (!tt.ptr<f32>, !tt.ptr<f32>) -> f32
    tt.store %output, %chosen : !tt.ptr<f32>
    tt.return
  }
  tt.func private @choose(
      %input: !tt.ptr<f32>, %scratch: !tt.ptr<f32>) -> f32 {
    %first = tt.load %input : !tt.ptr<f32>
    %zero = arith.constant 0.000000e+00 : f32
    tt.store %scratch, %zero : !tt.ptr<f32>
    %second = tt.load %input : !tt.ptr<f32>
    tt.return %first : f32
  }
}
'''


REPEATED_ATOMIC_RESULT_CONSUMER = r'''
module {
  tt.func public @repeated_atomic_result(
      %counter: !tt.ptr<f32>, %scratch: !tt.ptr<f32>,
      %output: !tt.ptr<f32>) {
    %one = arith.constant 1.000000e+00 : f32
    %first = tt.atomic_rmw add, %counter, %one : (!tt.ptr<f32>, f32) -> f32
    tt.store %scratch, %one : !tt.ptr<f32>
    %second = tt.atomic_rmw add, %counter, %one : (!tt.ptr<f32>, f32) -> f32
    tt.store %output, %first : !tt.ptr<f32>
    tt.return
  }
}
'''


REPEATED_LOAD_PUBLIC_RETURN = r'''
module {
  tt.func public @repeated_load_return(
      %input: !tt.ptr<f32>, %scratch: !tt.ptr<f32>) -> f32 {
    %first = tt.load %input : !tt.ptr<f32>
    %zero = arith.constant 0.000000e+00 : f32
    tt.store %scratch, %zero : !tt.ptr<f32>
    %second = tt.load %input : !tt.ptr<f32>
    tt.return %first : f32
  }
}
'''


@pytest.mark.parametrize(
    ("source", "replay"),
    [
        (
            TERMINAL_FORK_REPEATED_LOAD_CONSUMER,
            TERMINAL_FORK_REPEATED_LOAD_CONSUMER.replace(
                "tt.store %output, %first", "tt.store %output, %second"
            ),
        ),
        (
            TERMINAL_FORK_REPEATED_LOAD_PREDICATE,
            TERMINAL_FORK_REPEATED_LOAD_PREDICATE.replace(
                "cf.cond_br %first", "cf.cond_br %second"
            ),
        ),
        (
            OUTLINED_REPEATED_LOAD_RETURN,
            OUTLINED_REPEATED_LOAD_RETURN.replace(
                "tt.return %first : f32", "tt.return %second : f32"
            ),
        ),
        (
            REPEATED_ATOMIC_RESULT_CONSUMER,
            REPEATED_ATOMIC_RESULT_CONSUMER.replace(
                "tt.store %output, %first", "tt.store %output, %second"
            ),
        ),
        (
            REPEATED_LOAD_PUBLIC_RETURN,
            REPEATED_LOAD_PUBLIC_RETURN.replace(
                "tt.return %first : f32", "tt.return %second : f32"
            ),
        ),
    ],
    ids=[
        "terminal-arm-consumer",
        "entry-predicate",
        "private-helper-return",
        "atomic-result-consumer",
        "public-return-edge",
    ],
)
def test_effectful_producer_occurrences_do_not_content_hash_collapse(
    source: str, replay: str
) -> None:
    assert not compare_ttir(source, replay).hard_pass


PURE_IDENTICAL_PRODUCERS = r'''
module {
  tt.func public @pure_identical(
      %input: f32, %output: !tt.ptr<f32>) {
    %first = arith.addf %input, %input : f32
    %second = arith.addf %input, %input : f32
    tt.store %output, %first : !tt.ptr<f32>
    tt.return
  }
}
'''


INLINE_REPEATED_LOAD_RETURN = r'''
module {
  tt.func public @caller(
      %input: !tt.ptr<f32>, %scratch: !tt.ptr<f32>,
      %output: !tt.ptr<f32>) {
    %first = tt.load %input : !tt.ptr<f32>
    %zero = arith.constant 0.000000e+00 : f32
    tt.store %scratch, %zero : !tt.ptr<f32>
    %second = tt.load %input : !tt.ptr<f32>
    tt.store %output, %first : !tt.ptr<f32>
    tt.return
  }
}
'''


REPEATED_EFFECTFUL_REGIONS = r'''
module {
  tt.func public @repeated_effectful_regions(
      %input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %first_iv = %c0 to %n step %c1 : i32 {
      %first_value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %first_value : !tt.ptr<f32>
      scf.yield
    }
    scf.for %second_iv = %c0 to %n step %c1 : i32 {
      %second_value = tt.load %input : !tt.ptr<f32>
      tt.store %output, %second_value : !tt.ptr<f32>
      scf.yield
    }
    tt.return
  }
}
'''


def test_pure_identical_producers_remain_content_equivalent() -> None:
    replay = PURE_IDENTICAL_PRODUCERS.replace(
        "tt.store %output, %first", "tt.store %output, %second"
    )

    verdict = compare_ttir(PURE_IDENTICAL_PRODUCERS, replay)
    assert verdict.hard_pass, verdict.to_dict()


def test_effect_occurrences_are_assigned_after_interprocedural_expansion() -> None:
    verdict = compare_ttir(
        OUTLINED_REPEATED_LOAD_RETURN,
        INLINE_REPEATED_LOAD_RETURN,
    )

    assert verdict.hard_pass, verdict.to_dict()


def test_repeated_effectful_region_instances_have_distinct_event_paths() -> None:
    fingerprint = fingerprint_ttir(REPEATED_EFFECTFUL_REGIONS)
    root_children = {
        target
        for source, target, role in fingerprint.dataflow_edges
        if source == "region[root]" and role == "region.child"
    }

    assert len(root_children) == 2
    assert all("region[scf.for:" in child for child in root_children)
    event_producers = {
        source
        for source, _, _ in fingerprint.dataflow_edges
        if source.startswith("tt.load[")
    }
    assert len(event_producers) == 2
    assert all("@event[root/" in producer for producer in event_producers)


def test_closed_terminal_binary_fork_preserves_roles_and_control_edges() -> None:
    fingerprint = fingerprint_ttir(TERMINAL_BINARY_FORK)

    assert fingerprint.schema_version == "triton-ttir-3.6-gate1-v10"
    assert dict(fingerprint.op_multiset)["control.conditional"] == 1
    assert len(fingerprint.terminal_forks) == 1
    fork = fingerprint.terminal_forks[0]
    assert fork.true_arm.role == "true"
    assert fork.false_arm.role == "false"
    assert fork.true_arm.terminal == "tt.return:void"
    assert fork.false_arm.terminal == "tt.return:void"
    assert [item.split("[", 1)[0] for item in fork.true_arm.effect_order] == [
        "tt.load:tt.load",
        "tt.store:tt.store",
    ]
    assert [item.split("[", 1)[0] for item in fork.false_arm.effect_order] == [
        "tt.store:tt.store",
    ]
    roles = {
        edge[2]
        for edge in fingerprint.dataflow_edges
        if "control.conditional" in edge[1]
    }
    assert {"condition", "true", "false"}.issubset(roles)


def test_terminal_fork_physical_block_order_is_presentation_only() -> None:
    before, true_block = TERMINAL_BINARY_FORK.split("  ^bb_true:", 1)
    true_body, false_body = true_block.split("  ^bb_false:", 1)
    reordered = (
        before
        + "  ^bb_false:"
        + false_body.rsplit("  }", 1)[0]
        + "  ^bb_true:"
        + true_body
        + "  }\n}\n"
    )

    assert compare_ttir(TERMINAL_BINARY_FORK, reordered).hard_pass


def test_terminal_fork_arm_sorts_pure_members_but_orders_effects() -> None:
    two_values = TERMINAL_BINARY_FORK.replace(
        "    %value = tt.load %input : !tt.ptr<f32>\n"
        "    tt.store %output, %value : !tt.ptr<f32>",
        "    %one = arith.constant 1.000000e+00 : f32\n"
        "    %two = arith.constant 2.000000e+00 : f32\n"
        "    tt.store %output, %one : !tt.ptr<f32>\n"
        "    tt.store %output, %two : !tt.ptr<f32>",
    )
    pure_reordered = two_values.replace(
        "    %one = arith.constant 1.000000e+00 : f32\n"
        "    %two = arith.constant 2.000000e+00 : f32",
        "    %two = arith.constant 2.000000e+00 : f32\n"
        "    %one = arith.constant 1.000000e+00 : f32",
    )
    effects_reordered = two_values.replace(
        "    tt.store %output, %one : !tt.ptr<f32>\n"
        "    tt.store %output, %two : !tt.ptr<f32>",
        "    tt.store %output, %two : !tt.ptr<f32>\n"
        "    tt.store %output, %one : !tt.ptr<f32>",
    )

    assert compare_ttir(two_values, pure_reordered).hard_pass
    assert not compare_ttir(two_values, effects_reordered).hard_pass


@pytest.mark.parametrize(
    "mutated",
    [
        TERMINAL_BINARY_FORK.replace("arith.cmpi eq", "arith.cmpi ne"),
        TERMINAL_BINARY_FORK.replace(
            "cf.cond_br %condition, ^bb_true, ^bb_false",
            "cf.cond_br %condition, ^bb_false, ^bb_true",
        ),
        TERMINAL_BINARY_FORK.replace(
            "    %value = tt.load %input : !tt.ptr<f32>\n"
            "    tt.store %output, %value : !tt.ptr<f32>",
            "    %value = tt.load %input : !tt.ptr<f32>\n"
            "    %again = tt.load %input : !tt.ptr<f32>\n"
            "    tt.store %output, %again : !tt.ptr<f32>",
        ),
        TERMINAL_BINARY_FORK.replace(
            "    %value = tt.load %input : !tt.ptr<f32>\n"
            "    tt.store %output, %value : !tt.ptr<f32>",
            "    %value = tt.load %input : !tt.ptr<f32>\n"
            "    tt.store %output, %value : !tt.ptr<f32>\n"
            "    tt.store %output, %value : !tt.ptr<f32>",
        ),
    ],
)
def test_terminal_fork_predicate_role_member_and_effect_mutations_fail_hard(
    mutated: str,
) -> None:
    assert not compare_ttir(TERMINAL_BINARY_FORK, mutated).hard_pass


def test_outlined_terminal_fork_inherits_call_site_structured_context() -> None:
    fingerprint = fingerprint_ttir(OUTLINED_TERMINAL_FORK_IN_LOOP)

    assert "tt.call" not in dict(fingerprint.op_multiset)
    assert len(fingerprint.terminal_forks) == 1
    assert len(fingerprint.terminal_forks[0].region_context) == 1
    assert fingerprint.terminal_forks[0].region_context[0].startswith("scf.for[")
    assert dict(fingerprint.loops[0].body_op_multiset) == {
        "arith.constant": 1,
        "arith.cmpi": 1,
        "control.conditional": 1,
    }
    assert fingerprint.loops[0].body_effects == (
        ("control.conditional", 1),
    )


def test_root_cfg_helper_expands_to_same_synthetic_control_as_inline_fork() -> None:
    verdict = compare_ttir(
        TERMINAL_BINARY_FORK,
        OUTLINED_TERMINAL_FORK_AT_ROOT,
    )

    assert verdict.hard_pass, verdict.to_dict()
    assert "tt.call" not in dict(
        fingerprint_ttir(OUTLINED_TERMINAL_FORK_AT_ROOT).op_multiset
    )


@pytest.mark.parametrize(
    ("malformed", "message"),
    [
        (
            TERMINAL_BINARY_FORK.replace(
                "tt.func public @terminal_fork(\n",
                "tt.func public @terminal_fork(\n",
            ).replace(
                "%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flag: i32) {",
                "%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flag: i32) -> i32 {",
            ),
            "requires a void tt.func",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "cf.cond_br %condition, ^bb_true, ^bb_false",
                "tt.return\n"
                "    cf.cond_br %condition, ^bb_true, ^bb_false",
            ),
            "permits tt.return only.*arm",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "cf.cond_br %condition, ^bb_true, ^bb_false",
                "cf.cond_br %condition, ^bb_true(%flag : i32), ^bb_false",
            ),
            "operand-free cf.cond_br",
        ),
        (
            TERMINAL_BINARY_FORK.replace("  ^bb_true:", "  ^bb_true(%x: i32):"),
            "cannot take block arguments",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "    tt.return\n  ^bb_false:",
                "    cf.br ^bb_false\n  ^bb_false:",
            ),
            "only the exact operand-free",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "  ^bb_false:",
                "  ^bb_extra:\n    tt.return\n  ^bb_false:",
            ),
            "unexpected terminal-fork block",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "    tt.return\n  ^bb_false:",
                "  ^bb_false:",
            ),
            "true arm does not end",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "%condition = arith.cmpi eq, %flag, %zero_i32 : i32",
                "%condition = tt.splat %flag : i32 -> tensor<1xi32>",
            ),
            "predicate must be a scalar i1",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "    tt.store %output, %zero_f32 : !tt.ptr<f32>",
                "    tt.store %output, %value : !tt.ptr<f32>",
            ),
            "false arm uses SSA value from its true sibling",
        ),
        (
            TERMINAL_BINARY_FORK.replace(
                "    tt.return\n  ^bb_false:",
                "    tt.return %flag : i32\n  ^bb_false:",
            ),
            "tt.return explicit types.*do not match",
        ),
    ],
)
def test_terminal_fork_rejects_every_shape_outside_closed_profile(
    malformed: str,
    message: str,
) -> None:
    with pytest.raises(TTIRParseError, match=message):
        fingerprint_ttir(malformed)


def test_terminal_fork_helper_cannot_contain_an_interprocedural_call() -> None:
    malformed = OUTLINED_TERMINAL_FORK_IN_LOOP.replace(
        "    %c0 = arith.constant 0 : i32\n"
        "    %condition = arith.cmpi eq, %iv, %c0 : i32",
        "    %c0 = arith.constant 0 : i32\n"
        "    tt.call @leaf() : () -> ()\n"
        "    %condition = arith.cmpi eq, %iv, %c0 : i32",
    ).replace(
        "\n}\n",
        "\n  tt.func private @leaf() {\n    tt.return\n  }\n}\n",
    )

    with pytest.raises(TTIRParseError, match="cannot contain tt.call"):
        fingerprint_ttir(malformed)


_SAVED_REAL_TERMINAL_FORK_CORPUS = (
    (
        "cross_entropy",
        Path(
            "/tmp/intentir-scrc-task11/"
            "wave-20260725-b2-v7-v15-complex6-xhigh-01/liger_cross_entropy/"
            "fanout-001/candidate-01/attempt-01/gate1.json"
        ),
        0,
    ),
    (
        "flash_bwd_one_col",
        Path(
            "/tmp/intentir-scrc-task11/"
            "wave-20260725-b2-v7-v15-complex6-xhigh-01/"
            "flash_attn_flash_attn_triton_bwd_kernel_one_col_block/"
            "fanout-001/candidate-01/attempt-01/gate1.json"
        ),
        0,
    ),
    (
        "flash_full_bwd_helper",
        Path(
            "/tmp/intentir-scrc-task11/"
            "wave-20260725-b3-v6-v14-frontier5-xhigh-01/"
            "flash_attn_flash_attn_triton_bwd_kernel/"
            "fanout-001/candidate-03/attempt-04/gate1.json"
        ),
        1,
    ),
)


@pytest.mark.parametrize(
    ("case_name", "receipt_path", "expected_context_depth"),
    _SAVED_REAL_TERMINAL_FORK_CORPUS,
)
def test_saved_real_terminal_fork_corpus_and_role_mutation(
    case_name: str,
    receipt_path: Path,
    expected_context_depth: int,
) -> None:
    if not receipt_path.is_file():
        pytest.skip(f"saved Task11 corpus is unavailable: {case_name}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    ttir = payload["source_compile"]["ttir"]

    fingerprint = fingerprint_ttir(ttir)
    assert len(fingerprint.terminal_forks) == 1
    assert (
        len(fingerprint.terminal_forks[0].region_context)
        == expected_context_depth
    )

    branch = re.search(
        r"cf\.cond_br\s+(%[A-Za-z0-9_.$-]+)\s*,\s*"
        r"\^([A-Za-z0-9_.$-]+)\s*,\s*\^([A-Za-z0-9_.$-]+)",
        ttir,
    )
    assert branch is not None
    original = branch.group(0)
    role_swapped = (
        f"cf.cond_br {branch.group(1)}, ^{branch.group(3)}, "
        f"^{branch.group(2)}"
    )
    mutated = ttir[: branch.start()] + role_swapped + ttir[branch.end() :]
    verdict = compare_ttir(ttir, mutated)
    assert not verdict.hard_pass, (case_name, verdict.to_dict())

    with_successor_operand = original.replace(
        f"^{branch.group(2)}",
        f"^{branch.group(2)}({branch.group(1)} : i1)",
        1,
    )
    malformed = ttir.replace(original, with_successor_operand, 1)
    with pytest.raises(TTIRParseError, match="operand-free cf.cond_br"):
        fingerprint_ttir(malformed)


def test_reachable_multiblock_helper_is_explicitly_observer_unsupported() -> None:
    multiblock = OUTLINED_VOID_HELPER_IN_LOOP.replace(
        "    %source_at = tt.addptr %source, %index : !tt.ptr<f32>, i32",
        "    %condition = arith.constant true : i1\n"
        "    cf.cond_br %condition, ^bb1, ^bb2\n"
        "  ^bb1:\n"
        "    %source_at = tt.addptr %source, %index : !tt.ptr<f32>, i32",
    )

    with pytest.raises(TTIRParseError, match="exactly its true and false"):
        fingerprint_ttir(multiblock)


def test_public_multiblock_function_is_explicitly_observer_unsupported() -> None:
    multiblock = r'''
module {
  tt.func public @branch(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %condition = arith.constant true : i1
    cf.cond_br %condition, ^bb1, ^bb2
  ^bb1:
    %value = tt.load %x : !tt.ptr<f32>
    cf.br ^bb2
  ^bb2:
    tt.store %out, %value : !tt.ptr<f32>
    tt.return
  }
}
'''

    with pytest.raises(TTIRParseError, match="only the exact operand-free"):
        fingerprint_ttir(multiblock)


@pytest.mark.parametrize(
    "with_top_level_block",
    [
        r'''
module {
  tt.func public @public_block(%x: f32) {
  ^bb0:
    tt.return
  }
}
''',
        OUTLINED_NESTED_VALUE_HELPERS.replace(
            "  tt.func private @scale(%value: f32) -> f32 {\n",
            "  tt.func private @scale(%value: f32) -> f32 {\n  ^bb7:\n",
        ),
    ],
)
def test_reachable_top_level_block_label_without_cf_op_fails_closed(
    with_top_level_block: str,
) -> None:
    with pytest.raises(TTIRParseError, match="top-level block label"):
        fingerprint_ttir(with_top_level_block)


def test_structured_reduce_block_label_remains_supported() -> None:
    assert fingerprint_ttir(REDUCE_TTIR).reductions[0].combiner == "arith.addf"


@pytest.mark.parametrize(
    "operation_after_return",
    [
        r'''
module {
  tt.func public @void_return_then_op(%x: f32) {
    tt.return
    %unreachable = arith.addf %x, %x : f32
  }
}
''',
        OUTLINED_NESTED_VALUE_HELPERS.replace(
            "    tt.return %answer : f32\n  }\n}",
            "    tt.return %answer : f32\n"
            "    %unreachable = arith.addf %answer, %answer : f32\n"
            "  }\n}",
        ),
    ],
)
def test_operation_after_void_or_ssa_return_fails_closed(
    operation_after_return: str,
) -> None:
    with pytest.raises(TTIRParseError, match="operation after.*tt.return"):
        fingerprint_ttir(operation_after_return)


def test_duplicate_top_level_return_fails_closed() -> None:
    duplicate = INLINED_NESTED_VALUE_HELPERS.replace(
        "    tt.return\n",
        "    tt.return\n    tt.return\n",
    )

    with pytest.raises(TTIRParseError, match="more than one.*tt.return"):
        fingerprint_ttir(duplicate)


def test_second_block_in_structured_reduce_region_fails_closed() -> None:
    multiblock_reduce = REDUCE_TTIR.replace(
        "      tt.reduce.return %combined : f32\n",
        "      tt.reduce.return %combined : f32\n"
        "    ^bb1(%other_lhs: f32, %other_rhs: f32):\n"
        "      %other = arith.mulf %other_lhs, %other_rhs : f32\n"
        "      tt.reduce.return %other : f32\n",
    )

    with pytest.raises(TTIRParseError, match="more than one block"):
        fingerprint_ttir(multiblock_reduce)


OUTLINED_ORDERED_WRITES = r'''
module {
  tt.func public @ordered(%out: !tt.ptr<f32>, %first: f32, %second: f32) {
    tt.call @write(%out, %first) : (!tt.ptr<f32>, f32) -> ()
    tt.call @write(%out, %second) : (!tt.ptr<f32>, f32) -> ()
    tt.return
  }
  tt.func private @write(%destination: !tt.ptr<f32>, %value: f32) {
    tt.store %destination, %value : !tt.ptr<f32>
    tt.return
  }
}
'''


INLINED_ORDERED_WRITES = r'''
module {
  tt.func public @ordered_inline(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    tt.store %arg0, %arg1 : !tt.ptr<f32>
    tt.store %arg0, %arg2 : !tt.ptr<f32>
    tt.return
  }
}
'''


def test_ordered_outlined_effects_match_equivalent_inline_effects() -> None:
    verdict = compare_ttir(OUTLINED_ORDERED_WRITES, INLINED_ORDERED_WRITES)

    assert verdict.hard_pass, verdict.to_dict()


def test_swapping_same_callee_store_calls_changes_hard_effect_trace() -> None:
    swapped = OUTLINED_ORDERED_WRITES.replace(
        "    tt.call @write(%out, %first) : (!tt.ptr<f32>, f32) -> ()\n"
        "    tt.call @write(%out, %second) : (!tt.ptr<f32>, f32) -> ()",
        "    tt.call @write(%out, %second) : (!tt.ptr<f32>, f32) -> ()\n"
        "    tt.call @write(%out, %first) : (!tt.ptr<f32>, f32) -> ()",
    )
    verdict = compare_ttir(OUTLINED_ORDERED_WRITES, swapped)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["dataflow"]


NONISOMORPHIC_LOOP_TREE_A = r'''
module {
  tt.func public @tree_a(%out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %root = %c0 to %n step %c1 : i32 {
      tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
      scf.for %fork = %c0 to %n step %c1 : i32 {
        tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
        scf.for %fork_left = %c0 to %n step %c1 : i32 {
          tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
          scf.yield
        }
        scf.for %fork_right = %c0 to %n step %c1 : i32 {
          tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
          scf.yield
        }
        scf.yield
      }
      scf.for %chain = %c0 to %n step %c1 : i32 {
        tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
        scf.for %chain_leaf = %c0 to %n step %c1 : i32 {
          tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
          scf.yield
        }
        scf.yield
      }
      scf.yield
    }
    tt.return
  }
  tt.func private @bump(%out: !tt.ptr<f32>) {
    %one = arith.constant 1.000000e+00 : f32
    %prior = tt.atomic_rmw add, %out, %one : (!tt.ptr<f32>, f32) -> f32
    tt.return
  }
}
'''


NONISOMORPHIC_LOOP_TREE_B = r'''
module {
  tt.func public @tree_b(%out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %root = %c0 to %n step %c1 : i32 {
      tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
      scf.for %deep0 = %c0 to %n step %c1 : i32 {
        tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
        scf.for %deep1 = %c0 to %n step %c1 : i32 {
          tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
          scf.for %deep2 = %c0 to %n step %c1 : i32 {
            tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
            scf.yield
          }
          scf.yield
        }
        scf.yield
      }
      scf.for %leaf0 = %c0 to %n step %c1 : i32 {
        tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
        scf.yield
      }
      scf.for %leaf1 = %c0 to %n step %c1 : i32 {
        tt.call @bump(%out) : (!tt.ptr<f32>) -> ()
        scf.yield
      }
      scf.yield
    }
    tt.return
  }
  tt.func private @bump(%out: !tt.ptr<f32>) {
    %one = arith.constant 1.000000e+00 : f32
    %prior = tt.atomic_rmw add, %out, %one : (!tt.ptr<f32>, f32) -> f32
    tt.return
  }
}
'''


def test_nonisomorphic_nested_loop_trees_change_hard_region_topology() -> None:
    left = fingerprint_ttir(NONISOMORPHIC_LOOP_TREE_A)
    right = fingerprint_ttir(NONISOMORPHIC_LOOP_TREE_B)
    verdict = compare_fingerprints(left, right)

    assert len(left.loops) == len(right.loops) == 6
    assert dict(left.op_multiset) == dict(right.op_multiset)
    assert not verdict.hard_pass
    assert not verdict.hard_passes["dataflow"]


def test_call_explicit_type_cannot_lie_about_actual_ssa_producer_type() -> None:
    malformed = r'''
module {
  tt.func public @caller(%out: !tt.ptr<f32>) {
    %wrong = arith.constant 7 : i32
    %answer = tt.call @identity(%wrong) : (f32) -> f32
    tt.store %out, %answer : !tt.ptr<f32>
    tt.return
  }
  tt.func private @identity(%value: f32) -> f32 {
    tt.return %value : f32
  }
}
'''

    with pytest.raises(TTIRParseError, match="actual SSA type i32.*call type f32"):
        fingerprint_ttir(malformed)


def test_return_explicit_type_cannot_lie_about_actual_ssa_producer_type() -> None:
    malformed = r'''
module {
  tt.func public @caller(%out: !tt.ptr<f32>) {
    %answer = tt.call @bad_return() : () -> f32
    tt.store %out, %answer : !tt.ptr<f32>
    tt.return
  }
  tt.func private @bad_return() -> f32 {
    %wrong = arith.constant 7 : i32
    tt.return %wrong : f32
  }
}
'''

    with pytest.raises(TTIRParseError, match="actual SSA type i32.*return type f32"):
        fingerprint_ttir(malformed)


def test_private_noinline_boolean_is_the_only_erased_function_hint() -> None:
    with_hint = OUTLINED_NESTED_VALUE_HELPERS.replace(
        "tt.func private @scale(%value: f32) -> f32 {",
        "tt.func private @scale(%value: f32) -> f32 "
        "attributes {noinline = false} {",
    )

    assert compare_ttir(OUTLINED_NESTED_VALUE_HELPERS, with_hint).hard_pass

    public_with_hint = DATAFLOW_TTIR.replace(
        "tt.func public @dataflow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {",
        "tt.func public @dataflow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) "
        "attributes {noinline = true} {",
    )
    assert compare_ttir(DATAFLOW_TTIR, public_with_hint).hard_pass


@pytest.mark.parametrize(
    "unsupported_private_metadata",
    [
        OUTLINED_NESTED_VALUE_HELPERS.replace(
            "tt.func private @scale(%value: f32)",
            "tt.func private @scale(%value: f32 "
            "{tt.divisibility = 16 : i32})",
        ),
        OUTLINED_NESTED_VALUE_HELPERS.replace(
            "tt.func private @scale(%value: f32)",
            "tt.func private @scale(%value: f32 {mystery = true})",
        ),
        OUTLINED_NESTED_VALUE_HELPERS.replace(
            "tt.func private @scale(%value: f32) -> f32 {",
            "tt.func private @scale(%value: f32) -> f32 "
            "attributes {semantic_mode = fast} {",
        ),
    ],
)
def test_reachable_private_metadata_outside_closed_allowlist_fails_closed(
    unsupported_private_metadata: str,
) -> None:
    with pytest.raises(TTIRParseError, match="unsupported"):
        fingerprint_ttir(unsupported_private_metadata)


@pytest.mark.parametrize("name", ["semantic_mode", "tt.num_warps"])
def test_unknown_public_function_attribute_fails_closed(name: str) -> None:
    unsupported = DATAFLOW_TTIR.replace(
        "tt.func public @dataflow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) {",
        "tt.func public @dataflow(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>) "
        f"attributes {{{name} = 8 : i32}} {{",
    )

    with pytest.raises(TTIRParseError, match="unsupported function attribute"):
        fingerprint_ttir(unsupported)


def test_function_return_value_mapping_is_hard_dataflow_evidence() -> None:
    template = r'''
module {
  tt.func public @choose(%x: f32, %y: f32) -> f32 {
    %sum = arith.addf %x, %y : f32
    %product = arith.mulf %x, %y : f32
    tt.return %RETURN : f32
  }
}
'''
    returns_sum = template.replace("%RETURN", "%sum")
    returns_product = template.replace("%RETURN", "%product")

    verdict = compare_ttir(returns_sum, returns_product)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["dataflow"]


@pytest.mark.parametrize("replacement", ["%result#1, %result#0", "%result#1, %result#1"])
def test_equal_typed_multi_scan_result_slots_keep_identity(
    replacement: str,
) -> None:
    changed = SAME_TYPE_MULTI_SCAN_TTIR.replace(
        "tt.return %result#0, %result#1 :",
        f"tt.return {replacement} :",
        1,
    )

    verdict = compare_ttir(SAME_TYPE_MULTI_SCAN_TTIR, changed)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["dataflow"]


@pytest.mark.parametrize(
    "malformed, message",
    [
        (
            DATAFLOW_TTIR.replace("tt.return", "tt.return arbitrary-garbage", 1),
            "tt.return",
        ),
        (
            DATAFLOW_TTIR.replace(
                "tt.return", "tt.return {future_semantics = true}", 1
            ),
            "attributes are unsupported",
        ),
        (
            LOOP_CARRIED_POINTER_ACCESS_TTIR.replace(
                "      scf.yield %next_read, %next_write, %next_inout : "
                "tensor<8x!tt.ptr<f16>>, tensor<8x!tt.ptr<f16>>, "
                "tensor<8x!tt.ptr<f16>>\n",
                "",
                1,
            ),
            "exactly one scf.yield",
        ),
        (
            SIBLING_REGION_SSA_REUSE_TTIR.replace(
                "      scf.yield\n",
                "      scf.yield\n      scf.yield\n",
                1,
            ),
            "operation after scf.for terminator",
        ),
        (
            LOOP_CARRIED_POINTER_ACCESS_TTIR.replace(
                "scf.yield %next_read, %next_write, %next_inout : "
                "tensor<8x!tt.ptr<f16>>, tensor<8x!tt.ptr<f16>>, "
                "tensor<8x!tt.ptr<f16>>",
                "scf.yield %next_read, %next_write, %next_inout : i32, i32, i32",
                1,
            ),
            "yield types must close positionally",
        ),
        (
            REDUCE_TTIR.replace(
                "tt.reduce.return %combined : f32",
                "tt.reduce.return %combined : i32",
                1,
            ),
            "explicit types do not match reduce element types",
        ),
        (
            REDUCE_TTIR.replace(
                "tt.reduce.return %combined : f32",
                "tt.reduce.return %combined : f32 nonsense",
                1,
            ),
            "trailing or compound type syntax",
        ),
        (
            REDUCE_TTIR.replace(
                "      tt.reduce.return %combined : f32\n",
                "      tt.reduce.return %combined : f32\n"
                "      tt.reduce.return\n",
                1,
            ),
            "operation after tt.reduce terminator",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs {future_semantics = true} : f32",
                1,
            ),
            "attributes are unsupported",
        ),
        (
            SCAN_TTIR.replace(
                "tt.scan.return %lhs : f32",
                "tt.scan.return %lhs : (f32)",
                1,
            ),
            "unparenthesized list",
        ),
    ],
    ids=[
        "root-garbage",
        "root-attribute",
        "scf-missing-yield",
        "scf-duplicate-yield",
        "scf-lying-yield-type",
        "reduce-lying-return-type",
        "reduce-trailing-type-garbage",
        "reduce-duplicate-return",
        "scan-return-attribute",
        "scan-return-parenthesized-type",
    ],
)
def test_all_terminator_shells_and_region_closures_fail_closed(
    malformed: str,
    message: str,
) -> None:
    with pytest.raises(TTIRParseError, match=message):
        fingerprint_ttir(malformed)


def test_zero_result_scf_for_implicit_empty_yield_is_canonical() -> None:
    implicit = SIBLING_REGION_SSA_REUSE_TTIR.replace(
        "      scf.yield\n", "", 1
    )

    assert compare_ttir(SIBLING_REGION_SSA_REUSE_TTIR, implicit).hard_pass


def test_quoted_generic_result_type_cannot_have_unparsed_tail() -> None:
    malformed = DATAFLOW_TTIR.replace(
        "    tt.store %out, %answer",
        '    %junk = "arith.addf"(%answer, %answer) '
        ": (f32, f32) -> f32 nonsense\n"
        "    tt.store %out, %answer",
        1,
    )

    with pytest.raises(TTIRParseError, match="trailing or compound type syntax"):
        fingerprint_ttir(malformed)


def test_non_scan_tensor_encoding_alias_denotation_fails_closed_globally() -> None:
    template = r'''
#layout = #future.layout<PAYLOAD>
module {
  tt.func public @encoded(%x: tensor<8xf32, #layout>) -> tensor<8xf32, #layout> {
    %result = arith.addf %x, %x : tensor<8xf32, #layout>
    tt.return %result : tensor<8xf32, #layout>
  }
}
'''
    for payload in ("one", "two"):
        with pytest.raises(TTIRParseError, match="tensor encodings are unsupported"):
            fingerprint_ttir(template.replace("PAYLOAD", payload))


def test_module_type_alias_denotation_fails_closed_globally() -> None:
    template = r'''
!element = type PAYLOAD
module {
  tt.func public @aliased(%x: !element) -> !element {
    tt.return %x : !element
  }
}
'''
    for payload in ("f32", "i32"):
        with pytest.raises(TTIRParseError, match="type aliases are unsupported"):
            fingerprint_ttir(template.replace("PAYLOAD", payload))


def test_effect_order_inside_scf_region_is_hard_evidence() -> None:
    template = r'''
module {
  tt.func public @ordered(%out: !tt.ptr<f32>, %first: f32, %second: f32, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %iv = %c0 to %n step %c1 : i32 {
      tt.store %out, %FIRST : !tt.ptr<f32>
      tt.store %out, %SECOND : !tt.ptr<f32>
      scf.yield
    }
    tt.return
  }
}
'''
    source = template.replace("%FIRST", "%first").replace("%SECOND", "%second")
    reordered = template.replace("%FIRST", "%second").replace("%SECOND", "%first")

    verdict = compare_ttir(source, reordered)

    assert not verdict.hard_pass
    assert not verdict.hard_passes["dataflow"]
