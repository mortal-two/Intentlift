from __future__ import annotations

import pytest

from intentir.backend.runtime_contract import (
    RuntimeContract,
    RuntimeContractError,
    RuntimeLaunchFacts,
    RuntimePredicate,
    runtime_launch_facts_from_tensors,
)


def _contract() -> RuntimeContract:
    return RuntimeContract(
        contract_id="softmax.full-tile.sm120",
        predicates=(
            RuntimePredicate("p.n", "dimension_divisible", "N", 128, ("fact.n",)),
            RuntimePredicate("p.stride", "stride_eq", "X:1", 1, ("layout.x",)),
            RuntimePredicate("p.align", "pointer_alignment", "X", 16, ("access.x",)),
            RuntimePredicate("p.target", "target_capability", "sm_120a", True, ("target.sm120",)),
            RuntimePredicate("p.smem", "dynamic_smem_le", "dynamic_smem_bytes", 49152, ("schedule.smem",)),
        ),
    )


def _facts(**overrides) -> RuntimeLaunchFacts:
    values = {
        "dimensions": {"M": 32, "N": 1024},
        "strides": {"X": (1024, 1), "Y": (1024, 1)},
        "pointer_addresses": {"X": 0x1000, "Y": 0x2000},
        "target_capabilities": frozenset({"sm_120a", "tensor_core"}),
        "dynamic_smem_bytes": 32768,
    }
    values.update(overrides)
    return RuntimeLaunchFacts(**values)


def test_runtime_contract_matches_checkable_launch_facts() -> None:
    match = _contract().evaluate(_facts())

    assert match.matched is True
    assert all(result.matched for result in match.predicates)


def test_contract_miss_is_a_dispatch_result_not_a_verification_verdict() -> None:
    match = _contract().evaluate(
        _facts(dimensions={"M": 32, "N": 1000}, pointer_addresses={"X": 0x1004})
    )

    assert match.matched is False
    failed = {result.predicate_id for result in match.predicates if not result.matched}
    assert failed == {"p.n", "p.align"}
    assert "FULL" not in str(match.to_dict())
    assert "PARTIAL" not in str(match.to_dict())
    assert "OOS" not in str(match.to_dict())


def test_missing_runtime_fact_cleanly_misses_contract() -> None:
    match = _contract().evaluate(_facts(strides={}))

    stride = next(item for item in match.predicates if item.predicate_id == "p.stride")
    assert stride.matched is False
    assert stride.observed is None
    assert "absent" in stride.reason


def test_contract_round_trip_is_strict() -> None:
    value = _contract().to_dict()
    restored = RuntimeContract.from_dict(value)
    assert restored == _contract()

    value["gate1_verdict"] = "PASS"
    with pytest.raises(RuntimeContractError, match="unknown fields"):
        RuntimeContract.from_dict(value)


def test_arbitrary_expression_and_invalid_alignment_are_rejected() -> None:
    with pytest.raises(RuntimeContractError, match="unsupported"):
        RuntimePredicate("p.eval", "python_eval", "N", "N % 8 == 0", ("x",))
    with pytest.raises(RuntimeContractError, match="power of two"):
        RuntimePredicate("p.align", "pointer_alignment", "X", 24, ("x",))


def test_predicate_requires_evidence_reference() -> None:
    with pytest.raises(RuntimeContractError, match="evidence references"):
        RuntimePredicate("p.n", "dimension_eq", "N", 128, ())


class _FakeTensor:
    def __init__(self, shape, stride, pointer):
        self.shape = tuple(shape)
        self._stride = tuple(stride)
        self._pointer = pointer

    def stride(self):
        return self._stride

    def data_ptr(self):
        return self._pointer


def test_runtime_facts_are_derived_from_actual_tensor_descriptors() -> None:
    tensors = {
        "A": _FakeTensor((128, 256), (256, 1), 0x1000),
        "B": _FakeTensor((512, 256), (256, 1), 0x2000),
    }

    facts = runtime_launch_facts_from_tensors(
        tensors=tensors,
        dimension_sources={"M": ("A", 0), "N": ("B", 0), "K": ("A", 1)},
        target_capabilities={"sm_120a", "tensor_core"},
    )

    assert facts.dimensions == {"M": 128, "N": 512, "K": 256}
    assert facts.strides == {"A": (256, 1), "B": (256, 1)}
    assert facts.pointer_addresses == {"A": 0x1000, "B": 0x2000}
    assert facts.target_capabilities == frozenset({"sm_120a", "tensor_core"})


def test_runtime_fact_extraction_rejects_missing_or_invented_descriptors() -> None:
    with pytest.raises(RuntimeContractError, match="unknown tensor"):
        runtime_launch_facts_from_tensors(
            tensors={"A": _FakeTensor((8,), (1,), 0x1000)},
            dimension_sources={"N": ("B", 0)},
            target_capabilities={"sm_120a"},
        )

    with pytest.raises(RuntimeContractError, match=r"shape, stride\(\), and data_ptr"):
        runtime_launch_facts_from_tensors(
            tensors={"A": object()},
            dimension_sources={"N": ("A", 0)},
            target_capabilities={"sm_120a"},
        )

