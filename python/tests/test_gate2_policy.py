from __future__ import annotations

import numpy as np
import pytest

from intentir.verify.gate1.model import ResultRelationKind
from intentir.verify.gate2.compare import close_q
from intentir.verify.gate2.policy import (
    ComparisonPolicy,
    ComparisonPolicyError,
    NumericTolerance,
)


def _policy(*, equal_nan: bool = False) -> ComparisonPolicy:
    return ComparisonPolicy.create(
        "test.symmetric.q.v1",
        (("f32", NumericTolerance(0.0, 1.0, equal_nan)),),
    )


def test_close_q_is_symmetric_and_not_assumed_transitive() -> None:
    rule = _policy().rule_for(ResultRelationKind.TOLERANCE, "f32")
    source = np.asarray([0.0], dtype=np.float32)
    replay = np.asarray([0.9], dtype=np.float32)
    reference = np.asarray([-0.9], dtype=np.float32)

    assert close_q(source, replay, rule)
    assert close_q(replay, source, rule)
    assert close_q(source, reference, rule)
    assert not close_q(replay, reference, rule)
    assert not close_q(reference, replay, rule)


def test_close_q_handles_nan_inf_and_exact_domains_fail_closed() -> None:
    strict = _policy().rule_for("tolerance", "f32")
    equal_nan = _policy(equal_nan=True).rule_for("tolerance", "f32")
    nan = np.asarray([np.nan], dtype=np.float32)
    positive_inf = np.asarray([np.inf], dtype=np.float32)
    negative_inf = np.asarray([-np.inf], dtype=np.float32)

    assert not close_q(nan, nan, strict)
    assert close_q(nan, nan, equal_nan)
    assert close_q(positive_inf, positive_inf, strict)
    assert not close_q(positive_inf, negative_inf, strict)

    exact = _policy().rule_for("exact", "i32")
    assert close_q(
        np.asarray([1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        exact,
    )
    assert not close_q(
        np.asarray([1], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        exact,
    )


def test_q_rejects_unsupported_classes_and_dtype_pairs() -> None:
    policy = _policy()
    with pytest.raises(ComparisonPolicyError, match="set-valued"):
        policy.rule_for("one_of_updates", "f32")
    with pytest.raises(ComparisonPolicyError, match="floating"):
        policy.rule_for("tolerance", "i32")
    with pytest.raises(ComparisonPolicyError, match="no tolerance rule"):
        policy.rule_for("tolerance", "f64")
