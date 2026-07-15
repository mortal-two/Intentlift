from __future__ import annotations

import numpy as np
import pytest

from intentir.verify.gate2 import (
    InputCaseTemplate,
    InputDistribution,
    TensorInputSpec,
    boundary_1d_extents,
    generate_input_matrix,
    generate_tensor_input,
)


def test_gate2_input_generation_is_seed_deterministic() -> None:
    spec = TensorInputSpec((7, 5), "f32", InputDistribution.NORMAL, 2.0)
    first = generate_tensor_input(spec, seed=17)
    second = generate_tensor_input(spec, seed=17)
    different = generate_tensor_input(spec, seed=18)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (7, 5)
    assert first.dtype == np.float32


def test_gate2_domain_aware_distributions_preserve_expected_ranges() -> None:
    positive = generate_tensor_input(
        TensorInputSpec((257,), "f32", InputDistribution.POSITIVE, 4.0),
        seed=19,
    )
    near_one = generate_tensor_input(
        TensorInputSpec((257,), "f32", InputDistribution.NEAR_ONE, 0.125),
        seed=23,
    )

    assert np.all(positive >= 0.0)
    assert np.all(near_one >= 0.875)
    assert np.all(near_one <= 1.125)


def test_gate2_large_dynamic_and_ieee_boundary_patterns_are_explicit() -> None:
    dynamic = generate_tensor_input(
        TensorInputSpec(
            (15,), "f64", InputDistribution.LARGE_DYNAMIC, 1.0
        ),
        seed=29,
    )
    special = generate_tensor_input(
        TensorInputSpec((14,), "f32", InputDistribution.IEEE_SPECIAL),
        seed=31,
    )

    assert np.min(np.abs(dynamic[np.nonzero(dynamic)])) < 1.0e-3
    assert np.max(np.abs(dynamic)) > 1.0e3
    assert np.any(dynamic < 0.0) and np.any(dynamic > 0.0)
    assert np.isnan(special).any()
    assert np.isposinf(special).any()
    assert np.isneginf(special).any()


def test_gate2_integer_and_boolean_generation_uses_declared_dtype() -> None:
    integers = generate_tensor_input(
        TensorInputSpec((32,), "i32", InputDistribution.UNIFORM, 3.0),
        seed=37,
    )
    booleans = generate_tensor_input(
        TensorInputSpec((32,), "i1", InputDistribution.NORMAL), seed=41
    )

    assert integers.dtype == np.int32
    assert np.all(integers >= -3) and np.all(integers <= 3)
    assert booleans.dtype == np.bool_


def test_gate2_integer_boundary_distribution_is_explicit_and_repeatable() -> None:
    values = generate_tensor_input(
        TensorInputSpec(
            (12,), "i64", InputDistribution.INTEGER_BOUNDARY, 96.0
        ),
        seed=41,
    )

    expected = np.array([0, 96, 48, 1, 95, 0] * 2, dtype=np.int64)
    assert values.dtype == np.int64
    np.testing.assert_array_equal(values, expected)


def test_gate2_input_spec_rejects_ambiguous_or_invalid_requests() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TensorInputSpec((-1,), "f32")
    with pytest.raises(ValueError, match="exactly one"):
        generate_tensor_input(TensorInputSpec((1,), "f32"))
    with pytest.raises(ValueError, match="floating dtype"):
        generate_tensor_input(
            TensorInputSpec((4,), "i32", InputDistribution.IEEE_SPECIAL),
            seed=43,
        )
    with pytest.raises(ValueError, match="integer dtype"):
        generate_tensor_input(
            TensorInputSpec(
                (4,), "f32", InputDistribution.INTEGER_BOUNDARY
            ),
            seed=47,
        )
    with pytest.raises(ValueError, match="integer dtype"):
        generate_tensor_input(
            TensorInputSpec((4,), "i1", InputDistribution.INTEGER_BOUNDARY),
            seed=53,
        )


def test_gate2_boundary_extent_axis_matches_scrc_shape_requirements() -> None:
    extents = boundary_1d_extents(64)

    assert [(item.label, item.extent) for item in extents] == [
        ("one", 1),
        ("block_minus_one", 63),
        ("block", 64),
        ("block_plus_one", 65),
        ("non_divisible", 131),
        ("far_multitile", 529),
    ]


def test_gate2_input_matrix_expands_templates_by_seed_with_metadata() -> None:
    templates = (
        InputCaseTemplate(
            "tail.normal",
            {
                "X": TensorInputSpec(
                    (65,), "f32", InputDistribution.NORMAL, 0.5
                )
            },
            {"N": 65},
            {"shape_profile": "block_plus_one"},
        ),
        InputCaseTemplate(
            "tail.large",
            {
                "X": TensorInputSpec(
                    (65,), "f32", InputDistribution.LARGE_DYNAMIC, 1.0
                )
            },
            {"N": 65},
        ),
    )

    matrix = generate_input_matrix(templates, (7, 11, 13))

    assert len(matrix) == 6
    assert matrix[0].case_id == "tail.normal.seed7"
    assert matrix[-1].case_id == "tail.large.seed13"
    assert matrix[0].values["N"] == 65
    assert matrix[0].metadata["shape_profile"] == "block_plus_one"
    assert matrix[0].metadata["tensors"]["X"] == {
        "shape": [65],
        "dtype": "f32",
        "distribution": "normal",
        "scale": 0.5,
    }
    repeated = generate_input_matrix(templates, (7, 11, 13))
    np.testing.assert_array_equal(
        matrix[0].values["X"], repeated[0].values["X"]
    )


def test_gate2_input_matrix_rejects_duplicate_axes_and_overlapping_keys() -> None:
    template = InputCaseTemplate(
        "normal",
        {"X": TensorInputSpec((4,), "f32")},
    )
    with pytest.raises(ValueError, match="template_id"):
        generate_input_matrix((template, template), (1,))
    with pytest.raises(ValueError, match="seeds must be unique"):
        generate_input_matrix((template,), (1, 1))
    with pytest.raises(ValueError, match="overlap"):
        InputCaseTemplate(
            "bad",
            {"X": TensorInputSpec((4,), "f32")},
            {"X": 4},
        )
