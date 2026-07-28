from intentir.backend.roofline import RooflineCeilings, evaluate_logical_roofline


def test_logical_roofline_classifies_low_intensity_workload_as_memory_bound() -> None:
    ceilings = RooflineCeilings(1000.0, 100_000.0)
    result = evaluate_logical_roofline(
        {
            "complete": True,
            "boundary_io_bytes": 1_000_000,
            "floating_point_ops": 2_000_000,
            "special_function_ops": 0,
        },
        2.0,
        ceilings,
    )

    assert result["predicted_bottleneck"] == "memory"
    assert result["ceilings"]["ridge_point_flop_per_byte"] == 100.0
    assert result["memory_time_lower_bound_us"] == 1.0
    assert result["latency_over_logical_roof_lower_bound"] == 2.0
    assert result["observed_operational_regime"] == "memory_roof_approaching"
    assert result["supports_memory_wall_claim"] is True


def test_logical_roofline_keeps_sfu_caveat_separate_from_flops() -> None:
    result = evaluate_logical_roofline(
        {
            "complete": True,
            "boundary_io_bytes": 100,
            "floating_point_ops": 10_000,
            "special_function_ops": 50,
        },
        1.0,
        RooflineCeilings(10.0, 500.0),
    )

    assert result["predicted_bottleneck"] == "fp32_compute"
    assert any("special-function" in item for item in result["caveats"])


def test_tiny_low_intensity_point_is_not_mislabeled_memory_wall() -> None:
    result = evaluate_logical_roofline(
        {
            "complete": True,
            "boundary_io_bytes": 2_032,
            "floating_point_ops": 760,
            "special_function_ops": 254,
        },
        0.76,
        RooflineCeilings(986.0, 38_658.0),
    )

    assert result["asymptotic_roof_bottleneck"] == "memory"
    assert result["observed_operational_regime"] == (
        "latency_or_unmodeled_work_dominated"
    )
    assert result["supports_memory_wall_claim"] is False


def test_faster_than_logical_dram_roof_is_reported_as_model_mismatch() -> None:
    result = evaluate_logical_roofline(
        {
            "complete": True,
            "boundary_io_bytes": 1_000_000,
            "floating_point_ops": 2_000_000,
            "special_function_ops": 0,
        },
        0.75,
        RooflineCeilings(1000.0, 100_000.0),
    )

    assert result["latency_over_logical_roof_lower_bound"] == 0.75
    assert result["roof_utilization"] > 1.0
    assert result["observed_operational_regime"] == (
        "cache_reuse_or_roof_model_mismatch"
    )
    assert result["supports_memory_wall_claim"] is False
    assert "logical_roof_inconsistent_with_observation" in result[
        "memory_wall_claim_rejection_reasons"
    ]


def test_unmodeled_special_functions_forbid_memory_wall_claim() -> None:
    result = evaluate_logical_roofline(
        {
            "complete": True,
            "boundary_io_bytes": 1_000_000,
            "floating_point_ops": 2_000_000,
            "special_function_ops": 1,
        },
        1.25,
        RooflineCeilings(1000.0, 100_000.0),
    )

    assert result["predicted_bottleneck"] == "memory"
    assert result["supports_memory_wall_claim"] is False
    assert "special_function_throughput_not_modeled" in result[
        "memory_wall_claim_rejection_reasons"
    ]
