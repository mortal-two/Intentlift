from __future__ import annotations

from dataclasses import replace
import json

from intentir.backend.cuda.capabilities import load_high_performance_registry
from intentir.verify import workload_manifest as workload_module
from intentir.verify.workload_manifest import (
    SourceResolutionStatus,
    build_workload_manifest,
)


def test_scrc_manifest_is_exact_projection_of_28_kernel_registry() -> None:
    registry = load_high_performance_registry()
    manifest = build_workload_manifest(registry)

    assert [entry.ordinal for entry in manifest.entries] == list(range(1, 29))
    assert [entry.case_id for entry in manifest.entries] == [
        item.kernel_id for item in registry.kernels
    ]
    assert manifest.scenario_29["counted_as_direct_kernel"] is False
    assert {entry.suite for entry in manifest.entries} == {
        "liger", "flash_attention", "xformers"
    }
    assert {
        suite: sum(entry.suite == suite for entry in manifest.entries)
        for suite in ("liger", "flash_attention", "xformers")
    } == {"liger": 17, "flash_attention": 6, "xformers": 5}


def test_scrc_manifest_selects_six_real_representative_entries() -> None:
    manifest = build_workload_manifest()
    entries = manifest.select(tier="representative")

    assert len(entries) == 6
    assert {entry.representative_tier for entry in entries} == {
        "core.elementwise",
        "core.masked_pointwise",
        "core.single_axis_sum",
        "frontier.softmax",
        "frontier.normalization",
        "frontier.contraction",
    }
    assert all(entry.case_id in manifest.by_id for entry in entries)
    assert [entry.case_id for entry in entries] == [
        "high_performance:liger_dyt",
        "high_performance:liger_swiglu",
        (
            "high_performance:flash_attn.flash_attn_triton."
            "_bwd_preprocess_do_o_dot"
        ),
        "high_performance:liger_softmax",
        "high_performance:liger_rms_norm",
        "high_performance:flash_attn.ops.triton.linear.kernel_fwd",
    ]


def test_current_workspace_resolves_all_28_sources_and_symbols() -> None:
    validation = build_workload_manifest().validate()

    assert validation.passed, validation.to_json()
    assert validation.resolved_count == 28
    assert all(source.path is not None for source in validation.sources)


def test_missing_optional_source_is_case_unavailable(monkeypatch, tmp_path) -> None:
    entry = build_workload_manifest().by_id[
        "high_performance:xformers.ops._triton.rmsnorm_kernels._rms_norm_kernel"
    ]
    monkeypatch.setattr(workload_module, "_installed_module_path", lambda _: None)

    result = entry.resolve_source(dataset_root=tmp_path)

    assert result.status is SourceResolutionStatus.CASE_UNAVAILABLE
    assert not result.resolved
    assert "Dataset fallback is absent" in result.diagnostic


def test_wrong_entry_symbol_is_manifest_invalid() -> None:
    entry = replace(
        build_workload_manifest().entries[0],
        symbol="definitely_missing_scrc_symbol",
    )

    result = entry.resolve_source()

    assert result.status is SourceResolutionStatus.MANIFEST_INVALID
    assert "found 0 definitions" in result.diagnostic


def test_invalid_installed_source_falls_back_to_dataset(
    monkeypatch, tmp_path
) -> None:
    entry = build_workload_manifest().entries[0]
    installed = tmp_path / "installed.py"
    installed.write_text("this is not python ???\n", encoding="utf-8")
    fallback = (
        tmp_path / "Liger-Kernel" / "src"
    ).joinpath(*entry.source_module.split(".")).with_suffix(".py")
    fallback.parent.mkdir(parents=True)
    fallback.write_text(f"def {entry.symbol}():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(
        workload_module, "_installed_module_path", lambda _: installed
    )

    result = entry.resolve_source(dataset_root=tmp_path)

    assert result.status is SourceResolutionStatus.RESOLVED
    assert result.path == fallback.resolve()
    assert result.location.startswith("dataset://")


def test_nested_method_does_not_satisfy_top_level_entry_symbol(
    monkeypatch, tmp_path
) -> None:
    entry = build_workload_manifest().entries[0]
    installed = tmp_path / "installed.py"
    installed.write_text(
        f"class NotAKernel:\n    def {entry.symbol}(self):\n        pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        workload_module, "_installed_module_path", lambda _: installed
    )

    result = entry.resolve_source(dataset_root=tmp_path / "absent")

    assert result.status is SourceResolutionStatus.MANIFEST_INVALID
    assert "module top level" in result.diagnostic


def test_manifest_json_is_runner_metadata_not_a_replay_artifact() -> None:
    encoded = json.dumps(build_workload_manifest().to_json(), sort_keys=True)

    assert '"direct_kernel_count": 28' in encoded
    for forbidden in (
        "source_code",
        "binding_map",
        "concrete_c_values",
        "gate1_diff",
        "prior_replay",
        "conversation",
    ):
        assert forbidden not in encoded


def test_validation_json_uses_portable_source_locations() -> None:
    encoded = json.dumps(build_workload_manifest().validate().to_json())

    assert '"location"' in encoded
    assert '"path"' not in encoded
    assert "/home/" not in encoded


def test_unknown_or_empty_tier_fails_closed() -> None:
    manifest = build_workload_manifest()

    for tier in ("unknown", "core.not-real"):
        try:
            manifest.select(tier=tier)
        except KeyError:
            pass
        else:
            raise AssertionError(f"tier {tier!r} unexpectedly selected entries")
