"""Architecture guards for the selective pre-SCRC verifier archive."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import subprocess
import sys


PYTHON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_verify_package_exports_only_active_gates() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import intentir.verify as verify; "
                "assert verify.__all__ == ['gate1', 'gate2']; "
                "assert 'intentir.verify.interpret' not in sys.modules; "
                "assert 'intentir.verify.back' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_public_interpreter_path_is_only_a_legacy_compatibility_shim() -> None:
    shim = importlib.import_module("intentir.verify.interpret")
    archived = importlib.import_module(
        "intentir.verify.back.legacy_v0.interpret"
    )

    assert shim.interpret is archived.interpret
    assert shim.InterpError is archived.InterpError
    assert shim._philox_uniform is archived._philox_uniform


def test_repair_and_fuzzy_ratio_era_gate1_models_are_explicitly_archived() -> None:
    gate1 = importlib.import_module("intentir.verify.gate1")
    legacy_result = importlib.import_module(
        "intentir.verify.back.legacy_v0.gate1_result"
    )
    legacy_ratio = importlib.import_module(
        "intentir.verify.back.legacy_v0.masked_ratio"
    )
    model = importlib.import_module("intentir.verify.gate1.model")

    assert not hasattr(gate1, "Gate1Result")
    assert not hasattr(gate1, "leakage_ratio")
    assert not hasattr(gate1, "masked_ablation_passes")
    assert not hasattr(model, "Gate1Result")

    result = legacy_result.Gate1Result(
        model.Gate1Status.REPLAY_CLOSED,
        diagnostics=["historical"],
        relift_attempts=1,
        repair_attempts=1,
        leakage_ratio=0.25,
    )
    assert legacy_result.Gate1Result.from_json_str(
        result.to_json_str()
    ) == result
    assert legacy_ratio.leakage_ratio(1.0, 0.5) == 0.5
    assert legacy_ratio.masked_ablation_passes(1.0, 0.5)


def test_active_scrc_verification_does_not_import_archived_interpreter() -> None:
    active_files = [
        *(PYTHON_ROOT / "intentir" / "verify" / "gate1").glob("*.py"),
        *(PYTHON_ROOT / "intentir" / "verify" / "gate2").glob("*.py"),
        PYTHON_ROOT / "intentir" / "extraction" / "lifters" / "scrc.py",
    ]

    for path in active_files:
        source = path.read_text(encoding="utf-8")
        assert "intentir.verify.interpret" not in source, path
        assert "intentir.verify.back" not in source, path


def test_production_lowering_gate2_path_is_archived_and_not_active() -> None:
    archived = (
        PYTHON_ROOT
        / "intentir"
        / "verify"
        / "back"
        / "gate2_production_lowering_v1"
    )
    assert (archived / "lower2c.py").is_file()
    assert (archived / "cuda_lowering.py").is_file()
    assert (archived / "orchestrate.py").is_file()
    assert (archived / "model.py").is_file()

    active_gate2 = PYTHON_ROOT / "intentir" / "verify" / "gate2"
    assert not (active_gate2 / "lower2c.py").exists()
    assert not (active_gate2 / "cuda_lowering.py").exists()
    assert not (PROJECT_ROOT / "test/high_performance_live_benchmark.py").exists()
    assert (
        PROJECT_ROOT
        / "test/back/high_performance_live_gate_cuda_benchmark_v1.py"
    ).is_file()

    forbidden_tokens = {
        "Lower2C",
        "CudaLowering",
        "GateIntentBundle",
        "lowering_backend",
        "gate2_lowering_policy",
        "source_lowering",
        "replay_lowering",
        "LOWER2C_PROTOCOL",
        "CUDA_LOWERING_PROTOCOL",
    }
    active_paths = [
        *active_gate2.glob("*.py"),
        PROJECT_ROOT / "test" / "scrc_live_e2e.py",
        PROJECT_ROOT / "test" / "scrc_heldout_batch.py",
        PROJECT_ROOT / "test" / "high_performance_live_batch.py",
    ]
    for path in active_paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source, (path, token)


def test_importing_active_gate2_does_not_load_archived_or_backend_lowering() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import intentir.verify.gate2 as gate2; "
                "assert not hasattr(gate2, 'Lower2CSpec'); "
                "assert not hasattr(gate2, 'CudaLoweringSpec'); "
                "assert not any(name.startswith("
                "'intentir.verify.back.gate2_production_lowering_v1') "
                "for name in sys.modules); "
                "assert 'intentir.backend.cpu' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_inner_replay_repair_is_archived_and_absent_from_active_gate1() -> None:
    replay = importlib.import_module("intentir.verify.gate1.replay")
    forbidden_parameters = {
        "base_candidate",
        "control_candidate",
        "repair_candidate",
    }

    for writer in (replay.ResponsesReplayWriter, replay.StaticReplayWriter):
        parameters = set(inspect.signature(writer.generate).parameters)
        assert not (parameters & forbidden_parameters)

    forbidden_protocol_tokens = {
        "CURRENT_REPLAY_CANDIDATE",
        "CONTROLLED_REPLAY_UPDATE",
        "base_generation_id",
    }
    for path in (PYTHON_ROOT / "intentir" / "verify" / "gate1").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden_protocol_tokens:
            assert token not in source, (path, token)

    archived = (
        PYTHON_ROOT
        / "intentir"
        / "verify"
        / "back"
        / "legacy_v0"
        / "replay_repair_prompt.py"
    )
    archived_source = archived.read_text(encoding="utf-8")
    assert "CURRENT_REPLAY_CANDIDATE" in archived_source
    assert "CONTROLLED_REPLAY_UPDATE" in archived_source


def test_archived_replay_repair_helper_has_no_active_importer() -> None:
    active_roots = (
        PYTHON_ROOT / "intentir" / "verify" / "gate1",
        PYTHON_ROOT / "intentir" / "verify" / "gate2",
        PYTHON_ROOT / "intentir" / "extraction" / "lifters",
    )
    forbidden_tokens = {
        "replay_repair_prompt",
        "augment_replay_repair_prompt",
        "CURRENT_REPLAY_CANDIDATE",
        "CONTROLLED_REPLAY_UPDATE",
    }

    for root in active_roots:
        for path in root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in source, (path, token)


def test_active_entrypoints_expose_no_replay_repair_era_controls() -> None:
    """The removed subsystem must not survive as inert active-schema fields."""

    forbidden_tokens = {
        '"replay_code_repair"',
        '"maximum_replay_repairs"',
        '"pi_ablation_replay_repairs"',
        '"pi_ablation_replay_code_repair"',
        "--rewrite-repairs",
    }
    active_entrypoints = [
        PROJECT_ROOT / "test" / "scrc_live_e2e.py",
        PROJECT_ROOT / "test" / "scrc_heldout_batch.py",
    ]

    for path in active_entrypoints:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source, (path, token)


def test_legacy_internal_consumers_use_the_explicit_archive_path() -> None:
    legacy_consumers = [
        PYTHON_ROOT / "intentir" / "extraction" / "data" / "corpus.py",
        PYTHON_ROOT / "intentir" / "extraction" / "data" / "run_universe.py",
        PYTHON_ROOT
        / "intentir"
        / "extraction"
        / "lifters"
        / "cuda_frontend.py",
    ]

    for path in legacy_consumers:
        source = path.read_text(encoding="utf-8")
        assert "intentir.verify.back.legacy_v0.interpret" in source, path
        assert "from intentir.verify.interpret" not in source, path


def test_historical_probe_drivers_are_outside_current_live_entrypoints() -> None:
    archived = [
        PROJECT_ROOT / "test/back/scrc_gate1_capability_v5.py",
        PROJECT_ROOT / "test/back/scrc_probe_compile_v0.py",
        PROJECT_ROOT / "test/back/scrc_probe_compare_v0.py",
        PROJECT_ROOT / "test/back/scrc_probe_replays_v0.py",
    ]
    removed_roots = [
        PROJECT_ROOT / "test/scrc_gate1_capability.py",
        PROJECT_ROOT / "test/scrc_probe_compile.py",
        PROJECT_ROOT / "test/scrc_probe_compare.py",
        PROJECT_ROOT / "test/scrc_probe_replays.py",
    ]
    assert all(path.is_file() for path in archived)
    assert all(not path.exists() for path in removed_roots)

    for path in (
        PROJECT_ROOT / "test/scrc_live_e2e.py",
        PROJECT_ROOT / "test/scrc_heldout_batch.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "test/back" not in source, path
        assert "scrc_gate1_capability_v5" not in source, path
        assert "scrc_probe_replays_v0" not in source, path
