from __future__ import annotations

from pathlib import Path


def test_retired_control_plane_has_no_active_python_reference() -> None:
    root = Path(__file__).resolve().parents[2]
    tokens = [
        "Rule" + "Card",
        "Rule" + "Registry",
        "compose_" + "replay_plan",
        "materialize_" + "replay_plan",
        "Strategy" + "Spec",
        "compose_" + "provenance",
        "Intent" + "Program",
        "Logical" + "ABI",
        "Logical" + "Interface",
        "program_" + "adapter",
        "run_" + "semantic_interventions",
        "max_" + "semantic_interventions",
        "reduction_" + "mask_placement",
        "Replay" + "Plan",
        "Replay" + "Step",
        "Pi" + "Ablation",
        "pi_" + "ablations",
        "composer_" + "shadow",
        "rule_registry_" + "digest",
        "strategy_" + "digest",
        "rule_" + "release",
        "Strategy" + "Binding",
        "Strategy" + "Choice",
        "Strategy" + "Hint",
        "Logical" + "Argument",
        "Gate1Specialization" + "Family",
        "fixed_c_" + "family",
        "intent_program_" + "scalar_cpu_mlir_v1",
        "intent_layer_" + "c_json",
        "selected_" + "rule_cards",
    ]
    roots = [
        root / "python" / "intentir", root / "python" / "tests",
        root / "tools", root / "test",
    ]
    offenders: list[str] = []
    for active_root in roots:
        if not active_root.exists():
            continue
        for path in active_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in tokens:
                if token in text:
                    offenders.append(f"{path.relative_to(root)}: {token}")
    assert not offenders, "\n".join(offenders)
