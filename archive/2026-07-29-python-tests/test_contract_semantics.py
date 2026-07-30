from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from intentir.frontend.contract_expr_mlir import (
    ContractMLIRLoweringError,
    lower_contract_index,
    lower_typed_contract_program,
)
from intentir.ir.canonical_j import (
    ContractSort,
    parse_contract_expr,
    parse_index_expression,
)
from intentir.ir.contract_semantics import (
    ContractSemanticsError,
    TypedContractProgram,
    evaluate_contract_expr,
    evaluate_contract_index,
    infer_reference_sorts,
)


def _integer(expression: str):
    return parse_contract_expr(
        expression,
        expected=ContractSort.INTEGER,
        context="test integer expression",
    )


def _boolean(expression: str):
    return parse_contract_expr(
        expression,
        expected=ContractSort.BOOLEAN,
        context="test boolean expression",
    )


@pytest.mark.parametrize(
    ("text", "bindings", "expected"),
    [
        ("a + b * 3 - 2", {"a": 5, "b": 4}, 15),
        ("-a // b", {"a": 7, "b": 3}, -3),
        ("-a % b", {"a": 7, "b": 3}, 2),
        ("ceil_div(a, b)", {"a": 7, "b": 3}, 3),
        ("min(a, b)", {"a": 7, "b": 3}, 3),
        ("max(a, b)", {"a": 7, "b": 3}, 7),
    ],
)
def test_integer_contract_semantics(
    text: str, bindings: dict[str, int], expected: int
) -> None:
    assert evaluate_contract_expr(_integer(text), bindings) == expected


def test_contract_semantics_do_not_invent_a_machine_width() -> None:
    huge = 1 << 130
    assert evaluate_contract_expr(_integer("a * 2 + 1"), {"a": huge}) == huge * 2 + 1


def test_boolean_contract_semantics_are_typed() -> None:
    expression = _boolean("enabled and not disabled and 0 <= lane < extent")
    program = TypedContractProgram.compile(
        expression,
        {
            "enabled": ContractSort.BOOLEAN,
            "disabled": ContractSort.BOOLEAN,
            "lane": ContractSort.INTEGER,
            "extent": ContractSort.INTEGER,
        },
        expected_sort=ContractSort.BOOLEAN,
    )

    assert program.evaluate(
        {"enabled": True, "disabled": False, "lane": 3, "extent": 8}
    ) is True
    assert program.evaluate(
        {"enabled": True, "disabled": False, "lane": 8, "extent": 8}
    ) is False


def test_reference_sorts_are_uniquely_inferred_from_grammar_context() -> None:
    expression = _boolean("enabled and 0 <= lane < extent")
    assert infer_reference_sorts(
        expression,
        expected_sort=ContractSort.BOOLEAN,
    ) == {
        "enabled": ContractSort.BOOLEAN,
        "extent": ContractSort.INTEGER,
        "lane": ContractSort.INTEGER,
    }


def test_reference_sort_inference_rejects_conflicting_uses() -> None:
    expression = _boolean("value and value < 4")
    with pytest.raises(ContractSemanticsError, match="conflicting sorts"):
        infer_reference_sorts(
            expression,
            expected_sort=ContractSort.BOOLEAN,
        )


@pytest.mark.parametrize("factors", [(2, 3, 4), (1, 4, 6), (2, 2, 2)])
def test_mixed_radix_is_executed_as_generic_index_ast(
    factors: tuple[int, int, int]
) -> None:
    d0, d1, d2 = factors
    index = parse_index_expression(
        "[flat // (d1 * d2), (flat // d2) % d1, flat % d2]"
    )
    for flat in range(d0 * d1 * d2):
        digits = evaluate_contract_index(
            index,
            {"flat": flat, "d0": d0, "d1": d1, "d2": d2},
        )
        assert digits[0] * d1 * d2 + digits[1] * d2 + digits[2] == flat


def test_alpha_renaming_does_not_change_index_semantics() -> None:
    left = parse_index_expression("[flat // width, flat % width]")
    right = parse_index_expression("[lane // columns, lane % columns]")

    for flat in range(24):
        assert evaluate_contract_index(
            left, {"flat": flat, "width": 6}
        ) == evaluate_contract_index(
            right, {"lane": flat, "columns": 6}
        )


def test_contract_semantics_fail_closed_on_bad_environment() -> None:
    expression = _integer("a // b")
    with pytest.raises(ContractSemanticsError, match="lacks bindings"):
        evaluate_contract_expr(expression, {"a": 4})
    with pytest.raises(ContractSemanticsError, match="operand sorts|integer value"):
        evaluate_contract_expr(expression, {"a": True, "b": 2})
    with pytest.raises(ContractSemanticsError, match="divides by zero"):
        evaluate_contract_expr(expression, {"a": 4, "b": 0})


def test_generic_index_mlir_lowering_has_no_radix_or_rank_recipe() -> None:
    index = parse_index_expression(
        "[flat // (d1 * d2), (flat // d2) % d1, flat % d2]"
    )
    fragment = lower_contract_index(
        index,
        {
            "flat": ContractSort.INTEGER,
            "d1": ContractSort.INTEGER,
            "d2": ContractSort.INTEGER,
        },
        {"flat": "%arg0", "d1": "%arg1", "d2": "%arg2"},
        integer_type="index",
    )
    text = "\n".join(fragment.operations)

    assert len(fragment.results) == 3
    assert "arith.floordivsi" in text
    assert "arith.muli" in text
    assert "arith.subi" in text
    assert "mixed_radix" not in text
    assert any(item.kind == "nonzero_divisor" for item in fragment.obligations)
    assert any(item.kind == "representable" for item in fragment.obligations)


def test_alpha_renamed_contracts_lower_to_the_same_arith_shape() -> None:
    left = TypedContractProgram.compile(
        _integer("flat // width + flat % width"),
        {"flat": ContractSort.INTEGER, "width": ContractSort.INTEGER},
    )
    right = TypedContractProgram.compile(
        _integer("lane // columns + lane % columns"),
        {"lane": ContractSort.INTEGER, "columns": ContractSort.INTEGER},
    )
    left_mlir = lower_typed_contract_program(
        left, {"flat": "%arg0", "width": "%arg1"}, integer_type="index"
    )
    right_mlir = lower_typed_contract_program(
        right, {"lane": "%arg0", "columns": "%arg1"}, integer_type="index"
    )

    assert left_mlir.operations == right_mlir.operations
    assert left_mlir.results == right_mlir.results
    assert left_mlir.result_types == right_mlir.result_types
    assert [item.kind for item in left_mlir.obligations] == [
        item.kind for item in right_mlir.obligations
    ]


def test_boolean_contract_lowers_to_closed_i1_program() -> None:
    program = TypedContractProgram.compile(
        _boolean("enabled and 0 <= lane < extent"),
        {
            "enabled": ContractSort.BOOLEAN,
            "lane": ContractSort.INTEGER,
            "extent": ContractSort.INTEGER,
        },
    )
    fragment = lower_typed_contract_program(
        program,
        {"enabled": "%arg0", "lane": "%arg1", "extent": "%arg2"},
        integer_type="index",
    )

    assert fragment.result_types == ("i1",)
    assert sum("arith.cmpi" in item for item in fragment.operations) == 2
    assert any("arith.andi" in item for item in fragment.operations)


def test_emitted_contract_arith_is_valid_mlir_when_opt_is_available() -> None:
    root = Path(__file__).resolve().parents[2]
    intent_opt = root / "build/tools/intent-opt/intent-opt"
    if not intent_opt.exists():
        pytest.skip("intent-opt not built")
    index = parse_index_expression("[flat // width, flat % width]")
    fragment = lower_contract_index(
        index,
        {"flat": ContractSort.INTEGER, "width": ContractSort.INTEGER},
        {"flat": "%arg0", "width": "%arg1"},
        integer_type="index",
    )
    module = (
        "module {\n"
        "  func.func @contract(%arg0: index, %arg1: index) "
        "-> (index, index) {\n"
        f"{fragment.indented(4)}\n"
        f"    return {', '.join(fragment.results)} : index, index\n"
        "  }\n"
        "}\n"
    )
    result = subprocess.run(
        [str(intent_opt)],
        input=module,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_mlir_lowering_fails_closed_on_missing_reference_values() -> None:
    index = parse_index_expression("[flat // width]")
    with pytest.raises(ContractMLIRLoweringError, match="lacks MLIR values"):
        lower_contract_index(
            index,
            {"flat": ContractSort.INTEGER, "width": ContractSort.INTEGER},
            {"flat": "%arg0"},
            integer_type="index",
        )
