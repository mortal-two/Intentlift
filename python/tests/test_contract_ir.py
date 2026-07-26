from __future__ import annotations

import copy

import pytest

from direct_fixtures import make_copy_artifact
from intentir.ir.canonical_j import ContractSort, elaborate_j
from intentir.ir.contract_ir import (
    CONTRACT_IR_PROTOCOL,
    ContractIRProjectionError,
    project_contract_ir,
)
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    EffectKind,
    EffectSpec,
    ParameterKind,
    ParameterMode,
    ResultRelationKind,
    ResultRelationSpec,
    SignatureParameter,
)


def _indexed_copy_j(index_expression: str = "[k0 + 1]"):
    j = make_copy_artifact().j
    j.boundary = Boundary(
        domains={"d0": "0 <= k0 and k0 < x0"},
        accesses=[
            BoundaryAccess(
                "a0",
                AccessKind.LOAD,
                "t0",
                node_ref="n0",
                index_expression=index_expression,
            ),
            BoundaryAccess(
                "a1",
                AccessKind.STORE,
                "t1",
                node_ref="n0",
                index_expression=index_expression,
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
            EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
        ],
        result_relations=[
            ResultRelationSpec("t1", ResultRelationKind.EXACT),
        ],
    )
    return j


def _mixed_radix_copy_j():
    j = make_copy_artifact().j
    for tensor in [*j.intent.inputs, *j.intent.outputs]:
        tensor.shape = ["x0", "x1", "x2"]
    j.signature.parameters[0].rank = 3
    j.signature.parameters[1].rank = 3
    j.signature.parameters.extend(
        [
            SignatureParameter(
                "p3",
                ParameterKind.SCALAR,
                ParameterMode.VALUE,
                "i32",
                logical_ref="x1",
            ),
            SignatureParameter(
                "p4",
                ParameterKind.SCALAR,
                ParameterMode.VALUE,
                "i32",
                logical_ref="x2",
            ),
        ]
    )
    coordinate = "[k0 // (x1 * x2), (k0 // x2) % x1, k0 % x2]"
    j.boundary = Boundary(
        domains={"d0": "0 <= k0 and k0 < ((x0 * x1) * x2)"},
        accesses=[
            BoundaryAccess(
                "a0", AccessKind.LOAD, "t0", node_ref="n0",
                index_expression=coordinate,
            ),
            BoundaryAccess(
                "a1", AccessKind.STORE, "t1", node_ref="n0",
                index_expression=coordinate,
            ),
        ],
        effects=[
            EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
            EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
        ],
    )
    return j


def test_contract_ir_preserves_generic_index_ast_without_admission_matching() -> None:
    projection = project_contract_ir(_indexed_copy_j())

    assert projection.to_json()["protocol"] == CONTRACT_IR_PROTOCOL
    assert projection.accesses[0].index[0].expression.op == "add"
    assert dict(projection.reference_sorts)["k0"] is ContractSort.INTEGER
    assert dict(projection.reference_sorts)["x0"] is ContractSort.INTEGER
    assert not hasattr(projection, "admitted")


def test_contract_ir_preserves_every_non_expression_boundary_relation() -> None:
    canonical = elaborate_j(_indexed_copy_j("[k0]"))
    authored_boundary = copy.deepcopy(canonical.boundary.to_json())
    authored_boundary.pop("domains")
    authored_boundary.pop("accesses")

    projection = project_contract_ir(canonical)

    assert projection.remaining_contract["boundary_relations"] == authored_boundary
    assert projection.remaining_contract["node_ids"] == canonical.node_ids
    assert "kernel_id" not in projection.to_json()


def test_contract_ir_treats_mixed_radix_as_an_ordinary_typed_formula() -> None:
    projection = project_contract_ir(_mixed_radix_copy_j())
    first_index = projection.accesses[0].index

    assert [component.expression.op for component in first_index] == [
        "floordiv",
        "mod",
        "mod",
    ]
    assert "mixed_radix" not in str(projection.to_json())


def test_contract_ir_is_not_a_target_capability_or_bijection_oracle() -> None:
    j = _mixed_radix_copy_j()
    j.boundary.accesses[0].index_expression = (
        "[k0 // (x1 * x2), k0 % x1, k0 % x2]"
    )

    projection = project_contract_ir(j)

    assert projection.accesses[0].index[1].expression.op == "mod"


def test_contract_ir_fails_closed_on_cross_program_sort_conflict() -> None:
    j = _indexed_copy_j("[value]")
    j.boundary.domains = {"d0": "value and value < 4"}

    with pytest.raises(ContractIRProjectionError, match="conflicting sorts"):
        project_contract_ir(j)
