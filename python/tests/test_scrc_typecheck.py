from __future__ import annotations

import pytest

from direct_fixtures import make_copy_artifact, make_reduce_artifact
from intentir.ir.functional_boundary import project_functional_boundary
from intentir.ir.model import IntentOp, IntentReturn, Tensor
from intentir.verify.gate1.model import (
    AccessKind,
    BoundaryAccess,
    BroadcastSpec,
    ConsumerExtension,
    EffectKind,
    EffectSpec,
    ParameterKind,
    ParameterMode,
    RealizationAuthorization,
    RefinementKind,
    ResultRelationKind,
    ResultRelationSpec,
    ScheduleSlot,
    SchemaError,
    SignatureParameter,
    TypedOutsideValue,
)
from intentir.verify.gate1.pi import RealizationWitness
from intentir.verify.gate1.prompt import build_normal_rewriter_view
from intentir.verify.typecheck import (
    TYPECHECK_PROTOCOL,
    TypeCheckErrorCode,
    type_check,
)


def _codes(artifact) -> set[TypeCheckErrorCode]:
    return {
        item.code
        for item in type_check(artifact.j, artifact.c, artifact.pi).diagnostics
    }


def test_valid_dependent_witness_passes_before_replay() -> None:
    artifact = make_reduce_artifact()
    report = type_check(artifact.j, artifact.c, artifact.pi)
    assert report.passed
    assert report.protocol == TYPECHECK_PROTOCOL == "intentir.scrc.typecheck.v10"
    assert report.diagnostics == ()
    assert set(build_normal_rewriter_view(artifact)) == {"j", "c", "pi", "u_f"}


def _rank_zero_input_artifact(*, dtype: str = "i64"):
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = [4]
    artifact.j.intent.inputs.append(Tensor("t1", dtype, []))
    artifact.j.intent.outputs[0].name = "t2"
    artifact.j.intent.outputs[0].shape = [4]
    artifact.j.intent.body[0].result = "t2"
    artifact.j.signature.parameters[0].rank = 1
    artifact.j.signature.parameters[1].logical_ref = "t2"
    artifact.j.signature.parameters[1].rank = 1
    scalar = artifact.j.signature.parameters[2]
    scalar.dtype = dtype
    scalar.rank = 0
    scalar.logical_ref = "t1"
    return artifact


def test_typecheck_v7_rejects_dead_rank_zero_logical_input() -> None:
    artifact = _rank_zero_input_artifact()

    report = type_check(artifact.j, artifact.c, artifact.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.LOGICAL_INPUT_UNUSED
    ]

    assert not report.passed
    assert [item.path for item in matching] == [
        "j.signature.parameters[2].logical_ref"
    ]
    assert matching[0].ref == "t1"


def test_typecheck_v7_does_not_count_a_dead_node_as_output_liveness() -> None:
    artifact = _rank_zero_input_artifact()
    artifact.j.intent.body.append(
        IntentOp.from_json(
            {
                "id": "n1",
                "op": "map_unary",
                "result": "v0",
                "operands": ["t1"],
                "attrs": {"fn": "identity"},
            }
        )
    )
    artifact.j.node_ids["n1"] = "body[1]"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.LOGICAL_INPUT_UNUSED in {
        item.code for item in report.diagnostics
    }


def test_typecheck_v7_accepts_rank_zero_output_dependency() -> None:
    artifact = _rank_zero_input_artifact(dtype="f32")
    operation = artifact.j.intent.body[0]
    operation.op = "map_binary"
    operation.operands = ["t0", "t1"]
    operation.attrs = {"fn": "add"}

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_typecheck_v7_accepts_direct_rank_zero_identity_output() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = []
    artifact.j.intent.outputs[0].shape = []
    artifact.j.intent.body = []
    artifact.j.intent.returns = [IntentReturn("t1", "t0")]
    artifact.j.node_ids = {}
    artifact.j.signature.parameters[0].rank = 0
    artifact.j.signature.parameters[1].rank = 0
    artifact.j.signature.parameters = artifact.j.signature.parameters[:2]
    artifact.pi = RealizationWitness.from_json(
        {
            "regions": [{"id": "r0", "parent": None, "placement": "root"}],
            "instances": [],
            "routes": [],
            "order": [],
            "sharing": [],
            "refinements": [],
        }
    )

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_typecheck_v7_accepts_explicit_boundary_only_rank_zero_use() -> None:
    artifact = _rank_zero_input_artifact(dtype="i1")
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "t0",
            predicate="t1",
            fill_value=0.0,
            node_ref="n0",
            index_expression="[i0]",
        ),
        BoundaryAccess(
            "a1",
            AccessKind.STORE,
            "t2",
            predicate="t1",
            node_ref="n0",
            index_expression="[i0]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t2", "n0", "e1", "a1"),
    ]

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_typecheck_v7_accepts_representation_only_scalar_without_logical_ref() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = [4]
    artifact.j.intent.outputs[0].shape = [4]
    representation = artifact.j.signature.parameters[2]
    representation.dtype = "i64"
    representation.rank = 0
    representation.logical_ref = None
    payload = artifact.pi.to_json()
    payload["instances"].append(
        {
            "id": "i1",
            "semantic_ref": "p2",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    artifact.pi = RealizationWitness.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()

    artifact.j.boundary.domains = {"d0": "p2"}
    rejected = type_check(artifact.j, artifact.c, artifact.pi)
    assert TypeCheckErrorCode.REPRESENTATION_CARRIER_J_USE_FORBIDDEN in {
        item.code for item in rejected.diagnostics
    }


def _functional_masked_copy_artifact():
    artifact = make_copy_artifact()
    artifact.j.boundary.domains = {"d0": "0 <= i0 and i0 < x0"}
    artifact.j.boundary.accesses = [
        # This predicate is a physical padding guard over the complete logical
        # extent, not a request to drop predicate semantics.
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "t0",
            predicate="i0 < x0",
            fill_value=0.0,
            node_ref="n0",
            index_expression="[i0]",
        ),
        BoundaryAccess(
            "a1",
            AccessKind.STORE,
            "t1",
            predicate="i0 < x0",
            node_ref="n0",
            index_expression="[i0]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n0", "e1", "a1"),
    ]
    return artifact


def _functional_padded_reduction_artifact():
    artifact = make_reduce_artifact(explicit_accesses=False)
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < x0",
        "d1": "0 <= i1 and i1 < x1",
    }
    artifact.j.boundary.accesses = [
        BoundaryAccess(
            "a0",
            AccessKind.LOAD,
            "t0",
            predicate="i1 < x1",
            fill_value=0.0,
            node_ref="n0",
            index_expression="[i0, i1]",
        ),
        BoundaryAccess(
            "a1",
            AccessKind.STORE,
            "t1",
            node_ref="n1",
            index_expression="[i0]",
        ),
    ]
    artifact.j.boundary.effects = [
        EffectSpec(EffectKind.READ, "t0", "n0", "e0", "a0"),
        EffectSpec(EffectKind.WRITE, "t1", "n1", "e1", "a1"),
    ]
    artifact.j.boundary.reductions[0].tile_axis = 0
    artifact.j.boundary.consumer_extensions = [
        ConsumerExtension(
            "n1",
            0,
            "a0",
            0,
            TypedOutsideValue("f32", 0),
            "two_sided",
        )
    ]
    return artifact


def test_functional_boundary_v6_accepts_masked_and_padded_effect_contracts() -> None:
    masked = _functional_masked_copy_artifact()
    padded = _functional_padded_reduction_artifact()
    masked_effects = masked.j.boundary.to_json()["effects"]
    padded_effects = padded.j.boundary.to_json()["effects"]

    assert type_check(masked.j, masked.c, masked.pi).passed
    assert type_check(padded.j, padded.c, padded.pi).passed
    # TypeCheck validates the complete model-authored J; it never erases the
    # physical guard/fill or manufactures/rebinds an effect relation.
    assert masked.j.boundary.to_json()["effects"] == masked_effects
    assert padded.j.boundary.to_json()["effects"] == padded_effects


@pytest.mark.parametrize(
    "guard",
    ["flag", "not flag"],
)
def test_typecheck_v7_does_not_ban_guard_bearing_access(guard) -> None:
    artifact = _functional_masked_copy_artifact()
    artifact.j.intent.inputs.append(Tensor("flag", "i1", []))
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            "i1",
            0,
            logical_ref="flag",
        )
    )
    for access in artifact.j.boundary.accesses:
        access.predicate = f"{guard} and i0 < x0"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_consumer_extension_is_required_only_when_padding_reaches_consumer_axis() -> None:
    non_reduced_axis = _functional_padded_reduction_artifact()
    non_reduced_axis.j.boundary.accesses[0].predicate = "i0 < x0"
    non_reduced_axis.j.boundary.consumer_extensions = []

    report = type_check(
        non_reduced_axis.j,
        non_reduced_axis.c,
        non_reduced_axis.pi,
    )

    assert report.passed, report.to_json()


def test_logical_guard_does_not_become_physical_consumer_padding() -> None:
    artifact = _functional_padded_reduction_artifact()
    artifact.j.intent.inputs.append(Tensor("flag", "i1", []))
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            "i1",
            0,
            logical_ref="flag",
        )
    )
    artifact.j.boundary.accesses[0].predicate = "flag and i1 < x1"
    artifact.j.boundary.consumer_extensions = []

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_typecheck_v7_rejects_padded_mean_without_logical_count() -> None:
    artifact = _functional_padded_reduction_artifact()
    artifact.j.intent.body[1].attrs["fn"] = "mean"
    artifact.j.boundary.consumer_extensions = []

    report = type_check(artifact.j, artifact.c, artifact.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.PADDED_MEAN_COUNT_UNPROVEN
    ]

    assert not report.passed
    assert [item.path for item in matching] == [
        "j.intent.intent_function.body[1].attrs.fn"
    ]
    assert matching[0].ref == "n1"
    assert "logical valid count" in matching[0].detail


@pytest.mark.parametrize(
    "predicate",
    ["flag and i1 < x1", "not flag and i1 < x1"],
)
def test_typecheck_v7_does_not_reclassify_guarded_mean_as_padding(
    predicate: str,
) -> None:
    artifact = _functional_padded_reduction_artifact()
    artifact.j.intent.body[1].attrs["fn"] = "mean"
    artifact.j.boundary.consumer_extensions = []
    artifact.j.intent.inputs.append(Tensor("flag", "i1", []))
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            "i1",
            0,
            logical_ref="flag",
        )
    )
    artifact.j.boundary.accesses[0].predicate = predicate

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    with pytest.raises(NotImplementedError, match="guard-bearing"):
        project_functional_boundary(artifact.j)


def test_typecheck_v7_does_not_reclassify_cross_axis_mean_as_padding() -> None:
    artifact = _functional_padded_reduction_artifact()
    artifact.j.intent.body[1].attrs["fn"] = "mean"
    artifact.j.boundary.consumer_extensions = []
    artifact.j.boundary.accesses[0].predicate = "i1 < i0"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    with pytest.raises(NotImplementedError, match="cross-axis"):
        project_functional_boundary(artifact.j)


def test_typecheck_v7_accepts_dense_mean() -> None:
    artifact = make_reduce_artifact(explicit_accesses=False)
    artifact.j.intent.body[1].attrs["fn"] = "mean"
    artifact.j.boundary.consumer_extensions = []

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_typecheck_v7_does_not_blanket_ban_mean_with_unrelated_mask() -> None:
    artifact = _functional_padded_reduction_artifact()
    artifact.j.intent.body[1].attrs["fn"] = "mean"
    artifact.j.boundary.consumer_extensions = []
    artifact.j.boundary.accesses[0].predicate = "i0 < x0"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    assert TypeCheckErrorCode.PADDED_MEAN_COUNT_UNPROVEN not in {
        item.code for item in report.diagnostics
    }


def test_typecheck_v7_does_not_guess_padding_through_contraction() -> None:
    artifact = _functional_padded_reduction_artifact()
    contraction = artifact.j.intent.body[0]
    contraction.op = "contraction"
    contraction.operands = ["t0", "t0"]
    contraction.attrs = {"subscripts": "mk,nk->mn"}
    contraction.type = {"dtype": "f32", "shape": ["x0", "x0"]}
    artifact.j.intent.body[1].attrs["fn"] = "mean"
    artifact.j.boundary.consumer_extensions = []

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    assert TypeCheckErrorCode.PADDED_MEAN_COUNT_UNPROVEN not in {
        item.code for item in report.diagnostics
    }


@pytest.mark.parametrize(
    ("mutation", "path_suffix"),
    [
        ("missing_domain", ".predicate"),
        ("wrong_extent", ".predicate"),
        ("predicate_axis_not_indexed", ".predicate"),
    ],
)
def test_functional_boundary_v6_rejects_recognized_domain_contradictions(
    mutation, path_suffix
) -> None:
    artifact = _functional_padded_reduction_artifact()
    if mutation == "missing_domain":
        del artifact.j.boundary.domains["d1"]
    elif mutation == "wrong_extent":
        artifact.j.boundary.accesses[0].predicate = "i1 < 7"
    elif mutation == "predicate_axis_not_indexed":
        artifact.j.boundary.accesses[0].index_expression = "[i0, j0]"
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)

    report = type_check(artifact.j, artifact.c, artifact.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.FUNCTIONAL_BOUNDARY_INCOMPLETE
    ]

    assert not report.passed
    assert matching
    assert all(item.path.endswith(path_suffix) for item in matching)
    assert all(item.ref == "a0" for item in matching)

    if mutation == "missing_domain":
        with pytest.raises(ValueError, match=matching[0].detail):
            project_functional_boundary(artifact.j)


def test_functional_boundary_v6_does_not_ban_general_logical_predicates() -> None:
    artifact = _functional_masked_copy_artifact()
    artifact.j.intent.inputs[0].shape = ["x0", "x1"]
    artifact.j.intent.outputs[0].shape = ["x0", "x1"]
    artifact.j.signature.parameters[0].rank = 2
    artifact.j.signature.parameters[1].rank = 2
    for access in artifact.j.boundary.accesses:
        access.index_expression = "[i0, i1]"
        access.predicate = "i0 < i1"
    artifact.j.boundary.domains = {
        "d0": "0 <= i0 and i0 < x0",
        "d1": "0 <= i1 and i1 < x1",
    }

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    assert TypeCheckErrorCode.FUNCTIONAL_BOUNDARY_INCOMPLETE not in {
        item.code for item in report.diagnostics
    }


def test_reduce_contract_rejects_kind_alias_and_missing_canonical_attrs() -> None:
    artifact = make_reduce_artifact()
    artifact.j.intent.body[1].attrs = {
        "kind": "sum",
        "axes": [1],
        "keepdims": False,
    }

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID
    ]
    assert {item.path for item in matching} == {
        "j.intent.intent_function.body[1].attrs.assoc",
        "j.intent.intent_function.body[1].attrs.fn",
        "j.intent.intent_function.body[1].attrs.kind",
    }
    assert any("never translated by fixed code" in item.detail for item in matching)


def test_reduce_contract_rejects_kind_even_when_it_conflicts_with_fn() -> None:
    artifact = make_reduce_artifact()
    artifact.j.intent.body[1].attrs["kind"] = "max"

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    assert [item.path for item in report.diagnostics] == [
        "j.intent.intent_function.body[1].attrs.kind"
    ]


@pytest.mark.parametrize(
    ("attrs", "invalid_suffix"),
    [
        (
            {"fn": "median", "axes": [1], "assoc": "associative"},
            "attrs.fn",
        ),
        (
            {"fn": "sum", "axes": [], "assoc": "associative"},
            "attrs.axes",
        ),
        (
            {"fn": "sum", "axes": [True], "assoc": "associative"},
            "attrs.axes[0]",
        ),
        (
            {"fn": "sum", "axes": [2], "assoc": "associative"},
            "attrs.axes[0]",
        ),
        (
            {"fn": "sum", "axes": [1, -1], "assoc": "associative"},
            "attrs.axes[1]",
        ),
        (
            {"fn": "sum", "axes": [1], "assoc": "stable"},
            "attrs.assoc",
        ),
        (
            {
                "fn": "sum",
                "axes": [1],
                "assoc": "float_nondeterministic",
                "keepdims": 1,
            },
            "attrs.keepdims",
        ),
    ],
)
def test_reduce_contract_rejects_invalid_closed_fields(
    attrs, invalid_suffix
) -> None:
    artifact = make_reduce_artifact()
    artifact.j.intent.body[1].attrs = attrs

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    assert any(
        item.code is TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID
        and item.path.endswith(invalid_suffix)
        for item in report.diagnostics
    )


def test_reduce_contract_keeps_explicit_keepdims_and_result_shape_check() -> None:
    keepdims = make_reduce_artifact()
    keepdims.j.intent.body[1].attrs["keepdims"] = True
    keepdims.j.intent.outputs[0].shape = ["x0", 1]
    keepdims.j.signature.parameters[1].rank = 2
    keepdims.j.boundary.accesses[1].index_expression = "[v0, 0]"
    assert type_check(keepdims.j, keepdims.c, keepdims.pi).passed

    collapsed = make_reduce_artifact()
    collapsed.j.intent.body[1].attrs["keepdims"] = False
    assert type_check(collapsed.j, collapsed.c, collapsed.pi).passed

    negative_axis = make_reduce_artifact()
    negative_axis.j.intent.body[1].attrs["axes"] = [-1]
    assert type_check(
        negative_axis.j, negative_axis.c, negative_axis.pi
    ).passed

    mismatch = make_reduce_artifact()
    mismatch.j.intent.body[1].attrs["keepdims"] = True
    report = type_check(mismatch.j, mismatch.c, mismatch.pi)
    assert not report.passed
    assert report.diagnostics[0].code is TypeCheckErrorCode.J_ELABORATION_FAILED
    assert "conflicting types" in report.diagnostics[0].detail


def _contraction_artifact():
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0] = Tensor("t0", "f16", [2, 3])
    artifact.j.intent.inputs.append(Tensor("t2", "f16", [3, 4]))
    artifact.j.intent.outputs[0] = Tensor("t1", "f32", [2, 4])
    artifact.j.signature.parameters = [
        SignatureParameter(
            "p0",
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            "f16",
            2,
            logical_ref="t0",
        ),
        SignatureParameter(
            "p1",
            ParameterKind.POINTER,
            ParameterMode.OUTPUT,
            "f32",
            2,
            logical_ref="t1",
        ),
        SignatureParameter(
            "p2",
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            "f16",
            2,
            logical_ref="t2",
        ),
    ]
    operation = artifact.j.intent.body[0]
    operation.op = "contraction"
    operation.operands = ["t0", "t2"]
    operation.attrs = {"subscripts": "mk,kn->mn"}
    operation.type = {"dtype": "f32", "shape": [2, 4]}
    return artifact


def test_contraction_contract_accepts_canonical_typed_binary_einsum() -> None:
    artifact = _contraction_artifact()

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


@pytest.mark.parametrize(
    ("lhs_shape", "rhs_shape", "result_shape", "subscripts"),
    [
        ([2], [4], [2, 4], "m,n->mn"),
        ([5, 2, 3], [5, 3, 4], [5, 2, 4], "bmk,bkn->bmn"),
    ],
)
def test_contraction_typecheck_does_not_require_the_tl_dot_fast_path(
    lhs_shape, rhs_shape, result_shape, subscripts
) -> None:
    artifact = _contraction_artifact()
    artifact.j.intent.inputs[0].shape = lhs_shape
    artifact.j.intent.inputs[1].shape = rhs_shape
    artifact.j.intent.outputs[0].shape = result_shape
    artifact.j.signature.parameters[0].rank = len(lhs_shape)
    artifact.j.signature.parameters[1].rank = len(result_shape)
    artifact.j.signature.parameters[2].rank = len(rhs_shape)
    operation = artifact.j.intent.body[0]
    operation.attrs = {"subscripts": subscripts}
    operation.type = {"dtype": "f32", "shape": result_shape}

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_contraction_contract_rejects_axis_aliases_without_translation() -> None:
    artifact = _contraction_artifact()
    artifact.j.intent.body[0].attrs = {
        "lhs_contract_axes": [1],
        "rhs_contract_axes": [0],
    }

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID
    ]
    assert {item.path for item in matching} == {
        "j.intent.intent_function.body[0].attrs.lhs_contract_axes",
        "j.intent.intent_function.body[0].attrs.rhs_contract_axes",
        "j.intent.intent_function.body[0].attrs.subscripts",
    }
    assert all(
        "never translated by fixed code" in item.detail
        for item in matching
        if not item.path.endswith(".subscripts")
    )


@pytest.mark.parametrize(
    ("subscripts", "rhs_shape", "result_shape", "detail"),
    [
        ("mk,k...->m", [3, 4], [2], "closed binary einsum grammar"),
        ("mk,kn->mm", [3, 4], [2, 2], "output labels must be unique"),
        ("mk,kn->mp", [3, 4], [2, 4], "unknown labels"),
        ("mk,k->m", [3, 4], [2], "operand 1 labels have rank 1"),
        ("mk,kn->mn", [5, 4], [2, 4], "inconsistent typed extents"),
        ("mk,kn->mn", [3, 4], [2, 5], "does not match subscripts result"),
    ],
)
def test_contraction_contract_rejects_mechanically_invalid_typed_einsum(
    subscripts, rhs_shape, result_shape, detail
) -> None:
    artifact = _contraction_artifact()
    artifact.j.intent.inputs[1].shape = rhs_shape
    artifact.j.intent.outputs[0].shape = result_shape
    operation = artifact.j.intent.body[0]
    operation.attrs = {"subscripts": subscripts}
    operation.type = {"dtype": "f32", "shape": result_shape}

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    assert any(
        item.code is TypeCheckErrorCode.KNOWN_OP_CONTRACT_INVALID
        and detail in item.detail
        for item in report.diagnostics
    ), report.to_json()


def _row_scale_artifact():
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].shape = [2, "x0"]
    artifact.j.intent.outputs[0].shape = [2, "x0"]
    artifact.j.signature.parameters[0].rank = 2
    artifact.j.signature.parameters[1].rank = 2
    artifact.j.intent.inputs.append(Tensor("t2", "f32", [2]))
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.POINTER,
            ParameterMode.INPUT,
            "f32",
            1,
            logical_ref="t2",
        )
    )
    operation = artifact.j.intent.body[0]
    operation.op = "map_binary"
    operation.operands = ["t0", "t2"]
    operation.attrs = {"fn": "mul"}
    operation.type = None
    return artifact


def test_non_numpy_broadcast_requires_an_explicit_j_relation() -> None:
    artifact = _row_scale_artifact()

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    assert report.diagnostics[0].code is TypeCheckErrorCode.J_ELABORATION_FAILED
    assert "non-trailing broadcast must be declared explicitly" in (
        report.diagnostics[0].detail
    )


def test_explicit_nontrailing_broadcast_closes_value_typing() -> None:
    artifact = _row_scale_artifact()
    artifact.j.boundary.broadcasts = [BroadcastSpec("n0", "t2", [1])]

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()


def test_typecheck_rejects_a_graph_with_an_untyped_intermediate() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.body.insert(
        0,
        IntentOp.from_json(
            {
                "id": "n1",
                "op": "opaque_untyped",
                "result": "v0",
                "operands": [],
                "attrs": {},
            }
        ),
    )
    artifact.j.node_ids = {"n1": "body[0]", "n0": "body[1]"}

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert TypeCheckErrorCode.J_VALUE_TYPE_INCOMPLETE in {
        item.code for item in report.diagnostics
    }
    diagnostic = next(
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.J_VALUE_TYPE_INCOMPLETE
    )
    assert diagnostic.ref == "v0"


def test_fill_requires_a_predicated_load_before_replay() -> None:
    orphan = make_reduce_artifact()
    orphan.j.boundary.accesses[0].predicate = None
    orphan.j.boundary.accesses[0].fill_value = 0
    report = type_check(orphan.j, orphan.c, orphan.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.INVALID_ACCESS_FILL
    ]
    assert len(matching) == 1
    assert matching[0].ref == "a0"
    assert "non-null access predicate" in matching[0].detail
    with pytest.raises(SchemaError, match="invalid_access_fill"):
        build_normal_rewriter_view(orphan)

    non_load = make_reduce_artifact()
    store = non_load.j.boundary.accesses[1]
    assert store.kind is AccessKind.STORE and store.predicate is not None
    store.fill_value = 0
    report = type_check(non_load.j, non_load.c, non_load.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.INVALID_ACCESS_FILL
    ]
    assert len(matching) == 1
    assert matching[0].ref == "a1"
    assert "only on a load" in matching[0].detail


def test_dangling_anchor_fails_and_cannot_enter_replay() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["instances"][0]["semantic_ref"] = "n99"
    artifact.pi = RealizationWitness.from_json(payload)

    assert TypeCheckErrorCode.UNRESOLVED_ANCHOR in _codes(artifact)
    with pytest.raises(SchemaError, match="unresolved_anchor"):
        build_normal_rewriter_view(artifact)


def test_route_slot_and_carrier_type_are_checked_against_j() -> None:
    out_of_range = make_reduce_artifact()
    payload = out_of_range.pi.to_json()
    payload["routes"][0]["slots"] = [2]
    out_of_range.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.ROUTE_SLOT_OUT_OF_RANGE in _codes(out_of_range)

    wrong_type = make_reduce_artifact()
    payload = wrong_type.pi.to_json()
    payload["instances"][0]["semantic_ref"] = "p2"
    wrong_type.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.ROUTE_TYPE_MISMATCH in _codes(wrong_type)


def test_route_and_order_form_one_acyclic_dependency_graph() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["routes"].append({"source": "i3", "target": "i0", "slots": []})
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.ROUTE_CYCLE in _codes(artifact)


def test_c_value_type_is_not_an_executable_free_text_escape() -> None:
    artifact = make_reduce_artifact()
    artifact.c.values["c0"] = "code()"
    assert TypeCheckErrorCode.C_VALUE_TYPE_MISMATCH in _codes(artifact)

    artifact.c.slots[0].value_type = "string"
    assert TypeCheckErrorCode.C_VALUE_TYPE_MISMATCH in _codes(artifact)


def test_required_c_slot_must_have_the_actual_harness_valuation() -> None:
    artifact = make_reduce_artifact()
    artifact.c.values.clear()
    assert TypeCheckErrorCode.C_REQUIRED_VALUE_MISSING in _codes(artifact)


def test_c_attachment_uses_closed_subject_dependent_axis_domain() -> None:
    artifact = make_reduce_artifact()
    slot = artifact.c.slots[0]
    slot.subject_ref = "n1"
    slot.axis_ref = 1
    assert type_check(artifact.j, artifact.c, artifact.pi).passed

    view = build_normal_rewriter_view(artifact)
    assert view["c"]["slots"][0]["subject_ref"] == "n1"
    assert view["c"]["slots"][0]["axis_ref"] == 1

    slot.subject_ref = "d1"
    slot.axis_ref = "x1"
    assert type_check(artifact.j, artifact.c, artifact.pi).passed

    slot.subject_ref = "a1"
    slot.axis_ref = 0
    assert type_check(artifact.j, artifact.c, artifact.pi).passed
    slot.axis_ref = "x0"
    assert type_check(artifact.j, artifact.c, artifact.pi).passed

    slot.subject_ref = "n0"
    slot.axis_ref = 1
    assert TypeCheckErrorCode.C_AXIS_INVALID in _codes(artifact)

    slot.subject_ref = "n99"
    slot.axis_ref = None
    assert TypeCheckErrorCode.C_SUBJECT_UNKNOWN in _codes(artifact)


def test_c_axis_requires_an_explicit_closed_subject() -> None:
    with pytest.raises(SchemaError, match="axis_ref requires subject_ref"):
        ScheduleSlot.from_json(
            {
                "name": "c0",
                "kind": "constexpr",
                "role": "tile_extent",
                "axis_ref": 1,
            }
        )


def test_observable_realization_cannot_be_owned_by_c() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["instances"][3]["semantic_ref"] = "c0"
    payload["instances"][3]["mechanism"] = "transfer_direct"
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.OBSERVABLE_REQUIRES_J_ANCHOR in _codes(artifact)


def test_sensitive_atomic_requires_exact_j_authorization() -> None:
    artifact = make_reduce_artifact()
    payload = artifact.pi.to_json()
    payload["instances"][0]["mechanism"] = "update_atomic"
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.REFINEMENT_AUTH_MISSING in _codes(artifact)

    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.NATIVE_ATOMIC, "a0")
    ]
    payload["refinements"] = [
        {"instance": "i0", "kind": "native_atomic", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert type_check(artifact.j, artifact.c, artifact.pi).passed


def test_sensitive_authorization_subject_must_match_refined_anchor() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.NATIVE_ATOMIC, "n0")
    ]
    payload = artifact.pi.to_json()
    payload["instances"][0]["mechanism"] = "update_atomic"
    payload["refinements"] = [
        {"instance": "i0", "kind": "native_atomic", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.REFINEMENT_AUTH_SUBJECT_MISMATCH in _codes(artifact)


def test_typed_authorization_source_must_match_its_j_carrier() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization(
            "u0", RefinementKind.NUMERIC_CAST, "n0", "i32", "f32"
        )
    ]
    payload = artifact.pi.to_json()
    payload["refinements"] = [
        {"instance": "i1", "kind": "numeric_cast", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH in _codes(artifact)


def _integer_widen_artifact(
    *,
    kind: RefinementKind,
    source_dtype: str | None,
    target_dtype: str | None,
):
    artifact = make_reduce_artifact()
    carrier_dtype = source_dtype or (
        "i32" if kind is RefinementKind.SIGNED_WIDEN else "u32"
    )
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3",
            ParameterKind.SCALAR,
            ParameterMode.VALUE,
            carrier_dtype,
            rank=0,
            logical_ref=None,
        )
    )
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization(
            "u0",
            kind,
            "p3",
            source_dtype,
            target_dtype,
        )
    ]
    payload = artifact.pi.to_json()
    payload["instances"].append(
        {
            "id": "i4",
            "semantic_ref": "p3",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    payload["routes"].append(
        {"source": "i4", "target": "i0", "slots": [0]}
    )
    payload["refinements"].append(
        {"instance": "i4", "kind": kind.value, "auth_ref": "u0"}
    )
    artifact.pi = RealizationWitness.from_json(payload)
    return artifact


@pytest.mark.parametrize(
    ("kind", "source_dtype", "target_dtype"),
    [
        (RefinementKind.SIGNED_WIDEN, "i32", "i64"),
        (RefinementKind.UNSIGNED_WIDEN, "u32", "u64"),
    ],
)
def test_authoritative_typecheck_accepts_strict_integer_widen(
    kind: RefinementKind,
    source_dtype: str,
    target_dtype: str,
) -> None:
    artifact = _integer_widen_artifact(
        kind=kind,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
    )

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert report.passed, report.to_json()
    assert report.protocol == "intentir.scrc.typecheck.v10"


@pytest.mark.parametrize(
    ("kind", "source_dtype", "target_dtype"),
    [
        (RefinementKind.SIGNED_WIDEN, "i32", "i32"),
        (RefinementKind.SIGNED_WIDEN, "i64", "i32"),
        (RefinementKind.SIGNED_WIDEN, None, None),
        (RefinementKind.SIGNED_WIDEN, "u32", "u64"),
        (RefinementKind.SIGNED_WIDEN, "f32", "f64"),
        (RefinementKind.UNSIGNED_WIDEN, "i32", "i64"),
    ],
)
def test_authoritative_typecheck_rejects_invalid_integer_widen(
    kind: RefinementKind,
    source_dtype: str | None,
    target_dtype: str | None,
) -> None:
    artifact = _integer_widen_artifact(
        kind=kind,
        source_dtype=source_dtype,
        target_dtype=target_dtype,
    )

    report = type_check(artifact.j, artifact.c, artifact.pi)
    matching = [
        item
        for item in report.diagnostics
        if item.code is TypeCheckErrorCode.REFINEMENT_AUTH_INVALID_WIDEN
    ]

    assert not report.passed
    assert [item.path for item in matching] == [
        "j.boundary.realization_authorizations[0]"
    ]
    assert matching[0].ref == "u0"


def test_absent_none_carrier_cannot_gain_realization_semantics() -> None:
    artifact = make_copy_artifact()
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3", ParameterKind.OPAQUE, ParameterMode.VALUE, "none"
        )
    )
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization(
            "u0", RefinementKind.REASSOCIATION, "p3"
        )
    ]
    payload = artifact.pi.to_json()
    payload["instances"].append(
        {
            "id": "i1",
            "semantic_ref": "p3",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    payload["refinements"] = [
        {"instance": "i1", "kind": "reassociation", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)

    report = type_check(artifact.j, artifact.c, artifact.pi)

    assert not report.passed
    assert [item.code for item in report.diagnostics] == [
        TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
        TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
        TypeCheckErrorCode.ABSENT_CARRIER_SEMANTIC_USE_FORBIDDEN,
    ]
    assert [item.path for item in report.diagnostics] == [
        "j.boundary.realization_authorizations[0].subject_ref",
        "pi.instances[1].semantic_ref",
        "pi.refinements[0].instance",
    ]


def test_ordinary_opaque_carrier_may_still_own_a_realization() -> None:
    artifact = make_copy_artifact()
    artifact.j.signature.parameters.append(
        SignatureParameter(
            "p3", ParameterKind.OPAQUE, ParameterMode.VALUE, "stream"
        )
    )
    payload = artifact.pi.to_json()
    payload["instances"].append(
        {
            "id": "i1",
            "semantic_ref": "p3",
            "region": "r0",
            "mechanism": "realize",
        }
    )
    artifact.pi = RealizationWitness.from_json(payload)

    assert type_check(artifact.j, artifact.c, artifact.pi).passed


def test_numeric_cast_authorization_types_operand_and_result() -> None:
    artifact = make_copy_artifact()
    artifact.j.intent.inputs[0].dtype = "f16"
    artifact.j.signature.parameters[0].dtype = "f16"
    operation = artifact.j.intent.body[0]
    operation.op = "cast"
    operation.attrs = {"to_dtype": "f32"}
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization(
            "u0", RefinementKind.NUMERIC_CAST, "n0", "f16", "f32"
        )
    ]
    payload = artifact.pi.to_json()
    payload["refinements"] = [
        {"instance": "i0", "kind": "numeric_cast", "auth_ref": "u0"}
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert type_check(artifact.j, artifact.c, artifact.pi).passed

    artifact.j.boundary.realization_authorizations[0].source_dtype = "f32"
    assert TypeCheckErrorCode.REFINEMENT_AUTH_TYPE_MISMATCH in _codes(artifact)


def test_floating_transcendental_and_reduction_require_tolerance() -> None:
    reduction = make_reduce_artifact()
    reduction.j.boundary.result_relations = [
        ResultRelationSpec("t1", ResultRelationKind.EXACT)
    ]
    assert TypeCheckErrorCode.RESULT_RELATION_INCOMPATIBLE in _codes(reduction)

    reduction.j.boundary.result_relations = []
    assert TypeCheckErrorCode.RESULT_RELATION_INCOMPATIBLE in _codes(reduction)

    integer = make_reduce_artifact()
    for tensor in [*integer.j.intent.inputs, *integer.j.intent.outputs]:
        tensor.dtype = "i32"
    integer.j.signature.parameters[0].dtype = "i32"
    integer.j.signature.parameters[1].dtype = "i32"
    integer.j.boundary.consumer_extensions[0].outside.dtype = "i32"
    integer.j.boundary.result_relations = [
        ResultRelationSpec("t1", ResultRelationKind.EXACT)
    ]
    assert type_check(integer.j, integer.c, integer.pi).passed

    nonlinear = make_copy_artifact()
    nonlinear.j.intent.body[0].attrs["fn"] = "tanh"
    assert TypeCheckErrorCode.RESULT_RELATION_INCOMPATIBLE in _codes(nonlinear)

    nonlinear.j.boundary.result_relations = [
        ResultRelationSpec("t1", ResultRelationKind.TOLERANCE)
    ]
    assert type_check(nonlinear.j, nonlinear.c, nonlinear.pi).passed

    reassociated = make_copy_artifact()
    reassociated.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.REASSOCIATION, "n0")
    ]
    assert TypeCheckErrorCode.RESULT_RELATION_INCOMPATIBLE in _codes(
        reassociated
    )
    reassociated.j.boundary.result_relations = [
        ResultRelationSpec("t1", ResultRelationKind.TOLERANCE)
    ]
    assert type_check(
        reassociated.j, reassociated.c, reassociated.pi
    ).passed

    identity = make_copy_artifact()
    assert type_check(identity.j, identity.c, identity.pi).passed


def test_global_safe_refinement_is_closed_and_cannot_cite_candidate_auth() -> None:
    artifact = make_reduce_artifact()
    artifact.j.boundary.realization_authorizations = [
        RealizationAuthorization("u0", RefinementKind.REASSOCIATION, "n0")
    ]
    payload = artifact.pi.to_json()
    payload["refinements"] = [
        {
            "instance": "i1",
            "kind": "identity_materialization",
            "auth_ref": "u0",
        }
    ]
    artifact.pi = RealizationWitness.from_json(payload)
    assert TypeCheckErrorCode.GLOBAL_SAFE_AUTH_FORBIDDEN in _codes(artifact)
