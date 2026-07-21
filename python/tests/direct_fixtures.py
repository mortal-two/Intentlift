from __future__ import annotations

from intentir.ir.model import IntentFunction
from intentir.verify.gate1.model import (
    AccessKind,
    Boundary,
    BoundaryAccess,
    Frontend,
    Gate1Artifact,
    KernelInterface,
    KernelParameter,
    LocationEntry,
    LocationMap,
    ParameterKind,
    ParameterMode,
    ReductionBoundary,
    ScheduleSlot,
    ScheduleSlotKind,
    ScheduleSpec,
    SemanticArchive,
    SourceSpan,
)
from intentir.verify.gate1.pi import RealizationWitness


def make_reduce_artifact(*, explicit_accesses: bool = True) -> Gate1Artifact:
    function = IntentFunction.from_json(
        {
            "name": "scrc_j0",
            "inputs": [{"name": "t0", "dtype": "f32", "shape": ["x0", "x1"]}],
            "outputs": [{"name": "t1", "dtype": "f32", "shape": ["x0"]}],
            "body": [
                {
                    "id": "n0",
                    "op": "map_binary",
                    "result": "v0",
                    "operands": ["t0", "t0"],
                    "attrs": {"fn": "mul"},
                },
                {
                    "id": "n1",
                    "op": "reduce",
                    "result": "t1",
                    "operands": ["v0"],
                    "attrs": {
                        "fn": "sum",
                        "axes": [1],
                        "assoc": "float_nondeterministic",
                    },
                },
            ],
        }
    )
    accesses = (
        [
            BoundaryAccess(
                "a0",
                AccessKind.LOAD,
                "t0",
                predicate="v1 < x1",
                fill_value=0,
                node_ref="n0",
                index_expression="v2",
            ),
            BoundaryAccess(
                "a1",
                AccessKind.STORE,
                "t1",
                predicate="v3 < x0",
                node_ref="n1",
                index_expression="v3",
            ),
        ]
        if explicit_accesses
        else []
    )
    j = SemanticArchive(
        "scrc_k0",
        function,
        {"n0": "body[0]", "n1": "body[1]"},
        Boundary(
            domains={"d0": "v1 < x1"} if explicit_accesses else {},
            accesses=accesses,
            reductions=[
                ReductionBoundary(
                    "n1", "x1", identity=0, masked_contribution=0
                )
            ],
        ),
    )
    interface = KernelInterface(
        "scrc_k0",
        Frontend.TRITON,
        [
            KernelParameter(
                "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                "f32", 2, tensor_ref="t0",
            ),
            KernelParameter(
                "p1", ParameterKind.POINTER, ParameterMode.OUTPUT,
                "f32", 1, tensor_ref="t1",
            ),
            KernelParameter(
                "p2", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"
            ),
            KernelParameter("c0", ParameterKind.CONSTEXPR),
        ],
        ["ceil_div(p2, c0)"],
        function_name="scrc_k0",
    )
    schedule = ScheduleSpec(
        [ScheduleSlot("c0", ScheduleSlotKind.CONSTEXPR, "tile_extent")],
        {"c0": 128},
    )
    pi_payload = {
        "regions": [{"id": "r0", "parent": None, "placement": "root"}],
        "instances": [
            {"id": "i0", "semantic_ref": "a0", "region": "r0", "mechanism": "realize"},
            {"id": "i1", "semantic_ref": "n0", "region": "r0", "mechanism": "realize"},
            {"id": "i2", "semantic_ref": "n1", "region": "r0", "mechanism": "realize"},
            {"id": "i3", "semantic_ref": "a1", "region": "r0", "mechanism": "realize"},
        ],
        "routes": [
            {"source": "i0", "target": "i1", "slots": [0, 1]},
            {"source": "i1", "target": "i2", "slots": [0]},
            {"source": "i2", "target": "i3", "slots": [0]},
        ],
        "order": [
            {"before": "i0", "after": "i1"},
            {"before": "i1", "after": "i2"},
            {"before": "i2", "after": "i3"},
        ],
        "sharing": [{"source": "i0", "consumers": ["i1"]}],
    }
    if not explicit_accesses:
        pi_payload.update(
            instances=[
                {"id": "i0", "semantic_ref": "n0", "region": "r0", "mechanism": "realize"},
                {"id": "i1", "semantic_ref": "n1", "region": "r0", "mechanism": "realize"},
            ],
            routes=[{"source": "i0", "target": "i1", "slots": [0]}],
            order=[{"before": "i0", "after": "i1"}],
            sharing=[],
        )
    pi = RealizationWitness.from_json(pi_payload)
    return Gate1Artifact(
        j,
        interface,
        schedule,
        pi,
        LocationMap(
            "scrc_k0.py",
            [
                LocationEntry(
                    SourceSpan(1, 2),
                    ["n0", "n1"],
                    ["a0", "a1"] if explicit_accesses else [],
                )
            ],
        ),
        source="PRIVATE_SOURCE_CANARY",
    )


def make_copy_artifact() -> Gate1Artifact:
    function = IntentFunction.from_json(
        {
            "name": "scrc_j0",
            "inputs": [{"name": "t0", "dtype": "f32", "shape": ["x0"]}],
            "outputs": [{"name": "t1", "dtype": "f32", "shape": ["x0"]}],
            "body": [
                {
                    "id": "n0",
                    "op": "map_unary",
                    "result": "t1",
                    "operands": ["t0"],
                    "attrs": {"fn": "identity"},
                }
            ],
        }
    )
    return Gate1Artifact(
        SemanticArchive(
            "scrc_k0",
            function,
            {"n0": "body[0]"},
            Boundary(),
        ),
        KernelInterface(
            "scrc_k0",
            Frontend.TRITON,
            [
                KernelParameter(
                    "p0", ParameterKind.POINTER, ParameterMode.INPUT,
                    "f32", 1, tensor_ref="t0",
                ),
                KernelParameter(
                    "p1", ParameterKind.POINTER, ParameterMode.OUTPUT,
                    "f32", 1, tensor_ref="t1",
                ),
                KernelParameter(
                    "p2", ParameterKind.SCALAR, ParameterMode.VALUE, "i32"
                ),
                KernelParameter("c0", ParameterKind.CONSTEXPR),
            ],
            ["ceil_div(p2, c0)"],
            function_name="scrc_k0",
        ),
        ScheduleSpec(
            [ScheduleSlot("c0", ScheduleSlotKind.CONSTEXPR, "tile_extent")],
            {"c0": 1024},
        ),
        RealizationWitness.from_json(
            {
                "regions": [
                    {"id": "r0", "parent": None, "placement": "root"}
                ],
                "instances": [
                    {
                        "id": "i0",
                        "semantic_ref": "n0",
                        "region": "r0",
                        "mechanism": "realize",
                    }
                ],
                "routes": [],
                "order": [],
                "sharing": [],
            }
        ),
        LocationMap(
            "scrc_k0.py",
            [LocationEntry(SourceSpan(1, 1), ["n0"], [])],
        ),
        source="PRIVATE_SOURCE_CANARY",
    )
