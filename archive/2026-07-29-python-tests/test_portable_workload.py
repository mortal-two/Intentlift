from types import SimpleNamespace

import numpy as np

from intentir.backend.portable_workload import (
    decode_workload_recipe,
    encode_workload_recipe,
)
from intentir.verify.workload_adapters import (
    BufferBinding,
    BufferInitializer,
    BufferInputTransform,
    BufferRole,
    TRITON_WORKLOAD_ADAPTER_DERIVED_INPUT_PROTOCOL,
    TritonWorkloadAdapter,
)


def test_portable_workload_rebuilds_seeded_and_derived_inputs() -> None:
    adapter = TritonWorkloadAdapter(
        case_id="high_performance:test_portable_workload",
        buffers=(
            BufferBinding(
                "x",
                BufferRole.INPUT,
                "f32",
                (2, 3),
                initializer=BufferInitializer.RANDOM,
            ),
            BufferBinding(
                "sorted_x",
                BufferRole.INPUT,
                "f32",
                (2, 3),
                source_formal="x",
                input_transform=BufferInputTransform.SORT_DESCENDING_LAST_AXIS,
            ),
            BufferBinding(
                "state",
                BufferRole.INOUT,
                "i32",
                (2,),
                output_key="state_out",
                initializer=BufferInitializer.ZEROS,
            ),
            BufferBinding("y", BufferRole.OUTPUT, "f32", (2, 3), "y"),
        ),
        scalars=(),
        semantic_constexprs=(),
        c_values=(),
        grid=(1,),
        grid_contract=("1",),
        seed=37,
        protocol=TRITON_WORKLOAD_ADAPTER_DERIVED_INPUT_PROTOCOL,
    )
    rng = np.random.default_rng(adapter.seed)
    x = rng.normal(0.0, 0.25, size=(2, 3)).astype(np.float32)
    values = {
        "input_p0": x,
        "input_p1": np.ascontiguousarray(
            np.flip(np.sort(x, axis=-1), axis=-1)
        ),
        "input_p2": np.zeros((2,), dtype=np.int32),
        "logical_p4": np.asarray(7, dtype=np.int32),
    }
    arguments = (
        SimpleNamespace(name="p0", value_key="input_p0"),
        SimpleNamespace(name="p1", value_key="input_p1"),
        SimpleNamespace(name="p2", value_key="input_p2"),
        SimpleNamespace(name="p3", value_key=None),
    )

    recipe = encode_workload_recipe(
        adapter,
        {"x": "p0", "sorted_x": "p1", "state": "p2", "y": "p3"},
        arguments,
        values,
    )
    rebuilt = decode_workload_recipe(recipe)

    assert set(rebuilt) == set(values)
    for key in values:
        np.testing.assert_array_equal(rebuilt[key], values[key])
