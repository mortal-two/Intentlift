"""Typed leaf calls shared by pointwise CUDA capture rules."""
from intentlift.core.errors import require
from ._cuda_ast import callee, qual_type


MATH = {
    "exp2f": ("exp2", 1),
    "expf": ("exp", 1),
    "logf": ("log", 1),
    "rsqrtf": ("rsqrt", 1),
    "max": ("maximum_num", 2),
    "min": ("minimum_num", 2),
}


def capture_math(node, expression):
    function, operands = callee(node)
    name = function["referencedDecl"]["name"]
    require(name in MATH, "missing f32 CUDA math capability", cuda_function=name)
    intent, arity = MATH[name]
    require(len(operands) == arity and all(qual_type(arg) == "float" for arg in operands),
            "CUDA math requires f32 operands", cuda_function=name)
    return {"op": "math", "name": intent, "args": [expression(arg) for arg in operands]}
