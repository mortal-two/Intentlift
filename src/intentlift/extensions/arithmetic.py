"""Binary arithmetic for the checked pointwise f32 capture."""


def register(registry):
    registry.binary_operator("+", "add", "+")
    registry.binary_operator("*", "mul", "*")
