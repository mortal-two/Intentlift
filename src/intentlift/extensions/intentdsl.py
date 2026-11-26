"""Emit author-language source from the checked logical member representation."""
import ast
from intentlift.core.errors import require


def emit(program, context, registry):
    directions = {"input": "In", "output": "Out", "inout": "InOut"}
    annotations = [f"arg{i}: I.{directions[p['mode']]}[I.f32, ({program['extent']},)]"
                   for i, p in enumerate(program["parameters"])]

    def expr(node):
        if node["op"] == "read":
            return f"arg{node['slot']}[i]"
        require("spelling" in node and len(node.get("args", [])) == 2, "missing expression emitter")
        return f"({expr(node['args'][0])} {node['spelling']} {expr(node['args'][1])})"

    return ("import intentdsl as I\n\n@I.kernel\ndef lifted(" + ", ".join(annotations) + "):\n"
            + f"    for i in I.parallel(arg0.dim(0)):\n        arg{program['store']}[i] = {expr(program['expression'])}\n")


def verify(source):
    require(isinstance(source, str), "IntentDSL source must be text", "extension_invalid")
    ast.parse(source)


def register(registry):
    registry.representation("intentdsl_source", verify)
    registry.stage("intentdsl.emit", "pointwise", "intentdsl_source", emit)
