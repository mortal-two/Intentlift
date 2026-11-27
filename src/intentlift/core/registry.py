"""Representation contracts and registered stages; no product-specific semantics."""
from dataclasses import dataclass
from .errors import LiftError, require


@dataclass(frozen=True)
class Stage:
    name: str
    consumes: str
    produces: str
    run: object


class Registry:
    def __init__(self):
        self.representations = {}
        self.stages = {}
        self.binary = {}
        self.locations = {}
        self.owner = None

    def _add(self, table, name, value):
        require(name not in table, f"duplicate registration: {name}", "extension_invalid")
        table[name] = value
        self.locations[f"{id(table)}:{name}"] = self.owner

    def representation(self, name, verify):
        require(isinstance(name, str) and name and callable(verify), "invalid representation registration", "extension_invalid")
        self._add(self.representations, name, verify)

    def stage(self, name, consumes, produces, run):
        require(all(isinstance(x, str) and x for x in (name, consumes, produces)) and callable(run),
                "invalid stage registration", "extension_invalid")
        self._add(self.stages, name, Stage(name, consumes, produces, run))

    def binary_operator(self, cuda, intent, spelling):
        require(cuda not in self.binary, f"duplicate CUDA operator {cuda}", "extension_invalid")
        self.binary[cuda] = {"intent": intent, "spelling": spelling, "component": self.owner}

    def plan(self, pipeline):
        current = "cuda_request"
        stages = []
        for name in pipeline:
            require(name in self.stages, f"unregistered stage {name}", "extension_invalid")
            stage = self.stages[name]
            require(stage.consumes == current, f"invalid stage connection at {name}", "extension_invalid")
            require(stage.produces in self.representations, f"missing representation verifier {stage.produces}", "extension_invalid")
            stages.append(stage)
            current = stage.produces
        require(current == "intentdsl_source", "pipeline must produce IntentDSL source", "extension_invalid")
        return stages

    def execute(self, pipeline, value, context):
        for stage in self.plan(pipeline):
            try:
                value = stage.run(value, context, self)
                valid = self.representations[stage.produces](value)
                require(valid is not False, f"representation {stage.produces} rejected", "extension_invalid")
            except LiftError as exc:
                exc.details.setdefault("stage", stage.name)
                raise
        return value

    def describe(self):
        return {
            "representations": list(self.representations),
            "stages": [{"name": s.name, "consumes": s.consumes, "produces": s.produces,
                        "component": self.locations[f"{id(self.stages)}:{s.name}"]} for s in self.stages.values()],
            "binary_operators": self.binary,
        }
