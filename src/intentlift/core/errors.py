class LiftError(Exception):
    def __init__(self, kind, message, **details):
        super().__init__(message)
        self.kind = kind
        self.details = details

    def record(self):
        return {"ok": False, "kind": self.kind, "message": str(self), **self.details}


def require(condition, message, kind="capability_missing", **details):
    if not condition:
        raise LiftError(kind, message, **details)
