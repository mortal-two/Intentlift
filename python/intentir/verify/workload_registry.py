"""Frontend-neutral registry for real SCRC Harness workloads."""

from __future__ import annotations

from pathlib import Path

from intentir.verify.frontend_workloads import (
    FRONTEND_DATASET_HARNESSES,
    FrontendDatasetHarness,
    PreparedFrontendDatasetWorkload,
    prepare_frontend_dataset_workload,
)
from intentir.verify.workload_adapters import (
    TRITON_WORKLOAD_ADAPTERS,
    PreparedTritonWorkload,
    TritonWorkloadAdapter,
    prepare_triton_workload,
)


RegisteredHarness = TritonWorkloadAdapter | FrontendDatasetHarness
PreparedRegisteredWorkload = (
    PreparedTritonWorkload | PreparedFrontendDatasetWorkload
)


def registered_scrc_workloads() -> dict[str, RegisteredHarness]:
    overlap = set(TRITON_WORKLOAD_ADAPTERS) & set(FRONTEND_DATASET_HARNESSES)
    if overlap:
        raise RuntimeError(
            "SCRC workload registries contain duplicate case ids: "
            + ", ".join(sorted(overlap))
        )
    return {**TRITON_WORKLOAD_ADAPTERS, **FRONTEND_DATASET_HARNESSES}


def registered_scrc_case_ids(
    *, frontend: str | None = None
) -> tuple[str, ...]:
    rows = registered_scrc_workloads()
    if frontend is not None:
        if frontend not in {"triton", "tilelang", "cuda"}:
            raise ValueError(f"unsupported registered workload frontend {frontend!r}")
        rows = {
            case_id: adapter
            for case_id, adapter in rows.items()
            if getattr(adapter, "frontend", "triton") == frontend
        }
    return tuple(sorted(rows))


def prepare_registered_scrc_workload(
    case_id: str,
    *,
    dataset_root: str | Path | None = None,
) -> PreparedRegisteredWorkload:
    adapter = registered_scrc_workloads().get(case_id)
    if adapter is None:
        raise ValueError(f"case {case_id!r} has no registered SCRC Harness")
    if isinstance(adapter, TritonWorkloadAdapter):
        return prepare_triton_workload(adapter, dataset_root=dataset_root)
    if isinstance(adapter, FrontendDatasetHarness):
        return prepare_frontend_dataset_workload(
            adapter, dataset_root=dataset_root
        )
    raise TypeError(f"unsupported registered Harness {type(adapter).__name__}")


__all__ = [
    "PreparedRegisteredWorkload",
    "RegisteredHarness",
    "prepare_registered_scrc_workload",
    "registered_scrc_case_ids",
    "registered_scrc_workloads",
]
