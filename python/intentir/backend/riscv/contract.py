"""Fail-closed Layer-B ContractView for the native RISC-V backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from intentir.backend.contract_capabilities import derive_search_capabilities
from intentir.backend.materialize import BackendSearchState


CONTRACT_VIEW_VERSION = "intentir.backend.riscv.contract-view.v1"


class RiscvContractError(ValueError):
    pass


@dataclass(frozen=True)
class RiscvContractView:
    target: Mapping[str, Any]
    consumed: tuple[str, ...]
    unsupported: tuple[str, ...]
    capabilities: tuple[str, ...]
    version: str = CONTRACT_VIEW_VERSION

    @property
    def supported(self) -> bool:
        return not self.unsupported

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "target": dict(self.target),
            "supported": self.supported,
            "consumed": list(self.consumed),
            "unsupported": list(self.unsupported),
            "capabilities": list(self.capabilities),
        }


def derive_riscv_contract_view(
    state: BackendSearchState,
) -> RiscvContractView:
    if not isinstance(state, BackendSearchState):
        raise TypeError("RISC-V ContractView requires BackendSearchState")
    if state.target.family != "riscv":
        raise RiscvContractError("RISC-V ContractView requires riscv target")
    j = state.base.j
    projection = state.base.boundary_projection
    consumed = list(projection.consumed)
    projected_consumed = set(projection.consumed)
    unsupported: list[str] = list(projection.unsupported_lowering)
    if j.boundary.aliases:
        unsupported.append("alias_relation")
    if j.boundary.updates:
        unsupported.append("logical_update_relation")
    if j.boundary.access_guard_relations:
        guards_consumed = all(
            any(
                ref.startswith(
                    f"access_guard:{item.access_ref}:{item.value_ref}:"
                )
                and ref.endswith(f":{item.polarity.value}")
                for ref in projected_consumed
            )
            for item in j.boundary.access_guard_relations
        )
        if not guards_consumed:
            # The shared typed boundary materializer is the only authority that
            # may turn an authored access-local guard into an explicit select.
            # Consume its record when complete; otherwise fail closed instead
            # of reconstructing or guessing the guard in the RVV backend.
            unsupported.append("access_guard_materialization_not_implemented")
    for storage in j.storage:
        reference = f"storage:{storage.storage_id}"
        if (
            storage.mutability.value == "mutable"
            and reference not in projected_consumed
        ):
            unsupported.append(
                f"general_mutable_storage:{storage.storage_id}"
            )
        else:
            consumed.append(reference)
    consumed.extend(
        f"effect:{item.effect_id or index}:{item.kind.value}"
        for index, item in enumerate(j.boundary.effects)
    )
    consumed.extend(
        f"reduction:{item.node_ref}" for item in j.boundary.reductions
    )
    consumed.extend(
        f"result_relation:{item.output}:{item.kind.value}"
        for item in j.boundary.result_relations
    )
    consumed.extend(
        f"output:{item.output}" for item in j.boundary.output_mappings
    )
    consumed.extend(f"scope:{item.scope_id}" for item in j.scopes)
    consumed.extend(state.base.consumed_structures)
    view = RiscvContractView(
        target=state.target.to_dict(),
        consumed=tuple(sorted(set(consumed))),
        unsupported=tuple(sorted(set(unsupported))),
        capabilities=derive_search_capabilities(state.base),
    )
    if not view.supported:
        raise RiscvContractError(
            "RISC-V backend cannot consume complete Layer B: "
            + ", ".join(view.unsupported)
        )
    return view


__all__ = [
    "CONTRACT_VIEW_VERSION",
    "RiscvContractError",
    "RiscvContractView",
    "derive_riscv_contract_view",
]
