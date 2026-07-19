"""Target-neutral backend contracts.

This package is the migration boundary for logical ABI, runtime contracts, and
variant manifests.  Their legacy flat modules remain authoritative until each
implementation is moved in a dedicated compatibility-preserving commit.
"""

from intentir.backend.abi import LogicalABIError, LogicalArgument
from intentir.backend.common.capabilities import (
    CapabilityRegistryError,
    HighPerformanceCapabilityRegistry,
    KernelLoweringCapability,
    load_high_performance_registry,
)
from intentir.backend.runtime_contract import (
    ContractMatch,
    PredicateResult,
    RuntimeContract,
    RuntimeContractError,
    RuntimeLaunchFacts,
    RuntimePredicate,
    runtime_launch_facts_from_tensors,
)

__all__ = [
    "ContractMatch",
    "CapabilityRegistryError",
    "HighPerformanceCapabilityRegistry",
    "KernelLoweringCapability",
    "LogicalABIError",
    "LogicalArgument",
    "PredicateResult",
    "RuntimeContract",
    "RuntimeContractError",
    "RuntimeLaunchFacts",
    "RuntimePredicate",
    "load_high_performance_registry",
    "runtime_launch_facts_from_tensors",
]
