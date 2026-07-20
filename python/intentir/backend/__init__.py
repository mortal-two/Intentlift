"""Target-neutral backend interfaces and target-specific implementations.

Shared ABI, capability, manifest, and launch-plan contracts live directly in
this package.  Concrete compilation and execution stacks belong to a target
namespace such as :mod:`intentir.backend.cuda`.
"""
