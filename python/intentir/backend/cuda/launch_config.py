"""Structured launch plan (M6, R-RUNTIME-1): read grid/block/smem from the
``intent.launch_plan`` attribute the C++ passes now emit, instead of
regex-scraping the fully-lowered LLVM text.

Background
----------
Every perf-critical ``ConvertIntent*ToGPU`` pass bakes its launch geometry as
raw ``arith.constant`` index ops feeding ``gpu.launch``; the runtime
(``orchestrate.py`` / ``kernel_driver.py``) then recovers grid/block/smem by
``_resolve_const_index`` -- a regex over ``gpu.launch_func ... blocks in (...)``
that only works because ``-canonicalize`` happens to fold the extents to
literals (R5 P5, ``r5-backend-critique.md:102-111``). Brittle by construction.

The passes now ALSO attach a first-class ``Schedule`` (``lib/Intent/Conversion/
Schedule.h``) as a structured ``intent.launch_plan`` dictionary attribute on the
host func::

    func.func @rms_norm(...) attributes {intent.launch_plan =
        {block = [256, 1, 1], grid = [8, 1, 1], smem = 0 : i64,
         per_thread_tile = ... , pipeline_depth = ..., split_factor = ...,
         vector_width = ...}} { ... }

This module parses that attribute out of the PASS-OUTPUT MLIR (the memref+gpu
module the driver already has in hand, *before* the LLVM lowering the regex
path reads). The runtime PREFERS this plan and ASSERTs it equals the
regex-recovered grid/block/smem (``assert_matches``) so equivalence is proven on
the whole PK suite before the structured path is trusted; the regex path stays
as a validated fallback for any launch whose pass has not yet been taught to
emit a plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# One `key = [a, b, c]` (i64 array) or `key = N : i64` (scalar) field inside the
# `intent.launch_plan = { ... }` dictionary attribute.
_PLAN_RE = re.compile(r"intent\.launch_plan\s*=\s*\{([^}]*)\}")
_ARR_RE = re.compile(r"(\w+)\s*=\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]")
_INT_RE = re.compile(r"(\w+)\s*=\s*(-?\d+)\s*:\s*i\d+")


@dataclass(frozen=True)
class KernelLaunchConfig:
    """A pass-emitted structured launch plan (one gpu.launch's geometry)."""

    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    smem: int
    per_thread_tile: int = 1
    vector_width: int = 1
    pipeline_depth: int = 1
    split_factor: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "grid": list(self.grid),
            "block": list(self.block),
            "smem_bytes": self.smem,
            "per_thread_tile": self.per_thread_tile,
            "vector_width": self.vector_width,
            "pipeline_depth": self.pipeline_depth,
            "split_factor": self.split_factor,
        }

    def assert_matches(self, grid, block, shared: int, where: str = "") -> None:
        """Assert the regex-recovered (grid, block, shared) equals this plan.

        This is the equivalence proof the M6 refactor gates on: the structured
        plan must reproduce EXACTLY what the old regex path recovered, on every
        launch in the PK suite, before we trust it. A mismatch is a hard error
        (the launcher then knows to fall back / be fixed), not a silent drift.
        """
        g = tuple(int(x) for x in grid)
        b = tuple(int(x) for x in block)
        if g != self.grid or b != self.block or int(shared) != self.smem:
            raise AssertionError(
                f"launch_plan mismatch{(' ' + where) if where else ''}: "
                f"structured grid={self.grid} block={self.block} smem={self.smem} "
                f"vs regex grid={g} block={b} smem={int(shared)}"
            )


def parse_launch_configs(mlir_text: str) -> list[KernelLaunchConfig]:
    """Parse every ``intent.launch_plan`` attribute in a pass-output module.

    Returns them in source order (one per host func that emitted a plan). An
    empty list means no pass attached a plan (regex fallback applies).
    """
    plans: list[KernelLaunchConfig] = []
    for m in _PLAN_RE.finditer(mlir_text):
        body = m.group(1)
        arrs = {am.group(1): (int(am.group(2)), int(am.group(3)), int(am.group(4)))
                for am in _ARR_RE.finditer(body)}
        ints = {im.group(1): int(im.group(2)) for im in _INT_RE.finditer(body)}
        if "grid" not in arrs or "block" not in arrs:
            continue
        plans.append(
            KernelLaunchConfig(
                grid=arrs["grid"],
                block=arrs["block"],
                smem=ints.get("smem", 0),
                per_thread_tile=ints.get("per_thread_tile", 1),
                vector_width=ints.get("vector_width", 1),
                pipeline_depth=ints.get("pipeline_depth", 1),
                split_factor=ints.get("split_factor", 1),
            )
        )
    return plans


def single_launch_config(mlir_text: str) -> KernelLaunchConfig | None:
    """The sole launch plan in a single-kernel module, or None if absent.

    Raises if more than one plan is present (a single-launch driver expects at
    most one); multi-launch callers use ``parse_launch_configs``.
    """
    plans = parse_launch_configs(mlir_text)
    if not plans:
        return None
    if len(plans) > 1:
        raise RuntimeError(
            f"launch_plan: expected one plan, found {len(plans)}"
        )
    return plans[0]


__all__ = [
    "KernelLaunchConfig",
    "parse_launch_configs",
    "single_launch_config",
]
