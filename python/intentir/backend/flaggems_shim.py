"""Runtime compatibility shim for benchmarking flag_gems on triton >= 3.6.

flag_gems' ``flag_gems/utils/libentry.py`` (``LibEntry.run``) has two version
gates (``if major_version == 3 and 3 <= minor_version <= 5``) that build the
cached fast-path launch argument list. On triton 3.6 those gates fall through
to the ``else`` branch (``kernel[grid](*k_args.values())``), which DROPS every
constexpr placeholder from ``k_args``. triton 3.6's nvidia launcher still wants
the constexpr slots present as ``"O"`` placeholders, so the cached call passes
N-1 args where N are expected:

    TypeError: function takes exactly 32 arguments (31 given)

The FIRST call per process goes through the full JIT path and works; every
subsequent (cached) call throws -- which is exactly why single-call correctness
PKs pass but steady-state benchmarking never gets a number.

flag_gems' own 3.3-3.5 branch already contains a correct, constexpr-inclusive
launch path (rebuilds the full ordered arg list from
``k_args | tune_constexprs | heur_constexprs | constexprs``). Both gates key off
the module global ``minor_version``, so a single assignment routes triton 3.6
through that correct path. We mutate ONLY the imported module object -- the
site-packages file stays read-only.

Safety of lying ``minor_version = 5``: the only other consumer of this global in
the module is ``LibTuner.__init__`` (``major_version == 2 or (major_version == 3
and minor_version <= 1)``), which is False at both 5 and 6, so behaviour is
unchanged apart from the two intended launch gates.

Validated on RTX 5090D / sm_120a, triton 3.6.0: flag_gems ops run 100x+
in-process, including under ``triton.testing.do_bench_cudagraph``, with correct
output (see .trellis/tasks/07-07-coverage-and-perf-real/research/r2-flaggems-fix.md).
"""

from __future__ import annotations

import importlib

_APPLIED = False


def apply_flaggems_launch_shim() -> bool:
    """Route triton >= 3.6 through flag_gems' constexpr-inclusive launch path.

    Idempotent. Returns True if the shim is active (either just applied or
    already applied / not needed because triton < 3.6). Import ``flag_gems``
    before calling this so its ``libentry`` module is present.
    """
    global _APPLIED
    if _APPLIED:
        return True
    # NB: ``flag_gems.utils.libentry`` the attribute is the decorator function;
    # import the MODULE object explicitly to reach its globals.
    le = importlib.import_module("flag_gems.utils.libentry")
    if le.major_version == 3 and le.minor_version >= 6:
        le.minor_version = 5  # both run() gates now take the correct branch
    _APPLIED = True
    return True
