"""Session-wide test fixtures. Applies the flag_gems launch shim so PKs that
call flag_gems more than once per process do not hit the triton-3.6 constexpr-
drop ABI bug (see intentir.backend.flaggems_shim / task 07-07 M0)."""
try:
    from intentir.backend.flaggems_shim import apply_flaggems_launch_shim
    apply_flaggems_launch_shim()
except Exception:
    pass  # flag_gems not importable in this env -> PKs skip/guard themselves
