"""RISC-V target backend boundary.

No RISC-V lowering is claimed yet.  The package exists so target-neutral
callers can depend on an explicit namespace without adding RISC-V conditionals
to the CUDA implementation when that backend is introduced.
"""

__all__: list[str] = []
