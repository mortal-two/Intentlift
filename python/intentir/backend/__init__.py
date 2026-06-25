"""Back-end: lower the ``intent`` dialect to an executable GPU binary.

The spine is intent -> linalg/tensor/arith -> scf.parallel -> gpu -> nvvm ->
cubin, driven entirely on system MLIR 20.1.8 + ptxas 12.8 (verified to reach a
valid sm_120a cubin). Driver-API launch on torch pointers lives alongside.
"""
