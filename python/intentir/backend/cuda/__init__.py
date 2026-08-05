"""NVIDIA CUDA lowering capability registry.

Only the high-performance kernel capability registry lives here.  The
pre-SCRC CUDA compile/launch/orchestration modules were archived on
2026-08-01 (see ``archive/2026-08-01-pre-scrc-python/``); the production
backend that consumes accepted ``J+C`` lives on the separate
``TianchenIR-backend-certified-j-search`` branch.

This package intentionally re-exports nothing: importing
:mod:`intentir.backend.cuda.capabilities` must not pull in an execution
stack that this repository no longer owns.
"""
