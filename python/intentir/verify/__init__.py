"""Verification stack.

Schema/type checks, obligation checks (over evidence + LLM-reported claims),
and dynamic source-execution agreement. The numpy reference interpreter that
gives each atomic op an executable semantics lives here. Dynamic agreement can
rescue statically-UNKNOWN kernels (see .trellis/spec/verification/).
"""
