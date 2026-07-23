# mini-swe-agent source origin

This module was imported in full from
[SWE-agent/mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent):

- release: `v2.4.5`
- commit: `e187bcb2ff5825d85761a6f9c1f98c9fa6cfbc79`
- imported tree: upstream `src/minisweagent`
- license: MIT; see `LICENSE.md`

The source is now a first-class module of this repository and is modified
directly for IntentIR. There is no patch overlay and no requirement to retain
upstream runtime compatibility.

Within IntentIR, this module implements only the bounded, source-aware Lift
worker. The authoritative TypeCheck, source-isolated Replay, Gate1, Gate2,
candidate selection, and SCRC status machine remain outside this module.
