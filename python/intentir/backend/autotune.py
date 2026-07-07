"""Measured autotuner for the IntentIR GPU backend (M5).

The backend previously had NO autotuner: block/tile/vector geometry was either a
fixed constant baked into the C++ pass or a one-line ``autoThreads`` heuristic
(see research/r5-backend-critique.md P4). That means we could only ever *tie* the
library's chosen geometry, never *search past* it — and M1 showed the fixed
schedules are frequently wrong (argmax's 1024-thread default is latency-bound;
the shape-dependent optimum is 256).

This module is the GPU analog of the paper's RVV roofline search: a real
``lower -> do_bench -> argmin`` empirical search over the knobs already latent in
the passes (exposed as pass options), with a persisted per-(op, shape, dtype)
JSON cache and an occupancy-driven default for cold shapes.

Design (per the R-CODEGEN-5 requirement):
  * ``config_space(family, meta)`` — a **legal-config generator per op-family**.
      - mover family (elementwise / rowreduce / pool / argreduce / compact /
        gather / layout / rng / sort): sweep ``block-threads T in
        {128,256,512,1024}`` filtered by shape legality (T divides the tiled
        axis, power-of-two). ``vectorWidth`` / ``gridStyle`` are carried as
        single-value axes today (the passes expose only ``block-threads``); they
        become real sweep axes once M1/M6 land those pass options — the config
        dict already threads them so no caller changes then.
      - contraction / conv (the M3 mma pass): sweep ``BM,BN in {32,64,128}``,
        ``BK in {16,32,64}`` (the block-k option added in this milestone), and
        ``numStages in {1,2,3}``, filtered by tile-divisibility and the ~99 KiB
        smem cap.
  * ``AutotuneDriver.tune(task)`` — for each legal config: build the kernel via
    the pass with that config, ``do_bench`` it (the same steady-state timer
    bench_clusters uses), record the median; return the argmin config. Configs
    that fail to lower / launch / verify are skipped.
  * ``AutotuneDriver.lookup(op, shape, dtype)`` — the kernel-build path consults
    the JSON cache first; ``default_config`` supplies the occupancy default for
    cold shapes. Deterministic (fixed config-space order, first-wins ties) and
    resumable (cache is checked before searching).

The cache lives at ``autotune_cache.json`` next to this module.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

CACHE_PATH = Path(__file__).resolve().parent / "autotune_cache.json"

# sm_120a dynamic-shared budget (~100 KiB); leave a little headroom for the
# static shared the reduction/epilogue uses.
SMEM_CAP_BYTES = 99 * 1024

MOVER_T_SWEEP = [128, 256, 512, 1024]
TILE_SWEEP = [32, 64, 128]
BK_SWEEP = [16, 32, 64]
STAGES_SWEEP = [1, 2, 3]
# f16 padding on the smem K-tile leading dim (swizzle-free bank-conflict
# avoidance for the ldmatrix reads). 0 = off; 8 = one 128-bit lane. Only the
# nvgpu.mma.sync contraction path exposes ``smem-pad-k``.
PAD_SWEEP = [0, 8]


def _do_bench(thunk, want_graph: bool = True):
    """Median kernel us via the same steady-state path bench_clusters uses.

    Imported lazily so ``import intentir.backend.autotune`` stays cheap and does
    not pull in torch/triton/flag_gems at module load.
    """
    import triton

    if want_graph:
        try:
            ms = triton.testing.do_bench_cudagraph(thunk, rep=100,
                                                   return_mode="median")
            return ms * 1e3
        except Exception:
            import torch
            torch.cuda.synchronize()
    ms = triton.testing.do_bench(thunk, warmup=50, rep=200, return_mode="median")
    return ms * 1e3


@dataclass
class TuneTask:
    """One thing to tune: an (op, shape, dtype) plus how to build a candidate.

    ``build(config) -> (thunk, ok)`` lowers the op with ``config`` (a dict of
    pass-option values), returns a zero-arg NO-SYNC launch closure and a
    correctness flag (``ok=False`` / ``thunk=None`` -> the config is skipped). A
    config only changes the *schedule*, never the math, so a correct build stays
    correct — the flag guards configs that fail to lower/launch on this shape.
    """

    op: str
    shape: str
    dtype: str
    family: str            # "mover" | "contraction" | "conv"
    meta: dict             # shape info the config-space generator needs
    build: Callable[[dict], tuple]
    want_graph: bool = True


class AutotuneDriver:
    def __init__(self, cache_path: Path | str = CACHE_PATH):
        self.cache_path = Path(cache_path)
        self.cache: dict = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text())
            except Exception:
                self.cache = {}

    # ------------------------------------------------------------------ #
    # cache
    # ------------------------------------------------------------------ #
    @staticmethod
    def _key(op: str, shape: str, dtype: str) -> str:
        return f"{op}|{shape}|{dtype}"

    def lookup(self, op: str, shape: str, dtype: str) -> Optional[dict]:
        """The kernel-build path calls this FIRST; None -> cold shape."""
        rec = self.cache.get(self._key(op, shape, dtype))
        return rec["config"] if rec else None

    def config_for(self, op: str, shape: str, dtype: str, family: str,
                   meta: dict) -> dict:
        """Runtime-consult hook: cached winner, else the occupancy default."""
        cfg = self.lookup(op, shape, dtype)
        return cfg if cfg is not None else self.default_config(family, meta)

    def _save(self):
        tmp = self.cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.cache, indent=2, sort_keys=True))
        tmp.replace(self.cache_path)

    # ------------------------------------------------------------------ #
    # legal-config generators (per op-family)
    # ------------------------------------------------------------------ #
    def config_space(self, family: str, meta: dict) -> list[dict]:
        if family == "mover":
            return self._mover_space(meta)
        if family in ("contraction", "conv"):
            return self._contraction_space(family, meta)
        raise ValueError(f"unknown op-family {family!r}")

    def _mover_space(self, meta: dict) -> list[dict]:
        # T must divide the tiled axis (``divisor``) and be a power of two so the
        # shared-memory reduction tree is well-formed. vectorWidth/gridStyle are
        # future axes (single value today).
        divisor = meta.get("divisor")
        vw = meta.get("vector_widths", [1])
        grid = meta.get("grid_styles", ["default"])
        out = []
        for gs in grid:
            for v in vw:
                for T in MOVER_T_SWEEP:
                    if divisor is not None and divisor % T != 0:
                        continue
                    out.append({"T": T, "vectorWidth": v, "gridStyle": gs})
        return out or [self.default_config("mover", meta)]

    def _contraction_space(self, family: str, meta: dict) -> list[dict]:
        M, N = meta["M"], meta["N"]
        K = meta.get("K")
        out = []
        for BM in TILE_SWEEP:
            if BM % 16 != 0 or M % BM != 0:
                continue
            for BN in TILE_SWEEP:
                if BN % 8 != 0 or N % BN != 0:
                    continue
                bk_list = (BK_SWEEP if (family == "contraction" and K is not None)
                           else [None])
                stages_list = (STAGES_SWEEP if family == "contraction" else [None])
                # smemPadK (the ldmatrix bank-conflict pad) is only a knob on the
                # nvgpu.mma.sync contraction path; other families sweep pad=0.
                pad_list = (PAD_SWEEP if family == "contraction" else [0])
                for BK in bk_list:
                    if BK is not None and (K % BK != 0):
                        continue
                    for ns in stages_list:
                        for pad in pad_list:
                            cfg = {"BM": BM, "BN": BN}
                            if BK is not None:
                                cfg["BK"] = BK
                            if ns is not None:
                                cfg["numStages"] = ns
                            if family == "contraction":
                                cfg["smemPadK"] = pad
                            # smem = numStages*(BM+BN)*(BK+pad)*2B (f16 A/W tiles)
                            bk_eff = (BK if BK is not None else 32) + pad
                            ns_eff = ns if ns is not None else 1
                            smem = ns_eff * (BM + BN) * bk_eff * 2
                            if smem > SMEM_CAP_BYTES:
                                continue
                            out.append(cfg)
        return out or [self.default_config(family, meta)]

    # ------------------------------------------------------------------ #
    # occupancy default for cold shapes (M1 flatThreads / flat-grid style)
    # ------------------------------------------------------------------ #
    def default_config(self, family: str, meta: dict) -> dict:
        if family == "mover":
            divisor = meta.get("divisor")
            T = 256
            if divisor is not None:
                while T > 1 and divisor % T != 0:
                    T //= 2
            return {"T": T, "vectorWidth": 1, "gridStyle": "default"}
        if family == "conv":
            return {"BM": 64, "BN": 64}
        return {"BM": 64, "BN": 128, "BK": 32, "numStages": 3}

    # ------------------------------------------------------------------ #
    # the search
    # ------------------------------------------------------------------ #
    def tune(self, task: TuneTask, force: bool = False,
             verbose: bool = False) -> dict:
        """lower -> do_bench -> argmin over the legal config space; cache winner.

        Resumable: a cached (op, shape, dtype) is returned without re-searching
        unless ``force``. Deterministic: fixed config-space order, first-wins ties.
        """
        key = self._key(task.op, task.shape, task.dtype)
        if not force and key in self.cache:
            if verbose:
                print(f"  [cache hit] {key} -> {self.cache[key]['config']}")
            return self.cache[key]["config"]

        space = self.config_space(task.family, task.meta)
        tried = []
        for cfg in space:
            thunk = None
            ok = False
            try:
                thunk, ok = task.build(cfg)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    skip {cfg}: build {type(e).__name__}: {str(e)[:60]}")
                continue
            if thunk is None or not ok:
                if verbose:
                    print(f"    skip {cfg}: not ok")
                continue
            try:
                us = _do_bench(thunk, want_graph=task.want_graph)
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"    skip {cfg}: bench {type(e).__name__}: {str(e)[:60]}")
                continue
            tried.append((us, cfg))
            if verbose:
                print(f"    {cfg}  {us:8.2f} us")

        if tried:
            best_us, best_cfg = min(tried, key=lambda r: r[0])
        else:
            best_cfg, best_us = self.default_config(task.family, task.meta), None

        self.cache[key] = {
            "config": best_cfg,
            "us": round(best_us, 3) if best_us is not None else None,
            "family": task.family,
            "n_tried": len(tried),
            "tried": [{"config": c, "us": round(u, 3)} for u, c in tried],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._save()
        if verbose:
            print(f"  [best] {key} -> {best_cfg}  "
                  f"({best_us:.2f} us)" if best_us else f"  [best] {best_cfg}")
        return best_cfg
