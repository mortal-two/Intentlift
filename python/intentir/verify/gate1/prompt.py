"""Source-isolated Replay packets for the LLM-direct Gate1 protocol."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from intentir.ir.canonical_j import canonicalize_gate1_artifact
from .model import Frontend, Gate1Artifact, SchemaError
from .pi import (
    render_realization_mechanism_grammar,
    render_realization_witness_wire_grammar,
)


REPLAY_PACKET_SCHEMA = "intentir.scrc.replay-input.v1"
PUBLIC_FRONTEND_CONTEXT = {
    "triton": {
        "context_policy": "public_frontend_context_v16",
        "retrieval_scope": "public Triton documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Define exactly one @triton.jit scrc_k0 with J.signature handles "
            "followed by the anonymous MaskC(C) slots. Keep every C constexpr "
            "slot annotated tl.constexpr. Triton compile-time-only APIs do not "
            "preserve constexpr through ordinary Python aliases: pass a c* handle "
            "directly to tl.arange and any other compile-time-only API, never copy "
            "it to a local first. A single realize instance anchored to c* is "
            "satisfied by the parameter binding and its direct routed use; do not "
            "emit an identity assignment merely to materialize it. J reduce.axes "
            "are whole-tensor logical axes, while each matching "
            "J.boundary.reductions.tile_axis is the physical frontend reduction "
            "axis and must be used at the public reduction call. It is the local "
            "tile axis after program/grid axes select a program instance, not the "
            "whole-tensor logical ordinal in J reduce.attrs.axes; never copy or "
            "offset that logical ordinal to guess it. "
            "Each J.boundary access is closed: when predicate is absent, emit an "
            "unconditional load or store; never synthesize a defensive or "
            "redundant mask from a domain, grid extent, C slot, or safety guess. "
            "Every J.signature p* is one already-existing callable ABI carrier. "
            "All pi instances anchored to the same p* scope or route uses of that "
            "single carrier; they never create another external value or program "
            "coordinate. Duplicate p* instances therefore never authorize duplicate "
            "carrier reads or tl.program_id calls. "
            "Region placement='program' alone never authorizes "
            "tl.program_id. A domain-anchored realize stage in a program region "
            "whose scalar dependency path enters address_base is a program "
            "coordinate. Assign axes by first declaration of each distinct J "
            "domain semantic_ref: distinct stages anchored to the same d* remain "
            "distinct materializations of the same program axis and never allocate "
            "another axis. Only a distinct domain anchor may allocate the next "
            "tl.program_id(axis). A domain/C stage whose path enters address_lane "
            "is lane-varying and may use tl.arange. Do not vectorize a program "
            "coordinate or turn a lane coordinate into tl.program_id. Preserve "
            "every address_base -> address_lane route, then determine pointer "
            "operation count only from J.signature carrier kind plus pi routes. "
            "If a pointer/tensor p* carrier directly enters address_base, the base "
            "is already a pointer/address operation; its route to address_lane "
            "requires a second pointer/address operation for the lane contribution. "
            "If that carrier bypasses address_base and enters address_lane or the "
            "final address stage, while address_base receives only integer/domain/C "
            "contributions, preserve and merge the integer base and lane stages "
            "before exactly one pointer/address operation. In both cases preserve "
            "the program_id even if its grid extent is one. Never choose from the "
            "address_base -> address_lane edge alone, and never infer from Source "
            "names, paths, or text."
            " For a canonical binary J contraction whose per-program operands and "
            "result are rank-2 floating tiles with exactly one contracted label, "
            "and whose declared operand/result dtypes are accepted by public "
            "tl.dot, one pi realize instance anchored to that contraction node "
            "maps to one direct tl.dot realization (using the declared accumulation "
            "dtype). Do not replace that single stage with broadcasted outer-product "
            "multiplication plus tl.sum: those are additional compiler-visible "
            "stages. This direct rule does not apply to outer products, batched or "
            "otherwise unsupported contractions, unsupported tile/dtype contracts, "
            "or a pi chain that explicitly declares multiple contraction-anchored "
            "realize stages; never force tl.dot by changing J or dropping pi stages."
            " Every pi.iterations entry is mandatory realization topology. "
            "Resolve loop_instance to its J structured loop and tile_instance "
            "to the matching C tile slot. For induction_basis=extent_step, "
            "materialize a zero-based loop whose upper bound is the J trip extent "
            "and whose step is the C tile. For induction_basis=tile_ordinal, "
            "materialize a zero-based unit-step loop whose upper bound is the "
            "trip extent ceiling-divided by the matching C tile. Preserve this "
            "distinction even when "
            "the induction variable itself is unused: never insert or remove the "
            "trip-count ceiling division, and never choose a basis from an op "
            "family, concrete value, name, or frontend convention."
            " For induction_basis=rounded_extent_step, use the same zero lower "
            "and matching C tile step as extent_step, but use exactly "
            "ceil_div(the declared J trip extent, that C tile) multiplied by "
            "that C tile as the upper bound. Do not use the current tensor shape, "
            "a launch value, or an invented rounding formula."
            " Every pi.physical_iterations entry is mandatory realization "
            "topology for a Source-visible physical loop around an existing "
            "non-loop J reduce/contraction or boundary access/domain; it never "
            "creates a semantic J loop. Resolve body_instance to that authored "
            "subject and tile_instance to the required positive C slot attached "
            "to the same closed dependent axis. The "
            "axis attachment determines the logical extent, and induction_basis "
            "has the same extent_step, rounded_extent_step, or tile_ordinal "
            "denotation as pi.iterations. direction=forward visits that closed "
            "basis in ascending order; direction=reverse visits exactly the same "
            "basis points in descending order. Never infer the extent or direction "
            "from Source names, current shapes, C values, grid conventions, or "
            "frontend APIs. Every pi.physical_loop_carriers entry preserves the "
            "declared init/state/next/result temporal stages for its physical "
            "iteration region; do not collapse the loop-carried accumulator into "
            "one stateless reduction or invent an update formula absent from J."
            " Every pi.unit_iterations entry is mandatory physical topology over "
            "one already-authored non-loop J subject and axis. Materialize exactly "
            "the logical points [0, extent) with unit step, or the same points in "
            "descending order for direction=reverse. It has no C tile and never "
            "authorizes a new semantic J loop, rounded bound, launch-derived bound, "
            "formula, or operator."
            " Every J effect_if is mandatory effect control. Emit one real "
            "runtime `if` with explicit positive and negative arms, preserve "
            "predicate polarity, lexical arm scope, nested operation/effect order, "
            "and an explicitly empty arm. Never replace effect_if with tl.where, "
            "a value select, predicated unconditional effects, or compile-time "
            "control. Read the predicate and both arms only from J."
            " When pi.terminal_controls relates the two authored arm instances of "
            "that effect_if, emit the terminal arm as the sole body of a runtime "
            "`if` followed by a bare return, then emit the continuation arm after "
            "the `if` without an `else`. Use the authored predicate unchanged when "
            "the positive arm is terminal and its logical negation when the negative "
            "arm is terminal. Resolve both roles only through the typed pi relation; "
            "never infer termination from empty arms, effects, values, names, or "
            "frontend conventions."
            " Every J value_if is mandatory pure value control. Emit one real "
            "runtime `if` with explicit positive and negative arms in authored "
            "polarity and lexical order. Each arm may compute pure local "
            "temporaries, but its final statement must assign the authored yield "
            "to the same merge local in both arms. Never replace value_if with "
            "tl.where/select, effect_if, predicated unconditional computation, or "
            "compile-time control. effect_if, value_if, and select are "
            "hard-distinct; read the predicate, arms, yields, and result only "
            "from J."
            " Every pi.refinements entry is mandatory realization topology. "
            "Resolve its auth_ref in J.boundary.realization_authorizations and "
            "materialize the authorized kind at that exact instance before its "
            "routed consumers. For signed_widen, unsigned_widen, numeric_cast, "
            "and accumulation_dtype, use a public Triton cast/to operation with "
            "the explicitly authorized target_dtype; preserve distinct refined "
            "stages and do not algebraically remove or move the conversion. Never "
            "propagate one refinement to a sibling parameter/instance that merely "
            "shares a dtype or address expression. Signature dtypes are otherwise "
            "authoritative: do not guess a stride/index widening or emit `.to` for "
            "a p* carrier without its own routed refinement or an explicit J cast."
            " A reduction_linearization refinement is the exact source-neutral "
            "full-axis materialization immediately before its uniquely routed J "
            "reduce. For Triton, emit `tl.reshape(value, (value.numel,), "
            "can_reorder=True)` at that stage, then reduce on local axis 0. Never "
            "infer it from reduction kind or shape, apply it to another consumer, "
            "or omit it because the current carrier is already rank one."
            " A J const with value '-inf', 'inf', or 'nan' is the exact typed IEEE "
            "special authored by J. Materialize it directly at its declared dtype; "
            "never approximate it with a finite literal or promote it through an "
            "undeclared wider floating type."
            " A J.signature constexpr p* with logical_ref=x* is the callable "
            "carrier of that exact semantic dimension. Use that same p* directly "
            "where the authored pi lane/iteration realization requires x*; never "
            "replace it with a current-shape literal or an arbitrary unit extent."
            " A J.signature constexpr p* with logical_ref=null is only a "
            "Harness-fixed JIT-specialization carrier: preserve its constexpr "
            "ABI, emit the already-authored active graph, and never read it, "
            "reconstruct inactive control, or invent a dummy use. Its concrete "
            "value is absent from the Replay packet."
            " A scalar J.signature p* with logical_ref=null has no logical denotation. It "
            "may affect the body only through an instance anchored to that p* and "
            "a declared route to its realization consumer; if no such route exists, "
            "keep the ABI parameter but do not read it in the kernel body."
            " The generated module must obey the audited public import ABI: use "
            "`import triton`, `import triton.language as tl`, and, when a reviewed "
            "libdevice function is required, exactly "
            "`from triton.language.extra.cuda import libdevice` followed by "
            "`libdevice.<function>(...)`. Directly importing a libdevice function "
            "is rejected by the fixed module validator. In pinned Triton 3.6, "
            "spell tanh/tan/atan/atan2 as libdevice calls, elementwise power as "
            "the public `x ** y` operator, and isnan as `x != x`; `tl.tanh`, "
            "`tl.tan`, `tl.atan`, `tl.atan2`, `tl.pow`, and `tl.isnan` do not "
            "exist and are rejected before compilation."
        ),
    },
    "cute": {
        "context_policy": "public_frontend_context_v1",
        "retrieval_scope": "public NVIDIA CuTe DSL documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry": "scrc_entry",
        "compile_entry_contract": (
            "Define one @cute.jit scrc_entry after the requested @cute.kernel. "
            "Its positional arguments are J tensor inputs, inouts, then outputs, "
            "followed by stream; construct representation-only descriptors inside it."
        ),
    },
    "tilelang": {
        "context_policy": "public_frontend_context_v6",
        "retrieval_scope": "public TileLang documentation and public examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Return one self-contained Python module using exactly "
            "`import tilelang.language as T` and one `@T.prim_func` named "
            "`scrc_k0`. Express tensor ABI through public `T.Tensor` annotations, "
            "device launch through `T.Kernel`, and realization topology through "
            "public TileLang operations. C values are compile-time schedule "
            "constants, never additional logical runtime parameters. Bind each "
            "distinct J x* once with one uppercase `T.dynamic` module binding and "
            "reuse it in every `T.Tensor` shape; repeated `T.dynamic` calls for one "
            "x* create false ABI parameters. A program-region domain path entering "
            "address_base is a `T.Kernel` program coordinate; a domain path entering "
            "address_lane is participant-parallel and uses `T.Parallel`, not a "
            "serial loop. `pi.unit_iterations` alone authorizes `T.serial`. A J "
            "reduce fed by a lane-realized reduced domain uses the matching public "
            "`T.reduce_*` collective; never sequentialize the whole axis per lane. "
            "Preserve pi transfers as exact load/store or `T.copy` occurrences and "
            "preserve distinct node realizations as typed locals in authored order. "
            "A single-consumer transfer may stay inline only when pi order/region "
            "does not place it before an enclosing coordinate or control stage. A "
            "transfer ordered before the lane coordinate becomes one typed scalar "
            "before `T.Parallel`; a transfer inside that lane may remain inline. "
            "J.storage is the sole allocation authority: when it is empty, represent "
            "per-lane stages as annotated scalar locals and never emit `T.alloc_*`; "
            "use an allocation or `T.copy` destination only for authored storage. "
            "TIR structure is the hard frontend evidence; PTX details below its ABI "
            "envelope are diagnostic only. A MaskC compiler slot whose role is "
            "`launch_block_x|y|z` is an injected module-global schedule handle: use "
            "that c* directly for the corresponding `T.Kernel` thread extent, never "
            "turn it into a runtime parameter or guess its hidden value."
        ),
    },
    "cuda": {
        "context_policy": "public_frontend_context_v3",
        "retrieval_scope": "public CUDA C++ programming documentation and examples",
        "tool_access": ["public_document_lookup"],
        "compile_entry_contract": (
            "Return one standalone CUDA C++ translation unit containing exactly "
            "one `extern \"C\" __global__` kernel named `scrc_k0`. Its positional "
            "runtime parameters follow J.signature; C values are compiled "
            "schedule constants and are not runtime parameters. Use only public "
            "CUDA language constructs and intrinsics. Do not emit host code, "
            "launch syntax, file I/O, inline PTX, build directives, or `main`. "
            "A program-region domain path entering address_base maps to `blockIdx`; "
            "a domain path entering address_lane maps to `threadIdx`, with distinct "
            "axes assigned in declaration order and degenerate coordinates retained. "
            "Only `pi.unit_iterations` authorizes a serial unit-step `for` loop. A J "
            "reduce fed by a lane-realized reduced domain must remain a participant "
            "collective using public CUDA intrinsics; never load or unroll the whole "
            "logical axis independently in every thread. J alone supplies reducer "
            "semantics; pi supplies coordinate, route, order, and synchronization "
            "topology. Preserve every distinct transfer and realization stage. Each "
            "J boundary access predicate is exact: an absent predicate is an "
            "unconditional access and never authorizes a lane-leader guard. A MaskC "
            "compiler slot with role `launch_block_x|y|z` is injected as a compile-"
            "time macro; use its c* identifier directly for launch bounds or "
            "participant width, never as a runtime parameter or guessed literal."
        ),
    },
}

# Stored packet public-context digests and historical receipt digests are
# different evidence domains.  Existing receipt policies all hash the complete
# u_f object; changing that basis requires an explicit future policy version.
PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY = MappingProxyType(
    {
        "triton": "public_frontend_context_v16",
        "cute": "public_frontend_context_v1",
        "tilelang": "public_frontend_context_v6",
        "cuda": "public_frontend_context_v3",
    }
)
PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY = MappingProxyType(
    {
        "triton": MappingProxyType(
            {
                "public_frontend_context_v6": (
                    "5e180f9389fb124735f92ab4c5728e10121fa42bb51bb239902f9676e7f63339"
                ),
                "public_frontend_context_v7": (
                    "2f71b09d4cf3d25cb413b24252a59720822dd3a42b9f8af9eb606a36bfee9613"
                ),
                "public_frontend_context_v8": (
                    "060ddb29e6acdc024b69a122eccd8a3151f3205c909799e5a9356128f1ba67e5"
                ),
                "public_frontend_context_v9": (
                    "f83c5b223fd0b8edbc7b55d7fd626fa54b8d1e26db59f0e7ccdc489dcd30b7a8"
                ),
                "public_frontend_context_v10": (
                    "56d496604506f8e5dafa823cdae8f46e67e0f3e88f87fa5bf576d0334ed95d46"
                ),
                "public_frontend_context_v11": (
                    "b59f1ae42359ba97e26964ad6bde4450d2b7414ab2e210d709917952fa2c6c90"
                ),
                "public_frontend_context_v12": (
                    "15829eaa0e9e844ab6362c51adb7ec8cd2aa5045c6cd8c6ad8a2093276101ba9"
                ),
                "public_frontend_context_v13": (
                    "e93e54400c4fc3e52b048d3b21d85f674a60ee3ba71eefae74f29c081719f1c2"
                ),
                "public_frontend_context_v14": (
                    "d7b785bb9439c247d7dba481362969c5031de931ab368452192ad79296ec2c0f"
                ),
                "public_frontend_context_v15": (
                    "0fa2e4823267dcec5601c4f805ac91301a411efbb7479e5cc9c48b208a8d7510"
                ),
                "public_frontend_context_v16": (
                    "67845810703e1c0906950e4008d975fdab0167d17a8106ced006f0fdf627d402"
                ),
            }
        ),
        "cute": MappingProxyType(
            {
                "public_frontend_context_v1": (
                    "469bcb400c5f377d3b92a68a870189f0698bdd05a2154ccba16feed0502716dc"
                )
            }
        ),
        "tilelang": MappingProxyType(
            {
                "public_frontend_context_v1": (
                    "41d26224d6682b12f7761e6a4c6f61f3f9e29d5ad5bd4b3b473bdfdacbcf91f5"
                ),
                "public_frontend_context_v2": (
                    "ce254dc4037d8e03c1a68dd5f30c685c4cb1357be606101ef125ab437d6efb0d"
                ),
                "public_frontend_context_v3": (
                    "30d10e99d874733b4c98ceecbb476b2689c1be64bd6eac3ad61229d228e6b45c"
                ),
                "public_frontend_context_v4": (
                    "33f518d4726cd1bc64f94b4237eb19beae0ec6471b2c6ce39ee170895836a6ec"
                ),
                "public_frontend_context_v5": (
                    "d66e36be25876ecc20558944d6ccb7816b7fb5cf043fcd95d0b1ea87468ec7a0"
                ),
                "public_frontend_context_v6": (
                    "49ced776ce21868ef00a94a2576d5f9b0649413ff315c78ec3e44b543d87d2b5"
                ),
            }
        ),
        "cuda": MappingProxyType(
            {
                "public_frontend_context_v1": (
                    "97d17ef646dacef8b198afd70d09221a6c59aed6816e89ee19559aff3a22f1b3"
                ),
                "public_frontend_context_v2": (
                    "aa417c35c5da0c4ad083cbe869c3a46a3b0c245fa7fece54bc7d5043424f7d64"
                ),
                "public_frontend_context_v3": (
                    "e3f8509cdeb8e61b4ec6285cff73abc3b5c90697d1ad1728950dadde1d211bd2"
                ),
            }
        ),
    }
)
_PUBLIC_FRONTEND_CONTEXT_DIGEST_ALIASES = MappingProxyType(
    {
        "triton": MappingProxyType(
            {
                # Two private branches independently introduced v15.  Keep both
                # exact payloads auditable after the combined v16 supersedes them.
                "public_frontend_context_v15": frozenset(
                    {
                        "0fa2e4823267dcec5601c4f805ac91301a411efbb7479e5cc9c48b208a8d7510",
                        "7d68b5a2f77f614c952966c1b94a8eac052f6ebcad77412cd7db6af9ffbed8e6",
                    }
                )
            }
        )
    }
)
PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY = MappingProxyType(
    {
        "triton": MappingProxyType(
            {
                "public_frontend_context_v6": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "33f1d58a4c1e762fae6b8ff27c100adfef21bcc1d8a149409f85f8b4c7a281ec"
                        ),
                    }
                ),
                "public_frontend_context_v7": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "91eb3db3fb27a166c2fecb3e02c05ac8131be3b93fa7576272814d3050e388c6"
                        ),
                    }
                ),
                "public_frontend_context_v8": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "79e3af0228c349639b52f5e763a9bf07e01669d470de24caeb7826e7c31d2466"
                        ),
                    }
                ),
                "public_frontend_context_v9": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "9becf7924f26d695fc4390c6709a81fc729e12244c241482525ca44e5156d67b"
                        ),
                    }
                ),
                "public_frontend_context_v10": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "1e3143be1a4e0e4e293b63bd0a4a8cfd264abcd557c09b50b3d611fd2e8a993e"
                        ),
                    }
                ),
                "public_frontend_context_v11": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "8975154a14504466c714b581bd3a8dacadfb89fc7141542e3a6a476b54800ecd"
                        ),
                    }
                ),
                "public_frontend_context_v12": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "12b6972a0d9ef800ccde574d970132c2aa66ac5e1dfbdd8dc39ce8d367e1c3a2"
                        ),
                    }
                ),
                "public_frontend_context_v13": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "08640a27e471eff751f9dcaed01c5292a8ceb073fe988d89064714668611aef4"
                        ),
                    }
                ),
                "public_frontend_context_v14": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "60cd010e25be5a4089629ac56f3f6d27bc4e20765bb410472604301059f45014"
                        ),
                    }
                ),
                "public_frontend_context_v15": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "c7706c0d986a883cb30f703f67238885d9395b5fb9fa0251c443989edfce4b7e"
                        ),
                        "accepted_context_sha256": (
                            "c7706c0d986a883cb30f703f67238885d9395b5fb9fa0251c443989edfce4b7e",
                            "bd3175a91b8dfc0de729b5a7ba7c6dbb15ceab6238cc573f5c5c4c98632053f4",
                        ),
                    }
                ),
                "public_frontend_context_v16": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "31f39b26af2782660286f7f8de87e40114cfdc45c0082c7a40ab31a7d69abce4"
                        ),
                    }
                ),
            }
        ),
        "cute": MappingProxyType(
            {
                "public_frontend_context_v1": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "13c3f240eb4bd19bfbbc6982e3058ebb4ecc37c139a1e483cff0cff6a811bb37"
                        ),
                    }
                ),
            }
        ),
        "tilelang": MappingProxyType(
            {
                "public_frontend_context_v1": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "b54d1d1e65f5ad35a39b47aa43b7bee9eb68e4067c82d32e1b7739bc99c2660e"
                        ),
                    }
                ),
                "public_frontend_context_v2": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "33b3071963137ccb209c436d28ba57f756889dc0b68eadae1764c2d9402dea26"
                        ),
                    }
                ),
                "public_frontend_context_v3": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "13c6794712ea47de5f4636eff36978956ae89def5edac719563b5d3af084a5ec"
                        ),
                    }
                ),
                "public_frontend_context_v4": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "7830f9d36e818af0309fc9dcd259c1571836058e02a3f3931485b3033e7f6e7d"
                        ),
                    }
                ),
                "public_frontend_context_v5": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "8d03114a6f9b0cfe89f9a7cad12e48d429359d32c352737055717730a8be8451"
                        ),
                    }
                ),
                "public_frontend_context_v6": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "b600cb80e9c1496252b024b3a2257332a74a64ac77dc1c75ca70b7c8f5ea7140"
                        ),
                    }
                ),
            }
        ),
        "cuda": MappingProxyType(
            {
                "public_frontend_context_v1": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "83d5c95bd2c9bc28329e42987c4eb9628395990a5b2bb8c4d07aad81fb13a530"
                        ),
                    }
                ),
                "public_frontend_context_v2": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "a476dece1f8de76225e4352bf2dd63f04962a048f6f3fb6847c1d2cc9e66ed75"
                        ),
                    }
                ),
                "public_frontend_context_v3": MappingProxyType(
                    {
                        "receipt_hash_basis": "full_u_f",
                        "context_sha256": (
                            "1d5bee977f82a852fbde57428bb4b4bdca784c837ede9ae91a5f18d8b75e8e65"
                        ),
                    }
                ),
            }
        ),
    }
)

_FORBIDDEN_KEYS = frozenset(
    {
        "source_code",
        "source_span",
        "source_name",
        "location_map",
        "provenance",
        "response_text",
        "candidate",
        "compiler_diff",
        "selected_cards",
        "registry_digest",
        "harness",
        "interface",
        "binding_map",
        "compiler_fingerprint",
        "gate1_diff",
        "prior_replay",
        "conversation",
    }
)

_REPLAY_COMPONENT_KEYS = {
    "j": frozenset(
        {
            "kernel_id", "intent", "node_ids", "boundary", "signature",
            "scopes", "storage", "dimension_constraints",
            "value_dimension_relations",
        }
    ),
    "c": frozenset({"slots"}),
    "pi": frozenset(
        {
            "regions",
            "instances",
            "routes",
            "order",
            "sharing",
            "refinements",
            "iterations",
            "loop_carriers",
            "physical_iterations",
            "unit_iterations",
            "physical_loop_carriers",
            "terminal_controls",
        }
    ),
}
_HISTORICAL_J_KEYS_WITHOUT_VALUE_DIMENSION_RELATIONS = (
    _REPLAY_COMPONENT_KEYS["j"] - {"value_dimension_relations"}
)
_U_F_KEYS = frozenset(
    {
        "protocol",
        "frontend",
        "replay_function",
        "context_policy",
        "retrieval_scope",
        "tool_access",
        "compile_entry",
        "compile_entry_contract",
    }
)
_U_F_PACKET_KEYS = frozenset({"protocol", "frontend", "replay_function"})


def _canonical_mapping_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def public_frontend_context_digest(context: Mapping[str, Any]) -> str:
    """Digest public-context content independently of any receipt convention."""

    if not isinstance(context, Mapping):
        raise SchemaError("public frontend context must be an object")
    return _canonical_mapping_digest(context)


def audit_public_frontend_context_receipt(
    *,
    frontend: Frontend | str,
    context_policy: str,
    context_sha256: str,
) -> dict[str, str]:
    """Audit an exact historical receipt without upgrading its context version."""

    try:
        frontend = Frontend(frontend)
    except (TypeError, ValueError) as exc:
        raise SchemaError("public frontend context receipt has unknown frontend") from exc
    history = PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY.get(frontend.value)
    if history is None:
        raise SchemaError(
            "public frontend context receipt has no versioned digest ledger for "
            f"{frontend.value!r}"
        )
    policy = history.get(context_policy)
    if policy is None:
        raise SchemaError(
            "public frontend context receipt has unknown context_policy "
            f"{context_policy!r}"
        )
    expected = str(policy["context_sha256"])
    accepted = policy.get("accepted_context_sha256", (expected,))
    if not isinstance(accepted, (tuple, frozenset)):
        accepted = (expected,)
    if context_sha256 not in accepted:
        raise SchemaError(
            "public frontend context receipt digest does not match its declared "
            f"policy {context_policy!r}"
        )
    active = PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY[frontend.value]
    return {
        "frontend": frontend.value,
        "context_policy": context_policy,
        "context_sha256": context_sha256,
        "receipt_hash_basis": str(policy["receipt_hash_basis"]),
        "status": "current" if context_policy == active else "historical",
    }


def audit_public_frontend_context_payload(
    *,
    frontend: Frontend | str,
    context_policy: str,
    public_context_sha256: str,
) -> dict[str, str]:
    """Audit public context embedded in a stored packet, not its old receipt."""

    try:
        frontend = Frontend(frontend)
    except (TypeError, ValueError) as exc:
        raise SchemaError("public frontend context payload has unknown frontend") from exc
    history = PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY.get(frontend.value)
    if history is None or context_policy not in history:
        raise SchemaError(
            "public frontend context payload has unknown context_policy "
            f"{context_policy!r}"
        )
    aliases = _PUBLIC_FRONTEND_CONTEXT_DIGEST_ALIASES.get(frontend.value, {})
    accepted = aliases.get(context_policy, frozenset({history[context_policy]}))
    if public_context_sha256 not in accepted:
        raise SchemaError(
            "public frontend context payload digest does not match its declared "
            f"policy {context_policy!r}"
        )
    active = PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY[frontend.value]
    return {
        "frontend": frontend.value,
        "context_policy": context_policy,
        "public_context_sha256": public_context_sha256,
        "status": "current" if context_policy == active else "historical",
    }


def _assert_active_public_frontend_context(frontend: Frontend) -> None:
    active = PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY.get(frontend.value)
    if active is None:
        return
    context = PUBLIC_FRONTEND_CONTEXT.get(frontend.value)
    if not isinstance(context, Mapping) or context.get("context_policy") != active:
        raise SchemaError(
            f"active public frontend context policy drifted for {frontend.value!r}"
        )
    audit_public_frontend_context_payload(
        frontend=frontend,
        context_policy=active,
        public_context_sha256=public_frontend_context_digest(context),
    )


def _public_context_from_u_f(u_f: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in u_f.items()
        if key not in _U_F_PACKET_KEYS
    }


def build_public_frontend_context_receipt(
    u_f: Mapping[str, Any],
) -> dict[str, str]:
    """Build a receipt using the hash basis frozen for its declared policy."""

    if not isinstance(u_f, Mapping):
        raise SchemaError("public frontend context receipt input must be an object")
    try:
        frontend = Frontend(u_f.get("frontend"))
    except (TypeError, ValueError) as exc:
        raise SchemaError("public frontend context receipt has unknown frontend") from exc
    context_policy = u_f.get("context_policy")
    if not isinstance(context_policy, str):
        raise SchemaError("public frontend context receipt requires context_policy")
    history = PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY.get(frontend.value)
    policy = history.get(context_policy) if history is not None else None
    if policy is None:
        raise SchemaError(
            "public frontend context receipt has unknown context_policy "
            f"{context_policy!r}"
        )
    public_context = _public_context_from_u_f(u_f)
    audit_public_frontend_context_payload(
        frontend=frontend,
        context_policy=context_policy,
        public_context_sha256=public_frontend_context_digest(public_context),
    )
    basis = str(policy["receipt_hash_basis"])
    if basis == "full_u_f":
        digest = _canonical_mapping_digest(u_f)
    elif basis == "public_context":
        digest = public_frontend_context_digest(public_context)
    else:  # pragma: no cover - frozen ledger construction invariant
        raise SchemaError(f"unknown Replay receipt hash basis {basis!r}")
    return audit_public_frontend_context_receipt(
        frontend=frontend,
        context_policy=context_policy,
        context_sha256=digest,
    )

def _walk(value: Any, path: str = "$" ):
    yield path, value
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _validate_source_isolated_payload(
    payload: Mapping[str, Any],
    *,
    require_active_context: bool,
) -> None:
    if not isinstance(payload, Mapping):
        raise SchemaError("Replay payload must be an object")
    if set(payload) != {"j", "c", "pi", "u_f"}:
        raise SchemaError(
            "Replay payload root must contain exactly j, c, pi, and u_f"
        )
    for component, expected in _REPLAY_COMPONENT_KEYS.items():
        value = payload[component]
        actual = set(value) if isinstance(value, Mapping) else set()
        historical_j = (
            not require_active_context
            and component == "j"
            and actual == _HISTORICAL_J_KEYS_WITHOUT_VALUE_DIMENSION_RELATIONS
        )
        if not isinstance(value, Mapping) or (
            actual != expected and not historical_j
        ):
            raise SchemaError(
                f"Replay payload {component} must contain exactly "
                + ", ".join(sorted(expected))
            )
    u_f = payload["u_f"]
    if not isinstance(u_f, Mapping):
        raise SchemaError("Replay payload u_f must be an object")
    unknown_u_f = sorted(set(u_f) - _U_F_KEYS)
    if unknown_u_f:
        raise SchemaError(
            f"Replay payload u_f has unknown field(s): {unknown_u_f}"
        )
    required_u_f = set(_U_F_PACKET_KEYS)
    missing_u_f = sorted(required_u_f - set(u_f))
    if missing_u_f:
        raise SchemaError(
            f"Replay payload u_f is missing field(s): {missing_u_f}"
        )
    if u_f["protocol"] != REPLAY_PACKET_SCHEMA:
        raise SchemaError("Replay payload u_f uses an unsupported protocol")
    try:
        frontend = Frontend(u_f["frontend"])
    except (TypeError, ValueError) as exc:
        raise SchemaError("Replay payload u_f has an unsupported frontend") from exc
    if u_f["replay_function"] != "scrc_k0":
        raise SchemaError("Replay payload u_f must use canonical replay_function")
    if require_active_context:
        _assert_active_public_frontend_context(frontend)
        fixed_context = PUBLIC_FRONTEND_CONTEXT.get(frontend.value, {})
        missing_context = sorted(set(fixed_context) - set(u_f))
        if missing_context:
            raise SchemaError(
                "Replay payload u_f is missing fixed context field(s): "
                f"{missing_context}"
            )
        modified_context = sorted(
            key for key, value in fixed_context.items() if u_f.get(key) != value
        )
        if modified_context:
            raise SchemaError(
                "Replay payload u_f does not match the fixed public frontend "
                f"context field(s): {modified_context}"
            )
        if frontend.value in PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY:
            receipt = build_public_frontend_context_receipt(u_f)
            if receipt["status"] != "current":  # pragma: no cover
                raise SchemaError("fresh Replay payload uses a historical context")
    else:
        context_policy = u_f.get("context_policy")
        if not isinstance(context_policy, str):
            raise SchemaError(
                "historical Replay payload u_f requires context_policy"
            )
        audit_public_frontend_context_payload(
            frontend=frontend,
            context_policy=context_policy,
            public_context_sha256=public_frontend_context_digest(
                _public_context_from_u_f(u_f)
            ),
        )
    for path, value in _walk(dict(payload)):
        if isinstance(value, dict):
            forbidden = sorted(set(value) & _FORBIDDEN_KEYS)
            if "source" in value and not (
                path.startswith("$.pi.routes[")
                or path.startswith("$.pi.sharing[")
            ):
                forbidden.append("source")
            if forbidden:
                raise SchemaError(
                    f"Replay payload {path} contains privileged key(s): {forbidden}"
                )
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    for token in (
        "Rule" + "Card",
        "Rule" + "Registry",
        "compose_" + "provenance",
        "selected_" + "rule_cards",
    ):
        if token in encoded:
            raise SchemaError("Replay payload contains retired control-plane data")


def assert_source_isolated_payload(payload: Mapping[str, Any]) -> None:
    """Fail closed if a fresh Replay packet is not source-isolated and current."""

    _validate_source_isolated_payload(payload, require_active_context=True)


def _canonical_replay_packet_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_replay_packet_digest(
    payload: Mapping[str, Any],
    *,
    expected_digest: str,
) -> dict[str, str]:
    """Audit a stored current or historical packet without accepting it as fresh."""

    _validate_source_isolated_payload(payload, require_active_context=False)
    actual = _canonical_replay_packet_digest(payload)
    if actual != expected_digest:
        raise SchemaError("stored Replay packet digest mismatch")
    u_f = payload["u_f"]
    context_audit = audit_public_frontend_context_payload(
        frontend=str(u_f["frontend"]),
        context_policy=str(u_f["context_policy"]),
        public_context_sha256=public_frontend_context_digest(
            _public_context_from_u_f(u_f)
        ),
    )
    return {**context_audit, "packet_sha256": actual}


@dataclass(frozen=True)
class ReplayPacket:
    """Typed source-isolated request; this is not a fourth proof artifact."""

    j: Mapping[str, Any]
    c: Mapping[str, Any]
    pi: Mapping[str, Any]
    u_f: Mapping[str, Any]

    def __post_init__(self) -> None:
        assert_source_isolated_payload(self.to_json(validate=False))

    def to_json(self, *, validate: bool = True) -> dict[str, Any]:
        payload = {
            "j": copy.deepcopy(dict(self.j)),
            "c": copy.deepcopy(dict(self.c)),
            "pi": copy.deepcopy(dict(self.pi)),
            "u_f": copy.deepcopy(dict(self.u_f)),
        }
        if validate:
            assert_source_isolated_payload(payload)
        return payload




def build_rewriter_packet(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> ReplayPacket:
    """Build the one canonical ``J + pi + MaskC(C) + U_f`` packet."""

    # Local import avoids a package initialization cycle: TypeCheck depends on
    # the Gate1 data model, while this builder is its first production consumer.
    from intentir.verify.typecheck import type_check

    artifact = canonicalize_gate1_artifact(artifact)
    report = type_check(artifact.j, artifact.c, artifact.pi)
    if not report.passed:
        summary = ", ".join(
            f"{item.code.value}@{item.path}" for item in report.diagnostics
        )
        raise SchemaError("Replay requires TypeCheck(J,C,pi)=PASS: " + summary)
    view = artifact.rewriter_view(masked=masked)
    frontend = Frontend(frontend)
    _assert_active_public_frontend_context(frontend)
    context = copy.deepcopy(PUBLIC_FRONTEND_CONTEXT.get(frontend.value, {}))
    context.update(
        {
            "protocol": REPLAY_PACKET_SCHEMA,
            "frontend": frontend.value,
            "replay_function": "scrc_k0",
        }
    )
    return ReplayPacket(view["j"], view["c"], view["pi"], context)


def build_rewriter_payload(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> dict[str, Any]:
    """Build exactly ``J + pi + MaskC(C) + U_f``."""

    return build_rewriter_packet(
        artifact, frontend=frontend, masked=masked
    ).to_json()


def build_normal_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=False,
    )


def build_masked_rewriter_view(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> dict[str, Any]:
    return build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=True,
    )


def _instructions(*, frontend: Frontend, masked: bool) -> str:
    masked_clause = (
        "J has intentionally been masked for a single representation-partition audit. "
        if masked else ""
    )
    output_contract = (
        "Return CUDA C++ source only."
        if frontend is Frontend.CUDA
        else "Return Python source only."
    )
    frontend_realization_clause = (
        "Preserve every address_base -> address_lane route. If a pointer/tensor "
        "J.signature carrier directly enters address_base, materialize one base "
        "pointer/address operation and a second lane pointer/address operation. "
        "If the carrier bypasses address_base and enters address_lane or the final "
        "address stage while the base has only integer/domain/C inputs, merge the "
        "integer contributions before exactly one pointer/address operation. Decide "
        "only from J.signature kinds and pi carrier routes, never from Source names, "
        "paths, text, or the base-to-lane edge alone. Program axis identity belongs "
        "to the J domain anchor, not instance multiplicity: materialize repeated "
        "program-coordinate instances on one d* as distinct occurrences of the "
        "same axis, never as a new axis. For one pi realize instance anchored to a "
        "canonical, public-tl.dot-supported rank-2 floating J contraction with one "
        "contracted label, use one direct tl.dot with the declared accumulation "
        "dtype; do not expand it into outer-product multiply plus tl.sum. Do not "
        "apply this spelling outside those support conditions or when pi explicitly "
        "declares multiple contraction stages. Distinct ordered transfer_direct "
        "instances remain distinct compiler-visible transfers even when they read "
        "the same address. If Triton would CSE such transfers, use a public closed "
        "eviction_policy spelling to retain them; the observer normalizes the exact "
        "closed eviction spelling, so it carries no Source information. Never add "
        "a transfer absent from J/pi or collapse one that is present. "
        if frontend is Frontend.TRITON
        else ""
    )
    return (
        "You are the source-blind Replay LLM. Generate one complete, self-contained "
        f"{frontend.value} public-frontend kernel module. You never see "
        "the source implementation. Consume J, pure C and pi directly; pi is topology, "
        "not semantic authority. Treat every pi instance, route, order edge, sharing "
        "relation, structured or physical iteration relation, physical carrier, "
        "representation refinement and region placement as required "
        "realization topology. Region placement is exact lexical scope: emit each "
        "compiler-visible instance in its declared region, keep a loop-region "
        "instance lexically inside that loop even when it is induction-invariant, "
        "and never hoist, sink, clone, or fold it across a region boundary. In "
        "every pi order item, the instance named by "
        "before must execute before the instance named by after, including across "
        "regions and observable effects. Resolve each refinement auth_ref only through the "
        "matching J realization_authorization, materialize it at the refined "
        "instance before routed consumers, and never infer, omit, move, or copy a "
        "sensitive conversion onto a sibling instance. A logical_ref=null p* may "
        "be read only when pi contains its anchored instance and a route from that "
        "instance to a realization consumer; otherwise it remains an unused ABI "
        "parameter. Multiple "
        "instances anchored to one handle are distinct source stages: do not collapse "
        "their chain, discard J/C dependencies, duplicate a shared producer, "
        "or invent an undocumented stage. Obtain callable logical handles from "
        "J.signature and append the anonymous slots from MaskC(C). Obtain all "
        "logical operators, predicates, fills, constants, index relations and "
        "effects only from J. Preserve explicit J stages: reduce(sum) followed by "
        "division remains sum then division, never mean or a generated sum-of-ones "
        "count. Treat every J.value_dimension_relations item as an already "
        "J-certified precondition: relation=eq permits the declared integral "
        "rank-0 value_ref and dimension_ref to denote the same logical extent. "
        "Consume only the declared relation where J shapes, domains, predicates, "
        "or operators require it. Never infer, create, delete, repair, weaken, or "
        "strengthen a value/dimension relation from concrete values, parameter "
        "positions, C, pi, Harness conventions, or frontend behavior. A boundary "
        "access with no predicate is unconditional: "
        "never derive a redundant mask from domains, grid extents, C, or defensive "
        "safety. "
        "Treat each J effect_if as effect control, not a value expression. Ordinarily "
        "emit one real runtime if with explicit positive and negative arms in the "
        "authored polarity, keep every arm-local operation/effect in authored "
        "order, and spell an authored empty arm explicitly. If and only if one "
        "pi.terminal_controls item names the two realize instances anchored to that "
        "effect_if's sibling arm scopes, emit the named terminal arm as one runtime "
        "if ending in a bare return and emit the named continuation arm immediately "
        "after it without an else; negate the condition exactly when the negative "
        "arm is terminal. The relation supplies only arm topology: never guess, copy, "
        "or change the J predicate, polarity, arm operations, or effects. Never lower effect_if "
        "to tl.where/select or execute either arm unconditionally under a mask. "
        "Treat each J value_if as pure value control, distinct from effect_if and "
        "select: emit one real runtime if with explicit polarity-preserving arms; "
        "each arm may use pure local temporaries but must finally assign its J "
        "yield to the same merge local. Never lower value_if to tl.where/select, "
        "effect_if, predicated unconditional computation, or compile-time control. "
        "For every pi.iterations rounded_extent_step relation, resolve its "
        "loop_instance to the J loop trip extent and tile_instance to the matching "
        "C slot; use lower 0, step equal to that C tile, and upper exactly "
        "ceil_div(trip extent, C tile) * C tile. Do not infer a bound from current "
        "shapes/values or invent another rounding formula. "
        "For every pi.physical_iterations relation, keep J's operation or boundary "
        "traversal logical while materializing the explicit physical loop from its "
        "same-subject C-attached axis, induction_basis, and direction. Preserve every "
        "pi.physical_loop_carriers init/state/next/result temporal stage; neither "
        "relation authorizes a new semantic operator or formula. "
        "For every pi.unit_iterations relation, materialize exactly one unit-step "
        "physical loop over its authored J axis: [0, logical extent) for forward, or "
        "the same points descending for reverse. It has no C tile and authorizes no "
        "semantic J loop, formula, rounded bound, launch-derived bound, or operator. "
        "A participant-parallel domain carrier routed into address_lane is not a "
        "unit iteration. If a J reduce consumes its reduced axis through that lane "
        "path, preserve a frontend collective over those participants; never replace "
        "it with a per-participant serial traversal of the whole axis. Read the "
        "reducer only from J and the participant topology only from pi. "
        + frontend_realization_clause
        + "Do not output "
        "analysis, patches, placeholders, helper prose or markdown. Use only public "
        "frontend APIs and the canonical opaque handles in J.signature/MaskC(C). "
        "Pi graph "
        "reference contract: "
        + render_realization_witness_wire_grammar()
        + " Pi mechanism contract: "
        + render_realization_mechanism_grammar()
        + " "
        + masked_clause
        + output_contract
    )


def build_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
    masked: bool = False,
) -> tuple[str, str]:
    payload = build_rewriter_payload(
        artifact,
        frontend=frontend,
        masked=masked,
    )
    frontend = Frontend(frontend)
    return _instructions(frontend=frontend, masked=masked), json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False
    )


def build_normal_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact,
        frontend=frontend,
        masked=False,
    )


def build_masked_rewriter_prompt(
    artifact: Gate1Artifact,
    *,
    frontend: Frontend | str = Frontend.TRITON,
) -> tuple[str, str]:
    return build_rewriter_prompt(
        artifact,
        frontend=frontend,
        masked=True,
    )


def replay_packet_digest(payload: Mapping[str, Any]) -> str:
    assert_source_isolated_payload(payload)
    return _canonical_replay_packet_digest(payload)


__all__ = [
    "PUBLIC_FRONTEND_CONTEXT", "PUBLIC_FRONTEND_CONTEXT_ACTIVE_POLICY",
    "PUBLIC_FRONTEND_CONTEXT_DIGEST_HISTORY",
    "PUBLIC_FRONTEND_CONTEXT_RECEIPT_HISTORY", "REPLAY_PACKET_SCHEMA",
    "ReplayPacket", "audit_public_frontend_context_payload",
    "audit_public_frontend_context_receipt",
    "assert_source_isolated_payload", "audit_replay_packet_digest",
    "build_masked_rewriter_prompt",
    "build_masked_rewriter_view", "build_normal_rewriter_prompt",
    "build_normal_rewriter_view", "build_rewriter_payload",
    "build_public_frontend_context_receipt", "build_rewriter_packet",
    "build_rewriter_prompt",
    "public_frontend_context_digest", "replay_packet_digest",
]
