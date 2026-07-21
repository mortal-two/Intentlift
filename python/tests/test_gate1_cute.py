from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from intentir.evaluation.real_batch import (
    cute_entry_symbol,
    cute_compile_case,
    load_batch_manifest,
    probe_cute_compilation,
    resolve_source_path,
)
from intentir.verify.gate1.cute import (
    compare_cute_fingerprints,
    fingerprint_cute,
    normalize_cute_mlir,
)
from intentir.verify.gate1.cute_codegen import (
    UnsafeCuteReplayCodeError,
    validate_cute_module,
)
from intentir.verify.gate1.cute_compiler import CuteCompiler, CuteKernelSource


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "test" / "real_llm_batch_28.json"


class _RenameCuteEntrypoints(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        replacement = {"kernel": "scrc_k0", "go": "scrc_entry"}.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.name = {"kernel": "scrc_k0", "go": "scrc_entry"}.get(
            node.name, node.name
        )
        return node


def _source_blind_alpha_replay(source: str) -> str:
    """Build the policy-sized module used to test real replay compilation.

    This helper changes no kernel operation: it keeps the two public CuTe
    functions and compile-time literals, removes only source-side host running
    code/imports, and alpha-renames the required exported symbols.
    """

    tree = ast.parse(source)
    expanded_constants: list[ast.stmt] = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], (ast.Tuple, ast.List))
            and isinstance(node.value, (ast.Tuple, ast.List))
            and len(node.targets[0].elts) == len(node.value.elts)
        ):
            expanded_constants.extend(
                ast.Assign(targets=[target], value=value)
                for target, value in zip(
                    node.targets[0].elts, node.value.elts, strict=True
                )
            )
        else:
            expanded_constants.append(node)
    tree.body = expanded_constants
    allowed_imports = {"cutlass", "cutlass.cute", "cutlass.pipeline"}
    allowed_from_imports = {
        "cutlass",
        "cutlass.cute.nvgpu",
        "cutlass.pipeline.sm90",
    }
    tree.body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and all(alias.name in allowed_imports for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module in allowed_from_imports
        )
        or (
            isinstance(node, ast.Assign)
            and all(
                isinstance(target, ast.Name) and target.id.isupper()
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in {"kernel", "go"}
        )
    ]
    renamed = _RenameCuteEntrypoints().visit(tree)
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed) + "\n"


def _mlir(*, symbol: str, argument: str, result: str, location: str) -> str:
    return f"""
#loc = loc(unknown)
#loc1 = loc(\"{location}\":1:1)
module attributes {{gpu.container_module}} {{
  gpu.module @{symbol} {{
    cuda.kernel @{symbol}_kernel({argument}: !llvm.ptr<1> loc(#loc1)) attributes {{gpu.kernel}} {{
      {result} = nvvm.read.ptx.sreg.tid.x : i32 loc(#loc1)
      nvvm.mbarrier.init.shared {argument}, {result} : !llvm.ptr<1>, i32 loc(#loc1)
      cuda.return loc(#loc1)
    }} loc(#loc1)
  }} loc(#loc1)
}} loc(#loc1)
"""


def test_cute_observer_alpha_normalizes_names_and_locations() -> None:
    left = _mlir(symbol="source", argument="%arg0", result="%tid", location="a.py")
    right = _mlir(symbol="replay", argument="%input", result="%lane", location="b.py")
    assert normalize_cute_mlir(left) == normalize_cute_mlir(right)
    ptx = """
.version 8.8
.target sm_120a
    mbarrier.init.shared.b64 [%r1], 1;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r2], [%rd1], 16, [%r1];
    bar.sync 0;
"""
    source = fingerprint_cute(left, ptx, python_interface={"arity": 1})
    replay = fingerprint_cute(right, ptx, python_interface={"arity": 1})
    comparison = compare_cute_fingerprints(source, replay)
    assert source.digest == replay.digest
    assert comparison.hard_pass


def test_cute_observer_keeps_raw_bulk_and_tma_distinct() -> None:
    mlir = _mlir(symbol="module", argument="%arg0", result="%tid", location="x.py")
    raw = fingerprint_cute(
        mlir,
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r1], [%rd1], 16, [%r2];",
    )
    tensor = fingerprint_cute(
        mlir,
        "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r1], [%rd1], {%r2}, [%r3];",
    )
    comparison = compare_cute_fingerprints(raw, tensor)
    assert not comparison.hard_pass
    assert {item.category for item in comparison.differences} >= {
        "ptx_topology",
        "async_operations",
    }


_SAFE_REPLAY = """import cutlass
import cutlass.cute as cute

@cute.kernel
def scrc_k0(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    dst[tidx] = src[tidx]

@cute.jit
def scrc_entry(src, dst, stream):
    scrc_k0(src, dst).launch(
        grid=[1, 1, 1], block=[32, 1, 1], smem=0, stream=stream
    )
"""


def test_cute_replay_policy_accepts_public_kernel_and_rejects_host_io() -> None:
    validation = validate_cute_module(
        _SAFE_REPLAY,
        required_kernel="scrc_k0",
        required_entry="scrc_entry",
    )
    assert validation.kernel_name == "scrc_k0"
    unsafe = _SAFE_REPLAY.replace(
        "tidx, _, _ = cute.arch.thread_idx()", "open('answer.txt').read()"
    )
    with pytest.raises(UnsafeCuteReplayCodeError, match="host-side"):
        validate_cute_module(
            unsafe,
            required_kernel="scrc_k0",
            required_entry="scrc_entry",
        )


@pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="nvidia-cutlass-dsl is unavailable",
)
def test_frozen_cute_sources_compile_with_strict_observer(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manifest = load_batch_manifest(_MANIFEST)
    cases = [
        case for case in manifest.cases if case["source"]["frontend"] == "cute"
    ]
    assert [case["case_id"] for case in cases] == [
        "cute_async_bulk_copy",
        "cute_async_tma_copy",
    ]
    assert all(len(cute_compile_case(case).entry_arguments) == 2 for case in cases)
    receipt = probe_cute_compilation(manifest, cache_dir=tmp_path / "cache")
    assert receipt["cute_compiled"] == receipt["cute_total"] == 2
    async_ops = {
        item["case_id"]: item["fingerprint"]["async_operations"]
        for item in receipt["cases"]
    }
    assert any("cp.async.bulk.shared" in item for item in async_ops["cute_async_bulk_copy"])
    assert any(
        "cp.async.bulk.tensor.1d.shared" in item
        for item in async_ops["cute_async_tma_copy"]
    )


@pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None,
    reason="nvidia-cutlass-dsl is unavailable",
)
def test_real_cute_source_and_alpha_replay_strictly_close(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    manifest = load_batch_manifest(_MANIFEST)
    compiler = CuteCompiler(cache_dir=tmp_path / "cache", timeout=300.0)
    fingerprints = {}
    for case in (
        manifest.case("cute_async_bulk_copy"),
        manifest.case("cute_async_tma_copy"),
    ):
        source_path = resolve_source_path(manifest, case)
        source_code = source_path.read_text(encoding="utf-8")
        source = compiler.compile(
            CuteKernelSource.from_file(
                source_path,
                case["source"]["symbol"],
                cute_entry_symbol(source_code, case["source"]["symbol"]),
                trusted=True,
            ),
            cute_compile_case(case),
        )
        assert source.ok, source.diagnostic

        replay_code = _source_blind_alpha_replay(source_code)
        validate_cute_module(
            replay_code,
            required_kernel="scrc_k0",
            required_entry="scrc_entry",
        )
        replay = compiler.compile(
            CuteKernelSource.from_code(
                replay_code,
                "scrc_k0",
                "scrc_entry",
                trusted=False,
            ),
            cute_compile_case(case),
        )
        assert replay.ok, replay.diagnostic
        assert source.mlir is not None and source.ptx is not None
        assert replay.mlir is not None and replay.ptx is not None
        source_fingerprint = fingerprint_cute(source.mlir, source.ptx)
        replay_fingerprint = fingerprint_cute(replay.mlir, replay.ptx)
        assert compare_cute_fingerprints(
            source_fingerprint, replay_fingerprint
        ).hard_pass
        fingerprints[case["case_id"]] = source_fingerprint

    cross = compare_cute_fingerprints(
        fingerprints["cute_async_bulk_copy"],
        fingerprints["cute_async_tma_copy"],
    )
    assert not cross.hard_pass
    assert {item.category for item in cross.differences} >= {
        "ptx_topology",
        "async_operations",
    }
