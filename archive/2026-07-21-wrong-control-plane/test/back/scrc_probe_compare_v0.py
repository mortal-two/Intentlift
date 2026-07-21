"""Archived mechanical summary for the initial manual TTIR probe review.

This is deliberately not the proposed production fingerprint extractor.  It
only counts the structures that were then inspected in the raw unified diffs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


OP_NAME = r"(?:tt|arith|math|scf)\.[a-z_]+(?:\.[a-z_]+)*"
OP_RE = re.compile(
    rf'(?:=\s+|^\s*)(?:"(?P<quoted>{OP_NAME})"|(?P<plain>{OP_NAME}))(?=[\s(])'
)
REDUCE_RE = re.compile(r'"tt\.reduce".*axis = (?P<axis>-?\d+)')
CMP_RE = re.compile(r"arith\.cmpi\s+(?P<predicate>[a-z]+),")
CONST_RE = re.compile(r"arith\.constant\s+(?P<value>dense<[^>]+>|-?\d+(?:\.\d+)?)")


def _operand_count(line: str, opname: str) -> int:
    tail = line.split(opname, 1)[1].split(" : ", 1)[0]
    return len(re.findall(r"%[A-Za-z0-9_]+", tail))


def summarize(path: Path) -> dict[str, object]:
    lines = path.read_text().splitlines()
    body = [line for line in lines if not line.startswith("#loc")]
    ops: Counter[str] = Counter()
    cmp_predicates: Counter[str] = Counter()
    constants: Counter[str] = Counter()
    masked_loads = unmasked_loads = 0
    masked_stores = unmasked_stores = 0
    reductions: list[dict[str, object]] = []

    for i, line in enumerate(body):
        for match in OP_RE.finditer(line):
            ops[match.group("quoted") or match.group("plain")] += 1
        if match := CMP_RE.search(line):
            cmp_predicates[match.group("predicate")] += 1
        if match := CONST_RE.search(line):
            constants[match.group("value")] += 1
        if "tt.load" in line:
            if _operand_count(line, "tt.load") >= 3:
                masked_loads += 1
            else:
                unmasked_loads += 1
        if "tt.store" in line:
            if _operand_count(line, "tt.store") >= 3:
                masked_stores += 1
            else:
                unmasked_stores += 1
        if match := REDUCE_RE.search(line):
            combiner = None
            for region_line in body[i + 1 :]:
                if "tt.reduce.return" in region_line:
                    break
                for candidate in ("arith.addf", "arith.maxnumf", "arith.minnumf"):
                    if candidate in region_line:
                        combiner = candidate
                        break
                if combiner:
                    break
            reductions.append({"axis": int(match.group("axis")), "combiner": combiner})

    func_line = next(line.strip() for line in body if "tt.func public" in line)
    signature = func_line.split(" attributes ", 1)[0]
    signature = re.sub(r'\s+loc\("[^"]+"\(#loc\)\)', "", signature)
    return {
        "signature": signature,
        "ops": dict(sorted(ops.items())),
        "reductions": reductions,
        "masks": {
            "cmp_predicates": dict(sorted(cmp_predicates.items())),
            "masked_loads": masked_loads,
            "unmasked_loads": unmasked_loads,
            "masked_stores": masked_stores,
            "unmasked_stores": unmasked_stores,
            "and_nodes": ops["arith.andi"],
        },
        "constants": dict(sorted(constants.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", type=Path)
    args = parser.parse_args()

    report: dict[str, object] = {}
    for source in sorted(args.dump_dir.glob("*.source.ttir")):
        name = source.name.removesuffix(".source.ttir")
        replay = args.dump_dir / f"{name}.replay.ttir"
        source_summary = summarize(source)
        replay_summary = summarize(replay)
        report[name] = {
            "source": source_summary,
            "replay": replay_summary,
            "same_signature": source_summary["signature"] == replay_summary["signature"],
            "same_op_multiset": source_summary["ops"] == replay_summary["ops"],
            "same_reductions": source_summary["reductions"] == replay_summary["reductions"],
            "same_mask_summary": source_summary["masks"] == replay_summary["masks"],
            "same_constants": source_summary["constants"] == replay_summary["constants"],
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
