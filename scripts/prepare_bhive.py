"""M1: BHive preparation -> normalized train/validation/test manifest.

Record (data contract from the design):
  block_id, hex_bytes, instruction_tokens, cpu_arch, target_y, split
plus bookkeeping: n_instr, families, split_family, timeable.

Hygiene:
  * drop invalid / partially decoded blocks;
  * dedupe: blocks whose canonical token sequence is identical form one group and
    always land in the same split (no near-duplicate train/test leakage);
  * CPU label and units explicit: cpu_arch='skl', target_y in cycles/iteration;
  * a second split holds out whole program families (source-disjoint).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.decode import DecodeError, decode_block  # noqa: E402
from neuroopt.hwtime import timeable  # noqa: E402
from neuroopt.tokenizer import block_tokens  # noqa: E402

HELDOUT_FAMILIES = ("sqlite", "redis", "openssl")
MAX_INSTR = 64
MAX_Y = 100.0   # cycles/iteration; larger labels are dominated by measurement artefacts


def family(fname: str) -> str:
    return fname.split("-")[0].split(".")[0]


def _work(row):
    hx, y = row
    try:
        block = decode_block(hx)
    except DecodeError as e:
        return hx, y, None, str(e).split(":")[0]
    if len(block) > MAX_INSTR:
        return hx, y, None, f"more than {MAX_INSTR} instructions"
    return hx, y, (block_tokens(block), len(block), timeable(block)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="skl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    raw = ROOT / "data" / "raw"
    rows = []
    with open(raw / f"{args.arch}.csv") as f:
        for r in csv.reader(f):
            if r and r[0]:
                rows.append((r[0].lower(), float(r[1]) / 100.0))   # per-100-iterations -> per iteration
    if args.limit:
        rows = rows[: args.limit]

    fams = defaultdict(set)
    for p in sorted((raw / "sources").glob("*.csv")):
        with open(p) as f:
            for r in csv.reader(f):
                if r and r[0]:
                    fams[r[0].lower()].add(family(p.stem))

    stats = Counter(total=len(rows))
    with Pool() as pool:
        results = pool.map(_work, rows, chunksize=512)

    records, groups = [], defaultdict(list)
    for hx, y, info, err in results:
        if err:
            stats[f"dropped: {err}"] += 1
            continue
        if not (0 < y <= MAX_Y):
            stats["dropped: label outside (0, 100] cycles"] += 1
            continue
        toks, n, tm = info
        canon = "|".join(" ".join(t) for t in toks)
        gid = hashlib.sha1(canon.encode()).hexdigest()
        rec = dict(block_id=hashlib.sha1(hx.encode()).hexdigest()[:16], hex_bytes=hx,
                   instruction_tokens=toks, cpu_arch=args.arch, target_y=y,
                   n_instr=n, families=sorted(fams.get(hx, ())), timeable=tm, group=gid)
        groups[gid].append(len(records))
        records.append(rec)
    stats["kept"] = len(records)
    stats["token groups"] = len(groups)
    stats["blocks sharing a token group with another block"] = sum(len(v) for v in groups.values() if len(v) > 1)

    # Split 1: random by token group (80/10/10).
    for gid, idx in groups.items():
        h = int(gid[:8], 16) % 10
        split = "test" if h == 0 else "val" if h == 1 else "train"
        for k in idx:
            records[k]["split"] = split

    # Split 2: hold out whole program families; mixed groups are dropped from it.
    held = set(HELDOUT_FAMILIES)
    for gid, idx in groups.items():
        fs = [set(records[k]["families"]) for k in idx]
        all_fams = set().union(*fs)
        if not all_fams:
            s = "unknown"
        elif all_fams <= held:
            s = "test"
        elif all_fams & held:
            s = "excluded"
        else:
            s = "val" if int(gid[8:16], 16) % 10 == 0 else "train"
        for k in idx:
            records[k]["split_family"] = s

    out = ROOT / "data" / "processed" / f"bhive_{args.arch}.jsonl.gz"
    with gzip.open(out, "wt") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    summary = dict(stats)
    summary["split (random group)"] = dict(Counter(r["split"] for r in records))
    summary["split (held-out families: " + ",".join(HELDOUT_FAMILIES) + ")"] = dict(
        Counter(r["split_family"] for r in records))
    summary["timeable (register-only)"] = sum(r["timeable"] for r in records)
    summary["family counts"] = dict(Counter(f for r in records for f in r["families"]))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"m1_data_summary_{args.arch}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
