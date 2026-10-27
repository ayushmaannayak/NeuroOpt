"""Natural (NOT seeded) decision set for the optional extended rule pack.

Takes held-out TEST-split, hardware-timeable BHive blocks exactly as they are (no inverse rewrites, no inserted
instructions) and keeps those with at least one legal extended-rule opportunity. This answers the question the
seeded set cannot: does a learned oracle help on real code, where the benefit of a rewrite depends on context?
"""
import argparse, gzip, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuroopt.decode import decode_block  # noqa: E402
from neuroopt.rules import find_rewrites  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-blocks", type=int, default=1000)
    args = ap.parse_args()
    rows = [json.loads(l) for l in gzip.open(ROOT / "data/processed/bhive_skl.jsonl.gz", "rt")]
    cands = [r for r in rows if r["split"] == "test" and r["timeable"]]
    kept, n_any = [], 0
    for r in cands:
        b = decode_block(r["hex_bytes"])
        core = bool(find_rewrites(b)[0]); ext = find_rewrites(b, extended=True)[0]
        if ext:
            n_any += 1
            kept.append(dict(source="natural", block_id=r["block_id"], hex=r["hex_bytes"], notes=[], bhive_y=r["target_y"],
                             split="test", core_rule_applies=core))
    random.Random(args.seed).shuffle(kept)
    kept = kept[: args.max_blocks]
    out = ROOT / "data/processed/natural_decision_set.jsonl"
    with open(out, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    print(f"timeable test blocks: {len(cands)}; with >=1 legal extended opportunity: {n_any}; wrote {len(kept)} -> {out}")


if __name__ == "__main__":
    main()
