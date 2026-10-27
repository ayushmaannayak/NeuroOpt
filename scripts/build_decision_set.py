"""Build the decision-evaluation set.

Compilers already apply R1-R3 almost everywhere, so real BHive blocks rarely
contain a *legal* opportunity. We therefore use two sources:

  natural : held-out BHive blocks that already contain a legal opportunity;
  seeded  : held-out, register-only BHive blocks "de-optimized" with inverse
            rewrites:  xor r,r -> mov r,0 ;  shl r,k / add r,r -> imul r,r,2^k ;
            plus an inserted mov r64,r64 self-copy.

The seeded block is a new input program B. Whether the guards then allow the
forward rewrite depends on the surrounding flag uses (e.g. mov r,0 before an
adc stays), exactly as for real code. Only test-split blocks are used, so the
LSTM never trained on the source blocks.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.decode import GPR32, GPR64, block_bytes, decode_block  # noqa: E402
from neuroopt.emit import assemble_one  # noqa: E402
from neuroopt.hwtime import timeable  # noqa: E402
from neuroopt.rules import find_rewrites  # noqa: E402


def _gpr(op):
    return op.kind == "reg" and (op.reg in GPR64 or op.reg in GPR32)


def inverse_sites(block):
    """(index, replacement text, kind) for every place an inverse rewrite fits."""
    sites = []
    for i, ins in enumerate(block):
        ops = ins.operands
        if ins.mnemonic == "xor" and len(ops) == 2 and _gpr(ops[0]) and ops[1].kind == "reg" and ops[0].reg == ops[1].reg:
            sites.append((i, f"mov {ops[0].reg}, 0", "R1"))
        elif ins.mnemonic in ("shl", "sal") and len(ops) == 2 and _gpr(ops[0]) and ops[1].kind == "imm" and 1 <= ops[1].imm <= 30:
            sites.append((i, f"imul {ops[0].reg}, {ops[0].reg}, {2 ** ops[1].imm}", "R2"))
        elif ins.mnemonic == "add" and len(ops) == 2 and _gpr(ops[0]) and ops[1].kind == "reg" and ops[0].reg == ops[1].reg:
            sites.append((i, f"imul {ops[0].reg}, {ops[0].reg}, 2", "R2"))
    return sites


def seed_block(block, rng: random.Random):
    out = list(block)
    notes = []
    for i, text, kind in inverse_sites(block):
        if rng.random() < 0.8:
            out[i] = assemble_one(text)
            notes.append(f"{kind}-inverse @{i}: {block[i].text} -> {text}")
    used = sorted({r for ins in block for r in ins.regs_read + ins.regs_write if r in GPR64 and r != "rsp"})
    if used and rng.random() < 0.6:
        r = rng.choice(used)
        pos = rng.randint(0, len(out))
        out.insert(pos, assemble_one(f"mov {r}, {r}"))
        notes.append(f"R3-inverse: inserted mov {r}, {r} @{pos}")
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-r3-only", type=int, default=600)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    natural, seeded = [], []
    with gzip.open(ROOT / "data" / "processed" / "bhive_skl.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        # Natural register-only opportunities are rare, so take them from every split
        # (their split is recorded; the LSTM may have trained on train-split ones).
        if not r["timeable"]:
            continue
        block = decode_block(r["hex_bytes"])
        if find_rewrites(block)[0]:
            natural.append(dict(source="natural", block_id=r["block_id"], hex=r["hex_bytes"], notes=[],
                                bhive_y=r["target_y"], split=r["split"]))
    cands = [r for r in rows if r["split"] == "test" and r["timeable"]]
    rng.shuffle(cands)
    r3_only = 0
    for r in cands:
        block = decode_block(r["hex_bytes"])
        new, notes = seed_block(block, rng)
        if not notes or not timeable(new):
            continue
        if not find_rewrites(new)[0]:
            continue   # every injected site is refused by a guard: no decision to make
        if all(n.startswith("R3") for n in notes):
            if r3_only >= args.max_r3_only:
                continue
            r3_only += 1
        seeded.append(dict(source="seeded", block_id=r["block_id"], hex=block_bytes(new).hex(),
                           notes=notes, bhive_y=r["target_y"], split="test"))
    out = ROOT / "data" / "processed" / "decision_set.jsonl"
    with open(out, "w") as f:
        for rec in natural + seeded:
            f.write(json.dumps(rec) + "\n")
    print(f"natural={len(natural)} seeded={len(seeded)} -> {out}")


if __name__ == "__main__":
    main()
