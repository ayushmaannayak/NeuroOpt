"""Correctness evidence: fuzz the guarded rewriter in the emulator.

Guards ON : every rewrite the rules emit (single steps and every multi-step
            candidate in C(B)) must be equivalent to the original.
Guards OFF: apply the same rules with the guards removed. If the harness then
            catches bad rewrites, 'zero failures with guards on' is meaningful
            (a test that can never fail proves nothing).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.decode import block_bytes  # noqa: E402
from neuroopt.fuzz import random_block  # noqa: E402
from neuroopt.rules import apply, candidate_pool, find_rewrites  # noqa: E402
from neuroopt.verify import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=2000)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.time()

    on = Counter()
    off = Counter()
    rules_on, rules_off_caught = Counter(), Counter()
    examples = []
    for n in range(args.blocks):
        block = random_block(rng)
        legal, refused = find_rewrites(block, guards=True)
        legal_keys = {block_bytes(apply(block, rw)) for rw in legal}
        on["blocks"] += 1
        on["refusals"] += len(refused)
        for rw in legal:
            res = verify(block, apply(block, rw), args.trials, seed=n)
            on["single-step rewrites checked"] += 1
            rules_on[rw.rule] += 1
            if res.equivalent is False:
                on["ILLEGAL (single-step)"] += 1
                examples.append(dict(mode="guards on", block=[i.text for i in block], rewrite=rw.describe(),
                                     detail=res.detail))
        for cand in candidate_pool(block)[1:]:
            if len(cand.applied) > 1:
                res = verify(block, cand.block, args.trials, seed=n)
                on["multi-step candidates checked"] += 1
                if res.equivalent is False:
                    on["ILLEGAL (multi-step)"] += 1

        for rw in find_rewrites(block, guards=False)[0]:
            b2 = apply(block, rw)
            res = verify(block, b2, args.trials, seed=n)
            guarded = block_bytes(b2) not in legal_keys
            off["rewrites checked"] += 1
            off["of which a guard would refuse"] += guarded
            if res.equivalent is False:
                off["caught by harness"] += 1
                off["caught, among guard-refused"] += guarded
                rules_off_caught[rw.rule] += 1
                if len(examples) < 12:
                    examples.append(dict(mode="guards off", block=[i.text for i in block],
                                         rewrite=rw.describe(), detail=res.detail))
            elif not guarded:
                pass
            else:
                off["guard-refused but harness saw no difference"] += 1

    on.setdefault("ILLEGAL (single-step)", 0)
    on.setdefault("ILLEGAL (multi-step)", 0)
    out = dict(config=vars(args), guards_on=dict(on), guards_on_by_rule=dict(rules_on),
               guards_off=dict(off), guards_off_caught_by_rule=dict(rules_off_caught),
               examples=examples[:12], seconds=time.time() - t0)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "fuzz_guards.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "examples"}, indent=2))


if __name__ == "__main__":
    main()
