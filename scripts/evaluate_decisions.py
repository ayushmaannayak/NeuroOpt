"""M3: decision-level evaluation (the research question).

Golden rule: hold the candidates fixed and change only the ranking.
For every block B in the decision set:
  1. build the candidate pool C(B) ∪ {B} with the guards ON;
  2. verify every candidate against B in the emulator;
  3. time every candidate on this CPU (physical timing, repeated trials);
  4. let each oracle pick with the margin rule, and score the pick by its
     *measured* time: win rate, regret vs the best candidate, geomean gain.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.decode import decode_block  # noqa: E402
from neuroopt.hwtime import HardwareTimer  # noqa: E402
from neuroopt.metrics import bootstrap_ci, geomean  # noqa: E402
from neuroopt.model import LSTMOracle  # noqa: E402
from neuroopt.oracle import MCAOracle, StaticOracle  # noqa: E402
from neuroopt.rules import candidate_pool  # noqa: E402
from neuroopt.selector import choose  # noqa: E402
from neuroopt.verify import verify  # noqa: E402

CFG = dict(tag="", set="decision_set.jsonl", extended=False)
MEAS = ROOT / "results" / "decision_measurements.jsonl"
TAUS = [0.0, 0.01, 0.02, 0.05, 0.10]
EPS = 0.02   # a measured change below 2% counts as 'no change'


def measure(args):
    recs = [json.loads(line) for line in open(ROOT / "data" / "processed" / CFG["set"])]
    if args.limit:
        recs = recs[: args.limit]
    timer = HardwareTimer(trials=args.trials, cpu=args.cpu)
    t0 = time.time()
    with open(MEAS, "w") as f:
        for n, rec in enumerate(recs):
            block = decode_block(rec["hex"])
            pool = candidate_pool(block, extended=CFG["extended"])
            cands = []
            for c in pool:
                v = verify(block, c.block, trials=16, seed=n) if c.applied else None
                cands.append(dict(hex=c.key.hex(), text=[i.text for i in c.block], applied=list(c.applied),
                                  verified=None if v is None else v.equivalent,
                                  verify_detail="" if v is None else v.detail))
            # interleave candidates across rounds so drift hits all of them equally
            rounds = defaultdict(list)
            for _ in range(args.rounds):
                order = list(range(len(pool)))
                random.shuffle(order)
                for k in order:
                    t = timer.measure(pool[k].block)
                    rounds[k].append(t.cycles)
            for k, c in enumerate(cands):
                c["cycles"] = float(np.median(rounds[k]))
                c["cycles_rounds"] = rounds[k]
            f.write(json.dumps(dict(rec, candidates=cands)) + "\n")
            if n % 100 == 0:
                print(f"{n}/{len(recs)} blocks measured ({time.time() - t0:.0f}s)", flush=True)
    meta = dict(ticks_per_cycle=timer.ticks_per_cycle, trials=args.trials, rounds=args.rounds,
                cpu=args.cpu, cpu_model=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip("\t: "))
    (ROOT / "results" / f"decision_measurement_meta{CFG['tag']}.json").write_text(json.dumps(meta, indent=2))


def score():
    """Attach every oracle's predicted cost to every measured candidate."""
    oracles = [StaticOracle(), MCAOracle("cascadelake")]
    if (ROOT / "models" / "lstm_main.pt").exists():
        oracles.insert(1, LSTMOracle(ROOT / "models" / "lstm_main.pt"))
    else:
        print("WARNING: models/lstm_main.pt missing, evaluating without the LSTM oracle")
    blocks = [json.loads(line) for line in open(MEAS)]
    for o in oracles:
        flat = [decode_block(c["hex"]) if c["hex"] else [] for b in blocks for c in b["candidates"]]
        vals = iter(o.score_many(flat))
        for b in blocks:
            for c in b["candidates"]:
                c.setdefault("scores", {})[o.name] = next(vals)
    with open(MEAS, "w") as f:
        for b in blocks:
            f.write(json.dumps(b) + "\n")


def analyse():
    blocks = [json.loads(line) for line in open(MEAS)]
    names = list(blocks[0]["candidates"][0]["scores"].keys())
    out = {"n_blocks": len(blocks), "eps": EPS}

    # Correctness: every candidate the guards produced was checked in the emulator.
    vc = Counter()
    for b in blocks:
        for c in b["candidates"][1:]:
            vc[{True: "equivalent", False: "ILLEGAL", None: "emulator could not run"}[c["verified"]]] += 1
    out["verification"] = dict(vc)
    out["pool_size"] = dict(Counter(len(b["candidates"]) for b in blocks))

    # Measurement noise: spread of repeated rounds on the original block.
    cvs = [np.std(b["candidates"][0]["cycles_rounds"]) / np.mean(b["candidates"][0]["cycles_rounds"])
           for b in blocks if np.mean(b["candidates"][0]["cycles_rounds"]) > 0]
    out["noise"] = dict(median_cv=float(np.median(cvs)), p90_cv=float(np.percentile(cvs, 90)))

    # Ground truth: the measured effect of each single rule applied to B.
    eff = defaultdict(list)
    for b in blocks:
        t0 = b["candidates"][0]["cycles"]
        for c in b["candidates"][1:]:
            if len(c["applied"]) == 1 and c["cycles"] > 0:
                eff[c["applied"][0][:2]].append(t0 / c["cycles"])
    out["single_rule_measured_speedup"] = {
        r: dict(n=len(v), geomean=geomean(v), improved=float(np.mean(np.array(v) > 1 + EPS)),
                regressed=float(np.mean(np.array(v) < 1 - EPS))) for r, v in sorted(eff.items())}

    # Why do oracles disagree? Predicted vs measured gain of each single rule.
    pv = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        c0 = b["candidates"][0]
        for c in b["candidates"][1:]:
            if len(c["applied"]) == 1 and c["cycles"] > 0:
                r = c["applied"][0][:2]
                pv[r]["measured"].append(c0["cycles"] / c["cycles"])
                for name in names:
                    if c["scores"][name] > 0 and c0["scores"][name] > 0:
                        pv[r][name].append(c0["scores"][name] / c["scores"][name])
    out["single_rule_predicted_vs_measured"] = {
        r: {k: dict(geomean=geomean(v), says_faster=float(np.mean(np.array(v) > 1 + EPS)))
            for k, v in d.items()} for r, d in sorted(pv.items())}

    has_better = [min(c["cycles"] for c in b["candidates"]) < b["candidates"][0]["cycles"] * (1 - EPS)
                  for b in blocks]
    out["blocks_where_some_candidate_is_measurably_faster"] = float(np.mean(has_better))

    def evaluate(pick):
        speed, regret, improved, regressed, changed = [], [], [], [], []
        for b in blocks:
            cs = b["candidates"]
            k = pick(b)
            t0, tk = cs[0]["cycles"], cs[k]["cycles"]
            tbest = min(c["cycles"] for c in cs)
            speed.append(t0 / tk)
            regret.append(tk / tbest - 1)
            improved.append(t0 / tk > 1 + EPS)
            regressed.append(t0 / tk < 1 - EPS)
            changed.append(k != 0)
        lo, hi = bootstrap_ci(np.log(speed), np.mean)
        return dict(geomean_speedup=geomean(speed), geomean_ci95=[float(np.exp(lo)), float(np.exp(hi))],
                    win_rate=float(np.mean(improved)), regression_rate=float(np.mean(regressed)),
                    mean_regret=float(np.mean(regret)), zero_regret=float(np.mean(np.array(regret) <= EPS)),
                    rewrite_rate=float(np.mean(changed)))

    pol = {}
    pol["keep original"] = evaluate(lambda b: 0)
    pol["apply every legal rule"] = evaluate(lambda b: max(range(len(b["candidates"])),
                                                           key=lambda k: len(b["candidates"][k]["applied"])))
    rnd = random.Random(0)
    pol["random legal candidate"] = evaluate(lambda b: rnd.randrange(len(b["candidates"])))
    for name in names:
        for tau in TAUS:
            pol[f"{name} τ={tau:g}"] = evaluate(
                lambda b, name=name, tau=tau: choose([c["scores"][name] for c in b["candidates"]], tau, relative=True))
    pol["best measured (upper bound)"] = evaluate(
        lambda b: min(range(len(b["candidates"])), key=lambda k: b["candidates"][k]["cycles"]))
    out["policies"] = pol

    # Paired comparison of oracles on the same blocks (bootstrap CI of the
    # geomean ratio of their measured speedups).
    def picks(name):
        return [b["candidates"][choose([c["scores"][name] for c in b["candidates"]], 0.0, relative=True)]["cycles"]
                for b in blocks]
    paired = {}
    for a in names:
        for b_ in names:
            if a >= b_:
                continue
            ta, tb = np.array(picks(a)), np.array(picks(b_))
            d = np.log(tb / ta)            # > 0: oracle a's pick is faster
            lo, hi = bootstrap_ci(d, np.mean)
            paired[f"{a} vs {b_}"] = dict(geomean_ratio=float(np.exp(d.mean())), ci95=[float(np.exp(lo)), float(np.exp(hi))],
                                          a_faster=float(np.mean(d > np.log(1 + EPS))),
                                          b_faster=float(np.mean(d < -np.log(1 + EPS))))
    out["paired_oracles_tau0"] = paired

    # Ranking quality inside each pool: pairwise ordering accuracy against hardware.
    rank = {}
    for name in names:
        good = tot = 0.0
        for b in blocks:
            cs = b["candidates"]
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    ti, tj = cs[i]["cycles"], cs[j]["cycles"]
                    if abs(ti - tj) <= EPS * max(ti, tj):
                        continue
                    si, sj = cs[i]["scores"][name], cs[j]["scores"][name]
                    tot += 1
                    if si == sj:
                        good += 0.5
                    elif (si < sj) == (ti < tj):
                        good += 1
        rank[name] = dict(pairs=int(tot), pairwise_accuracy=good / tot if tot else float("nan"))
    out["pairwise_ranking"] = rank

    # Breakdown by which rules were seeded.
    by_kind = defaultdict(list)
    for b in blocks:
        kinds = sorted({n[:2] for n in b["notes"]}) or ["natural"]
        by_kind["+".join(kinds)].append(b)
    out["by_seeded_rules"] = {}
    for kind, bs in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        row = {"n": len(bs)}
        for name in names + ["best"]:
            sp = []
            for b in bs:
                cs = b["candidates"]
                if name == "best":
                    k = min(range(len(cs)), key=lambda k: cs[k]["cycles"])
                else:
                    k = choose([c["scores"][name] for c in cs], 0.0, relative=True)
                sp.append(cs[0]["cycles"] / cs[k]["cycles"])
            row[name] = geomean(sp)
        out["by_seeded_rules"][kind] = row

    (ROOT / "results" / f"decision_eval{CFG['tag']}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "measure", "score", "analyse"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--decision-set", default="decision_set.jsonl", help="file under data/processed/")
    ap.add_argument("--tag", default="", help="suffix for result files, e.g. _natural")
    ap.add_argument("--extended", action="store_true", help="enable the optional X1-X6 rule pack")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cpu", type=int, default=min(3, (os.cpu_count() or 1) - 1))
    args = ap.parse_args()
    global MEAS
    CFG.update(tag=args.tag, set=args.decision_set, extended=args.extended)
    MEAS = ROOT / "results" / f"decision_measurements{args.tag}.jsonl"
    random.seed(0)
    if args.stage in ("all", "measure"):
        measure(args)
    if args.stage in ("all", "score"):
        score()
    if args.stage in ("all", "analyse"):
        analyse()


if __name__ == "__main__":
    main()
