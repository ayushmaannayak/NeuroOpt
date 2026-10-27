"""Second, more precise timing pass over every candidate of a measurement file (test-retest).

Why: with n candidates per block and per-measurement noise sigma, 'best measured' is biased upward (winner's curse):
picking the minimum of noisy timings always looks like a gain. A second independent pass lets us (a) quantify how
reproducible the timings are, and (b) score a policy's/oracle's pick on timings that did NOT take part in choosing it.
"""
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuroopt.decode import decode_block  # noqa: E402
from neuroopt.hwtime import HardwareTimer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_natural")
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--cpu", type=int, default=0)
    a = ap.parse_args()
    src = ROOT / "results" / f"decision_measurements{a.tag}.jsonl"
    dst = ROOT / "results" / f"decision_measurements{a.tag}_retest.jsonl"
    timer = HardwareTimer(trials=a.trials, cpu=a.cpu)
    random.seed(1); t0 = time.time()
    with open(dst, "w") as f:
        for n, line in enumerate(open(src)):
            b = json.loads(line)
            blocks = [decode_block(c["hex"]) for c in b["candidates"]]
            rounds = defaultdict(list)
            for _ in range(a.rounds):
                order = list(range(len(blocks))); random.shuffle(order)
                for k in order:
                    rounds[k].append(timer.measure(blocks[k]).cycles)
            for k, c in enumerate(b["candidates"]):
                c["cycles_pass1"] = c["cycles"]; c["cycles_rounds_pass1"] = c["cycles_rounds"]
                c["cycles"] = float(np.median(rounds[k])); c["cycles_rounds"] = rounds[k]
            f.write(json.dumps(b) + "\n")
            if n % 100 == 0:
                print(f"{n} blocks re-timed ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
