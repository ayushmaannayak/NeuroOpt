"""Guard audit on NATURAL code: every single-step rewrite the rules propose on real BHive blocks is checked in the
emulator, with the guards ON (must be 0 illegal) and OFF (shows what the guards prevent). Includes memory-touching
blocks (the emulator maps memory lazily), which the register-only timing set cannot cover."""
import argparse, collections, gzip, json, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neuroopt.decode import decode_block  # noqa: E402
from neuroopt.rules import apply, find_rewrites  # noqa: E402
from neuroopt.verify import verify  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=5000)
    ap.add_argument("--trials", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in gzip.open(ROOT / "data/processed/bhive_skl.jsonl.gz", "rt")]
    random.Random(0).shuffle(rows)
    rows = rows[: a.blocks]
    out = {"blocks": len(rows)}
    for mode in ("guards on", "guards off"):
        guards = mode == "guards on"
        per = collections.defaultdict(lambda: [0, 0, 0])      # candidates, illegal, emulator-could-not-run
        for n, r in enumerate(rows):
            b = decode_block(r["hex_bytes"])
            ok, _ = find_rewrites(b, guards=guards, extended=True)
            for rw in ok:
                v = verify(b, apply(b, rw), trials=a.trials, seed=n)
                p = per[rw.rule]; p[0] += 1
                if v.equivalent is False: p[1] += 1
                elif v.equivalent is None: p[2] += 1
        out[mode] = dict(candidates=sum(v[0] for v in per.values()), illegal=sum(v[1] for v in per.values()),
                         emulator_could_not_run=sum(v[2] for v in per.values()),
                         by_rule={k: dict(n=v[0], illegal=v[1]) for k, v in sorted(per.items())})
        print(mode, out[mode]["candidates"], "illegal", out[mode]["illegal"], flush=True)
    (ROOT / "results" / "audit_natural.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
