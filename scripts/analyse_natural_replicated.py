"""Winner's-curse-free analysis of the natural-code decision experiment (uses the two timing passes)."""
import argparse, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
EPS = 0.02


def boot(x, n=3000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.exp(x.mean())), [float(np.exp(np.percentile(m, 2.5))), float(np.exp(np.percentile(m, 97.5)))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_natural"); a = ap.parse_args()
    B = [json.loads(l) for l in open(ROOT / "results" / f"decision_measurements{a.tag}_retest.jsonl")]
    names = list(B[0]["candidates"][0]["scores"])
    t1 = [np.array([c["cycles_pass1"] for c in b["candidates"]]) for b in B]
    t2 = [np.array([c["cycles"] for c in b["candidates"]]) for b in B]
    S = {n: [np.array([c["scores"][n] for c in b["candidates"]]) for b in B] for n in names}

    # ---- reproducibility of the timings themselves
    r1 = np.concatenate([np.log(a_[0] / a_[1:]) for a_ in t1]); r2 = np.concatenate([np.log(b_[0] / b_[1:]) for b_ in t2])
    from scipy.stats import spearmanr
    sig1 = np.abs(r1) > EPS
    rep = float(np.mean((np.sign(r2[sig1]) == np.sign(r1[sig1])) & (np.abs(r2[sig1]) > EPS))) if sig1.any() else float("nan")
    out = {"n_blocks": len(B), "n_candidates": int(len(r1)),
           "timing_reproducibility": dict(spearman_pass1_vs_pass2=float(spearmanr(r1, r2)[0]),
                                          sd_of_pass_difference_pct=float(np.std(r1 - r2) * 100),
                                          replication_rate_of_pass1_gt2pct_effects=rep,
                                          median_cv_pass2=float(np.median([np.std(b["candidates"][0]["cycles_rounds"]) / np.mean(b["candidates"][0]["cycles_rounds"]) for b in B])))}

    def pick_score(sc, tau):
        j = int(np.argmin(sc)); return j if (sc[0] - sc[j]) / max(sc[0], 1e-9) > tau else 0

    policies = {"keep original": lambda i: 0}
    for n in names:
        for tau in (0.0, 0.02, 0.05):
            policies[f"{n} tau={tau:g}"] = (lambda i, n=n, tau=tau: pick_score(S[n][i], tau))
    policies["oracle, selected on pass 1 (replicated, unbiased)"] = lambda i: int(np.argmin(t1[i]))
    res = {}
    for name, pol in policies.items():
        ch = [pol(i) for i in range(len(B))]
        lg2 = np.array([np.log(t2[i][0] / t2[i][j]) for i, j in enumerate(ch)])
        lg1 = np.array([np.log(t1[i][0] / t1[i][j]) for i, j in enumerate(ch)])
        g, ci = boot(lg2)
        res[name] = dict(geomean_speedup_pass2=g, ci95=ci, geomean_speedup_pass1=float(np.exp(lg1.mean())),
                         changed=int(sum(j != 0 for j in ch)), wins=int((lg2 > EPS).sum()), losses=int((lg2 < -EPS).sum()),
                         worst=float(np.exp(lg2.min())))
    lg = np.array([np.log(t2[i][0] / t2[i]).mean() for i in range(len(B))])      # random legal candidate (incl. original)
    res["random legal candidate (expected)"] = dict(geomean_speedup_pass2=boot(lg)[0], ci95=boot(lg)[1])
    res["oracle, selected AND scored on pass 2 (biased: winner's curse)"] = dict(
        geomean_speedup_pass2=float(np.exp(np.mean([np.log(t[0] / t.min()) for t in t2]))))
    out["policies"] = res

    # ---- paired: LSTM vs static / mca (same blocks, pass 2)
    def lg_of(pol):
        return np.array([np.log(t2[i][0] / t2[i][pol(i)]) for i in range(len(B))])
    paired = {}
    for other in names:
        if other == "lstm":
            continue
        d = lg_of(policies["lstm tau=0"]) - lg_of(policies[f"{other} tau=0"])
        g, ci = boot(d); paired[f"lstm(tau=0) vs {other}(tau=0)"] = dict(ratio=g, ci95=ci)
    out["paired"] = paired

    # ---- can any oracle order candidates?  only pairs whose difference REPLICATES in both passes
    pair = {}
    for n in names:
        ok = tot = ties = 0
        for i in range(len(B)):
            k = len(t1[i])
            for x in range(k):
                for y in range(x + 1, k):
                    d1, d2 = np.log(t1[i][x] / t1[i][y]), np.log(t2[i][x] / t2[i][y])
                    if abs(d1) > 0.03 and abs(d2) > 0.03 and np.sign(d1) == np.sign(d2):
                        tot += 1; s = S[n][i][x] - S[n][i][y]
                        if s == 0: ties += 1
                        elif np.sign(s) == np.sign(d2): ok += 1
        pair[n] = dict(replicated_pairs=tot, accuracy_excluding_ties=ok / max(tot - ties, 1), tie_fraction=ties / max(tot, 1))
    out["pairwise_on_replicated_pairs"] = pair
    (ROOT / "results" / f"natural_replicated{a.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
