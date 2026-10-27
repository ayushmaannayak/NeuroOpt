"""Figures + RESULTS.md from the JSON results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, INK2, GRID, SURFACE = "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
ORACLE_COLOR = {"static": BLUE, "lstm": ORANGE, "llvm-mca(cascadelake)": AQUA}
ORACLE_LABEL = {"static": "Static table", "lstm": "LSTM", "llvm-mca(cascadelake)": "llvm-mca"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)


def fig_prediction(pred):
    arr = np.load(RES / "prediction_arrays.npz")
    y = arr["y"]
    idx, mca = arr["mca_idx"], arr["mca"]
    ok = np.isfinite(mca)
    panels = [("Static table (calibrated)", arr["static_(calibrated)"][idx][ok], y[idx][ok]),
              ("llvm-mca (skylake)", mca[ok], y[idx][ok]),
              ("LSTM (main)", arr["lstm[main]"][idx][ok], y[idx][ok])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    s = pred["mca_sample"]
    keys = ["static (calibrated)", "llvm-mca(skylake)", "lstm[main]"]
    for ax, (title, p, t), k in zip(axes, panels, keys):
        ax.hexbin(t, p, gridsize=45, xscale="log", yscale="log", bins="log", cmap="Blues", mincnt=1,
                  extent=(np.log10(0.1), np.log10(60), np.log10(0.1), np.log10(60)))
        ax.plot([0.1, 60], [0.1, 60], color=INK2, lw=1, ls="--")
        ax.set_title(title)
        ax.text(0.04, 0.96, f"MAPE {s[k]['mape']:.1f}%\nSpearman {s[k]['spearman']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
        ax.set_xlabel("measured (BHive, cycles/iter)")
    axes[0].set_ylabel("predicted (cycles/iter)")
    fig.suptitle(f"Prediction on the same {s['n'] - s['llvm-mca failures']} held-out BHive blocks",
                 x=0.01, ha="left", fontweight="bold")
    save(fig, "prediction_scatter.png")


def fig_inversion(pred, dec):
    s = pred["mca_sample"]
    pol = dec["policies"]
    rows = [("Static table", s["static (calibrated)"], pol["static τ=0"], BLUE),
            ("llvm-mca", s["llvm-mca(skylake)"], pol["llvm-mca(cascadelake) τ=0"], AQUA),
            ("LSTM", s["lstm[main]"], pol["lstm τ=0"], ORANGE)]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ys = np.arange(len(rows))[::-1]
    for y, (label, p, d, color) in zip(ys, rows):
        axes[0].barh(y, p["mape"], color=color, height=0.6)
        axes[0].text(p["mape"] + 0.8, y, f"{p['mape']:.1f}%", va="center", fontsize=9)
        axes[1].barh(y, p["spearman"], color=color, height=0.6)
        axes[1].text(p["spearman"] + 0.004, y, f"{p['spearman']:.3f}", va="center", fontsize=9)
        g = (d["geomean_speedup"] - 1) * 100
        lo, hi = [(v - 1) * 100 for v in d["geomean_ci95"]]
        axes[2].barh(y, g, color=color, height=0.6)
        axes[2].errorbar(g, y, xerr=[[g - lo], [hi - g]], color=INK, lw=1, capsize=3)
        axes[2].text(hi + 0.9, y, f"{g:+.1f}%", va="center", fontsize=9)
    best = (pol["best measured (upper bound)"]["geomean_speedup"] - 1) * 100
    axes[2].axvline(best, color=INK2, lw=1, ls="--")
    axes[2].text(best - 0.5, -0.62, f"best possible {best:+.1f}%", ha="right", va="center", fontsize=8.5, color=INK2)
    for ax in axes:
        ax.set_ylim(-0.85, len(rows) - 0.5)
    axes[0].set_title("Prediction error (MAPE, lower is better)")
    axes[1].set_title("Prediction ranking (Spearman)")
    axes[2].set_title("Decision quality (measured speedup)")
    axes[0].set_xlim(0, 55)
    axes[1].set_xlim(0.75, 1.0)
    axes[2].set_xlim(0, 50)
    for ax in axes:
        ax.set_yticks(ys, [r[0] for r in rows])
        ax.grid(axis="y", visible=False)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_xlabel("% error on held-out BHive blocks")
    axes[1].set_xlabel("rank correlation with BHive labels")
    axes[2].set_xlabel("geomean speedup of chosen rewrites on this CPU (%)")
    save(fig, "prediction_vs_decision.png")


def fig_training():
    fig, ax = plt.subplots(figsize=(7, 4))
    runs = [("main", "full data", ORANGE), ("frac30", "30% of data", BLUE), ("frac10", "10% of data", AQUA),
            ("noreg", "no register names", VIOLET)]
    for tag, label, color in runs:
        m = load(f"m2_train_{tag}.json")
        if not m:
            continue
        h = m["history"]
        x = [e["epoch"] for e in h]
        v = [e["val_mape"] for e in h]
        ax.plot(x, v, color=color, lw=2)
        ax.annotate(f"{label} ({min(v):.1f}%)", (x[-1], v[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MAPE (%)")
    ax.set_title("LSTM training: validation error per epoch")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.35)
    save(fig, "training_curves.png")


def fig_decisions(dec):
    pol = dec["policies"]
    rows = [("keep original", "Keep original", GRAY), ("random legal candidate", "Random legal", GRAY),
            ("apply every legal rule", "Apply every rule", GRAY),
            ("static τ=0", "Static table", BLUE), ("llvm-mca(cascadelake) τ=0", "llvm-mca", AQUA),
            ("lstm τ=0", "LSTM", ORANGE), ("best measured (upper bound)", "Best possible", INK2)]
    rows = [r for r in rows if r[0] in pol]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    ys = np.arange(len(rows))[::-1]
    for y, (k, label, color) in zip(ys, rows):
        g = pol[k]["geomean_speedup"]
        lo, hi = pol[k]["geomean_ci95"]
        ax.barh(y, (g - 1) * 100, color=color, height=0.6)
        ax.errorbar((g - 1) * 100, y, xerr=[[(g - lo) * 100], [(hi - g) * 100]], color=INK, lw=1, capsize=3)
        ax.text(max((hi - 1) * 100, 0) + 0.6, y, f"{(g - 1) * 100:+.1f}%", va="center", fontsize=9)
    ax.set_yticks(ys, [r[1] for r in rows])
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    ax.set_xlabel("geometric-mean measured speedup vs original (%)")
    ax.set_title("Measured throughput gain of each policy's choices")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    for y, (k, label, _) in zip(ys, rows):
        w, r = pol[k]["win_rate"] * 100, pol[k]["regression_rate"] * 100
        n = 100 - w - r
        ax.barh(y, w, color=BLUE, height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.barh(y, n, left=w, color=GRID, height=0.6, edgecolor=SURFACE, linewidth=2)
        ax.barh(y, r, left=w + n, color=ORANGE, height=0.6, edgecolor=SURFACE, linewidth=2)
        if w >= 6:
            ax.text(w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=8.5, color="white")
        if r >= 4:
            ax.text(w + n + r / 2, y, f"{r:.0f}%", ha="center", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [r[1] for r in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of blocks")
    ax.set_title(f"Outcome per block (|change| > {dec['eps'] * 100:.0f}% counts)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="faster"), Patch(color=GRID, label="no change"),
                       Patch(color=ORANGE, label="slower")], loc="lower right", frameon=False, fontsize=9,
              bbox_to_anchor=(1.0, -0.32), ncol=3)
    save(fig, "decision_policies.png")


def fig_tau(dec):
    pol = dec["policies"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, color in ORACLE_COLOR.items():
        taus = sorted(float(k.split("τ=")[1]) for k in pol if k.startswith(name + " τ="))
        if not taus:
            continue
        g = [(pol[f"{name} τ={t:g}"]["geomean_speedup"] - 1) * 100 for t in taus]
        rg = [pol[f"{name} τ={t:g}"]["regression_rate"] * 100 for t in taus]
        x = [t * 100 for t in taus]
        for ax, v in zip(axes, (g, rg)):
            ax.plot(x, v, color=color, lw=2, marker="o", ms=5)
            ax.annotate(ORACLE_LABEL[name], (x[-1], v[-1]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
    axes[0].set_title("Geometric-mean speedup vs margin τ")
    axes[0].set_ylabel("speedup vs original (%)")
    axes[1].set_title("Blocks made slower vs margin τ")
    axes[1].set_ylabel("% of blocks regressed")
    for ax in axes:
        ax.set_xlabel("τ (required predicted gain, % of original cost)")
        ax.set_xlim(right=ax.get_xlim()[1] * 1.25)
    save(fig, "tau_sweep.png")


def fig_rules(dec):
    eff = dec["single_rule_measured_speedup"]
    blocks = [json.loads(line) for line in open(RES / "decision_measurements.jsonl")]
    data = {"R1": [], "R2": [], "R3": []}
    for b in blocks:
        t0 = b["candidates"][0]["cycles"]
        for c in b["candidates"][1:]:
            if len(c["applied"]) == 1 and c["cycles"] > 0:
                data[c["applied"][0][:2]].append((t0 / c["cycles"] - 1) * 100)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    names = {"R1": "R1  mov r,0 → xor r,r", "R2": "R2  imul r,r,2^k → shl r,k", "R3": "R3  delete mov r64,r64"}
    rng = np.random.default_rng(0)
    cap = 150
    for y, (r, v) in enumerate(data.items()):
        v = np.clip(np.array(v), -40, cap)
        ax.scatter(v, y + rng.uniform(-0.25, 0.25, len(v)), s=10, color=[BLUE, ORANGE, AQUA][y], alpha=0.45,
                   linewidths=0)
    ax.set_yticks(range(3), [f"{names[r]}\nn={eff[r]['n']}, geomean {(eff[r]['geomean'] - 1) * 100:+.1f}%"
                             for r in data])
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlim(-42, cap + 5)
    ax.set_xlabel(f"measured speedup of the single rewrite on this CPU (%, values above {cap}% drawn at {cap}%)")
    ax.set_title("Ground truth: what each rule really does to throughput")
    ax.grid(axis="y", visible=False)
    save(fig, "rule_effects.png")


def pct(x):
    return f"{x * 100:.1f}%"


def write_md(data, pred, dec, fuzz, train, hwmeta):
    L = []
    w = L.append
    w("# NeuroOpt — results\n")
    w("All numbers below are produced by the scripts in this repository and saved as JSON in `results/`.\n")
    if pred and dec:
        s_, pol = pred["mca_sample"], dec["policies"]
        pr = dec["paired_oracles_tau0"]["lstm vs static"]
        w("## Answer to the research question\n")
        w("> *Does a learned block-throughput model select better legal rewrites than a fixed-cost table?*\n")
        w(f"**No, not in this setting.** The LSTM is by far the best *predictor* "
          f"(MAPE {s_['lstm[main]']['mape']:.1f}% vs {s_['static (calibrated)']['mape']:.1f}% static, "
          f"{s_['llvm-mca(skylake)']['mape']:.1f}% llvm-mca), yet it makes the *worst* decisions: its picks give "
          f"{(pol['lstm τ=0']['geomean_speedup'] - 1) * 100:+.1f}% measured geomean speedup vs "
          f"{(pol['static τ=0']['geomean_speedup'] - 1) * 100:+.1f}% for the static table "
          f"(upper bound {(pol['best measured (upper bound)']['geomean_speedup'] - 1) * 100:+.1f}%). "
          f"On the same blocks, the LSTM's choices are {(1 - pr['geomean_ratio']) * 100:.1f}% slower than the "
          f"static table's (95% CI {(1 - pr['ci95'][1]) * 100:.1f} to {(1 - pr['ci95'][0]) * 100:.1f}%). "
          "Ordered by rank correlation (Spearman: LSTM > llvm-mca > static), the three oracles come out in exactly "
          "the reverse order at the decision level (static > llvm-mca > LSTM). Low prediction error did **not** "
          "translate into better optimization choices. This is the link the project set out to test, and the "
          "hypothesis is not supported for these three rules on this CPU.\n")
        w("Why: the decision only needs the *sign* and rough size of each rewrite's gain. All three rules "
          "are almost never harmful on this CPU, so a table that always says \"faster\" is nearly optimal. "
          "The LSTM under-predicts the gains of R2 and R3 and predicts R1 as a slowdown, so it often keeps the "
          "slower original (see the diagnosis table in section 4).\n")
        w("![headline](figures/prediction_vs_decision.png)\n")

    w("## 1. Correctness (guards + emulator harness)\n")
    on, off = fuzz["guards_on"], fuzz["guards_off"]
    w(f"Random register-only blocks dense in rewrite sites and flag traps: **{on['blocks']:,} blocks**, "
      f"{fuzz['config']['trials']} random register/flag states per check.\n")
    w("| Experiment | Result |\n|---|---|")
    w(f"| Guards ON — single-step rewrites | **{on['ILLEGAL (single-step)']} illegal of {on['single-step rewrites checked']:,}** |")
    w(f"| Guards ON — multi-step candidates | **{on['ILLEGAL (multi-step)']} illegal of {on['multi-step candidates checked']:,}** |")
    w(f"| Guards OFF — rewrites a guard would refuse | {off['of which a guard would refuse']:,} |")
    w(f"| Guards OFF — of those, caught by the harness | **{off['caught, among guard-refused']:,}** "
      f"({off['caught, among guard-refused'] / off['of which a guard would refuse'] * 100:.1f}%) |")
    w(f"| Guards OFF — guard-approved rewrites flagged by harness (false alarms) | "
      f"{off['caught by harness'] - off['caught, among guard-refused']} |")
    if dec:
        v = dec["verification"]
        w(f"| Decision set — every candidate in every pool, emulator check | "
          f"{v.get('equivalent', 0):,} equivalent, {v.get('ILLEGAL', 0)} illegal, "
          f"{v.get('emulator could not run', 0)} not runnable in the emulator |")
    w("\nThe guards-off row is what makes the guards-on row meaningful: the harness *can* catch bad "
      "rewrites. The uncaught guard-refused rewrites are conservative refusals (e.g. flags assumed live at "
      "block exit) where no difference is observable. This is strong evidence, **not a proof**.\n")

    w("## 2. Data (M1)\n")
    w(f"BHive Skylake throughput labels: {data['total']:,} blocks → **{data['kept']:,} kept** "
      f"({data['total'] - data['kept']} dropped: >64 instructions or label outside (0, 100] cycles; "
      f"0 decode failures). {data['token groups']:,} distinct token sequences; identical-token blocks always "
      f"share a split.\n")
    sr = data["split (random group)"]
    w(f"* Random group split: train {sr['train']:,} / val {sr['val']:,} / test {sr['test']:,}")
    fam_key = next(k for k in data if k.startswith("split (held-out families"))
    sf = data[fam_key]
    w(f"* Program-family split ({fam_key[26:-1]} held out): train {sf['train']:,} / val {sf['val']:,} / "
      f"test {sf['test']:,} ({sf.get('excluded', 0):,} blocks shared with held-out programs excluded)")
    w(f"* Register-only (timeable on this machine): {data['timeable (register-only)']:,}\n")

    w("## 3. Prediction (M2)\n")
    if pred:
        s = pred["mca_sample"]
        w(f"Same {s['n'] - s['llvm-mca failures']} held-out test blocks for all three "
          f"({s['llvm-mca failures']} blocks llvm-mca could not parse were dropped from the comparison):\n")
        w("| Model | MAPE | MAE (cycles) | Spearman | Kendall |\n|---|---|---|---|---|")
        for k in ["static (calibrated)", "llvm-mca(skylake)", "lstm[main]"]:
            r = s[k]
            w(f"| {k} | {r['mape']:.1f}% | {r['mae']:.3f} | {r['spearman']:.3f} | {r['kendall']:.3f} |")
        w(f"\nStatic table MAPE uses one global scale fitted on train (×{pred['static_scale_fitted_on_train']:.3f}); "
          "its ranking metrics are scale-free.\n")
        w(f"Full test split ({pred['n_test']:,} blocks):\n")
        w("| Model | MAPE | Spearman |\n|---|---|---|")
        for k, r in pred["full_test"].items():
            w(f"| {k} | {r['mape']:.1f}% | {r['spearman']:.3f} |")
        if "heldout_families" in pred:
            h = pred["heldout_families"]
            w(f"\nHeld-out program families ({h['families']}, {h['n']:,} blocks never seen in training): "
              f"LSTM MAPE **{h['lstm[family]']['mape']:.1f}%** (Spearman {h['lstm[family]']['spearman']:.3f}); "
              f"static table {h['static (calibrated)']['mape']:.1f}%.\n")
        w("\nAblations: `main` = full data with register names; `noreg` = tokens without register names; "
          "`frac30`/`frac10` = 30% / 10% of the training set.\n")
        w("![prediction](figures/prediction_scatter.png)\n![training](figures/training_curves.png)\n")

    w("## 4. Decisions (M3) — the research question\n")
    if dec:
        w(f"**{dec['n_blocks']:,} blocks**, every candidate timed on this CPU ({hwmeta['cpu_model']}, "
          f"median of {hwmeta['rounds']} interleaved rounds × {hwmeta['trials']} trials; median run-to-run "
          f"CV {pct(dec['noise']['median_cv'])}). Candidate pools are identical for every oracle; only the "
          f"ranking changes. A change counts only if larger than {dec['eps'] * 100:.0f}%. "
          f"A measurably faster candidate exists in {pct(dec['blocks_where_some_candidate_is_measurably_faster'])} of blocks.\n")
        w("| Policy | Geomean speedup (95% CI) | Faster | Slower | Mean regret | Picks best |\n|---|---|---|---|---|---|")
        for k, r in dec["policies"].items():
            if "τ=" in k and not k.endswith("τ=0"):
                continue
            lo, hi = r["geomean_ci95"]
            w(f"| {k} | {(r['geomean_speedup'] - 1) * 100:+.2f}% ({(lo - 1) * 100:+.2f}, {(hi - 1) * 100:+.2f}) | "
              f"{pct(r['win_rate'])} | {pct(r['regression_rate'])} | {pct(r['mean_regret'])} | {pct(r['zero_regret'])} |")
        w("\nPaired comparison on the same blocks (τ=0): geomean ratio of measured time of B's pick to A's pick "
          "(> 1 means A picks faster code):\n")
        w("| A vs B | Ratio (95% CI) | A's pick faster | B's pick faster |\n|---|---|---|---|")
        for k, r in dec["paired_oracles_tau0"].items():
            w(f"| {k} | {r['geomean_ratio']:.3f} ({r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}) | {pct(r['a_faster'])} | {pct(r['b_faster'])} |")
        w("\nDiagnosis: predicted vs measured gain of each single rewrite (geomean of original/rewritten; "
          "'says faster' = share of cases where the oracle predicts more than 2% gain):\n")
        w("| Rule | Measured | Static table | llvm-mca | LSTM |\n|---|---|---|---|---|")
        for r, d in dec["single_rule_predicted_vs_measured"].items():
            cells = [f"{(d[k]['geomean'] - 1) * 100:+.1f}% ({pct(d[k]['says_faster'])} faster)"
                     for k in ("measured", "static", "llvm-mca(cascadelake)", "lstm")]
            w(f"| {r} | " + " | ".join(cells) + " |")
        w("\nPairwise ranking accuracy inside pools (pairs whose measured times differ by more than "
          f"{dec['eps'] * 100:.0f}%):\n")
        w("| Oracle | Pairs | Correctly ordered |\n|---|---|---|")
        for k, r in dec["pairwise_ranking"].items():
            w(f"| {k} | {r['pairs']:,} | {pct(r['pairwise_accuracy'])} |")
        w("\nMeasured effect of each single rule (ground truth):\n")
        w("| Rule | n | Geomean speedup | Faster | Slower |\n|---|---|---|---|---|")
        for k, r in dec["single_rule_measured_speedup"].items():
            w(f"| {k} | {r['n']} | {(r['geomean'] - 1) * 100:+.2f}% | {pct(r['improved'])} | {pct(r['regressed'])} |")
        w("\nτ sweep (relative margin):\n")
        w("| Oracle | " + " | ".join(f"τ={t}" for t in ["0", "0.01", "0.02", "0.05", "0.1"]) + " |")
        w("|---|" + "---|" * 5)
        for name in ORACLE_LABEL:
            cells = []
            for t in ["0", "0.01", "0.02", "0.05", "0.1"]:
                r = dec["policies"].get(f"{name} τ={t}")
                cells.append(f"{(r['geomean_speedup'] - 1) * 100:+.2f}% / {pct(r['regression_rate'])} slower" if r else "-")
            w(f"| {name} | " + " | ".join(cells) + " |")
        w("\n![policies](figures/decision_policies.png)\n![tau](figures/tau_sweep.png)\n![rules](figures/rule_effects.png)\n")
    (RES / "RESULTS.md").write_text("\n".join(L) + "\n")


def main():
    data = load("m1_data_summary_skl.json")
    pred = load("prediction_eval.json")
    dec = load("decision_eval.json")
    fuzz = load("fuzz_guards.json")
    hw = load("decision_measurement_meta.json")
    train = load("m2_train_main.json")
    if pred:
        fig_prediction(pred)
    if pred and dec:
        fig_inversion(pred, dec)
    fig_training()
    if dec:
        fig_decisions(dec)
        fig_tau(dec)
        fig_rules(dec)
    write_md(data, pred, dec, fuzz, train, hw)
    print("wrote", RES / "RESULTS.md", "and", FIG)


if __name__ == "__main__":
    sys.exit(main())
