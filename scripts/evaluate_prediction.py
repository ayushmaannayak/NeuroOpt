"""Prediction-level evaluation on held-out BHive blocks (CPU-matched skl labels).

Compares, on the same blocks:
  static table  (raw sum; MAPE after one global scale fitted on train)
  llvm-mca      (-mcpu=skylake; on a fixed random sample of the test split)
  LSTM          (every trained model found in models/)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.decode import decode_block  # noqa: E402
from neuroopt.metrics import prediction_report  # noqa: E402
from neuroopt.model import Encoded, load, predict  # noqa: E402
from neuroopt.oracle import MCAOracle, StaticOracle  # noqa: E402
from neuroopt.tokenizer import block_tokens  # noqa: E402


def rows_for(split_key: str, split: str):
    with gzip.open(ROOT / "data" / "processed" / "bhive_skl.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r[split_key] == split:
                yield r


def lstm_predict(path: Path, rows):
    model, vocab, meta = load(path)
    wr = meta.get("with_registers", True)
    toks = [r["instruction_tokens"] if wr else block_tokens(decode_block(r["hex_bytes"]), False) for r in rows]
    return predict(model, Encoded(toks, vocab)), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mca-sample", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    static = StaticOracle()

    # Static table: one global scale (geometric mean ratio) fitted on train.
    train = list(rows_for("split", "train"))
    s_train = np.array([static.score(decode_block(r["hex_bytes"])) for r in train])
    y_train = np.array([r["target_y"] for r in train])
    scale = float(np.exp(np.mean(np.log(y_train) - np.log(s_train))))
    del train

    test = list(rows_for("split", "test"))
    y = np.array([r["target_y"] for r in test])
    s_raw = np.array([static.score(decode_block(r["hex_bytes"])) for r in test])
    out = {"static_scale_fitted_on_train": scale, "n_test": len(test), "full_test": {}, "mca_sample": {}}
    out["full_test"]["static (calibrated)"] = prediction_report(s_raw * scale, y)

    preds = {"static (calibrated)": s_raw * scale}
    for path in sorted((ROOT / "models").glob("lstm_*.pt")):
        meta_split = load(path)[2].get("split", "split")
        if meta_split != "split":
            continue
        p, meta = lstm_predict(path, test)
        name = f"lstm[{path.stem[5:]}]"
        preds[name] = p
        out["full_test"][name] = prediction_report(p, y)
        print(name, out["full_test"][name], flush=True)

    # Same-sample comparison including llvm-mca.
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(test), size=min(args.mca_sample, len(test)), replace=False))
    mca = MCAOracle("skylake")
    m = np.array(mca.score_many([decode_block(test[k]["hex_bytes"]) for k in idx]))
    out["mca_sample"]["n"] = int(len(idx))
    out["mca_sample"]["llvm-mca failures"] = int(np.isnan(m).sum())
    ok = np.isfinite(m)
    out["mca_sample"]["llvm-mca(skylake)"] = prediction_report(m[ok], y[idx][ok])
    for name, p in preds.items():
        out["mca_sample"][name] = prediction_report(p[idx][ok], y[idx][ok])

    # Per-category breakdown (BHive categories) for the main model.
    cats = {}
    with open(ROOT / "data" / "raw" / "categories.csv") as f:
        for r in csv.reader(f):
            if r:
                cats[r[0].lower()] = r[1]
    by = defaultdict(list)
    for k, r in enumerate(test):
        by[cats.get(r["hex_bytes"], "unknown")].append(k)
    main_name = "lstm[main]" if "lstm[main]" in preds else next((n for n in preds if n.startswith("lstm")), None)
    out["by_category"] = {}
    for c, ks in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ks = np.array(ks)
        row = {"n": int(len(ks)), "static (calibrated)": prediction_report(preds["static (calibrated)"][ks], y[ks])["mape"]}
        if main_name:
            row[main_name] = prediction_report(preds[main_name][ks], y[ks])["mape"]
        out["by_category"][c] = row

    # Held-out program families (source-disjoint split), if that model exists.
    fam_path = ROOT / "models" / "lstm_family.pt"
    if fam_path.exists():
        fam_test = list(rows_for("split_family", "test"))
        p, _ = lstm_predict(fam_path, fam_test)
        yf = np.array([r["target_y"] for r in fam_test])
        sf = np.array([static.score(decode_block(r["hex_bytes"])) for r in fam_test]) * scale
        out["heldout_families"] = {"families": "sqlite, redis, openssl", "n": len(fam_test),
                                   "lstm[family]": prediction_report(p, yf),
                                   "static (calibrated)": prediction_report(sf, yf)}

    np.savez_compressed(ROOT / "results" / "prediction_arrays.npz", y=y, mca_idx=idx, mca=m,
                        **{k.replace(" ", "_"): v for k, v in preds.items()})
    (ROOT / "results" / "prediction_eval.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
