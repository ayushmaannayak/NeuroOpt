"""Evaluation metrics: prediction (MAPE, MAE, Spearman) and decisions
(win rate, regret, geometric-mean gain)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mape(pred, y) -> float:
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    return float(np.mean(np.abs(pred - y) / y) * 100)


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def spearman(pred, y) -> float:
    return float(stats.spearmanr(pred, y).statistic)


def kendall(pred, y) -> float:
    return float(stats.kendalltau(pred, y).statistic)


def geomean(ratios) -> float:
    r = np.asarray(ratios, float)
    return float(np.exp(np.mean(np.log(r))))


def prediction_report(pred, y) -> dict:
    ok = np.isfinite(pred)
    p, t = np.asarray(pred)[ok], np.asarray(y)[ok]
    return dict(n=int(ok.sum()), mape=mape(p, t), mae=mae(p, t),
                spearman=spearman(p, t), kendall=kendall(p, t))


def bootstrap_ci(values, fn=np.mean, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    sims = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))
