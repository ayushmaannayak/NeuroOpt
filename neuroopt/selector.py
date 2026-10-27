"""Oracle + Select.

    B* = argmin ĉ(B') over B' in C(B) ∪ {B},  subject to  ĉ(B) − ĉ(B') > τ

C(B) only ever contains guard-approved candidates, so legality sits outside the
formula: the oracle ranks, it never permits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .decode import Instr
from .rules import Refusal, apply, candidate_pool, find_rewrites


def gain_ok(c_orig: float, c_new: float, tau: float, relative: bool) -> bool:
    gain = c_orig - c_new
    if relative:
        return c_orig > 0 and gain / c_orig > tau
    return gain > tau


def choose(costs: list[float], tau: float = 0.0, relative: bool = False) -> int:
    """Index of the chosen candidate; index 0 is the original block (keep it if
    nothing clears the margin)."""
    best = min(range(len(costs)), key=lambda k: (costs[k], k))
    return best if best != 0 and gain_ok(costs[0], costs[best], tau, relative) else 0


@dataclass
class Selection:
    original: list[Instr]
    chosen: list[Instr]
    cost_before: float
    cost_after: float
    steps: list[str] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.steps)


def select_greedy(block: list[Instr], oracle, tau: float = 0.0, relative: bool = False) -> Selection:
    """Greedy selector: repeatedly take the best single legal rewrite while its
    predicted gain clears τ. Simple and fast, not guaranteed optimal."""
    current = list(block)
    c_cur = oracle.score(current)
    c0 = c_cur
    steps: list[str] = []
    refusals: list[Refusal] = []
    seen_refusals = set()
    while True:
        rewrites, refused = find_rewrites(current)
        for r in refused:
            if (r.rule, r.before, r.reason) not in seen_refusals:
                seen_refusals.add((r.rule, r.before, r.reason))
                refusals.append(r)
        if not rewrites:
            break
        cands = [apply(current, rw) for rw in rewrites]
        costs = oracle.score_many(cands)
        k = min(range(len(cands)), key=lambda j: costs[j])
        if not gain_ok(c_cur, costs[k], tau, relative):
            break
        steps.append(rewrites[k].describe())
        current, c_cur = cands[k], costs[k]
    return Selection(block, current, c0, c_cur, steps, refusals)


def select_exhaustive(block: list[Instr], oracle, tau: float = 0.0, relative: bool = False) -> Selection:
    """Score the whole candidate pool C(B) ∪ {B} and take the formula's argmin."""
    pool = candidate_pool(block)
    costs = oracle.score_many([c.block for c in pool])
    k = choose(costs, tau, relative)
    return Selection(block, pool[k].block, costs[0], costs[k], list(pool[k].applied),
                     find_rewrites(block)[1])
