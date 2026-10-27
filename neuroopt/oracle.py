"""The cost oracle: anything with ``score(block) -> float`` (lower = faster).

The oracle only ranks candidates that the guards already declared legal, so
swapping the implementation (static table -> LSTM -> llvm-mca) can change speed
but never correctness.
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from .decode import Instr, block_bytes


class Oracle(Protocol):
    name: str

    def score(self, block: list[Instr]) -> float: ...

    def score_many(self, blocks: list[list[Instr]]) -> list[float]: ...


# --------------------------------------------------------------------------- #
# Baseline: fixed per-opcode proxy table (prices each instruction alone).
# --------------------------------------------------------------------------- #
_PRICE = {}
for _m in ("imul", "mul", "popcnt", "lzcnt", "tzcnt", "bsf", "bsr", "pdep", "pext", "mulx"):
    _PRICE[_m] = 3.0
for _m in ("div", "idiv"):
    _PRICE[_m] = 25.0
for _m in ("call", "ret"):
    _PRICE[_m] = 2.0
_FP_ARITH = re.compile(r"^v?(add|sub|mul|min|max|fmadd|fmsub|fnmadd|fnmsub|cvt|round|dp|hadd|hsub)")
_FP_DIV = re.compile(r"^v?(div|sqrt|rcp|rsqrt)")
_ZERO_IDIOMS = {"xor", "sub", "pxor", "vpxor", "vpxord", "vpxorq", "xorps", "vxorps", "xorpd", "vxorpd"}
ZERO_IDIOM_PRICE = 0.25


def static_price(ins: Instr) -> float:
    m = ins.mnemonic
    regs = [o.reg for o in ins.operands if o.kind == "reg"]
    if m in _ZERO_IDIOMS and len(regs) >= 2 and len(set(regs)) == 1 and len(regs) == len(ins.operands):
        return ZERO_IDIOM_PRICE  # 'xor r, r' is recognised as a dependency-breaking zero idiom
    if m in _PRICE:
        return _PRICE[m]
    vec = any(r and r[0] in "xyz" and r[1:3] == "mm" for r in regs)
    if vec and _FP_DIV.match(m):
        return 14.0
    if vec and _FP_ARITH.match(m):
        return 4.0
    return 1.0


class StaticOracle:
    """Sum of per-instruction prices. Units are proxy 'cycles', not measurements."""
    name = "static"

    def score(self, block):
        return float(sum(static_price(i) for i in block))

    def score_many(self, blocks):
        return [self.score(b) for b in blocks]


# --------------------------------------------------------------------------- #
# Analytical baseline: llvm-mca.
# --------------------------------------------------------------------------- #
_CYCLES_RE = re.compile(r"Total Cycles:\s+(\d+)")


def to_intel_asm(block: list[Instr]) -> str:
    lines = [".intel_syntax noprefix"]
    for i in block:
        # Capstone prints e.g. 'movabs'; LLVM accepts that in Intel syntax.
        lines.append(i.text)
    return "\n".join(lines) + "\n"


class MCAOracle:
    """Predicted cycles per iteration from llvm-mca's scheduling model."""

    def __init__(self, mcpu: str = "skylake", iterations: int = 100, workers: int | None = None):
        self.mcpu = mcpu
        self.iterations = iterations
        self.workers = workers or os.cpu_count() or 4
        self.name = f"llvm-mca({mcpu})"
        self._cache: dict[bytes, float] = {}

    def _run(self, asm: str) -> float:
        with tempfile.NamedTemporaryFile("w", suffix=".s", delete=False) as f:
            f.write(asm)
            path = f.name
        try:
            out = subprocess.run(
                ["llvm-mca", f"-mcpu={self.mcpu}", f"-iterations={self.iterations}",
                 "-mtriple=x86_64-unknown-linux-gnu", "-all-views=false", "-summary-view", path],
                capture_output=True, text=True, timeout=60)
        finally:
            os.unlink(path)
        m = _CYCLES_RE.search(out.stdout)
        return int(m.group(1)) / self.iterations if m else math.nan

    def score(self, block):
        key = block_bytes(block)
        if key not in self._cache:
            self._cache[key] = self._run(to_intel_asm(block)) if block else 0.0
        return self._cache[key]

    def score_many(self, blocks):
        todo = {block_bytes(b): b for b in blocks if block_bytes(b) not in self._cache}
        with ThreadPoolExecutor(self.workers) as ex:
            for key, val in zip(todo, ex.map(lambda b: self._run(to_intel_asm(b)) if b else 0.0,
                                             todo.values())):
                self._cache[key] = val
        return [self._cache[block_bytes(b)] for b in blocks]
