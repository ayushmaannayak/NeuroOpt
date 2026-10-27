"""NeuroOpt command line.

  python -m neuroopt optimize --asm "mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx"
  python -m neuroopt optimize --hex 48c7c000000000486bdb084889c94801d8 --oracle lstm --measure
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .decode import DecodeError, block_bytes, decode_block
from .emit import assemble
from .flags import FLAGS, liveness
from .oracle import MCAOracle, StaticOracle
from .rules import find_rewrites
from .selector import select_greedy
from .verify import verify

ROOT = Path(__file__).resolve().parents[1]


def make_oracle(name: str):
    if name == "static":
        return StaticOracle()
    if name == "mca":
        return MCAOracle("cascadelake")
    if name == "lstm":
        from .model import LSTMOracle
        return LSTMOracle(ROOT / "models" / "lstm_main.pt")
    raise SystemExit(f"unknown oracle {name}")


def optimize(code: bytes, oracle_name: str, tau: float, relative: bool, measure: bool, out=sys.stdout):
    p = lambda *a: print(*a, file=out)  # noqa: E731
    try:
        block = decode_block(code)
    except DecodeError as e:
        p(f"REJECTED (left untouched): {e}")
        return None
    oracle = make_oracle(oracle_name)

    p("1. Decode (Capstone)")
    live = liveness(block)
    for k, ins in enumerate(block):
        flags = " ".join(f if f in live[k] else "." * len(f) for f in FLAGS)
        p(f"   {k:2d}  {ins.raw.hex():<22} {ins.text:<34} live after: {flags}")

    sel = select_greedy(block, oracle, tau, relative)
    p("2. Rewrite + guards")
    for r in sel.refusals:
        p(f"   refused  {r.describe()}")
    p("3. Oracle + select" + f"  [oracle={oracle.name}, τ={tau:g}{' relative' if relative else ''}]")
    for s in sel.steps:
        p(f"   accepted {s}")
    if not sel.steps:
        legal = find_rewrites(block)[0]
        p("   no candidate cleared the margin: keeping the original" if legal
          else "   no legal rewrite: keeping the original")
    p(f"   predicted cost {sel.cost_before:.2f} -> {sel.cost_after:.2f}")

    p("4. Emit + verify")
    out_bytes = block_bytes(sel.chosen)
    for ins in sel.chosen:
        p(f"       {ins.raw.hex():<22} {ins.text}")
    p(f"   bytes {len(code)} -> {len(out_bytes)}   hex {out_bytes.hex()}")
    if sel.changed:
        v = verify(block, sel.chosen, trials=256)
        p(f"   emulator: {'EQUIVALENT' if v.equivalent else 'MISMATCH ' + v.detail} over {v.trials} random states")

    if measure:
        from .hwtime import HardwareTimer, timeable
        if timeable(block) and timeable(sel.chosen):
            t = HardwareTimer(trials=9)
            a, b = t.measure(block), t.measure(sel.chosen)
            p(f"5. Hardware timing (this CPU): {a.cycles:.2f} -> {b.cycles:.2f} cycles/iteration "
              f"(x{a.cycles / b.cycles if b.cycles else float('inf'):.2f})")
        else:
            p("5. Hardware timing skipped: block touches memory or the stack")
    return sel


def main(argv=None):
    ap = argparse.ArgumentParser(prog="neuroopt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("optimize", help="optimize one basic block")
    g = o.add_mutually_exclusive_group(required=True)
    g.add_argument("--hex")
    g.add_argument("--asm", help="Intel syntax, ';' separated")
    o.add_argument("--oracle", default="static", choices=["static", "lstm", "mca"])
    o.add_argument("--tau", type=float, default=0.0)
    o.add_argument("--relative", action="store_true", help="τ is a fraction of the original cost")
    o.add_argument("--measure", action="store_true", help="time before/after on this CPU")
    a = ap.parse_args(argv)
    code = bytes.fromhex(a.hex) if a.hex else assemble(a.asm)
    optimize(code, a.oracle, a.tau, a.relative, a.measure)


if __name__ == "__main__":
    main()
