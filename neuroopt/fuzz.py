"""Random register-only blocks that are dense in rewrite opportunities and in
flag traps (adc/sbb/setcc/cmov readers, inc/dec that keep CF, cmc, shift by CL,
32-bit self-moves)."""
from __future__ import annotations

import random

from .decode import GPR32, GPR64, Instr, decode_block
from .emit import assemble

_R64 = [r for r in GPR64 if r != "rsp"]
_R32 = [r for r in GPR32 if r != "esp"]
_R8 = ["al", "bl", "cl", "dl", "sil", "dil", "r8b", "r9b", "r10b", "r11b"]
_CC = ["e", "ne", "b", "ae", "l", "ge", "s", "ns", "o", "p", "a", "be"]

TEMPLATES = [
    # rewrite opportunities
    (6, "mov {r}, 0"), (5, "imul {r}, {r}, {p2}"), (3, "mov {q}, {q}"), (2, "mov {d}, {d}"),
    # flag writers
    (4, "add {r}, {s}"), (3, "sub {r}, {s}"), (3, "xor {r}, {s}"), (2, "and {r}, {s}"),
    (2, "or {r}, {s}"), (3, "cmp {r}, {s}"), (2, "test {r}, {s}"), (1, "neg {r}"),
    (1, "bt {r}, {imm}"), (1, "popcnt {r}, {s}"),
    # partial writers / CF-preserving
    (2, "inc {r}"), (2, "dec {r}"), (1, "shl {r}, cl"), (1, "shr {r}, {imm}"), (1, "rol {r}, 1"),
    # flag readers
    (3, "adc {r}, {s}"), (2, "sbb {r}, {s}"), (3, "cmov{cc} {r}, {s}"), (3, "set{cc} {b}"),
    (1, "cmc"), (1, "stc"), (1, "clc"), (1, "lahf"),
    # flag neutral
    (2, "lea {q}, [{q2} + {q3}*4 + 8]"), (1, "not {r}"), (2, "mov {r}, {s}"), (1, "bswap {r}"),
]
_WEIGHTS = [w for w, _ in TEMPLATES]


def _instr(rng: random.Random) -> str:
    tpl = rng.choices([t for _, t in TEMPLATES], _WEIGHTS)[0]
    wide = rng.random() < 0.6
    regs = _R64 if wide else _R32
    r, s = rng.choice(regs), rng.choice(regs)
    return tpl.format(
        r=r, s=s, q=rng.choice(_R64), q2=rng.choice(_R64), q3=rng.choice([x for x in _R64 if x != "rsp"]),
        d=rng.choice(_R32), b=rng.choice(_R8), cc=rng.choice(_CC),
        p2=2 ** rng.randint(1, 30), imm=rng.randint(1, 31),
    )


def random_block(rng: random.Random, min_len: int = 2, max_len: int = 8) -> list[Instr]:
    n = rng.randint(min_len, max_len)
    return decode_block(assemble("; ".join(_instr(rng) for _ in range(n))))
