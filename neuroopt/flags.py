"""Flag semantics and backward flag liveness (the safety engine).

    live_before = reads ∪ (live_after − kills)

Conservative choices (when unsure, assume the worst):
  * all six flags are live at block exit (we cannot see the next block);
  * jumps and calls read all flags;
  * a shift/rotate by CL kills nothing (the count may be zero);
  * instructions we do not understand read all flags;
  * ``cmc`` reads CF (Capstone's metadata omits it).
"""
from __future__ import annotations

from dataclasses import dataclass

from capstone import x86 as cx

from .decode import Instr

FLAGS = ("CF", "PF", "AF", "ZF", "SF", "OF")
ALL = frozenset(FLAGS)
NONE: frozenset[str] = frozenset()
# EFLAGS bit positions, used by the emulator harness.
FLAG_BITS = {"CF": 0, "PF": 2, "AF": 4, "ZF": 6, "SF": 7, "OF": 11}


def _bits(kind: str) -> dict[str, int]:
    return {f: getattr(cx, f"X86_EFLAGS_{kind}_{f}") for f in FLAGS}


_TEST, _MODIFY, _RESET, _SET, _UNDEF = (_bits(k) for k in ("TEST", "MODIFY", "RESET", "SET", "UNDEFINED"))
_PRIOR = {f: getattr(cx, f"X86_EFLAGS_PRIOR_{f}", 0) for f in FLAGS}

# Instructions known not to touch the six arithmetic flags. Anything with no
# Capstone flag metadata that is *not* listed here is treated as unknown.
FLAG_NEUTRAL = {
    "mov", "movabs", "movzx", "movsx", "movsxd", "lea", "push", "pop", "nop", "xchg",
    "bswap", "not", "cqo", "cdq", "cdqe", "cwde", "cbw", "cwd", "movbe", "pause",
    "prefetcht0", "prefetcht1", "prefetcht2", "prefetchnta", "prefetchw", "lfence",
    "mfence", "sfence", "vzeroupper", "vzeroall", "xlatb", "endbr64", "leave", "enter",
    "rorx", "sarx", "shlx", "shrx", "pdep", "pext", "mulx", "movnti", "crc32",
    "kmovb", "kmovw", "kmovd", "kmovq",
}
# Vector instructions that DO write the arithmetic flags.
VECTOR_FLAG_WRITERS = {
    "comiss", "comisd", "ucomiss", "ucomisd", "vcomiss", "vcomisd", "vucomiss", "vucomisd",
    "ptest", "vptest", "vtestps", "vtestpd", "pcmpestri", "pcmpestrm", "pcmpistri",
    "pcmpistrm", "vpcmpestri", "vpcmpestrm", "vpcmpistri", "vpcmpistrm",
}
SHIFTS = {"shl", "sal", "shr", "sar", "rol", "ror", "rcl", "rcr", "shld", "shrd"}
_VEC_PREFIXES = ("xmm", "ymm", "zmm", "mm", "k")


@dataclass(frozen=True)
class FlagEffect:
    reads: frozenset[str]     # flags the instruction may read
    kills: frozenset[str]     # flags it definitely overwrites (incl. 'undefined')
    may_change: frozenset[str]  # flags whose value may differ afterwards
    undefined: frozenset[str]   # flags left undefined (when written)
    known: bool = True


def _uses_vector_regs(ins: Instr) -> bool:
    regs = [o.reg for o in ins.operands if o.kind == "reg"]
    return bool(regs) and any(r.startswith(_VEC_PREFIXES) for r in regs)


def effect(ins: Instr) -> FlagEffect:
    m = ins.mnemonic
    e = ins.eflags

    if ins.is_branch or m.startswith("j") or m.startswith("loop"):
        return FlagEffect(ALL, NONE, NONE, NONE)
    if m == "cmc":  # Capstone omits the CF read.
        return FlagEffect(frozenset({"CF"}), frozenset({"CF"}), frozenset({"CF"}), NONE)
    if m in ("pushf", "pushfq", "pushfw", "lahf"):
        return FlagEffect(ALL, NONE, NONE, NONE)
    if m == "sahf":
        k = frozenset({"SF", "ZF", "AF", "PF", "CF"})
        return FlagEffect(NONE, k, k, NONE)
    if m in ("popf", "popfq", "popfw"):
        return FlagEffect(NONE, ALL, ALL, NONE)

    if e == 0:
        if m in FLAG_NEUTRAL or (_uses_vector_regs(ins) and m not in VECTOR_FLAG_WRITERS):
            return FlagEffect(NONE, NONE, NONE, NONE)
        # Unknown: assume it reads every flag and overwrites none.
        return FlagEffect(ALL, NONE, NONE, NONE, known=False)

    reads = frozenset(f for f in FLAGS if e & (_TEST[f] | _PRIOR[f]))
    written = frozenset(f for f in FLAGS if e & (_MODIFY[f] | _RESET[f] | _SET[f] | _UNDEF[f]))
    undefined = frozenset(f for f in FLAGS if e & _UNDEF[f])

    if m in SHIFTS:
        count = ins.operands[-1] if ins.operands else None
        if count is not None and count.kind == "imm":
            width = ins.operands[0].size * 8
            mask = 0x3F if width == 64 else 0x1F
            if (count.imm & mask) == 0:   # shift by zero: flags untouched
                return FlagEffect(reads, NONE, NONE, NONE)
        elif count is not None and count.kind == "reg":
            # Count may be zero, so nothing is definitely overwritten.
            return FlagEffect(reads, NONE, written, undefined)

    return FlagEffect(reads, written, written, undefined)


def liveness(block: list[Instr], live_out: frozenset[str] = ALL) -> list[frozenset[str]]:
    """Return live_after[i] for every instruction (flags live right after i)."""
    live_after = [NONE] * len(block)
    live = live_out
    for i in range(len(block) - 1, -1, -1):
        live_after[i] = live
        fx = effect(block[i])
        live = fx.reads | (live - fx.kills)
    return live_after


def first_reader(block: list[Instr], start: int, flags: frozenset[str]) -> tuple[int, str] | None:
    """Find who keeps ``flags`` alive after position ``start`` (for refusal messages).
    Returns (index, flag) of the first instruction reading one of them before it is
    overwritten, or (len(block), flag) when the flag survives to block exit."""
    pending = set(flags)
    for j in range(start + 1, len(block)):
        fx = effect(block[j])
        hit = sorted(pending & fx.reads)
        if hit:
            return j, hit[0]
        pending -= fx.kills
        if not pending:
            return None
    return (len(block), sorted(pending)[0]) if pending else None


def defined_at_exit(block: list[Instr]) -> frozenset[str]:
    """Flags whose value at block exit is architecturally defined, assuming every
    flag is defined on entry. The emulator compares only these."""
    defined = set(FLAGS)
    for ins in block:
        fx = effect(ins)
        defined -= fx.undefined
        defined |= fx.kills - fx.undefined
    return frozenset(defined)
