"""M4 Rewrite: three guarded peephole rules on exact operands.

  R1  mov r, 0        ->  xor r, r     guard: all six flags dead afterwards
  R2  imul r, r, 2^k  ->  shl r, k     guard: all six flags dead afterwards
  R3  mov r64, r64    ->  (delete)     guard: 64-bit only (32-bit writes zero-extend)

Legality is decided here and only here. The cost oracle never sees a candidate
that failed a guard.
"""
from __future__ import annotations

from dataclasses import dataclass

from .decode import GPR32, GPR64, Instr, block_bytes
from .emit import assemble_one
from .flags import effect, first_reader, liveness


@dataclass(frozen=True)
class Rewrite:
    rule: str
    index: int
    before: str
    after: str | None          # None means the instruction is deleted
    new_instr: Instr | None
    span: int = 1              # instructions replaced (extended rules may replace 2)
    new_instrs: tuple | None = None   # explicit replacement tuple (extended rules); overrides new_instr

    def describe(self) -> str:
        tgt = "(delete)" if self.after is None else self.after
        return f"{self.rule} @{self.index}: {self.before}  ->  {tgt}"


@dataclass(frozen=True)
class Refusal:
    rule: str
    index: int
    before: str
    reason: str

    def describe(self) -> str:
        return f"{self.rule} refused @{self.index}: {self.before}  ({self.reason})"


def _gpr(op) -> str | None:
    if op.kind == "reg" and (op.reg in GPR64 or op.reg in GPR32):
        return op.reg
    return None


def _match_r1(ins: Instr):
    if ins.mnemonic == "mov" and len(ins.operands) == 2:
        dst, src = ins.operands
        r = _gpr(dst)
        if r and src.kind == "imm" and src.imm == 0:
            return f"xor {r}, {r}"
    return None


def _match_r2(ins: Instr):
    if ins.mnemonic == "imul" and len(ins.operands) == 3:
        dst, src, imm = ins.operands
        r = _gpr(dst)
        if r and src.kind == "reg" and src.reg == r and imm.kind == "imm":
            v = imm.imm
            width = dst.size * 8
            if v > 1 and (v & (v - 1)) == 0 and v.bit_length() - 1 < width:
                return f"shl {r}, {v.bit_length() - 1}"
    return None


def _match_r3(ins: Instr):
    """Returns (is_64bit, reg) for a register self-copy."""
    if ins.mnemonic == "mov" and len(ins.operands) == 2:
        dst, src = ins.operands
        if dst.kind == "reg" and src.kind == "reg" and dst.reg == src.reg:
            return dst.reg in GPR64, dst.reg
    return None


def _flag_reason(block, i, changed, live_after):
    hit = changed & live_after[i]
    if not hit:
        return None
    who = first_reader(block, i, hit)
    if who is None:
        return f"{'/'.join(sorted(hit))} live after"
    j, flag = who
    if j == len(block):
        return f"{flag} live at block exit"
    return f"{flag} read by {block[j].mnemonic} @{j} before being overwritten"


def find_rewrites(block: list[Instr], guards: bool = True, extended: bool = False) -> tuple[list[Rewrite], list[Refusal]]:
    """Every single-step rewrite of ``block``. With ``guards=False`` the flag and
    width guards are skipped (used only to show the verifier catches bad rewrites).
    ``extended=True`` additionally enables the optional X1-X6 pack (``rules_ext``)."""
    live_after = liveness(block)
    ok: list[Rewrite] = []
    refused: list[Refusal] = []
    for i, ins in enumerate(block):
        for rule, matcher in (("R1", _match_r1), ("R2", _match_r2)):
            target = matcher(ins)
            if target is None:
                continue
            new = assemble_one(target)
            reason = _flag_reason(block, i, effect(new).may_change, live_after)
            if reason and guards:
                refused.append(Refusal(rule, i, ins.text, reason))
            else:
                ok.append(Rewrite(rule, i, ins.text, new.text, new))
        m3 = _match_r3(ins)
        if m3 is not None:
            is64, reg = m3
            if is64 or not guards:
                ok.append(Rewrite("R3", i, ins.text, None, None))
            else:
                why = ("32-bit write zero-extends into the upper half, not a no-op"
                       if reg in GPR32 else "only 32/64-bit GPRs are in scope")
                refused.append(Refusal("R3", i, ins.text, why))
    if extended:
        from .rules_ext import extended_edits
        eok, eref = extended_edits(block, live_after, guards)
        for rule, i, span, new, _ in eok:
            after = "; ".join(x.text for x in new) if new else None
            before = "; ".join(x.text for x in block[i:i + span])
            ok.append(Rewrite(rule, i, before, after, new[0] if len(new) == 1 else None, span, tuple(new)))
        for rule, i, span, new, why in eref:
            refused.append(Refusal(rule, i, "; ".join(x.text for x in block[i:i + span]), why))
    return ok, refused


def apply(block: list[Instr], rw: Rewrite) -> list[Instr]:
    out = list(block)
    if rw.new_instrs is not None:
        out[rw.index:rw.index + rw.span] = list(rw.new_instrs)
        return out
    if rw.new_instr is None:
        del out[rw.index]
    else:
        out[rw.index] = rw.new_instr
    return out


@dataclass
class Candidate:
    block: list[Instr]
    applied: tuple[str, ...]   # human-readable rewrite descriptions, in order

    @property
    def key(self) -> bytes:
        return block_bytes(self.block)


def candidate_pool(block: list[Instr], guards: bool = True, limit: int = 64, extended: bool = False) -> list[Candidate]:
    """C(B) ∪ {B}: every block reachable by repeatedly applying legal rewrites.
    Each step is individually legal on an already-equivalent block, so liveness
    is recomputed after each step. The original block is always candidate 0."""
    pool = [Candidate(list(block), ())]
    seen = {pool[0].key}
    frontier = [pool[0]]
    while frontier and len(pool) < limit:
        nxt = []
        for cand in frontier:
            for rw in find_rewrites(cand.block, guards, extended)[0]:
                b = apply(cand.block, rw)
                c = Candidate(b, cand.applied + (rw.describe(),))
                if c.key not in seen and len(pool) < limit:
                    seen.add(c.key)
                    pool.append(c)
                    nxt.append(c)
        frontier = nxt
    return pool
