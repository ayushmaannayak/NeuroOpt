"""Optional EXTENDED rule pack (X1-X6), outside the three rules of the design.

Why it exists: on real BHive code R1-R3 almost never apply (compilers already did it), so the three rules
alone cannot show whether a learned oracle helps on natural code. These rules rewrite between instruction
forms whose real cost depends on context (ports, dependency chains, scheduling), which is exactly where a
learned oracle could beat a table. They use the SAME guard machinery (backward flag liveness) and are
checked by the SAME emulator harness; nothing here is trusted beyond that.

  X1  add r,1 <-> inc r ; sub r,1 <-> dec r           guard: CF dead (inc/dec keep CF)
  X2  add r,s|imm <-> lea r,[r+s|imm]                 guard: all six flags dead (lea writes none)
  X3  shl r,1 <-> add r,r                             guard: all six flags dead
  X4  imul r,s,{3,5,9} <-> lea r,[s+s*{2,4,8}]        guard: all six flags dead
  X5  not r ; inc r -> neg r   (and neg;dec -> not)   guard: all six flags dead after the pair
  X6  swap two adjacent independent ALU instructions  guard: no register/flag hazard (Capstone's regs_read /
                                                       regs_write, so implicit operands such as one-operand
                                                       imul on rax:rdx are covered)
"""
from __future__ import annotations

from .decode import GPR32, GPR64, GPR_FAMILY, Instr
from .emit import assemble_one
from .flags import ALL, FLAGS, effect

_GPR = set(GPR64) | set(GPR32)
_SWAP_OK = {"mov", "add", "sub", "and", "or", "xor", "adc", "sbb", "inc", "dec", "neg", "not", "lea", "imul", "shl",
            "shr", "sar", "rol", "ror", "movzx", "movsx", "movsxd", "cmp", "test", "bswap"}


def _gpr(op):
    return op.reg if op.kind == "reg" and op.reg in _GPR else None


def _fam(regs):
    return {GPR_FAMILY[r] for r in regs if r in GPR_FAMILY}


def _swap_ok_mnemonic(m: str) -> bool:
    return m in _SWAP_OK or m.startswith("cmov") or (m.startswith("set") and m != "setssbsy")


def _region_changes(instrs: list[Instr]) -> frozenset:
    out = frozenset()
    for i in instrs:
        out |= effect(i).may_change
    return out


def _single_edits(ins: Instr):
    """(rule, replacement-text list) for the one-instruction rewrites."""
    m, ops = ins.mnemonic, ins.operands
    out = []
    if len(ops) == 2 and _gpr(ops[0]):
        r = ops[0].reg
        if m in ("add", "sub") and ops[1].kind == "imm" and ops[1].imm == 1:
            out.append(("X1", [f"{'inc' if m == 'add' else 'dec'} {r}"]))
        if m == "add" and ops[1].kind == "reg" and ops[1].reg == r:
            out.append(("X3", [f"shl {r}, 1"]))
        if m == "add" and GPR_FAMILY.get(r) and ops[1].kind == "reg" and ops[1].size == ops[0].size and ops[0].size in (4, 8) \
                and ops[1].reg != r:
            out.append(("X2", [f"lea {r}, [{r} + {ops[1].reg}]"]))
        if m == "add" and ops[1].kind == "imm" and ops[0].size in (4, 8) and -2**31 <= ops[1].imm < 2**31 and ops[1].imm != 1:
            out.append(("X2", [f"lea {r}, [{r} + {ops[1].imm}]"]))
        if m == "shl" and ops[1].kind == "imm" and ops[1].imm == 1:
            out.append(("X3", [f"add {r}, {r}"]))
    if m in ("inc", "dec") and len(ops) == 1 and _gpr(ops[0]):
        r = ops[0].reg
        out.append(("X1", [f"{'add' if m == 'inc' else 'sub'} {r}, 1"]))
    if m == "lea" and len(ops) == 2 and _gpr(ops[0]) and ops[1].kind == "mem" and ops[0].size in (4, 8):
        mem, r = ops[1], ops[0].reg
        if mem.base == r and mem.index and mem.scale == 1 and not mem.disp and mem.index in _GPR and ops[0].size == 8:
            out.append(("X2", [f"add {r}, {mem.index}"]))
        if mem.base == r and not mem.index and mem.disp and ops[0].size == 8 and -2**31 <= mem.disp < 2**31:
            out.append(("X2", [f"add {r}, {mem.disp}"]))
        if mem.base and mem.base == mem.index and mem.scale in (2, 4, 8) and not mem.disp and ops[0].size == 8 and mem.base in GPR64:
            out.append(("X4", [f"imul {r}, {mem.base}, {mem.scale + 1}"]))
    if m == "imul" and len(ops) == 3 and _gpr(ops[0]) and ops[1].kind == "reg" and ops[2].kind == "imm" \
            and ops[2].imm in (3, 5, 9) and ops[0].size == 8 and ops[1].size == 8:
        out.append(("X4", [f"lea {ops[0].reg}, [{ops[1].reg} + {ops[1].reg}*{ops[2].imm - 1}]"]))
    return out


def extended_edits(block: list[Instr], live_after, guards: bool = True):
    """-> (rewrites_as_tuples, refusals_as_tuples); tuples are
    (rule, index, span, new_instrs_tuple, reason_if_refused)."""
    ok, refused = [], []
    for i, ins in enumerate(block):
        for rule, texts in _single_edits(ins):
            new = [assemble_one(t) for t in texts]
            diff = (effect(ins).may_change | _region_changes(new)) & live_after[i]
            if diff and guards:
                refused.append((rule, i, 1, tuple(new), f"{'/'.join(sorted(diff))} live after"))
            else:
                ok.append((rule, i, 1, tuple(new), ""))
        if i + 1 < len(block):
            j = block[i + 1]
            # X5
            if ins.mnemonic == "not" and j.mnemonic == "inc" and _gpr(ins.operands[0]) and ins.operands[0].reg == j.operands[0].reg:
                new = (assemble_one(f"neg {ins.operands[0].reg}"),)
                kind = "X5"
            elif ins.mnemonic == "neg" and j.mnemonic == "dec" and _gpr(ins.operands[0]) and ins.operands[0].reg == j.operands[0].reg:
                new = (assemble_one(f"not {ins.operands[0].reg}"),)
                kind = "X5"
            else:
                new = None
            if new is not None:
                diff = (_region_changes([ins, j]) | _region_changes(list(new))) & live_after[i + 1]
                (refused if diff and guards else ok).append((kind, i, 2, new, "flags live after pair" if diff else ""))
            # X6 swap
            if _swap_ok_mnemonic(ins.mnemonic) and _swap_ok_mnemonic(j.mnemonic) and not ins.has_memory_access \
                    and not j.has_memory_access and not ins.is_branch and not j.is_branch and ins.raw != j.raw:
                ra, wa = _fam(ins.regs_read), _fam(ins.regs_write)
                rb, wb = _fam(j.regs_read), _fam(j.regs_write)
                fa, fb = effect(ins), effect(j)
                reg_ok = not (wa & (rb | wb)) and not (wb & ra)
                flag_ok = not (fb.reads & fa.may_change) and not (fa.reads & fb.may_change) \
                    and not ((fa.may_change & fb.may_change) & live_after[i + 1])
                if not (fa.known and fb.known):
                    flag_ok = False
                if reg_ok and flag_ok:
                    ok.append(("X6", i, 2, (j, ins), ""))
                elif not guards:
                    ok.append(("X6", i, 2, (j, ins), ""))
    return ok, refused
