"""Emit + verify: run a rewritten block beside its original in the Unicorn
emulator on random register and flag states.

Registers (16 GPRs and xmm0-15) and every written memory byte must match.
Flags are compared only where the original block leaves them defined. This is
strong evidence, not a proof: it is random testing on an emulator.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import unicorn
from unicorn import x86_const as U

from .decode import Instr, block_bytes
from .flags import FLAG_BITS, defined_at_exit

CODE_BASE = 0x1000_0000
CODE_SIZE = 0x10_0000
PAGE = 0x1000
GPRS = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"] + [f"r{i}" for i in range(8, 16)]
_GPR_IDS = [getattr(U, f"UC_X86_REG_{r.upper()}") for r in GPRS]
_XMM_IDS = [getattr(U, f"UC_X86_REG_XMM{i}") for i in range(16)]
_INTERESTING = [0, 1, 2, 3, 7, 8, 0xFF, 0xFFFF, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF,
                0x1_0000_0000, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, 0xFFFF_FFFF_FFFF_FFFF]


@dataclass
class VerifyResult:
    equivalent: bool | None   # None: emulator could not run the original
    trials: int
    detail: str = ""


def _random_state(rng: random.Random, pointer_like: bool):
    regs = []
    for _ in GPRS:
        if pointer_like:
            regs.append(rng.randrange(0x0010_0000, 0x7000_0000) & ~0x7)
        elif rng.random() < 0.3:
            regs.append(rng.choice(_INTERESTING))
        else:
            regs.append(rng.getrandbits(64))
    xmms = [rng.getrandbits(128) for _ in range(16)]
    flags = 0x2
    for bit in FLAG_BITS.values():
        if rng.random() < 0.5:
            flags |= 1 << bit
    return regs, xmms, flags


class _Machine:
    MAX_LAZY_PAGES = 64

    def __init__(self):
        self._init_uc()

    def _init_uc(self):
        self.uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.uc.mem_map(CODE_BASE, CODE_SIZE)
        self.lazy: dict[int, bytes] = {}
        self.written: set[int] = set()
        self.uc.hook_add(unicorn.UC_HOOK_MEM_UNMAPPED, self._on_unmapped)
        self.uc.hook_add(unicorn.UC_HOOK_MEM_WRITE, self._on_write)

    def _on_unmapped(self, uc, access, address, size, value, data):
        page = address & ~(PAGE - 1)
        if not (0 < page < 0x7FFF_FFFF_F000) or CODE_BASE <= page < CODE_BASE + CODE_SIZE:
            return False
        for p in {page, (address + size - 1) & ~(PAGE - 1)}:
            if p not in self.lazy:
                content = random.Random(p).randbytes(PAGE)  # deterministic per page
                uc.mem_map(p, PAGE)
                uc.mem_write(p, content)
                self.lazy[p] = content
        return True

    def _on_write(self, uc, access, address, size, value, data):
        for a in range(address, address + size):
            self.written.add(a)

    def run(self, code: bytes, state):
        regs, xmms, flags = state
        # Unmapping and re-mapping pages inside one Unicorn instance leaves stale cached translations, which made
        # the verifier report identical blocks as "different" (~2.5% of seeds on memory blocks; found by auditing
        # natural BHive code). So: restore the content of already-mapped pages IN PLACE, and rebuild the whole
        # emulator once too many pages have accumulated.
        if len(self.lazy) > self.MAX_LAZY_PAGES:
            self._init_uc()
        uc = self.uc
        for p, content in self.lazy.items():
            uc.mem_write(p, content)
        self.written.clear()
        uc.mem_write(CODE_BASE, code + b"\x90")
        for rid, v in zip(_GPR_IDS, regs):
            uc.reg_write(rid, v)
        for rid, v in zip(_XMM_IDS, xmms):
            uc.reg_write(rid, v)
        uc.reg_write(U.UC_X86_REG_EFLAGS, flags)
        if code:
            uc.emu_start(CODE_BASE, CODE_BASE + len(code), count=10_000)
        mem = {a: uc.mem_read(a, 1)[0] for a in sorted(self.written)}
        return ([uc.reg_read(r) for r in _GPR_IDS], [uc.reg_read(r) for r in _XMM_IDS],
                uc.reg_read(U.UC_X86_REG_EFLAGS), mem)


_MACHINE: _Machine | None = None


def _machine() -> _Machine:
    global _MACHINE
    if _MACHINE is None:
        _MACHINE = _Machine()
    return _MACHINE


def _compare(m, a, b, state, flag_mask, defined, t, mb=None):
    """Run both byte strings on ``state``; return a VerifyResult describing the first difference, else None."""
    try:
        ra = m.run(a, state)
    except unicorn.UcError as e:
        return VerifyResult(None, t, f"emulator cannot run original: {e}")
    try:
        rb = (mb or m).run(b, state)
    except unicorn.UcError as e:
        return VerifyResult(False, t + 1, f"rewrite faults: {e}")
    if ra[0] != rb[0]:
        k = next(i for i in range(16) if ra[0][i] != rb[0][i])
        return VerifyResult(False, t + 1, f"{GPRS[k]}: {ra[0][k]:#x} != {rb[0][k]:#x}")
    if ra[1] != rb[1]:
        return VerifyResult(False, t + 1, "xmm registers differ")
    if (ra[2] ^ rb[2]) & flag_mask:
        diff = [f for f, bit in FLAG_BITS.items() if f in defined and ((ra[2] ^ rb[2]) >> bit) & 1]
        return VerifyResult(False, t + 1, f"flags differ: {'/'.join(diff)}")
    if ra[3] != rb[3]:
        return VerifyResult(False, t + 1, "memory differs")
    return None


def verify(original: list[Instr], rewritten: list[Instr], trials: int = 16, seed: int = 0) -> VerifyResult:
    a, b = block_bytes(original), block_bytes(rewritten)
    defined = defined_at_exit(original)
    lost = sorted(defined - defined_at_exit(rewritten))
    if lost:
        return VerifyResult(False, 0, f"rewrite leaves {'/'.join(lost)} undefined")
    flag_mask = sum(1 << FLAG_BITS[f] for f in defined)
    pointer_like = any(i.has_memory_access for i in original)
    rng = random.Random(seed)
    m = _machine()
    for t in range(trials):
        state = _random_state(rng, pointer_like)
        verdict = _compare(m, a, b, state, flag_mask, defined, t)
        if isinstance(verdict, VerifyResult) and verdict.equivalent is False and "faults" not in verdict.detail:
            # Confirm with a brand-new emulator PER SIDE: a real illegal rewrite reproduces, stale emulator state does not.
            again = _compare(_Machine(), a, b, state, flag_mask, defined, t, mb=_Machine())
            if not (isinstance(again, VerifyResult) and again.equivalent is False):
                continue
            return again
        if isinstance(verdict, VerifyResult):
            return verdict
    return VerifyResult(True, trials)
