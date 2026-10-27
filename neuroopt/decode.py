"""M1 Decode: turn x86-64 machine-code bytes into exact instructions with Capstone.

Blocks that do not decode completely are rejected, so the rest of the pipeline
never works on code it does not fully understand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import capstone
from capstone import x86 as cx

_MD = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
_MD.detail = True

# 64-bit general-purpose register family for every GPR name Capstone can emit.
_GPR64 = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"] + [f"r{i}" for i in range(8, 16)]
_ALIASES = {
    "rax": ["eax", "ax", "al", "ah"], "rbx": ["ebx", "bx", "bl", "bh"],
    "rcx": ["ecx", "cx", "cl", "ch"], "rdx": ["edx", "dx", "dl", "dh"],
    "rsi": ["esi", "si", "sil"], "rdi": ["edi", "di", "dil"],
    "rbp": ["ebp", "bp", "bpl"], "rsp": ["esp", "sp", "spl"],
}
for _i in range(8, 16):
    _ALIASES[f"r{_i}"] = [f"r{_i}d", f"r{_i}w", f"r{_i}b"]
GPR_FAMILY = {}
for _full, _subs in _ALIASES.items():
    GPR_FAMILY[_full] = _full
    for _s in _subs:
        GPR_FAMILY[_s] = _full
GPR64 = tuple(_GPR64)
GPR32 = tuple(_ALIASES[r][0] for r in _GPR64)


class DecodeError(ValueError):
    """Raised for bytes that are invalid or only partially decodable."""


@dataclass(frozen=True)
class Operand:
    kind: str                 # "reg" | "imm" | "mem"
    size: int                 # bytes
    reg: str | None = None    # for kind == "reg"
    imm: int | None = None    # for kind == "imm"
    base: str | None = None   # for kind == "mem"
    index: str | None = None
    scale: int = 1
    disp: int = 0


@dataclass(frozen=True)
class Instr:
    raw: bytes
    mnemonic: str
    op_str: str
    operands: tuple[Operand, ...]
    eflags: int               # Capstone X86_EFLAGS_* bitmask
    groups: tuple[int, ...]
    regs_read: tuple[str, ...]
    regs_write: tuple[str, ...]
    prefix: tuple[int, ...] = field(default=())

    @property
    def text(self) -> str:
        return f"{self.mnemonic} {self.op_str}".strip()

    @property
    def is_branch(self) -> bool:
        g = set(self.groups)
        return bool(g & {cx.X86_GRP_JUMP, cx.X86_GRP_CALL, cx.X86_GRP_RET,
                         cx.X86_GRP_INT, cx.X86_GRP_IRET, cx.X86_GRP_BRANCH_RELATIVE})

    @property
    def has_memory_access(self) -> bool:
        """True if the instruction touches memory (lea only computes an address)."""
        if self.mnemonic == "lea" or self.mnemonic.startswith("nop"):
            return False
        if any(o.kind == "mem" for o in self.operands):
            return True
        return self.mnemonic in _IMPLICIT_MEMORY

    def __str__(self) -> str:
        return self.text


_IMPLICIT_MEMORY = {
    "push", "pop", "pushfq", "popfq", "pushf", "popf", "call", "ret", "leave", "enter",
    "movsb", "movsw", "movsd", "movsq", "stosb", "stosw", "stosd", "stosq",
    "lodsb", "lodsw", "lodsd", "lodsq", "scasb", "scasw", "scasd", "scasq",
    "cmpsb", "cmpsw", "cmpsd", "cmpsq", "xlatb",
}


def _operand(op, insn) -> Operand:
    if op.type == cx.X86_OP_REG:
        return Operand("reg", op.size, reg=insn.reg_name(op.reg))
    if op.type == cx.X86_OP_IMM:
        return Operand("imm", op.size, imm=op.imm)
    m = op.mem
    return Operand(
        "mem", op.size,
        base=insn.reg_name(m.base) if m.base else None,
        index=insn.reg_name(m.index) if m.index else None,
        scale=m.scale, disp=m.disp,
    )


def _convert(insn) -> Instr:
    rr, rw = insn.regs_access()
    return Instr(
        raw=bytes(insn.bytes),
        mnemonic=insn.mnemonic,
        op_str=insn.op_str,
        operands=tuple(_operand(o, insn) for o in insn.operands),
        eflags=insn.eflags,
        groups=tuple(insn.groups),
        regs_read=tuple(insn.reg_name(r) for r in rr),
        regs_write=tuple(insn.reg_name(r) for r in rw),
        prefix=tuple(p for p in insn.prefix if p),
    )


def decode_one(code: bytes) -> Instr:
    block = decode_block(code)
    if len(block) != 1:
        raise DecodeError(f"expected one instruction, got {len(block)}")
    return block[0]


def decode_block(code: bytes | str) -> list[Instr]:
    """Decode a whole basic block. Rejects invalid or partially decoded input,
    and blocks with control flow anywhere but the final instruction."""
    if isinstance(code, str):
        try:
            code = bytes.fromhex(code)
        except ValueError as e:
            raise DecodeError(f"bad hex: {e}") from None
    if not code:
        raise DecodeError("empty block")
    out, consumed = [], 0
    for insn in _MD.disasm(code, 0):
        out.append(_convert(insn))
        consumed += insn.size
    if consumed != len(code):
        raise DecodeError(f"partial decode: {consumed} of {len(code)} bytes understood")
    for ins in out[:-1]:
        if ins.is_branch:
            raise DecodeError(f"control flow inside block: {ins.text}")
    return out


def block_bytes(block: list[Instr]) -> bytes:
    return b"".join(i.raw for i in block)


def block_text(block: list[Instr]) -> str:
    return "\n".join(i.text for i in block)
