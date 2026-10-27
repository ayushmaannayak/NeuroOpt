"""M2 Tokenize: instructions -> token ids (+ lengths and masks) for the model.

Each instruction becomes: mnemonic, then per operand either the register name,
<IMM>, or <MEM> base index </MEM>. This is deliberately lossy (immediate values
and displacements are dropped), so it is never used to decide legality.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .decode import Instr

PAD, UNK, IMM, MEM, MEM_END = "<PAD>", "<UNK>", "<IMM>", "<MEM>", "</MEM>"
SPECIAL = [PAD, UNK, IMM, MEM, MEM_END]
MAX_TOKENS = 10   # per instruction (mnemonic + up to 3 operands with memory parts)


def instr_tokens(ins: Instr, with_registers: bool = True) -> list[str]:
    out = [ins.mnemonic]
    for op in ins.operands:
        if op.kind == "reg":
            out.append(op.reg if with_registers else f"<REG{op.size * 8}>")
        elif op.kind == "imm":
            out.append(IMM)
        else:
            out.append(MEM)
            if with_registers:
                if op.base:
                    out.append(op.base)
                if op.index:
                    out.append(op.index)
            out.append(MEM_END)
    return out[:MAX_TOKENS]


def block_tokens(block: list[Instr], with_registers: bool = True) -> list[list[str]]:
    return [instr_tokens(i, with_registers) for i in block]


class Vocab:
    def __init__(self, tokens: list[str]):
        self.itos = list(tokens)
        self.stoi = {t: k for k, t in enumerate(self.itos)}

    @classmethod
    def build(cls, token_lists, min_count: int = 2) -> "Vocab":
        c = Counter(t for block in token_lists for ins in block for t in ins)
        return cls(SPECIAL + sorted(t for t, n in c.items() if n >= min_count and t not in SPECIAL))

    def __len__(self):
        return len(self.itos)

    def encode(self, block_toks: list[list[str]]) -> list[list[int]]:
        unk = self.stoi[UNK]
        return [[self.stoi.get(t, unk) for t in ins] for ins in block_toks]

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


def pad_batch(encoded: list[list[list[int]]]):
    """Pack a batch into (ids[B, I, T], n_tokens[B, I], n_instr[B], mask[B, I])."""
    import torch
    B = len(encoded)
    I = max(len(b) for b in encoded)
    T = max(len(ins) for b in encoded for ins in b)
    ids = torch.zeros(B, I, T, dtype=torch.long)
    ntok = torch.zeros(B, I, dtype=torch.long)
    for bi, b in enumerate(encoded):
        for ii, ins in enumerate(b):
            ids[bi, ii, :len(ins)] = torch.tensor(ins)
            ntok[bi, ii] = len(ins)
    ninstr = torch.tensor([len(b) for b in encoded])
    mask = ntok > 0
    return ids, ntok, ninstr, mask
