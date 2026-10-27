"""Re-encode instructions. Untouched instructions keep their original bytes;
only instructions produced by a rewrite rule are assembled (with Keystone)."""
from __future__ import annotations

from functools import lru_cache

import keystone

from .decode import Instr, block_bytes, decode_one

_KS = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)


@lru_cache(maxsize=4096)
def assemble_one(text: str) -> Instr:
    encoding, _ = _KS.asm(text)
    return decode_one(bytes(encoding))


def assemble(text: str) -> bytes:
    """Assemble one or more ';' / newline separated instructions."""
    encoding, _ = _KS.asm(text.replace("\n", ";"))
    return bytes(encoding)


def emit(block: list[Instr]) -> bytes:
    return block_bytes(block)
