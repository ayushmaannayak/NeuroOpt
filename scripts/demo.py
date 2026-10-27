"""Live demo for the viva: the worked example, then the legality traps, then the
same block under each oracle. Output is also saved to results/demo_output.txt."""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.__main__ import optimize  # noqa: E402
from neuroopt.emit import assemble  # noqa: E402

CASES = [
    ("Worked example (slide 23)", "mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx"),
    ("Trap 1: imul -> shl changes flags; adc reads CF", "imul rbx, rbx, 8; adc rax, rbx"),
    ("Trap 2: mov -> xor clobbers ZF that sete needs", "cmp rdi, rsi; mov eax, 0; sete al"),
    ("Trap 3: mov eax, eax is a zero-extension, not a no-op", "mov eax, eax; mov rcx, rcx; add rax, rcx"),
    ("Real BHive-style block, mixed", "xor edx, edx; mov ecx, 0; imul r8, r8, 16; mov r9, r9; add r8, rcx; cmp r8, rdx"),
]


def main():
    buf = io.StringIO()

    class Tee:
        def write(self, s):
            sys.stdout.write(s)
            buf.write(s)

        def flush(self):
            sys.stdout.flush()

    out = Tee()
    oracles = ["static", "mca"]
    if (ROOT / "models" / "lstm_main.pt").exists():
        oracles.append("lstm")
    for title, asm in CASES:
        print("=" * 100, file=out)
        print(f"{title}\n    {asm}", file=out)
        print("=" * 100, file=out)
        optimize(assemble(asm), "static", 0.0, False, measure=True, out=out)
        print(file=out)
    print("=" * 100, file=out)
    print("Same worked example, swapping only the oracle", file=out)
    print("=" * 100, file=out)
    for o in oracles:
        print(f"--- oracle = {o}", file=out)
        optimize(assemble(CASES[0][1]), o, 0.0, False, measure=True, out=out)
        print(file=out)
    (ROOT / "results" / "demo_output.txt").write_text(buf.getvalue())


if __name__ == "__main__":
    main()
