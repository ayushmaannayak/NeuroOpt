"""Physical timing on the host CPU.

Only register-only blocks are timed (no memory access, no stack-pointer writes,
no traps), because we do not replicate BHive's page-mapping machinery.
The raw clock is the TSC; results are converted to core cycles using a measured
1-cycle dependency chain (``add rax, 1``).
"""
from __future__ import annotations

import ctypes
import os
import statistics
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .decode import GPR_FAMILY, Instr, block_bytes, decode_block
from .emit import assemble

_HERE = Path(__file__).parent
_SO = _HERE / "_hwtime.so"

_DENY = {
    "div", "idiv", "push", "pop", "call", "ret", "syscall", "sysenter", "int", "int3", "hlt",
    "cpuid", "rdtsc", "rdtscp", "rdpmc", "rdrand", "rdseed", "xgetbv", "xsetbv", "in", "out",
    "cli", "sti", "ud2", "enter", "leave", "pushfq", "popfq", "pushf", "popf", "lfence",
    "mfence", "sfence", "pause", "xbegin", "xend", "xabort", "xtest", "vzeroall", "vzeroupper",
    "ldmxcsr", "stmxcsr", "fxsave", "fxrstor", "xsave", "xrstor", "rdfsbase", "rdgsbase",
    "wrfsbase", "wrgsbase", "rdpid", "lock", "monitor", "mwait", "clflush", "clflushopt",
}
_SEGMENT = {"cs", "ds", "es", "fs", "gs", "ss"}


def timeable(block: list[Instr]) -> bool:
    for ins in block:
        m = ins.mnemonic
        if m in _DENY or m.startswith(("j", "loop", "rep", "f")) or ins.is_branch:
            return False
        if ins.has_memory_access:
            return False
        if any(r in ("rsp", "esp", "sp", "spl") for r in ins.regs_write):
            return False
        regs = set(ins.regs_read) | set(ins.regs_write)
        if regs & _SEGMENT or any(r.startswith(("cr", "dr", "st(")) for r in regs):
            return False
    return True


_INIT_REGS = ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp"] + [f"r{i}" for i in range(8, 16)]
_PROLOGUE = assemble(
    "push rbx; push rbp; push r12; push r13; push r14; push r15; vzeroall; "
    + "; ".join(f"mov {r}, {0x1000 + 0x111 * k}" for k, r in enumerate(_INIT_REGS)))
_EPILOGUE = assemble("vzeroupper; pop r15; pop r14; pop r13; pop r12; pop rbp; pop rbx; ret")


def _build_lib():
    if not _SO.exists() or _SO.stat().st_mtime < (_HERE / "hwtime.c").stat().st_mtime:
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", str(_SO), str(_HERE / "hwtime.c")],
                       check=True)
    lib = ctypes.CDLL(str(_SO))
    lib.nt_load.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.nt_time.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint64)]
    if lib.nt_init() != 0:
        raise RuntimeError("could not map an executable page")
    return lib


@dataclass
class Timing:
    cycles: float             # median core cycles / iteration across trials
    trials: list[float]
    fault: int = 0            # signal number if the block faulted

    @property
    def cv(self) -> float:
        if len(self.trials) < 2 or self.cycles <= 0:
            return 0.0
        return statistics.pstdev(self.trials) / self.cycles


class HardwareTimer:
    """Times a block in a hot loop:

        loop:  body x K ; dec ctr ; jnz loop        (M iterations)

    measured with K and 2K copies and the same M. The loop overhead and the call
    overhead are identical in both, so (t(2K) - t(K)) / (M*K) is the block's own
    steady-state throughput. K is kept small so the loop runs from the uop cache:
    a 100x/200x straight-line unroll (as in BHive) is front-end bound for short
    blocks and can even invert the ranking of two candidates.
    """

    def __init__(self, body_bytes: int = 256, executions: int = 2000, reps: int = 20, trials: int = 5,
                 cpu: int | None = None):
        self.lib = _build_lib()
        self.body_bytes, self.executions, self.reps, self.trials = body_bytes, executions, reps, trials
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        self.ticks_per_cycle = 1.0
        # Calibrate on a dependent REGISTER add chain (1 cycle/iteration on every x86 core). A chain of
        # `add rax, 1` is NOT safe: Golden Cove / Sapphire Rapids-class cores fold dependent add-immediate
        # chains in the renamer (~9 per clock here), which made every reported "cycle" ~6x too large.
        calib = self.measure_bytes(assemble("add rax, rbx"))
        self.ticks_per_cycle = calib.cycles
        # Self-check: a dependent imul chain must come out at its architectural latency (3 cycles).
        chk = self.measure_bytes(assemble("imul rax, rax, 8")).cycles
        self.calibration_check_imul_latency = chk
        self.calibration_ok = 2.5 <= chk <= 3.5
        if not self.calibration_ok:
            import warnings
            warnings.warn(f"timer calibration looks wrong: dependent imul chain measured {chk:.2f} cycles (expected ~3). "
                          "Cycle numbers from this machine are unreliable (noisy VM, frequency scaling, or unusual core).")

    @staticmethod
    def _counter(block: list[Instr]) -> str | None:
        used = {GPR_FAMILY.get(r, r) for ins in block for r in ins.regs_read + ins.regs_write}
        for r in ("r15", "r14", "r13", "r12", "r11", "r10", "r9", "r8", "rbp", "rsi", "rdi", "rbx",
                  "rdx", "rcx", "rax"):
            if r not in used:
                return r
        return None

    def _code(self, body: bytes, k: int, m: int, ctr: str) -> bytes:
        head = _PROLOGUE + assemble(f"mov {ctr}, {m}")
        head += b"\x90" * ((-len(head)) % 32)
        loop = body * k + assemble(f"dec {ctr}")
        rel = -(len(loop) + 6)
        loop += b"\x0f\x85" + struct.pack("<i", rel)
        return head + loop + _EPILOGUE

    def _min_ticks(self, code: bytes) -> tuple[int, int]:
        self.lib.nt_load(code, len(code))
        out = ctypes.c_uint64()
        sig = self.lib.nt_time(3, self.reps, ctypes.byref(out))
        return sig, out.value

    def measure_bytes(self, body: bytes) -> Timing:
        if not body:
            return Timing(0.0, [0.0] * self.trials)
        ctr = self._counter(decode_block(body))
        if ctr is None:
            return Timing(float("nan"), [], fault=-1)
        k = max(1, min(16, self.body_bytes // len(body)))
        m = max(1, self.executions // k)
        c1, c2 = self._code(body, k, m, ctr), self._code(body, 2 * k, m, ctr)
        vals = []
        for _ in range(self.trials):
            s1, t1 = self._min_ticks(c1)
            s2, t2 = self._min_ticks(c2)
            if s1 or s2:
                return Timing(float("nan"), [], fault=s1 or s2)
            vals.append(max(t2 - t1, 0) / (m * k) / self.ticks_per_cycle)
        return Timing(statistics.median(vals), vals)

    def measure(self, block: list[Instr]) -> Timing:
        return self.measure_bytes(block_bytes(block))
