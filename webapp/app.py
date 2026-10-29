"""NeuroOpt web app: paste an x86-64 basic block, see every pipeline stage.

Run:   python -m webapp                       (then open http://127.0.0.1:8000)

It reuses the project's own modules, so the page shows exactly what the CLI does:
decode (Capstone) -> flag liveness -> guarded rewrites and refusals -> oracle scores ->
greedy selection with margin tau -> re-encode (Keystone) -> emulator check (Unicorn).

Security: the server binds to 127.0.0.1 by default. Hardware timing executes the submitted
instructions natively on THIS machine, so /api/measure is OFF unless the server is started with
--enable-measure, and it should only ever be enabled for a local, single-user session.
"""
from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroopt.decode import DecodeError, block_bytes, decode_block  # noqa: E402
from neuroopt.emit import assemble  # noqa: E402
from neuroopt.flags import FLAGS, liveness  # noqa: E402
from neuroopt.oracle import MCAOracle, StaticOracle  # noqa: E402
from neuroopt.rules import apply, find_rewrites  # noqa: E402
from neuroopt.selector import gain_ok, select_greedy  # noqa: E402
from neuroopt.verify import verify  # noqa: E402

MAX_CODE_BYTES = 512
MAX_INSTRUCTIONS = 48
ORACLE_LABELS = {
    "static": "Static proxy table (fixed cost per opcode)",
    "lstm": "LSTM oracle (trained on BHive throughput)",
    "mca": "llvm-mca (Cascade Lake model)",
}

EXAMPLES = [
    {"title": "Worked example (Review 1)", "asm": "mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx",
     "note": "All three rules fire: the final add overwrites every flag, so the flag-changing rewrites are legal."},
    {"title": "Trap 1: adc reads the carry XOR would clear", "asm": "mov rax, 0; adc rax, rbx",
     "note": "mov rax,0 -> xor rax,rax is REFUSED: CF is live because adc reads it."},
    {"title": "Trap 2: mov eax, eax clears the upper half of RAX", "asm": "mov eax, eax; add rax, rbx",
     "note": "A 32-bit write zero-extends into RAX, so the self-move is kept; mov rcx,rcx would be deleted."},
    {"title": "Trap 3: flags are live at block exit", "asm": "mov rax, 0",
     "note": "Nothing later overwrites the flags, so MOV stays (a standalone block keeps the conservative contract)."},
    {"title": "Multiply by 8 before an overwriting add", "asm": "imul rbx, rbx, 8; add rax, rbx",
     "note": "imul -> shl is legal because the add kills every flag."},
    {"title": "Extended rules (tick the box)", "asm": "add rax, rbx; sub rcx, 1; imul rdx, rsi, 5; add rdi, rdi; not r8; inc r8",
     "note": "Optional X1-X6 pack: inc/dec, lea forms, shl 1, not;inc -> neg, instruction swaps. Same guards."},
]

_lock = threading.RLock()
_oracles: dict = {}


def create_app(enable_measure: bool = False) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["ENABLE_MEASURE"] = enable_measure
    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    def index():
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/config")
    def config():
        return jsonify(oracles=[{"id": k, "label": ORACLE_LABELS[k]} for k in available_oracles()],
                       measure=bool(app.config["ENABLE_MEASURE"]),
                       limits={"bytes": MAX_CODE_BYTES, "instructions": MAX_INSTRUCTIONS},
                       examples=EXAMPLES)

    @app.post("/api/optimize")
    def api_optimize():
        try:
            return jsonify(run_pipeline(request.get_json(silent=True) or {}))
        except UserError as e:
            return jsonify(error=str(e)), 400

    @app.post("/api/measure")
    def api_measure():
        if not app.config["ENABLE_MEASURE"]:
            return jsonify(error="Hardware timing is disabled. Start the server with --enable-measure "
                                 "(local use only: it executes the instructions on this machine)."), 403
        try:
            return jsonify(run_measure(request.get_json(silent=True) or {}))
        except UserError as e:
            return jsonify(error=str(e)), 400

    return app


class UserError(ValueError):
    pass


def available_oracles() -> list[str]:
    out = ["static"]
    if (ROOT / "models" / "lstm_main.pt").exists():
        try:
            import torch  # noqa: F401
            out.append("lstm")
        except Exception:
            pass
    if shutil.which("llvm-mca"):
        out.append("mca")
    return out


def get_oracle(name: str):
    if name not in ORACLE_LABELS:
        raise UserError(f"unknown oracle '{name}'")
    if name not in available_oracles():
        raise UserError(f"oracle '{name}' is not available on this machine")
    with _lock:
        if name not in _oracles:
            if name == "static":
                _oracles[name] = StaticOracle()
            elif name == "mca":
                _oracles[name] = MCAOracle("cascadelake")
            else:
                from neuroopt.model import LSTMOracle
                _oracles[name] = LSTMOracle(ROOT / "models" / "lstm_main.pt")
        return _oracles[name]


def parse_input(body: dict) -> bytes:
    asm, hx = (body.get("asm") or "").strip(), (body.get("hex") or "").strip()
    if not asm and not hx:
        raise UserError("Enter assembly (Intel syntax) or hex bytes.")
    if hx:
        try:
            code = bytes.fromhex("".join(hx.split()))
        except ValueError:
            raise UserError("The hex input is not valid hex bytes.")
    else:
        lines = [ln.split("#")[0].split("//")[0].strip() for ln in asm.replace("\r", "").split("\n")]
        text = "; ".join(ln.strip("; ") for ln in lines if ln.strip("; "))
        if not text:
            raise UserError("Enter at least one instruction.")
        try:
            code = assemble(text)
        except Exception as e:  # keystone raises KsError with a readable message
            raise UserError(f"Could not assemble: {e}")
    if not code:
        raise UserError("The input assembled to zero bytes.")
    if len(code) > MAX_CODE_BYTES:
        raise UserError(f"Block too large ({len(code)} bytes; limit {MAX_CODE_BYTES}).")
    return code


def parse_options(body: dict):
    oracle = str(body.get("oracle") or "static")
    try:
        tau = float(body.get("tau") or 0.0)
    except (TypeError, ValueError):
        raise UserError("tau must be a number.")
    if tau < 0 or tau > 1e6:
        raise UserError("tau must be between 0 and 1,000,000.")
    return oracle, tau, bool(body.get("relative")), bool(body.get("extended"))


def _ins_row(k, ins, live=None):
    row = {"index": k, "hex": ins.raw.hex(), "text": ins.text}
    if live is not None:
        row["live_after"] = [f for f in FLAGS if f in live]
    return row


def _verdict(v):
    if v is None:
        return None
    return {"equivalent": v.equivalent, "trials": v.trials, "detail": v.detail}


def run_pipeline(body: dict) -> dict:
    code = parse_input(body)
    oracle_name, tau, relative, extended = parse_options(body)
    try:
        block = decode_block(code)
    except DecodeError as e:
        raise UserError(f"Rejected (block left untouched): {e}")
    if len(block) > MAX_INSTRUCTIONS:
        raise UserError(f"Too many instructions ({len(block)}; limit {MAX_INSTRUCTIONS}).")
    oracle = get_oracle(oracle_name)
    with _lock:
        live = liveness(block)
        rewrites, refused = find_rewrites(block, extended=extended)
        cands = [apply(block, rw) for rw in rewrites]
        costs = oracle.score_many([block] + cands)
        c0 = costs[0]
        candidate_rows = []
        for rw, cand, c in zip(rewrites, cands, costs[1:]):
            v = verify(block, cand, trials=64)
            candidate_rows.append({
                "rule": rw.rule, "index": rw.index, "before": rw.before,
                "after": "(delete)" if rw.after is None else rw.after,
                "score": c, "gain": c0 - c, "gain_pct": (100.0 * (c0 - c) / c0) if c0 else 0.0,
                "clears_tau": gain_ok(c0, c, tau, relative), "emulator": _verdict(v)})
        sel = select_greedy(block, oracle, tau, relative, extended=extended)
        final_bytes = block_bytes(sel.chosen)
        final_v = verify(block, sel.chosen, trials=256) if sel.changed else None
        compare = {}
        for name in available_oracles():
            o = get_oracle(name)
            compare[name] = {"label": ORACLE_LABELS[name], "original": o.score(block), "optimized": o.score(sel.chosen)}
    return {
        "oracle": {"id": oracle_name, "name": oracle.name, "label": ORACLE_LABELS[oracle_name]},
        "tau": tau, "relative": relative, "extended": extended,
        "original": [_ins_row(k, i, live[k]) for k, i in enumerate(block)],
        "original_bytes": len(code), "original_hex": code.hex(),
        "legal": candidate_rows,
        "refused": [{"rule": r.rule, "index": r.index, "before": r.before, "reason": r.reason} for r in sel.refusals] or
                   [{"rule": r.rule, "index": r.index, "before": r.before, "reason": r.reason} for r in refused],
        "steps": list(sel.steps),
        "cost_before": sel.cost_before, "cost_after": sel.cost_after,
        "reduction_pct": (100.0 * (sel.cost_before - sel.cost_after) / sel.cost_before) if sel.cost_before else 0.0,
        "changed": sel.changed,
        "optimized": [_ins_row(k, i) for k, i in enumerate(sel.chosen)],
        "optimized_bytes": len(final_bytes), "optimized_hex": final_bytes.hex(),
        "verification": _verdict(final_v),
        "compare": compare,
    }


def run_measure(body: dict) -> dict:
    from neuroopt.hwtime import HardwareTimer, timeable
    code = parse_input(body)
    oracle_name, tau, relative, extended = parse_options(body)
    try:
        block = decode_block(code)
    except DecodeError as e:
        raise UserError(f"Rejected: {e}")
    if len(block) > MAX_INSTRUCTIONS:
        raise UserError("Too many instructions.")
    with _lock:
        sel = select_greedy(block, get_oracle(oracle_name), tau, relative, extended=extended)
        if not (timeable(block) and timeable(sel.chosen)):
            return {"timed": False, "reason": "The block touches memory, the stack, or a control-flow instruction, "
                                              "so it cannot be timed safely."}
        timer = HardwareTimer(trials=9)
        a, b = timer.measure(block), timer.measure(sel.chosen)
    return {"timed": True, "before": a.cycles, "after": b.cycles,
            "speedup": (a.cycles / b.cycles) if b.cycles else None, "calibration_ok": getattr(timer, "calibration_ok", None)}
