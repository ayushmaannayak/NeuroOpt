"""Prototype test suite: decode, tokenize, guards, liveness, selection, verification."""
import random

import pytest

from neuroopt.decode import DecodeError, block_bytes, decode_block
from neuroopt.emit import assemble
from neuroopt.flags import ALL, defined_at_exit, effect, liveness
from neuroopt.fuzz import random_block
from neuroopt.oracle import StaticOracle
from neuroopt.rules import apply, candidate_pool, find_rewrites
from neuroopt.selector import choose, select_exhaustive, select_greedy
from neuroopt.tokenizer import Vocab, block_tokens
from neuroopt.verify import verify

WORKED_EXAMPLE = "48c7c000000000486bdb084889c94801d8"   # mov rax,0; imul rbx,rbx,8; mov rcx,rcx; add rax,rbx


def blk(text):
    return decode_block(assemble(text))


# --- M1 decode ------------------------------------------------------------- #
def test_decode_worked_example():
    b = decode_block(WORKED_EXAMPLE)
    assert [i.text for i in b] == ["mov rax, 0", "imul rbx, rbx, 8", "mov rcx, rcx", "add rax, rbx"]


def test_decode_rejects_partial_and_invalid():
    with pytest.raises(DecodeError):
        decode_block("4801")          # truncated add
    with pytest.raises(DecodeError):
        decode_block("zz")
    with pytest.raises(DecodeError):
        decode_block(assemble("jmp 0x10; add rax, rbx"))   # control flow inside the block


# --- M2 tokenize ----------------------------------------------------------- #
def test_tokenizer_is_lossy_but_consistent():
    a, b = blk("mov rax, 5"), blk("mov rax, 7")
    assert block_tokens(a) == block_tokens(b) == [["mov", "rax", "<IMM>"]]
    m = block_tokens(blk("mov rax, qword ptr [rbx + rcx*8 + 16]"))[0]
    assert m == ["mov", "rax", "<MEM>", "rbx", "rcx", "</MEM>"]
    v = Vocab.build([block_tokens(a)], min_count=1)
    assert v.encode([["mov", "never-seen"]])[0][1] == v.stoi["<UNK>"]


# --- flags / liveness ------------------------------------------------------ #
def test_liveness_backward_rule():
    b = blk("mov rax, 0; adc rbx, rcx; add rdx, rdx")
    live = liveness(b)
    assert "CF" in live[0]           # adc reads CF
    assert live[1] == frozenset()    # add overwrites every flag, so nothing is live after adc
    assert liveness(blk("add rax, rbx"))[-1] == ALL


def test_conservative_flag_effects():
    assert effect(blk("shl rax, cl")[0]).kills == frozenset()      # count may be zero
    assert effect(blk("cmc")[0]).reads == {"CF"}                    # Capstone omits this
    assert effect(blk("inc rax")[0]).kills == ALL - {"CF"}          # inc keeps CF
    assert effect(blk("pushfq")[0]).reads == ALL


# --- guards ---------------------------------------------------------------- #
def test_r1_refused_when_cf_read_by_adc():
    ok, refused = find_rewrites(blk("mov rax, 0; adc rbx, rcx; add rax, rbx"))
    assert not [r for r in ok if r.rule == "R1"]
    assert "CF read by adc" in refused[0].reason


def test_r2_refused_when_flags_reach_setcc():
    _, refused = find_rewrites(blk("imul rbx, rbx, 8; sete al; add rax, rax"))
    assert refused and refused[0].rule == "R2" and "ZF" in refused[0].reason


def test_r3_64bit_only():
    ok, refused = find_rewrites(blk("mov eax, eax; mov rcx, rcx"))
    assert [r.before for r in ok] == ["mov rcx, rcx"]
    assert "zero-extends" in refused[0].reason


def test_flags_live_at_block_exit_block_rewrite():
    ok, refused = find_rewrites(blk("mov rax, 0"))
    assert not ok and "block exit" in refused[0].reason


# --- selection ------------------------------------------------------------- #
def test_worked_example_static_proxy():
    s = select_greedy(decode_block(WORKED_EXAMPLE), StaticOracle())
    assert [i.text for i in s.chosen] == ["xor rax, rax", "shl rbx, 3", "add rax, rbx"]
    assert (s.cost_before, s.cost_after) == (6.0, 2.25)


def test_margin_keeps_original():
    assert choose([1.00, 0.99], tau=0.05, relative=True) == 0
    assert choose([1.00, 0.90], tau=0.05, relative=True) == 1
    assert choose([2.0, 3.0]) == 0                      # original wins: never forced to change


def test_exhaustive_matches_greedy_on_example():
    b = decode_block(WORKED_EXAMPLE)
    assert block_bytes(select_exhaustive(b, StaticOracle()).chosen) == block_bytes(select_greedy(b, StaticOracle()).chosen)
    assert len(candidate_pool(b)) == 8                  # 3 independent sites -> 2^3 candidates


def test_oracle_swap_does_not_change_candidates():
    class Reverse:
        name = "reverse"
        def score(self, b): return -StaticOracle().score(b)
        def score_many(self, bs): return [self.score(b) for b in bs]
    b = decode_block(WORKED_EXAMPLE)
    s = select_greedy(b, Reverse())
    assert not s.changed                                # a bad oracle can only lose speed


# --- verification ---------------------------------------------------------- #
def test_verify_worked_example_equivalent():
    b = decode_block(WORKED_EXAMPLE)
    assert verify(b, select_greedy(b, StaticOracle()).chosen, trials=64).equivalent


def test_verify_catches_unguarded_rewrites():
    b = blk("mov rax, 0; adc rbx, rcx")
    bad = find_rewrites(b, guards=False)[0][0]
    assert verify(b, apply(b, bad)).equivalent is False
    b = blk("mov eax, eax")
    assert verify(b, apply(b, find_rewrites(b, guards=False)[0][0])).equivalent is False


def test_verify_compares_only_defined_flags():
    assert defined_at_exit(blk("imul rbx, rbx, 4")) == {"CF", "SF", "OF"}
    a, b = blk("xor eax, eax"), blk("mov eax, 0; and eax, eax")   # AF undefined in both
    assert defined_at_exit(a) == ALL - {"AF"}
    assert verify(a, b, trials=64).equivalent
    # a rewrite that leaves a defined flag undefined is rejected outright
    assert verify(blk("add rax, rbx; add rcx, rdx"), blk("add rax, rbx; imul rcx, rcx, 1")).equivalent is False


def test_verify_memory_blocks():
    b = blk("mov rcx, rcx; mov qword ptr [rdi + 8], rax; add rax, qword ptr [rsi]")
    rw = find_rewrites(b)[0]
    assert rw and verify(b, apply(b, rw[0])).equivalent


def test_fuzz_guards_on_never_illegal():
    rng = random.Random(1234)
    checked = 0
    for n in range(150):
        b = random_block(rng)
        for cand in candidate_pool(b)[1:]:
            assert verify(b, cand.block, trials=16, seed=n).equivalent, (b, cand.applied)
            checked += 1
    assert checked > 50


# --- hardware timing harness ------------------------------------------------ #
@pytest.mark.skipif(not __import__("platform").machine() in ("x86_64", "AMD64"), reason="x86-64 only")
def test_hardware_timer_recovers_known_latencies():
    from neuroopt.hwtime import HardwareTimer
    t = HardwareTimer(trials=5)
    imul = t.measure_bytes(assemble("imul rax, rax, 8")).cycles     # 3-cycle latency chain
    shl = t.measure_bytes(assemble("shl rax, 3")).cycles            # 1-cycle latency chain
    assert 2.5 < imul < 3.5 and 0.7 < shl < 1.3


# ---------------------------------------------------------------- additions: calibration + extended rule pack
def test_timer_calibration_is_validated_and_correct():
    from neuroopt.hwtime import HardwareTimer
    t = HardwareTimer(trials=5)
    assert t.calibration_ok, f"imul chain measured {t.calibration_check_imul_latency:.2f} cycles"


def test_extended_pool_is_equivalent_in_emulator():
    from neuroopt.rules import candidate_pool
    b = decode_block(assemble("add rax, rbx; sub rcx, 1; imul rdx, rsi, 5; add rdi, rdi; not r8; inc r8; xor r9, r9; add r9, r10"))
    pool = candidate_pool(b, extended=True, limit=40)
    assert len(pool) > 10
    assert all(verify(b, c.block, trials=16).equivalent for c in pool[1:])


def test_extended_guards_refuse_when_flags_are_live():
    from neuroopt.rules import find_rewrites
    b = decode_block(assemble("add rax, rbx; adc rcx, rdx"))          # adc reads the CF that add produced
    ok, ref = find_rewrites(b, extended=True)
    assert not any(r.rule == "X2" and r.index == 0 for r in ok)
    assert any(r.rule == "X2" and r.index == 0 for r in ref)


def test_swap_respects_implicit_operands():
    from neuroopt.rules import find_rewrites
    b = decode_block(assemble("mov eax, edi; imul r9d"))               # one-operand imul implicitly uses eax/edx
    assert not any(r.rule == "X6" for r in find_rewrites(b, extended=True)[0])


def test_verifier_catches_extended_rules_with_guards_off():
    from neuroopt.rules import candidate_pool
    bad = 0
    for text in ["add rax, rbx; adc rcx, rdx", "mov eax, edi; imul r9d", "sub rax, 1; adc rbx, rcx", "add rax, rbx; cmovc rcx, rdx"]:
        b = decode_block(assemble(text))
        for c in candidate_pool(b, guards=False, extended=True, limit=20)[1:]:
            bad += not verify(b, c.block, trials=24).equivalent
    assert bad > 0, "emulator failed to expose any unguarded illegal rewrite"


def test_verifier_has_no_false_illegals_on_memory_blocks():
    # Regression: the emulator used to report a block as different from an identical copy of itself (~1-2% of seeds
    # on blocks that read and write the same memory), because state leaked between runs.
    a = decode_block(assemble("mov eax, dword ptr [rsp + 0x28]; add qword ptr [rsp + 0x28], 1; add eax, 1; cmp eax, esi"))
    b = decode_block(assemble("mov eax, dword ptr [rsp + 0x28]; add qword ptr [rsp + 0x28], 1; inc eax; cmp eax, esi"))
    c = decode_block(assemble("mov eax, dword ptr [rsp + 0x28]; add qword ptr [rsp + 0x28], 1; add eax, 2; cmp eax, esi"))
    assert all(verify(a, a, trials=16, seed=s).equivalent for s in range(300))
    assert all(verify(a, b, trials=16, seed=s).equivalent for s in range(300))
    assert not any(verify(a, c, trials=16, seed=s).equivalent for s in range(30))      # real differences still caught
