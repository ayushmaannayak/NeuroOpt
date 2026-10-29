"""Web app tests: API behaviour, the legality traps through the HTTP layer, and the safety defaults."""
import pytest

from webapp.app import create_app


@pytest.fixture(scope="module")
def client():
    return create_app(enable_measure=False).test_client()


def post(client, **body):
    r = client.post("/api/optimize", json=body)
    return r.status_code, r.get_json()


def test_config_lists_examples_and_static_oracle(client):
    d = client.get("/api/config").get_json()
    assert "static" in [o["id"] for o in d["oracles"]]
    assert d["measure"] is False and len(d["examples"]) >= 5


def test_index_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and b"NeuroOpt" in r.data


def test_worked_example_matches_the_paper_numbers(client):
    code, d = post(client, asm="mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx", oracle="static")
    assert code == 200
    assert [s.split(" @")[0] for s in d["steps"]] == ["R2", "R3", "R1"]       # greedy: biggest saving first
    assert (d["cost_before"], d["cost_after"]) == (6.0, 2.25)
    assert round(d["reduction_pct"], 1) == 62.5
    assert [r["text"] for r in d["optimized"]] == ["xor rax, rax", "shl rbx, 3", "add rax, rbx"]
    assert d["verification"]["equivalent"] is True and d["verification"]["trials"] == 256
    assert d["original_bytes"] == 17 and d["optimized_bytes"] == 10


def test_adc_trap_is_refused_with_a_reason(client):
    code, d = post(client, asm="mov rax, 0; adc rax, rbx")
    assert code == 200 and not d["changed"]
    assert any(r["rule"] == "R1" and "CF" in r["reason"] for r in d["refused"])


def test_32bit_self_move_is_kept(client):
    _, d = post(client, asm="mov eax, eax; add rax, rbx")
    assert not d["changed"] and any(r["rule"] == "R3" and "zero-extend" in r["reason"] for r in d["refused"])


def test_flags_live_at_exit_blocks_a_standalone_zero(client):
    _, d = post(client, asm="mov rax, 0")
    assert not d["changed"] and any(r["rule"] == "R1" for r in d["refused"])


def test_every_listed_legal_candidate_is_emulator_equivalent(client):
    _, d = post(client, asm="add rax, rbx; sub rcx, 1; imul rdx, rsi, 5; add rdi, rdi; not r8; inc r8", extended=True)
    assert d["legal"] and all(c["emulator"]["equivalent"] is True for c in d["legal"])


def test_margin_tau_keeps_the_original_when_gain_is_small(client):
    _, d = post(client, asm="mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx", tau=100)
    assert not d["changed"] and d["steps"] == []


def test_flag_liveness_rows_for_the_worked_example(client):
    _, d = post(client, asm="mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx")
    live = [r["live_after"] for r in d["original"]]
    assert live[:3] == [[], [], []] and set(live[3]) == {"CF", "PF", "AF", "ZF", "SF", "OF"}


@pytest.mark.parametrize("body,msg", [
    ({}, "Enter assembly"),
    ({"asm": "frobnicate rax"}, "Could not assemble"),
    ({"hex": "zz"}, "not valid hex"),
    ({"hex": "48c7c0"}, "Rejected"),
    ({"asm": "mov rax, 0", "oracle": "nope"}, "unknown oracle"),
    ({"asm": "mov rax, 0", "tau": "abc"}, "tau must be a number"),
    ({"asm": "; ".join(["add rax, rbx"] * 60)}, "Too many instructions"),
])
def test_bad_input_gives_a_400_with_a_readable_message(client, body, msg):
    code, d = post(client, **body)
    assert code == 400 and msg in d["error"]


def test_hardware_timing_is_off_by_default(client):
    assert client.post("/api/measure", json={"asm": "add rax, rbx"}).status_code == 403


def test_hardware_timing_when_explicitly_enabled():
    c = create_app(enable_measure=True).test_client()
    r = c.post("/api/measure", json={"asm": "mov rax, 0; imul rbx, rbx, 8; mov rcx, rcx; add rax, rbx"})
    d = r.get_json()
    assert r.status_code == 200 and d["timed"] and d["before"] > 0 and d["after"] > 0
    m = c.post("/api/measure", json={"asm": "mov rax, qword ptr [rsp + 8]"}).get_json()
    assert m["timed"] is False                                                   # memory blocks are never executed
