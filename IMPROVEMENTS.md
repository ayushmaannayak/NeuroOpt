# Improvements (v2) - what changed, what was found, what is still limited

Everything below was run. Numbers come from `results/` and can be regenerated with `scripts/run_natural.sh`.
Nothing in the original design was removed; the default behaviour (three rules R1-R3, the deck's pipeline) is unchanged.

## 1. Bugs found and fixed

**a) Hardware timer calibration broke on newer Intel cores** (`neuroopt/hwtime.py`).
Cycles were calibrated from a dependent `add rax, 1` chain, assumed to take 1 cycle. Golden Cove / Sapphire
Rapids-class cores fold dependent add-immediate chains in the renamer, so on such a CPU the chain ran at 0.106 TSC
ticks/iteration instead of ~0.64, every reported "cycle" came out ~6x too large (a dependent `imul` chain read 17.8
cycles instead of 3) and `test_hardware_timer_recovers_known_latencies` failed. Calibration now uses a dependent
register chain (`add rax, rbx`) and is **self-checked** against an `imul` chain (must read 2.5-3.5 cycles, else a
warning is raised and `timer.calibration_ok` is False). Your published results were measured on Cascade Lake and are
unaffected; the fix matters if you demo on a 12th-gen-or-newer Intel machine.

**b) The emulator verifier could report a block as different from an identical copy of itself** (`neuroopt/verify.py`).
Found by auditing natural BHive code: comparing a memory-touching block with itself returned "ILLEGAL" for ~1.5-2.5%
of seeds (state leaked from one emulator run into the next). Fix: restore lazily-mapped pages in place, rebuild the
emulator when too many pages accumulate, and **confirm every mismatch on a brand-new emulator per side** before
calling it illegal. Check: 0 false alarms in 1,500 seeds (identical block, and an equivalent rewrite), while a
genuinely different rewrite is still caught 300/300. Effect on earlier results: the false alarms could only ADD
"illegal" verdicts. Re-running `scripts/fuzz_guards.py --blocks 3000` reproduces your numbers **exactly** (0 illegal
with guards on; 1,672 of 1,864 guard-refused rewrites caught with guards off), so nothing published changes.

## 2. Additions

* `neuroopt/rules_ext.py` - **optional** extended rule pack X1-X6 (`candidate_pool(..., extended=True)`), outside the
  deck's three rules and off by default: add/sub 1 <-> inc/dec, add <-> lea, shl 1 <-> add, imul 3/5/9 <-> lea,
  not;inc -> neg, and swapping adjacent independent instructions (uses Capstone's implicit register sets, so e.g.
  one-operand `imul` is never moved past `mov eax, ...`). Same flag-liveness guards, same emulator check.
* `scripts/build_natural_decision_set.py`, `evaluate_decisions.py --decision-set/--tag/--extended` - a decision
  benchmark made of **natural** held-out test blocks (no seeding, no inserted instructions).
* `scripts/retest_natural.py`, `analyse_natural_replicated.py` - test-retest timing and a winner's-curse-free analysis.
* `scripts/audit_natural.py` - guard audit on natural code, including memory-touching blocks.
* Tests: 19 -> 25 (calibration self-check; extended pool equivalent in the emulator; guards refuse when flags are live;
  swap respects implicit operands; verifier exposes unguarded extended rewrites; verifier has no false illegals).

* `webapp/` - a Flask web app over the same pipeline (see README, "Web app"); 18 API tests and a headless-DOM page test.

## 3. Findings

**Guard audit on natural code** (5,000 random BHive blocks, any instruction mix, single-step rewrites):
guards ON - 4,382 rewrites proposed, **0 illegal**; guards OFF - 7,579 proposed, **1,605 illegal** (swap 1,402/6,262,
add->lea 165/928, inc/dec 20/249 ...). Limit: 62% of the guards-on candidates (2,719) could not be emulated, almost
all because Unicorn cannot execute AVX/VEX instructions here (no control over XCR0), so on those blocks the guards are
trusted but not independently verified; 1,663 were verified equivalent.

**Natural-code decisions** (1,000 held-out test blocks, 9,434 candidates, extended rules):
* Every candidate checked in the emulator: 9,432 equivalent, 0 illegal, 2 not runnable.
* **Timing on this machine is too noisy to resolve small effects**: median run-to-run CV 2.3%; two independent passes
  agree only weakly (Spearman 0.32 on speedup ratios, SD of the difference 11.8%; only 46% of >2% effects replicate).
  The sandbox is a shared cloud VM, not your quiet machine, so treat this as a method demonstration.
* "Best measured" is biased upward by winner's curse: +5.0% (pass 1) and +3.7% (selected and scored on pass 2). Selected
  on pass 1 and scored on pass 2 (unbiased, and a LOWER bound because the selection is noisy): **+0.96% (95% CI +0.51 to +1.40%)**.
* Policies, scored on pass 2 (geomean speedup, 95% CI): keep 1.000; static table +0.01% (changes only 3 blocks - it
  scores swaps and most form changes as ties); llvm-mca -0.02% (47 changed, 10 wins/11 losses); **LSTM -0.14%
  (-0.63 to +0.34%; 154 wins/140 losses, worst 0.60x)**; random legal candidate -0.30% (-0.62 to +0.04%).
* Ordering ability on 34,624 candidate pairs whose difference replicates in both passes: LSTM 53.0% correct, llvm-mca 54.0%
  (85% ties), static ties 97.6%. **No oracle can yet rank context-dependent rewrites on natural code.**

Read together with your headline: the seeded set (1,073 of 1,082 blocks) rewards deleting inserted no-op self-moves
(R3 = 890 of 1,242 single-rule sites), where "apply every legal rule" (+39.2%) ties the static table exactly; on
natural code the achievable gain is about 1% and the learned oracle shows no edge. The research question's answer
("better prediction did not give better decisions") is consistent across both.

## 4. Still limited (be explicit in the viva)
* LSTM trained on Skylake labels; timing hosts differ (yours Cascade Lake, this sandbox a newer core).
* Natural-code timing needs a quiet machine; run `scripts/run_natural.sh` there and read `timing_reproducibility`
  (Spearman between passes should be >0.8 before trusting any decision number).
* AVX blocks cannot be emulated, so the verifier does not cover them.
* Extended rules are an experiment, not part of the deck's three-rule scope.
