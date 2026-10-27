# Viva notes: what this build changes in the story

The study guide describes M0 as done and M1 to M3 as planned. This repository implements all of
them, so several slides need new numbers and some "planned" items become results.

## Numbers to update on the slides

| Slide claim (study guide) | This build | Where it comes from |
|---|---|---|
| 17/17 tests pass | **19/19** (`python -m pytest`) | `tests/test_neuroopt.py` |
| 0 of 3,488 illegal rewrites with guards ON | **0 of 2,600** fuzzed candidates (1,858 single-step + 742 multi-step), plus **0 of 1,464** decision-set candidates | `results/fuzz_guards.json`, `results/decision_eval.json` |
| 894 of 1,690 caught with guards OFF | **1,672 of 1,864** guard-refused rewrites caught (89.7%), 0 false alarms | `results/fuzz_guards.json` |
| 6.00 → 2.25 proxy score (62.5%) | unchanged: **6.00 → 2.25** proxy | `scripts/demo.py` |
| no hardware timing yet | worked example **measured 3.00 → 1.04 cycles/iteration** | `results/demo_output.txt` |

The fuzz counts differ from the deck because this is a fresh implementation with its own random
generator. Quote the numbers from this repository and keep the deck consistent with them.

## What you can now claim (all measured, all reproducible)

* M1: 314,173 BHive Skylake blocks prepared, deduplicated by token sequence, with a random split
  and a held-out-programs split.
* M2: the LSTM reaches 12.5% test MAPE (Spearman 0.958), against 36.2% for the calibrated static
  table and 45.8% for llvm-mca on the same blocks. On unseen programs it gets 15.0%.
  Ablations: no register names 16.2%, 30% of data 14.5%, 10% of data 16.7%.
* M3: decision-level evaluation on 1,082 held-out blocks, every candidate emulator-verified and
  timed on real hardware. Static +39.2%, llvm-mca +38.1%, LSTM +34.5%; best possible +39.8%.
* **The answer to the research question is "no" for these three rules.** The best predictor made
  the worst decisions. This is a valid, falsifiable result (study guide Q48). It directly
  supports the project's key insight that prediction error and decision quality are different things.

## What you still must NOT say

* "The LSTM makes code faster." It made *worse* choices than the table.
* "We achieved a 62.5% speedup." That figure is a proxy score. The measured numbers are separate.
* "Formally verified." The guards are still tested, not proven.
* "Works on any CPU." Timing is register-only blocks on one Cascade Lake machine, while the LSTM
  labels come from BHive's Skylake client machine.

## New questions you should expect, with answers

**Why did the LSTM lose if it predicts better?** Decisions only need the *sign* and rough size of
each rewrite's gain on a block. All three rules are almost never harmful on this CPU, so a table
that always says "faster" is close to optimal (+39.2% vs +39.8% best). The LSTM under-predicts
R2/R3 gains and predicts R1 as a slight slowdown, so it keeps the slower original. On the worked
example it predicts 1.07 cycles for a block that really takes 3.0, missing the loop-carried `imul`
latency.

**Then is the learned oracle useless?** Not shown. It is useless *for these three rules*, which are
nearly always profitable. The value of a learned oracle should appear for rewrites whose benefit
depends on context (e.g. LEA vs ADD/SHL sequences, instruction selection with port pressure).
That is the natural next experiment, and the pipeline supports it: add rules, rerun
`evaluate_decisions.py`.

**Why does R1 (mov r,0 → xor r,r) show no speedup?** Both are single cheap uops with no input
dependency on this core. The benefit is code size (5 or 7 bytes → 2 or 3), not throughput in a hot
loop. The static table predicts a 27% gain, which is a prediction error that happens to cause no
harm.

**Why does deleting mov rax, rax give +45%?** Intel's move elimination does not apply to a
same-register move, so on a loop-carried chain it costs a full cycle of latency.

**Your hardware numbers are from your own harness, not BHive's. Why?** A 100×/200× straight-line
unroll turned out to be front-end bound for short blocks and even inverted rankings (it made
`xor r,r` look slower than `mov r,0`). The effect flipped with the unroll factor, so it was a
harness artifact. The harness times the block in a small hot loop with K and 2K copies; loop
overhead cancels. It recovers textbook latencies (imul 3, shl 1, zero idiom 0.25).

**Why a "seeded" decision set?** Compilers already apply these rules. Only 181 BHive blocks have
a legal opportunity (9 register-only). Meanwhile the guards refuse 1,417 real `mov r32,r32`
zero-extensions and 532 `mov r,0` with live flags, which is good real-world evidence for the guards.
So held-out register-only BHive blocks were de-optimized with inverse rewrites, and the guards
decided legality as usual.

**Is the LSTM-vs-table difference significant?** Yes. In a paired bootstrap over the same 1,082
blocks, the LSTM's picks are 3.4% slower (95% CI 2.6 to 4.3%).


## v2 additions - likely questions
**Why is the +39% not a real-world speedup?** The decision set is 99% seeded: inverse rewrites plus inserted `mov r,r`
no-ops. Deleting them helps, and a static table that always says "cheaper" gets it right (apply-every-rule = static).
On natural held-out blocks (extended rules) the unbiased headroom is about +1% and no oracle beats doing nothing.

**What is the winner's curse here?** "Best measured" takes the minimum over ~10 noisy timings, so it is biased upward
(+5.0% vs an unbiased +0.96%). We select on one timing pass and score on an independent pass.

**How do you know the verifier is trustworthy?** It is tested both ways: 0 illegal with guards on (4,382 natural-code
and 2,600 fuzzed rewrites), 1,605 of 7,579 illegal with guards off, and since v2 it has no false alarms on identical
blocks (0 of 1,500 seeds) while real differences are still caught (300/300). Limit: AVX blocks cannot be emulated.

**What bugs did you find in your own tools?** (1) Timer calibrated on `add rax,1`, which newer Intel cores fold, giving
cycles ~6x too large (fixed, with an `imul` self-check). (2) The emulator leaked state between runs and could call an
identical block "illegal" (fixed; mismatches are confirmed on fresh emulators).

**Why not just report the best numbers?** The research question is whether better prediction gives better decisions;
a benchmark where apply-everything is optimal cannot answer it, so we added a natural benchmark and a test-retest design.
