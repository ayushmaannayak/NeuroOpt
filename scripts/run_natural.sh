#!/bin/bash
# Natural-code experiments (optional extended rules). Run on a QUIET machine: stop other jobs, pin the CPU.
set -e
cd "$(dirname "$0")/.."
python3 scripts/build_natural_decision_set.py --max-blocks 1000
python3 scripts/evaluate_decisions.py --decision-set natural_decision_set.jsonl --tag _natural --extended
python3 scripts/retest_natural.py --tag _natural
python3 scripts/analyse_natural_replicated.py --tag _natural      # check timing_reproducibility.spearman first
python3 scripts/audit_natural.py --blocks 5000
