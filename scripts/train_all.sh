#!/usr/bin/env bash
# All M2 training runs: main model, held-out program families, and ablations.
set -euo pipefail
cd "$(dirname "$0")/.."
taskset -c 0-2 python3 scripts/train_lstm.py --threads 3 --tag main   --epochs 20                      > results/log_train_main.txt 2>&1
taskset -c 0-2 python3 scripts/train_lstm.py --threads 3 --tag family --epochs 12 --split split_family > results/log_train_family.txt 2>&1
taskset -c 0-2 python3 scripts/train_lstm.py --threads 3 --tag noreg  --epochs 12 --no-registers      > results/log_train_noreg.txt 2>&1
taskset -c 0-2 python3 scripts/train_lstm.py --threads 3 --tag frac30 --epochs 12 --frac 0.3          > results/log_train_frac30.txt 2>&1
taskset -c 0-2 python3 scripts/train_lstm.py --threads 3 --tag frac10 --epochs 12 --frac 0.1          > results/log_train_frac10.txt 2>&1
echo ALL_TRAINING_DONE
