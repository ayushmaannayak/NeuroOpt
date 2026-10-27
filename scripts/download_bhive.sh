#!/usr/bin/env bash
# Fetch the BHive benchmark (Chen et al., IISWC 2019): measured throughput labels,
# categories and per-program source lists. Apache-2.0, see data/raw/BHIVE_LICENSE.txt.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
BASE=https://raw.githubusercontent.com/ithemal/bhive/master/benchmark
curl -sSfLO "$BASE/throughput/skl.csv"
curl -sSfLO "$BASE/throughput/hsw.csv"
curl -sSfLO "$BASE/categories.csv"
if [ ! -d sources ]; then
  tmp=$(mktemp -d)
  git clone -q --depth 1 https://github.com/ithemal/bhive.git "$tmp/bhive"
  cp -r "$tmp/bhive/benchmark/sources" .
  cp "$tmp/bhive/LICENSE.txt" BHIVE_LICENSE.txt
  rm -rf "$tmp"
fi
echo "BHive data in $(pwd)"
