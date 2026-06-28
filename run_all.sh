#!/usr/bin/env bash
# Run the full benchmark pipeline.
# Recommended: have a GPU available before running stages 4-5.
set -euo pipefail

echo "=== Stage 1: Smoke test (1 asset, classical only) ==="
python -m src.experiment --fast --assets BTC-USD --models naive,arima,garch

echo "=== Stage 2: Full classical + ML baselines (CPU) ==="
python -m src.experiment --models naive,hist_mean,arima,garch,xgboost

echo "=== Stage 3: Neural forecasters (GPU recommended) ==="
python -m src.experiment --models lstm,nbeats,patchtst,tft

echo "=== Stage 4: Foundation model (zero-shot) ==="
python -m src.experiment --models chronos

echo "=== Stage 5: Aggregate + figures ==="
python -m src.analyze

echo "=== Done. Results in results/, figures in results/figures/ ==="
