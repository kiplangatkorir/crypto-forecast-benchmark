# Crypto Forecast Benchmark

Rigorous walk-forward benchmark comparing classical, ML, deep learning, and foundation models
for cryptocurrency returns and volatility forecasting.

**Target venue:** ICICPE 2026 (deadline May 31, 2026)

## Project Structure

```
crypto_forecast_benchmark/
├── src/
│   ├── data.py              # Data loading + feature engineering
│   ├── splits.py            # Walk-forward expanding-window splits
│   ├── metrics.py           # RMSE, MAE, MAPE, DM test, directional accuracy
│   ├── baselines.py         # Naive, ARIMA, GARCH, XGBoost
│   ├── neural.py            # LSTM, N-BEATS, PatchTST, TFT (neuralforecast)
│   ├── foundation.py        # Chronos zero-shot
│   ├── experiment.py        # Main experiment runner
│   └── analyze.py           # Aggregate results, generate tables/figures
├── configs/
│   └── config.yaml          # Asset list, date ranges, model hyperparams
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── results/                 # Output: predictions, metrics, figures
├── requirements.txt
└── run_all.sh               # Reproducibility script
```

## Quickstart

```bash
# 1. Install (use a fresh venv; Python 3.10+)
pip install -r requirements.txt

# 2. Smoke test on CPU (classical models only, 1 asset, fast)
python -m src.experiment --models naive,arima,garch --assets BTC-USD --fast

# 3. Full classical + ML run (CPU, ~30 min)
python -m src.experiment --models naive,arima,garch,xgboost

# 4. Neural models (GPU recommended, ~3-6 hours total)
python -m src.experiment --models lstm,nbeats,patchtst,tft

# 5. Foundation model (GPU, ~30 min)
python -m src.experiment --models chronos

# 6. Aggregate and analyze
python -m src.analyze
```

## Compute Budget Estimate

| Stage | Hardware | Time | Est. Cost |
|---|---|---|---|
| Classical + XGBoost | CPU | ~30 min | $0 (local) |
| LSTM, N-BEATS | L4 GPU | ~1.5 hr | ~$2 |
| PatchTST, TFT | L4 GPU | ~3 hr | ~$4 |
| Chronos zero-shot | L4 GPU | ~30 min | ~$1 |
| Tuning / re-runs | L4 GPU | ~6 hr | ~$8 |
| **Total** |  |  | **~$15–25** |

Lambda Labs, RunPod, or Vast.ai for cheapest L4/A10 instances. Colab Pro+ also works.

## Citation
If this benchmark is useful for your research, please cite the accompanying ICICPE 2026 paper.
