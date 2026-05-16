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
python -m src.experiment --models naive,hist_mean,arima,garch,xgboost

# 4. Neural models (GPU recommended, ~3-6 hours total)
python -m src.experiment --models lstm,nbeats,patchtst,tft

# 5. Foundation model (GPU, ~30 min)
python -m src.experiment --models chronos

# 6. Aggregate and analyze
python -m src.analyze
```

## Modal GPU Runs

Use Modal for neural and foundation-model runs. On Windows PowerShell, set
UTF-8 output first so Modal's CLI status output renders correctly:

```powershell
$env:PYTHONIOENCODING='utf-8'

# Cheap GPU smoke test: all GPU-backed models, BTC only, one split, five steps.
python -m modal run modal_experiment.py --models lstm,nbeats,patchtst,tft,chronos --assets BTC-USD --targets log_return --horizons 1 --max-splits 1 --model-max-steps 5 --run-name smoke-all-gpu

# Deadline run: keep all assets/targets/horizons, but cap splits and steps first.
python -m modal run modal_experiment.py --models lstm,nbeats,patchtst,tft,chronos --full --max-splits 10 --model-max-steps 200 --run-name neural-deadline-v1

# Full sequential run using config max_steps. This can be very expensive because
# it retrains every neural model on every walk-forward split.
python -m modal run modal_experiment.py --models lstm,nbeats,patchtst,tft,chronos --full --max-splits 0 --model-max-steps 0 --run-name neural-full

# Full classical + XGBoost run on Modal CPU workers, useful if local xgboost is
# not installed.
python -m modal run modal_experiment.py --models naive,hist_mean,arima,garch,xgboost --full --gpu CPU --max-splits 0 --model-max-steps 0 --run-name classical-full
```

Modal results are saved in the `crypto-forecast-results` Modal volume and
downloaded locally as zip archives under `results/modal_archives/`.

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
