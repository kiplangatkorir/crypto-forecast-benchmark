# AGENTS.md

Context for Codex when working on this repository.

## Project

**Crypto Forecast Benchmark** — a rigorous walk-forward benchmark comparing
classical, ML, deep learning, and foundation models for cryptocurrency
returns and volatility forecasting.

**Target:** Paper submission to ICICPE 2026 (Chiang Mai, Thailand, Aug 19–21, 2026).
**Submission deadline:** May 31, 2026.
**Affiliation on submission:** WorldQuant University (WQU).

## Core thesis

Most crypto-forecasting papers suffer from methodological weaknesses (lookahead
bias, single train-test splits, weak baselines). We present a rigorous benchmark
with:

1. Expanding-window walk-forward validation (no lookahead).
2. Strong classical baselines (ARIMA, GARCH) tuned correctly.
3. Modern neural forecasters (LSTM, N-BEATS, PatchTST, TFT).
4. A time-series foundation model (Chronos) for zero-shot comparison.
5. Diebold-Mariano significance tests with HLN small-sample correction.

The publishable story is the *contrast*: neural models likely fail to beat
random walk on **returns** (near-random-walk) but should provide significant
gains on **realized volatility** (more predictable).

## Repository layout

```
src/
  data.py          Data loading (yfinance) + feature engineering
  splits.py        Walk-forward expanding-window splits
  metrics.py       RMSE/MAE/MAPE/dir_acc + Diebold-Mariano test
  baselines.py     Naive, ARIMA, GARCH, XGBoost
  neural.py        LSTM, N-BEATS, PatchTST, TFT via neuralforecast
  foundation.py    Chronos zero-shot
  experiment.py    Main runner (walk-forward orchestration)
  analyze.py       Aggregate results, DM tests, figures
configs/
  config.yaml      All hyperparameters and asset list
results/           Generated: metrics.csv, predictions.parquet, figures/
```

## Conventions

- **Python 3.10+**. Type hints everywhere; `from __future__ import annotations` at top of modules.
- **Logging, not prints.** Each module has `logger = logging.getLogger(__name__)`.
- **No lookahead.** Any feature at time `t` must use only data `<= t-1`. Lag
  features always use `shift(1)` before rolling windows.
- **Reproducibility.** All randomness keyed off `cfg["seed"]` (currently 42).
- **Forecaster interface.** Every model is dispatched via `experiment.forecast_one()`,
  which takes `(model_name, train_df, horizon, target, cfg)` and returns a
  numpy array of length `horizon`. Adding a new model = add a branch there and
  a function in the relevant module.
- **Data flow.** `experiment.py` writes two files per run: `metrics.csv`
  (one row per asset/model/target/horizon/split) and `predictions.parquet`
  (one row per forecasted timestep). `analyze.py` reads both.

## Commands

```bash
# Smoke test (CPU, <5 min)
python -m src.experiment --fast --assets BTC-USD --models naive,arima,garch

# Full classical + ML sweep (CPU, ~30 min)
python -m src.experiment --models naive,hist_mean,arima,garch,xgboost

# Neural models (GPU recommended)
python -m src.experiment --models lstm,nbeats,patchtst,tft

# Foundation model (GPU)
python -m src.experiment --models chronos

# Aggregate + generate figures and DM tables
python -m src.analyze
```

## Working with Codex on this repo

When starting a session, useful first prompts:

- "Read AGENTS.md and the src/ directory, then summarize what's implemented and what's still TODO."
- "Run the smoke test and report any failures."
- "Help me draft the methodology section based on the code in src/."

## Known TODOs / open work

- [ ] **Hyperparameter tuning** for neural models — currently using sensible defaults; should add a small grid search per model on the first split's validation slice.
- [ ] **Error analysis by volatility regime** — split test windows into high-vol / low-vol periods (using ex-ante GARCH-implied vol) and report metrics per regime. This is a key paper figure.
- [ ] **Add probabilistic metrics** — CRPS or pinball loss for models that produce distributions (GARCH, Chronos). Currently we only use the median.
- [ ] **Robustness check** — re-run with hourly data on 2024–2025 only, to show results generalize across frequencies.
- [ ] **Paper draft** — `paper/` directory with LaTeX (likely IEEE template per ICICPE call).
- [ ] **Unit tests** — at least for `metrics.diebold_mariano` (currently smoke-tested only).

## Paper draft outline (working)

1. **Introduction** — motivation: methodological weaknesses in crypto forecasting literature.
2. **Related Work** — survey of (a) classical financial econometrics, (b) DL for finance, (c) time-series foundation models.
3. **Methodology** — data, walk-forward protocol, models, metrics including DM test.
4. **Experimental Setup** — assets, date ranges, splits, hyperparameters, compute.
5. **Results** — three subsections: returns, volatility, regime analysis.
6. **Discussion** — when do DL models help? Foundation-model zero-shot performance. Cost of getting validation wrong.
7. **Conclusion + Limitations** — daily-only, 5 assets, English-language audience.

## Key references to cite

- Diebold & Mariano (1995) — DM test
- Harvey, Leybourne & Newbold (1997) — HLN correction
- Bollerslev (1986) — GARCH
- Lim et al. (2021) — Temporal Fusion Transformer
- Nie et al. (2023) — PatchTST
- Oreshkin et al. (2020) — N-BEATS
- Ansari et al. (2024) — Chronos
- Makridakis et al. (M4/M5 competitions) — forecasting benchmark methodology
