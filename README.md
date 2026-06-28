# Crypto Forecast Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A rigorous walk-forward benchmark comparing classical econometric models, machine learning algorithms, deep learning architectures, and time-series foundation models for cryptocurrency returns and volatility forecasting.

This repository contains the full source code, data pipelines, and experimental orchestration to reproduce the benchmark results.

## Overview

Forecasting cryptocurrency markets presents unique challenges due to extreme volatility and non-stationary dynamics. Many existing studies suffer from methodological flaws such as look-ahead bias or weak baselines. This benchmark provides a standardized, rigorous evaluation framework utilizing an expanding-window walk-forward validation protocol to ensure robust and unbiased performance estimation.

**Key Features:**
- **Strict Walk-Forward Validation:** Expanding-window splits with zero look-ahead bias.
- **Comprehensive Model Suite:**
  - *Classical:* Naive, Historical Mean, ARIMA, GARCH
  - *Machine Learning:* XGBoost
  - *Deep Learning:* LSTM, N-BEATS, PatchTST, Temporal Fusion Transformer (TFT)
  - *Foundation Models:* Chronos (Zero-shot)
- **Rigorous Evaluation:** RMSE, MAE, MAPE, Directional Accuracy, and Diebold-Mariano tests with HLN correction.
- **Scalable Execution:** Local CPU/GPU support and seamless integration with Modal for distributed cloud execution.

## Project Structure

```text
crypto_forecast_benchmark/
├── src/
│   ├── data.py              # Data loading & feature engineering (yfinance)
│   ├── splits.py            # Walk-forward expanding-window generator
│   ├── metrics.py           # Evaluation metrics & Diebold-Mariano test
│   ├── baselines.py         # Naive, ARIMA, GARCH, and XGBoost implementations
│   ├── neural.py            # Deep learning models via neuralforecast
│   ├── foundation.py        # Chronos foundation model inference
│   ├── experiment.py        # Core experiment orchestration
│   └── analyze.py           # Result aggregation, table generation, and plotting
├── configs/
│   └── config.yaml          # Centralized configuration (assets, dates, hyperparams)
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── results/                 # Output directory for predictions, metrics, and figures
├── run_all.sh               # End-to-end reproducibility script
└── requirements.txt         # Project dependencies
```

## Quickstart (Local Execution)

1. **Environment Setup** (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Smoke Test (Fast CPU Run)**:
   Test the pipeline end-to-end using classical models on a single asset.
   ```bash
   python -m src.experiment --models naive,arima,garch --assets BTC-USD --fast
   ```

3. **Full Classical & ML Sweep (CPU)**:
   ```bash
   python -m src.experiment --models naive,hist_mean,arima,garch,xgboost
   ```

4. **Deep Learning Suite (GPU Recommended)**:
   ```bash
   python -m src.experiment --models lstm,nbeats,patchtst,tft
   ```

5. **Foundation Model (GPU Required)**:
   ```bash
   python -m src.experiment --models chronos
   ```

6. **Analysis & Reporting**:
   Aggregate results and generate performance tables/figures.
   ```bash
   python -m src.analyze
   ```

## Cloud Execution (Modal)

For compute-intensive deep learning and foundation model experiments, this project supports [Modal](https://modal.com/).

*Note for Windows PowerShell users:* Set UTF-8 encoding for proper output rendering.
```powershell
$env:PYTHONIOENCODING='utf-8'
```

**Example Modal Commands:**

- **GPU Smoke Test** (Fast verification of neural models):
  ```bash
  python -m modal run modal_experiment.py --models lstm,nbeats,patchtst,tft,chronos --assets BTC-USD --targets log_return --horizons 1 --max-splits 1 --model-max-steps 5 --run-name smoke-all-gpu
  ```

- **Full Sequential Run** (Complete benchmark suite):
  ```bash
  python -m modal run modal_experiment.py --models lstm,nbeats,patchtst,tft,chronos --full --max-splits 0 --model-max-steps 0 --run-name neural-full
  ```

Results are saved to the `crypto-forecast-results` Modal volume and downloaded locally to `results/modal_archives/`.

## Compute Requirements

| Experiment Stage | Recommended Hardware | Est. Duration |
|------------------|----------------------|---------------|
| Classical + XGBoost | Modern CPU | ~30 mins |
| LSTM, N-BEATS | 1x NVIDIA L4 / T4 | ~1.5 hours |
| PatchTST, TFT | 1x NVIDIA L4 / T4 | ~3 hours |
| Chronos (Zero-shot) | 1x NVIDIA L4 / T4 | ~30 mins |

*For cost-effective GPU compute, consider services like Lambda Labs, RunPod, Vast.ai, or Google Colab Pro+.*

## Citation

If you utilize this benchmark or codebase in your research, please cite our paper published on [ResearchGate](https://www.researchgate.net/publication/405856354_When_Do_Neural_Forecasters_Help_A_Walk-Forward_Benchmark_on_Cryptocurrency_Returns_and_Volatility):

```bibtex
@misc{korir2026neuralforecasters,
  title={When Do Neural Forecasters Help? A Walk-Forward Benchmark on Cryptocurrency Returns and Volatility},
  author={Korir, Gilbert and Mbalu, Ndetto and Aranotu, Chinedum and Tekenah, Harris},
  year={2026},
  publisher={ResearchGate},
  url={https://www.researchgate.net/publication/405856354_When_Do_Neural_Forecasters_Help_A_Walk-Forward_Benchmark_on_Cryptocurrency_Returns_and_Volatility}
}
```
