"""Main experiment runner.

Orchestrates walk-forward evaluation across (asset, model, target) combinations.

Usage:
    python -m src.experiment --models naive,arima,garch,xgboost
    python -m src.experiment --models lstm,nbeats,patchtst,tft
    python -m src.experiment --models chronos
    python -m src.experiment --fast --assets BTC-USD --models naive,arima
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src import baselines, neural, foundation
from src.data import load_all_assets
from src.metrics import all_metrics
from src.splits import expanding_window_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("experiment")


# ---------------------------------------------------------------------------
# Forecast dispatcher
# ---------------------------------------------------------------------------

def forecast_one(
    model_name: str,
    train_df: pd.DataFrame,
    horizon: int,
    target: str,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """Dispatch to the right forecaster. Returns array of length `horizon`."""
    feats_cfg = cfg.get("features", {})
    models_cfg = cfg.get("models", {})

    if model_name == "naive":
        return baselines.naive_forecast(train_df, horizon, target)
    if model_name == "hist_mean":
        return baselines.historical_mean_forecast(train_df, horizon, target)
    if model_name == "arima":
        order = tuple(models_cfg.get("arima", {}).get("order", (2, 0, 2)))
        return baselines.arima_forecast(train_df, horizon, target, order=order)
    if model_name == "garch":
        g = models_cfg.get("garch", {})
        # GARCH forecasts VOLATILITY (std of returns), not returns themselves.
        # Only meaningful when target is a volatility column.
        return baselines.garch_volatility_forecast(
            train_df, horizon,
            return_col="log_return",
            p=g.get("p", 1), q=g.get("q", 1),
            mean=g.get("mean", "AR"), lags=g.get("lags", 1),
        )
    if model_name == "xgboost":
        return baselines.xgboost_recursive_forecast(
            train_df, horizon, target,
            lags=feats_cfg.get("lags", [1, 2, 3, 5, 7, 14, 30]),
            rolling_windows=feats_cfg.get("rolling_windows", [7, 14, 30]),
            rolling_stats=feats_cfg.get("rolling_stats", ["mean", "std"]),
            xgb_params=models_cfg.get("xgboost", {}),
        )
    if model_name in ("lstm", "nbeats", "patchtst", "tft"):
        return neural.neural_forecast(
            train_df, horizon, target,
            model_name=model_name,
            model_config=models_cfg.get(model_name, {}),
        )
    if model_name == "chronos":
        c = models_cfg.get("chronos", {})
        return foundation.chronos_forecast(
            train_df, horizon, target,
            model_name=c.get("model_name", "amazon/chronos-t5-small"),
            num_samples=c.get("num_samples", 20),
        )
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_experiment(cfg: Dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    assets = args.assets.split(",") if args.assets else cfg["data"]["assets"]
    models = args.models.split(",") if args.models else ["naive", "arima"]
    targets = cfg["targets"]
    horizons = cfg["horizons"]

    if args.fast:
        cfg["walk_forward"]["max_splits"] = 5

    data = load_all_assets(
        assets,
        start=cfg["data"]["start_date"],
        end=cfg["data"]["end_date"],
        interval=cfg["data"]["interval"],
    )

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / "predictions.parquet"
    metrics_path = results_dir / "metrics.csv"

    all_predictions: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, Any]] = []

    for asset, df in data.items():
        for target in targets:
            if target not in df.columns:
                logger.warning("Target %s not in %s columns, skipping", target, asset)
                continue
            for horizon in horizons:
                splits = list(expanding_window_splits(
                    df,
                    initial_train_days=cfg["walk_forward"]["initial_train_days"],
                    test_horizon_days=max(cfg["walk_forward"]["test_horizon_days"], horizon),
                    step_days=cfg["walk_forward"]["step_days"],
                    max_splits=cfg["walk_forward"].get("max_splits"),
                ))
                for model_name in models:
                    pbar = tqdm(
                        splits,
                        desc=f"{asset} | {target} | h={horizon} | {model_name}",
                    )
                    for sp in pbar:
                        try:
                            pred = forecast_one(model_name, sp.train, horizon, target, cfg)
                        except Exception as e:
                            logger.error(
                                "Forecast failed: asset=%s model=%s target=%s split=%d: %s",
                                asset, model_name, target, sp.split_id, e,
                            )
                            continue

                        # Compare against the first `horizon` test observations
                        y_true = sp.test[target].iloc[:horizon].values
                        y_pred = np.asarray(pred[:horizon]).ravel()

                        metrics = all_metrics(y_true, y_pred)
                        record = {
                            "asset": asset,
                            "target": target,
                            "horizon": horizon,
                            "model": model_name,
                            "split_id": sp.split_id,
                            "test_start": sp.test_start,
                            "test_end": sp.test_start + pd.Timedelta(days=horizon-1),
                            **metrics,
                        }
                        all_metrics.append(record)

                        # Store all predictions for downstream DM tests
                        for k in range(horizon):
                            all_predictions.append({
                                "asset": asset,
                                "target": target,
                                "horizon": horizon,
                                "model": model_name,
                                "split_id": sp.split_id,
                                "step": k + 1,
                                "date": sp.test.index[k],
                                "y_true": float(y_true[k]) if k < len(y_true) else np.nan,
                                "y_pred": float(y_pred[k]) if k < len(y_pred) else np.nan,
                            })

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.DataFrame(all_predictions)
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_parquet(predictions_path)
    logger.info("Saved metrics: %s (%d rows)", metrics_path, len(metrics_df))
    logger.info("Saved predictions: %s (%d rows)", predictions_path, len(preds_df))
    return metrics_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", default="naive,arima",
                        help="Comma-separated list. Options: naive,hist_mean,arima,"
                             "garch,xgboost,lstm,nbeats,patchtst,tft,chronos")
    parser.add_argument("--assets", default=None,
                        help="Comma-separated list (overrides config).")
    parser.add_argument("--fast", action="store_true",
                        help="Use only 5 splits, for smoke testing.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg, args)


if __name__ == "__main__":
    main()
