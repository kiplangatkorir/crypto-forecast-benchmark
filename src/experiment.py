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
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src import baselines, neural, foundation
from src.data import load_all_assets
from src.metrics import all_metrics as compute_all_metrics
from src.splits import expanding_window_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("experiment")


def is_model_applicable(model_name: str, target: str) -> bool:
    """Return whether a model should be evaluated for the given target."""
    if model_name == "garch" and "vol" not in target:
        return False
    return True


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated CLI value into a list."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: Optional[str]) -> Optional[List[int]]:
    parsed = _parse_csv(value)
    if parsed is None:
        return None
    return [int(item) for item in parsed]


def _set_seed(seed: int) -> None:
    """Seed common RNGs used by baseline and neural models."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed; skipping torch seed setup")


def apply_runtime_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI/runtime overrides to a loaded config dictionary."""
    if getattr(args, "targets", None):
        cfg["targets"] = _parse_csv(args.targets)
    if getattr(args, "horizons", None):
        cfg["horizons"] = _parse_int_csv(args.horizons)
    if getattr(args, "results_dir", None):
        cfg["output"]["results_dir"] = args.results_dir
        cfg["output"]["figures_dir"] = str(Path(args.results_dir) / "figures")
    if getattr(args, "max_splits", None) is not None:
        cfg["walk_forward"]["max_splits"] = args.max_splits
    if getattr(args, "model_max_steps", None) is not None:
        for model_name in ("lstm", "nbeats", "patchtst", "tft"):
            cfg.setdefault("models", {}).setdefault(model_name, {})["max_steps"] = (
                args.model_max_steps
            )
    return cfg


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
        if not is_model_applicable(model_name, target):
            raise ValueError("GARCH forecasts conditional volatility, not returns")
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

def run_experiment(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    on_checkpoint: Optional[Callable[[], None]] = None,
) -> pd.DataFrame:
    _set_seed(int(cfg.get("seed", 42)))

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
    predictions_path = results_dir / cfg["output"].get("predictions_file", "predictions.parquet")
    metrics_path = results_dir / cfg["output"].get("metrics_file", "metrics.csv")

    all_predictions: List[Dict[str, Any]] = []
    metric_records: List[Dict[str, Any]] = []

    def save_outputs() -> None:
        metrics_df = pd.DataFrame(metric_records)
        preds_df = pd.DataFrame(all_predictions)
        metrics_df.to_csv(metrics_path, index=False)
        preds_df.to_parquet(predictions_path)
        if on_checkpoint is not None:
            on_checkpoint()

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
                    if not is_model_applicable(model_name, target):
                        logger.info(
                            "Skipping model=%s for target=%s because it is not applicable",
                            model_name,
                            target,
                        )
                        continue
                    pbar = tqdm(
                        splits,
                        desc=f"{asset} | {target} | h={horizon} | {model_name}",
                    )
                    for sp in pbar:
                        try:
                            pred = forecast_one(model_name, sp.train, horizon, target, cfg)
                        except Exception as e:
                            logger.exception(
                                "Forecast failed: asset=%s model=%s target=%s split=%d: %s",
                                asset, model_name, target, sp.split_id, e,
                            )
                            continue

                        # Compare against the first `horizon` test observations
                        y_true = sp.test[target].iloc[:horizon].values
                        y_pred = np.asarray(pred[:horizon]).ravel()

                        metrics = compute_all_metrics(y_true, y_pred)
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
                        metric_records.append(record)

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
                    if metric_records:
                        save_outputs()

    metrics_df = pd.DataFrame(metric_records)
    preds_df = pd.DataFrame(all_predictions)
    save_outputs()
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
    parser.add_argument("--targets", default=None,
                        help="Comma-separated targets, e.g. log_return,realized_vol_7d.")
    parser.add_argument("--horizons", default=None,
                        help="Comma-separated forecast horizons, e.g. 1,7.")
    parser.add_argument("--results-dir", default=None,
                        help="Override output results directory.")
    parser.add_argument("--max-splits", type=int, default=None,
                        help="Override max walk-forward splits.")
    parser.add_argument("--model-max-steps", type=int, default=None,
                        help="Override max_steps for neural models.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = apply_runtime_overrides(cfg, args)
    run_experiment(cfg, args)


if __name__ == "__main__":
    main()
