"""Baseline forecasters: Naive, ARIMA, GARCH, XGBoost.

All baselines share a simple interface:
    fit(train_df, target, config) -> model_state
    forecast(model_state, horizon, target) -> np.ndarray of length `horizon`
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------

def naive_forecast(train_df: pd.DataFrame, horizon: int, target: str) -> np.ndarray:
    """Predict last observed value for all future steps (random walk)."""
    last = train_df[target].iloc[-1]
    return np.full(horizon, last, dtype=float)


def historical_mean_forecast(
    train_df: pd.DataFrame, horizon: int, target: str
) -> np.ndarray:
    """Predict the historical mean for all steps."""
    return np.full(horizon, train_df[target].mean(), dtype=float)


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def arima_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    target: str,
    order: tuple = (2, 0, 2),
) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA

    series = train_df[target].dropna().values
    try:
        model = ARIMA(series, order=order)
        fit = model.fit()
        return fit.forecast(steps=horizon)
    except Exception as e:
        logger.warning("ARIMA failed: %s. Falling back to naive.", e)
        return naive_forecast(train_df, horizon, target)


# ---------------------------------------------------------------------------
# GARCH (forecasts conditional volatility)
# ---------------------------------------------------------------------------

def garch_volatility_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    return_col: str = "log_return",
    p: int = 1,
    q: int = 1,
    mean: str = "AR",
    lags: int = 1,
) -> np.ndarray:
    """GARCH conditional volatility forecast.

    Returns predicted daily-return std for each of the next `horizon` days.
    """
    from arch import arch_model

    returns = train_df[return_col].dropna().values * 100  # scale for numerical stability
    try:
        model = arch_model(returns, vol="GARCH", p=p, q=q, mean=mean, lags=lags)
        fit = model.fit(disp="off", show_warning=False)
        fc = fit.forecast(horizon=horizon, reindex=False)
        # variance forecast row [-1] holds the path
        cond_var = fc.variance.values[-1]
        cond_std = np.sqrt(cond_var) / 100  # un-scale
        return cond_std
    except Exception as e:
        logger.warning("GARCH failed: %s. Returning historical std.", e)
        return np.full(horizon, train_df[return_col].std(), dtype=float)


# ---------------------------------------------------------------------------
# XGBoost on lag features
# ---------------------------------------------------------------------------

def build_lag_matrix(
    series: pd.Series,
    lags: List[int],
    rolling_windows: List[int],
    rolling_stats: List[str],
) -> pd.DataFrame:
    """Build lag-feature matrix; same logic as data.make_lag_features."""
    from src.data import make_lag_features
    return make_lag_features(series, lags, rolling_windows, rolling_stats)


def xgboost_recursive_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    target: str,
    lags: List[int],
    rolling_windows: List[int],
    rolling_stats: List[str],
    xgb_params: Dict[str, Any],
) -> np.ndarray:
    """Train an XGBoost regressor on lag features; forecast recursively.

    Recursive: predict t+1, append to history, predict t+2, etc.
    """
    import xgboost as xgb

    series = train_df[target].copy()
    feats = build_lag_matrix(series, lags, rolling_windows, rolling_stats)
    # Align target (= series at time t) with features (built from data <= t-1)
    y = series.loc[feats.index]
    X = feats.values
    y_arr = y.values

    params = {k: v for k, v in xgb_params.items() if k != "early_stopping_rounds"}
    model = xgb.XGBRegressor(**params, verbosity=0)
    model.fit(X, y_arr)

    # Recursive multi-step forecast
    history = series.copy()
    preds = []
    for _ in range(horizon):
        feat_row = build_lag_matrix(history, lags, rolling_windows, rolling_stats).iloc[[-1]]
        next_val = float(model.predict(feat_row.values)[0])
        preds.append(next_val)
        # Append prediction as the new "observation" for the next step
        next_idx = history.index[-1] + pd.Timedelta(days=1)
        history.loc[next_idx] = next_val
    return np.array(preds)
