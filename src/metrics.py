"""Forecast evaluation metrics including Diebold-Mariano significance tests."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (guarded against zeros)."""
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of forecasts with the correct sign.

    For returns, this is often a more decision-relevant metric than RMSE.
    """
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {k: float("nan") for k in ["rmse", "mae", "mape", "dir_acc"]}
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "dir_acc": directional_accuracy(y_true, y_pred),
    }


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    h: int = 1,
    loss: str = "mse",
) -> Tuple[float, float]:
    """Diebold-Mariano test for equal predictive accuracy.

    H0: forecasts A and B have equal expected loss.
    Returns (dm_statistic, two_sided_p_value).

    Negative DM stat -> A has lower loss (A better than B).

    Uses the Harvey, Leybourne, and Newbold (1997) small-sample correction.

    Args:
        y_true:  realized values
        pred_a:  forecasts from model A
        pred_b:  forecasts from model B
        h:       forecast horizon (for autocovariance lag count)
        loss:    "mse" or "mae"
    """
    y_true = np.asarray(y_true).ravel()
    pred_a = np.asarray(pred_a).ravel()
    pred_b = np.asarray(pred_b).ravel()

    mask = ~(np.isnan(y_true) | np.isnan(pred_a) | np.isnan(pred_b))
    y_true, pred_a, pred_b = y_true[mask], pred_a[mask], pred_b[mask]

    if loss == "mse":
        loss_a = (y_true - pred_a) ** 2
        loss_b = (y_true - pred_b) ** 2
    elif loss == "mae":
        loss_a = np.abs(y_true - pred_a)
        loss_b = np.abs(y_true - pred_b)
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = loss_a - loss_b
    n = len(d)
    if n < h + 2:
        return float("nan"), float("nan")

    mean_d = d.mean()
    # Newey-West-style variance estimate with lag h-1
    gamma_0 = np.var(d, ddof=0)
    var_d = gamma_0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1]
        var_d += 2 * cov
    var_d = var_d / n

    if var_d <= 0:
        return float("nan"), float("nan")

    dm_stat = mean_d / np.sqrt(var_d)
    # HLN small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat *= correction
    p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p_value)


def pairwise_dm_matrix(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    h: int = 1,
    loss: str = "mse",
) -> pd.DataFrame:
    """Build a pairwise DM-test p-value matrix across models."""
    names = list(predictions.keys())
    m = len(names)
    pmat = pd.DataFrame(
        np.full((m, m), np.nan), index=names, columns=names,
    )
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            _, p = diebold_mariano(y_true, predictions[a], predictions[b], h, loss)
            pmat.iloc[i, j] = p
    return pmat
