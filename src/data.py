"""Data loading and feature engineering for crypto forecasting benchmark."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_asset(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Download or load cached OHLCV data for a single asset."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{ticker.replace('/', '_')}_{interval}.parquet"
        if cache_file.exists():
            logger.info("Loading %s from cache", ticker)
            return pd.read_parquet(cache_file)

    logger.info("Downloading %s from yfinance", ticker)
    df = yf.download(
        ticker, start=start, end=end, interval=interval,
        progress=False, auto_adjust=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if cache_dir is not None:
        df.to_parquet(cache_file)
    return df


def engineer_features(
    df: pd.DataFrame,
    vol_window: int = 7,
) -> pd.DataFrame:
    """Add log returns, realized volatility, and engineered features."""
    out = df.copy()
    out["log_return"] = np.log(out["Close"] / out["Close"].shift(1))
    out[f"realized_vol_{vol_window}d"] = (
        out["log_return"].rolling(vol_window).std()
    )
    # Squared returns (proxy for variance, useful for GARCH baseline comparison)
    out["squared_return"] = out["log_return"] ** 2
    return out.dropna()


def make_lag_features(
    series: pd.Series,
    lags: List[int],
    rolling_windows: List[int],
    rolling_stats: List[str],
) -> pd.DataFrame:
    """Build a lag-feature dataframe for tabular models (XGBoost).

    Important: no lookahead. All features at time t use only data <= t-1.
    """
    feats = pd.DataFrame(index=series.index)
    for lag in lags:
        feats[f"lag_{lag}"] = series.shift(lag)
    for w in rolling_windows:
        # shift(1) ensures the rolling window does NOT include the current obs
        rolled = series.shift(1).rolling(w)
        if "mean" in rolling_stats:
            feats[f"roll_mean_{w}"] = rolled.mean()
        if "std" in rolling_stats:
            feats[f"roll_std_{w}"] = rolled.std()
        if "min" in rolling_stats:
            feats[f"roll_min_{w}"] = rolled.min()
        if "max" in rolling_stats:
            feats[f"roll_max_{w}"] = rolled.max()
    return feats.dropna()


def load_all_assets(
    assets: List[str],
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: Path | None = Path("data_cache"),
    vol_window: int = 7,
) -> Dict[str, pd.DataFrame]:
    """Load and engineer features for all configured assets."""
    out: Dict[str, pd.DataFrame] = {}
    for a in assets:
        raw = download_asset(a, start, end, interval, cache_dir)
        out[a] = engineer_features(raw, vol_window=vol_window)
        logger.info("%s: %d rows from %s to %s",
                    a, len(out[a]), out[a].index.min(), out[a].index.max())
    return out
