"""Neural forecasters: LSTM, N-BEATS, PatchTST, TFT via neuralforecast."""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_nf_format(series: pd.Series, unique_id: str) -> pd.DataFrame:
    """Convert a single time series to neuralforecast's expected long format."""
    df = pd.DataFrame({
        "unique_id": unique_id,
        "ds": series.index,
        "y": series.values,
    }).dropna()
    return df


def neural_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    target: str,
    model_name: str,
    model_config: Dict[str, Any],
    unique_id: str = "series",
) -> np.ndarray:
    """Train a neural forecaster and predict `horizon` steps ahead.

    Supported model_name values: 'lstm', 'nbeats', 'patchtst', 'tft'.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, NBEATS, PatchTST, TFT

    nf_df = _to_nf_format(train_df[target], unique_id)

    model_cls = {
        "lstm": LSTM,
        "nbeats": NBEATS,
        "patchtst": PatchTST,
        "tft": TFT,
    }[model_name.lower()]

    # Common params
    common = {"h": horizon, "input_size": model_config.get("input_size", 60)}

    # Model-specific param mapping
    if model_name == "lstm":
        model = LSTM(
            **common,
            encoder_hidden_size=model_config.get("hidden_size", 128),
            encoder_n_layers=model_config.get("num_layers", 2),
            learning_rate=model_config.get("learning_rate", 1e-3),
            max_steps=model_config.get("max_steps", 1000),
        )
    elif model_name == "nbeats":
        model = NBEATS(
            **common,
            learning_rate=model_config.get("learning_rate", 1e-3),
            max_steps=model_config.get("max_steps", 1000),
        )
    elif model_name == "patchtst":
        model = PatchTST(
            **common,
            patch_len=model_config.get("patch_len", 16),
            stride=model_config.get("stride", 8),
            n_heads=model_config.get("n_heads", 4),
            learning_rate=model_config.get("learning_rate", 1e-4),
            max_steps=model_config.get("max_steps", 1000),
        )
    elif model_name == "tft":
        model = TFT(
            **common,
            hidden_size=model_config.get("hidden_size", 64),
            n_head=model_config.get("n_head", 4),
            learning_rate=model_config.get("learning_rate", 1e-3),
            max_steps=model_config.get("max_steps", 1000),
        )
    else:
        raise ValueError(f"Unknown neural model: {model_name}")

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=nf_df)
    fc = nf.predict()
    # Column is named after the model class (e.g., 'LSTM', 'NBEATS')
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    return fc[col].values[:horizon]
