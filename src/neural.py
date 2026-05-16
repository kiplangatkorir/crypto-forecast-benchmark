"""Neural forecasters: LSTM, N-BEATS, PatchTST, TFT via neuralforecast."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cfg_int(model_config: Dict[str, Any], key: str, default: int) -> int:
    return int(model_config.get(key, default))


def _cfg_float(model_config: Dict[str, Any], key: str, default: float) -> float:
    return float(model_config.get(key, default))


def _cfg_list(model_config: Dict[str, Any], key: str, default: list[str]) -> list[str]:
    value = model_config.get(key, default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


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
    os.environ.setdefault("TQDM_DISABLE", "1")
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning_fabric").setLevel(logging.WARNING)

    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM, NBEATS, PatchTST, TFT
    try:
        import torch

        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("medium")
    except ImportError:
        logger.debug("torch not installed; skipping matmul precision setup")

    nf_df = _to_nf_format(train_df[target], unique_id)

    model_cls = {
        "lstm": LSTM,
        "nbeats": NBEATS,
        "patchtst": PatchTST,
        "tft": TFT,
    }[model_name.lower()]

    # Common params
    common = {
        "h": horizon,
        "input_size": _cfg_int(model_config, "input_size", 60),
        "enable_progress_bar": False,
        "logger": False,
    }

    # Model-specific param mapping
    if model_name == "lstm":
        model = LSTM(
            **common,
            encoder_hidden_size=_cfg_int(model_config, "hidden_size", 128),
            encoder_n_layers=_cfg_int(model_config, "num_layers", 2),
            learning_rate=_cfg_float(model_config, "learning_rate", 1e-3),
            max_steps=_cfg_int(model_config, "max_steps", 1000),
        )
    elif model_name == "nbeats":
        model = NBEATS(
            **common,
            stack_types=_cfg_list(
                model_config,
                "stack_types",
                ["identity", "identity", "identity"],
            ),
            learning_rate=_cfg_float(model_config, "learning_rate", 1e-3),
            max_steps=_cfg_int(model_config, "max_steps", 1000),
        )
    elif model_name == "patchtst":
        model = PatchTST(
            **common,
            patch_len=_cfg_int(model_config, "patch_len", 16),
            stride=_cfg_int(model_config, "stride", 8),
            n_heads=_cfg_int(model_config, "n_heads", 4),
            learning_rate=_cfg_float(model_config, "learning_rate", 1e-4),
            max_steps=_cfg_int(model_config, "max_steps", 1000),
        )
    elif model_name == "tft":
        model = TFT(
            **common,
            hidden_size=_cfg_int(model_config, "hidden_size", 64),
            n_head=_cfg_int(model_config, "n_head", 4),
            learning_rate=_cfg_float(model_config, "learning_rate", 1e-3),
            max_steps=_cfg_int(model_config, "max_steps", 1000),
        )
    else:
        raise ValueError(f"Unknown neural model: {model_name}")

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=nf_df)
    fc = nf.predict()
    # Column is named after the model class (e.g., 'LSTM', 'NBEATS')
    col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
    return fc[col].values[:horizon]
