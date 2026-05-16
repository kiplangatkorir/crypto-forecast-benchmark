"""Chronos zero-shot time-series forecasting (Amazon, 2024).

Paper: https://arxiv.org/abs/2403.07815
Reference: https://github.com/amazon-science/chronos-forecasting
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _load_chronos(model_name: str, device: str):
    """Cache Chronos pipeline across splits to avoid repeated loading."""
    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    return pipeline


def chronos_forecast(
    train_df: pd.DataFrame,
    horizon: int,
    target: str,
    model_name: str = "amazon/chronos-t5-small",
    num_samples: int = 20,
    device: Optional[str] = None,
) -> np.ndarray:
    """Zero-shot probabilistic forecast via Chronos. Returns median forecast."""
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline = _load_chronos(model_name, device)
    series = train_df[target].dropna().values
    context = torch.tensor(series, dtype=torch.float32)

    forecast = pipeline.predict(
        context=context,
        prediction_length=horizon,
        num_samples=num_samples,
    )
    # forecast shape: [num_series=1, num_samples, prediction_length]
    samples = forecast[0].numpy()  # shape (num_samples, horizon)
    return np.median(samples, axis=0)
