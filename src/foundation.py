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
    try:
        from chronos import BaseChronosPipeline as Pipeline
    except ImportError:
        from chronos import ChronosPipeline as Pipeline

    kwargs = {
        "device_map": device,
        "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
    }
    try:
        pipeline = Pipeline.from_pretrained(model_name, **kwargs)
    except OSError:
        logger.warning("Chronos load failed; retrying with force_download=True")
        pipeline = Pipeline.from_pretrained(
            model_name,
            force_download=True,
            **kwargs,
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

    try:
        forecast = pipeline.predict(context, horizon, num_samples=num_samples)
    except TypeError:
        forecast = pipeline.predict(context, horizon)
    # forecast shape: [num_series=1, num_samples, prediction_length]
    samples = forecast[0].detach().cpu().numpy()  # shape (num_samples, horizon)
    return np.median(samples, axis=0)
