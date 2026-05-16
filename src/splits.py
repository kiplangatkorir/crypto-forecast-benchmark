"""Walk-forward expanding-window splits.

Critical for avoiding lookahead bias in financial forecasting benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd


@dataclass
class Split:
    """A single walk-forward (train, test) split."""
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train: pd.DataFrame
    test: pd.DataFrame


def expanding_window_splits(
    df: pd.DataFrame,
    initial_train_days: int = 730,
    test_horizon_days: int = 30,
    step_days: int = 30,
    max_splits: Optional[int] = None,
) -> Iterator[Split]:
    """Yield (train, test) splits with an expanding training window.

    Args:
        df: time-indexed DataFrame, sorted ascending.
        initial_train_days: rows in the FIRST training window.
        test_horizon_days: rows in each test window.
        step_days: how far the test window moves between splits.
        max_splits: optional cap on number of splits.

    Yields:
        Split objects with train and test slices.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted ascending.")

    n = len(df)
    start = initial_train_days
    split_id = 0

    while start + test_horizon_days <= n:
        train = df.iloc[:start]
        test = df.iloc[start:start + test_horizon_days]
        yield Split(
            split_id=split_id,
            train_start=train.index[0],
            train_end=train.index[-1],
            test_start=test.index[0],
            test_end=test.index[-1],
            train=train,
            test=test,
        )
        split_id += 1
        if max_splits is not None and split_id >= max_splits:
            return
        start += step_days


def count_splits(
    n_rows: int,
    initial_train_days: int = 730,
    test_horizon_days: int = 30,
    step_days: int = 30,
) -> int:
    """Compute number of splits without iterating."""
    if n_rows < initial_train_days + test_horizon_days:
        return 0
    return 1 + (n_rows - initial_train_days - test_horizon_days) // step_days
