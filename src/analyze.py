"""Analyze results: aggregate metrics, run DM tests, generate publication figures.

Usage:
    python -m src.analyze
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.metrics import pairwise_dm_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("analyze")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def load_results(results_dir: Path):
    metrics = pd.read_csv(results_dir / "metrics.csv", parse_dates=["test_start", "test_end"])
    preds = pd.read_parquet(results_dir / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    return metrics, preds


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each metric per (target, horizon, model)."""
    agg = (
        metrics
        .groupby(["target", "horizon", "model"])[["rmse", "mae", "mape", "dir_acc"]]
        .agg(["mean", "std"])
        .round(5)
    )
    return agg


def rank_models(metrics: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """Mean rank across splits per (target, horizon)."""
    ranks = (
        metrics
        .groupby(["target", "horizon", "split_id"])
        .apply(lambda g: g.set_index("model")[metric].rank())
        .reset_index()
        .groupby(["target", "horizon", "model"])[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: f"mean_rank_{metric}"})
    )
    return ranks.sort_values(["target", "horizon", f"mean_rank_{metric}"])


def dm_tests_per_pair(preds: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Run pairwise DM tests for one (target, horizon) configuration."""
    sub = preds[(preds["target"] == target) & (preds["horizon"] == horizon)].copy()
    sub = sub.dropna(subset=["y_true", "y_pred"])

    # Build a wide matrix: rows = (asset, date, step), cols = model -> pred
    wide = sub.pivot_table(
        index=["asset", "date", "step"], columns="model",
        values="y_pred", aggfunc="first",
    ).dropna()
    truth = (
        sub.groupby(["asset", "date", "step"])["y_true"].first().loc[wide.index]
    )

    preds_dict = {m: wide[m].values for m in wide.columns}
    pmat = pairwise_dm_matrix(truth.values, preds_dict, h=horizon, loss="mse")
    return pmat


def plot_metric_by_model(
    metrics: pd.DataFrame,
    metric: str = "rmse",
    out_dir: Path = Path("results/figures"),
):
    """Boxplot of a metric across splits, faceted by (target, horizon)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for (target, horizon), g in metrics.groupby(["target", "horizon"]):
        plt.figure(figsize=(10, 5))
        order = g.groupby("model")[metric].median().sort_values().index
        sns.boxplot(data=g, x="model", y=metric, order=order)
        plt.title(f"{metric.upper()} across walk-forward splits — {target}, h={horizon}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        fp = out_dir / f"{metric}_{target}_h{horizon}.png"
        plt.savefig(fp, dpi=200)
        plt.close()
        logger.info("Saved %s", fp)


def plot_dm_heatmap(
    pmat: pd.DataFrame,
    title: str,
    out_path: Path,
):
    """Heatmap of DM-test p-values (row beats column when cell < 0.05)."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pmat, annot=True, fmt=".3f", cmap="RdYlGn_r",
        cbar_kws={"label": "DM p-value"}, vmin=0, vmax=0.2,
    )
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Saved %s", out_path)


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg["output"]["results_dir"])
    fig_dir = Path(cfg["output"]["figures_dir"])

    metrics, preds = load_results(results_dir)
    logger.info("Loaded %d metric rows, %d prediction rows", len(metrics), len(preds))

    # 1. Summary tables
    summary = summary_table(metrics)
    summary.to_csv(results_dir / "summary_table.csv")
    logger.info("Saved summary table")
    print("\n=== SUMMARY (mean +/- std across splits) ===")
    print(summary)

    # 2. Mean ranks
    for m in ["rmse", "mae"]:
        ranks = rank_models(metrics, metric=m)
        ranks.to_csv(results_dir / f"ranks_{m}.csv", index=False)
        logger.info("Saved ranks for %s", m)

    # 3. Plots
    for m in ["rmse", "mae", "dir_acc"]:
        plot_metric_by_model(metrics, metric=m, out_dir=fig_dir)

    # 4. DM tests
    for (target, horizon), _ in metrics.groupby(["target", "horizon"]):
        pmat = dm_tests_per_pair(preds, target, horizon)
        if pmat.empty:
            continue
        pmat.to_csv(results_dir / f"dm_pvalues_{target}_h{horizon}.csv")
        plot_dm_heatmap(
            pmat,
            f"Diebold-Mariano p-values — {target}, h={horizon}",
            fig_dir / f"dm_heatmap_{target}_h{horizon}.png",
        )


if __name__ == "__main__":
    main()
