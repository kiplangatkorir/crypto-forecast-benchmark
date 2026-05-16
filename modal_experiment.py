"""Modal GPU runner for the crypto forecasting benchmark.

Examples:
    python -m modal run modal_experiment.py --models lstm --assets BTC-USD --max-splits 1 --model-max-steps 5 --run-name smoke-lstm
    python -m modal run modal_experiment.py --models lstm,nbeats,patchtst,tft,chronos --full --max-splits 0 --model-max-steps 0 --run-name neural-full
    python -m modal run modal_experiment.py --models naive,hist_mean,arima,garch,xgboost --full --gpu CPU --run-name classical-full
"""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import modal

APP_DIR = "/root/app"
RESULTS_DIR = "/results"
CACHE_DIR = "/cache"
DATA_CACHE_DIR = f"{APP_DIR}/data_cache"

app = modal.App("crypto-forecast-benchmark")

results_volume = modal.Volume.from_name(
    "crypto-forecast-results", create_if_missing=True
)
data_volume = modal.Volume.from_name("crypto-forecast-data", create_if_missing=True)
model_cache_volume = modal.Volume.from_name(
    "crypto-forecast-model-cache", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .workdir(APP_DIR)
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "TORCH_HOME": f"{CACHE_DIR}/torch",
        }
    )
    .add_local_dir("src", remote_path=f"{APP_DIR}/src")
    .add_local_dir("configs", remote_path=f"{APP_DIR}/configs")
)


def _timestamped_run_name(prefix: str = "modal") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def _clean_csv(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return ",".join(parsed) if parsed else None


def _run_experiment_impl(
    models: str,
    assets: Optional[str],
    targets: Optional[str],
    horizons: Optional[str],
    run_name: str,
    max_splits: Optional[int],
    model_max_steps: Optional[int],
) -> dict[str, Any]:
    import argparse

    import torch
    import yaml

    from src.experiment import apply_runtime_overrides, run_experiment

    run_results_dir = str(Path(RESULTS_DIR) / run_name)
    args = argparse.Namespace(
        config="configs/config.yaml",
        models=_clean_csv(models),
        assets=_clean_csv(assets),
        fast=False,
        targets=_clean_csv(targets),
        horizons=_clean_csv(horizons),
        results_dir=run_results_dir,
        max_splits=max_splits,
        model_max_steps=model_max_steps,
    )

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_runtime_overrides(cfg, args)

    def commit_checkpoint() -> None:
        results_volume.commit()

    metrics_df = run_experiment(cfg, args, on_checkpoint=commit_checkpoint)
    results_volume.commit()
    data_volume.commit()
    model_cache_volume.commit()

    return {
        "run_name": run_name,
        "results_dir": run_results_dir,
        "metrics_rows": int(len(metrics_df)),
        "models": _clean_csv(models),
        "assets": _clean_csv(assets) or ",".join(cfg["data"]["assets"]),
        "targets": ",".join(cfg["targets"]),
        "horizons": ",".join(str(h) for h in cfg["horizons"]),
        "max_splits": cfg["walk_forward"].get("max_splits"),
        "model_max_steps": model_max_steps,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "cuda_device_count": int(torch.cuda.device_count())
        if torch.cuda.is_available()
        else 0,
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 12,
    volumes={
        RESULTS_DIR: results_volume,
        CACHE_DIR: model_cache_volume,
        DATA_CACHE_DIR: data_volume,
    },
)
def run_remote_experiment_cpu(
    models: str,
    assets: Optional[str],
    targets: Optional[str],
    horizons: Optional[str],
    run_name: str,
    max_splits: Optional[int],
    model_max_steps: Optional[int],
) -> dict[str, Any]:
    """Run the existing experiment code on Modal CPU workers."""
    return _run_experiment_impl(
        models=models,
        assets=assets,
        targets=targets,
        horizons=horizons,
        run_name=run_name,
        max_splits=max_splits,
        model_max_steps=model_max_steps,
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=60 * 60 * 12,
    volumes={
        RESULTS_DIR: results_volume,
        CACHE_DIR: model_cache_volume,
        DATA_CACHE_DIR: data_volume,
    },
)
def run_remote_experiment(
    models: str,
    assets: Optional[str],
    targets: Optional[str],
    horizons: Optional[str],
    run_name: str,
    max_splits: Optional[int],
    model_max_steps: Optional[int],
) -> dict[str, Any]:
    """Run the existing experiment code on a Modal L4 GPU."""
    return _run_experiment_impl(
        models=models,
        assets=assets,
        targets=targets,
        horizons=horizons,
        run_name=run_name,
        max_splits=max_splits,
        model_max_steps=model_max_steps,
    )


@app.function(
    image=image,
    gpu="A100",
    cpu=8,
    memory=65536,
    timeout=60 * 60 * 24,
    volumes={
        RESULTS_DIR: results_volume,
        CACHE_DIR: model_cache_volume,
        DATA_CACHE_DIR: data_volume,
    },
)
def run_remote_experiment_a100(
    models: str,
    assets: Optional[str],
    targets: Optional[str],
    horizons: Optional[str],
    run_name: str,
    max_splits: Optional[int],
    model_max_steps: Optional[int],
) -> dict[str, Any]:
    """Run the existing experiment code on one Modal A100 GPU."""
    return _run_experiment_impl(
        models=models,
        assets=assets,
        targets=targets,
        horizons=horizons,
        run_name=run_name,
        max_splits=max_splits,
        model_max_steps=model_max_steps,
    )


@app.function(
    image=image,
    timeout=10 * 60,
    volumes={RESULTS_DIR: results_volume},
)
def zip_results(run_name: str) -> bytes:
    """Return a zip archive of one run from the Modal results volume."""
    run_dir = Path(RESULTS_DIR) / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"No Modal result directory found at {run_dir}")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(RESULTS_DIR))
    return buf.getvalue()


@app.local_entrypoint()
def main(
    models: str = "lstm",
    assets: Optional[str] = "BTC-USD",
    targets: Optional[str] = None,
    horizons: Optional[str] = None,
    run_name: Optional[str] = None,
    full: bool = False,
    max_splits: Optional[int] = 1,
    model_max_steps: Optional[int] = 5,
    download: bool = True,
    gpu: str = "L4",
    parallel_models: bool = False,
    detach: bool = False,
    download_runs: Optional[str] = None,
) -> None:
    """Launch a Modal GPU experiment.

    By default this runs a cheap smoke test: BTC only, one split, five neural
    training steps. Use --full with --model-max-steps 0 for the production run.
    """
    if download_runs:
        archive_dir = Path("results") / "modal_archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        for name in [item.strip() for item in download_runs.split(",") if item.strip()]:
            archive = zip_results.remote(name)
            archive_path = archive_dir / f"{name}.zip"
            archive_path.write_bytes(archive)
            print(f"Downloaded Modal results archive: {archive_path}")
        return

    if full:
        if assets == "BTC-USD":
            assets = None
    if max_splits is not None and max_splits <= 0:
        max_splits = None
    if model_max_steps is not None and model_max_steps <= 0:
        model_max_steps = None

    gpu_choice = gpu.upper()
    if gpu_choice in {"CPU", "NONE", "NO_GPU"}:
        remote_fn = run_remote_experiment_cpu
    elif gpu_choice == "A100":
        remote_fn = run_remote_experiment_a100
    else:
        remote_fn = run_remote_experiment
    model_shards = [
        item.strip() for item in models.split(",") if item.strip()
    ] if parallel_models else [models]

    calls = []
    for model_shard in model_shards:
        base_name = run_name or _timestamped_run_name("modal")
        resolved_run_name = (
            f"{base_name}-{model_shard.lower()}"
            if parallel_models
            else base_name
        )
        call = remote_fn.spawn(
            models=model_shard,
            assets=assets,
            targets=targets,
            horizons=horizons,
            run_name=resolved_run_name,
            max_splits=max_splits,
            model_max_steps=model_max_steps,
        )
        calls.append((resolved_run_name, call))

    if detach:
        summaries = [
            {
                "run_name": name,
                "function_call_id": call.object_id,
                "gpu": gpu,
                "models": name.rsplit("-", 1)[-1] if parallel_models else models,
            }
            for name, call in calls
        ]
        print(json.dumps({"detached": True, "runs": summaries}, indent=2))
        return

    summaries = []
    for _, call in calls:
        summaries.append(call.get())

    print(json.dumps(summaries if parallel_models else summaries[0], indent=2, sort_keys=True))

    if not download:
        return

    archive_dir = Path("results") / "modal_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for summary in summaries:
        archive = zip_results.remote(summary["run_name"])
        archive_path = archive_dir / f"{summary['run_name']}.zip"
        archive_path.write_bytes(archive)
        print(f"Downloaded Modal results archive: {archive_path}")
