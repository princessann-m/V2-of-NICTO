"""Training pipelines for each model type."""

from __future__ import annotations

from typing import Any, Callable, Iterator


def get_pipeline(model_type: str) -> Callable:
    mapping = {
        "mamba": mamba_pipeline,
        "encoder_decoder": encoder_decoder_pipeline,
        "gan": gan_pipeline,
        "simulation": simulation_pipeline,
        "virtual_lab": virtual_lab_pipeline,
    }
    if model_type not in mapping:
        raise ValueError(f"Unknown pipeline: {model_type}")
    return mapping[model_type]


def mamba_pipeline(
    model: Any,
    train_loader: Iterator,
    val_loader: Iterator | None = None,
    **kwargs: Any,
) -> Any:
    return {"model": model, "loader": train_loader, "val_loader": val_loader}


def encoder_decoder_pipeline(
    model: Any,
    train_loader: Iterator,
    val_loader: Iterator | None = None,
    **kwargs: Any,
) -> Any:
    return {"model": model, "loader": train_loader, "val_loader": val_loader}


def gan_pipeline(
    model: Any,
    train_loader: Iterator,
    val_loader: Iterator | None = None,
    **kwargs: Any,
) -> Any:
    return {"model": model, "loader": train_loader, "val_loader": val_loader}


def simulation_pipeline(
    model: Any,
    train_loader: Iterator,
    val_loader: Iterator | None = None,
    **kwargs: Any,
) -> Any:
    return {"model": model, "loader": train_loader, "val_loader": val_loader}


def virtual_lab_pipeline(
    model: Any,
    train_loader: Iterator,
    val_loader: Iterator | None = None,
    **kwargs: Any,
) -> Any:
    return {"model": model, "loader": train_loader, "val_loader": val_loader}
