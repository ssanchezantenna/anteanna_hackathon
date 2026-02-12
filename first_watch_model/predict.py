"""Inference utilities for the First Watch prediction system.

Provides a high-level API that loads a saved ensemble model and generates
title-probability predictions from a subscriber DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from first_watch_model.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MODEL_DIR,
    SMOOTHING_ALPHA,
    TEMPERATURE,
)
from first_watch_model.ensemble import EnsemblePredictor
from first_watch_model.features import UNKNOWN_CATEGORY

logger = logging.getLogger(__name__)


def load_model(model_dir: str | Path | None = None) -> EnsemblePredictor:
    """Load a saved :class:`EnsemblePredictor` from disk.

    Args:
        model_dir: Directory containing ``ensemble.joblib``.  Defaults to
            ``config.MODEL_DIR``.

    Returns:
        Loaded EnsemblePredictor ready for inference.
    """
    directory = Path(model_dir) if model_dir is not None else Path(MODEL_DIR)
    path = directory / "ensemble.joblib"
    logger.info("Loading ensemble model from %s", path)
    ensemble: EnsemblePredictor = joblib.load(path)
    return ensemble


def _fill_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature columns with safe defaults.

    Numeric features default to 0.  Categorical features default to
    ``__UNKNOWN__``.

    Args:
        df: Input DataFrame (modified in place).

    Returns:
        The same DataFrame with defaults applied.
    """
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = UNKNOWN_CATEGORY
        else:
            df[col] = df[col].fillna(UNKNOWN_CATEGORY).astype(str)

    return df


def predict_from_df(
    df: pd.DataFrame,
    model: EnsemblePredictor | None = None,
    top_k: int = 10,
    alpha: float = SMOOTHING_ALPHA,
    temperature: float = TEMPERATURE,
) -> pd.DataFrame:
    """Generate first-watch predictions for a subscriber DataFrame.

    Args:
        df: Input DataFrame.  Must contain a ``service`` column.  Feature
            columns are optional -- missing ones are filled with defaults.
            If a ``subscriber_id`` column is present it is propagated to
            the output; otherwise positional indices are used.
        model: Pre-loaded :class:`EnsemblePredictor`.  When *None* the
            model is loaded from disk automatically.
        top_k: Number of top predictions per subscriber.
        alpha: Blending weight for the empirical baseline.
        temperature: Temperature scaling for the classifier.

    Returns:
        DataFrame with columns ``[subscriber_id, service, rank, title,
        probability]``.
    """
    if model is None:
        model = load_model()

    work = df.copy()
    work = _fill_defaults(work)

    has_subscriber_id = "subscriber_id" in work.columns

    all_results: list[pd.DataFrame] = []

    for service, group in work.groupby("service"):
        service = str(service)
        features_df = group.reset_index(drop=True)

        preds = model.predict(
            service=service,
            features_df=features_df,
            alpha=alpha,
            temperature=temperature,
            top_k=top_k,
        )

        # Ensure row_index column exists for the join below
        if "row_index" not in preds.columns:
            preds["row_index"] = 0

        preds["service"] = service

        if has_subscriber_id:
            row_ids = features_df["subscriber_id"].values
            preds["subscriber_id"] = preds["row_index"].map(
                lambda idx, ids=row_ids: ids[idx] if idx < len(ids) else idx
            )
        else:
            preds["subscriber_id"] = preds["row_index"]

        all_results.append(preds)

    if not all_results:
        return pd.DataFrame(
            columns=["subscriber_id", "service", "rank", "title", "probability"]
        )

    result = pd.concat(all_results, ignore_index=True)
    result = result[["subscriber_id", "service", "rank", "title", "probability"]]
    result = result.sort_values(
        ["subscriber_id", "service", "rank"]
    ).reset_index(drop=True)

    return result
