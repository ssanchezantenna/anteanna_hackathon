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
from first_watch_model.features import UNKNOWN_CATEGORY, derive_signup_date_features

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
            If a ``signup_date`` column is present, temporal features
            (``signup_month``, ``signup_day_of_week``,
            ``signup_week_of_year``) are derived automatically and the
            month-specific popularity baseline is used.
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

    # Derive temporal features from signup_date when available
    work = derive_signup_date_features(work)

    work = _fill_defaults(work)

    has_subscriber_id = "subscriber_id" in work.columns

    # Determine signup_month per row (0 means unknown)
    has_signup_month = "signup_month" in work.columns

    all_results: list[pd.DataFrame] = []

    # Group by (service, signup_month) so each group gets its own
    # time-specific popularity baseline.
    if has_signup_month:
        group_cols = ["service", "signup_month"]
    else:
        group_cols = ["service"]

    for group_key, group in work.groupby(group_cols):
        if has_signup_month:
            service, signup_month_val = str(group_key[0]), int(group_key[1])
            signup_month = signup_month_val if signup_month_val > 0 else None
        else:
            service = str(group_key) if isinstance(group_key, str) else str(group_key[0])
            signup_month = None

        features_df = group.reset_index(drop=True)

        preds = model.predict(
            service=service,
            features_df=features_df,
            alpha=alpha,
            temperature=temperature,
            top_k=top_k,
            signup_month=signup_month,
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
