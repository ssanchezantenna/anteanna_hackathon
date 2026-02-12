"""Training pipeline for the First Watch prediction system.

Orchestrates the end-to-end flow:

1. Load training data from BigQuery (``split='train'``).
2. Load the title popularity table.
3. Fit :class:`TitleEncoder` per service.
4. Prepare the feature matrix (fit categorical encoders).
5. Train a :class:`PopularityBaseline`.
6. Train one :class:`ServiceClassifier` per service (skip if fewer than
   ``MIN_SAMPLES_PER_SERVICE`` examples).
7. Assemble an :class:`EnsemblePredictor`.
8. Persist all artifacts to ``models/`` via joblib.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from first_watch_model.baseline import PopularityBaseline
from first_watch_model.classifier import ServiceClassifier
from first_watch_model.config import (
    LGBM_PARAMS,
    MAX_TITLES_PER_SERVICE,
    MIN_SAMPLES_PER_SERVICE,
    MODEL_DIR,
    TARGET_SERVICES,
)
from first_watch_model.data_loader import load_title_popularity, load_training_data
from first_watch_model.ensemble import EnsemblePredictor
from first_watch_model.features import TitleEncoder, prepare_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full training pipeline."""

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading training data from BigQuery (split='train')...")
    train_df = load_training_data(split="train")
    logger.info("Training rows: %d", len(train_df))

    logger.info("Loading title popularity table...")
    popularity_df = load_title_popularity()
    logger.info("Popularity rows: %d", len(popularity_df))

    # ------------------------------------------------------------------
    # 2. Fit title encoders
    # ------------------------------------------------------------------
    logger.info(
        "Fitting TitleEncoder (max_titles=%d)...", MAX_TITLES_PER_SERVICE
    )
    title_encoder = TitleEncoder(max_titles=MAX_TITLES_PER_SERVICE)
    title_encoder.fit(train_df)

    for svc in title_encoder.services:
        logger.info("  %s: %d classes", svc, title_encoder.num_classes(svc))

    # ------------------------------------------------------------------
    # 3. Prepare features (fit categorical encoders)
    # ------------------------------------------------------------------
    logger.info("Preparing feature matrix...")
    X_all, feature_encoders = prepare_features(train_df, fit=True)
    logger.info("Feature matrix shape: %s", X_all.shape)

    # ------------------------------------------------------------------
    # 4. Train popularity baseline
    # ------------------------------------------------------------------
    logger.info("Training PopularityBaseline...")
    baseline = PopularityBaseline()
    baseline.fit(popularity_df)

    for svc in baseline.services:
        top = baseline.predict_top_k(svc, k=3)
        top_str = ", ".join(f"{t} ({p:.3f})" for t, p in top)
        logger.info("  %s top-3: %s", svc, top_str)

    # ------------------------------------------------------------------
    # 5. Train one ServiceClassifier per service
    # ------------------------------------------------------------------
    logger.info("Training per-service classifiers...")
    classifiers: dict[str, ServiceClassifier] = {}

    for service in TARGET_SERVICES:
        svc_mask = train_df["service"] == service
        n_samples = int(svc_mask.sum())

        if n_samples < MIN_SAMPLES_PER_SERVICE:
            logger.warning(
                "Skipping %s: only %d samples (min=%d)",
                service,
                n_samples,
                MIN_SAMPLES_PER_SERVICE,
            )
            continue

        logger.info(
            "Training classifier for %s (%d samples)...",
            service,
            n_samples,
        )

        # Encode targets
        y_encoded = title_encoder.encode(
            service, train_df.loc[svc_mask, "first_watch_title"]
        )

        # Extract corresponding feature rows
        X_svc = X_all[svc_mask.values]

        clf = ServiceClassifier()
        clf.fit(X_svc, y_encoded, params=LGBM_PARAMS)

        classifiers[service] = clf
        logger.info("  Done. Classes: %d", clf.num_classes_)

    # ------------------------------------------------------------------
    # 6. Build ensemble
    # ------------------------------------------------------------------
    logger.info("Building EnsemblePredictor...")
    ensemble = EnsemblePredictor(
        classifiers=classifiers,
        baseline=baseline,
        title_encoder=title_encoder,
        feature_encoders=feature_encoders,
    )

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    artifacts: dict[str, object] = {
        "ensemble": ensemble,
        "title_encoder": title_encoder,
        "feature_encoders": feature_encoders,
        "baseline": baseline,
        "classifiers": classifiers,
    }

    for name, obj in artifacts.items():
        path = model_dir / f"{name}.joblib"
        joblib.dump(obj, path)
        logger.info("Saved %s -> %s", name, path)

    logger.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
