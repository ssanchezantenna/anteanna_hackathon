"""Ensemble predictor blending classifier output with popularity baseline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from first_watch_model.baseline import PopularityBaseline
from first_watch_model.classifier import ServiceClassifier
from first_watch_model.config import SMOOTHING_ALPHA, TEMPERATURE
from first_watch_model.features import (
    OTHER_BUCKET,
    TitleEncoder,
    prepare_features,
)

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """Blended ensemble of per-service classifiers and a popularity baseline.

    The final prediction is a weighted mix of the classifier's probability
    distribution and the empirical popularity distribution::

        P_final = (1 - alpha) * P_classifier + alpha * P_empirical

    For services where no classifier was trained (due to insufficient
    data), the predictor falls back to the pure popularity baseline.

    Args:
        classifiers: ``{service: ServiceClassifier}`` mapping.
        baseline: Fitted PopularityBaseline.
        title_encoder: Fitted TitleEncoder.
        feature_encoders: Ordinal-encoding mappings for categorical
            features (as returned by ``features.prepare_features``).
    """

    def __init__(
        self,
        classifiers: dict[str, ServiceClassifier],
        baseline: PopularityBaseline,
        title_encoder: TitleEncoder,
        feature_encoders: dict[str, dict[str, int]],
    ) -> None:
        self.classifiers = classifiers
        self.baseline = baseline
        self.title_encoder = title_encoder
        self.feature_encoders = feature_encoders

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_empirical_vector(
        self, service: str, month: int | None = None
    ) -> tuple[np.ndarray, list[str]]:
        """Build a probability vector aligned to the TitleEncoder classes.

        Args:
            service: Target streaming service.
            month: Optional signup month (1--12) for time-aware lookup.
                When provided the monthly popularity distribution is used
                (falling back to global if the month is unavailable).

        Returns:
            Tuple of ``(prob_vector, title_list)`` where *prob_vector* has
            length equal to the number of classes for *service* and
            *title_list* maps each index to a title name.
        """
        num_classes = self.title_encoder.num_classes(service)
        idx_to_title = self.title_encoder.idx_to_title.get(service, {})
        title_to_idx = self.title_encoder.title_to_idx.get(service, {})

        empirical_dist = self.baseline.predict(service, month=month)

        title_list = [
            idx_to_title.get(i, OTHER_BUCKET) for i in range(num_classes)
        ]
        prob_vector = np.zeros(num_classes, dtype=np.float64)

        for title, prob in empirical_dist.items():
            idx = title_to_idx.get(title)
            if idx is not None:
                prob_vector[idx] = prob

        # Assign residual empirical mass to __OTHER__
        other_idx = title_to_idx.get(OTHER_BUCKET)
        if other_idx is not None:
            assigned = prob_vector.sum()
            if assigned < 1.0:
                prob_vector[other_idx] += 1.0 - assigned

        # Re-normalize
        total = prob_vector.sum()
        if total > 0:
            prob_vector /= total

        return prob_vector, title_list

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def predict(
        self,
        service: str,
        features_df: pd.DataFrame,
        alpha: float = SMOOTHING_ALPHA,
        temperature: float = TEMPERATURE,
        top_k: int = 10,
        signup_month: int | None = None,
    ) -> pd.DataFrame:
        """Generate blended title-probability predictions.

        Args:
            service: Target streaming service.
            features_df: Feature DataFrame for one or more subscribers.
                Must contain the numeric and categorical feature columns
                defined in ``config``.
            alpha: Weight for the empirical baseline (0 = classifier only,
                1 = baseline only).
            temperature: Temperature scaling applied to classifier logits.
            top_k: Number of top titles to return per subscriber.
            signup_month: Optional signup month (1--12).  When provided the
                time-windowed popularity distribution for that month is
                used as the empirical baseline component.

        Returns:
            DataFrame with columns ``[title, probability, rank]`` for each
            subscriber row.  When multiple rows are provided the result
            also contains a ``row_index`` column to disambiguate.
        """
        has_classifier = service in self.classifiers

        # --- popularity vector (shared across all subscribers) ---------------
        empirical_vec, title_list = self._build_empirical_vector(
            service, month=signup_month
        )

        # --- classifier probabilities ----------------------------------------
        if has_classifier:
            X, _ = prepare_features(
                features_df,
                fit=False,
                encoders=self.feature_encoders,
            )
            classifier_proba = self.classifiers[service].predict_proba(
                X, temperature=temperature
            )

            # Align classifier columns to title encoder width
            n_classes_model = classifier_proba.shape[1]
            n_classes_encoder = len(title_list)
            if n_classes_model < n_classes_encoder:
                pad = np.zeros(
                    (classifier_proba.shape[0], n_classes_encoder - n_classes_model),
                    dtype=np.float64,
                )
                classifier_proba = np.hstack([classifier_proba, pad])
            elif n_classes_model > n_classes_encoder:
                classifier_proba = classifier_proba[:, :n_classes_encoder]
        else:
            logger.info(
                "No classifier for %s -- falling back to popularity baseline.",
                service,
            )

        # --- blend and build output ------------------------------------------
        rows: list[dict[str, Any]] = []

        for i in range(len(features_df)):
            if has_classifier:
                clf_vec = classifier_proba[i]
                blended = (1.0 - alpha) * clf_vec + alpha * empirical_vec
            else:
                blended = empirical_vec.copy()

            # Re-normalize
            total = blended.sum()
            if total > 0:
                blended /= total

            # Top-k
            top_indices = np.argsort(blended)[::-1][:top_k]
            for rank, idx in enumerate(top_indices, start=1):
                rows.append(
                    {
                        "row_index": i,
                        "title": (
                            title_list[idx]
                            if idx < len(title_list)
                            else OTHER_BUCKET
                        ),
                        "probability": float(blended[idx]),
                        "rank": rank,
                    }
                )

        result = pd.DataFrame(
            rows, columns=["row_index", "title", "probability", "rank"]
        )

        # Drop row_index when there is only a single subscriber
        if len(features_df) == 1:
            result = result.drop(columns=["row_index"])

        return result
