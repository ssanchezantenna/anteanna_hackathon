"""LightGBM-based multi-class classifier with temperature scaling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import lightgbm as lgb
import numpy as np

from first_watch_model.config import LGBM_PARAMS


class ServiceClassifier:
    """Per-service multi-class title classifier.

    Wraps a LightGBM classifier that predicts the probability of each
    title being the first watch for subscribers of one specific streaming
    service.

    Attributes:
        model: The underlying LightGBM classifier (set after ``fit``).
        num_classes_: Number of output classes (set after ``fit``).
    """

    def __init__(self) -> None:
        self.model: lgb.LGBMClassifier | None = None
        self.num_classes_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> "ServiceClassifier":
        """Train the multi-class classifier.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y_encoded: Integer-encoded target labels.
            params: LightGBM parameters.  Falls back to
                ``config.LGBM_PARAMS`` when *None*.

        Returns:
            Self, for method chaining.
        """
        if params is None:
            params = deepcopy(LGBM_PARAMS)
        else:
            params = deepcopy(params)

        self.num_classes_ = int(np.max(y_encoded) + 1)
        params["num_class"] = self.num_classes_

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(X, y_encoded)
        return self

    def predict_proba(
        self, X: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        """Predict class probabilities with optional temperature scaling.

        Temperature scaling adjusts the sharpness of the output
        distribution:

        * ``T = 1.0`` -- raw model output (no change).
        * ``T > 1.0`` -- flatter / more uniform distribution.
        * ``T < 1.0`` -- sharper / more peaked distribution.

        The formula is::

            logits = log(raw_proba + eps)
            P_scaled(i) = softmax(logits / T)

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            temperature: Temperature parameter for scaling.

        Returns:
            Probability matrix of shape ``(n_samples, num_classes_)``.
        """
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted yet.")

        raw_proba: np.ndarray = self.model.predict_proba(X)

        if temperature == 1.0:
            return raw_proba

        # Convert probabilities to log-space, apply temperature, then
        # re-normalize via softmax.
        eps = 1e-12
        logits = np.log(raw_proba + eps)
        scaled_logits = logits / temperature

        # Numerically stable softmax
        shifted = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return proba

    @property
    def classes_(self) -> np.ndarray:
        """Return the class labels known to the fitted model."""
        if self.model is None:
            raise RuntimeError("Classifier has not been fitted yet.")
        return self.model.classes_

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Return LightGBM feature importances, or None if not fitted."""
        if self.model is None:
            return None
        return self.model.feature_importances_
