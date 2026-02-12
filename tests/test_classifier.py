"""Tests for first_watch_model.classifier -- ServiceClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from first_watch_model.classifier import ServiceClassifier


# ---------------------------------------------------------------------------
# Fixtures: small synthetic classification problem
# ---------------------------------------------------------------------------


def _make_classification_data(
    n_samples: int = 300,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small multi-class classification dataset.

    Features are random floats; labels are drawn uniformly from
    ``[0, n_classes)``.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randint(0, n_classes, size=n_samples).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestServiceClassifier:
    """Tests for ServiceClassifier."""

    def test_predict_proba_shape(self) -> None:
        """predict_proba output should have shape (n_samples, n_classes)."""
        X, y = _make_classification_data(n_samples=200, n_classes=5)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 5,
            "n_estimators": 10,
            "verbose": -1,
        })

        proba = clf.predict_proba(X)
        assert proba.shape == (200, 5)

    def test_probabilities_sum_to_one(self) -> None:
        """Each row of predict_proba should sum to 1.0."""
        X, y = _make_classification_data(n_samples=100, n_classes=4)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 4,
            "n_estimators": 10,
            "verbose": -1,
        })

        proba = clf.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_temperature_scaling_high_flattens(self) -> None:
        """T > 1 should produce a flatter (more uniform) distribution."""
        X, y = _make_classification_data(n_samples=100, n_classes=4)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 4,
            "n_estimators": 20,
            "verbose": -1,
        })

        proba_raw = clf.predict_proba(X, temperature=1.0)
        proba_flat = clf.predict_proba(X, temperature=5.0)

        # Variance across classes should be smaller for T > 1 (flatter)
        var_raw = np.var(proba_raw, axis=1).mean()
        var_flat = np.var(proba_flat, axis=1).mean()

        assert var_flat < var_raw, (
            f"High temperature should flatten distribution: "
            f"var_flat={var_flat:.6f} >= var_raw={var_raw:.6f}"
        )

    def test_temperature_scaling_low_sharpens(self) -> None:
        """T < 1 should produce a sharper (more peaked) distribution."""
        X, y = _make_classification_data(n_samples=100, n_classes=4)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 4,
            "n_estimators": 20,
            "verbose": -1,
        })

        proba_raw = clf.predict_proba(X, temperature=1.0)
        proba_sharp = clf.predict_proba(X, temperature=0.3)

        # Variance across classes should be larger for T < 1 (sharper)
        var_raw = np.var(proba_raw, axis=1).mean()
        var_sharp = np.var(proba_sharp, axis=1).mean()

        assert var_sharp > var_raw, (
            f"Low temperature should sharpen distribution: "
            f"var_sharp={var_sharp:.6f} <= var_raw={var_raw:.6f}"
        )

    def test_temperature_one_is_identity(self) -> None:
        """Temperature=1.0 should return the raw model output."""
        X, y = _make_classification_data(n_samples=50, n_classes=3)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 10,
            "verbose": -1,
        })

        proba_raw = clf.predict_proba(X, temperature=1.0)
        proba_default = clf.predict_proba(X)  # default is also T=1.0

        np.testing.assert_allclose(proba_raw, proba_default, atol=1e-10)

    def test_scaled_probabilities_still_sum_to_one(self) -> None:
        """After temperature scaling, rows should still sum to 1.0."""
        X, y = _make_classification_data(n_samples=50, n_classes=4)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 4,
            "n_estimators": 10,
            "verbose": -1,
        })

        for temp in [0.1, 0.5, 1.0, 2.0, 10.0]:
            proba = clf.predict_proba(X, temperature=temp)
            row_sums = proba.sum(axis=1)
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-6,
                err_msg=f"Rows do not sum to 1.0 for temperature={temp}",
            )

    def test_classes_property(self) -> None:
        X, y = _make_classification_data(n_samples=100, n_classes=5)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 5,
            "n_estimators": 10,
            "verbose": -1,
        })

        classes = clf.classes_
        assert len(classes) == 5

    def test_predict_proba_before_fit_raises(self) -> None:
        clf = ServiceClassifier()
        X = np.random.randn(5, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(X)

    def test_all_probabilities_non_negative(self) -> None:
        """All predicted probabilities should be >= 0."""
        X, y = _make_classification_data(n_samples=100, n_classes=4)
        clf = ServiceClassifier()
        clf.fit(X, y, params={
            "objective": "multiclass",
            "num_class": 4,
            "n_estimators": 10,
            "verbose": -1,
        })

        for temp in [0.5, 1.0, 3.0]:
            proba = clf.predict_proba(X, temperature=temp)
            assert (proba >= 0).all(), (
                f"Negative probabilities found at temperature={temp}"
            )
