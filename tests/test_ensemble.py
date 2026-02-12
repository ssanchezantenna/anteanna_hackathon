"""Tests for first_watch_model.ensemble -- EnsemblePredictor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from first_watch_model.baseline import PopularityBaseline
from first_watch_model.classifier import ServiceClassifier
from first_watch_model.ensemble import EnsemblePredictor
from first_watch_model.features import OTHER_BUCKET, TitleEncoder


# ---------------------------------------------------------------------------
# Helpers: build components for testing
# ---------------------------------------------------------------------------

SERVICE = "Netflix"
TITLES = ["Squid Game", "Wednesday", "Stranger Things", "The Night Agent"]


def _build_title_encoder(service: str, titles: list[str]) -> TitleEncoder:
    """Fit a TitleEncoder on a minimal DataFrame for one service."""
    n = len(titles) * 10
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "service": [service] * n,
            "first_watch_title": rng.choice(titles, size=n),
        }
    )
    enc = TitleEncoder(max_titles=len(titles))
    enc.fit(df)
    return enc


def _build_baseline(service: str, titles: list[str]) -> PopularityBaseline:
    """Build a PopularityBaseline with uniform distribution over titles."""
    n = len(titles)
    prob = 1.0 / n
    df = pd.DataFrame(
        {
            "service": [service] * n,
            "first_watch_title": titles,
            "empirical_probability": [prob] * n,
        }
    )
    baseline = PopularityBaseline()
    baseline.fit(df)
    return baseline


def _mock_classifier_uniform(n_classes: int) -> MagicMock:
    """Mock classifier returning a uniform distribution."""
    clf = MagicMock(spec=ServiceClassifier)
    uniform_row = np.ones(n_classes) / n_classes

    def fake_predict_proba(X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        n = X.shape[0]
        return np.tile(uniform_row, (n, 1))

    clf.predict_proba = MagicMock(side_effect=fake_predict_proba)
    clf.num_classes_ = n_classes
    return clf


def _mock_classifier_peaked(n_classes: int, peak_idx: int = 0) -> MagicMock:
    """Mock classifier that puts all mass on one class."""
    clf = MagicMock(spec=ServiceClassifier)

    def fake_predict_proba(X: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        n = X.shape[0]
        proba = np.zeros((n, n_classes))
        proba[:, peak_idx] = 1.0
        return proba

    clf.predict_proba = MagicMock(side_effect=fake_predict_proba)
    clf.num_classes_ = n_classes
    return clf


def _make_feature_df(n: int = 3) -> pd.DataFrame:
    """Minimal feature DataFrame matching config.FEATURE_COLUMNS."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "service": [SERVICE] * n,
            "has_netflix": rng.randint(0, 2, n),
            "has_amazon": rng.randint(0, 2, n),
            "has_disney": rng.randint(0, 2, n),
            "has_max": rng.randint(0, 2, n),
            "has_apple_tv": rng.randint(0, 2, n),
            "has_hulu": rng.randint(0, 2, n),
            "has_peacock": rng.randint(0, 2, n),
            "has_paramount": rng.randint(0, 2, n),
            "has_linear": rng.randint(0, 2, n),
            "active_days_30d": rng.randint(1, 30, n),
            "total_duration_30d": rng.randint(1000, 80000, n),
            "distinct_titles_30d": rng.randint(1, 30, n),
            "total_sessions_30d": rng.randint(1, 80, n),
            "distinct_services_30d": rng.randint(1, 6, n),
            "avg_hour_of_day": rng.uniform(8, 23, n),
            "stddev_hour_of_day": rng.uniform(0.5, 5, n),
            "weekend_ratio": rng.uniform(0, 1, n),
            "primetime_ratio": rng.uniform(0, 1, n),
            "dominant_content_type": rng.choice(["series", "movie"], size=n),
            "avg_session_duration": rng.uniform(300, 3600, n),
            "city": rng.choice(["New York", "LA"], size=n),
        }
    )


def _build_feature_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build ordinal encoders using prepare_features in fit mode."""
    from first_watch_model.features import prepare_features

    _, encoders = prepare_features(df, fit=True)
    return encoders


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""

    def test_predict_returns_dataframe(self) -> None:
        """predict() should return a DataFrame with title, probability, rank."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=2)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        clf = _mock_classifier_uniform(n_classes)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, top_k=5)

        assert isinstance(result, pd.DataFrame)
        assert "title" in result.columns
        assert "probability" in result.columns
        assert "rank" in result.columns

    def test_predict_probabilities_are_valid(self) -> None:
        """All probabilities in predict output should be non-negative."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=2)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        clf = _mock_classifier_uniform(n_classes)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, top_k=5)
        assert (result["probability"] >= 0).all()

    def test_baseline_fallback_when_no_classifier(self) -> None:
        """Without a classifier, predict should still return results using baseline."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=2)
        feature_encoders = _build_feature_encoders(feature_df)

        ensemble = EnsemblePredictor(
            classifiers={},  # No classifiers
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, top_k=3)

        assert len(result) > 0
        assert (result["probability"] >= 0).all()

    def test_baseline_fallback_all_rows_identical(self) -> None:
        """With no classifier, all subscribers should get the same predictions."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=3)
        feature_encoders = _build_feature_encoders(feature_df)

        ensemble = EnsemblePredictor(
            classifiers={},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, top_k=3)

        # Group by row_index and compare predictions
        if "row_index" in result.columns:
            groups = [grp for _, grp in result.groupby("row_index")]
            if len(groups) > 1:
                titles_0 = groups[0]["title"].tolist()
                probs_0 = groups[0]["probability"].tolist()
                for grp in groups[1:]:
                    assert grp["title"].tolist() == titles_0
                    np.testing.assert_allclose(
                        grp["probability"].tolist(), probs_0, atol=1e-9
                    )

    def test_alpha_zero_gives_pure_classifier(self) -> None:
        """With alpha=0, the output should come entirely from the classifier."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=1)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        clf = _mock_classifier_peaked(n_classes, peak_idx=0)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, alpha=0.0, top_k=n_classes)

        # With alpha=0, rank 1 should have nearly all the mass
        top_row = result[result["rank"] == 1].iloc[0]
        assert top_row["probability"] > 0.9

    def test_alpha_one_gives_pure_baseline(self) -> None:
        """With alpha=1, the output should come entirely from the baseline."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=1)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        # Classifier puts all mass on class 0
        clf = _mock_classifier_peaked(n_classes, peak_idx=0)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, alpha=1.0, top_k=n_classes)

        # With alpha=1.0, the peaked classifier should be completely ignored.
        # The baseline is roughly uniform, so the top prediction should NOT
        # dominate.
        top_row = result[result["rank"] == 1].iloc[0]
        assert top_row["probability"] < 0.9, (
            "alpha=1.0 should ignore classifier; top title should not dominate"
        )

    def test_blending_intermediate_alpha(self) -> None:
        """With 0 < alpha < 1, output should be a mix of classifier and baseline."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=1)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        clf = _mock_classifier_peaked(n_classes, peak_idx=0)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, alpha=0.5, top_k=n_classes)

        top_row = result[result["rank"] == 1].iloc[0]
        # Should be between pure classifier (1.0) and pure baseline (~0.2)
        assert 0.2 < top_row["probability"] < 1.0

    def test_top_k_limits_output(self) -> None:
        """predict(top_k=k) should return at most k rows per subscriber."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=1)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        clf = _mock_classifier_uniform(n_classes)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, top_k=3)
        assert len(result) == 3

    def test_ranks_are_sequential(self) -> None:
        """Rank values should start at 1 and increase sequentially."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=1)
        feature_encoders = _build_feature_encoders(feature_df)
        n_classes = enc.num_classes(SERVICE)
        clf = _mock_classifier_uniform(n_classes)

        ensemble = EnsemblePredictor(
            classifiers={SERVICE: clf},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        result = ensemble.predict(SERVICE, feature_df, top_k=4)
        assert result["rank"].tolist() == [1, 2, 3, 4]

    def test_empirical_vector_sums_to_one(self) -> None:
        """The internal empirical vector should be a valid probability distribution."""
        enc = _build_title_encoder(SERVICE, TITLES)
        baseline = _build_baseline(SERVICE, TITLES)
        feature_df = _make_feature_df(n=1)
        feature_encoders = _build_feature_encoders(feature_df)

        ensemble = EnsemblePredictor(
            classifiers={},
            baseline=baseline,
            title_encoder=enc,
            feature_encoders=feature_encoders,
        )

        vec, title_list = ensemble._build_empirical_vector(SERVICE)
        assert abs(vec.sum() - 1.0) < 1e-9
        assert len(title_list) == enc.num_classes(SERVICE)
