"""Tests for first_watch_model.baseline -- PopularityBaseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from first_watch_model.baseline import PopularityBaseline


# ---------------------------------------------------------------------------
# Fixtures: synthetic popularity data
# ---------------------------------------------------------------------------


def _make_popularity_df() -> pd.DataFrame:
    """Create a synthetic title-popularity DataFrame for two services."""
    rows = []
    # Netflix titles
    netflix_titles = [
        ("Squid Game", 0.12),
        ("Wednesday", 0.10),
        ("Stranger Things", 0.08),
        ("The Night Agent", 0.07),
        ("You", 0.06),
        ("Outer Banks", 0.05),
        ("Love Is Blind", 0.04),
        ("Ginny & Georgia", 0.03),
    ]
    # Intentionally do NOT sum to 1.0 so normalization is tested
    for title, prob in netflix_titles:
        rows.append(
            {
                "service": "Netflix",
                "first_watch_title": title,
                "empirical_probability": prob,
            }
        )

    # Disney+ titles
    disney_titles = [
        ("Bluey", 0.15),
        ("Moana", 0.10),
        ("The Mandalorian", 0.09),
        ("Percy Jackson", 0.08),
        ("Inside Out 2", 0.07),
    ]
    for title, prob in disney_titles:
        rows.append(
            {
                "service": "Disney+",
                "first_watch_title": title,
                "empirical_probability": prob,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPopularityBaseline:
    """Tests for PopularityBaseline."""

    def test_fit_stores_distributions(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        assert "Netflix" in baseline.services
        assert "Disney+" in baseline.services

    def test_probabilities_sum_to_one(self) -> None:
        """After fitting, each service distribution should sum to 1.0."""
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        for service in baseline.services:
            dist = baseline.predict(service)
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-9, (
                f"{service}: probabilities sum to {total}, expected 1.0"
            )

    def test_predict_returns_all_titles(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        netflix_dist = baseline.predict("Netflix")
        assert len(netflix_dist) == 8

    def test_predict_top_k_sorted_descending(self) -> None:
        """predict_top_k should return titles sorted by probability."""
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        top_5 = baseline.predict_top_k("Netflix", k=5)
        probs = [p for _, p in top_5]

        assert probs == sorted(probs, reverse=True)

    def test_predict_top_k_respects_k(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        top_3 = baseline.predict_top_k("Netflix", k=3)
        assert len(top_3) == 3

    def test_predict_top_k_returns_highest_first(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        top_1 = baseline.predict_top_k("Netflix", k=1)
        # Squid Game has the highest probability in our fixture
        assert top_1[0][0] == "Squid Game"

    def test_predict_with_n_titles_renormalizes(self) -> None:
        """predict(n_titles=...) should re-normalize the truncated distribution."""
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        top = baseline.predict("Netflix", n_titles=3)
        total = sum(top.values())
        assert abs(total - 1.0) < 1e-9

    def test_predict_unknown_service_returns_empty(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        dist = baseline.predict("NonexistentService")
        assert dist == {}

    def test_predict_top_k_unknown_service_returns_empty(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        result = baseline.predict_top_k("NonexistentService", k=5)
        assert result == []

    def test_all_probabilities_positive(self) -> None:
        df = _make_popularity_df()
        baseline = PopularityBaseline().fit(df)

        for service in baseline.services:
            dist = baseline.predict(service)
            for title, prob in dist.items():
                assert prob >= 0.0, f"{service}/{title}: negative probability {prob}"
