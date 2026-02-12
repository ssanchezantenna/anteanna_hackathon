"""Popularity-based baseline predictor.

The baseline returns the empirical P(title | service) distribution
derived from the title-popularity mart. Every subscriber of a given
service receives the identical distribution -- no personalization.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class PopularityBaseline:
    """Service-level popularity baseline.

    Attributes:
        distributions: ``{service: {title: probability}}`` mapping where
            probabilities sum to 1.0 for each service.
    """

    def __init__(self) -> None:
        self.distributions: dict[str, dict[str, float]] = {}

    def fit(self, popularity_df: pd.DataFrame) -> "PopularityBaseline":
        """Fit the baseline from the title popularity table.

        Args:
            popularity_df: DataFrame with columns ``service``,
                ``first_watch_title``, and ``empirical_probability``.

        Returns:
            Self, for method chaining.
        """
        self.distributions = {}

        for service, group in popularity_df.groupby("service"):
            service = str(service)
            dist: dict[str, float] = {}
            total = 0.0
            for _, row in group.iterrows():
                title = str(row["first_watch_title"])
                prob = float(row["empirical_probability"])
                dist[title] = prob
                total += prob

            # Normalize to ensure probabilities sum to 1.0
            if total > 0:
                dist = {t: p / total for t, p in dist.items()}

            self.distributions[service] = dist

        return self

    def predict(
        self, service: str, n_titles: int | None = None
    ) -> dict[str, float]:
        """Return the popularity distribution for a service.

        Args:
            service: Name of the streaming service.
            n_titles: If provided, return only the top-N titles (re-normalized).

        Returns:
            Dictionary mapping title names to probabilities.
        """
        dist = self.distributions.get(service, {})

        if not dist:
            return {}

        if n_titles is not None:
            sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
            top = dict(sorted_items[:n_titles])
            total = sum(top.values())
            if total > 0:
                top = {t: p / total for t, p in top.items()}
            return top

        return dict(dist)

    def predict_top_k(
        self, service: str, k: int = 10
    ) -> list[tuple[str, float]]:
        """Return the top-k titles by popularity for a service.

        Args:
            service: Name of the streaming service.
            k: Number of top titles to return.

        Returns:
            List of ``(title, probability)`` tuples sorted descending.
        """
        dist = self.distributions.get(service, {})
        sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    @property
    def services(self) -> list[str]:
        """Return the list of services this baseline knows about."""
        return list(self.distributions.keys())
