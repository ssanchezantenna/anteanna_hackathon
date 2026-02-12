"""Popularity-based baseline predictor.

The baseline returns the empirical P(title | service) distribution
derived from the title-popularity mart.  Every subscriber of a given
service (and month) receives the identical distribution -- no
personalization.

Supports time-windowed distributions: when a ``signup_month`` is
provided the baseline returns the monthly distribution for that month,
falling back to the global (all-time) distribution when the month is
unknown.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class PopularityBaseline:
    """Service-level popularity baseline with optional monthly granularity.

    Attributes:
        global_distributions: ``{service: {title: probability}}`` mapping
            aggregated over all months.
        monthly_distributions: ``{service: {month: {title: probability}}}``
            mapping at monthly granularity.
    """

    def __init__(self) -> None:
        self.global_distributions: dict[str, dict[str, float]] = {}
        self.monthly_distributions: dict[str, dict[int, dict[str, float]]] = {}

    @staticmethod
    def _normalize(dist: dict[str, float]) -> dict[str, float]:
        total = sum(dist.values())
        if total > 0:
            return {t: p / total for t, p in dist.items()}
        return dist

    def fit(self, popularity_df: pd.DataFrame) -> "PopularityBaseline":
        """Fit the baseline from the title popularity table.

        The DataFrame may contain both monthly rows (``signup_month``
        is not null) and global rows (``signup_month`` is null).  If
        ``signup_month`` is not present at all the entire table is
        treated as global.

        Args:
            popularity_df: DataFrame with columns ``service``,
                ``first_watch_title``, ``empirical_probability``, and
                optionally ``signup_month``.

        Returns:
            Self, for method chaining.
        """
        self.global_distributions = {}
        self.monthly_distributions = {}

        has_month = "signup_month" in popularity_df.columns

        for service, svc_group in popularity_df.groupby("service"):
            service = str(service)

            if has_month:
                # --- monthly rows ------------------------------------------------
                monthly = svc_group[svc_group["signup_month"].notna()]
                for month, mgroup in monthly.groupby("signup_month"):
                    month_int = int(month)
                    dist: dict[str, float] = {}
                    for _, row in mgroup.iterrows():
                        dist[str(row["first_watch_title"])] = float(
                            row["empirical_probability"]
                        )
                    self.monthly_distributions.setdefault(service, {})[
                        month_int
                    ] = self._normalize(dist)

                # --- global rows (signup_month IS NULL) --------------------------
                global_rows = svc_group[svc_group["signup_month"].isna()]
                if not global_rows.empty:
                    gdist: dict[str, float] = {}
                    for _, row in global_rows.iterrows():
                        gdist[str(row["first_watch_title"])] = float(
                            row["empirical_probability"]
                        )
                    self.global_distributions[service] = self._normalize(gdist)
                else:
                    # No explicit global rows -- aggregate monthly counts
                    self.global_distributions[service] = (
                        self._build_global_from_monthly(service)
                    )
            else:
                # No month column at all -- everything is global
                dist = {}
                for _, row in svc_group.iterrows():
                    dist[str(row["first_watch_title"])] = float(
                        row["empirical_probability"]
                    )
                self.global_distributions[service] = self._normalize(dist)

        return self

    def _build_global_from_monthly(self, service: str) -> dict[str, float]:
        """Aggregate monthly distributions into a single global one."""
        combined: dict[str, float] = {}
        for _month, dist in self.monthly_distributions.get(service, {}).items():
            for title, prob in dist.items():
                combined[title] = combined.get(title, 0.0) + prob
        return self._normalize(combined)

    def predict(
        self,
        service: str,
        n_titles: int | None = None,
        month: int | None = None,
    ) -> dict[str, float]:
        """Return the popularity distribution for a service.

        Args:
            service: Name of the streaming service.
            n_titles: If provided, return only the top-N titles
                (re-normalized).
            month: Signup month (1--12).  When provided, the monthly
                distribution is used if available; otherwise falls back
                to the global distribution.

        Returns:
            Dictionary mapping title names to probabilities.
        """
        # Try monthly first, then global
        dist: dict[str, float] = {}
        if month is not None:
            dist = (
                self.monthly_distributions
                .get(service, {})
                .get(month, {})
            )
        if not dist:
            dist = self.global_distributions.get(service, {})

        if not dist:
            return {}

        if n_titles is not None:
            sorted_items = sorted(
                dist.items(), key=lambda x: x[1], reverse=True
            )
            top = dict(sorted_items[:n_titles])
            total = sum(top.values())
            if total > 0:
                top = {t: p / total for t, p in top.items()}
            return top

        return dict(dist)

    def predict_top_k(
        self,
        service: str,
        k: int = 10,
        month: int | None = None,
    ) -> list[tuple[str, float]]:
        """Return the top-k titles by popularity for a service.

        Args:
            service: Name of the streaming service.
            k: Number of top titles to return.
            month: Optional signup month for time-aware lookup.

        Returns:
            List of ``(title, probability)`` tuples sorted descending.
        """
        dist = self.predict(service, month=month)
        sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]

    @property
    def services(self) -> list[str]:
        """Return the list of services this baseline knows about."""
        return list(self.global_distributions.keys())
