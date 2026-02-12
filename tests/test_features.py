"""Tests for first_watch_model.features -- TitleEncoder and prepare_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from first_watch_model.features import (
    OTHER_BUCKET,
    UNKNOWN_CATEGORY,
    TitleEncoder,
    prepare_features,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic data
# ---------------------------------------------------------------------------


def _make_training_df(n: int = 200) -> pd.DataFrame:
    """Build a synthetic training DataFrame with two services."""
    rng = np.random.RandomState(42)
    services = rng.choice(["Netflix", "Disney+"], size=n)
    titles_netflix = [
        "Squid Game", "Wednesday", "Stranger Things", "The Night Agent",
        "You", "Outer Banks", "Love Is Blind", "Ginny & Georgia",
    ]
    titles_disney = [
        "Bluey", "Moana", "The Mandalorian", "Percy Jackson",
        "Inside Out 2", "Encanto", "Frozen", "Ahsoka",
    ]
    titles = []
    for svc in services:
        pool = titles_netflix if svc == "Netflix" else titles_disney
        titles.append(rng.choice(pool))

    return pd.DataFrame(
        {
            "service": services,
            "first_watch_title": titles,
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
            "dominant_content_type": rng.choice(
                ["series", "movie", "other"], size=n
            ),
            "avg_session_duration": rng.uniform(300, 3600, n),
            "city": rng.choice(
                ["New York", "Los Angeles", "Chicago"], size=n
            ),
        }
    )


# ---------------------------------------------------------------------------
# TitleEncoder tests
# ---------------------------------------------------------------------------


class TestTitleEncoder:
    """Tests for the TitleEncoder class."""

    def test_fit_creates_vocabulary(self) -> None:
        df = _make_training_df()
        enc = TitleEncoder(max_titles=5)
        enc.fit(df)

        assert "Netflix" in enc.services
        assert "Disney+" in enc.services

    def test_other_bucket_in_vocabulary(self) -> None:
        df = _make_training_df()
        enc = TitleEncoder(max_titles=5)
        enc.fit(df)

        for service in enc.services:
            assert OTHER_BUCKET in enc.title_to_idx[service]

    def test_max_titles_limit(self) -> None:
        df = _make_training_df()
        enc = TitleEncoder(max_titles=3)
        enc.fit(df)

        for service in enc.services:
            # 3 titles + __OTHER__ = 4 entries
            assert enc.num_classes(service) == 4

    def test_encode_known_titles(self) -> None:
        df = _make_training_df()
        enc = TitleEncoder(max_titles=10)
        enc.fit(df)

        netflix_titles = pd.Series(["Squid Game", "Wednesday"])
        codes = enc.encode("Netflix", netflix_titles)

        assert codes.shape == (2,)
        assert codes.dtype == np.int32

    def test_encode_unknown_title_maps_to_other(self) -> None:
        df = _make_training_df()
        enc = TitleEncoder(max_titles=10)
        enc.fit(df)

        unknown = pd.Series(["Title That Does Not Exist"])
        codes = enc.encode("Netflix", unknown)
        other_idx = enc.title_to_idx["Netflix"][OTHER_BUCKET]

        assert codes[0] == other_idx

    def test_encode_null_maps_to_other(self) -> None:
        """NaN titles should map to __OTHER__ code."""
        df = _make_training_df()
        enc = TitleEncoder(max_titles=10)
        enc.fit(df)

        with_null = pd.Series([None, np.nan, "Squid Game"])
        codes = enc.encode("Netflix", with_null)
        other_idx = enc.title_to_idx["Netflix"][OTHER_BUCKET]

        # None and NaN should map to other
        assert codes[0] == other_idx
        assert codes[1] == other_idx
        # Known title should NOT map to other
        assert codes[2] != other_idx

    def test_encoding_roundtrip(self) -> None:
        """Encode then decode should recover original titles."""
        df = _make_training_df()
        enc = TitleEncoder(max_titles=10)
        enc.fit(df)

        originals = pd.Series(["Squid Game", "Wednesday"])
        codes = enc.encode("Netflix", originals)
        decoded = enc.decode("Netflix", codes)

        assert decoded == ["Squid Game", "Wednesday"]

    def test_rare_titles_become_other(self) -> None:
        """With max_titles=2, most titles should end up as __OTHER__."""
        df = _make_training_df(n=500)
        enc = TitleEncoder(max_titles=2)
        enc.fit(df)

        # There are 8 Netflix titles in the data but we keep only 2
        all_netflix_titles = df.loc[
            df["service"] == "Netflix", "first_watch_title"
        ]
        codes = enc.encode("Netflix", all_netflix_titles)
        other_idx = enc.title_to_idx["Netflix"][OTHER_BUCKET]

        # At least some should be __OTHER__
        assert (codes == other_idx).sum() > 0


# ---------------------------------------------------------------------------
# prepare_features tests
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    """Tests for the prepare_features function."""

    def test_output_shape(self) -> None:
        df = _make_training_df(n=50)
        matrix, encoders = prepare_features(df, fit=True)

        assert matrix.shape[0] == 50
        assert matrix.ndim == 2
        # Numeric features (FEATURE_COLUMNS) + categorical features
        assert matrix.shape[1] > 0

    def test_output_dtype(self) -> None:
        df = _make_training_df(n=10)
        matrix, _ = prepare_features(df, fit=True)

        assert matrix.dtype == np.float32

    def test_null_handling_numeric(self) -> None:
        """Missing numeric features should be filled with 0."""
        df = _make_training_df(n=10)
        df.loc[0, "active_days_30d"] = np.nan
        df.loc[1, "total_duration_30d"] = np.nan

        matrix, _ = prepare_features(df, fit=True)

        # No NaN values should remain
        assert not np.isnan(matrix).any()

    def test_null_handling_categorical(self) -> None:
        """Missing categorical features should be filled with __UNKNOWN__."""
        df = _make_training_df(n=10)
        df.loc[0, "city"] = None
        df.loc[1, "dominant_content_type"] = np.nan

        matrix, _ = prepare_features(df, fit=True)

        assert not np.isnan(matrix).any()

    def test_missing_columns_filled_with_defaults(self) -> None:
        """Columns absent from the input should be created with defaults."""
        df = pd.DataFrame(
            {
                "service": ["Netflix", "Disney+"],
                "has_netflix": [1, 0],
            }
        )
        matrix, _ = prepare_features(df, fit=True)

        assert matrix.shape[0] == 2
        assert not np.isnan(matrix).any()

    def test_reuse_encoders(self) -> None:
        """Using fit=False with pre-fitted encoders should work."""
        df = _make_training_df(n=50)
        _, encoders = prepare_features(df, fit=True)

        df2 = _make_training_df(n=10)
        matrix2, encoders2 = prepare_features(
            df2, fit=False, encoders=encoders
        )

        assert matrix2.shape[0] == 10
        assert encoders2 is encoders
