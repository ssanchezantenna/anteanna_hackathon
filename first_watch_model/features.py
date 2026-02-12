"""Feature engineering: title encoding and feature matrix preparation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from first_watch_model.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    MAX_TITLES_PER_SERVICE,
)

OTHER_BUCKET: str = "__OTHER__"
UNKNOWN_CATEGORY: str = "__UNKNOWN__"


def derive_signup_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive temporal features from a ``signup_date`` column.

    Extracts ``signup_month``, ``signup_day_of_week``, and
    ``signup_week_of_year`` using conventions consistent with BigQuery:

    - ``signup_day_of_week``: 1 = Sunday, 7 = Saturday (BQ DAYOFWEEK).
    - ``signup_month``: 1--12.
    - ``signup_week_of_year``: ISO week number 1--53.

    If ``signup_date`` is not present in *df* the DataFrame is returned
    unchanged.  Columns that already exist are **not** overwritten, so
    dbt-computed values from training data are preserved.

    Args:
        df: Input DataFrame, potentially containing a ``signup_date``
            column (string or datetime).

    Returns:
        DataFrame with the three temporal columns added (or unchanged).
    """
    if "signup_date" not in df.columns:
        return df

    work = df.copy()
    dt = pd.to_datetime(work["signup_date"], errors="coerce")

    if "signup_month" not in work.columns:
        work["signup_month"] = dt.dt.month.fillna(0).astype(int)

    if "signup_day_of_week" not in work.columns:
        # Python: Monday=0..Sunday=6  ->  BQ: Sunday=1..Saturday=7
        py_dow = dt.dt.dayofweek  # 0=Mon .. 6=Sun
        work["signup_day_of_week"] = (
            ((py_dow + 1) % 7 + 1).fillna(0).astype(int)
        )

    if "signup_week_of_year" not in work.columns:
        work["signup_week_of_year"] = (
            dt.dt.isocalendar().week.fillna(0).astype(int)
        )

    return work


class TitleEncoder:
    """Maps titles to integer codes per service.

    For each service the encoder keeps the top-N most frequent titles and
    maps everything else to the ``__OTHER__`` bucket.

    Attributes:
        max_titles: Maximum number of distinct titles to keep per service.
        title_to_idx: ``{service: {title: int}}`` mapping.
        idx_to_title: ``{service: {int: title}}`` reverse mapping.
    """

    def __init__(self, max_titles: int = MAX_TITLES_PER_SERVICE) -> None:
        self.max_titles = max_titles
        self.title_to_idx: dict[str, dict[str, int]] = {}
        self.idx_to_title: dict[str, dict[int, str]] = {}

    def fit(self, df: pd.DataFrame) -> "TitleEncoder":
        """Fit the encoder on training data.

        Expects a DataFrame with at least ``service`` and
        ``first_watch_title`` columns.

        Args:
            df: Training DataFrame.

        Returns:
            Self, for method chaining.
        """
        self.title_to_idx = {}
        self.idx_to_title = {}

        for service, group in df.groupby("service"):
            service = str(service)
            title_counts = (
                group["first_watch_title"]
                .value_counts()
                .head(self.max_titles)
            )

            t2i: dict[str, int] = {}
            i2t: dict[int, str] = {}
            for idx, title in enumerate(title_counts.index):
                t2i[title] = idx
                i2t[idx] = title

            # Reserve last index for __OTHER__
            other_idx = len(t2i)
            t2i[OTHER_BUCKET] = other_idx
            i2t[other_idx] = OTHER_BUCKET

            self.title_to_idx[service] = t2i
            self.idx_to_title[service] = i2t

        return self

    def encode(self, service: str, titles: pd.Series) -> np.ndarray:
        """Encode a series of titles to integer codes.

        Args:
            service: The streaming service name.
            titles: Series of title strings.

        Returns:
            1-D integer array of encoded title indices.
        """
        mapping = self.title_to_idx.get(service, {})
        other_idx = mapping.get(OTHER_BUCKET, 0)
        return np.array(
            [mapping.get(t, other_idx) for t in titles], dtype=np.int32
        )

    def decode(self, service: str, indices: np.ndarray) -> list[str]:
        """Decode integer codes back to title strings.

        Args:
            service: The streaming service name.
            indices: 1-D integer array of title codes.

        Returns:
            List of title strings.
        """
        mapping = self.idx_to_title.get(service, {})
        return [mapping.get(int(i), OTHER_BUCKET) for i in indices]

    def num_classes(self, service: str) -> int:
        """Return the number of classes (including __OTHER__) for a service."""
        return len(self.title_to_idx.get(service, {}))

    @property
    def services(self) -> list[str]:
        """Return the list of services this encoder knows about."""
        return list(self.title_to_idx.keys())


def prepare_features(
    df: pd.DataFrame,
    categorical_features: list[str] | None = None,
    fit: bool = False,
    encoders: dict[str, dict[str, int]] | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Build a numeric feature matrix from the raw DataFrame.

    Numeric features are filled with 0 for missing values.  Categorical
    features are ordinal-encoded (unseen categories become 0 which
    corresponds to ``__UNKNOWN__``).

    Args:
        df: Raw feature DataFrame.
        categorical_features: List of categorical column names.  Defaults
            to ``config.CATEGORICAL_FEATURES``.
        fit: If *True*, build new ordinal encoders from the data.  If
            *False*, reuse the provided *encoders*.
        encoders: Existing ``{column: {category: int}}`` mappings.
            Required when ``fit=False``.

    Returns:
        A tuple of ``(feature_matrix, encoders)`` where
        ``feature_matrix`` is a 2-D float32 ``np.ndarray`` and
        ``encoders`` is the (possibly newly-fitted) ordinal mapping dict.
    """
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    if encoders is None:
        encoders = {}

    work = df.copy()

    # --- numeric features ---------------------------------------------------
    numeric_cols = [c for c in FEATURE_COLUMNS if c in work.columns]
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)

    # Ensure all expected numeric columns exist (fill with 0 if absent)
    for col in FEATURE_COLUMNS:
        if col not in work.columns:
            work[col] = 0.0

    # --- categorical features -----------------------------------------------
    for col in categorical_features:
        if col not in work.columns:
            work[col] = UNKNOWN_CATEGORY
        else:
            work[col] = work[col].fillna(UNKNOWN_CATEGORY).astype(str)

        if fit:
            unique_vals = sorted(work[col].unique())
            mapping: dict[str, int] = {UNKNOWN_CATEGORY: 0}
            code = 1
            for val in unique_vals:
                if val != UNKNOWN_CATEGORY:
                    mapping[val] = code
                    code += 1
            encoders[col] = mapping
        else:
            if col not in encoders:
                encoders[col] = {UNKNOWN_CATEGORY: 0}

        col_map = encoders[col]
        work[col] = work[col].map(
            lambda v, m=col_map: m.get(v, m.get(UNKNOWN_CATEGORY, 0))
        )

    # --- assemble matrix -----------------------------------------------------
    all_cols = FEATURE_COLUMNS + categorical_features
    feature_matrix = work[all_cols].to_numpy(dtype=np.float32)

    return feature_matrix, encoders
