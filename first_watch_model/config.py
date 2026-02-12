"""Configuration constants for the First Watch prediction model."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Target streaming services
# ---------------------------------------------------------------------------
TARGET_SERVICES: list[str] = [
    "Amazon Prime Video",
    "Apple TV+",
    "Disney+",
    "Max",
    "Netflix",
    "Hulu",
    "Peacock",
    "Paramount+",
]

# ---------------------------------------------------------------------------
# BigQuery references
# ---------------------------------------------------------------------------
BQ_PROJECT: str = "antenna-reporting"
BQ_DATASET: str = "dbt_first_watch"
TRAINING_TABLE: str = "mart_training_dataset"
POPULARITY_TABLE: str = "mart_title_popularity"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
FEATURE_COLUMNS: list[str] = [
    # Service flags (binary 0/1)
    "has_netflix",
    "has_amazon",
    "has_disney",
    "has_max",
    "has_apple_tv",
    "has_hulu",
    "has_peacock",
    "has_paramount",
    "has_linear",
    # Activity metrics
    "active_days_30d",
    "total_duration_30d",
    "distinct_titles_30d",
    "total_sessions_30d",
    # Service diversity
    "distinct_services_30d",
    # Temporal patterns
    "avg_hour_of_day",
    "stddev_hour_of_day",
    "weekend_ratio",
    "primetime_ratio",
    # Content preferences
    "avg_session_duration",
    # Signup date temporal features
    "signup_month",
    "signup_day_of_week",
    "signup_week_of_year",
]

CATEGORICAL_FEATURES: list[str] = [
    "dominant_content_type",
    "city",
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
MAX_TITLES_PER_SERVICE: int = 75
SMOOTHING_ALPHA: float = 0.3
TEMPERATURE: float = 1.5

LGBM_PARAMS: dict = {
    "objective": "multiclass",
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "n_estimators": 300,
    "verbose": -1,
    "random_state": 42,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Paths and thresholds
# ---------------------------------------------------------------------------
MODEL_DIR: Path = Path("models")
MIN_SAMPLES_PER_SERVICE: int = 50
