"""Data loading utilities for BigQuery training and popularity tables."""

from __future__ import annotations

import pandas as pd
from google.cloud import bigquery

from first_watch_model.config import (
    BQ_DATASET,
    BQ_PROJECT,
    POPULARITY_TABLE,
    TRAINING_TABLE,
)


def _get_client() -> bigquery.Client:
    """Return a BigQuery client bound to the project."""
    return bigquery.Client(project=BQ_PROJECT)


def load_training_data(split: str | None = None) -> pd.DataFrame:
    """Load the training dataset from BigQuery.

    Args:
        split: Optional data split to filter on (e.g. ``'train'`` or
            ``'test'``). When *None*, all rows are returned.

    Returns:
        DataFrame with all columns from ``mart_training_dataset``.
    """
    table_ref = f"`{BQ_PROJECT}.{BQ_DATASET}.{TRAINING_TABLE}`"
    query = f"SELECT * FROM {table_ref}"

    if split is not None:
        query += f" WHERE data_split = @split"

    client = _get_client()

    if split is not None:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("split", "STRING", split),
            ]
        )
        df: pd.DataFrame = client.query(query, job_config=job_config).to_dataframe()
    else:
        df = client.query(query).to_dataframe()

    # dbt mart uses "target_service"; downstream Python expects "service"
    if "target_service" in df.columns and "service" not in df.columns:
        df = df.rename(columns={"target_service": "service"})

    return df


def load_title_popularity() -> pd.DataFrame:
    """Load the title popularity table from BigQuery.

    Returns:
        DataFrame with columns including ``service``,
        ``first_watch_title``, ``empirical_probability``, and
        ``popularity_rank``.
    """
    table_ref = f"`{BQ_PROJECT}.{BQ_DATASET}.{POPULARITY_TABLE}`"
    query = f"SELECT * FROM {table_ref}"

    client = _get_client()
    df: pd.DataFrame = client.query(query).to_dataframe()

    # dbt mart uses "target_service"; downstream Python expects "service"
    if "target_service" in df.columns and "service" not in df.columns:
        df = df.rename(columns={"target_service": "service"})

    return df
