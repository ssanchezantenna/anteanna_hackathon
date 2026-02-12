"""Evaluation metrics for the First Watch prediction system.

Computes per-service and macro-averaged metrics:

- Top-k accuracy (k = 1, 3, 5, 10)
- Mean Reciprocal Rank (MRR)
- Log-likelihood
- KL divergence (predicted aggregate distribution vs actual distribution)

Compares three model variants side-by-side:
  Baseline-only | Classifier-only | Ensemble
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from first_watch_model.baseline import PopularityBaseline
from first_watch_model.classifier import ServiceClassifier
from first_watch_model.config import (
    MODEL_DIR,
    SMOOTHING_ALPHA,
    TARGET_SERVICES,
    TEMPERATURE,
)
from first_watch_model.data_loader import load_training_data
from first_watch_model.ensemble import EnsemblePredictor
from first_watch_model.features import (
    OTHER_BUCKET,
    TitleEncoder,
    prepare_features,
)
from first_watch_model.predict import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Evaluation k values
TOP_K_VALUES: list[int] = [1, 3, 5, 10]


# -----------------------------------------------------------------------
# Metric computation helpers
# -----------------------------------------------------------------------


def _top_k_accuracy(
    ranked_titles: list[list[str]],
    true_titles: list[str],
    k: int,
) -> float:
    """Fraction of samples where the true title appears in the top-k.

    Args:
        ranked_titles: Per-sample list of titles ordered by descending
            predicted probability.
        true_titles: Ground-truth title for each sample.
        k: Number of top predictions to consider.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    if not true_titles:
        return 0.0
    hits = sum(
        1 for ranked, true_t in zip(ranked_titles, true_titles)
        if true_t in ranked[:k]
    )
    return hits / len(true_titles)


def _mrr(ranked_titles: list[list[str]], true_titles: list[str]) -> float:
    """Mean Reciprocal Rank.

    Args:
        ranked_titles: Per-sample ranked title lists.
        true_titles: Ground-truth titles.

    Returns:
        MRR score in ``[0, 1]``.
    """
    if not true_titles:
        return 0.0
    rr_sum = 0.0
    for ranked, true_t in zip(ranked_titles, true_titles):
        try:
            rank = ranked.index(true_t) + 1
            rr_sum += 1.0 / rank
        except ValueError:
            pass  # true title not in ranked list
    return rr_sum / len(true_titles)


def _log_likelihood(prob_of_true: np.ndarray) -> float:
    """Average log-likelihood of the true title under predicted probs.

    Args:
        prob_of_true: Array of ``P(true_title)`` for each sample.

    Returns:
        Mean log-likelihood (higher is better, maximum 0).
    """
    eps = 1e-12
    return float(np.mean(np.log(prob_of_true + eps)))


def _kl_divergence(
    predicted_dist: dict[str, float],
    actual_dist: dict[str, float],
) -> float:
    """KL(actual || predicted) over the title vocabulary.

    Args:
        predicted_dist: Aggregated predicted distribution ``{title: prob}``.
        actual_dist: Empirical distribution ``{title: prob}``.

    Returns:
        KL divergence (>= 0, lower is better).
    """
    eps = 1e-12
    all_titles = set(predicted_dist) | set(actual_dist)
    kl = 0.0
    for title in all_titles:
        p = actual_dist.get(title, eps)
        q = predicted_dist.get(title, eps)
        kl += p * np.log(p / q)
    return float(kl)


# -----------------------------------------------------------------------
# Per-variant metric extraction
# -----------------------------------------------------------------------


def _extract_metrics(
    proba_matrix: np.ndarray,
    idx_to_title: dict[int, str],
    title_to_idx: dict[str, int],
    true_titles: list[str],
) -> dict[str, float]:
    """Compute all metrics from a probability matrix and ground truth.

    Args:
        proba_matrix: Shape ``(n_samples, n_classes)``.
        idx_to_title: Index-to-title mapping.
        title_to_idx: Title-to-index mapping.
        true_titles: Ground-truth titles.

    Returns:
        Dict of metric name to value.
    """
    n_samples = proba_matrix.shape[0]

    # Build ranked title lists and P(true)
    ranked_titles: list[list[str]] = []
    p_true = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        row = proba_matrix[i]
        order = np.argsort(row)[::-1]
        ranked_titles.append(
            [idx_to_title.get(int(j), OTHER_BUCKET) for j in order]
        )
        t_idx = title_to_idx.get(true_titles[i])
        if t_idx is not None and t_idx < len(row):
            p_true[i] = row[t_idx]

    # Aggregate distributions for KL divergence
    actual_counts: dict[str, int] = defaultdict(int)
    for t in true_titles:
        actual_counts[t] += 1
    total_actual = sum(actual_counts.values())
    actual_dist = {t: c / total_actual for t, c in actual_counts.items()}

    pred_mean = proba_matrix.mean(axis=0)
    pred_dist: dict[str, float] = {}
    for idx, prob in enumerate(pred_mean):
        title = idx_to_title.get(idx, OTHER_BUCKET)
        pred_dist[title] = pred_dist.get(title, 0.0) + float(prob)

    # Assemble metrics
    metrics: dict[str, float] = {}
    for k in TOP_K_VALUES:
        metrics[f"top_{k}_accuracy"] = _top_k_accuracy(
            ranked_titles, true_titles, k
        )
    metrics["mrr"] = _mrr(ranked_titles, true_titles)
    metrics["log_likelihood"] = _log_likelihood(p_true)
    metrics["kl_divergence"] = _kl_divergence(pred_dist, actual_dist)
    metrics["n_samples"] = float(n_samples)

    return metrics


# -----------------------------------------------------------------------
# Per-service evaluation (all three variants)
# -----------------------------------------------------------------------


def _build_baseline_matrix(
    svc_df: pd.DataFrame,
    service: str,
    ensemble: EnsemblePredictor,
    n_classes: int,
) -> np.ndarray:
    """Build a per-subscriber baseline probability matrix.

    When ``signup_month`` is available in *svc_df*, each subscriber gets
    the month-specific empirical vector.  Otherwise a single global
    vector is tiled across all subscribers.

    Returns:
        Array of shape ``(len(svc_df), n_classes)``.
    """
    has_month = "signup_month" in svc_df.columns

    if not has_month:
        vec = ensemble._build_empirical_vector(service)[0]
        return np.tile(vec, (len(svc_df), 1))

    # Build one empirical vector per unique month, then assemble per row
    months = svc_df["signup_month"].values
    unique_months = set(int(m) for m in months if pd.notna(m) and int(m) > 0)
    unique_months.add(0)  # fallback for unknown months

    cache: dict[int, np.ndarray] = {}
    for m in unique_months:
        month_arg = m if m > 0 else None
        cache[m] = ensemble._build_empirical_vector(service, month=month_arg)[0]

    matrix = np.zeros((len(svc_df), n_classes), dtype=np.float64)
    for i, m in enumerate(months):
        m_int = int(m) if pd.notna(m) and int(m) > 0 else 0
        vec = cache.get(m_int, cache[0])
        matrix[i, :len(vec)] = vec

    return matrix


def evaluate_service(
    service: str,
    test_df: pd.DataFrame,
    ensemble: EnsemblePredictor,
    title_encoder: TitleEncoder,
    feature_encoders: dict[str, dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Evaluate Baseline, Classifier-only, and Ensemble on one service.

    Args:
        service: Target streaming service name.
        test_df: Full test DataFrame (will be filtered to *service*).
        ensemble: Fitted EnsemblePredictor.
        title_encoder: Fitted TitleEncoder.
        feature_encoders: Categorical encoding mappings.

    Returns:
        Dict keyed by variant (``baseline``, ``classifier``, ``ensemble``)
        mapping to a metrics dict.  Empty dict when no test data exists
        for *service*.
    """
    svc_df = test_df[test_df["service"] == service].reset_index(drop=True)
    if len(svc_df) == 0:
        return {}

    true_titles: list[str] = svc_df["first_watch_title"].tolist()
    n_classes = title_encoder.num_classes(service)
    idx_to_title = title_encoder.idx_to_title.get(service, {})
    title_to_idx = title_encoder.title_to_idx.get(service, {})

    X_test, _ = prepare_features(
        svc_df, fit=False, encoders=feature_encoders
    )

    results: dict[str, dict[str, float]] = {}

    # --- 1. Baseline-only (pure popularity, time-aware) ---------------------
    baseline_matrix = _build_baseline_matrix(
        svc_df, service, ensemble, n_classes
    )
    results["baseline"] = _extract_metrics(
        baseline_matrix, idx_to_title, title_to_idx, true_titles
    )

    # --- 2. Classifier-only (alpha=0, no blending) --------------------------
    if service in ensemble.classifiers:
        clf_proba = ensemble.classifiers[service].predict_proba(
            X_test, temperature=TEMPERATURE
        )
        # Pad / truncate to match title encoder width
        if clf_proba.shape[1] < n_classes:
            pad = np.zeros(
                (clf_proba.shape[0], n_classes - clf_proba.shape[1]),
                dtype=np.float64,
            )
            clf_proba = np.hstack([clf_proba, pad])
        elif clf_proba.shape[1] > n_classes:
            clf_proba = clf_proba[:, :n_classes]
        results["classifier"] = _extract_metrics(
            clf_proba, idx_to_title, title_to_idx, true_titles
        )
    else:
        # No classifier -- classifier variant equals baseline
        results["classifier"] = results["baseline"].copy()

    # --- 3. Ensemble (blended with time-aware baseline) ---------------------
    if service in ensemble.classifiers:
        ensemble_proba = (
            (1.0 - SMOOTHING_ALPHA) * clf_proba
            + SMOOTHING_ALPHA * baseline_matrix
        )
        # Re-normalize rows
        row_sums = ensemble_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        ensemble_proba /= row_sums
    else:
        ensemble_proba = baseline_matrix.copy()

    results["ensemble"] = _extract_metrics(
        ensemble_proba, idx_to_title, title_to_idx, true_titles
    )

    return results


# -----------------------------------------------------------------------
# Full evaluation across all services
# -----------------------------------------------------------------------


def run_full_evaluation(
    test_df: pd.DataFrame,
    ensemble: EnsemblePredictor,
    title_encoder: TitleEncoder,
    feature_encoders: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Evaluate all services and return a tidy results DataFrame.

    Args:
        test_df: Test-split DataFrame with ``service`` and
            ``first_watch_title`` columns plus feature columns.
        ensemble: Fitted EnsemblePredictor.
        title_encoder: Fitted TitleEncoder.
        feature_encoders: Categorical encoding mappings.

    Returns:
        DataFrame with columns ``[service, variant, metric, value]``.
    """
    records: list[dict[str, Any]] = []

    for service in TARGET_SERVICES:
        svc_results = evaluate_service(
            service=service,
            test_df=test_df,
            ensemble=ensemble,
            title_encoder=title_encoder,
            feature_encoders=feature_encoders,
        )
        if not svc_results:
            logger.warning("No test data for %s -- skipping.", service)
            continue

        for variant, metrics in svc_results.items():
            for metric_name, value in metrics.items():
                records.append(
                    {
                        "service": service,
                        "variant": variant,
                        "metric": metric_name,
                        "value": value,
                    }
                )

    return pd.DataFrame(records)


def print_results_table(
    results_df: pd.DataFrame, detailed: bool = False
) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results_df: Tidy DataFrame as returned by
            :func:`run_full_evaluation`.
        detailed: When *True*, include per-service breakdowns in the
            output table.  When *False*, only the macro-averaged summary
            is printed.
    """
    if results_df.empty:
        print("No evaluation results to display.")
        return

    # Determine which variants and metrics are present
    variants = [
        v for v in ["baseline", "classifier", "ensemble"]
        if v in results_df["variant"].unique()
    ]
    metric_names = [
        m for m in results_df["metric"].unique() if m != "n_samples"
    ]

    # Compute macro averages
    macro_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        for variant in variants:
            subset = results_df[
                (results_df["variant"] == variant)
                & (results_df["metric"] == metric)
            ]
            if not subset.empty:
                macro_rows.append(
                    {
                        "service": "** MACRO AVG **",
                        "variant": variant,
                        "metric": metric,
                        "value": subset["value"].mean(),
                    }
                )

    if detailed:
        full_df = pd.concat(
            [results_df, pd.DataFrame(macro_rows)], ignore_index=True
        )
    else:
        full_df = pd.DataFrame(macro_rows)

    # Build pivot: rows = (service, metric), columns = variant
    pivot = full_df.pivot_table(
        index=["service", "metric"],
        columns="variant",
        values="value",
        aggfunc="first",
    )

    # ---- formatted output ---------------------------------------------------
    col_width = 14
    print()
    print("=" * 92)
    print("FIRST WATCH MODEL -- EVALUATION RESULTS")
    print("=" * 92)

    header = f"{'Service':<24} {'Metric':<20}"
    for v in variants:
        header += f" {v:>{col_width}}"
    print(header)
    print("-" * 92)

    current_service = ""
    for (service, metric), row in pivot.iterrows():
        if service != current_service:
            if current_service:
                print("-" * 92)
            current_service = service

        line = f"{service:<24} {metric:<20}"
        for v in variants:
            val = row.get(v, float("nan"))
            if metric == "n_samples":
                line += f" {val:>{col_width}.0f}"
            else:
                line += f" {val:>{col_width}.4f}"
        print(line)

    print("=" * 92)


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------


def main(detailed: bool = False) -> None:
    """Load test data and model, run full evaluation, print results.

    Args:
        detailed: When *True*, print per-service metric breakdowns in the
            results table.  When *False*, only macro-averaged metrics are
            shown.
    """
    logger.info("Loading test data from BigQuery (split='test')...")
    test_df = load_training_data(split="test")
    logger.info("Test rows: %d", len(test_df))

    logger.info("Loading model artifacts...")
    ensemble = load_model()

    model_dir = Path(MODEL_DIR)
    title_encoder: TitleEncoder = joblib.load(
        model_dir / "title_encoder.joblib"
    )
    feature_encoders: dict[str, dict[str, int]] = joblib.load(
        model_dir / "feature_encoders.joblib"
    )

    logger.info("Running evaluation across all services...")
    results_df = run_full_evaluation(
        test_df=test_df,
        ensemble=ensemble,
        title_encoder=title_encoder,
        feature_encoders=feature_encoders,
    )

    print_results_table(results_df, detailed=detailed)

    # Persist results
    output_path = model_dir / "evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info("Saved evaluation results to %s", output_path)


if __name__ == "__main__":
    main()
