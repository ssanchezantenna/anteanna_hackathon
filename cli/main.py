"""Click-based CLI for the First Watch prediction system.

Usage
-----
::

    # Train the model (requires BigQuery access)
    first-watch train

    # Generate predictions from a CSV
    first-watch predict --input examples/sample_input.csv --output predictions.csv

    # Evaluate on the held-out test set
    first-watch evaluate --detailed
"""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


@click.group()
def cli() -> None:
    """First Watch -- predict the first title a new streaming subscriber will watch."""


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@cli.command()
def train() -> None:
    """Train the model (requires BigQuery access)."""
    click.echo("Starting training pipeline ...")
    from first_watch_model.train import main as train_main

    train_main()
    click.echo("Training complete.")


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to input CSV with subscriber features.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to write the output predictions CSV.",
)
@click.option(
    "--top-k",
    default=10,
    show_default=True,
    type=int,
    help="Number of top title predictions per subscriber.",
)
@click.option(
    "--format",
    "output_format",
    default="long",
    show_default=True,
    type=click.Choice(["long", "wide"], case_sensitive=False),
    help="Output format: 'long' (one row per prediction) or 'wide' (one row per subscriber).",
)
def predict(
    input_path: str,
    output_path: str,
    top_k: int,
    output_format: str,
) -> None:
    """Generate title predictions from a CSV of subscriber features."""
    from first_watch_model.predict import predict_from_df

    click.echo(f"Reading input from {input_path} ...")
    df = pd.read_csv(input_path)
    click.echo(f"  {len(df)} subscribers loaded.")

    click.echo("Generating predictions ...")
    result = predict_from_df(df, top_k=top_k)

    if output_format == "wide":
        result = _pivot_wide(result)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    click.echo(f"Predictions written to {out}")


def _pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format predictions to one row per subscriber.

    Each rank becomes a pair of columns: ``title_1``, ``prob_1``, etc.
    """
    if long_df.empty:
        return long_df

    rows: list[dict] = []
    for (sub_id, service), grp in long_df.groupby(
        ["subscriber_id", "service"]
    ):
        row: dict = {"subscriber_id": sub_id, "service": service}
        for _, r in grp.iterrows():
            rank = int(r["rank"])
            row[f"title_{rank}"] = r["title"]
            row[f"prob_{rank}"] = r["probability"]
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--detailed",
    is_flag=True,
    default=False,
    help="Show per-service metric breakdowns.",
)
def evaluate(detailed: bool) -> None:
    """Evaluate model on the held-out test set (requires BigQuery access)."""
    click.echo("Starting evaluation ...")
    from first_watch_model.evaluate import main as eval_main

    eval_main(detailed=detailed)
    click.echo("Evaluation complete.")


if __name__ == "__main__":
    cli()
