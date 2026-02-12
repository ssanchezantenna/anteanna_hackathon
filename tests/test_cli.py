"""Tests for the CLI (cli.main) using Click's CliRunner.

All BigQuery and model-loading calls are mocked so tests run offline.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from cli.main import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_INPUT_CSV = """\
service,has_netflix,has_amazon,has_disney,has_max,has_apple_tv,has_hulu,has_peacock,has_paramount,has_linear,active_days_30d,total_duration_30d,distinct_titles_30d,total_sessions_30d,distinct_services_30d,avg_hour_of_day,stddev_hour_of_day,weekend_ratio,primetime_ratio,dominant_content_type,avg_session_duration,city
Netflix,0,1,1,0,0,0,0,0,1,22,54000,18,45,3,20.5,2.1,0.35,0.62,series,1200,New York
Disney+,1,1,0,0,1,0,0,0,0,15,32000,12,28,3,18.0,3.4,0.55,0.48,movie,1143,Los Angeles
"""


def _mock_predict_from_df(
    df: pd.DataFrame,
    model=None,
    top_k: int = 10,
    alpha: float = 0.3,
    temperature: float = 1.5,
) -> pd.DataFrame:
    """Return a fake predictions DataFrame matching the real API signature."""
    rows = []
    for idx, row in df.iterrows():
        for rank in range(1, top_k + 1):
            rows.append(
                {
                    "subscriber_id": int(idx),
                    "service": row["service"],
                    "rank": rank,
                    "title": f"Title {rank}",
                    "probability": round(1.0 / (rank + 1), 4),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCliTrain:
    """Smoke tests for the 'train' command."""

    @patch("first_watch_model.train.main")
    def test_train_invokes_main(self, mock_train_main: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])

        assert result.exit_code == 0, result.output
        mock_train_main.assert_called_once()

    @patch("first_watch_model.train.main")
    def test_train_output_message(self, mock_train_main: MagicMock) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train"])

        assert "Starting training pipeline" in result.output
        assert "Training complete" in result.output


class TestCliPredict:
    """Tests for the 'predict' command."""

    @patch("first_watch_model.predict.predict_from_df", side_effect=_mock_predict_from_df)
    def test_predict_writes_output(self, mock_predict: MagicMock) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            output_path = os.path.join(tmpdir, "output.csv")

            with open(input_path, "w") as f:
                f.write(SAMPLE_INPUT_CSV)

            result = runner.invoke(
                cli,
                ["predict", "--input", input_path, "--output", output_path],
            )

            assert result.exit_code == 0, result.output
            assert os.path.exists(output_path)

            out_df = pd.read_csv(output_path)
            assert "subscriber_id" in out_df.columns
            assert "title" in out_df.columns
            assert "probability" in out_df.columns

    @patch("first_watch_model.predict.predict_from_df", side_effect=_mock_predict_from_df)
    def test_predict_top_k_passed(self, mock_predict: MagicMock) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            output_path = os.path.join(tmpdir, "output.csv")

            with open(input_path, "w") as f:
                f.write(SAMPLE_INPUT_CSV)

            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--input", input_path,
                    "--output", output_path,
                    "--top-k", "5",
                ],
            )

            assert result.exit_code == 0, result.output
            mock_predict.assert_called_once()
            # Verify top_k was passed through
            _, kwargs = mock_predict.call_args
            assert kwargs.get("top_k") == 5

    @patch("first_watch_model.predict.predict_from_df", side_effect=_mock_predict_from_df)
    def test_predict_wide_format(self, mock_predict: MagicMock) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.csv")
            output_path = os.path.join(tmpdir, "output.csv")

            with open(input_path, "w") as f:
                f.write(SAMPLE_INPUT_CSV)

            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--input", input_path,
                    "--output", output_path,
                    "--format", "wide",
                ],
            )

            assert result.exit_code == 0, result.output
            out_df = pd.read_csv(output_path)
            # Wide format should have title_1, prob_1, etc.
            assert "title_1" in out_df.columns
            assert "prob_1" in out_df.columns

    @patch("first_watch_model.predict.predict_from_df", side_effect=_mock_predict_from_df)
    def test_predict_with_sample_input_file(self, mock_predict: MagicMock) -> None:
        """Use the actual examples/sample_input.csv if it exists."""
        sample_path = (
            Path(__file__).resolve().parent.parent / "examples" / "sample_input.csv"
        )
        if not sample_path.exists():
            pytest.skip("examples/sample_input.csv not found")

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "predictions.csv")

            result = runner.invoke(
                cli,
                [
                    "predict",
                    "--input", str(sample_path),
                    "--output", output_path,
                ],
            )

            assert result.exit_code == 0, result.output
            assert os.path.exists(output_path)

    def test_predict_missing_input_fails(self) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["predict", "--input", "/nonexistent/file.csv", "--output", "out.csv"],
        )
        assert result.exit_code != 0


class TestCliEvaluate:
    """Smoke tests for the 'evaluate' command."""

    @patch("first_watch_model.evaluate.main")
    def test_evaluate_invokes_main(self, mock_eval_main: MagicMock) -> None:
        mock_eval_main.return_value = None
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate"])

        assert result.exit_code == 0, result.output
        mock_eval_main.assert_called_once_with(detailed=False)

    @patch("first_watch_model.evaluate.main")
    def test_evaluate_detailed_flag(self, mock_eval_main: MagicMock) -> None:
        mock_eval_main.return_value = None
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--detailed"])

        assert result.exit_code == 0, result.output
        mock_eval_main.assert_called_once_with(detailed=True)

    @patch("first_watch_model.evaluate.main")
    def test_evaluate_output_message(self, mock_eval_main: MagicMock) -> None:
        mock_eval_main.return_value = None
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate"])

        assert "Starting evaluation" in result.output
        assert "Evaluation complete" in result.output


class TestCliHelp:
    """Verify help text is available."""

    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "First Watch" in result.output

    def test_train_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0

    def test_predict_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["predict", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--output" in result.output

    def test_evaluate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["evaluate", "--help"])
        assert result.exit_code == 0
        assert "--detailed" in result.output
