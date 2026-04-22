from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .models import CommunityRecord


def save_outputs(
    output_dir: str,
    communities: list[CommunityRecord],
    account_scores: pd.DataFrame,
    metrics: dict[str, float],
    window_stats: pd.DataFrame,
) -> dict[str, str]:
    """Persist method-level outputs to disk.

    Args:
        output_dir: Destination directory for exported files.
        communities: Final detected communities.
        account_scores: Account-level risk score table.
        metrics: Metric dictionary for the method.
        window_stats: Window-level summary table.

    Returns:
        A dictionary mapping artifact names to their file-system paths.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    communities_path = path / "communities.json"
    metrics_path = path / "metrics.json"
    account_scores_path = path / "account_scores.csv"
    window_stats_path = path / "window_stats.csv"
    report_path = path / "community_report.csv"

    with communities_path.open("w", encoding="utf-8") as handle:
        json.dump([item.to_dict() for item in communities], handle, ensure_ascii=False, indent=2)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    account_scores.to_csv(account_scores_path, index=False, encoding="utf-8-sig")
    window_stats.to_csv(window_stats_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([item.to_dict() for item in communities]).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "communities": str(communities_path),
        "metrics": str(metrics_path),
        "account_scores": str(account_scores_path),
        "window_stats": str(window_stats_path),
        "community_report": str(report_path),
    }


def save_comparison_metrics(output_dir: str, comparison: dict[str, object]) -> str:
    """Persist cross-method comparison metrics to disk.

    Args:
        output_dir: Destination directory for the comparison artifact.
        comparison: Comparison dictionary produced by the pipeline.

    Returns:
        The path to the saved comparison JSON file.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    comparison_path = path / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
    return str(comparison_path)
