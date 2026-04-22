from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score

from .models import CommunityRecord


def build_account_scores(
    events: pd.DataFrame,
    communities: list[CommunityRecord],
) -> pd.DataFrame:
    """Convert community-level detections into account-level scores.

    Args:
        events: Full event table containing ground-truth account labels.
        communities: Final detected communities.

    Returns:
        A table assigning each account its best community-derived risk score and
        corresponding predicted community label.
    """
    account_truth = (
        events.groupby("user_id")
        .agg(
            is_bot=("is_bot", "max"),
            true_group=("true_group", lambda values: values.mode().iloc[0]),
        )
        .reset_index()
    )

    best_score = {user_id: 0.0 for user_id in account_truth["user_id"]}
    best_group = {user_id: "background" for user_id in account_truth["user_id"]}

    for index, community in enumerate(sorted(communities, key=lambda item: item.risk_score, reverse=True)):
        predicted_group = f"detected_{index}"
        for member in community.members:
            if community.risk_score > best_score.get(member, 0.0):
                best_score[member] = community.risk_score
                best_group[member] = predicted_group

    account_truth["risk_score"] = account_truth["user_id"].map(best_score).fillna(0.0)
    account_truth["pred_group"] = account_truth["user_id"].map(best_group).fillna("background")
    return account_truth.sort_values("risk_score", ascending=False).reset_index(drop=True)


def evaluate(
    events: pd.DataFrame,
    communities: list[CommunityRecord],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate the final detections using account and community metrics.

    Args:
        events: Full event table containing ground-truth labels.
        communities: Final detected communities.

    Returns:
        A tuple containing a metric dictionary and the account-level score
        table used to compute those metrics.
    """
    account_scores = build_account_scores(events, communities)
    true_labels = account_scores["is_bot"].astype(int).to_numpy()
    risk_scores = account_scores["risk_score"].astype(float).to_numpy()

    metrics: dict[str, float] = {}
    if len(np.unique(true_labels)) == 2:
        metrics["auroc"] = float(roc_auc_score(true_labels, risk_scores))
    else:
        metrics["auroc"] = math.nan

    bot_count = int(true_labels.sum())
    top_k = max(bot_count, 1)
    top_slice = account_scores.head(top_k)
    metrics["precision_at_k"] = float(top_slice["is_bot"].mean()) if not top_slice.empty else math.nan
    metrics["recall_at_k"] = float(top_slice["is_bot"].sum() / bot_count) if bot_count else math.nan

    bot_scores = account_scores[account_scores["is_bot"]].copy()
    if len(bot_scores) >= 2 and bot_scores["true_group"].nunique() >= 2:
        metrics["nmi"] = float(
            normalized_mutual_info_score(
                bot_scores["true_group"],
                bot_scores["pred_group"],
            )
        )
        metrics["ari"] = float(
            adjusted_rand_score(
                bot_scores["true_group"],
                bot_scores["pred_group"],
            )
        )
    else:
        metrics["nmi"] = math.nan
        metrics["ari"] = math.nan

    return metrics, account_scores
