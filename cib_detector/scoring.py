from __future__ import annotations

from collections import Counter
from dataclasses import replace

import networkx as nx
import numpy as np
import pandas as pd

from .config import DetectorConfig
from .models import CommunityRecord


def community_metrics_from_graph(graph: nx.Graph, members: list[str]) -> dict[str, float | list[str]]:
    """Aggregate edge-level metrics for a candidate community.

    Args:
        graph: Weighted user similarity graph.
        members: User identifiers belonging to the candidate community.

    Returns:
        A dictionary of averaged coordination scores, structural density,
        leader centralization, and explanatory metadata such as shared targets
        and top leaders.
    """
    subgraph = graph.subgraph(members)
    edges = list(subgraph.edges(data=True))
    if not edges:
        return {
            "sync": 0.0,
            "popularity": 0.0,
            "residual": 0.0,
            "jaccard": 0.0,
            "dtw": 0.0,
            "session": 0.0,
            "campaign": 0.0,
            "leader": 0.0,
            "density": 0.0,
            "centralization": 0.0,
            "shared_targets": [],
            "top_leaders": [],
        }

    shared_counter: Counter[str] = Counter()
    lead_counter: Counter[str] = Counter()
    for _, _, edge_data in edges:
        for target in edge_data.get("shared_targets", []):
            shared_counter[str(target)] += 1
    for user_a, user_b, edge_data in edges:
        leader_strength = float(edge_data.get("leader", 0.0))
        direction = float(edge_data.get("lead_direction", 0.0))
        if leader_strength <= 0 or direction == 0:
            continue
        if direction > 0:
            lead_counter[str(user_a)] += leader_strength
        else:
            lead_counter[str(user_b)] += leader_strength

    total_lead_mass = float(sum(lead_counter.values()))
    centralization = 0.0
    if total_lead_mass > 0 and len(members) > 1:
        centralization = float(max(lead_counter.values(), default=0.0) / total_lead_mass)

    return {
        "sync": float(np.mean([edge_data.get("sync", 0.0) for _, _, edge_data in edges])),
        "popularity": float(np.mean([edge_data.get("popularity", 0.0) for _, _, edge_data in edges])),
        "residual": float(np.mean([edge_data.get("residual", 0.0) for _, _, edge_data in edges])),
        "jaccard": float(np.mean([edge_data.get("jaccard", 0.0) for _, _, edge_data in edges])),
        "dtw": float(np.mean([edge_data.get("dtw", 0.0) for _, _, edge_data in edges])),
        "session": float(np.mean([edge_data.get("session", 0.0) for _, _, edge_data in edges])),
        "campaign": float(np.mean([edge_data.get("campaign", 0.0) for _, _, edge_data in edges])),
        "leader": float(np.mean([edge_data.get("leader", 0.0) for _, _, edge_data in edges])),
        "density": float(nx.density(subgraph)) if len(members) > 1 else 0.0,
        "centralization": centralization,
        "shared_targets": [target for target, _ in shared_counter.most_common(5)],
        "top_leaders": [target for target, _ in lead_counter.most_common(3)],
    }


def calculate_risk_score(metrics: dict[str, float | list[str]], config: DetectorConfig) -> float:
    """Compute the composite community risk score.

    Args:
        metrics: Community-level feature aggregates.
        config: Detector configuration containing community risk weights.

    Returns:
        A bounded risk score in the interval ``[0, 1]``.
    """
    weights = config.risk_weights
    risk_score = (
        weights["sync"] * float(metrics["sync"])
        + weights["popularity"] * float(metrics["popularity"])
        + weights["residual"] * float(metrics["residual"])
        + weights["jaccard"] * float(metrics["jaccard"])
        + weights["dtw"] * float(metrics["dtw"])
        + weights["density"] * float(metrics["density"])
        + weights["campaign"] * float(metrics["campaign"])
        + weights["leader"] * float(metrics["leader"])
        + weights["centralization"] * float(metrics["centralization"])
    )
    return float(min(max(risk_score, 0.0), 1.0))


def risk_level_from_score(score: float) -> str:
    """Map a numeric risk score to a qualitative risk label.

    Args:
        score: Community risk score.

    Returns:
        One of ``"low"``, ``"medium"``, or ``"high"``.
    """
    if score >= 0.75:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def score_window_communities(
    graph: nx.Graph,
    communities: list[list[str]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    config: DetectorConfig,
) -> list[CommunityRecord]:
    """Assign risk scores to communities detected in one time window.

    Args:
        graph: Weighted user similarity graph.
        communities: Candidate communities represented as member lists.
        window_start: Start timestamp of the current window.
        window_end: End timestamp of the current window.
        config: Detector configuration.

    Returns:
        A risk-sorted list of :class:`CommunityRecord` objects.
    """
    scored: list[CommunityRecord] = []
    for index, members in enumerate(communities):
        metrics = community_metrics_from_graph(graph, members)
        risk_score = calculate_risk_score(metrics, config)
        scored.append(
            CommunityRecord(
                community_id=f"{window_start.strftime('%H%M')}_{index}",
                members=members,
                window_start=window_start,
                window_end=window_end,
                sync_score=float(metrics["sync"]),
                popularity_score=float(metrics["popularity"]),
                residual_score=float(metrics["residual"]),
                jaccard_score=float(metrics["jaccard"]),
                dtw_score=float(metrics["dtw"]),
                session_score=float(metrics["session"]),
                campaign_score=float(metrics["campaign"]),
                leader_score=float(metrics["leader"]),
                density_score=float(metrics["density"]),
                centralization_score=float(metrics["centralization"]),
                risk_score=risk_score,
                risk_level=risk_level_from_score(risk_score),
                shared_targets=list(metrics["shared_targets"]),
                top_leaders=list(metrics["top_leaders"]),
            )
        )
    return sorted(scored, key=lambda item: item.risk_score, reverse=True)


def _overlap_ratio(left: list[str], right: list[str]) -> float:
    """Compute overlap relative to the smaller community.

    Args:
        left: First member list.
        right: Second member list.

    Returns:
        The overlap size divided by the size of the smaller set.
    """
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def consolidate_communities(
    communities: list[CommunityRecord],
    config: DetectorConfig,
) -> list[CommunityRecord]:
    """Merge highly overlapping communities across time windows.

    Args:
        communities: Candidate community records collected across windows.
        config: Detector configuration containing merge thresholds.

    Returns:
        A de-duplicated list of consolidated community records sorted by risk.
    """
    merged: list[CommunityRecord] = []

    for candidate in sorted(communities, key=lambda item: item.risk_score, reverse=True):
        matched = False
        for index, existing in enumerate(merged):
            if _overlap_ratio(existing.members, candidate.members) < config.overlap_merge_threshold:
                continue

            old_support = existing.support_windows
            new_support = old_support + 1
            merged_members = sorted(set(existing.members) | set(candidate.members))
            merged_targets = list(dict.fromkeys(existing.shared_targets + candidate.shared_targets))[:5]
            merged_leaders = list(dict.fromkeys(existing.top_leaders + candidate.top_leaders))[:3]

            def _rolling_average(left: float, right: float) -> float:
                """Compute a support-weighted rolling average.

                Args:
                    left: Historical average value.
                    right: New value to incorporate.

                Returns:
                    The updated average after incorporating one additional
                    supporting window.
                """
                return (left * old_support + right) / new_support

            avg_sync = _rolling_average(existing.sync_score, candidate.sync_score)
            avg_popularity = _rolling_average(existing.popularity_score, candidate.popularity_score)
            avg_residual = _rolling_average(existing.residual_score, candidate.residual_score)
            avg_jaccard = _rolling_average(existing.jaccard_score, candidate.jaccard_score)
            avg_dtw = _rolling_average(existing.dtw_score, candidate.dtw_score)
            avg_session = _rolling_average(existing.session_score, candidate.session_score)
            avg_campaign = _rolling_average(existing.campaign_score, candidate.campaign_score)
            avg_leader = _rolling_average(existing.leader_score, candidate.leader_score)
            avg_density = _rolling_average(existing.density_score, candidate.density_score)
            avg_centralization = _rolling_average(existing.centralization_score, candidate.centralization_score)

            base_score = calculate_risk_score(
                {
                    "sync": avg_sync,
                    "popularity": avg_popularity,
                    "residual": avg_residual,
                    "jaccard": avg_jaccard,
                    "dtw": avg_dtw,
                    "density": avg_density,
                    "session": avg_session,
                    "campaign": avg_campaign,
                    "leader": avg_leader,
                    "centralization": avg_centralization,
                    "shared_targets": merged_targets,
                },
                config,
            )
            support_bonus = min(0.03 * (new_support - 1), 0.12)
            final_score = min(1.0, max(base_score, existing.risk_score, candidate.risk_score) + support_bonus)

            merged[index] = replace(
                existing,
                members=merged_members,
                window_start=min(existing.window_start, candidate.window_start),
                window_end=max(existing.window_end, candidate.window_end),
                sync_score=avg_sync,
                popularity_score=avg_popularity,
                residual_score=avg_residual,
                jaccard_score=avg_jaccard,
                dtw_score=avg_dtw,
                session_score=avg_session,
                campaign_score=avg_campaign,
                leader_score=avg_leader,
                density_score=avg_density,
                centralization_score=avg_centralization,
                risk_score=final_score,
                risk_level=risk_level_from_score(final_score),
                shared_targets=merged_targets,
                top_leaders=merged_leaders,
                support_windows=new_support,
                window_mdcs=min(
                    value
                    for value in [existing.window_mdcs, candidate.window_mdcs]
                    if value is not None
                )
                if existing.window_mdcs is not None or candidate.window_mdcs is not None
                else None,
                passes_mdcs=existing.passes_mdcs and candidate.passes_mdcs,
            )
            matched = True
            break

        if not matched:
            merged.append(candidate)

    return sorted(merged, key=lambda item: item.risk_score, reverse=True)
