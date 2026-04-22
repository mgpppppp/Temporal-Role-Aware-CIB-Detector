from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .config import DetectorConfig


def build_bipartite_graph(window_events: pd.DataFrame) -> nx.Graph:
    """Build a user-content bipartite graph for a single time window.

    Args:
        window_events: Event records that fall within one temporal window.

    Returns:
        A bipartite :class:`networkx.Graph` connecting user nodes to content
        nodes, with edge attributes summarizing interaction counts and timing.
    """
    graph = nx.Graph()
    for row in window_events.itertuples(index=False):
        user_node = f"user::{row.user_id}"
        content_node = f"content::{row.content_id}"
        graph.add_node(user_node, bipartite="user", raw_id=str(row.user_id))
        graph.add_node(content_node, bipartite="content", raw_id=str(row.content_id))

        if graph.has_edge(user_node, content_node):
            graph[user_node][content_node]["count"] += 1
            graph[user_node][content_node]["last_timestamp"] = row.timestamp
        else:
            graph.add_edge(
                user_node,
                content_node,
                count=1,
                first_timestamp=row.timestamp,
                last_timestamp=row.timestamp,
            )
    return graph


def build_user_similarity_graph(
    pairwise_features: pd.DataFrame,
    profiles: dict[str, dict[str, object]],
    config: DetectorConfig,
) -> nx.Graph:
    """Construct the weighted user similarity graph used for mining.

    Args:
        pairwise_features: Pairwise coordination feature table.
        profiles: Per-user behavioral profiles for the current time window.
        config: Detector configuration containing edge weighting thresholds.

    Returns:
        A weighted :class:`networkx.Graph` whose nodes are users and whose
        edges encode coordination strength and explanatory feature attributes.
    """
    graph = nx.Graph()
    for user_id, profile in profiles.items():
        graph.add_node(
            user_id,
            event_count=profile["event_count"],
            active_span=profile["active_span"],
            burstiness=profile["burstiness"],
            content_focus=profile["content_focus"],
            action_entropy=profile["action_entropy"],
            role_tag=profile["role_tag"],
            leader_score_window=profile.get("leader_score_window", 0.0),
            follower_score_window=profile.get("follower_score_window", 0.0),
            role_balance_window=profile.get("role_balance_window", 0.5),
            role_consistency_window=profile.get("role_consistency_window", 0.0),
            is_bot=profile["is_bot"],
            true_group=profile["true_group"],
        )

    if pairwise_features.empty:
        return graph

    weights = config.similarity_weights
    df = pairwise_features.copy()
    df["weight"] = (
        weights["sync"] * df["sync"]
        + weights["popularity"] * df["popularity"]
        + weights["jaccard"] * df["jaccard"]
        + weights["dtw"] * df["dtw"]
        + weights["session"] * df["session"]
        + weights["campaign"] * df["campaign"]
        + weights["residual"] * df["residual"]
        + weights["leader"] * df["leader"]
    )

    threshold = float(df["weight"].quantile(config.edge_quantile)) if len(df) > 1 else config.min_edge_weight
    threshold = max(threshold, config.min_edge_weight)
    kept = df[df["weight"] >= threshold].copy()

    if kept.empty:
        kept = df.nlargest(min(5, len(df)), "weight").copy()

    for row in kept.itertuples(index=False):
        graph.add_edge(
            row.user_a,
            row.user_b,
            weight=float(row.weight),
            sync=float(row.sync),
            popularity=float(row.popularity),
            jaccard=float(row.jaccard),
            dtw=float(row.dtw),
            session=float(row.session),
            campaign=float(row.campaign),
            residual=float(row.residual),
            leader=float(row.leader),
            lead_direction=float(row.lead_direction),
            mean_lead_lag=float(row.mean_lead_lag),
            lead_dominance=float(row.lead_dominance),
            expected_coordination=float(row.expected_coordination),
            shared_targets=list(row.shared_targets),
            shared_target_count=int(row.shared_target_count),
        )

    for node in list(graph.nodes):
        if graph.degree(node) == 0 and node in profiles:
            graph.remove_node(node)

    return graph


def graph_stats(graph: nx.Graph) -> dict[str, float]:
    """Compute coarse summary statistics for a similarity graph.

    Args:
        graph: Weighted user similarity graph.

    Returns:
        A dictionary containing node count, edge count, graph density, and the
        mean retained edge weight.
    """
    if graph.number_of_nodes() == 0:
        return {"nodes": 0.0, "edges": 0.0, "density": 0.0, "mean_weight": 0.0}

    weights = [data.get("weight", 0.0) for _, _, data in graph.edges(data=True)]
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "mean_weight": float(np.mean(weights)) if weights else 0.0,
    }
