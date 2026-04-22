from __future__ import annotations

import networkx as nx
import numpy as np

from .config import DetectorConfig
from .mining import discover_louvain_communities
from .models import CommunityRecord
from .scoring import calculate_risk_score, community_metrics_from_graph


def _rewired_graph(graph: nx.Graph, rng: np.random.Generator) -> nx.Graph:
    """Generate a null graph by rewiring edges while preserving attributes.

    Args:
        graph: Observed user similarity graph.
        rng: Random number generator used for rewiring.

    Returns:
        A rewired graph with shuffled structure and reassigned edge attributes.
    """
    structure = nx.Graph()
    structure.add_nodes_from(graph.nodes(data=True))
    structure.add_edges_from(graph.edges())

    if structure.number_of_edges() >= 4:
        try:
            nx.double_edge_swap(
                structure,
                nswap=min(structure.number_of_edges() * 2, 80),
                max_tries=500,
                seed=int(rng.integers(0, 1_000_000)),
            )
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            pass

    original_edge_attrs = [dict(data) for _, _, data in graph.edges(data=True)]
    rng.shuffle(original_edge_attrs)

    rewired = nx.Graph()
    rewired.add_nodes_from(graph.nodes(data=True))
    for (user_a, user_b), edge_attrs in zip(structure.edges(), original_edge_attrs):
        rewired.add_edge(user_a, user_b, **edge_attrs)
    return rewired


def estimate_window_mdcs(
    graph: nx.Graph,
    observed_communities: list[CommunityRecord],
    config: DetectorConfig,
) -> tuple[int | None, dict[int, dict[str, float]]]:
    """Estimate the minimum detectable community size for one window.

    Args:
        graph: Observed user similarity graph.
        observed_communities: Community candidates scored on the observed graph.
        config: Detector configuration containing null-model parameters.

    Returns:
        A tuple containing the estimated MDCS threshold and per-size null-model
        summary statistics.
    """
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0 or not observed_communities:
        return None, {}

    max_size = max(len(item.members) for item in observed_communities)
    observed_best = {size: 0.0 for size in range(2, max_size + 1)}
    for item in observed_communities:
        for size in range(2, len(item.members) + 1):
            observed_best[size] = max(observed_best[size], item.risk_score)

    rng = np.random.default_rng(config.random_seed)
    null_scores: dict[int, list[float]] = {size: [] for size in range(2, max_size + 1)}

    for _ in range(config.null_trials):
        null_graph = _rewired_graph(graph, rng)
        null_communities = discover_louvain_communities(null_graph, config)
        trial_best = {size: 0.0 for size in range(2, max_size + 1)}

        for members in null_communities:
            metrics = community_metrics_from_graph(null_graph, members)
            risk_score = calculate_risk_score(metrics, config)
            for size in range(2, min(len(members), max_size) + 1):
                trial_best[size] = max(trial_best[size], risk_score)

        for size, best_score in trial_best.items():
            null_scores[size].append(best_score)

    mdcs: int | None = None
    details: dict[int, dict[str, float]] = {}
    for size in range(2, max_size + 1):
        samples = np.array(null_scores[size], dtype=float)
        mean_score = float(samples.mean()) if len(samples) else 0.0
        std_score = float(samples.std()) if len(samples) else 0.0
        threshold = mean_score + config.mdcs_zscore * std_score
        details[size] = {
            "observed": float(observed_best[size]),
            "null_mean": mean_score,
            "null_std": std_score,
            "threshold": float(threshold),
        }
        if mdcs is None and observed_best[size] >= max(threshold, config.mdcs_min_observed_risk):
            mdcs = size

    return mdcs, details
