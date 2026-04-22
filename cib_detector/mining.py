from __future__ import annotations

from collections import defaultdict
import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning

from .config import DetectorConfig

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


def discover_louvain_communities(graph: nx.Graph, config: DetectorConfig) -> list[list[str]]:
    """Detect communities with the Louvain algorithm or a greedy fallback.

    Args:
        graph: Weighted user similarity graph.
        config: Detector configuration containing Louvain parameters.

    Returns:
        A list of community member lists that satisfy minimum size and density
        constraints.
    """
    if graph.number_of_nodes() < config.initial_min_cluster_size or graph.number_of_edges() == 0:
        return []

    try:
        raw_communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=config.louvain_resolution,
            seed=config.random_seed,
        )
    except AttributeError:
        raw_communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))

    return _filter_candidate_communities(graph, raw_communities, config)


class _MaxPoolSAGELayer(nn.Module):
    """Single max-pooling GraphSAGE aggregation layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize the layer.

        Args:
            in_dim: Input feature dimensionality.
            hidden_dim: Hidden dimensionality for the pooling MLP.
            out_dim: Output feature dimensionality.
        """
        super().__init__()
        self.pool_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.output_linear = nn.Linear(in_dim + hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, neighbors: list[list[int]]) -> torch.Tensor:
        """Aggregate neighbor features and return updated node states.

        Args:
            x: Node feature matrix.
            neighbors: Adjacency list represented as index lists.

        Returns:
            Updated node representations after max-pooling aggregation.
        """
        pooled_vectors: list[torch.Tensor] = []
        for node_index, neighbor_indices in enumerate(neighbors):
            if neighbor_indices:
                neighbor_tensor = self.pool_mlp(x[neighbor_indices])
                pooled = torch.max(neighbor_tensor, dim=0).values
            else:
                pooled = torch.zeros(
                    self.pool_mlp[-2].out_features,
                    device=x.device,
                    dtype=x.dtype,
                )
            pooled_vectors.append(torch.cat([x[node_index], pooled], dim=0))
        stacked = torch.stack(pooled_vectors, dim=0)
        return F.relu(self.output_linear(stacked))


class _GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder used for unsupervised node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        """Initialize the encoder.

        Args:
            in_dim: Input feature dimensionality.
            hidden_dim: Hidden representation dimensionality.
            embedding_dim: Output embedding dimensionality.
        """
        super().__init__()
        self.layer1 = _MaxPoolSAGELayer(in_dim, hidden_dim, hidden_dim)
        self.layer2 = _MaxPoolSAGELayer(hidden_dim, hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, neighbors: list[list[int]]) -> torch.Tensor:
        """Encode nodes into normalized embeddings.

        Args:
            x: Node feature matrix.
            neighbors: Adjacency list represented as index lists.

        Returns:
            L2-normalized node embeddings.
        """
        hidden = self.layer1(x, neighbors)
        embedding = self.layer2(hidden, neighbors)
        return F.normalize(embedding, p=2, dim=1)


def _filter_candidate_communities(
    graph: nx.Graph,
    raw_communities: list[set[str]] | list[list[str]],
    config: DetectorConfig,
) -> list[list[str]]:
    """Filter raw communities using minimum size and density constraints.

    Args:
        graph: Weighted user similarity graph.
        raw_communities: Raw community assignments produced by a mining method.
        config: Detector configuration containing filtering thresholds.

    Returns:
        A filtered list of community member lists.
    """
    communities: list[list[str]] = []
    for members in raw_communities:
        member_list = sorted(str(member) for member in members)
        if len(member_list) < config.initial_min_cluster_size:
            continue
        subgraph = graph.subgraph(member_list)
        density = nx.density(subgraph) if len(member_list) > 1 else 0.0
        if density < config.min_density:
            continue
        communities.append(member_list)
    return communities


def _build_node_feature_matrix(
    graph: nx.Graph,
    profiles: dict[str, dict[str, object]],
    config: DetectorConfig,
) -> tuple[list[str], np.ndarray, list[list[int]]]:
    """Create the feature matrix and adjacency lists for GraphSAGE.

    Args:
        graph: Weighted user similarity graph.
        profiles: Per-user behavioral profiles for the current window.
        config: Detector configuration specifying feature toggles.

    Returns:
        A tuple containing node identifiers, a standardized feature matrix, and
        neighbor index lists.
    """
    nodes = sorted(graph.nodes())
    node_to_index = {node: index for index, node in enumerate(nodes)}
    clustering = nx.clustering(graph, weight="weight") if graph.number_of_edges() else {}

    feature_rows: list[list[float]] = []
    neighbor_lists: list[list[int]] = []
    for node in nodes:
        profile = profiles[node]
        weighted_degree = float(graph.degree(node, weight="weight"))
        degree = float(graph.degree(node))
        avg_edge_weight = weighted_degree / degree if degree > 0 else 0.0
        role_features = (
            [
                float(profile.get("leader_score_window", 0.0)),
                float(profile.get("follower_score_window", 0.0)),
                float(profile.get("role_balance_window", 0.5)),
                float(profile.get("role_consistency_window", 0.0)),
            ]
            if config.graphsage_use_role_features
            else [0.0, 0.0, 0.5, 0.0]
        )
        feature_rows.append(
            [
                float(profile["event_count"]),
                float(len(profile["content_set"])),
                float(profile["mean_dwell"]),
                float(profile["mean_session_events"]),
                float(profile["active_span"]),
                float(profile["burstiness"]),
                float(profile["content_focus"]),
                float(profile["action_entropy"]),
                *role_features,
                degree,
                weighted_degree,
                avg_edge_weight,
                float(clustering.get(node, 0.0)),
            ]
        )
        neighbor_lists.append([node_to_index[neighbor] for neighbor in graph.neighbors(node)])

    features = np.asarray(feature_rows, dtype=np.float32)
    if len(features) > 0:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        features = (features - mean) / std
    return nodes, features, neighbor_lists


def _sample_negative_edges(
    node_count: int,
    positive_edges: set[tuple[int, int]],
    sample_count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Sample non-edge pairs for unsupervised GraphSAGE training.

    Args:
        node_count: Number of graph nodes.
        positive_edges: Existing graph edges expressed as sorted index pairs.
        sample_count: Desired number of negative pairs.
        rng: Random number generator.

    Returns:
        A list of node-index pairs that are not present as positive edges.
    """
    if node_count < 2 or sample_count <= 0:
        return []

    negatives: set[tuple[int, int]] = set()
    max_attempts = max(sample_count * 10, 50)
    attempts = 0
    while len(negatives) < sample_count and attempts < max_attempts:
        left = int(rng.integers(0, node_count))
        right = int(rng.integers(0, node_count))
        attempts += 1
        if left == right:
            continue
        edge = tuple(sorted((left, right)))
        if edge in positive_edges or edge in negatives:
            continue
        negatives.add(edge)
    return list(negatives)


def _train_graphsage_embeddings(
    graph: nx.Graph,
    profiles: dict[str, dict[str, object]],
    config: DetectorConfig,
) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """Train unsupervised GraphSAGE embeddings from one similarity graph.

    Args:
        graph: Weighted user similarity graph.
        profiles: Per-user behavioral profiles for the current window.
        config: Detector configuration containing GraphSAGE hyperparameters.

    Returns:
        A tuple containing node identifiers, learned embeddings, and training
        diagnostics such as the final loss.
    """
    nodes, feature_matrix, neighbors = _build_node_feature_matrix(graph, profiles, config)
    if len(nodes) < config.initial_min_cluster_size:
        return nodes, np.zeros((len(nodes), config.graphsage_embedding_dim), dtype=np.float32), {"loss": 0.0}

    if torch is None:
        return nodes, feature_matrix, {"loss": 0.0}

    device = torch.device("cpu")
    x = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
    model = _GraphSAGEEncoder(
        in_dim=x.shape[1],
        hidden_dim=config.graphsage_hidden_dim,
        embedding_dim=config.graphsage_embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.graphsage_lr)
    rng = np.random.default_rng(config.random_seed)

    node_to_index = {node: index for index, node in enumerate(nodes)}
    positive_edges = {
        tuple(sorted((node_to_index[left], node_to_index[right])))
        for left, right in graph.edges()
    }
    positive_list = sorted(positive_edges)
    positive_weights = np.asarray(
        [
            float(graph[nodes[left]][nodes[right]].get("weight", 0.0))
            for left, right in positive_list
        ],
        dtype=np.float32,
    )
    if len(positive_weights) > 0:
        positive_weights = positive_weights / max(float(positive_weights.max()), 1e-6)
    if not positive_list:
        return nodes, feature_matrix, {"loss": 0.0}

    last_loss = 0.0
    for _ in range(config.graphsage_epochs):
        optimizer.zero_grad()
        embeddings = model(x, neighbors)

        pos_tensor = torch.tensor(positive_list, dtype=torch.long, device=device)
        negative_edges = _sample_negative_edges(
            node_count=len(nodes),
            positive_edges=positive_edges,
            sample_count=max(1, int(len(positive_list) * config.graphsage_negative_ratio)),
            rng=rng,
        )
        neg_tensor = torch.tensor(negative_edges, dtype=torch.long, device=device) if negative_edges else None

        pos_scores = (embeddings[pos_tensor[:, 0]] * embeddings[pos_tensor[:, 1]]).sum(dim=1)
        pos_labels = torch.ones_like(pos_scores)
        pos_weight_tensor = torch.tensor(positive_weights, dtype=torch.float32, device=device)
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight=0.4 + 0.6 * pos_weight_tensor)

        if neg_tensor is not None and len(negative_edges) > 0:
            neg_scores = (embeddings[neg_tensor[:, 0]] * embeddings[neg_tensor[:, 1]]).sum(dim=1)
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        else:
            neg_loss = torch.tensor(0.0, device=device)

        reg_loss = 1e-4 * embeddings.pow(2).mean()
        loss = pos_loss + neg_loss + reg_loss
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())

    with torch.no_grad():
        final_embeddings = model(x, neighbors).cpu().numpy()
    return nodes, final_embeddings, {"loss": last_loss}


def _choose_graphsage_cluster_count(
    embeddings: np.ndarray,
    config: DetectorConfig,
) -> int | None:
    """Choose the number of KMeans clusters using silhouette score.

    Args:
        embeddings: Node embedding matrix.
        config: Detector configuration containing cluster-count bounds.

    Returns:
        The selected number of clusters, or ``None`` when no admissible choice
        satisfies the minimum-size constraints.
    """
    node_count = len(embeddings)
    if node_count < max(3, config.initial_min_cluster_size + 1):
        return None

    max_clusters = min(config.graphsage_max_clusters, node_count - 1)
    if max_clusters < 2:
        return None

    best_k: int | None = None
    best_score = float("-inf")
    for cluster_count in range(2, max_clusters + 1):
        model = KMeans(n_clusters=cluster_count, n_init=4, random_state=config.random_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = model.fit_predict(embeddings)
        if len(set(labels)) <= 1:
            continue
        if min(np.bincount(labels)) < config.initial_min_cluster_size:
            continue
        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > best_score:
            best_score = float(score)
            best_k = cluster_count
    return best_k


def discover_graphsage_communities(
    graph: nx.Graph,
    profiles: dict[str, dict[str, object]],
    config: DetectorConfig,
) -> tuple[list[list[str]], dict[str, float]]:
    """Detect communities from GraphSAGE embeddings followed by clustering.

    Args:
        graph: Weighted user similarity graph.
        profiles: Per-user behavioral profiles for the current window.
        config: Detector configuration containing GraphSAGE settings.

    Returns:
        A tuple containing filtered community member lists and auxiliary
        training statistics.
    """
    if graph.number_of_nodes() < config.initial_min_cluster_size or graph.number_of_edges() == 0:
        return [], {"loss": 0.0, "cluster_count": 0.0}

    nodes, embeddings, train_stats = _train_graphsage_embeddings(graph, profiles, config)
    if len(nodes) < config.initial_min_cluster_size:
        return [], {"loss": float(train_stats.get("loss", 0.0)), "cluster_count": 0.0}

    cluster_count = _choose_graphsage_cluster_count(embeddings, config)
    if cluster_count is None:
        return [], {"loss": float(train_stats.get("loss", 0.0)), "cluster_count": 0.0}

    clusterer = KMeans(n_clusters=cluster_count, n_init=4, random_state=config.random_seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        labels = clusterer.fit_predict(embeddings)
    grouped: dict[int, list[str]] = defaultdict(list)
    for node, label in zip(nodes, labels):
        grouped[int(label)].append(node)

    raw_communities = [members for members in grouped.values()]
    communities = _filter_candidate_communities(graph, raw_communities, config)
    return communities, {
        "loss": float(train_stats.get("loss", 0.0)),
        "cluster_count": float(cluster_count),
        "embedding_dim": float(embeddings.shape[1]) if len(embeddings.shape) > 1 else 0.0,
    }
