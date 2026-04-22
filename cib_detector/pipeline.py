from __future__ import annotations

import pandas as pd

from .config import DetectorConfig
from .evaluation import evaluate
from .features import attach_role_features, build_content_weights, build_user_profiles, compute_pairwise_features
from .graph_build import build_bipartite_graph, build_user_similarity_graph, graph_stats
from .mdcs import estimate_window_mdcs
from .mining import discover_graphsage_communities, discover_louvain_communities
from .models import CommunityRecord
from .preprocess import iter_sliding_windows, prepare_events
from .scoring import consolidate_communities, score_window_communities
from .visualize import save_comparison_metrics, save_outputs


def _update_campaign_history(
    pair_history: dict[tuple[str, str], dict[str, object]],
    pairwise_features: pd.DataFrame,
    current_window_index: int,
    config: DetectorConfig,
) -> None:
    """Update pairwise historical memory for campaign persistence features.

    Args:
        pair_history: Mutable historical memory indexed by user pair.
        pairwise_features: Pairwise feature table for the current window.
        current_window_index: Index of the current sliding window.
        config: Detector configuration containing campaign memory settings.
    """
    if pairwise_features.empty:
        return

    for row in pairwise_features.itertuples(index=False):
        pair_strength = 0.5 * float(row.sync) + 0.5 * float(row.popularity)
        if int(row.shared_target_count) == 0 or pair_strength < config.campaign_update_min_strength:
            continue

        pair_key = tuple(sorted((str(row.user_a), str(row.user_b))))
        history = pair_history.setdefault(pair_key, {"windows": [], "target_sets": []})
        history["windows"].append(int(current_window_index))
        history["target_sets"].append(set(row.shared_targets))

        overflow = max(0, len(history["windows"]) - config.campaign_memory_windows)
        if overflow > 0:
            history["windows"] = history["windows"][overflow:]
            history["target_sets"] = history["target_sets"][overflow:]


def _build_method_window_row(
    method: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    window_events: pd.DataFrame,
    bipartite_graph,
    graph_summary: dict[str, float],
    pairwise_features: pd.DataFrame,
    scored_communities: list[CommunityRecord],
    kept_communities: int,
    mdcs: int | None,
    mdcs_details: dict[int, dict[str, float]],
    extra_stats: dict[str, float] | None = None,
) -> dict[str, object]:
    """Create a tabular summary row for one method-window combination.

    Args:
        method: Name of the community discovery method.
        window_start: Start timestamp of the current window.
        window_end: End timestamp of the current window.
        window_events: Event records in the current window.
        bipartite_graph: User-content bipartite graph.
        graph_summary: Precomputed summary statistics for the similarity graph.
        pairwise_features: Pairwise feature table.
        scored_communities: Communities scored within the current window.
        kept_communities: Number of communities retained after filtering.
        mdcs: Estimated MDCS threshold.
        mdcs_details: Detailed null-model thresholds by community size.
        extra_stats: Optional method-specific diagnostics.

    Returns:
        A dictionary suitable for inclusion in the exported window statistics
        table.
    """
    row = {
        "method": method,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "events": int(len(window_events)),
        "active_users": int(window_events["user_id"].nunique()),
        "active_targets": int(window_events["content_id"].nunique()),
        "bipartite_edges": int(bipartite_graph.number_of_edges()),
        "user_graph_nodes": int(graph_summary["nodes"]),
        "user_graph_edges": int(graph_summary["edges"]),
        "user_graph_density": round(float(graph_summary["density"]), 4),
        "user_graph_mean_weight": round(float(graph_summary["mean_weight"]), 4),
        "mean_pair_sync": round(float(pairwise_features["sync"].mean()), 4) if not pairwise_features.empty else 0.0,
        "mean_pair_popularity": round(float(pairwise_features["popularity"].mean()), 4)
        if not pairwise_features.empty
        else 0.0,
        "mean_pair_residual": round(float(pairwise_features["residual"].mean()), 4)
        if not pairwise_features.empty
        else 0.0,
        "mean_pair_campaign": round(float(pairwise_features["campaign"].mean()), 4)
        if not pairwise_features.empty
        else 0.0,
        "mean_pair_leader": round(float(pairwise_features["leader"].mean()), 4)
        if not pairwise_features.empty
        else 0.0,
        "raw_communities": int(len(scored_communities)),
        "kept_communities": int(kept_communities),
        "window_mdcs": mdcs,
        "mdcs_thresholds": str({size: round(detail["threshold"], 4) for size, detail in mdcs_details.items()}),
    }
    if extra_stats:
        row.update({key: round(float(value), 4) for key, value in extra_stats.items()})
    return row


def _compare_metrics(method_results: dict[str, dict[str, object]]) -> dict[str, object]:
    """Compare evaluation metrics across discovery methods.

    Args:
        method_results: Method-specific pipeline outputs keyed by method name.

    Returns:
        A dictionary containing both per-method metrics and pairwise deltas.
    """
    louvain_metrics = method_results["louvain"]["metrics"]
    graphsage_metrics = method_results["graphsage"]["metrics"]
    delta = {}
    for key in sorted(set(louvain_metrics) | set(graphsage_metrics)):
        left = louvain_metrics.get(key)
        right = graphsage_metrics.get(key)
        if isinstance(left, float) and isinstance(right, float):
            delta[f"graphsage_minus_louvain_{key}"] = round(right - left, 6)
    return {
        "louvain": louvain_metrics,
        "graphsage": graphsage_metrics,
        "delta": delta,
    }


def run_pipeline(
    events: pd.DataFrame,
    config: DetectorConfig,
    output_dir: str = "outputs",
) -> dict[str, object]:
    """Execute the full coordinated-behavior detection pipeline.

    Args:
        events: Raw or partially processed event table.
        config: Detector configuration controlling all pipeline stages.
        output_dir: Root directory used to store exported artifacts.

    Returns:
        A dictionary containing method-specific results, cross-method
        comparison metrics, exported output paths, and the prepared event table.
    """
    prepared_events = prepare_events(events)
    content_weights = build_content_weights(prepared_events, config)

    method_states: dict[str, dict[str, object]] = {
        "louvain": {"candidates": [], "window_rows": []},
        "graphsage": {"candidates": [], "window_rows": []},
    }
    pair_history: dict[tuple[str, str], dict[str, object]] = {}

    for window_index, (window_start, window_end, window_events) in enumerate(
        iter_sliding_windows(prepared_events, config)
    ):
        bipartite_graph = build_bipartite_graph(window_events)
        profiles = build_user_profiles(window_events, window_start, config)
        pairwise_features = compute_pairwise_features(
            profiles,
            config,
            content_weights=content_weights,
            pair_history=pair_history,
            current_window_index=window_index,
        )
        attach_role_features(profiles, pairwise_features)
        user_graph = build_user_similarity_graph(pairwise_features, profiles, config)
        graph_summary = graph_stats(user_graph)

        method_to_candidates: dict[str, tuple[list[list[str]], dict[str, float]]] = {
            "louvain": (discover_louvain_communities(user_graph, config), {}),
            "graphsage": discover_graphsage_communities(user_graph, profiles, config),
        }

        for method, (communities, method_extra_stats) in method_to_candidates.items():
            scored_communities = score_window_communities(
                user_graph,
                communities,
                window_start,
                window_end,
                config,
            )
            mdcs, mdcs_details = (
                estimate_window_mdcs(user_graph, scored_communities, config)
                if config.enable_mdcs_filter
                else (None, {})
            )

            filtered_count = 0
            for community in scored_communities:
                community.window_mdcs = mdcs
                community.passes_mdcs = mdcs is None or len(community.members) >= mdcs
                if community.passes_mdcs and community.risk_score >= config.min_output_risk:
                    method_states[method]["candidates"].append(community)
                    filtered_count += 1

            method_states[method]["window_rows"].append(
                _build_method_window_row(
                    method=method,
                    window_start=window_start,
                    window_end=window_end,
                    window_events=window_events,
                    bipartite_graph=bipartite_graph,
                    graph_summary=graph_summary,
                    pairwise_features=pairwise_features,
                    scored_communities=scored_communities,
                    kept_communities=filtered_count,
                    mdcs=mdcs,
                    mdcs_details=mdcs_details,
                    extra_stats=method_extra_stats,
                )
            )

        _update_campaign_history(pair_history, pairwise_features, window_index, config)

    method_results: dict[str, dict[str, object]] = {}
    output_paths: dict[str, object] = {}

    for method, state in method_states.items():
        final_communities = consolidate_communities(state["candidates"], config)
        metrics, account_scores = evaluate(prepared_events, final_communities)
        window_stats = pd.DataFrame(state["window_rows"])
        method_output_dir = f"{output_dir}/{method}"
        method_output_paths = save_outputs(
            method_output_dir,
            final_communities,
            account_scores,
            metrics,
            window_stats,
        )
        method_results[method] = {
            "communities": final_communities,
            "metrics": metrics,
            "account_scores": account_scores,
            "window_stats": window_stats,
            "output_paths": method_output_paths,
        }
        output_paths[method] = method_output_paths

    comparison = _compare_metrics(method_results)
    output_paths["comparison"] = save_comparison_metrics(output_dir, comparison)

    return {
        "methods": method_results,
        "comparison": comparison,
        "output_paths": output_paths,
        "events": prepared_events,
    }
