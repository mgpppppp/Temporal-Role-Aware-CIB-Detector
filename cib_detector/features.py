from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from .config import DetectorConfig


def _normalized_entropy(values: list[object]) -> float:
    """Compute entropy normalized to the interval ``[0, 1]``.

    Args:
        values: Categorical observations drawn from an arbitrary domain.

    Returns:
        The Shannon entropy of the empirical distribution divided by the
        maximum achievable entropy for the observed support size.
    """
    if len(values) <= 1:
        return 0.0

    counts = pd.Series(values).value_counts(normalize=True).to_numpy(dtype=float)
    entropy = -float(np.sum(counts * np.log2(np.clip(counts, 1e-12, None))))
    return float(entropy / math.log2(len(counts))) if len(counts) > 1 else 0.0


def build_user_profiles(
    window_events: pd.DataFrame,
    window_start: pd.Timestamp,
    config: DetectorConfig,
) -> dict[str, dict[str, object]]:
    """Assemble per-user behavioral profiles within one time window.

    Args:
        window_events: Event records assigned to the current sliding window.
        window_start: Timestamp marking the start of the current window.
        config: Detector configuration containing temporal normalization
            parameters.

    Returns:
        A mapping from user identifiers to dictionaries containing target sets,
        temporal sequences, dwell statistics, session summaries, and labels used
        by downstream pairwise feature extraction.
    """
    profiles: dict[str, dict[str, object]] = {}
    window_seconds = max(config.window_size_minutes * 60, 1)

    for user_id, user_events in window_events.groupby("user_id"):
        sorted_events = user_events.sort_values("timestamp")
        content_times: dict[str, list[float]] = defaultdict(list)
        first_content_time: dict[str, float] = {}
        sequence: list[np.ndarray] = []

        for row in sorted_events.itertuples(index=False):
            offset_seconds = (row.timestamp - window_start).total_seconds()
            dwell_norm = min(float(row.dwell_time), 120.0) / 120.0
            action_norm = float(row.action_code) / 4.0
            sequence.append(
                np.array(
                    [
                        offset_seconds / window_seconds,
                        action_norm,
                        dwell_norm,
                    ],
                    dtype=float,
                )
            )
            content_times[str(row.content_id)].append(offset_seconds)
            first_content_time.setdefault(str(row.content_id), offset_seconds)

        event_offsets = [float(item[0] * window_seconds) for item in sequence]
        active_span = max(event_offsets[-1] - event_offsets[0], 1.0) if event_offsets else 1.0
        session_sizes = sorted_events.groupby("session_id").size().tolist()
        action_entropy = _normalized_entropy(sorted_events["action_type"].astype(str).tolist())
        content_focus = len(content_times) / max(len(sorted_events), 1)
        burstiness = min(float(len(sorted_events) / active_span), 1.0)

        profiles[str(user_id)] = {
            "user_id": str(user_id),
            "content_set": set(content_times.keys()),
            "content_times": dict(content_times),
            "first_content_time": first_content_time,
            "sequence": np.stack(sequence) if sequence else np.zeros((0, 3), dtype=float),
            "mean_dwell": float(sorted_events["dwell_time"].mean()),
            "mean_session_events": float(np.mean(session_sizes)) if session_sizes else 0.0,
            "event_count": int(len(sorted_events)),
            "active_span": float(active_span / window_seconds),
            "burstiness": float(burstiness),
            "action_entropy": float(action_entropy),
            "content_focus": float(content_focus),
            "is_bot": bool(sorted_events["is_bot"].max()),
            "true_group": str(sorted_events["true_group"].mode().iloc[0]),
            "role_tag": str(sorted_events["role_tag"].mode().iloc[0]),
        }
    return profiles


def build_content_weights(
    events: pd.DataFrame,
    config: DetectorConfig,
) -> dict[str, float]:
    """Estimate inverse-popularity weights for content items.

    Args:
        events: Full event table used to estimate global content popularity.
        config: Detector configuration containing smoothing constants.

    Returns:
        A mapping from content identifiers to inverse-popularity weights so
        that highly popular targets contribute less to coordination strength.
    """
    if events.empty:
        return {}

    popularity = events.groupby("content_id")["user_id"].nunique().to_dict()
    weights: dict[str, float] = {}
    for content_id, user_count in popularity.items():
        adjusted_count = max(float(user_count) + config.popularity_smoothing, 1.0)
        weights[str(content_id)] = float(1.0 / math.log2(2.0 + adjusted_count))
    return weights


def _weighted_average(items: dict[str, float], content_weights: dict[str, float] | None) -> float:
    """Compute a weighted mean over per-target values.

    Args:
        items: Mapping from content identifiers to scalar values.
        content_weights: Optional inverse-popularity weights.

    Returns:
        The weighted arithmetic mean, or ``0.0`` if no values are provided.
    """
    if not items:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for content_id, value in items.items():
        weight = float(content_weights.get(content_id, 1.0)) if content_weights else 1.0
        weighted_sum += weight * float(value)
        total_weight += weight
    return float(weighted_sum / total_weight) if total_weight > 0 else 0.0


def _weighted_mass(items: set[str], content_weights: dict[str, float] | None) -> float:
    """Compute the total weighted mass of a target set.

    Args:
        items: Set of content identifiers.
        content_weights: Optional inverse-popularity weights.

    Returns:
        The sum of weights assigned to the provided targets.
    """
    if not items:
        return 0.0
    return float(
        sum(float(content_weights.get(content_id, 1.0)) if content_weights else 1.0 for content_id in items)
    )


def weighted_jaccard_similarity(
    content_set_a: set[str],
    content_set_b: set[str],
    content_weights: dict[str, float] | None,
) -> float:
    """Compute weighted Jaccard similarity between two target sets.

    Args:
        content_set_a: First content set.
        content_set_b: Second content set.
        content_weights: Optional inverse-popularity weights.

    Returns:
        A weighted Jaccard coefficient in the interval ``[0, 1]``.
    """
    union = content_set_a | content_set_b
    if not union:
        return 0.0

    intersection = content_set_a & content_set_b
    numerator = sum(float(content_weights.get(content_id, 1.0)) if content_weights else 1.0 for content_id in intersection)
    denominator = sum(float(content_weights.get(content_id, 1.0)) if content_weights else 1.0 for content_id in union)
    return float(numerator / denominator) if denominator > 0 else 0.0


def build_window_background_stats(
    profiles: dict[str, dict[str, object]],
    content_index: dict[str, set[str]],
    content_weights: dict[str, float] | None,
) -> dict[str, object]:
    """Summarize background exposure statistics for one time window.

    Args:
        profiles: Per-user behavioral profiles for the current window.
        content_index: Inverted index from content identifiers to exposed users.
        content_weights: Optional inverse-popularity weights.

    Returns:
        Aggregate statistics describing activity volume, target coverage, and
        empirical target hit probabilities.
    """
    active_user_count = max(len(profiles), 1)
    active_target_set = {content_id for profile in profiles.values() for content_id in profile["content_set"]}
    total_weighted_target_mass = max(_weighted_mass(active_target_set, content_weights), 1.0)
    mean_event_count = float(np.mean([profile["event_count"] for profile in profiles.values()])) if profiles else 0.0
    mean_weighted_mass = (
        float(np.mean([_weighted_mass(profile["content_set"], content_weights) for profile in profiles.values()]))
        if profiles
        else 0.0
    )
    content_hit_prob = {
        str(content_id): float(len(users) / active_user_count)
        for content_id, users in content_index.items()
    }
    mean_content_hit_prob = float(np.mean(list(content_hit_prob.values()))) if content_hit_prob else 0.0
    return {
        "active_user_count": float(active_user_count),
        "active_target_count": float(len(active_target_set)),
        "total_weighted_target_mass": float(total_weighted_target_mass),
        "mean_event_count": float(mean_event_count),
        "mean_weighted_mass": float(mean_weighted_mass),
        "content_hit_prob": content_hit_prob,
        "mean_content_hit_prob": float(mean_content_hit_prob),
    }


def residual_coordination_score(
    profile_a: dict[str, object],
    profile_b: dict[str, object],
    shared_targets: list[str],
    sync_score: float,
    popularity_score: float,
    jaccard_score: float,
    campaign_score: float,
    background_stats: dict[str, object],
    content_weights: dict[str, float] | None,
    config: DetectorConfig,
) -> tuple[float, float]:
    """Estimate residual coordination beyond background exposure.

    Args:
        profile_a: Behavioral profile for the first user.
        profile_b: Behavioral profile for the second user.
        shared_targets: Targets that satisfy the synchronization criterion.
        sync_score: Temporal synchronization score.
        popularity_score: Weighted overlap score.
        jaccard_score: Unweighted overlap score.
        campaign_score: Cross-window campaign persistence score.
        background_stats: Window-level baseline statistics.
        content_weights: Optional inverse-popularity weights.
        config: Detector configuration containing residual model coefficients.

    Returns:
        A tuple containing the residual coordination score and the expected
        coordination level implied by the background model.
    """
    observed_coordination = float(
        0.35 * sync_score
        + 0.25 * popularity_score
        + 0.20 * jaccard_score
        + 0.20 * campaign_score
    )

    weighted_mass_a = _weighted_mass(profile_a["content_set"], content_weights)
    weighted_mass_b = _weighted_mass(profile_b["content_set"], content_weights)
    total_weighted_target_mass = max(float(background_stats["total_weighted_target_mass"]), 1.0)
    exposure_a = min(1.0, weighted_mass_a / total_weighted_target_mass)
    exposure_b = min(1.0, weighted_mass_b / total_weighted_target_mass)
    activity_overlap = min(1.0, math.sqrt(exposure_a * exposure_b))

    mean_event_count = max(float(background_stats["mean_event_count"]), 1.0)
    normalized_activity = min(
        1.0,
        math.sqrt(float(profile_a["event_count"]) * float(profile_b["event_count"])) / mean_event_count,
    )

    hit_prob_lookup: dict[str, float] = background_stats["content_hit_prob"]  # type: ignore[assignment]
    shared_target_popularity = (
        float(np.mean([hit_prob_lookup.get(target, float(background_stats["mean_content_hit_prob"])) for target in shared_targets]))
        if shared_targets
        else float(background_stats["mean_content_hit_prob"])
    )

    expected_coordination = float(
        min(
            1.0,
            config.residual_shared_target_weight * activity_overlap
            + config.residual_activity_weight * normalized_activity
            + config.residual_popularity_weight * shared_target_popularity,
        )
    )
    residual_score = float(max(config.residual_floor, observed_coordination - expected_coordination))
    return residual_score, expected_coordination


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute average dynamic time warping distance between two sequences.

    Args:
        seq_a: First behavioral sequence.
        seq_b: Second behavioral sequence.

    Returns:
        The length-normalized dynamic time warping distance, or infinity if one
        of the sequences is empty.
    """
    if len(seq_a) == 0 or len(seq_b) == 0:
        return float("inf")

    n_rows, n_cols = len(seq_a), len(seq_b)
    dp = np.full((n_rows + 1, n_cols + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            cost = float(np.linalg.norm(seq_a[i - 1] - seq_b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[n_rows, n_cols] / max(n_rows + n_cols, 1))


def dtw_similarity(seq_a: np.ndarray, seq_b: np.ndarray, sigma: float = 0.8) -> float:
    """Convert dynamic time warping distance into a similarity score.

    Args:
        seq_a: First behavioral sequence.
        seq_b: Second behavioral sequence.
        sigma: Scale parameter controlling exponential decay.

    Returns:
        A similarity score in the interval ``[0, 1]``.
    """
    distance = _dtw_distance(seq_a, seq_b)
    if math.isinf(distance):
        return 0.0
    return float(math.exp(-distance / max(sigma, 1e-6)))


def jaccard_similarity(content_set_a: set[str], content_set_b: set[str]) -> float:
    """Compute standard Jaccard similarity between two content sets.

    Args:
        content_set_a: First content set.
        content_set_b: Second content set.

    Returns:
        The unweighted Jaccard coefficient.
    """
    union = content_set_a | content_set_b
    if not union:
        return 0.0
    return float(len(content_set_a & content_set_b) / len(union))


def synchronization_score(
    content_times_a: dict[str, list[float]],
    content_times_b: dict[str, list[float]],
    sync_tolerance_seconds: int,
    content_weights: dict[str, float] | None = None,
    tau_seconds: float | None = None,
    min_match_strength: float | None = None,
) -> tuple[float, list[str]]:
    """Measure temporal synchronization across shared targets.

    Args:
        content_times_a: Per-target timestamps for the first user.
        content_times_b: Per-target timestamps for the second user.
        sync_tolerance_seconds: Characteristic synchronization tolerance.
        content_weights: Optional inverse-popularity weights.
        tau_seconds: Exponential decay parameter for time differences.
        min_match_strength: Minimum per-target strength required to count a
            target as synchronized.

    Returns:
        A tuple containing the weighted synchronization score and the list of
        targets whose best temporal match exceeds the threshold.
    """
    shared_contents = sorted(set(content_times_a) & set(content_times_b))
    if not shared_contents:
        return 0.0, []

    decay_tau = max(float(tau_seconds or sync_tolerance_seconds or 1.0), 1e-6)
    match_threshold = (
        float(min_match_strength)
        if min_match_strength is not None
        else float(math.exp(-float(sync_tolerance_seconds) / decay_tau))
    )

    matched_contents: list[str] = []
    per_content_scores: dict[str, float] = {}
    for content_id in shared_contents:
        best_strength = 0.0
        for time_a in content_times_a[content_id]:
            for time_b in content_times_b[content_id]:
                strength = float(math.exp(-abs(time_a - time_b) / decay_tau))
                if strength > best_strength:
                    best_strength = strength
        per_content_scores[content_id] = best_strength
        if best_strength >= match_threshold:
            matched_contents.append(content_id)

    return _weighted_average(per_content_scores, content_weights), matched_contents


def leader_follower_score(
    profile_a: dict[str, object],
    profile_b: dict[str, object],
    shared_targets: list[str],
    config: DetectorConfig,
    content_weights: dict[str, float] | None = None,
) -> tuple[float, float, float, float]:
    """Estimate directional leader-follower dynamics between two users.

    Args:
        profile_a: Behavioral profile for the first user.
        profile_b: Behavioral profile for the second user.
        shared_targets: Targets that satisfy the synchronization criterion.
        config: Detector configuration containing lag constraints.
        content_weights: Optional inverse-popularity weights.

    Returns:
        A tuple containing leader score, direction sign, mean lag, and
        directional dominance.
    """
    if not config.enable_leader_feature:
        return 0.0, 0.0, 0.0, 0.0

    if len(shared_targets) < config.leader_min_shared_targets:
        return 0.0, 0.0, 0.0, 0.0

    min_lag = max(config.leader_min_lag_seconds, 0.0)
    max_lag = max(config.leader_max_lag_seconds, min_lag + 1.0)
    lead_mass_a = 0.0
    lead_mass_b = 0.0
    directional_mass = 0.0
    lag_values: list[float] = []

    first_times_a: dict[str, float] = profile_a["first_content_time"]  # type: ignore[assignment]
    first_times_b: dict[str, float] = profile_b["first_content_time"]  # type: ignore[assignment]

    for target in shared_targets:
        if target not in first_times_a or target not in first_times_b:
            continue
        lag = float(first_times_b[target] - first_times_a[target])
        abs_lag = abs(lag)
        if abs_lag < min_lag or abs_lag > max_lag:
            continue

        weight = float(content_weights.get(target, 1.0)) if content_weights else 1.0
        strength = weight * (1.0 - (abs_lag - min_lag) / max(max_lag - min_lag, 1e-6))
        lag_values.append(abs_lag)
        directional_mass += strength
        if lag > 0:
            lead_mass_a += strength
        else:
            lead_mass_b += strength

    if directional_mass <= 0:
        return 0.0, 0.0, 0.0, 0.0

    dominant_mass = max(lead_mass_a, lead_mass_b)
    direction = 1.0 if lead_mass_a > lead_mass_b else -1.0 if lead_mass_b > lead_mass_a else 0.0
    coverage = min(1.0, directional_mass / max(float(len(shared_targets)), 1.0))
    dominance = dominant_mass / directional_mass
    mean_lag = float(np.mean(lag_values)) if lag_values else 0.0
    lag_strength = 1.0 - min(1.0, mean_lag / max_lag)
    score = float(coverage * dominance * (0.6 + 0.4 * lag_strength))
    return score, direction, mean_lag, dominance


def session_similarity(profile_a: dict[str, object], profile_b: dict[str, object]) -> float:
    """Compare two users through session-level behavioral rhythm.

    Args:
        profile_a: Behavioral profile for the first user.
        profile_b: Behavioral profile for the second user.

    Returns:
        A bounded similarity score derived from dwell time, session size, and
        event-count differences.
    """
    dwell_gap = abs(float(profile_a["mean_dwell"]) - float(profile_b["mean_dwell"])) / 30.0
    session_gap = abs(float(profile_a["mean_session_events"]) - float(profile_b["mean_session_events"])) / 10.0
    event_gap = abs(int(profile_a["event_count"]) - int(profile_b["event_count"])) / 15.0
    return float(1.0 / (1.0 + dwell_gap + session_gap + event_gap))


def compute_pairwise_features(
    profiles: dict[str, dict[str, object]],
    config: DetectorConfig,
    content_weights: dict[str, float] | None = None,
    pair_history: dict[tuple[str, str], dict[str, object]] | None = None,
    current_window_index: int = 0,
) -> pd.DataFrame:
    """Compute pairwise coordination features for all candidate user pairs.

    Args:
        profiles: Per-user behavioral profiles for the current window.
        config: Detector configuration.
        content_weights: Optional inverse-popularity weights.
        pair_history: Historical pair memory used for campaign persistence.
        current_window_index: Index of the current sliding window.

    Returns:
        A :class:`pandas.DataFrame` whose rows describe pairwise coordination
        strength, overlap, role dynamics, and background-adjusted residuals.
    """
    if len(profiles) < 2:
        return pd.DataFrame()

    content_index: dict[str, set[str]] = defaultdict(set)
    for user_id, profile in profiles.items():
        for content_id in profile["content_set"]:
            content_index[str(content_id)].add(user_id)

    background_stats = build_window_background_stats(
        profiles=profiles,
        content_index=content_index,
        content_weights=content_weights,
    )

    candidate_pairs: set[tuple[str, str]] = set()
    for users in content_index.values():
        for user_a, user_b in combinations(sorted(users), 2):
            candidate_pairs.add((user_a, user_b))

    rows: list[dict[str, object]] = []
    for user_a, user_b in sorted(candidate_pairs):
        profile_a = profiles[user_a]
        profile_b = profiles[user_b]
        sync_score, matched_contents = synchronization_score(
            profile_a["content_times"],
            profile_b["content_times"],
            config.sync_tolerance_seconds,
            content_weights=content_weights,
            tau_seconds=config.sync_decay_tau_seconds,
            min_match_strength=config.sync_match_min_strength,
        )
        popularity_score = weighted_jaccard_similarity(
            profile_a["content_set"],
            profile_b["content_set"],
            content_weights,
        )
        jaccard_score = jaccard_similarity(profile_a["content_set"], profile_b["content_set"])
        dtw_score = dtw_similarity(profile_a["sequence"], profile_b["sequence"])
        session_score = session_similarity(profile_a, profile_b)
        leader_score, lead_direction, mean_lag, lead_dominance = leader_follower_score(
            profile_a=profile_a,
            profile_b=profile_b,
            shared_targets=matched_contents,
            config=config,
            content_weights=content_weights,
        )
        campaign_score = campaign_consistency_score(
            user_a=user_a,
            user_b=user_b,
            shared_targets=matched_contents,
            pair_history=pair_history,
            content_weights=content_weights,
            current_window_index=current_window_index,
            config=config,
        )
        residual_score, expected_coordination = residual_coordination_score(
            profile_a=profile_a,
            profile_b=profile_b,
            shared_targets=matched_contents,
            sync_score=sync_score,
            popularity_score=popularity_score,
            jaccard_score=jaccard_score,
            campaign_score=campaign_score,
            background_stats=background_stats,
            content_weights=content_weights,
            config=config,
        )
        if not config.enable_residual_feature:
            residual_score = 0.0

        rows.append(
            {
                "user_a": user_a,
                "user_b": user_b,
                "sync": sync_score,
                "popularity": popularity_score,
                "jaccard": jaccard_score,
                "dtw": dtw_score,
                "session": session_score,
                "campaign": campaign_score,
                "residual": residual_score,
                "leader": leader_score,
                "lead_direction": lead_direction,
                "mean_lead_lag": mean_lag,
                "lead_dominance": lead_dominance,
                "expected_coordination": expected_coordination,
                "shared_targets": matched_contents,
                "shared_target_count": len(matched_contents),
            }
        )

    return pd.DataFrame(rows)


def attach_role_features(
    profiles: dict[str, dict[str, object]],
    pairwise_features: pd.DataFrame,
) -> None:
    """Attach window-level role summaries to user profiles.

    Args:
        profiles: Mutable mapping of user profiles for the current window.
        pairwise_features: Pairwise feature table containing leader-follower
            statistics.
    """
    if pairwise_features.empty or "leader" not in pairwise_features.columns:
        for profile in profiles.values():
            profile["leader_score_window"] = 0.0
            profile["follower_score_window"] = 0.0
            profile["role_balance_window"] = 0.5
            profile["role_consistency_window"] = 0.0
        return

    role_stats = {
        user_id: {
            "leader_score_window": 0.0,
            "follower_score_window": 0.0,
            "role_consistency_window": 0.0,
            "pair_count_window": 0.0,
        }
        for user_id in profiles
    }

    for row in pairwise_features.itertuples(index=False):
        role_strength = float(row.leader)
        if role_strength <= 0:
            continue

        user_a = str(row.user_a)
        user_b = str(row.user_b)
        role_stats[user_a]["pair_count_window"] += 1.0
        role_stats[user_b]["pair_count_window"] += 1.0
        if float(row.lead_direction) > 0:
            role_stats[user_a]["leader_score_window"] += role_strength
            role_stats[user_b]["follower_score_window"] += role_strength
            role_stats[user_a]["role_consistency_window"] += float(row.lead_dominance)
            role_stats[user_b]["role_consistency_window"] += float(row.lead_dominance)
        elif float(row.lead_direction) < 0:
            role_stats[user_b]["leader_score_window"] += role_strength
            role_stats[user_a]["follower_score_window"] += role_strength
            role_stats[user_a]["role_consistency_window"] += float(row.lead_dominance)
            role_stats[user_b]["role_consistency_window"] += float(row.lead_dominance)

    for user_id, profile in profiles.items():
        stats = role_stats[user_id]
        leader_score = float(stats["leader_score_window"])
        follower_score = float(stats["follower_score_window"])
        pair_count = max(float(stats["pair_count_window"]), 1.0)
        profile["leader_score_window"] = leader_score
        profile["follower_score_window"] = follower_score
        profile["role_balance_window"] = float(leader_score / max(leader_score + follower_score, 1e-6))
        profile["role_consistency_window"] = float(stats["role_consistency_window"] / pair_count)


def _weighted_target_overlap(
    current_targets: set[str],
    historical_targets: set[str],
    content_weights: dict[str, float] | None,
) -> float:
    """Measure weighted overlap between current and historical targets.

    Args:
        current_targets: Targets observed in the current window.
        historical_targets: Targets observed in previous windows.
        content_weights: Optional inverse-popularity weights.

    Returns:
        The fraction of current weighted target mass that was also observed in
        the historical target set.
    """
    if not current_targets:
        return 0.0

    numerator = sum(
        float(content_weights.get(content_id, 1.0)) if content_weights else 1.0
        for content_id in (current_targets & historical_targets)
    )
    denominator = sum(
        float(content_weights.get(content_id, 1.0)) if content_weights else 1.0
        for content_id in current_targets
    )
    return float(numerator / denominator) if denominator > 0 else 0.0


def campaign_consistency_score(
    user_a: str,
    user_b: str,
    shared_targets: list[str],
    pair_history: dict[tuple[str, str], dict[str, object]] | None,
    content_weights: dict[str, float] | None,
    current_window_index: int,
    config: DetectorConfig,
) -> float:
    """Quantify repeated coordination across multiple time windows.

    Args:
        user_a: First user identifier.
        user_b: Second user identifier.
        shared_targets: Synchronized targets in the current window.
        pair_history: Historical memory for the user pair.
        content_weights: Optional inverse-popularity weights.
        current_window_index: Index of the current sliding window.
        config: Detector configuration containing campaign decay parameters.

    Returns:
        A score in ``[0, 1]`` describing how consistently the pair has
        coordinated across recent windows.
    """
    if not config.enable_campaign_feature:
        return 0.0

    if not shared_targets or not pair_history:
        return 0.0

    pair_key = tuple(sorted((str(user_a), str(user_b))))
    history = pair_history.get(pair_key)
    if not history:
        return 0.0

    previous_windows = history.get("windows", [])
    previous_target_sets = history.get("target_sets", [])
    if not previous_windows:
        return 0.0

    current_targets = set(shared_targets)
    historical_targets = set().union(*previous_target_sets) if previous_target_sets else set()
    target_reuse_score = _weighted_target_overlap(current_targets, historical_targets, content_weights)

    repeat_score = min(
        1.0,
        len(previous_windows) / max(float(config.campaign_repeat_saturation), 1.0),
    )
    window_gap = max(0, current_window_index - int(previous_windows[-1]))
    recency_score = float(math.exp(-window_gap / max(config.campaign_recency_decay, 1.0)))
    temporal_repeat = repeat_score * recency_score
    return float(min(1.0, 0.55 * target_reuse_score + 0.45 * temporal_repeat))
