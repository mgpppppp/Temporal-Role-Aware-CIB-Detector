from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _pick_from_topics(
    rng: np.random.Generator,
    content_by_topic: dict[int, list[str]],
    preferred_topics: list[int],
    size: int,
) -> list[str]:
    """Sample unique content items from preferred topical pools.

    Args:
        rng: Random number generator.
        content_by_topic: Mapping from topic identifiers to content lists.
        preferred_topics: Topic identifiers preferred by the current user.
        size: Desired number of sampled content items.

    Returns:
        A list of unique content identifiers sampled without replacement.
    """
    pool: list[str] = []
    for topic in preferred_topics:
        pool.extend(content_by_topic[topic])
    if not pool:
        pool = [item for items in content_by_topic.values() for item in items]
    unique_pool = list(dict.fromkeys(pool))
    if len(unique_pool) <= size:
        return unique_pool
    return rng.choice(unique_pool, size=size, replace=False).tolist()


def _merge_unique(primary: list[str], secondary: list[str], size: int) -> list[str]:
    """Merge two target lists while preserving order and uniqueness.

    Args:
        primary: First ordered target list.
        secondary: Second ordered target list.
        size: Maximum output length.

    Returns:
        A truncated ordered list of unique target identifiers.
    """
    merged = list(dict.fromkeys(primary + secondary))
    return merged[:size]


def _blend(low: float, high: float, strength: float) -> float:
    """Linearly interpolate between two values.

    Args:
        low: Value at camouflage strength ``0.0``.
        high: Value at camouflage strength ``1.0``.
        strength: Camouflage strength in ``[0, 1]``.

    Returns:
        The interpolated scalar value.
    """
    clipped = min(max(float(strength), 0.0), 1.0)
    return float(low + (high - low) * clipped)


def _camouflage_profile(adversarial_strength: float) -> dict[str, float]:
    """Create a synthetic camouflage profile for adversarial bots.

    Args:
        adversarial_strength: Scalar camouflage intensity in ``[0, 1]``.

    Returns:
        A dictionary of generator parameters controlling timing noise, target
        dilution, role disruption, and organic-looking session behavior.
    """
    strength = min(max(float(adversarial_strength), 0.0), 1.0)
    return {
        "sync_anchor_jitter": _blend(0.0, 70.0, strength),
        "step_jitter": _blend(4.0, 30.0, strength),
        "leader_shuffle_rate": _blend(0.05, 0.70, strength),
        "target_dropout_rate": _blend(0.18, 0.60, strength),
        "decoy_insert_rate": _blend(0.22, 0.78, strength),
        "decoy_target_count": _blend(1.0, 4.0, strength),
        "hot_overlap_rate": _blend(0.25, 0.85, strength),
        "hot_overlap_count": _blend(2.0, 6.0, strength),
        "organic_session_bias": _blend(0.15, 0.72, strength),
        "benign_rhythm_shift": _blend(0.10, 0.75, strength),
        "follower_gap_multiplier": _blend(1.0, 2.8, strength),
        "repeat_action_rate": _blend(0.24, 0.08, strength),
        "subteam_target_divergence": _blend(0.12, 0.55, strength),
    }


def _insert_decoys(
    rng: np.random.Generator,
    base_targets: list[str],
    decoys: list[str],
    count: int,
) -> list[str]:
    """Insert decoy targets into an ordered target list.

    Args:
        rng: Random number generator.
        base_targets: Ordered list of coordinated targets.
        decoys: Candidate decoy targets to insert.
        count: Number of decoys to insert.

    Returns:
        A target list with decoys inserted at random positions.
    """
    updated = list(base_targets)
    if not decoys or count <= 0:
        return updated

    for _ in range(min(count, len(decoys))):
        insertion_index = int(rng.integers(0, len(updated) + 1))
        updated = updated[:insertion_index] + [decoys.pop(0)] + updated[insertion_index:]
    return updated


def _sample_bot_action(
    rng: np.random.Generator,
    action_types: list[str],
    benign_rhythm_shift: float,
    filler: bool = False,
) -> str:
    """Sample a bot action under a chosen camouflage regime.

    Args:
        rng: Random number generator.
        action_types: Available action labels.
        benign_rhythm_shift: Strength of organic-looking rhythm imitation.
        filler: Whether the action belongs to a filler session.

    Returns:
        An action label sampled from a camouflage-dependent distribution.
    """
    if filler:
        probs = [0.58, 0.28, 0.14] if benign_rhythm_shift < 0.4 else [0.69, 0.23, 0.08]
    else:
        probs = [0.38, 0.47, 0.15] if benign_rhythm_shift < 0.35 else [0.61, 0.29, 0.10]
    return rng.choice(action_types, p=probs).item()


def _sample_bot_dwell(
    rng: np.random.Generator,
    benign_rhythm_shift: float,
    mode: str,
) -> float:
    """Sample a dwell time under a chosen camouflage regime.

    Args:
        rng: Random number generator.
        benign_rhythm_shift: Strength of organic-looking rhythm imitation.
        mode: Session context such as ``"filler"``, ``"warmup"``,
            ``"burst"``, or ``"repeat"``.

    Returns:
        A non-negative dwell time.
    """
    if mode == "filler":
        mean, std = _blend(15.0, 21.0, benign_rhythm_shift), 6.5
    elif mode == "warmup":
        mean, std = _blend(11.0, 18.0, benign_rhythm_shift), 4.0
    elif mode == "repeat":
        mean, std = _blend(4.0, 7.0, benign_rhythm_shift), 1.8
    else:
        mean, std = _blend(6.5, 13.5, benign_rhythm_shift), 3.0
    return max(1.0 if mode in {"burst", "repeat"} else 2.0, rng.normal(mean, std))


def _camouflaged_targets(
    rng: np.random.Generator,
    active_targets: list[str],
    hot_contents: list[str],
    content_by_topic: dict[int, list[str]],
    topic_count: int,
    camouflage: dict[str, float],
) -> list[str]:
    """Apply target-level camouflage to a coordinated target set.

    Args:
        rng: Random number generator.
        active_targets: Base coordinated targets.
        hot_contents: Globally popular content identifiers.
        content_by_topic: Mapping from topic identifiers to content lists.
        topic_count: Number of synthetic topics.
        camouflage: Camouflage profile parameters.

    Returns:
        A modified target list with dropped targets, decoys, and popular-item
        masking applied.
    """
    targets = list(active_targets)
    if len(targets) > 3 and rng.random() < camouflage["target_dropout_rate"]:
        drop_cap = min(
            max(1, int(round(1 + camouflage["target_dropout_rate"] * 3))),
            len(targets) - 2,
        )
        drop_count = int(rng.integers(0, drop_cap + 1))
        if drop_count > 0:
            dropped = set(rng.choice(targets, size=drop_count, replace=False).tolist())
            targets = [target for target in targets if target not in dropped]

    if rng.random() < camouflage["decoy_insert_rate"]:
        decoys = _pick_from_topics(
            rng,
            content_by_topic,
            [int(rng.integers(0, topic_count))],
            size=max(2, int(round(camouflage["decoy_target_count"])) + 1),
        )
        targets = _insert_decoys(rng, targets, list(decoys), max(1, int(round(camouflage["decoy_target_count"]))))

    if rng.random() < camouflage["hot_overlap_rate"]:
        hot_mask_count = min(len(hot_contents), max(1, int(round(camouflage["hot_overlap_count"]))))
        organic_overlap_targets = rng.choice(hot_contents, size=hot_mask_count, replace=False).tolist()
        targets = _merge_unique(organic_overlap_targets, targets, size=max(len(targets), hot_mask_count))
    return targets


def _sample_burst_step_seconds(
    rng: np.random.Generator,
    is_leader: bool,
    step_jitter: float,
) -> int:
    """Sample an inter-event lag for burst activity.

    Args:
        rng: Random number generator.
        is_leader: Whether the current user acts as the burst initiator.
        step_jitter: Jitter magnitude controlling synchronization degradation.

    Returns:
        An integer number of seconds between consecutive burst actions.
    """
    jitter_ratio = step_jitter / 30.0
    if is_leader:
        low = _blend(7.0, 14.0, jitter_ratio)
        high = _blend(18.0, 40.0, jitter_ratio)
    else:
        low = _blend(10.0, 18.0, jitter_ratio)
        high = _blend(24.0, 58.0, jitter_ratio)
    base = int(rng.integers(int(low), int(high) + 1))
    extra = int(rng.integers(0, max(2, int(round(step_jitter)) + 1)))
    return base + extra


def generate_synthetic_events(
    seed: int = 7,
    benign_users: int = 80,
    bot_group_sizes: list[int] | None = None,
    content_count: int = 180,
    adversarial_strength: float = 0.0,
) -> pd.DataFrame:
    """Generate a synthetic benchmark for coordinated behavior detection.

    The generator mixes benign users, borderline organic promoters, and bot
    groups with configurable levels of adversarial camouflage. Higher
    camouflage strength weakens synchronization, dilutes shared targets,
    disrupts leader-follower structure, and increases overlap with popular
    organic content.

    Args:
        seed: Random seed for reproducibility.
        benign_users: Number of benign users in the generated dataset.
        bot_group_sizes: Sizes of coordinated bot groups.
        content_count: Number of unique content items in the universe.
        adversarial_strength: Camouflage intensity in ``[0, 1]``.

    Returns:
        A chronologically sorted :class:`pandas.DataFrame` containing the
        synthetic event log.
    """
    rng = np.random.default_rng(seed)
    bot_group_sizes = bot_group_sizes or [4, 5, 6]
    camouflage = _camouflage_profile(adversarial_strength)

    base_start = pd.Timestamp("2026-01-01 08:00:00")
    action_types = ["view", "like", "follow"]
    topic_count = 12
    content_ids = [f"content_{idx:04d}" for idx in range(content_count)]
    content_by_topic: dict[int, list[str]] = defaultdict(list)
    for idx, content_id in enumerate(content_ids):
        content_by_topic[idx % topic_count].append(content_id)

    records: list[dict[str, object]] = []
    session_counter = 0

    def add_record(
        user_id: str,
        content_id: str,
        action_type: str,
        timestamp: pd.Timestamp,
        dwell_time: float,
        is_bot: bool,
        true_group: str,
        session_id: str,
        role_tag: str = "organic",
    ) -> None:
        """Append a single event record to the synthetic log.

        Args:
            user_id: User identifier.
            content_id: Content identifier.
            action_type: Action label.
            timestamp: Event timestamp.
            dwell_time: Simulated dwell time in seconds.
            is_bot: Whether the user belongs to a bot group.
            true_group: Ground-truth community label.
            session_id: Session identifier.
            role_tag: Synthetic role annotation.
        """
        records.append(
            {
                "user_id": user_id,
                "content_id": content_id,
                "action_type": action_type,
                "timestamp": timestamp,
                "dwell_time": round(float(dwell_time), 2),
                "session_id": session_id,
                "is_bot": is_bot,
                "true_group": true_group,
                "role_tag": role_tag,
            }
        )

    hot_content_count = max(14, content_count // 10)
    hot_contents = rng.choice(content_ids, size=hot_content_count, replace=False).tolist()
    trend_minutes = [30, 65, 110, 155, 225, 300]
    trend_windows: list[dict[str, object]] = []
    for minute in trend_minutes:
        trend_windows.append(
            {
                "minute": minute,
                "targets": rng.choice(
                    hot_contents,
                    size=min(len(hot_contents), int(rng.integers(3, 6))),
                    replace=False,
                ).tolist(),
                "theme_topic": int(rng.integers(0, topic_count)),
            }
        )

    benign_cohort_size = max(5, benign_users // 8)
    benign_cohorts: list[dict[str, object]] = []
    for cohort_idx in range(4):
        benign_cohorts.append(
            {
                "cohort_id": f"organic_cohort_{cohort_idx}",
                "members": set(
                    f"user_{user_idx:03d}"
                    for user_idx in range(
                        cohort_idx * benign_cohort_size,
                        min((cohort_idx + 1) * benign_cohort_size, benign_users),
                    )
                ),
                "topics": rng.choice(topic_count, size=2, replace=False).tolist(),
                "fan_targets": rng.choice(
                    hot_contents,
                    size=min(len(hot_contents), int(rng.integers(4, 7))),
                    replace=False,
                ).tolist(),
            }
        )

    for user_idx in range(benign_users):
        user_id = f"user_{user_idx:03d}"
        cohort = next((item for item in benign_cohorts if user_id in item["members"]), None)
        cohort_topics = list(cohort["topics"]) if cohort else []
        preferred_topics = cohort_topics or rng.choice(topic_count, size=2, replace=False).tolist()
        if rng.random() < 0.35:
            extra_topic = int(rng.integers(0, topic_count))
            if extra_topic not in preferred_topics:
                preferred_topics.append(extra_topic)

        trend_affinity = float(rng.uniform(0.08, 0.35))
        if cohort:
            trend_affinity += 0.06
        heavy_user = rng.random() < 0.12
        num_sessions = int(rng.integers(5, 10 if not heavy_user else 14))

        for _ in range(num_sessions):
            session_counter += 1
            session_id = f"sess_{session_counter:05d}"
            session_start = base_start + pd.Timedelta(
                minutes=int(rng.integers(0, 360)),
                seconds=int(rng.integers(0, 59)),
            )
            session_length = int(rng.integers(4, 10 if not heavy_user else 15))

            trend_driven = rng.random() < trend_affinity
            if trend_driven:
                chosen_trend = trend_windows[int(rng.integers(0, len(trend_windows)))]
                session_start = base_start + pd.Timedelta(
                    minutes=int(chosen_trend["minute"]) + int(rng.integers(-18, 19)),
                    seconds=int(rng.integers(0, 59)),
                )
                trend_targets = list(chosen_trend["targets"])
                if cohort and rng.random() < 0.55:
                    trend_targets = _merge_unique(list(cohort["fan_targets"]), trend_targets, size=session_length)
                contents = _merge_unique(
                    trend_targets,
                    _pick_from_topics(rng, content_by_topic, preferred_topics, size=session_length),
                    size=session_length,
                )
            else:
                contents = _pick_from_topics(rng, content_by_topic, preferred_topics, size=session_length)

            current_time = session_start
            for position, content_id in enumerate(contents):
                base_gap = int(rng.integers(12, 130 if not heavy_user else 90))
                if trend_driven and position < 3:
                    base_gap = int(rng.integers(10, 60))
                current_time += pd.Timedelta(seconds=base_gap)

                if content_id in hot_contents:
                    action_type = rng.choice(action_types, p=[0.68, 0.24, 0.08]).item()
                else:
                    action_type = rng.choice(action_types, p=[0.76, 0.20, 0.04]).item()

                dwell_mean = 22.0 if not trend_driven else 16.0
                dwell_time = max(2.0, rng.normal(dwell_mean, 9.5))
                add_record(
                    user_id=user_id,
                    content_id=content_id,
                    action_type=action_type,
                    timestamp=current_time,
                    dwell_time=dwell_time,
                    is_bot=False,
                    true_group="benign",
                    session_id=session_id,
                    role_tag="organic",
                )

                if trend_driven and content_id in hot_contents and rng.random() < 0.10:
                    current_time += pd.Timedelta(seconds=int(rng.integers(25, 180)))
                    add_record(
                        user_id=user_id,
                        content_id=content_id,
                        action_type=rng.choice(["like", "follow"], p=[0.8, 0.2]).item(),
                        timestamp=current_time,
                        dwell_time=max(1.0, rng.normal(6.0, 2.5)),
                        is_bot=False,
                        true_group="benign",
                        session_id=session_id,
                        role_tag="organic",
                    )

    borderline_users = max(3, benign_users // 16)
    for borderline_idx in range(borderline_users):
        user_id = f"benign_promoter_{borderline_idx:02d}"
        promoter_targets = rng.choice(hot_contents, size=min(len(hot_contents), 8), replace=False).tolist()
        for window in trend_windows:
            if rng.random() < 0.45:
                session_counter += 1
                session_id = f"sess_{session_counter:05d}"
                current_time = base_start + pd.Timedelta(
                    minutes=int(window["minute"]) + int(rng.integers(-18, 19)),
                    seconds=int(rng.integers(0, 50)),
                )
                contents = _merge_unique(
                    promoter_targets,
                    list(window["targets"]),
                    size=int(rng.integers(3, 5)),
                )
                for content_id in contents:
                    current_time += pd.Timedelta(seconds=int(rng.integers(15, 95)))
                    add_record(
                        user_id=user_id,
                        content_id=content_id,
                        action_type=rng.choice(action_types, p=[0.48, 0.42, 0.10]).item(),
                        timestamp=current_time,
                        dwell_time=max(2.0, rng.normal(10.0, 4.0)),
                        is_bot=False,
                        true_group="benign",
                        session_id=session_id,
                        role_tag="organic_promoter",
                    )

    burst_minutes = [45, 85, 150, 210, 290]
    for group_idx, group_size in enumerate(bot_group_sizes):
        true_group = f"botnet_{group_idx}"
        hot_overlap_count = min(len(hot_contents), max(2, int(round(camouflage["hot_overlap_count"]))))
        hot_overlap = rng.choice(hot_contents, size=hot_overlap_count, replace=False).tolist()
        niche_targets = rng.choice(content_ids, size=int(rng.integers(8, 12)), replace=False).tolist()
        target_pool = _merge_unique(hot_overlap, niche_targets, size=int(rng.integers(10, 15)))
        bot_users = [f"bot_{group_idx}_{member_idx:02d}" for member_idx in range(group_size)]
        subteam_split = max(2, group_size // 2)
        subteams = [bot_users[:subteam_split], bot_users[subteam_split:]]
        subteams = [team for team in subteams if team]
        lead_lag_base = int(rng.integers(6, 18))

        for user_id in bot_users:
            filler_sessions = int(rng.integers(4, 7 + int(round(camouflage["organic_session_bias"] * 4))))
            filler_topics = rng.choice(topic_count, size=2, replace=False).tolist()
            role_tag = "leader" if user_id == bot_users[0] else "follower"
            for _ in range(filler_sessions):
                session_counter += 1
                session_id = f"sess_{session_counter:05d}"
                session_start = base_start + pd.Timedelta(
                    minutes=int(rng.integers(0, 360)),
                    seconds=int(rng.integers(0, 59)),
                )
                session_length = int(rng.integers(4, 9))
                mixed_targets = _merge_unique(
                    rng.choice(target_pool, size=min(len(target_pool), int(rng.integers(1, 3))), replace=False).tolist(),
                    _pick_from_topics(rng, content_by_topic, filler_topics, size=session_length),
                    size=session_length,
                )
                current_time = session_start
                for content_id in mixed_targets:
                    filler_gap_low = int(_blend(18.0, 28.0, camouflage["benign_rhythm_shift"]))
                    filler_gap_high = int(_blend(150.0, 210.0, camouflage["benign_rhythm_shift"]))
                    current_time += pd.Timedelta(seconds=int(rng.integers(filler_gap_low, filler_gap_high)))
                    if rng.random() < camouflage["hot_overlap_rate"]:
                        content_id = rng.choice(hot_contents).item()
                    add_record(
                        user_id=user_id,
                        content_id=content_id,
                        action_type=_sample_bot_action(
                            rng,
                            action_types,
                            camouflage["benign_rhythm_shift"],
                            filler=True,
                        ),
                        timestamp=current_time,
                        dwell_time=_sample_bot_dwell(
                            rng,
                            camouflage["benign_rhythm_shift"],
                            mode="filler",
                        ),
                        is_bot=True,
                        true_group=true_group,
                        session_id=session_id,
                        role_tag=role_tag,
                    )

        for burst_index, burst_minute in enumerate(burst_minutes):
            burst_start = base_start + pd.Timedelta(minutes=burst_minute + int(rng.integers(-6, 7)))
            burst_targets = rng.choice(
                target_pool,
                size=min(len(target_pool), int(rng.integers(4, 6))),
                replace=False,
            ).tolist()
            if rng.random() < 0.25:
                burst_targets = _merge_unique(
                    burst_targets,
                    list(trend_windows[burst_index % len(trend_windows)]["targets"]),
                    size=min(6, len(burst_targets) + 1),
                )

            for team_idx, team_members in enumerate(subteams):
                team_offset = int(rng.integers(0, 45)) + team_idx * int(rng.integers(8, 26))
                team_leader = (
                    rng.choice(team_members).item()
                    if rng.random() < camouflage["leader_shuffle_rate"]
                    else team_members[0]
                )
                subteam_shared_targets = burst_targets.copy()
                divergence_drop = int(
                    rng.integers(
                        0,
                        min(
                            len(subteam_shared_targets),
                            max(1, int(round(len(subteam_shared_targets) * camouflage["subteam_target_divergence"]))),
                        )
                        + 1,
                    )
                )
                if divergence_drop > 0 and len(subteam_shared_targets) - divergence_drop >= 2:
                    drop_targets = set(
                        rng.choice(subteam_shared_targets, size=divergence_drop, replace=False).tolist()
                    )
                    subteam_shared_targets = [target for target in subteam_shared_targets if target not in drop_targets]

                for user_id in team_members:
                    if rng.random() < 0.08:
                        continue

                    session_counter += 1
                    session_id = f"sess_{session_counter:05d}"
                    lag_multiplier = team_members.index(user_id)
                    anchor_noise = int(
                        rng.integers(
                            0,
                            max(2, int(round(camouflage["sync_anchor_jitter"])) + 1),
                        )
                    )
                    user_anchor = burst_start + pd.Timedelta(
                        seconds=(
                            team_offset
                            + int(round(lead_lag_base * lag_multiplier * camouflage["follower_gap_multiplier"]))
                            + int(rng.integers(0, 8))
                            + anchor_noise
                        )
                    )
                    role_tag = "leader" if user_id == team_leader else "amplifier"

                    active_targets = _camouflaged_targets(
                        rng,
                        subteam_shared_targets,
                        hot_contents,
                        content_by_topic,
                        topic_count,
                        camouflage,
                    )

                    warmup_targets = rng.choice(hot_contents, size=min(2, len(hot_contents)), replace=False).tolist()
                    current_time = user_anchor - pd.Timedelta(seconds=int(rng.integers(30, 90)))
                    for content_id in warmup_targets:
                        current_time += pd.Timedelta(
                            seconds=int(
                                rng.integers(
                                    int(_blend(10.0, 14.0, camouflage["benign_rhythm_shift"])),
                                    int(_blend(40.0, 70.0, camouflage["benign_rhythm_shift"])) + 1,
                                )
                            )
                        )
                        add_record(
                            user_id=user_id,
                            content_id=content_id,
                            action_type="view",
                            timestamp=current_time,
                            dwell_time=_sample_bot_dwell(
                                rng,
                                camouflage["benign_rhythm_shift"],
                                mode="warmup",
                            ),
                            is_bot=True,
                            true_group=true_group,
                            session_id=session_id,
                            role_tag=role_tag,
                        )

                    current_time = user_anchor
                    for content_id in active_targets:
                        step_seconds = _sample_burst_step_seconds(
                            rng,
                            is_leader=user_id == team_leader,
                            step_jitter=camouflage["step_jitter"],
                        )
                        current_time += pd.Timedelta(seconds=step_seconds)
                        add_record(
                            user_id=user_id,
                            content_id=content_id,
                            action_type=_sample_bot_action(
                                rng,
                                action_types,
                                camouflage["benign_rhythm_shift"],
                            ),
                            timestamp=current_time,
                            dwell_time=_sample_bot_dwell(
                                rng,
                                camouflage["benign_rhythm_shift"],
                                mode="burst",
                            ),
                            is_bot=True,
                            true_group=true_group,
                            session_id=session_id,
                            role_tag=role_tag,
                        )

                    if active_targets and rng.random() < camouflage["repeat_action_rate"]:
                        current_time += pd.Timedelta(seconds=int(rng.integers(35, 140)))
                        add_record(
                            user_id=user_id,
                            content_id=rng.choice(active_targets).item(),
                            action_type=(
                                rng.choice(["like", "follow"], p=[0.75, 0.25]).item()
                                if camouflage["benign_rhythm_shift"] < 0.45
                                else "like"
                            ),
                            timestamp=current_time,
                            dwell_time=_sample_bot_dwell(
                                rng,
                                camouflage["benign_rhythm_shift"],
                                mode="repeat",
                            ),
                            is_bot=True,
                            true_group=true_group,
                            session_id=session_id,
                            role_tag=role_tag,
                        )

    events = pd.DataFrame.from_records(records)
    events = events.sort_values(["timestamp", "user_id", "content_id"]).reset_index(drop=True)
    return events
