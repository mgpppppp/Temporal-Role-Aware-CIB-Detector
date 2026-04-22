from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field


@dataclass
class DetectorConfig:
    """Central configuration object for the detection pipeline.

    The configuration consolidates temporal window parameters, graph
    construction thresholds, role-modeling settings, GraphSAGE hyperparameters,
    and feature toggles used for ablation studies.
    """
    window_size_minutes: int = 5
    window_step_minutes: int = 1
    sync_tolerance_seconds: int = 20
    sync_decay_tau_seconds: float = 20.0
    sync_match_min_strength: float = 0.37
    popularity_smoothing: float = 1.0
    residual_floor: float = 0.0
    residual_shared_target_weight: float = 0.45
    residual_activity_weight: float = 0.20
    residual_popularity_weight: float = 0.35
    edge_quantile: float = 0.90
    min_edge_weight: float = 0.35
    initial_min_cluster_size: int = 2
    min_density: float = 0.45
    min_output_risk: float = 0.55
    louvain_resolution: float = 1.0
    overlap_merge_threshold: float = 0.60
    null_trials: int = 12
    mdcs_zscore: float = 2.5
    mdcs_min_observed_risk: float = 0.60
    campaign_memory_windows: int = 12
    campaign_repeat_saturation: int = 4
    campaign_recency_decay: float = 4.0
    campaign_update_min_strength: float = 0.35
    leader_min_lag_seconds: float = 3.0
    leader_max_lag_seconds: float = 90.0
    leader_min_shared_targets: int = 2
    graphsage_hidden_dim: int = 20
    graphsage_embedding_dim: int = 14
    graphsage_epochs: int = 16
    graphsage_lr: float = 0.01
    graphsage_negative_ratio: float = 1.5
    graphsage_max_clusters: int = 5
    enable_campaign_feature: bool = True
    enable_residual_feature: bool = True
    enable_leader_feature: bool = True
    enable_mdcs_filter: bool = True
    graphsage_use_role_features: bool = True
    random_seed: int = 7
    similarity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sync": 0.24,
            "popularity": 0.12,
            "jaccard": 0.10,
            "dtw": 0.14,
            "session": 0.05,
            "campaign": 0.12,
            "residual": 0.13,
            "leader": 0.20,
        }
    )
    risk_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sync": 0.20,
            "popularity": 0.10,
            "jaccard": 0.08,
            "dtw": 0.12,
            "density": 0.14,
            "campaign": 0.12,
            "residual": 0.12,
            "leader": 0.18,
            "centralization": 0.14,
        }
    )

    def clone_with(self, **updates: object) -> "DetectorConfig":
        """Create a modified copy of the current configuration.

        Args:
            **updates: Field overrides to apply to the cloned configuration.

        Returns:
            A new :class:`DetectorConfig` instance containing the original
            values plus the requested updates.
        """
        payload = asdict(self)
        payload.update(updates)
        return DetectorConfig(**payload)
