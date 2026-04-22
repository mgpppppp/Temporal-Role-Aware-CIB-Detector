from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class CommunityRecord:
    """Structured representation of a detected coordinated community.

    The record stores temporal scope, membership, community-level feature
    aggregates, risk estimates, explanatory metadata, and post-processing
    annotations such as MDCS thresholds and cross-window support.
    """
    community_id: str
    members: list[str]
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    sync_score: float
    popularity_score: float
    residual_score: float
    jaccard_score: float
    dtw_score: float
    session_score: float
    campaign_score: float
    leader_score: float
    density_score: float
    centralization_score: float
    risk_score: float
    risk_level: str
    shared_targets: list[str] = field(default_factory=list)
    top_leaders: list[str] = field(default_factory=list)
    support_windows: int = 1
    window_mdcs: int | None = None
    passes_mdcs: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize the community record into a JSON-friendly dictionary.

        Returns:
            A dictionary containing rounded scalar metrics and auxiliary fields
            suitable for export to JSON or tabular output formats.
        """
        return {
            "community_id": self.community_id,
            "members": self.members,
            "member_count": len(self.members),
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "sync_score": round(self.sync_score, 4),
            "popularity_score": round(self.popularity_score, 4),
            "residual_score": round(self.residual_score, 4),
            "jaccard_score": round(self.jaccard_score, 4),
            "dtw_score": round(self.dtw_score, 4),
            "session_score": round(self.session_score, 4),
            "campaign_score": round(self.campaign_score, 4),
            "leader_score": round(self.leader_score, 4),
            "density_score": round(self.density_score, 4),
            "centralization_score": round(self.centralization_score, 4),
            "risk_score": round(self.risk_score, 4),
            "risk_level": self.risk_level,
            "shared_targets": self.shared_targets,
            "top_leaders": self.top_leaders,
            "support_windows": self.support_windows,
            "window_mdcs": self.window_mdcs,
            "passes_mdcs": self.passes_mdcs,
        }
