from __future__ import annotations

from typing import Iterator

import pandas as pd

from .config import DetectorConfig


ACTION_CODE_MAP = {"view": 0, "like": 1, "follow": 2}


def load_events(csv_path: str) -> pd.DataFrame:
    """Load an event log from a CSV file.

    Args:
        csv_path: Path to the CSV file containing the event log.

    Returns:
        A :class:`pandas.DataFrame` with the raw event records.
    """
    return pd.read_csv(csv_path)


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize a raw event table.

    The function enforces the required schema, parses timestamps, injects
    default fields when optional columns are missing, and derives an integer
    action code used by downstream sequence features.

    Args:
        events: Raw event log as a :class:`pandas.DataFrame`.

    Returns:
        A cleaned and standardized event table sorted by time, user, and
        content identifier.

    Raises:
        ValueError: If one or more required input columns are missing.
    """
    df = events.copy()
    required = {"user_id", "content_id", "action_type", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False)
    df["dwell_time"] = df.get("dwell_time", 0.0)
    df["dwell_time"] = pd.to_numeric(df["dwell_time"], errors="coerce").fillna(0.0)

    if "session_id" not in df.columns:
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        gap = df.groupby("user_id")["timestamp"].diff().gt(pd.Timedelta(minutes=30))
        df["session_id"] = df["user_id"] + "_sess_" + gap.groupby(df["user_id"]).cumsum().astype(str)

    if "is_bot" not in df.columns:
        df["is_bot"] = False
    df["is_bot"] = df["is_bot"].astype(bool)

    if "true_group" not in df.columns:
        df["true_group"] = df["user_id"]
    df["true_group"] = df["true_group"].astype(str)

    if "role_tag" not in df.columns:
        df["role_tag"] = "unknown"
    df["role_tag"] = df["role_tag"].astype(str)

    df["action_type"] = df["action_type"].astype(str).str.lower()
    df["action_code"] = df["action_type"].map(ACTION_CODE_MAP).fillna(len(ACTION_CODE_MAP)).astype(int)
    df = df.sort_values(["timestamp", "user_id", "content_id"]).reset_index(drop=True)
    return df


def iter_sliding_windows(
    events: pd.DataFrame,
    config: DetectorConfig,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """Yield non-empty sliding windows over the event stream.

    Args:
        events: Standardized event table sorted by timestamp.
        config: Detector configuration containing window size and step length.

    Yields:
        Tuples of ``(window_start, window_end, window_events)`` describing each
        non-empty temporal slice.
    """
    if events.empty:
        return

    window_size = pd.Timedelta(minutes=config.window_size_minutes)
    window_step = pd.Timedelta(minutes=config.window_step_minutes)

    start = events["timestamp"].min().floor("min")
    last_timestamp = events["timestamp"].max().ceil("min")

    while start <= last_timestamp:
        end = start + window_size
        window_events = events[
            (events["timestamp"] >= start) & (events["timestamp"] < end)
        ].copy()
        if not window_events.empty:
            yield start, end, window_events
        start += window_step
