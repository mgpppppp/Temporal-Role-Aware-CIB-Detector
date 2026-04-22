"""Academic prototype for coordinated inauthentic behavior detection."""

from .config import DetectorConfig
from .pipeline import run_pipeline

__all__ = ["DetectorConfig", "run_pipeline"]
