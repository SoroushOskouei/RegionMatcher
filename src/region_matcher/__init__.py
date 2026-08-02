"""Local region-matching inference package."""

from .inference import InferenceOptions, MatchResult, run_inference
from .modeling import RegionMatcher

__all__ = ["InferenceOptions", "MatchResult", "RegionMatcher", "run_inference"]
__version__ = "0.1.0"
