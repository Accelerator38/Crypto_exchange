"""Thin, execution-independent strategy research reset."""

from .dataset import build_feature_tape, load_feature_tape
from .simulator import CostModel, SimulationResult, simulate_targets
from .strategies import build_signal

__all__ = [
    "CostModel",
    "SimulationResult",
    "build_feature_tape",
    "build_signal",
    "load_feature_tape",
    "simulate_targets",
]
