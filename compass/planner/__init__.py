"""Compass offline phase 2: deployment planning.

Profile each feasible configuration on the target hardware, extract the
Pareto frontier over (accuracy, P95 latency), and derive AQM-based
switching thresholds for the runtime controller.
"""

from compass.planner.aqm import (
    ThresholdPoint,
    WorkThreshold,
    compute_thresholds,
    compute_work_thresholds,
    load_pareto,
)
from compass.planner.pareto import extract_pareto, select_operating_points
from compass.planner.profiler import LatencyProfile, Profiler

__all__ = [
    "Profiler",
    "LatencyProfile",
    "extract_pareto",
    "select_operating_points",
    "ThresholdPoint",
    "WorkThreshold",
    "compute_thresholds",
    "compute_work_thresholds",
    "load_pareto",
]
