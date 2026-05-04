"""Analytical queuing model — derives switching thresholds from latency profiles.

For each Pareto-optimal configuration the AQM computes the maximum queue
length under which the latency SLO can still be met:

    slack_ms     = SLO_ms - P95_service_ms
    n_up         = floor(slack_ms / mean_service_ms)            (integer thresholds)
    n_down       = max(0, n_up - hysteresis)

Two flavors:
- ThresholdPoint:  integer N_up / N_down (used by the paper's experiments).
- WorkThreshold:   continuous slack budget (alternative formulation).
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ThresholdPoint:
    """Integer-based switching threshold."""

    config: Dict[str, Any]
    model_name: str
    accuracy: float
    mean_ms: float
    p95_ms: float
    n_up: int
    n_down: int
    feasible: bool


@dataclass
class WorkThreshold:
    """Continuous (work-budget) switching threshold."""

    config: Dict[str, Any]
    model_name: str
    accuracy: float
    mean_ms: float
    p95_ms: float
    slack_ms: float
    feasible: bool


def load_pareto(path: str) -> List[Dict]:
    """Load a Pareto frontier JSON, accepting any of the historical schemas."""
    with open(path) as f:
        data = json.load(f)
    return data.get(
        "pareto_frontier",
        data.get("pareto_points", data.get("configs", data)),
    )


def _name(config: Dict[str, Any]) -> str:
    return config.get("generator_model", config.get("detector_model", "unknown"))


def compute_thresholds(
    pareto: List[Dict],
    slo_ms: float,
    hysteresis: int = 2,
) -> List[ThresholdPoint]:
    """Integer thresholds. Switch up when N > n_up; eligible to switch down
    when N < n_down."""
    out = []
    for p in pareto:
        cfg = p.get("config", p)
        mean = p.get("latency_mean_ms", p.get("mean_service_ms", 0))
        p95 = p.get("latency_p95_ms", p.get("p95_service_ms", 0))
        slack = slo_ms - p95

        if slack <= 0 or mean <= 0:
            n_up, feasible = 0, False
        else:
            n_up = int(slack / mean)
            feasible = True

        out.append(ThresholdPoint(
            config=cfg,
            model_name=_name(cfg),
            accuracy=p.get("accuracy", 0),
            mean_ms=mean,
            p95_ms=p95,
            n_up=n_up,
            n_down=max(0, n_up - hysteresis),
            feasible=feasible,
        ))
    return sorted(out, key=lambda x: x.mean_ms)


def compute_work_thresholds(
    pareto: List[Dict],
    slo_ms: float,
) -> List[WorkThreshold]:
    """Continuous thresholds. Switch up when N * mean > slack_ms."""
    out = []
    for p in pareto:
        cfg = p.get("config", p)
        mean = p.get("latency_mean_ms", p.get("mean_service_ms", 0))
        p95 = p.get("latency_p95_ms", p.get("p95_service_ms", 0))
        slack = slo_ms - p95
        out.append(WorkThreshold(
            config=cfg,
            model_name=_name(cfg),
            accuracy=p.get("accuracy", 0),
            mean_ms=mean,
            p95_ms=p95,
            slack_ms=slack,
            feasible=slack > 0,
        ))
    return sorted(out, key=lambda x: x.mean_ms)
