"""Pareto frontier extraction over (accuracy, latency)."""

from typing import Any, Dict, List, Optional


def is_dominated(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """True iff b dominates a: b has higher accuracy AND lower latency,
    with at least one strict inequality."""
    if b["accuracy"] >= a["accuracy"] and b["latency_p95_ms"] <= a["latency_p95_ms"]:
        if b["accuracy"] > a["accuracy"] or b["latency_p95_ms"] < a["latency_p95_ms"]:
            return True
    return False


def extract_pareto(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pareto-optimal configs maximizing accuracy and minimizing P95 latency.

    Each profile dict must contain `accuracy` and `latency_p95_ms`. Returns
    a list sorted by latency (fastest first).
    """
    out = []
    for i, a in enumerate(profiles):
        dominated = any(is_dominated(a, b) for j, b in enumerate(profiles) if i != j)
        if not dominated:
            out.append(a)
    return sorted(out, key=lambda x: x["latency_p95_ms"])


def select_operating_points(
    pareto: List[Dict[str, Any]],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Pick the canonical (fastest, balanced, most accurate) operating points
    used for static-baseline comparisons."""
    if not pareto:
        return None
    fastest = min(pareto, key=lambda x: x["latency_p95_ms"])
    most_accurate = max(pareto, key=lambda x: x["accuracy"])
    balanced = max(
        pareto,
        key=lambda x: (
            x["accuracy"] / x["latency_p95_ms"] if x["latency_p95_ms"] > 0 else 0
        ),
    )
    return {
        "fastest": fastest,
        "balanced": balanced,
        "most_accurate": most_accurate,
    }
