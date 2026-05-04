"""Latency profiling for feasible configurations.

Measures end-to-end execution latency on the target deployment hardware.
Profiles include warmup runs (discarded) and measurement runs (recorded as
mean / P50 / P95 / P99 / std). Workflow-agnostic: the profiler drives any
object that exposes `configure(config)` and `execute(sample)`.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Protocol

import numpy as np
from tqdm import tqdm


class _ExecutorLike(Protocol):
    def configure(self, config: Dict[str, Any]) -> None: ...
    def execute(self, sample: Any) -> Any: ...


@dataclass
class LatencyProfile:
    """Latency statistics for one configuration."""

    config: Dict[str, Any]
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    n_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "latency_mean_ms": self.mean_ms,
            "latency_p50_ms": self.p50_ms,
            "latency_p95_ms": self.p95_ms,
            "latency_p99_ms": self.p99_ms,
            "latency_std_ms": self.std_ms,
            "n_samples": self.n_samples,
        }


class Profiler:
    """End-to-end latency profiler.

    Args:
        executor: object with `configure(config)` + `execute(sample)`.
        samples: dataset samples to drive the executor.
        n_warmup: warmup runs per config (discarded).
        n_samples: measurement runs per config.
        seed: RNG seed for reproducible sample selection.
    """

    def __init__(
        self,
        executor: _ExecutorLike,
        samples: List[Any],
        n_warmup: int = 3,
        n_samples: int = 50,
        seed: int = 42,
    ):
        self.executor = executor
        self.n_warmup = n_warmup
        self.n_samples = n_samples

        rng = np.random.RandomState(seed)
        n_needed = n_warmup + n_samples
        if len(samples) < n_needed:
            indices = (list(range(len(samples))) * (n_needed // len(samples) + 1))[:n_needed]
        else:
            indices = rng.choice(len(samples), size=n_needed, replace=False).tolist()

        self.warmup_samples = [samples[i] for i in indices[:n_warmup]]
        self.measure_samples = [samples[i] for i in indices[n_warmup:n_needed]]

    def profile_config(self, config: Dict[str, Any]) -> LatencyProfile:
        self.executor.configure(config)

        for s in self.warmup_samples:
            self.executor.execute(s)

        latencies_ms: List[float] = []
        for s in self.measure_samples:
            t0 = time.perf_counter()
            self.executor.execute(s)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

        arr = np.asarray(latencies_ms)
        return LatencyProfile(
            config=config,
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            std_ms=float(arr.std()),
            n_samples=len(arr),
        )

    def profile_all(
        self,
        configs: List[Dict[str, Any]],
        verbose: bool = True,
    ) -> List[LatencyProfile]:
        return [
            self.profile_config(c)
            for c in tqdm(configs, desc="Profiling", disable=not verbose)
        ]
