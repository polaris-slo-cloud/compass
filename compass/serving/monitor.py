"""Queue and load monitoring."""

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import List


@dataclass
class Snapshot:
    """Point-in-time snapshot."""
    timestamp: float
    queue_length: int
    config_idx: int
    model: str
    arrival_rate: float


class Monitor:
    """Monitors queue state and arrival rate."""

    def __init__(self, rate_window: float = 5.0):
        self._lock = RLock()
        self._arrivals: deque = deque()
        self._rate_window = rate_window
        self.snapshots: List[Snapshot] = []

    def record_arrival(self, timestamp: float) -> None:
        with self._lock:
            self._arrivals.append(timestamp)
            cutoff = timestamp - self._rate_window
            while self._arrivals and self._arrivals[0] < cutoff:
                self._arrivals.popleft()

    def get_arrival_rate(self, timestamp: float) -> float:
        with self._lock:
            cutoff = timestamp - self._rate_window
            count = sum(1 for t in self._arrivals if t >= cutoff)
            return count / self._rate_window

    def snapshot(
        self,
        timestamp: float,
        queue_length: int,
        config_idx: int,
        model: str,
    ) -> None:
        with self._lock:
            self.snapshots.append(Snapshot(
                timestamp=timestamp,
                queue_length=queue_length,
                config_idx=config_idx,
                model=model,
                arrival_rate=self.get_arrival_rate(timestamp),
            ))

    def summary(self) -> dict:
        if not self.snapshots:
            return {}
        lengths = [s.queue_length for s in self.snapshots]
        return {
            "n_snapshots": len(self.snapshots),
            "avg_queue": sum(lengths) / len(lengths),
            "max_queue": max(lengths),
            "duration_s": self.snapshots[-1].timestamp - self.snapshots[0].timestamp,
        }
