"""Elastico controller: queue-length-driven configuration switching.

Implements the runtime adaptation logic from Section V of the Compass paper.
On each request arrival, the queue length is checked against the active
configuration's upscale threshold; on completion, against the next-slower
configuration's downscale threshold. Asymmetric hysteresis prevents
oscillation under fluctuating load.
"""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Union

from compass.planner.aqm import ThresholdPoint, WorkThreshold


@dataclass
class SwitchEvent:
    """A single config switch."""

    timestamp: float
    from_idx: int
    to_idx: int
    queue_length: int
    reason: str


class Controller:
    """Queue-length driven Elastico controller."""

    def __init__(
        self,
        thresholds: List[Union[ThresholdPoint, WorkThreshold]],
        h_time_ms: float = 50.0,
        cooldown_s: float = 5.0,
        upscale_cooldown_s: float = 0.0,
    ):
        if not thresholds:
            raise ValueError("Need at least one threshold")

        self.thresholds = thresholds
        self.h_time_ms = h_time_ms
        self.cooldown_s = cooldown_s
        self.upscale_cooldown_s = upscale_cooldown_s
        self.work_based = isinstance(thresholds[0], WorkThreshold)

        self._lock = Lock()
        self.current_idx = len(thresholds) - 1  # start most accurate
        self.queue_length = 0
        self.last_upscale = 0.0
        self.last_downscale = 0.0
        self.switches: List[SwitchEvent] = []

    @property
    def current(self) -> Union[ThresholdPoint, WorkThreshold]:
        return self.thresholds[self.current_idx]

    @property
    def config(self) -> Dict[str, Any]:
        return self.current.config

    def on_arrival(self, timestamp: float) -> Dict[str, Any]:
        with self._lock:
            self.queue_length += 1
            self._check_upscale(timestamp)
            return self.config

    def on_completion(self, timestamp: float) -> None:
        with self._lock:
            self.queue_length = max(0, self.queue_length - 1)
            self._check_downscale(timestamp)

    def _check_upscale(self, timestamp: float) -> None:
        if self.current_idx == 0:
            return
        if timestamp - self.last_upscale < self.upscale_cooldown_s:
            return

        t = self.current
        if self.work_based:
            should_upscale = self.queue_length * t.mean_ms > t.slack_ms
        else:
            should_upscale = self.queue_length > t.n_up

        if should_upscale:
            self._switch(self.current_idx - 1, "upscale", timestamp)

    def _check_downscale(self, timestamp: float) -> None:
        if self.current_idx == len(self.thresholds) - 1:
            return
        if timestamp - self.last_downscale < self.cooldown_s:
            return

        slower_idx = self.current_idx + 1
        slower = self.thresholds[slower_idx]

        if self.work_based:
            can_downscale = (
                self.queue_length * slower.mean_ms
                <= (slower.slack_ms - self.h_time_ms)
            )
        else:
            t = self.current
            can_downscale = self.queue_length < t.n_down

        if can_downscale:
            self._switch(slower_idx, "downscale", timestamp)

    def _switch(self, new_idx: int, reason: str, timestamp: float) -> None:
        self.switches.append(SwitchEvent(
            timestamp=timestamp,
            from_idx=self.current_idx,
            to_idx=new_idx,
            queue_length=self.queue_length,
            reason=reason,
        ))
        self.current_idx = new_idx
        if reason == "upscale":
            self.last_upscale = timestamp
        else:
            self.last_downscale = timestamp

    def reset(self) -> None:
        with self._lock:
            self.current_idx = len(self.thresholds) - 1
            self.queue_length = 0
            self.last_upscale = 0.0
            self.last_downscale = 0.0
            self.switches.clear()
