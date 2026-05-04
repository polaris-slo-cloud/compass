"""Load pattern generation."""

import math
import random
from typing import List


class LoadGenerator:
    """Generates arrival times for different load patterns."""

    @staticmethod
    def constant(qps: float, duration: float) -> List[float]:
        """Constant arrival rate."""
        arrivals = []
        t = 0.0
        while t < duration:
            t += random.expovariate(qps)
            if t < duration:
                arrivals.append(t)
        return arrivals

    @staticmethod
    def step(qps_levels: List[float], step_duration: float) -> List[float]:
        """Step function through QPS levels."""
        arrivals = []
        t = 0.0
        for qps in qps_levels:
            step_end = t + step_duration
            if qps > 0:
                while t < step_end:
                    t += random.expovariate(qps)
                    if t < step_end:
                        arrivals.append(t)
            t = step_end
        return arrivals

    @staticmethod
    def ramp(start_qps: float, end_qps: float, duration: float) -> List[float]:
        """Linear ramp between QPS values."""
        arrivals = []
        t = 0.0
        while t < duration:
            progress = t / duration
            current_qps = start_qps + (end_qps - start_qps) * progress
            if current_qps > 0:
                t += random.expovariate(current_qps)
                if t < duration:
                    arrivals.append(t)
            else:
                t += 0.1
        return arrivals

    @staticmethod
    def sine(base_qps: float, amplitude: float, period: float, duration: float) -> List[float]:
        """Sinusoidal load pattern."""
        arrivals = []
        t = 0.0
        while t < duration:
            current_qps = base_qps + amplitude * math.sin(2 * math.pi * t / period)
            current_qps = max(0.1, current_qps)
            t += random.expovariate(current_qps)
            if t < duration:
                arrivals.append(t)
        return arrivals

    @staticmethod
    def spike(
        base_qps: float,
        spike_qps: float,
        base_duration: float,
        spike_duration: float,
        recovery_duration: float,
    ) -> List[float]:
        """Base -> spike -> recovery pattern."""
        arrivals = []

        # Base phase
        t = 0.0
        while t < base_duration:
            t += random.expovariate(base_qps)
            if t < base_duration:
                arrivals.append(t)

        # Spike phase
        spike_start = base_duration
        t = spike_start
        while t < spike_start + spike_duration:
            t += random.expovariate(spike_qps)
            if t < spike_start + spike_duration:
                arrivals.append(t)

        # Recovery phase
        recovery_start = spike_start + spike_duration
        t = recovery_start
        while t < recovery_start + recovery_duration:
            t += random.expovariate(base_qps)
            if t < recovery_start + recovery_duration:
                arrivals.append(t)

        return arrivals

    @staticmethod
    def bursty(
        base_qps: float,
        burst_qps: float,
        duration: float,
        burst_duration_mean: float = 2.0,
        burst_interval_mean: float = 10.0,
    ) -> List[float]:
        """
        Bursty load with random short bursts.

        Args:
            base_qps: Background arrival rate
            burst_qps: Arrival rate during bursts
            duration: Total duration in seconds
            burst_duration_mean: Mean burst length in seconds (default 2s)
            burst_interval_mean: Mean time between bursts in seconds (default 10s)
        """
        arrivals = []
        t = 0.0

        # Generate burst start times and durations
        bursts = []
        burst_start = random.expovariate(1.0 / burst_interval_mean)
        while burst_start < duration:
            burst_len = random.expovariate(1.0 / burst_duration_mean)
            burst_len = min(burst_len, 5.0)  # Cap burst length at 5s
            bursts.append((burst_start, burst_start + burst_len))
            burst_start += burst_len + random.expovariate(1.0 / burst_interval_mean)

        def in_burst(time: float) -> bool:
            for start, end in bursts:
                if start <= time <= end:
                    return True
            return False

        # Generate arrivals
        while t < duration:
            current_qps = burst_qps if in_burst(t) else base_qps
            t += random.expovariate(current_qps)
            if t < duration:
                arrivals.append(t)

        return arrivals
