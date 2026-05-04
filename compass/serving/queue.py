"""Thread-safe request queue."""

from dataclasses import dataclass, field
from queue import Queue, Empty
from threading import Lock
from typing import Any, Dict, List, Optional


@dataclass
class Request:
    """A request in the system."""
    id: int
    question: str
    ground_truths: List[str]  # All valid answers
    arrival_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    config_idx: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""

    @property
    def wait_time_ms(self) -> float:
        return (self.start_time - self.arrival_time) * 1000

    @property
    def service_time_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def response_time_ms(self) -> float:
        return (self.end_time - self.arrival_time) * 1000


class RequestQueue:
    """Thread-safe queue for requests."""

    def __init__(self):
        self._queue: Queue = Queue()
        self._lock = Lock()
        self._completed: List[Request] = []

    def put(self, request: Request) -> None:
        self._queue.put(request)

    def get(self, timeout: float = 0.5) -> Optional[Request]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    def mark_completed(self, request: Request) -> None:
        with self._lock:
            self._completed.append(request)

    @property
    def pending_size(self) -> int:
        return self._queue.qsize()

    @property
    def completed(self) -> List[Request]:
        with self._lock:
            return list(self._completed)

    def empty(self) -> bool:
        return self._queue.empty()
