"""Evaluator interface for COMPASS-V.

Subclass `Evaluator` to plug a workflow into Compass. The only required
method is `evaluate_partial(config, indices)`, which executes the workflow
under `config` against the samples at `indices` and returns one accuracy
score per sample (in [0, 1]).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Evaluator(ABC):
    """Workflow-agnostic evaluator interface."""

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Total number of samples in the dataset."""

    @abstractmethod
    def evaluate_partial(
        self, config: Dict[str, Any], indices: List[int]
    ) -> List[float]:
        """Run the workflow under `config` on samples at `indices`.

        Returns one accuracy score per sample, in the same order as
        `indices`. Required by COMPASS-V for progressive budgeting.
        """

    def evaluate(self, config: Dict[str, Any]) -> float:
        """Mean accuracy over the full dataset.

        Default implementation calls `evaluate_partial` over all indices.
        Override if your workflow has a faster batched path.
        """
        scores = self.evaluate_partial(config, list(range(self.n_samples)))
        return sum(scores) / len(scores) if scores else 0.0
