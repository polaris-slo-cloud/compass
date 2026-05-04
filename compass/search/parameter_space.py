"""Parameter space definition for COMPASS-V.

A workflow's tunable parameters are described by a `ParameterSpace`:
    - `params`: name -> ordered list of values (best-first by convention)
    - `norm_types`: per-parameter normalization (LINEAR / LOG / CATEGORICAL).
      If unspecified, Compass infers from value types and ranges.
    - `constraints`: optional list of validity predicates. A configuration
      is valid only if every predicate returns True.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class NormType(Enum):
    LINEAR = "linear"
    LOG = "log"
    CATEGORICAL = "categorical"


Constraint = Callable[[Dict[str, Any]], bool]


@dataclass
class ParameterSpace:
    """Configuration space for a workflow.

    Example:
        space = ParameterSpace(
            params={
                "model": ["llama3.1:8b", "gemma3:4b", "gemma3:1b"],
                "top_k": [20, 10, 5],
            },
            norm_types={"top_k": NormType.LOG},  # model auto-CATEGORICAL
        )
    """

    params: Dict[str, List[Any]]
    norm_types: Dict[str, NormType] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)

    # Cached normalized values: param -> [float per value]
    _norm_vals: Dict[str, List[float]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for name, values in self.params.items():
            if name not in self.norm_types:
                self.norm_types[name] = self._infer_norm_type(values)
            self._norm_vals[name] = self._compute_norm_values(name, values)

    def is_valid(self, config: Dict[str, Any]) -> bool:
        """True iff `config` satisfies every constraint."""
        return all(c(config) for c in self.constraints)

    def enumerate_valid(self) -> List[Dict[str, Any]]:
        """Cartesian product filtered by constraints."""
        names = list(self.params)
        out = []
        for combo in product(*(self.params[n] for n in names)):
            cfg = dict(zip(names, combo))
            if self.is_valid(cfg):
                out.append(cfg)
        return out

    def normalize(self, config: Dict[str, Any]) -> Dict[str, float]:
        """Map a configuration into [0, 1]^d for distance computations."""
        out = {}
        for name, val in config.items():
            if name not in self.params:
                continue
            values = self.params[name]
            idx = values.index(val) if val in values else 0
            norms = self._norm_vals[name]
            if len(norms) <= 1 or max(norms) == min(norms):
                out[name] = 0.5
            else:
                out[name] = (norms[idx] - min(norms)) / (max(norms) - min(norms))
        return out

    def neighbor_value(
        self, name: str, current: Any, direction: str
    ) -> Optional[Any]:
        """Next value for `name` in the given direction along the normalized axis.

        `direction` is "up" (higher normalized value) or "down".
        Returns None if `current` is already at the extremum.
        """
        values = self.params[name]
        norms = self._norm_vals[name]
        idx = values.index(current)
        cur_x = norms[idx]

        best_val, best_x = None, None
        for i, v in enumerate(values):
            if v == current:
                continue
            vx = norms[i]
            if direction == "up" and vx > cur_x:
                if best_x is None or vx < best_x:
                    best_val, best_x = v, vx
            elif direction == "down" and vx < cur_x:
                if best_x is None or vx > best_x:
                    best_val, best_x = v, vx
        return best_val

    @staticmethod
    def _extract_numeric(values: List[Any]) -> List[float]:
        nums = []
        for v in values:
            if isinstance(v, (int, float)):
                nums.append(float(v))
            elif isinstance(v, str):
                m = re.search(r"(\d+(?:\.\d+)?)(b|m)?\b", v.lower())
                if m:
                    n = float(m.group(1))
                    if m.group(2) == "m":
                        n /= 1000
                    nums.append(n)
        return nums

    def _infer_norm_type(self, values: List[Any]) -> NormType:
        nums = self._extract_numeric(values)
        if len(nums) != len(values):
            return NormType.CATEGORICAL
        if not nums:
            return NormType.CATEGORICAL
        if max(nums) / max(min(nums), 1e-3) > 10:
            return NormType.LOG
        return NormType.LINEAR

    def _compute_norm_values(self, name: str, values: List[Any]) -> List[float]:
        norm_type = self.norm_types[name]
        if norm_type == NormType.CATEGORICAL:
            return list(range(len(values)))
        nums = self._extract_numeric(values)
        if len(nums) != len(values):
            return list(range(len(values)))
        if norm_type == NormType.LOG:
            return [float(np.log(n + 1)) for n in nums]
        return [float(n) for n in nums]


def config_key(config: Dict[str, Any]) -> tuple:
    """Hashable key for a configuration dict."""
    return tuple(sorted(config.items()))
