"""COMPASS-V: feasible configuration discovery via vector navigation.

Implements the algorithm from Section IV of:
    Gravara, Herrera, Nastic. "Compass: Optimizing Compound AI Workflows
    for Dynamic Adaptation." CCGrid 2026.

Pipeline per configuration:
    1. LHS bootstrap for diverse seeding.
    2. Progressive evaluation with Wilson-CI early stopping.
    3. On feasible: lateral expansion along low-gradient axes.
    4. On infeasible: hill-climb up the IDW-estimated gradient.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube
from tqdm import tqdm

from compass.search.evaluator import Evaluator
from compass.search.parameter_space import ParameterSpace, config_key


@dataclass
class CompassVConfig:
    """Hyperparameters for COMPASS-V."""

    n_bootstrap: int = 20
    budgets: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    confidence: float = 0.95
    margin: float = 0.02
    idw_k: int = 5
    idw_power: float = 2.0
    max_lateral: int = 6
    max_iterations: int = 50
    seed: int = 42


class CompassV:
    """COMPASS-V search.

    Args:
        space: parameter space describing the configurations to search over.
        evaluator: workflow evaluator implementing `Evaluator`.
        slo: minimum acceptable accuracy threshold (`tau` in the paper).
        config: optional algorithm hyperparameters.

    Returns from `search()`:
        list of `{"config": ..., "accuracy": ...}` for every configuration
        classified as feasible.
    """

    def __init__(
        self,
        space: ParameterSpace,
        evaluator: Evaluator,
        slo: float,
        config: Optional[CompassVConfig] = None,
    ):
        self.space = space
        self.evaluator = evaluator
        self.slo = slo
        self.cfg = config or CompassVConfig()

        self.state: Dict[tuple, Dict] = {}
        self.total_evals = 0

    # --- Wilson-CI early stopping ---

    def _wilson_ci(self, acc: float, n: int) -> Tuple[float, float]:
        if n == 0:
            return 0.0, 1.0
        z = norm.ppf(1 - (1 - self.cfg.confidence) / 2)
        denom = 1 + z**2 / n
        center = (acc + z**2 / (2 * n)) / denom
        spread = z * np.sqrt((acc * (1 - acc) + z**2 / (4 * n)) / n) / denom
        return max(0.0, center - spread), min(1.0, center + spread)

    def _decide(self, acc: float, n: int, is_final: bool) -> str:
        if is_final:
            return "feasible" if acc >= self.slo else "infeasible"
        lo, hi = self._wilson_ci(acc, n)
        if lo > self.slo + self.cfg.margin:
            return "feasible"
        if hi < self.slo - self.cfg.margin:
            return "infeasible"
        return "uncertain"

    # --- IDW gradient and navigation ---

    @staticmethod
    def _distance(x1: Dict[str, float], x2: Dict[str, float]) -> float:
        return float(np.sqrt(sum((x1.get(p, 0) - x2.get(p, 0)) ** 2 for p in x1)))

    def _idw_gradient(self, config: Dict[str, Any]) -> Dict[str, float]:
        x_c = self.space.normalize(config)
        acc_c = self.state.get(config_key(config), {}).get("accuracy", 0)

        neighbors = []
        for key, st in self.state.items():
            if st.get("accuracy") is None or key == config_key(config):
                continue
            d = self._distance(x_c, st["normalized"])
            if d > 1e-10:
                neighbors.append((d, st["normalized"], st["accuracy"]))
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[: self.cfg.idw_k]

        grads = {}
        for param in self.space.params:
            num, denom = 0.0, 0.0
            for d, x_n, acc_n in neighbors:
                w = 1 / (d ** self.cfg.idw_power)
                dx = x_n.get(param, 0) - x_c.get(param, 0)
                if abs(dx) > 1e-10:
                    num += w * (acc_n - acc_c) / dx
                    denom += w
            grads[param] = num / denom if denom > 1e-10 else 0.0
        return grads

    def _climb_candidate(
        self, config: Dict[str, Any], grad: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        if not grad:
            return None
        best_param = max(grad, key=lambda p: grad[p])
        if grad[best_param] <= 0:
            return None
        direction = "up" if grad[best_param] > 0 else "down"
        next_val = self.space.neighbor_value(best_param, config[best_param], direction)
        if next_val is None:
            return None
        new = config.copy()
        new[best_param] = next_val
        if not self.space.is_valid(new):
            return None
        return new

    def _lateral_candidates(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        grad = self._idw_gradient(config)
        for param in sorted(grad, key=lambda p: abs(grad[p])):
            for direction in ("up", "down"):
                nv = self.space.neighbor_value(param, config[param], direction)
                if nv is None:
                    continue
                new = config.copy()
                new[param] = nv
                if not self.space.is_valid(new):
                    continue
                if config_key(new) not in self.state:
                    out.append(new)
            if len(out) >= self.cfg.max_lateral:
                break
        return out

    # --- evaluation with progressive budgeting ---

    def _evaluate(
        self, config: Dict[str, Any], budget: int, prev: Optional[Dict]
    ) -> Dict:
        if prev is None:
            indices = list(range(budget))
            scores = self.evaluator.evaluate_partial(config, indices)
            self.total_evals += budget
            return {
                "indices": set(indices),
                "scores": list(scores),
                "n": budget,
                "accuracy": float(np.mean(scores)),
            }

        new_idx = [i for i in range(budget) if i not in prev["indices"]]
        if not new_idx:
            return prev
        new_scores = self.evaluator.evaluate_partial(config, new_idx)
        self.total_evals += len(new_idx)
        all_scores = prev["scores"] + list(new_scores)
        return {
            "indices": prev["indices"] | set(new_idx),
            "scores": all_scores,
            "n": len(all_scores),
            "accuracy": float(np.mean(all_scores)),
        }

    # --- LHS bootstrap (constraint-aware) ---

    def _lhs_bootstrap(self) -> List[Dict[str, Any]]:
        names = list(self.space.params)
        sampler = LatinHypercube(d=len(names), seed=self.cfg.seed)
        samples = sampler.random(n=self.cfg.n_bootstrap)

        configs: List[Dict[str, Any]] = []
        for s in samples:
            cfg = {}
            for i, p in enumerate(names):
                vals = self.space.params[p]
                idx = min(int(s[i] * len(vals)), len(vals) - 1)
                cfg[p] = vals[idx]
            if self.space.is_valid(cfg):
                configs.append(cfg)

        # Always include extreme corners if they're valid.
        for corner in (
            {p: self.space.params[p][0] for p in names},
            {p: self.space.params[p][-1] for p in names},
        ):
            if self.space.is_valid(corner):
                configs.append(corner)

        # Top up with random valid configs if constraints reduced the count.
        if len(configs) < self.cfg.n_bootstrap:
            valid_pool = self.space.enumerate_valid()
            rng = np.random.RandomState(self.cfg.seed)
            seen = {config_key(c) for c in configs}
            while len(configs) < self.cfg.n_bootstrap and len(seen) < len(valid_pool):
                c = valid_pool[rng.randint(len(valid_pool))]
                k = config_key(c)
                if k not in seen:
                    seen.add(k)
                    configs.append(c)

        # Dedupe while preserving order.
        seen = set()
        unique = []
        for c in configs:
            k = config_key(c)
            if k not in seen:
                seen.add(k)
                unique.append(c)
        return unique

    # --- main loop ---

    def search(self, verbose: bool = True) -> List[Dict[str, Any]]:
        budgets = self.cfg.budgets

        if verbose:
            print(f"COMPASS-V: SLO={self.slo}, budgets={budgets}")

        bootstrap = self._lhs_bootstrap()
        if verbose:
            print(f"Bootstrap: {len(bootstrap)} configs")

        for cfg in tqdm(bootstrap, desc="Bootstrap", disable=not verbose):
            results = self._evaluate(cfg, budgets[0], None)
            self.state[config_key(cfg)] = {
                "config": cfg,
                "normalized": self.space.normalize(cfg),
                "results": results,
                "accuracy": results["accuracy"],
                "budget_idx": 0,
                "status": "pending",
            }

        for iteration in range(self.cfg.max_iterations):
            active = [
                k for k, s in self.state.items()
                if s["status"] in ("pending", "uncertain")
            ]
            if not active:
                break

            if verbose:
                print(f"Iteration {iteration + 1}: {len(active)} active")

            new_candidates: List[Dict[str, Any]] = []
            for key in tqdm(active, desc="Evaluating", disable=not verbose):
                st = self.state[key]
                cfg = st["config"]
                bidx = st.get("budget_idx", 0)

                if bidx >= len(budgets):
                    continue

                budget = budgets[bidx]
                is_final = bidx == len(budgets) - 1

                results = self._evaluate(cfg, budget, st["results"])
                st["results"] = results
                st["accuracy"] = results["accuracy"]
                st["budget_idx"] = bidx + 1

                decision = self._decide(results["accuracy"], results["n"], is_final)

                if decision == "feasible":
                    st["status"] = "feasible"
                    for lc in self._lateral_candidates(cfg):
                        if config_key(lc) not in self.state:
                            new_candidates.append(lc)
                elif decision == "infeasible":
                    st["status"] = "infeasible"
                    climb = self._climb_candidate(cfg, self._idw_gradient(cfg))
                    if climb and config_key(climb) not in self.state:
                        new_candidates.append(climb)
                else:
                    st["status"] = "uncertain"

            for nc in new_candidates:
                key = config_key(nc)
                if key in self.state:
                    continue
                self.state[key] = {
                    "config": nc,
                    "normalized": self.space.normalize(nc),
                    "results": None,
                    "accuracy": None,
                    "budget_idx": 0,
                    "status": "pending",
                }

            if verbose:
                n_feas = sum(1 for s in self.state.values() if s["status"] == "feasible")
                print(f"  Feasible: {n_feas}, New candidates: {len(new_candidates)}")

        feasible = [
            {"config": st["config"], "accuracy": st["accuracy"]}
            for st in self.state.values()
            if st["status"] == "feasible"
        ]

        if verbose:
            print(
                f"\nFound {len(feasible)} feasible configs, "
                f"{self.total_evals} total sample evals"
            )
        return feasible
