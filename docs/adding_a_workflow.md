# Adding a new workflow

To plug a new Compound AI workflow into Compass you implement two things:

1. An **`Evaluator`** that knows how to run your workflow under a given configuration on a list of dataset samples.
2. A **`ParameterSpace`** that describes the configurations to search over.

Both shipped workflows (`workflows/rag/`, `workflows/vision/`) are reference implementations.

---

## 1. The `Evaluator`

```python
# workflows/myflow/evaluator.py
from typing import Any, Dict, List
from compass.search import Evaluator

class MyEvaluator(Evaluator):
    def __init__(self, dataset):
        self.dataset = dataset
        # ... build pipeline, load model weights, etc.

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    def evaluate_partial(self, config: Dict[str, Any], indices: List[int]) -> List[float]:
        """Run the workflow under `config` on samples at `indices`.
        Return one accuracy score per sample, in [0, 1], in the same order.
        """
        self._configure(config)
        return [self._score_one(self.dataset[i]) for i in indices]

    def _configure(self, config): ...
    def _score_one(self, sample) -> float: ...
```

Only `n_samples` and `evaluate_partial` are required. The default `evaluate(config)` averages `evaluate_partial(config, range(n_samples))`; override it if you have a faster batched path.

The score must be in `[0, 1]`. Wilson confidence intervals assume Bernoulli-like statistics, so a smooth metric such as F1, IoU, or accuracy works best.

---

## 2. The `ParameterSpace`

```python
# workflows/myflow/configs.py
from compass.search import ParameterSpace, NormType

def parameter_space() -> ParameterSpace:
    return ParameterSpace(
        params={
            "model": ["big", "medium", "small"],     # best-first ordering
            "top_k": [50, 20, 10, 5, 1],
            "temperature": [0.0, 0.3, 0.7, 1.0],
        },
        norm_types={
            "top_k": NormType.LOG,                   # log-scale numeric
            "temperature": NormType.LINEAR,          # linear numeric
            "model": NormType.CATEGORICAL,           # auto-inferred for strings
        },
        constraints=[
            lambda c: not (c["model"] == "small" and c["top_k"] > 20),
        ],
    )
```

- `params` lists the discrete values for each parameter, **best-first**. COMPASS-V uses this order to seed corner configurations and to decide gradient direction.
- `norm_types` controls how values are mapped into `[0, 1]` for distance / IDW computations. If unspecified, Compass infers `LINEAR` / `LOG` from numeric ranges and `CATEGORICAL` for strings.
- `constraints` are predicates run on every candidate; constraint-invalid configs are skipped during bootstrap and navigation.

---

## 3. Run the search

```python
from compass.search import CompassV, CompassVConfig
from workflows.myflow.configs import parameter_space
from workflows.myflow.evaluator import MyEvaluator

space = parameter_space()
evaluator = MyEvaluator(dataset=...)
search = CompassV(
    space=space,
    evaluator=evaluator,
    slo=0.75,
    config=CompassVConfig(n_bootstrap=20, budgets=[10, 25, 50, 100]),
)
feasible = search.search()
```

Or via the runner script — wire your workflow into `experiments/run_search.py::_load_workflow`:

```python
if name == "myflow":
    from workflows.myflow import MyEvaluator, parameter_space
    return parameter_space(), MyEvaluator(...)
```

then:

```bash
python experiments/run_search.py --workflow myflow --method compass_v --slo 0.75 --n-samples 100
```

---

## 4. Tips

- **Best-first ordering matters.** COMPASS-V seeds with the best-corner and worst-corner configurations and uses the per-parameter ordering for hill-climbing. If your "best" model isn't first, the bootstrap may miss the feasible region at tight SLOs.
- **Constraints are cheap.** Add them eagerly — every constraint-rejection saves a workflow execution.
- **Stochastic workflows.** If `evaluate_partial(config, [i])` is non-deterministic, fix a seed inside `_score_one`. Wilson CIs assume independent samples.
