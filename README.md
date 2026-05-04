# Compass

**Compass** optimizes Compound AI workflows for dynamic adaptation. It has two components:

- **COMPASS-V** — offline feasible-configuration search via LHS bootstrap, IDW pseudo-gradients, Wilson-CI early stopping, and hill-climb / lateral expansion.
- **Elastico** — online runtime controller that switches between Pareto-optimal configurations as load changes, using analytical-queuing-model (AQM) thresholds.

This repository contains the implementation evaluated in:

> M. Gravara, J. L. Herrera, S. Nastic.
> *Compass: Optimizing Compound AI Workflows for Dynamic Adaptation.*
> IEEE/ACM CCGrid 2026, Sydney, Australia.

---

## Repository layout

```
compass/                  Framework — three components mirror the paper:
├── search/               Offline phase 1: COMPASS-V task optimization
├── planner/              Offline phase 2: profiling, Pareto, AQM thresholds
└── serving/              Online phase: Elastico runtime adaptation

workflows/                Workflows being optimized (plug your own here):
├── rag/                  RAG pipeline on SQuAD2.0 (LangChain + Ollama)
└── vision/               Cascaded YOLO detection on COCO val2017

experiments/              Thin runner scripts (argparse → compass APIs):
├── run_search.py         COMPASS-V or Grid Search on a workflow
├── run_planner.py        Profile → Pareto → AQM (--stage profile|pareto|aqm|all)
├── run_serving.py        Elastico online experiment
└── run_baseline.py       Static-baseline online experiment

plotting/                 Paper-figure regenerators:
├── plot_search.py        --figure {convergence|efficiency|all}
└── plot_serving.py       --figure {bars|scatter|cdf|timeseries|all}

results/                  Paper-result cache (~8 MB, shipped):
├── search/{rag,vision}/multi_slo/    COMPASS-V per-SLO results
├── planner/                          Pareto frontier, latency profiles
└── serving/run5/<pattern>_slo<ms>/   Elastico + static-baseline runs

scripts/
├── setup_ollama.sh       Pull the 6 LLMs used by RAG
└── reproduce_paper.sh    End-to-end orchestrator

docs/
├── adding_a_workflow.md  Plug in a new workflow (Evaluator + ParameterSpace)
└── reproducing_paper.md  Exact commands per figure
```

---

## Install

```bash
git clone https://github.com/distributed-systems-tuwien/compass.git
cd compass

# pick the extras matching your workflows:
pip install -e .[rag]         # RAG only
pip install -e .[vision]      # vision only
pip install -e .[all]         # both
```

For bit-exact paper reproduction, use the lock file: `pip install -r requirements-lock.txt`.

### External services

- **Ollama** (RAG): install from <https://ollama.com>, then `bash scripts/setup_ollama.sh` to pull the 6 LLMs.
- **COCO val2017** (vision): `python -m workflows.vision.download_coco` (~800 MB).
- **SQuAD FAISS index** (RAG): not shipped. Build once with `python -m workflows.rag.build_index`.

---

## Quickstart

Reproduce all paper figures from the cached results in <30 s:

```bash
python plotting/plot_search.py --figure all     # → figures/search/
python plotting/plot_serving.py --figure all    # → figures/serving/
```

Run COMPASS-V on RAG at one SLO:

```bash
python experiments/run_search.py \
    --workflow rag --method compass_v --slo 0.75 --n-samples 100
```

Run Elastico under a spike load:

```bash
python experiments/run_serving.py \
    --pareto results/planner/pareto_frontier.json \
    --slo 1000 --pattern spike --duration 180 --warmup
```

End-to-end: `bash scripts/reproduce_paper.sh`.

---

## Add your own workflow

Implement two interfaces and pass them to `CompassV`:

```python
from compass.search import CompassV, Evaluator, ParameterSpace, NormType

class MyEvaluator(Evaluator):
    @property
    def n_samples(self) -> int: ...
    def evaluate_partial(self, config, indices) -> list[float]: ...

space = ParameterSpace(
    params={"model": ["A", "B", "C"], "k": [10, 20, 50]},
    norm_types={"k": NormType.LOG},
    constraints=[lambda c: c["k"] <= 50],
)

feasible = CompassV(space, MyEvaluator(...), slo=0.75).search()
```

Full walkthrough: [docs/adding_a_workflow.md](docs/adding_a_workflow.md).

---

## Citation

```bibtex
@inproceedings{gravara2026compass,
  title     = {Compass: Optimizing Compound AI Workflows for Dynamic Adaptation},
  author    = {Gravara, Milos and Herrera, Juan Luis and Nastic, Stefan},
  booktitle = {IEEE/ACM International Symposium on Cluster, Cloud and Internet Computing (CCGrid)},
  year      = {2026},
  address   = {Sydney, Australia},
}
```

---

## Contact

Milos Gravara — `milos.gravara@tuwien.ac.at`
Distributed Systems Group, TU Wien.
