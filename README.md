# Compass

**Compass** is a framework for serving Compound AI workflows under accuracy
and latency SLOs by switching between configurations at runtime. Given a
workflow `W`, a dataset `D`, an accuracy threshold `τ`, and a latency SLO
`δ` on target hardware `H`, Compass produces an inference serving system
that maintains both SLOs as load changes.

The framework has **three components**, split across an offline preparation
phase and an online serving phase.

## Architecture

### Offline phase

**1. COMPASS-V — Task Optimization** (`compass/search/`)
Explores the configuration space `C` and returns the **feasible set**
`F = { c ∈ C : Acc(c) ≥ τ }`. Combines Latin-Hypercube bootstrap,
inverse-distance-weighted pseudo-gradients, Wilson-CI early stopping, and
hill-climb / lateral expansion to discover `F` with far fewer evaluations
than exhaustive grid search.
*Inputs:* `W, D, τ`. *Output:* `F`.

**2. Planner — Deployment Planning** (`compass/planner/`)
Takes the feasible set and prepares it for a specific deployment in three
sub-stages:
- *Profiler* — measures end-to-end P50 / P95 / P99 latency for every config
  in `F` on hardware `H`.
- *Pareto extractor* — keeps only configurations that are non-dominated on
  (accuracy, P95-latency).
- *AQM* — derives a queue-length-based switching threshold for each
  Pareto-optimal config using an analytical queuing model with the SLO
  budget `δ - P95`.

*Inputs:* `F, H, δ`. *Output:* Pareto front + per-config switching policy.

### Online phase

**3. Inference Serving System — Elastico** (`compass/serving/`)
Wraps the Pareto front into a live serving stack: a request queue, a load
monitor, the **Elastico** controller, and a workflow executor. As load
changes, Elastico moves up the Pareto front (faster configs) when the
queue grows past the active threshold, and down (more accurate configs)
when load slack returns — with asymmetric hysteresis to prevent
oscillation.
*Input:* Pareto front + thresholds. *Output:* served requests that
respect both `τ` and `δ`.

```
   offline                 offline                        online
┌───────────────┐    ┌────────────────┐    ┌───────────────────────────┐
│  COMPASS-V    │    │    Planner     │    │  Inference Serving        │
│  W, D, τ ──►  │ F  │  F, H, δ ──►   │    │  Pareto + thresholds      │
│  feasible set │───►│  Pareto +      │───►│  → Elastico controller    │
│       F       │    │  AQM thresholds│    │  → live request serving   │
└───────────────┘    └────────────────┘    └───────────────────────────┘
   compass/search      compass/planner             compass/serving
```

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

plotting/                 Generates the paper figures from results/:
├── plot_search.py        --figure {convergence|efficiency|all}
└── plot_serving.py       --figure {bars|scatter|cdf|timeseries|all}

results/                  Experiment outputs (gitignored — populate by
                          running experiments or scripts/reproduce_paper.sh):
├── search/{rag,vision}/multi_slo/    COMPASS-V per-SLO results
├── planner/                          Pareto frontier, latency profiles
└── serving/<pattern>_slo<ms>/        Elastico + static-baseline runs

scripts/
├── setup_ollama.sh       Pull the 6 LLMs used by RAG
└── reproduce_paper.sh    End-to-end orchestrator

docs/
├── adding_a_workflow.md  Plug in a new workflow (Evaluator + ParameterSpace)
└── reproducing_paper.md  Exact commands per figure

data/
└── squad_questions.json  Tracked: 2.9 MB SQuAD2 questions used by RAG
                          (the FAISS index and COCO/YOLO weights are NOT
                          shipped — see "Workflow setup" below).
```

---

## Requirements

| Resource     | Minimum / recommended                                                  |
| ---          | ---                                                                    |
| OS           | Linux (tested on Ubuntu 22.04). macOS works for COMPASS-V only.        |
| Python       | 3.10+                                                                  |
| GPU          | NVIDIA GPU with ≥ 16 GB VRAM (paper uses RTX 4090, 24 GB)              |
| Disk         | ~40 GB free: 30 GB Ollama models + 800 MB COCO + 1 GB indexes / weights |
| Network      | Required for first-time downloads (Ollama, COCO, sentence-transformers) |

A full paper reproduction (`bash scripts/reproduce_paper.sh`) takes roughly
**12–15 hours** on the reference hardware: ~6 h for COMPASS-V on RAG,
~3 h on vision, ~1 h for the planner, ~3 h for online serving runs. Pass
`--plot-only` to regenerate figures from a previously-computed `results/`
without re-running anything.

---

## Install

```bash
git clone https://github.com/polaris-slo-cloud/compass.git
cd compass

# Core install (just the COMPASS-V algorithm + planner + plotting):
pip install -e .

# To also run the bundled workflows, install the matching extras:
pip install -e ".[rag]"        # adds langchain + faiss + sentence-transformers
pip install -e ".[vision]"     # adds ultralytics + torch
pip install -e ".[all]"        # both
```

Quote the extras (`".[rag]"`) — bare `[rag]` is interpreted as a glob in zsh.

For bit-exact paper reproduction, pin to the lock file:

```bash
pip install -r requirements-lock.txt
```

---

## Workflow setup

Each workflow needs a one-time setup before its first run.

### RAG (SQuAD2.0)

```bash
# 1. Install Ollama from https://ollama.com, then start the daemon.
# 2. Pull the 6 LLMs used in the paper (~30 GB total):
bash scripts/setup_ollama.sh

# 3. Build the FAISS index from the shipped questions file (~5 min, ~10 MB on disk):
python -m workflows.rag.build_index
```

### Vision (COCO val2017)

```bash
# Download images + annotations (~1 GB) into data/coco/:
python -m workflows.vision.download_coco
```

YOLOv8 weights (`yolov8n/s/m/l/x.pt`) are auto-downloaded by the
`ultralytics` library on first use (~290 MB total) — no manual step needed.

---

## Quickstart

### Reproduce all paper figures end-to-end

```bash
bash scripts/reproduce_paper.sh         # 12-15 h on reference hardware
```

### Or run each phase manually

```bash
# 1. Offline: COMPASS-V search at one SLO threshold τ.
python experiments/run_search.py \
    --workflow rag --method compass_v --slo 0.75 --n-samples 100 \
    --output results/search/rag/multi_slo/slo_0.75.json

# 2. Offline: profile the feasible set, extract Pareto, derive AQM thresholds.
python experiments/run_planner.py --stage all \
    --workflow rag \
    --feasible results/search/rag/multi_slo/slo_0.75.json \
    --slo-ms 1000 --work-based

# 3. Online: serve under a spike load with Elastico (--warmup preloads
#    every Pareto-front model into Ollama before measurement begins).
python experiments/run_serving.py \
    --pareto results/planner/pareto_frontier.json \
    --slo 1000 --pattern spike --duration 180 --warmup \
    --output results/serving/spike_slo1000/elastico.json

# 4. Plot every figure under figures/.
python plotting/plot_search.py  --figure all
python plotting/plot_serving.py --figure all
```

See [docs/reproducing_paper.md](docs/reproducing_paper.md) for the exact
commands behind each figure.

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
