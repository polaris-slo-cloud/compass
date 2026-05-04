# Reproducing the paper figures

The repository ships ~8 MB of cached experiment results under `results/`.
You can regenerate every paper figure in <30 s without running any
experiment.

## Quick: regenerate from cache

```bash
python plotting/plot_search.py  --figure all
python plotting/plot_serving.py --figure all
```

Outputs land in `figures/search/` and `figures/serving/` as both PDF and PNG.

| Paper figure          | Script                       | Cached input                                            |
| ---                   | ---                          | ---                                                     |
| Fig. 1 (convergence)  | `plot_search.py --figure convergence` | `results/search/rag/multi_slo/compass/slo_*.json` |
| Fig. 4 (efficiency)   | `plot_search.py --figure efficiency`  | hard-coded summary points (paper-final values)    |
| Fig. bars             | `plot_serving.py --figure bars`       | `results/serving/run5/*/*.json`                   |
| Fig. scatter          | `plot_serving.py --figure scatter`    | same                                              |
| Fig. CDF              | `plot_serving.py --figure cdf`        | same                                              |
| Fig. timeseries       | `plot_serving.py --figure timeseries` | same                                              |

## Full: rerun every experiment

```bash
bash scripts/reproduce_paper.sh
```

Stages:

1. Build the SQuAD FAISS index (`python -m workflows.rag.build_index`) and
   download COCO val2017 (`python -m workflows.vision.download_coco`).
2. Run COMPASS-V at every paper SLO on both workflows. ~hours per workflow.
3. Profile feasible RAG configs → extract Pareto → derive AQM thresholds.
4. Run Elastico + three static baselines under spike / ramp / step / bursty.
5. Regenerate every figure under `figures/`.

Skip stages 1–4 with `--plot-only` to just replot from the shipped cache.

## Reproducing one figure end-to-end

For example, RAG convergence at SLO=0.75:

```bash
python experiments/run_search.py \
    --workflow rag --method compass_v --slo 0.75 --n-samples 100 \
    --output results/search/rag/multi_slo/compass/slo_0.75.json

python plotting/plot_search.py --figure convergence
```
