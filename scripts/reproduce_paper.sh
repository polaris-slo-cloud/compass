#!/usr/bin/env bash
# End-to-end paper reproduction.
#
# Stages:
#   1. (optional) Build the SQuAD index for RAG and download COCO for vision.
#   2. Run COMPASS-V at every paper SLO on both workflows. ~hours each.
#   3. Profile feasible RAG configs, extract Pareto, derive AQM thresholds.
#   4. Run Elastico + static baselines under spike / bursty / ramp / step.
#   5. Regenerate every figure under figures/.
#
# Skip stages 1-4 with --plot-only to just regenerate figures from the
# shipped paper cache in results/.

set -euo pipefail

PLOT_ONLY=0
WORKFLOWS=(rag vision)
RAG_SLOS=(0.30 0.40 0.50 0.60 0.75 0.80 0.85 0.90)
VISION_SLOS=(0.55 0.60 0.65 0.70 0.73 0.75 0.77 0.78)
SERVING_SLOS=(500 750 1000 1500)
PATTERNS=(spike ramp step bursty)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plot-only) PLOT_ONLY=1; shift ;;
        --workflows) shift; IFS="," read -r -a WORKFLOWS <<< "$1"; shift ;;
        -h|--help)
            head -30 "$0" | sed 's/^#//; s/^ //'
            exit 0
            ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ "$PLOT_ONLY" -eq 0 ]]; then
    echo "=== [1/4] Workflow data ==="
    if [[ " ${WORKFLOWS[*]} " == *" rag "* ]] && [[ ! -d data/squad_index ]]; then
        echo "Building SQuAD FAISS index..."
        python -m workflows.rag.build_index
    fi
    if [[ " ${WORKFLOWS[*]} " == *" vision "* ]] && [[ ! -d data/coco/val2017 ]]; then
        echo "Downloading COCO val2017..."
        python -m workflows.vision.download_coco
    fi

    echo "=== [2/4] COMPASS-V search ==="
    if [[ " ${WORKFLOWS[*]} " == *" rag "* ]]; then
        for slo in "${RAG_SLOS[@]}"; do
            python experiments/run_search.py \
                --workflow rag --method compass_v --slo "$slo" --n-samples 100
        done
    fi
    if [[ " ${WORKFLOWS[*]} " == *" vision "* ]]; then
        for slo in "${VISION_SLOS[@]}"; do
            python experiments/run_search.py \
                --workflow vision --method compass_v --slo "$slo" --n-samples 200
        done
    fi

    echo "=== [3/4] Planner (RAG only — Elastico runs against RAG) ==="
    python experiments/run_planner.py --stage all \
        --workflow rag \
        --feasible results/search/rag/multi_slo/slo_0.75.json \
        --slo-ms 1000 --work-based

    echo "=== [4/4] Online runtime ==="
    for pat in "${PATTERNS[@]}"; do
        for slo in "${SERVING_SLOS[@]}"; do
            out_dir="results/serving/run_local/${pat}_slo${slo}"
            mkdir -p "$out_dir"
            for sel in fastest balanced accurate; do
                python experiments/run_baseline.py \
                    --pareto results/planner/pareto_frontier.json \
                    --slo "$slo" --pattern "$pat" --duration 180 --warmup \
                    --config-select "$sel" \
                    --output "$out_dir/baseline_${sel}.json"
            done
            python experiments/run_serving.py \
                --pareto results/planner/pareto_frontier.json \
                --slo "$slo" --pattern "$pat" --duration 180 --warmup \
                --output "$out_dir/elastico.json"
        done
    done
fi

echo "=== Plot all figures ==="
python plotting/plot_search.py --figure all
python plotting/plot_serving.py --figure all

echo "Done. Figures in figures/."
