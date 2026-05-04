#!/usr/bin/env python3
"""COMPASS-V or Grid Search on a Compass workflow.

Usage:
    python experiments/run_search.py --workflow rag --slo 0.75 --method compass_v
    python experiments/run_search.py --workflow rag --slo 0.6  --method grid
    python experiments/run_search.py --workflow vision --slo 0.65 --method compass_v \
        --n-samples 200 --output results/search/vision/multi_slo/slo_0.65.json

The script loads the workflow's parameter space + evaluator, runs the chosen
search method, and writes the feasible set (with sample-eval count) as JSON.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compass.search import CompassV, CompassVConfig, ParameterSpace, config_key


def _load_workflow(name: str, n_samples: int):
    """Return (parameter_space, evaluator)."""
    if name == "rag":
        from workflows.rag import RagEvaluator, parameter_space
        from workflows.rag.configs import load_dataset
        dataset = load_dataset(n=n_samples)
        return parameter_space(), RagEvaluator(dataset=dataset)

    if name == "vision":
        from workflows.vision import VisionEvaluator, parameter_space
        images_dir = "data/coco/val2017"
        annotations_path = "data/coco/annotations/instances_val2017.json"
        if not Path(annotations_path).exists():
            raise FileNotFoundError(
                "COCO val2017 not found. Run `python -m workflows.vision.download_coco` first."
            )
        evaluator = VisionEvaluator(images_dir, annotations_path)
        # Subset to first n_samples images for tractable runs.
        evaluator.image_ids = evaluator.image_ids[:n_samples]
        return parameter_space(), evaluator

    raise ValueError(f"Unknown workflow: {name}")


def run_grid_search(
    space: ParameterSpace,
    evaluator,
    slo: float,
    n_samples: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Exhaustive evaluation over every valid configuration."""
    configs = space.enumerate_valid()
    if verbose:
        print(f"Grid search: {len(configs)} configs, {n_samples} samples each")

    results = []
    feasible = []
    total_evals = 0

    for cfg in tqdm(configs, desc="Grid", disable=not verbose):
        scores = evaluator.evaluate_partial(cfg, list(range(n_samples)))
        acc = float(np.mean(scores)) if scores else 0.0
        total_evals += n_samples
        record = {"config": cfg, "accuracy": acc}
        results.append(record)
        if acc >= slo:
            feasible.append(record)

    return {
        "method": "grid",
        "slo": slo,
        "total_configs": len(configs),
        "total_evaluated": len(configs),
        "feasible_count": len(feasible),
        "sample_evals": total_evals,
        "feasible_configs": feasible,
        "all_results": results,
    }


def run_compass_v(
    space: ParameterSpace,
    evaluator,
    slo: float,
    n_samples: int,
    n_bootstrap: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """COMPASS-V search."""
    cfg = CompassVConfig(
        n_bootstrap=n_bootstrap,
        budgets=[10, 25, 50, n_samples] if n_samples >= 50 else [n_samples],
        confidence=0.95,
        margin=0.02,
    )
    search = CompassV(space=space, evaluator=evaluator, slo=slo, config=cfg)
    feasible = search.search(verbose=verbose)
    return {
        "method": "compass_v",
        "slo": slo,
        "n_bootstrap": n_bootstrap,
        "total_configs": len(space.enumerate_valid()),
        "total_evaluated": len(search.state),
        "feasible_count": len(feasible),
        "sample_evals": search.total_evals,
        "feasible_configs": feasible,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflow", required=True, choices=["rag", "vision"])
    parser.add_argument("--method", required=True, choices=["compass_v", "grid"])
    parser.add_argument("--slo", type=float, required=True, help="Accuracy SLO threshold")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Samples per evaluation (default 100 for RAG, 200 typical for vision)")
    parser.add_argument("--n-bootstrap", type=int, default=20,
                        help="LHS bootstrap size for COMPASS-V (default 20)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/search/{workflow}/multi_slo/slo_{slo}.json)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            f"results/search/{args.workflow}/multi_slo/slo_{args.slo}.json"
        )

    print(f"Loading {args.workflow} workflow with {args.n_samples} samples...")
    space, evaluator = _load_workflow(args.workflow, args.n_samples)
    print(f"Configuration space: {len(space.enumerate_valid())} valid configs")

    start = time.time()
    if args.method == "grid":
        result = run_grid_search(space, evaluator, args.slo, args.n_samples,
                                 verbose=not args.quiet)
    else:
        result = run_compass_v(space, evaluator, args.slo, args.n_samples,
                               args.n_bootstrap, verbose=not args.quiet)
    result["elapsed_sec"] = time.time() - start
    result["timestamp"] = datetime.now().isoformat()
    result["workflow"] = args.workflow
    result["n_samples"] = args.n_samples

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{result['method']} on {args.workflow} @ SLO={args.slo}: "
          f"{result['feasible_count']} feasible, "
          f"{result['sample_evals']} sample evals, "
          f"{result['elapsed_sec']:.1f}s")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
