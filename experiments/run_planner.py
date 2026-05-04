#!/usr/bin/env python3
"""Compass deployment planner: profile → Pareto → AQM thresholds.

Usage:
    # Run all stages end-to-end (default)
    python experiments/run_planner.py \
        --feasible results/search/rag/multi_slo/slo_0.75.json \
        --slo-ms 1000

    # Run only one stage
    python experiments/run_planner.py --stage profile --feasible ...
    python experiments/run_planner.py --stage pareto  --profiles ...
    python experiments/run_planner.py --stage aqm     --pareto ...

The full pipeline writes three files under `results/planner/`:
    latency_profiles.json    -- per-config P50/P95/P99 latency
    pareto_frontier.json     -- accuracy/latency Pareto-optimal subset
    thresholds_slo<ms>.json  -- AQM thresholds for the controller
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compass.planner import (
    Profiler,
    LatencyProfile,
    extract_pareto,
    select_operating_points,
    compute_thresholds,
    compute_work_thresholds,
)


def _load_workflow_executor(workflow: str, n_samples: int):
    """Return (executor, samples) for the requested workflow."""
    if workflow == "rag":
        from compass.serving import WorkflowExecutor
        from workflows.rag.configs import load_dataset
        return WorkflowExecutor(keep_models_loaded=True), load_dataset(n=n_samples)

    if workflow == "vision":
        from workflows.vision import VisionEvaluator
        # The vision evaluator is also the executor: VisionEvaluator.evaluate_partial
        # measures latency implicitly. For a clean profiler we wrap it.
        raise NotImplementedError(
            "Vision profiling: use VisionEvaluator with the Profiler directly "
            "(samples = image indices)."
        )

    raise ValueError(f"Unknown workflow: {workflow}")


def stage_profile(args: argparse.Namespace) -> Path:
    with open(args.feasible) as f:
        feasible_data = json.load(f)
    feasible = feasible_data["feasible_configs"]
    print(f"Profiling {len(feasible)} feasible configurations...")

    executor, samples = _load_workflow_executor(args.workflow, args.profile_samples + args.warmup)
    profiler = Profiler(
        executor=executor,
        samples=samples,
        n_warmup=args.warmup,
        n_samples=args.profile_samples,
    )

    profiles = []
    start = time.time()
    for entry in feasible:
        prof = profiler.profile_config(entry["config"])
        profiles.append({
            **prof.to_dict(),
            "accuracy": entry["accuracy"],
        })

    out = {
        "metadata": {
            "source": str(args.feasible),
            "workflow": args.workflow,
            "n_warmup": args.warmup,
            "n_samples": args.profile_samples,
            "elapsed_sec": time.time() - start,
            "timestamp": datetime.now().isoformat(),
        },
        "profiles": profiles,
    }
    out_path = Path(args.profiles_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Wrote: {out_path}")
    return out_path


def stage_pareto(args: argparse.Namespace) -> Path:
    with open(args.profiles) as f:
        profile_data = json.load(f)
    profiles = profile_data["profiles"]

    pareto = extract_pareto(profiles)
    operating = select_operating_points(pareto)

    out = {
        "metadata": {
            "source": str(args.profiles),
            "n_pareto": len(pareto),
            "timestamp": datetime.now().isoformat(),
        },
        "pareto_frontier": pareto,
        "operating_points": operating,
    }
    out_path = Path(args.pareto_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Pareto frontier: {len(pareto)} configs -> {out_path}")
    return out_path


def stage_aqm(args: argparse.Namespace) -> Path:
    with open(args.pareto) as f:
        pareto_data = json.load(f)
    pareto = pareto_data["pareto_frontier"]

    if args.work_based:
        thresholds = compute_work_thresholds(pareto, args.slo_ms)
    else:
        thresholds = compute_thresholds(pareto, args.slo_ms, args.hysteresis)

    out = {
        "metadata": {
            "source": str(args.pareto),
            "slo_ms": args.slo_ms,
            "work_based": args.work_based,
            "hysteresis": args.hysteresis,
            "timestamp": datetime.now().isoformat(),
        },
        "thresholds": [asdict(t) for t in thresholds],
    }
    out_path = Path(args.thresholds_out or
                    f"results/planner/thresholds_slo{int(args.slo_ms)}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  AQM thresholds (SLO={args.slo_ms}ms): {len(thresholds)} -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["profile", "pareto", "aqm", "all"],
                        default="all")
    parser.add_argument("--workflow", default="rag", choices=["rag", "vision"])

    # profile stage
    parser.add_argument("--feasible", help="search results JSON (input to profile stage)")
    parser.add_argument("--profile-samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--profiles-out",
                        default="results/planner/latency_profiles.json")

    # pareto stage
    parser.add_argument("--profiles", help="profiles JSON (input to pareto stage)")
    parser.add_argument("--pareto-out",
                        default="results/planner/pareto_frontier.json")

    # aqm stage
    parser.add_argument("--pareto", help="pareto JSON (input to aqm stage)")
    parser.add_argument("--slo-ms", type=float, default=1000.0)
    parser.add_argument("--work-based", action="store_true",
                        help="continuous slack thresholds (default integer)")
    parser.add_argument("--hysteresis", type=int, default=2)
    parser.add_argument("--thresholds-out", default=None)

    args = parser.parse_args()

    if args.stage in ("profile", "all"):
        if not args.feasible:
            parser.error("--feasible is required for profile stage")
        args.profiles = stage_profile(args)

    if args.stage in ("pareto", "all"):
        if args.stage == "pareto" and not args.profiles:
            parser.error("--profiles is required for pareto stage")
        args.pareto = stage_pareto(args)

    if args.stage in ("aqm", "all"):
        if args.stage == "aqm" and not args.pareto:
            parser.error("--pareto is required for aqm stage")
        stage_aqm(args)


if __name__ == "__main__":
    main()
