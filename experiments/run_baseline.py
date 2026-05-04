#!/usr/bin/env python3
"""Static-baseline runtime experiment (no Elastico).

Runs the RAG workflow with a single fixed configuration drawn from the Pareto
frontier (fastest / balanced / most-accurate / explicit index). Used as a
control against the Elastico controller — see Section VI of the paper.

Usage:
    python experiments/run_baseline.py \
        --pareto results/planner/pareto_frontier.json \
        --config-select fastest \
        --slo 1000 \
        --pattern spike --base-qps 1.5 --spike-qps 5.5 \
        --duration 180 \
        --warmup \
        --output results/serving/spike_slo1000/baseline_fastest.json
"""

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compass.planner import compute_work_thresholds, load_pareto
from compass.serving import (
    LoadGenerator,
    Request,
    RequestQueue,
    WorkflowExecutor,
)
from workflows.rag.utils import compute_f1


@dataclass
class BaselineResult:
    requests: List[Request]
    total: int = 0
    avg_latency_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    slo_violations: int = 0
    slo_compliance: float = 0.0
    avg_accuracy: float = 0.0


def load_questions(path: str = "data/squad_questions.json", last_percent: float = 10.0) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    answerable = [q for q in data if not q.get("is_impossible", False)]
    n = max(1, int(len(answerable) * last_percent / 100))
    return answerable[-n:]


def _arrivals_for(pattern: str, duration: float, args: argparse.Namespace) -> List[float]:
    if pattern == "constant":
        return LoadGenerator.constant(args.qps, duration)
    if pattern == "step":
        levels = [0.3, 0.5, 1.0, 0.5, 0.3]
        return LoadGenerator.step(levels, duration / len(levels))
    if pattern == "ramp":
        return LoadGenerator.ramp(args.start_qps, args.end_qps, duration)
    if pattern == "spike":
        return LoadGenerator.spike(
            args.base_qps, args.spike_qps,
            duration / 3, duration / 3, duration / 3,
        )
    if pattern == "bursty":
        return LoadGenerator.bursty(
            args.base_qps, args.burst_qps, duration,
            args.burst_duration, args.burst_interval,
        )
    raise ValueError(f"Unknown pattern: {pattern}")


class BaselineExperiment:
    """Single fixed config — no controller, no switching."""

    def __init__(self, config: Dict[str, Any], slo_ms: float):
        self.config = config
        self.slo_ms = slo_ms
        self.executor = WorkflowExecutor(keep_models_loaded=True)
        self.queue = RequestQueue()
        self._stop_event = threading.Event()
        self._start_time = 0.0
        self._pbar: Optional[tqdm] = None

    def warmup(self, keep_alive: str = "30m", ollama_url: str = "http://localhost:11434") -> None:
        model = self.config.get("generator_model", "llama3.1:8b")
        print(f"\nWarming up LLM: {model} (keep_alive={keep_alive})...")
        try:
            requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": "", "keep_alive": keep_alive},
                timeout=300,
            ).raise_for_status()
        except requests.RequestException as e:
            print(f"  Warning: failed to preload {model}: {e}")

        print("Warming pipeline (retriever, reranker)...")
        self.executor.configure(self.config)
        self.executor.execute("What is the capital of France?")
        print("Warmup complete.\n")

    def _worker(self) -> None:
        self.executor.configure(self.config)
        while not self._stop_event.is_set() or not self.queue.empty():
            req = self.queue.get(timeout=0.5)
            if req is None:
                continue
            req.start_time = time.time() - self._start_time
            req.answer = self.executor.execute(req.question)
            req.end_time = time.time() - self._start_time
            self.queue.mark_completed(req)
            self.queue.task_done()
            if self._pbar is not None:
                self._pbar.update(1)

    def run(self, pattern: str, duration: float, args: argparse.Namespace) -> BaselineResult:
        arrivals = _arrivals_for(pattern, duration, args)
        questions = load_questions()
        print(f"\nPattern={pattern} arrivals={len(arrivals)} questions={len(questions)}")

        self.queue = RequestQueue()
        self._stop_event.clear()
        self._start_time = time.time()

        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        self._pbar = tqdm(total=len(arrivals), desc="Completed", unit="req")

        for i, arrival_t in enumerate(arrivals):
            target = self._start_time + arrival_t
            now = time.time()
            if target > now:
                time.sleep(target - now)
            t = time.time() - self._start_time

            q = questions[i % len(questions)]
            req = Request(
                id=i,
                question=q["question"],
                ground_truths=q.get("answers", []),
                arrival_time=t,
            )
            req.config = self.config
            self.queue.put(req)

        self._stop_event.set()
        worker.join()
        self._pbar.close()

        result = BaselineResult(requests=self.queue.completed)
        self._compute_metrics(result)
        return result

    def _compute_metrics(self, result: BaselineResult) -> None:
        if not result.requests:
            return
        latencies = []
        accuracies = []
        for req in result.requests:
            lat = req.response_time_ms
            latencies.append(lat)
            if lat > self.slo_ms:
                result.slo_violations += 1
            if req.answer and req.ground_truths:
                accuracies.append(max(compute_f1(req.answer, gt) for gt in req.ground_truths))

        result.total = len(result.requests)
        latencies.sort()
        result.avg_latency_ms = sum(latencies) / len(latencies)
        result.p50_ms = latencies[int(len(latencies) * 0.50)]
        result.p95_ms = latencies[int(len(latencies) * 0.95)]
        result.p99_ms = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        result.slo_compliance = 100 * (1 - result.slo_violations / len(latencies))
        if accuracies:
            result.avg_accuracy = sum(accuracies) / len(accuracies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pareto", required=True, help="Pareto frontier JSON")
    parser.add_argument("--slo", type=float, required=True, help="SLO latency in ms")
    parser.add_argument("--config-select", default="balanced",
                        choices=["fastest", "balanced", "most_accurate", "index"])
    parser.add_argument("--config-index", type=int, default=0,
                        help="Config index when --config-select=index")
    parser.add_argument("--pattern", default="constant",
                        choices=["constant", "step", "ramp", "spike", "bursty"])
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--qps", type=float, default=0.5)
    parser.add_argument("--base-qps", type=float, default=1.5)
    parser.add_argument("--spike-qps", type=float, default=5.5)
    parser.add_argument("--start-qps", type=float, default=0.5)
    parser.add_argument("--end-qps", type=float, default=4.0)
    parser.add_argument("--burst-qps", type=float, default=6.0)
    parser.add_argument("--burst-duration", type=float, default=2.0)
    parser.add_argument("--burst-interval", type=float, default=10.0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--keep-alive", type=str, default="30m")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    pareto = load_pareto(args.pareto)
    thresholds = compute_work_thresholds(pareto, args.slo)
    print("Available configs (sorted by latency):")
    for i, t in enumerate(thresholds):
        print(f"  [{i}] {t.model_name}: mean={t.mean_ms:.0f}ms acc={t.accuracy:.3f}")

    if args.config_select == "fastest":
        idx = 0
    elif args.config_select == "most_accurate":
        idx = len(thresholds) - 1
    elif args.config_select == "balanced":
        idx = len(thresholds) // 2
    else:
        idx = args.config_index

    if not 0 <= idx < len(thresholds):
        sys.exit(f"Invalid config index {idx}")

    selected = thresholds[idx]
    print(f"\nSelected [{idx}]: {selected.model_name}, config={selected.config}")

    exp = BaselineExperiment(config=selected.config, slo_ms=args.slo)
    if args.warmup:
        exp.warmup(keep_alive=args.keep_alive)
    result = exp.run(args.pattern, args.duration, args)

    print(f"\n{'='*50}\nRESULTS\n{'='*50}")
    print(f"Config: {selected.model_name} (idx={idx})")
    print(f"Total requests:     {result.total}")
    print(f"Latency P50/P95/P99: {result.p50_ms:.0f}/{result.p95_ms:.0f}/{result.p99_ms:.0f} ms")
    print(f"SLO compliance:     {result.slo_compliance:.1f}% ({result.slo_violations} violations)")
    print(f"Average accuracy:   {result.avg_accuracy:.4f}")

    if args.output:
        out = {
            "config": {
                "slo_ms": args.slo, "pattern": args.pattern,
                "duration": args.duration,
                "config_select": args.config_select, "config_index": idx,
                "selected_config": selected.config,
                "model_name": selected.model_name,
            },
            "metrics": {
                "total": result.total,
                "p50_ms": result.p50_ms, "p95_ms": result.p95_ms, "p99_ms": result.p99_ms,
                "slo_compliance": result.slo_compliance,
                "avg_accuracy": result.avg_accuracy,
            },
            "requests": [
                {
                    "id": r.id, "arrival": r.arrival_time,
                    "start": r.start_time, "end": r.end_time,
                    "response_ms": r.response_time_ms, "wait_ms": r.wait_time_ms,
                }
                for r in result.requests
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
