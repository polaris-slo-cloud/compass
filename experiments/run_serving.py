#!/usr/bin/env python3
"""Online Elastico runtime experiment.

Runs the queue-length-driven Elastico controller against a live RAG workflow
under a chosen load pattern (constant, step, ramp, spike, bursty). Records
per-request latency, configuration switches, and SLO compliance.

Usage:
    python experiments/run_serving.py \
        --pareto results/planner/pareto_frontier.json \
        --slo 1000 \
        --pattern spike --base-qps 1.5 --spike-qps 5.5 \
        --duration 180 \
        --warmup \
        --output results/serving/spike_slo1000/elastico.json
"""

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compass.planner import (
    compute_thresholds,
    compute_work_thresholds,
    load_pareto,
)
from compass.serving import (
    Controller,
    LoadGenerator,
    Monitor,
    Request,
    RequestQueue,
    WorkflowExecutor,
)
from workflows.rag.utils import compute_f1


@dataclass
class ExperimentResult:
    requests: List[Request]
    switches: List[dict]
    monitor_summary: dict
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


class Experiment:
    """Elastico online experiment driver."""

    def __init__(
        self,
        pareto_path: str,
        slo_ms: float,
        work_based: bool = True,
        cooldown_s: float = 5.0,
        upscale_cooldown_s: float = 0.0,
        h_time_ms: float = 50.0,
        hysteresis: int = 2,
    ):
        self.slo_ms = slo_ms
        self.work_based = work_based

        pareto = load_pareto(pareto_path)
        if work_based:
            thresholds = compute_work_thresholds(pareto, slo_ms)
        else:
            thresholds = compute_thresholds(pareto, slo_ms, hysteresis)

        self.controller = Controller(
            thresholds,
            h_time_ms=h_time_ms,
            cooldown_s=cooldown_s,
            upscale_cooldown_s=upscale_cooldown_s,
        )
        self.executor = WorkflowExecutor(keep_models_loaded=True)
        self.queue = RequestQueue()
        self.monitor = Monitor()

        self._stop_event = threading.Event()
        self._start_time = 0.0
        self._pbar: Optional[tqdm] = None

        print(f"Loaded {len(thresholds)} configs, SLO={slo_ms}ms, "
              f"mode={'work' if work_based else 'integer'}")
        for i, t in enumerate(thresholds):
            print(f"  [{i}] {t.model_name}: mean={t.mean_ms:.0f}ms "
                  f"p95={t.p95_ms:.0f}ms acc={t.accuracy:.3f}")

    def warmup(
        self,
        keep_alive: str = "30m",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        thresholds = self.controller.thresholds
        llm_models = sorted({t.model_name for t in thresholds})

        print(f"\nWarming up {len(llm_models)} LLM model(s) (keep_alive={keep_alive})...")
        for model in tqdm(llm_models, desc="Loading LLMs"):
            try:
                requests.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": keep_alive},
                    timeout=300,
                ).raise_for_status()
            except requests.RequestException as e:
                print(f"  Warning: failed to preload {model}: {e}")

        print("Warming pipeline (retriever, reranker)...")
        seen = set()
        warmup_q = "What is the capital of France?"
        for t in tqdm(thresholds, desc="Warming"):
            key = (t.config.get("reranker_model"), t.config.get("retriever_k"))
            if key in seen:
                continue
            seen.add(key)
            self.executor.configure(t.config)
            self.executor.execute(warmup_q)
        print("Warmup complete.\n")

    def _worker(self) -> None:
        current_cfg_idx = None
        while not self._stop_event.is_set() or not self.queue.empty():
            req = self.queue.get(timeout=0.5)
            if req is None:
                continue
            if req.config_idx != current_cfg_idx:
                self.executor.configure(req.config)
                current_cfg_idx = req.config_idx
            req.start_time = time.time() - self._start_time
            req.answer = self.executor.execute(req.question)
            req.end_time = time.time() - self._start_time
            self.controller.on_completion(req.end_time)
            self.queue.mark_completed(req)
            self.queue.task_done()
            self.monitor.snapshot(
                req.end_time,
                self.controller.queue_length,
                self.controller.current_idx,
                self.controller.current.model_name,
            )
            if self._pbar is not None:
                self._pbar.update(1)

    def run(self, pattern: str, duration: float, args: argparse.Namespace) -> ExperimentResult:
        arrivals = _arrivals_for(pattern, duration, args)
        questions = load_questions()
        print(f"\nPattern={pattern} arrivals={len(arrivals)} questions={len(questions)}")

        self.controller.reset()
        self.queue = RequestQueue()
        self.monitor = Monitor()
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
            req.config = self.controller.on_arrival(t)
            req.config_idx = self.controller.current_idx
            self.monitor.record_arrival(t)
            self.monitor.snapshot(
                t, self.controller.queue_length,
                self.controller.current_idx, self.controller.current.model_name,
            )
            self.queue.put(req)

        self._stop_event.set()
        worker.join()
        self._pbar.close()

        result = ExperimentResult(
            requests=self.queue.completed,
            switches=[
                {"t": s.timestamp, "from": s.from_idx, "to": s.to_idx,
                 "N": s.queue_length, "reason": s.reason}
                for s in self.controller.switches
            ],
            monitor_summary=self.monitor.summary(),
        )
        self._compute_metrics(result)
        return result

    def _compute_metrics(self, result: ExperimentResult) -> None:
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
    parser.add_argument("--pattern", default="constant",
                        choices=["constant", "step", "ramp", "spike", "bursty"])
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--qps", type=float, default=0.5, help="QPS for constant pattern")
    parser.add_argument("--base-qps", type=float, default=1.5, help="spike/bursty base")
    parser.add_argument("--spike-qps", type=float, default=5.5, help="spike peak")
    parser.add_argument("--start-qps", type=float, default=0.5, help="ramp start")
    parser.add_argument("--end-qps", type=float, default=4.0, help="ramp end")
    parser.add_argument("--burst-qps", type=float, default=6.0)
    parser.add_argument("--burst-duration", type=float, default=2.0)
    parser.add_argument("--burst-interval", type=float, default=10.0)
    parser.add_argument("--integer", action="store_true",
                        help="Use integer thresholds instead of work-based")
    parser.add_argument("--cooldown", type=float, default=5.0)
    parser.add_argument("--upscale-cooldown", type=float, default=0.0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--keep-alive", type=str, default="30m")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    exp = Experiment(
        pareto_path=args.pareto,
        slo_ms=args.slo,
        work_based=not args.integer,
        cooldown_s=args.cooldown,
        upscale_cooldown_s=args.upscale_cooldown,
    )
    if args.warmup:
        exp.warmup(keep_alive=args.keep_alive)

    result = exp.run(args.pattern, args.duration, args)

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Total requests:     {result.total}")
    print(f"Latency P50/P95/P99: {result.p50_ms:.0f}/{result.p95_ms:.0f}/{result.p99_ms:.0f} ms")
    print(f"SLO compliance:     {result.slo_compliance:.1f}% ({result.slo_violations} violations)")
    print(f"Average accuracy:   {result.avg_accuracy:.4f}")
    print(f"Config switches:    {len(result.switches)}")

    if args.output:
        out = {
            "config": {
                "slo_ms": args.slo, "pattern": args.pattern,
                "duration": args.duration, "work_based": not args.integer,
            },
            "metrics": {
                "total": result.total,
                "p50_ms": result.p50_ms, "p95_ms": result.p95_ms, "p99_ms": result.p99_ms,
                "slo_compliance": result.slo_compliance,
                "avg_accuracy": result.avg_accuracy,
                "switches": len(result.switches),
            },
            "switches": result.switches,
            "requests": [
                {
                    "id": r.id, "arrival": r.arrival_time,
                    "start": r.start_time, "end": r.end_time,
                    "config_idx": r.config_idx,
                    "response_ms": r.response_time_ms,
                    "wait_ms": r.wait_time_ms,
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
