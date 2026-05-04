#!/usr/bin/env python3
"""Generate the Elastico runtime figures from cached paper results.

Figures (Section VI.C of the paper):
    bars         SLO compliance / accuracy bars vs SLO target  (fig:bars)
    scatter      Accuracy vs SLO-compliance scatter            (fig:scatter)
    cdf          Latency CDF across SLO targets                (fig:cdf)
    timeseries   Per-request latency + config switches         (fig:timeseries)

Reads the same JSON layout produced by `experiments/run_serving.py` and
`experiments/run_baseline.py`:
    <results-dir>/<pattern>_slo<ms>/{elastico,baseline_fastest,baseline_balanced,baseline_accurate}.json

Usage:
    python plotting/plot_serving.py --figure all
    python plotting/plot_serving.py --figure cdf       --pattern spike
    python plotting/plot_serving.py --figure timeseries --pattern bursty --slo 1000
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = Path("results/serving/run5")
DEFAULT_OUT = Path("figures/serving")

METHOD_ORDER = ["baseline_fastest", "baseline_balanced", "baseline_accurate", "elastico"]
METHOD_LABEL = {
    "baseline_fastest":  "Static-Fast",
    "baseline_balanced": "Static-Balanced",
    "baseline_accurate": "Static-Accurate",
    # Older runs used "baseline_middle" — accept it as an alias for balanced.
    "baseline_middle":   "Static-Balanced",
    "elastico":          "Elastico",
}
METHOD_COLOR = {
    "baseline_fastest":  "#1f77b4",
    "baseline_balanced": "#ff7f0e",
    "baseline_middle":   "#ff7f0e",
    "baseline_accurate": "#2ca02c",
    "elastico":          "#d62728",
}
METHOD_HATCH = {
    "baseline_fastest":  "",
    "baseline_balanced": "//",
    "baseline_middle":   "//",
    "baseline_accurate": "\\\\",
    "elastico":          "xx",
}


def _ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    })


# -- data loading --

_DIR_RE = re.compile(r"^(?P<pattern>\w+)_slo(?P<slo>\d+)$")


def _load_results(results_dir: Path) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    """Return dict keyed by (pattern, slo_ms, method)."""
    out: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for d in sorted(results_dir.iterdir()) if results_dir.exists() else []:
        if not d.is_dir():
            continue
        m = _DIR_RE.match(d.name)
        if not m:
            continue
        pattern = m.group("pattern")
        slo = int(m.group("slo"))
        for f in d.glob("*.json"):
            method = f.stem
            with open(f) as fp:
                out[(pattern, slo, method)] = json.load(fp)
    return out


def _slos_for(data: Dict, pattern: str) -> List[int]:
    return sorted({k[1] for k in data if k[0] == pattern})


def _methods_for(data: Dict, pattern: str, slo: int) -> List[str]:
    have = {k[2] for k in data if k[0] == pattern and k[1] == slo}
    return [m for m in METHOD_ORDER if m in have or
            (m == "baseline_balanced" and "baseline_middle" in have)]


# -- bars --

def plot_bars(data: Dict, pattern: str, out_dir: Path) -> None:
    slos = _slos_for(data, pattern)
    if not slos:
        return
    methods = _methods_for(data, pattern, slos[0])
    if not methods:
        return

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    metrics = [
        ("slo_compliance", "SLO Compliance (%)", (0, 110)),
        ("avg_accuracy",   "Accuracy (F1)",       (0.70, 0.90)),
        ("p95_ms",         "P95 Latency (ms)",    None),
    ]

    x = np.arange(len(slos))
    width = 0.8 / len(methods)

    for ax, (mname, ylabel, ylim) in zip(axes, metrics):
        for i, method in enumerate(methods):
            values = []
            for slo in slos:
                key = (pattern, slo, method)
                if key not in data and method == "baseline_balanced":
                    key = (pattern, slo, "baseline_middle")
                values.append(data[key]["metrics"][mname] if key in data else 0)
            ax.bar(
                x + i * width, values, width,
                label=METHOD_LABEL[method],
                color=METHOD_COLOR[method],
                hatch=METHOD_HATCH[method],
                edgecolor="black", linewidth=0.5,
            )
        if mname == "p95_ms":
            for i, slo in enumerate(slos):
                ax.hlines(slo, xmin=i - 0.05, xmax=i + width * len(methods) + 0.05,
                          colors="black", linestyles="--", linewidth=1.2)
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([str(s) for s in slos])
        ax.set_xlabel("SLO Target (ms)")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods),
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.suptitle(f"{pattern.capitalize()} Pattern", y=1.06, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"bars_{pattern}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [bars] -> {out_dir}/bars_{pattern}.{{pdf,png}}")


# -- scatter --

def plot_scatter(data: Dict, pattern: str, out_dir: Path) -> None:
    slos = _slos_for(data, pattern)
    if not slos:
        return
    methods = _methods_for(data, pattern, slos[0])

    markers = ["o", "s", "^", "D"]
    while len(markers) < len(slos):
        markers.append("v")

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        for slo, marker in zip(slos, markers):
            key = (pattern, slo, method)
            if key not in data and method == "baseline_balanced":
                key = (pattern, slo, "baseline_middle")
            if key not in data:
                continue
            m = data[key]["metrics"]
            label = METHOD_LABEL[method] if slo == slos[0] else None
            ax.scatter(
                m["slo_compliance"], m["avg_accuracy"],
                color=METHOD_COLOR[method], marker=marker, s=110,
                label=label,
                edgecolors="black", linewidths=0.5, alpha=0.85,
            )
    for slo, marker in zip(slos, markers):
        ax.scatter([], [], c="gray", marker=marker, s=80,
                   label=f"SLO={slo}ms", edgecolors="black", linewidths=0.5)

    ax.axvline(x=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("SLO Compliance (%)")
    ax.set_ylabel("Accuracy (F1)")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower left", framealpha=0.9, ncol=2)
    ax.set_title(f"{pattern.capitalize()} Pattern: Accuracy vs SLO Compliance",
                 fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"scatter_{pattern}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [scatter] -> {out_dir}/scatter_{pattern}.{{pdf,png}}")


# -- cdf --

def plot_cdf(data: Dict, pattern: str, out_dir: Path) -> None:
    slos = _slos_for(data, pattern)
    if not slos:
        return

    n = len(slos)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7, 2.7 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, slo in zip(axes, slos):
        for method, ls in (
            ("elastico", "-"),
            ("baseline_fastest", "--"),
            ("baseline_accurate", ":"),
        ):
            key = (pattern, slo, method)
            if key not in data:
                continue
            latencies = sorted(r["response_ms"] for r in data[key]["requests"])
            if not latencies:
                continue
            percentiles = np.linspace(0, 100, len(latencies))
            ax.plot(latencies, percentiles,
                    label=METHOD_LABEL[method],
                    color=METHOD_COLOR[method], linestyle=ls)

        ax.axvline(x=slo, color="black", linestyle="-", linewidth=1, alpha=0.7)
        ax.axhline(y=95, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Response Latency (ms)")
        ax.set_ylabel("Percentile")
        ax.set_title(f"SLO = {slo} ms")
        ax.set_xlim(0, slo * 2.5)
        ax.set_ylim(0, 100)
        if slo == slos[0]:
            ax.legend(loc="lower right", framealpha=0.9)

    for j in range(len(slos), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"{pattern.capitalize()} Pattern: Latency CDF",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"cdf_{pattern}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [cdf] -> {out_dir}/cdf_{pattern}.{{pdf,png}}")


# -- timeseries --

def _detect_bursts(arrivals: List[float], duration: float,
                   bin_size: float = 5.0, threshold: float = 1.8) -> List[Tuple[float, float]]:
    bins: Dict[int, int] = defaultdict(int)
    for a in arrivals:
        bins[int(a // bin_size)] += 1
    out = []
    in_burst = False
    burst_start = 0.0
    for i in range(int(duration // bin_size) + 1):
        rate = bins.get(i, 0) / bin_size
        t = i * bin_size
        if rate > threshold and not in_burst:
            in_burst = True
            burst_start = t
        elif rate <= threshold and in_burst:
            in_burst = False
            out.append((burst_start, t))
    if in_burst:
        out.append((burst_start, duration))
    return out


def plot_timeseries(data: Dict, pattern: str, slo: int, out_dir: Path) -> None:
    key = (pattern, slo, "elastico")
    if key not in data:
        print(f"  [timeseries] no elastico data for {pattern} SLO={slo}, skipping")
        return

    d = data[key]
    requests = d["requests"]
    switches = d["switches"]
    duration = d["config"]["duration"]
    metrics = d["metrics"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})

    arrivals = [r["arrival"] for r in requests]
    latencies = [r["response_ms"] for r in requests]
    configs = [r["config_idx"] for r in requests]

    cfg_colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    cfg_names = ["Fast", "Balanced", "Accurate"]
    config_counts: Dict[int, int] = defaultdict(int)
    for c in configs:
        config_counts[c] += 1
    total = max(1, len(configs))

    ax1 = axes[0]
    for cfg in range(3):
        ca = [a for a, c in zip(arrivals, configs) if c == cfg]
        cl = [l for l, c in zip(latencies, configs) if c == cfg]
        if not ca:
            continue
        pct = 100 * config_counts[cfg] / total
        ax1.scatter(ca, cl, color=cfg_colors[cfg], s=12, alpha=0.6,
                    rasterized=True,
                    label=f"{cfg_names[cfg]} ({pct:.0f}%)")

    ax1.axhline(y=slo, color="black", linestyle="--", linewidth=1.5, label="SLO")
    ax1.set_ylabel("Response Latency (ms)")
    if latencies:
        max_lat = float(np.percentile(latencies, 99))
        ax1.set_ylim(0, min(slo * 2.2, max_lat * 1.3))
    ax1.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=8)

    if pattern == "spike":
        ax1.axvspan(duration / 3, 2 * duration / 3, alpha=0.15, color="red", zorder=0)
    elif pattern == "bursty":
        for start, end in _detect_bursts(arrivals, duration):
            ax1.axvspan(start, end, alpha=0.15, color="red", zorder=0)

    ax2 = axes[1]
    times = [0.0]
    cfg_vals = [requests[0]["config_idx"]] if requests else [2]
    for s in switches:
        times.append(s["t"])
        cfg_vals.append(s["to"])
    times.append(duration)
    cfg_vals.append(cfg_vals[-1])

    for i in range(len(times) - 1):
        cfg = cfg_vals[i]
        ax2.fill_between([times[i], times[i + 1]], [cfg, cfg],
                         step="post", alpha=0.6, color=cfg_colors[cfg])
    ax2.step(times, cfg_vals, where="post", linewidth=1, color="black")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Config")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(cfg_names)
    ax2.set_ylim(-0.3, 2.5)
    ax2.set_xlim(0, duration)

    fig.suptitle(
        f"{pattern.capitalize()} | SLO = {slo}ms | "
        f"Compliance: {metrics['slo_compliance']:.1f}% | "
        f"Accuracy: {metrics['avg_accuracy']:.3f} | "
        f"Switches: {metrics['switches']}",
        fontweight="bold", y=0.99,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"timeseries_{pattern}_slo{slo}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  [timeseries] -> {out_dir}/timeseries_{pattern}_slo{slo}.{{pdf,png}}")


# -- main --

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure",
                        choices=["bars", "scatter", "cdf", "timeseries", "all"],
                        default="all")
    parser.add_argument("--pattern", default=None,
                        help="Restrict to one pattern (default: all patterns present)")
    parser.add_argument("--slo", type=int, default=None,
                        help="For --figure timeseries: which SLO to plot (default: all SLOs present)")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    _ieee_style()
    print(f"Generating runtime figures in {args.out_dir}/")

    data = _load_results(args.results_dir)
    if not data:
        print(f"  No results found under {args.results_dir}")
        return

    patterns = sorted({k[0] for k in data})
    if args.pattern:
        patterns = [p for p in patterns if p == args.pattern]

    for pat in patterns:
        if args.figure in ("bars", "all"):
            plot_bars(data, pat, args.out_dir)
        if args.figure in ("scatter", "all"):
            plot_scatter(data, pat, args.out_dir)
        if args.figure in ("cdf", "all"):
            plot_cdf(data, pat, args.out_dir)
        if args.figure in ("timeseries", "all"):
            slos = _slos_for(data, pat)
            if args.slo is not None:
                slos = [args.slo] if args.slo in slos else []
            for slo in slos:
                plot_timeseries(data, pat, slo, args.out_dir)


if __name__ == "__main__":
    main()
