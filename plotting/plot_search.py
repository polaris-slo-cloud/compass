#!/usr/bin/env python3
"""Generate the COMPASS-V search figures from cached paper results.

Figures (Section VI.B of the paper):
    convergence   COMPASS-V vs Grid Search bounds at each SLO       (fig1)
    efficiency    Eval savings vs feasible-fraction across workflows (fig4)

Usage:
    python plotting/plot_search.py --figure all
    python plotting/plot_search.py --figure convergence
    python plotting/plot_search.py --figure efficiency
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RESULTS = Path("results/search")
DEFAULT_OUT = Path("figures/search")

COLORS = {
    "compass": "#0072B2",
    "grid_best": "#009E73",
    "grid_worst": "#D55E00",
    "grid_fill": "#CCCCCC",
}

# Cross-workflow efficiency summary (paper Table-equivalent).
RAG_EFFICIENCY = [
    {"slo": 0.30, "f_pct": 99.1, "savings": 68.4},
    {"slo": 0.40, "f_pct": 91.0, "savings": 48.0},
    {"slo": 0.50, "f_pct": 81.6, "savings": 32.1},
    {"slo": 0.60, "f_pct": 69.2, "savings": 20.3},
    {"slo": 0.75, "f_pct": 32.9, "savings": 45.2},
    {"slo": 0.80, "f_pct": 13.2, "savings": 66.1},
    {"slo": 0.85, "f_pct":  2.1, "savings": 84.7},
]
VISION_EFFICIENCY = [
    {"slo": 0.55, "f_pct": 98.7, "savings": 79.3},
    {"slo": 0.60, "f_pct": 96.1, "savings": 68.0},
    {"slo": 0.65, "f_pct": 90.6, "savings": 55.4},
    {"slo": 0.70, "f_pct": 76.9, "savings": 51.1},
    {"slo": 0.73, "f_pct": 56.4, "savings": 53.4},
    {"slo": 0.75, "f_pct": 39.5, "savings": 59.7},
    {"slo": 0.77, "f_pct": 17.7, "savings": 67.0},
    {"slo": 0.78, "f_pct":  9.9, "savings": 73.7},
]


def _ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2,
    })


def _load_compass_results(results_dir: Path) -> Dict[float, Dict[str, Any]]:
    """Load slo_*.json files (skipping *_timeline.json) under multi_slo/compass/."""
    out: Dict[float, Dict[str, Any]] = {}
    compass_dir = results_dir / "multi_slo" / "compass"
    if not compass_dir.exists():
        return out
    for f in sorted(compass_dir.glob("slo_*.json")):
        if "timeline" in f.name:
            continue
        try:
            slo = float(f.stem.replace("slo_", ""))
        except ValueError:
            continue
        out[slo] = json.load(open(f))
    return out


def _ground_truth_counts(results_dir: Path, fallback_from_compass: bool = True) -> Dict[float, int]:
    """Per-SLO feasible-count ground truth.

    Reads from `grid_100.json` if present (preferred — exhaustive). Otherwise
    falls back to COMPASS-V's reported feasible_count, which equals the GT
    when COMPASS-V hits 100% recall (the paper's reported result).
    """
    grid_path = results_dir / "grid_100.json"
    if grid_path.exists():
        grid = json.load(open(grid_path))
        all_evaluated = grid.get("all_evaluated") or grid.get("all_results", [])
        if all_evaluated:
            return {
                slo: sum(1 for c in all_evaluated if c["accuracy"] >= slo)
                for slo in (0.3, 0.4, 0.5, 0.6, 0.75, 0.8, 0.85, 0.9)
            }
    if fallback_from_compass:
        compass = _load_compass_results(results_dir)
        return {slo: d["metadata"]["feasible_count"] for slo, d in compass.items()}
    return {}


def _grid_bounds(
    gt_count: int, total_configs: int = 234, evals_per_config: int = 100
) -> Tuple[Tuple[List[float], List[int]], Tuple[List[float], List[int]]]:
    """Best/worst-case Grid Search convergence curves."""
    infeas = total_configs - gt_count
    best_x, best_y = [0], [0]
    worst_x, worst_y = [0], [0]
    for k in range(1, total_configs + 1):
        evals = k * evals_per_config
        best_x.append(evals)
        best_y.append(min(k, gt_count))
        worst_x.append(evals)
        worst_y.append(max(0, k - infeas))
    return (best_x, best_y), (worst_x, worst_y)


def plot_convergence(
    results_dir: Path,
    out_dir: Path,
    total_configs: int = 234,
) -> None:
    """fig1: per-SLO convergence subplots, COMPASS-V vs Grid bounds."""
    compass = _load_compass_results(results_dir)
    if not compass:
        print(f"  [convergence] no compass results under {results_dir}/multi_slo/compass — skipping")
        return
    gt = _ground_truth_counts(results_dir)

    slos = sorted(compass.keys())
    n = len(slos)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, slo in enumerate(slos):
        ax = axes[idx]
        gt_count = gt.get(slo, compass[slo]["metadata"]["feasible_count"])

        (bx, by), (wx, wy) = _grid_bounds(gt_count, total_configs=total_configs)
        bx_k = [v / 1000 for v in bx]
        wx_k = [v / 1000 for v in wx]
        ax.fill_between(bx_k, by, wy, color=COLORS["grid_fill"], alpha=0.4,
                        label="Grid range")
        ax.plot(bx_k, by, "--", color=COLORS["grid_best"], linewidth=1.2,
                alpha=0.9, label="Grid (best)")
        ax.plot(wx_k, wy, "--", color=COLORS["grid_worst"], linewidth=1.2,
                alpha=0.9, label="Grid (worst)")

        trace = compass[slo].get("anytime_trace") or []
        if trace:
            x, y = zip(*trace)
            ax.plot([v / 1000 for v in x], y, color=COLORS["compass"],
                    linewidth=2, label="COMPASS-V")

        if gt_count > 0:
            ax.axhline(gt_count, color="#666666", linestyle=":", linewidth=1.0,
                       alpha=0.7)

        ax.set_title(rf"$\tau$ = {slo:.2f}")
        ax.set_xlim(0, total_configs * 100 / 1000)
        ax.set_xlabel("Sample evals (k)")
        ax.set_ylabel("Feasible found")

    # Hide any unused axes
    for j in range(len(slos), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.02), frameon=True)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig1_convergence.{ext}")
    plt.close(fig)
    print(f"  [convergence] -> {out_dir}/fig1_convergence.{{pdf,png}}")


def plot_efficiency(out_dir: Path) -> None:
    """fig4: cross-workflow eval-savings vs feasible-fraction scatter."""
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    workflows = [
        ("RAG (234 configs)", RAG_EFFICIENCY, "o", "#1f77b4"),
        ("Object Detection (385 configs)", VISION_EFFICIENCY, "D", "#e07b39"),
    ]
    for label, data, marker, color in workflows:
        s = sorted(data, key=lambda d: d["f_pct"])
        ax.plot([d["f_pct"] for d in s], [d["savings"] for d in s],
                marker=marker, markersize=6, color=color,
                linewidth=1.4, label=label,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xlabel("Feasible Configs (%)")
    ax.set_ylabel("Eval Savings (%)")
    ax.set_xlim(-2, 105)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=len(workflows), frameon=True, columnspacing=1.0,
              handletextpad=0.4)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"fig4_efficiency_scatter.{ext}")
    plt.close(fig)
    print(f"  [efficiency] -> {out_dir}/fig4_efficiency_scatter.{{pdf,png}}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figure", choices=["convergence", "efficiency", "all"],
                        default="all")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS / "rag",
                        help="Directory with multi_slo/compass/ + grid_100.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    _ieee_style()
    print(f"Generating search figures in {args.out_dir}/")

    if args.figure in ("convergence", "all"):
        plot_convergence(args.results_dir, args.out_dir)

    if args.figure in ("efficiency", "all"):
        plot_efficiency(args.out_dir)


if __name__ == "__main__":
    main()
