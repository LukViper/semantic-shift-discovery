#!/usr/bin/env python3
"""Regenerate paper figures from experiment JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def main() -> None:
    style()
    disc = json.loads((ROOT / "data/reports/experiments/exp_discovery_benchmarks.json").read_text())
    exp3 = json.loads((ROOT / "data/reports/experiments/exp3_false_discovery.json").read_text())
    abl = json.loads((ROOT / "data/reports/experiments/ablations.json").read_text())
    compact = json.loads((ROOT / "data/reports/experiments/summary_compact.json").read_text())
    reports = json.loads((ROOT / "data/reports/demo/shift_reports.json").read_text())

    # discovery precision
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, ds, title in [(axes[0], "liverpoolfc", "LiverpoolFC"), (axes[1], "semeval_en", "SemEval-EN")]:
        methods = ["contextual_aggregate", "sense_cluster", "ecos"]
        labels = ["Contextual", "Sense cluster", "ECOS"]
        x = np.arange(3)
        width = 0.25
        for i, k in enumerate([5, 10, 15]):
            vals = [disc[ds]["methods"][m][f"P@{k}"] for m in methods]
            ax.bar(
                x + (i - 1) * width,
                vals,
                width,
                label=f"P@{k}",
                color=["#1b4f72", "#2874a6", "#5dade2"][i],
                edgecolor="white",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        if ds == "liverpoolfc":
            ax.set_ylabel("Precision")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Open-vocabulary discovery precision", fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig_discovery_precision")

    # FDR
    scenarios = ["original", "topic_only", "frequency_only"]
    labels = ["Original", "Topic-only\nperturbation", "Frequency-only\nperturbation"]
    base = [exp3[s]["baseline_fdr@10"] for s in scenarios]
    ecos = [exp3[s]["ecos_fdr@10"] for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - width / 2, base, width, label="Contextual baseline", color="#8d6e63", edgecolor="white")
    ax.bar(x + width / 2, ecos, width, label="ECOS", color="#0b3d5c", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FDR@10 (lower is better)")
    ax.set_ylim(0, 1.0)
    ax.set_title("False-discovery resistance under confounder perturbations")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_fdr_resistance")

    # ablation
    order = ["full", "no_counterfactual", "no_calibration", "weak_controls"]
    names = ["Full ECOS", "− Counterfactual", "− Calibration", "Weak controls"]
    fdr = [abl[k]["FDR@10"] for k in order]
    p5 = [abl[k]["P@5"] for k in order]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(order))
    ax.plot(x, fdr, "o-", color="#a93226", linewidth=2, markersize=7, label="FDR@10")
    ax.plot(x, p5, "s--", color="#1e8449", linewidth=2, markersize=7, label="P@5")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Ablation on SEM-STREAM pilot")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_ablation")

    # semstream ranking
    methods = ["static_embedding", "contextual_aggregate", "sense_cluster", "ecos"]
    mlabels = ["Static", "Contextual", "Sense", "ECOS"]
    metrics = ["P@5", "MAP", "NDCG@10"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(mlabels))
    width = 0.25
    palette = ["#1a5276", "#148f77", "#b9770e"]
    for i, m in enumerate(metrics):
        vals = [compact["exp1"][meth][m] for meth in methods]
        ax.bar(x + (i - 1) * width, vals, width, label=m, color=palette[i], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(mlabels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SEM-STREAM pilot: ranking quality")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    save(fig, "fig_semstream_ranking")

    # signal breakdown
    terms = [r["term"] for r in reports[:6]]
    ctx = [r["contextual_shift"] for r in reports[:6]]
    sense = [r["sense_redistribution"] for r in reports[:6]]
    cf = [r["counterfactual_robustness"] for r in reports[:6]]
    conf = [r["calibrated_confidence"] for r in reports[:6]]
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(terms))
    w = 0.2
    ax.bar(x - 1.5 * w, ctx, w, label="Contextual", color="#1a5276")
    ax.bar(x - 0.5 * w, sense, w, label="Sense", color="#148f77")
    ax.bar(x + 0.5 * w, cf, w, label="CF robustness", color="#b9770e")
    ax.bar(x + 1.5 * w, conf, w, label="Confidence", color="#6c3483")
    ax.set_xticks(x)
    ax.set_xticklabels(terms, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Signal breakdown for top SEM-STREAM discoveries")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_signal_breakdown")

    # architecture boxes
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    boxes = [
        (0.3, 4.2, 1.8, 0.9, "Longitudinal\ncorpus", "#eaf2f8"),
        (2.4, 4.2, 1.8, 0.9, "Candidate\ndiscovery", "#eaf2f8"),
        (4.5, 4.2, 1.8, 0.9, "Contextual\n+ senses", "#d5f5e3"),
        (6.6, 4.2, 1.8, 0.9, "Shift\nengine", "#d5f5e3"),
        (1.3, 2.5, 1.8, 0.9, "Confounder\ncontrols", "#fdebd0"),
        (3.4, 2.5, 1.8, 0.9, "Counterfactual\nmatching", "#fdebd0"),
        (5.5, 2.5, 1.8, 0.9, "Evidence +\nECSS / calib.", "#f5b7b1"),
        (7.6, 2.5, 1.8, 0.9, "Shift report +\nObservatory", "#f5b7b1"),
    ]
    for x0, y0, w, h, txt, c in boxes:
        rect = plt.Rectangle((x0, y0), w, h, facecolor=c, edgecolor="#2c3e50", linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=8.5, zorder=3)
    for x0 in [2.1, 4.2, 6.3]:
        ax.annotate(
            "",
            xy=(x0 + 0.25, 4.65),
            xytext=(x0 - 0.05, 4.65),
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
        )
    ax.annotate(
        "",
        xy=(4.2, 3.4),
        xytext=(7.5, 4.2),
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.0, connectionstyle="arc3,rad=0.15"),
    )
    for x0 in [3.1, 5.2, 7.3]:
        ax.annotate(
            "",
            xy=(x0 + 0.25, 2.95),
            xytext=(x0 - 0.05, 2.95),
            arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2),
        )
    ax.set_title("ECOS architecture: discovery → validation → evidence", fontsize=12, pad=8)
    fig.tight_layout()
    save(fig, "fig_architecture")


if __name__ == "__main__":
    main()
