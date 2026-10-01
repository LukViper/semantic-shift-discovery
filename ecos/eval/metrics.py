"""Evaluation metrics for LSC, discovery, FDR, and ranking."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


def spearman_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) < 2:
        return 0.0
    coef, _ = spearmanr(y_true, y_pred)
    return float(0.0 if np.isnan(coef) else coef)


def pearson_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    if len(y_true) < 2:
        return 0.0
    coef, _ = pearsonr(y_true, y_pred)
    return float(0.0 if np.isnan(coef) else coef)


def precision_at_k(y_true: dict[str, int], ranked: Sequence[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    hits = sum(1 for w in top if y_true.get(w, 0) == 1)
    return hits / len(top)


def recall_at_k(y_true: dict[str, int], ranked: Sequence[str], k: int) -> float:
    positives = [w for w, y in y_true.items() if y == 1]
    if not positives:
        return 0.0
    top = set(ranked[:k])
    hits = sum(1 for w in positives if w in top)
    return hits / len(positives)


def average_precision(y_true: dict[str, int], ranked: Sequence[str]) -> float:
    hits = 0
    sum_prec = 0.0
    for i, w in enumerate(ranked, 1):
        if y_true.get(w, 0) == 1:
            hits += 1
            sum_prec += hits / i
    denom = sum(1 for y in y_true.values() if y == 1) or 1
    return sum_prec / denom


def map_at_k(y_true: dict[str, int], ranked: Sequence[str], k: int) -> float:
    return average_precision(y_true, ranked[:k])


def ndcg_at_k(y_true: dict[str, int], ranked: Sequence[str], k: int) -> float:
    def dcg(rels: list[int]) -> float:
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(rels))

    rels = [int(y_true.get(w, 0)) for w in ranked[:k]]
    ideal = sorted((int(v) for v in y_true.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg <= 0:
        return 0.0
    return float(dcg(rels) / idcg)


def false_discovery_rate(y_true: dict[str, int], ranked: Sequence[str], k: int) -> float:
    top = ranked[:k]
    if not top:
        return 0.0
    tp = sum(1 for w in top if y_true.get(w, 0) == 1)
    fp = sum(1 for w in top if y_true.get(w, 0) != 1)
    return fp / (tp + fp) if (tp + fp) else 0.0


def ranking_metrics(
    y_true: dict[str, int],
    ranked: Sequence[str],
    k_values: Iterable[int] = (5, 10, 20),
) -> dict[str, float]:
    out: dict[str, float] = {
        "MAP": average_precision(y_true, ranked),
    }
    for k in k_values:
        out[f"P@{k}"] = precision_at_k(y_true, ranked, k)
        out[f"R@{k}"] = recall_at_k(y_true, ranked, k)
        out[f"MAP@{k}"] = map_at_k(y_true, ranked, k)
        out[f"NDCG@{k}"] = ndcg_at_k(y_true, ranked, k)
        out[f"FDR@{k}"] = false_discovery_rate(y_true, ranked, k)
    return out
