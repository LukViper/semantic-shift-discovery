"""Semantic shift scoring signals and ECSS aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon

from ecos.representations.encoder import cosine_distance, mean_embedding
from ecos.sense.induction import align_sense_distributions, classify_shift_type


@dataclass
class ShiftWeights:
    context: float = 0.35
    sense: float = 0.30
    neighborhood: float = 0.20
    temporal: float = 0.15


@dataclass
class ShiftSignals:
    contextual: float
    sense: float
    neighborhood: float
    temporal: float
    changepoint: str | None
    shift_type: str
    early_sense_dist: list[float]
    late_sense_dist: list[float]


def contextual_divergence(early_emb: np.ndarray, late_emb: np.ndarray) -> float:
    return cosine_distance(mean_embedding(early_emb), mean_embedding(late_emb))


def sense_shift_score(early_emb: np.ndarray, late_emb: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, str]:
    early_d, late_d = align_sense_distributions(
        np.array([]),
        np.array([]),
        early_emb,
        late_emb,
    )
    # Pad to same length
    n = max(len(early_d), len(late_d))
    e = np.zeros(n)
    l = np.zeros(n)
    e[: len(early_d)] = early_d
    l[: len(late_d)] = late_d
    js = float(jensenshannon(e + 1e-12, l + 1e-12))
    # jensenshannon returns sqrt(JS); already in [0,1]
    shift_type = classify_shift_type(e, l)
    return js, e, l, shift_type


def neighborhood_shift(
    early_neigh: list[str],
    late_neigh: list[str],
) -> float:
    """Jaccard distance between top lexical neighborhoods."""
    a, b = set(early_neigh), set(late_neigh)
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return 1.0 - inter / union


def top_neighbors_from_contexts(contexts: list[str], term: str, k: int = 10) -> list[str]:
    from collections import Counter

    from ecos.preprocess.pipeline import tokenize

    counts: Counter[str] = Counter()
    term_l = term.lower()
    for ctx in contexts:
        for tok in tokenize(ctx):
            if tok != term_l:
                counts[tok] += 1
    return [w for w, _ in counts.most_common(k)]


def temporal_changepoint(
    period_embeddings: dict[str, np.ndarray],
) -> tuple[float, str | None]:
    """
    Detect period with largest consecutive mean-embedding jump.
    Returns (max_jump, changepoint_label as late period of the jump).
    """
    periods = sorted(period_embeddings.keys())
    if len(periods) < 2:
        return 0.0, None
    jumps: list[tuple[float, str]] = []
    prev_mean = mean_embedding(period_embeddings[periods[0]])
    for p in periods[1:]:
        cur = mean_embedding(period_embeddings[p])
        jumps.append((cosine_distance(prev_mean, cur), p))
        prev_mean = cur
    best = max(jumps, key=lambda x: x[0])
    # Normalize temporal score relative to max possible 1.0
    return float(best[0]), best[1]


def combine_signals(signals: ShiftSignals, weights: ShiftWeights | None = None) -> float:
    weights = weights or ShiftWeights()
    raw = (
        weights.context * signals.contextual
        + weights.sense * signals.sense
        + weights.neighborhood * signals.neighborhood
        + weights.temporal * signals.temporal
    )
    return float(max(0.0, min(1.0, raw)))


def compute_shift_signals(
    term: str,
    early_emb: np.ndarray,
    late_emb: np.ndarray,
    early_contexts: list[str],
    late_contexts: list[str],
    period_embeddings: dict[str, np.ndarray] | None = None,
) -> ShiftSignals:
    contextual = contextual_divergence(early_emb, late_emb)
    sense, e_dist, l_dist, shift_type = sense_shift_score(early_emb, late_emb)
    early_n = top_neighbors_from_contexts(early_contexts, term)
    late_n = top_neighbors_from_contexts(late_contexts, term)
    neigh = neighborhood_shift(early_n, late_n)
    temporal = 0.0
    cp = None
    if period_embeddings:
        temporal, cp = temporal_changepoint(period_embeddings)
    return ShiftSignals(
        contextual=contextual,
        sense=sense,
        neighborhood=neigh,
        temporal=temporal,
        changepoint=cp,
        shift_type=shift_type,
        early_sense_dist=e_dist.tolist(),
        late_sense_dist=l_dist.tolist(),
    )
