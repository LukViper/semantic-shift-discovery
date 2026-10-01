"""Counterfactual matching / reweighting for robust shift verification."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ecos.schema import Document


@dataclass
class CounterfactualConfig:
    matching: str = "reweight"  # reweight | subsample
    n_bins: int = 5
    attrs: tuple[str, ...] = ("topic", "source", "community")


def _attr_value(doc: Document, attr: str) -> str:
    return str(getattr(doc, attr, None) or "unknown")


def stratum_key(doc: Document, attrs: tuple[str, ...]) -> str:
    return "||".join(_attr_value(doc, a) for a in attrs)


def target_stratum_weights(early_docs: list[Document], attrs: tuple[str, ...]) -> dict[str, float]:
    counts: Counter[str] = Counter(stratum_key(d, attrs) for d in early_docs)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def reweight_late_docs(
    late_docs: list[Document],
    target_weights: dict[str, float],
    attrs: tuple[str, ...],
) -> tuple[list[Document], np.ndarray]:
    """
    Assign importance weights to late documents to match early stratum margins.
    Returns (docs, weights).
    """
    if not late_docs:
        return [], np.array([])

    groups: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(late_docs):
        groups[stratum_key(d, attrs)].append(i)

    weights = np.zeros(len(late_docs), dtype=np.float64)
    for key, idxs in groups.items():
        target = target_weights.get(key, 0.0)
        # Distribute target mass uniformly across docs in stratum
        w = target / len(idxs) if idxs else 0.0
        for i in idxs:
            weights[i] = w

    # If late has strata missing from early, their weight stays 0 (downweighted)
    # Renormalize
    s = weights.sum()
    if s > 0:
        weights /= s
    else:
        weights[:] = 1.0 / len(weights)
    return late_docs, weights


def subsample_matched(
    late_docs: list[Document],
    target_weights: dict[str, float],
    attrs: tuple[str, ...],
    n: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[Document]:
    """Multinomial subsample of late docs toward early stratum distribution."""
    rng = rng or np.random.default_rng(42)
    n = n or len(late_docs)
    if not late_docs:
        return []
    _, weights = reweight_late_docs(late_docs, target_weights, attrs)
    idxs = rng.choice(len(late_docs), size=min(n, len(late_docs)), replace=True, p=weights)
    return [late_docs[i] for i in idxs]


def counterfactual_shift(
    early_docs: list[Document],
    late_docs: list[Document],
    observed_score: float,
    score_fn: Callable[[list[Document], list[Document]], float],
    cfg: CounterfactualConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Recompute semantic score under matched late distribution.

    score_fn(early_docs, late_docs_or_weighted_proxy) -> float
    For reweighting, we subsample as a practical proxy usable by text pipelines.
    """
    cfg = cfg or CounterfactualConfig()
    target = target_stratum_weights(early_docs, cfg.attrs)
    matched = subsample_matched(late_docs, target, cfg.attrs, rng=rng)
    cf_score = float(score_fn(early_docs, matched)) if matched else 0.0
    robustness = 1.0 - abs(observed_score - cf_score) / (observed_score + 1e-9)
    robustness = float(max(0.0, min(1.0, robustness)))
    # If CF score remains high, shift is robust; if collapses, confounder-driven
    survive = cf_score / (observed_score + 1e-9)
    return {
        "observed": float(observed_score),
        "counterfactual": cf_score,
        "robustness": robustness,
        "survival_ratio": float(min(survive, 2.0)),
    }
