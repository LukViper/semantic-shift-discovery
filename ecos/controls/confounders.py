"""Confounder controls: topic, source, frequency, corpus composition."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon

from ecos.preprocess.pipeline import PreprocessConfig, preprocess_document
from ecos.schema import ControlResult, Document


@dataclass
class ControlThresholds:
    topic: float = 0.35
    source: float = 0.35
    frequency: float = 0.50


def _categorical_distribution(docs: list[Document], attr: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for d in docs:
        val = getattr(d, attr, None) or "unknown"
        counts[str(val)] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def _js_from_dicts(p: dict[str, float], q: dict[str, float]) -> float:
    keys = sorted(set(p) | set(q))
    if not keys:
        return 0.0
    a = np.array([p.get(k, 0.0) for k in keys], dtype=np.float64)
    b = np.array([q.get(k, 0.0) for k in keys], dtype=np.float64)
    return float(jensenshannon(a + 1e-12, b + 1e-12))


def topic_effect(early_docs: list[Document], late_docs: list[Document]) -> ControlResult:
    dist = _js_from_dicts(
        _categorical_distribution(early_docs, "topic"),
        _categorical_distribution(late_docs, "topic"),
    )
    # PASS means shift is NOT fully explained by topic change alone —
    # i.e. topic divergence is below threshold OR we flag low topic confound.
    # Interpretation: lower score = less confounder pressure.
    return ControlResult(
        name="topic",
        passed=dist < 0.9,  # refined by thresholds in evaluate_controls
        score=dist,
        detail=f"topic JS divergence={dist:.3f}",
    )


def source_effect(early_docs: list[Document], late_docs: list[Document]) -> ControlResult:
    dist = _js_from_dicts(
        _categorical_distribution(early_docs, "source"),
        _categorical_distribution(late_docs, "source"),
    )
    # Also consider community skew
    dist_c = _js_from_dicts(
        _categorical_distribution(early_docs, "community"),
        _categorical_distribution(late_docs, "community"),
    )
    score = max(dist, dist_c)
    return ControlResult(
        name="source",
        passed=True,
        score=score,
        detail=f"source/community JS={score:.3f}",
    )


def frequency_effect(
    term: str,
    early_docs: list[Document],
    late_docs: list[Document],
    prep_cfg: PreprocessConfig | None = None,
) -> ControlResult:
    prep_cfg = prep_cfg or PreprocessConfig()
    term_l = term.lower()

    def rel_freq(docs: list[Document]) -> float:
        total_tok = 0
        hit = 0
        for d in docs:
            toks = preprocess_document(d, prep_cfg)
            total_tok += len(toks)
            hit += sum(1 for t in toks if t == term_l)
        return hit / (total_tok or 1)

    e = rel_freq(early_docs)
    l = rel_freq(late_docs)
    # Relative change magnitude
    score = abs(l - e) / (e + l + 1e-9)
    return ControlResult(
        name="frequency",
        passed=True,
        score=float(score),
        detail=f"rel_freq early={e:.4g} late={l:.4g}",
    )


def composition_effect(early_docs: list[Document], late_docs: list[Document]) -> ControlResult:
    """Domain composition divergence."""
    dist = _js_from_dicts(
        _categorical_distribution(early_docs, "domain"),
        _categorical_distribution(late_docs, "domain"),
    )
    return ControlResult(
        name="composition",
        passed=True,
        score=dist,
        detail=f"domain JS={dist:.3f}",
    )


def evaluate_controls(
    term: str,
    early_docs: list[Document],
    late_docs: list[Document],
    thresholds: ControlThresholds | None = None,
    observed_shift: float = 0.0,
) -> list[ControlResult]:
    """
    A control PASSes when the confounder magnitude is not large enough to
    fully explain the observed semantic shift (heuristic).
    """
    thresholds = thresholds or ControlThresholds()
    topic = topic_effect(early_docs, late_docs)
    source = source_effect(early_docs, late_docs)
    freq = frequency_effect(term, early_docs, late_docs)
    comp = composition_effect(early_docs, late_docs)

    def decide(ctrl: ControlResult, thr: float) -> ControlResult:
        # Fail only when confounder is strong AND rivals/exceeds the semantic signal.
        # Mild corpus drift alone is not enough to reject a shift hypothesis.
        explains = ctrl.score >= thr and ctrl.score >= max(0.45, 0.85 * max(observed_shift, 1e-6))
        return ControlResult(
            name=ctrl.name,
            passed=not explains,
            score=ctrl.score,
            detail=ctrl.detail + (" | FAIL: confounder-dominated" if explains else " | PASS"),
        )

    return [
        decide(topic, thresholds.topic),
        decide(source, thresholds.source),
        decide(freq, thresholds.frequency),
        decide(comp, thresholds.topic),
    ]
