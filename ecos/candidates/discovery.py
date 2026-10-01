"""Open-vocabulary candidate discovery."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ecos.preprocess.pipeline import PreprocessConfig, preprocess_document
from ecos.schema import Document, TemporalCorpus


@dataclass
class CandidateConfig:
    top_k: int = 100
    alpha: float = 0.25  # frequency change
    beta: float = 0.35  # distributional change
    gamma: float = 0.20  # contextual variability
    delta: float = 0.20  # temporal novelty
    min_freq: int = 3


def _period_token_counts(
    docs: list[Document],
    cfg: PreprocessConfig,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(preprocess_document(doc, cfg))
    return counts


def _normalize(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values()) or 1
    return {k: v / total for k, v in counter.items()}


def _cooccurrence_vector(
    docs: list[Document],
    term: str,
    cfg: PreprocessConfig,
    vocab: list[str],
    window: int = 5,
) -> np.ndarray:
    idx = {w: i for i, w in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=np.float64)
    for doc in docs:
        tokens = preprocess_document(doc, cfg)
        positions = [i for i, t in enumerate(tokens) if t == term]
        for pos in positions:
            left = max(0, pos - window)
            right = min(len(tokens), pos + window + 1)
            for j in range(left, right):
                if j == pos:
                    continue
                w = tokens[j]
                if w in idx:
                    vec[idx[w]] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def discover_candidates(
    corpus: TemporalCorpus,
    early_period: str,
    late_period: str,
    cand_cfg: CandidateConfig | None = None,
    prep_cfg: PreprocessConfig | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """
    Rank vocabulary by cheap multi-signal candidate score:
    C(w) = α F(w) + β D(w) + γ V(w) + δ T(w)
    """
    cand_cfg = cand_cfg or CandidateConfig()
    prep_cfg = prep_cfg or PreprocessConfig()
    slices = corpus.slices()
    if early_period not in slices or late_period not in slices:
        available = list(slices.keys())
        raise ValueError(f"Periods {early_period}/{late_period} not in {available}")

    early_docs = slices[early_period]
    late_docs = slices[late_period]
    early_counts = _period_token_counts(early_docs, prep_cfg)
    late_counts = _period_token_counts(late_docs, prep_cfg)
    vocab = sorted(
        {
            w
            for w, c in (early_counts + late_counts).items()
            if early_counts[w] + late_counts[w] >= cand_cfg.min_freq
        }
    )
    if not vocab:
        return []

    # Frequency change F: relative frequency delta
    early_rel = _normalize(early_counts)
    late_rel = _normalize(late_counts)
    f_raw = {
        w: abs(late_rel.get(w, 0.0) - early_rel.get(w, 0.0)) / (early_rel.get(w, 0.0) + late_rel.get(w, 0.0) + 1e-9)
        for w in vocab
    }

    # Distributional change D: cosine distance of co-occurrence vectors
    # Use top co-occurrence features from union vocab (capped)
    feature_vocab = vocab[: min(200, len(vocab))]
    d_raw: dict[str, float] = {}
    for w in vocab:
        v1 = _cooccurrence_vector(early_docs, w, prep_cfg, feature_vocab)
        v2 = _cooccurrence_vector(late_docs, w, prep_cfg, feature_vocab)
        sim = float(np.dot(v1, v2))
        d_raw[w] = 1.0 - max(min(sim, 1.0), -1.0)

    # Contextual variability V: entropy of neighbor distribution in late period
    v_raw: dict[str, float] = {}
    for w in vocab:
        vec = _cooccurrence_vector(late_docs, w, prep_cfg, feature_vocab)
        s = vec.sum()
        if s <= 0:
            v_raw[w] = 0.0
            continue
        p = vec / s
        p = p[p > 0]
        v_raw[w] = float(-(p * np.log(p + 1e-12)).sum())

    # Temporal novelty T: appears much more in late than early
    t_raw = {
        w: late_counts[w] / (early_counts[w] + late_counts[w] + 1e-9)
        for w in vocab
    }

    def _minmax(d: dict[str, float]) -> dict[str, float]:
        vals = np.array(list(d.values()), dtype=np.float64)
        lo, hi = float(vals.min()), float(vals.max())
        if hi - lo < 1e-12:
            return {k: 0.0 for k in d}
        return {k: (v - lo) / (hi - lo) for k, v in d.items()}

    F, D, V, T = _minmax(f_raw), _minmax(d_raw), _minmax(v_raw), _minmax(t_raw)
    ranked: list[tuple[str, float, dict[str, float]]] = []
    for w in vocab:
        score = (
            cand_cfg.alpha * F[w]
            + cand_cfg.beta * D[w]
            + cand_cfg.gamma * V[w]
            + cand_cfg.delta * T[w]
        )
        ranked.append(
            (
                w,
                float(score),
                {"F": F[w], "D": D[w], "V": V[w], "T": T[w]},
            )
        )
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked[: cand_cfg.top_k]


def collect_contexts(
    corpus: TemporalCorpus,
    term: str,
    period: str | None = None,
    window: int = 8,
    max_contexts: int = 50,
    prep_cfg: PreprocessConfig | None = None,
) -> list[dict]:
    """Collect contextual windows containing `term`."""
    prep_cfg = prep_cfg or PreprocessConfig(remove_stopwords=False)
    docs = corpus.documents if period is None else corpus.slices().get(period, [])
    contexts: list[dict] = []
    term_l = term.lower()
    for doc in docs:
        tokens = preprocess_document(doc, prep_cfg)
        for i, tok in enumerate(tokens):
            if tok != term_l:
                continue
            left = max(0, i - window)
            right = min(len(tokens), i + window + 1)
            snippet = " ".join(tokens[left:right])
            contexts.append(
                {
                    "document_id": doc.document_id,
                    "period": period
                    or (
                        str(doc.timestamp.year)
                        if corpus.window == "year"
                        else doc.timestamp.isoformat()
                    ),
                    "text": snippet,
                    "source": doc.source,
                    "topic": doc.topic,
                    "full_text": doc.text,
                }
            )
            if len(contexts) >= max_contexts:
                return contexts
    return contexts
