"""Baseline semantic-change methods for comparison."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from ecos.candidates.discovery import collect_contexts
from ecos.preprocess.pipeline import PreprocessConfig, preprocess_document
from ecos.representations.encoder import ContextualEncoder, EncoderConfig, cosine_distance, mean_embedding
from ecos.schema import TemporalCorpus
from ecos.sense.induction import SenseConfig, induce_senses, sense_distribution
from scipy.spatial.distance import jensenshannon


def static_embedding_shift(
    corpus: TemporalCorpus,
    term: str,
    early_period: str,
    late_period: str,
) -> float:
    """
    Cheap static baseline: cosine distance between averaged co-occurrence
    (PPMI-like) vectors approximated via hashing BoW of contexts.
    """
    early = [c["text"] for c in collect_contexts(corpus, term, early_period)]
    late = [c["text"] for c in collect_contexts(corpus, term, late_period)]
    if not early or not late:
        return 0.0
    enc = ContextualEncoder(EncoderConfig(backend="bag_of_words"))
    enc.fit(early + late)
    e = mean_embedding(enc.encode(early))
    l = mean_embedding(enc.encode(late))
    return cosine_distance(e, l)


def contextual_aggregate_shift(
    corpus: TemporalCorpus,
    term: str,
    early_period: str,
    late_period: str,
    encoder: ContextualEncoder | None = None,
) -> float:
    """Contextual baseline: distance between mean contextual embeddings."""
    encoder = encoder or ContextualEncoder()
    early = [c["text"] for c in collect_contexts(corpus, term, early_period)]
    late = [c["text"] for c in collect_contexts(corpus, term, late_period)]
    if not early or not late:
        return 0.0
    encoder.fit(early + late)
    return cosine_distance(mean_embedding(encoder.encode(early)), mean_embedding(encoder.encode(late)))


def sense_cluster_shift(
    corpus: TemporalCorpus,
    term: str,
    early_period: str,
    late_period: str,
    encoder: ContextualEncoder | None = None,
    n_clusters: int = 3,
) -> float:
    """Sense-based baseline: JS divergence of sense distributions."""
    encoder = encoder or ContextualEncoder()
    early = [c["text"] for c in collect_contexts(corpus, term, early_period)]
    late = [c["text"] for c in collect_contexts(corpus, term, late_period)]
    if not early or not late:
        return 0.0
    encoder.fit(early + late)
    e_emb = encoder.encode(early)
    l_emb = encoder.encode(late)
    # Fit shared kmeans on combined
    combined = np.vstack([e_emb, l_emb])
    model = induce_senses(combined, SenseConfig(n_clusters=n_clusters))
    if model.centers is None:
        return 0.0
    e_labels = np.linalg.norm(e_emb[:, None, :] - model.centers[None, :, :], axis=2).argmin(axis=1)
    l_labels = np.linalg.norm(l_emb[:, None, :] - model.centers[None, :, :], axis=2).argmin(axis=1)
    e_dist = sense_distribution(e_labels, model.n_senses)
    l_dist = sense_distribution(l_labels, model.n_senses)
    return float(jensenshannon(e_dist + 1e-12, l_dist + 1e-12))


def score_vocabulary(
    corpus: TemporalCorpus,
    terms: Sequence[str],
    early_period: str,
    late_period: str,
    method: str = "contextual_aggregate",
) -> dict[str, float]:
    enc = ContextualEncoder()
    scores: dict[str, float] = {}
    for term in terms:
        if method == "static_embedding":
            scores[term] = static_embedding_shift(corpus, term, early_period, late_period)
        elif method == "sense_cluster":
            scores[term] = sense_cluster_shift(corpus, term, early_period, late_period, encoder=enc)
        else:
            scores[term] = contextual_aggregate_shift(
                corpus, term, early_period, late_period, encoder=enc
            )
    return scores


def frequency_only_shift(
    corpus: TemporalCorpus,
    term: str,
    early_period: str,
    late_period: str,
) -> float:
    """Confounder baseline: relative frequency change only."""
    prep = PreprocessConfig()
    slices = corpus.slices()
    term_l = term.lower()

    def rf(docs):
        total = hit = 0
        for d in docs:
            toks = preprocess_document(d, prep)
            total += len(toks)
            hit += sum(1 for t in toks if t == term_l)
        return hit / (total or 1)

    e, l = rf(slices.get(early_period, [])), rf(slices.get(late_period, []))
    return abs(l - e) / (e + l + 1e-9)
