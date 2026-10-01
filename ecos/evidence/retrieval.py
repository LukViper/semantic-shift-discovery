"""Representative evidence retrieval for shift reports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecos.schema import EvidenceExample


@dataclass
class EvidenceConfig:
    n_per_period: int = 5
    max_chars: int = 240


def _medoid_indices(embeddings: np.ndarray, k: int) -> list[int]:
    """Select k diverse medoid-like examples via farthest-point + nearest-to-centroid."""
    n = embeddings.shape[0]
    if n == 0:
        return []
    k = min(k, n)
    centroid = embeddings.mean(axis=0)
    dists = np.linalg.norm(embeddings - centroid, axis=1)
    # Start with closest to centroid
    selected = [int(dists.argmin())]
    while len(selected) < k:
        # Farthest from already selected
        sel_emb = embeddings[selected]
        dmin = np.linalg.norm(embeddings[:, None, :] - sel_emb[None, :, :], axis=2).min(axis=1)
        dmin[selected] = -1
        selected.append(int(dmin.argmax()))
    return selected


def retrieve_evidence(
    early_texts: list[str],
    late_texts: list[str],
    early_emb: np.ndarray,
    late_emb: np.ndarray,
    early_period: str,
    late_period: str,
    cfg: EvidenceConfig | None = None,
) -> list[EvidenceExample]:
    cfg = cfg or EvidenceConfig()
    evidence: list[EvidenceExample] = []

    for period, texts, emb in (
        (early_period, early_texts, early_emb),
        (late_period, late_texts, late_emb),
    ):
        if not texts or emb.size == 0:
            continue
        idxs = _medoid_indices(emb, cfg.n_per_period)
        centroid = emb.mean(axis=0)
        for i in idxs:
            text = texts[i]
            if len(text) > cfg.max_chars:
                text = text[: cfg.max_chars - 3] + "..."
            score = float(
                1.0
                - (
                    np.linalg.norm(emb[i] - centroid)
                    / (np.linalg.norm(centroid) + 1e-9)
                )
            )
            evidence.append(
                EvidenceExample(period=period, text=text, sense_id=None, score=max(0.0, score))
            )
    return evidence


def evidence_consistency(
    early_emb: np.ndarray,
    late_emb: np.ndarray,
    evidence_early_idx: list[int] | None = None,
    evidence_late_idx: list[int] | None = None,
) -> float:
    """
    Consistency = agreement between selected evidence centroids and full period centroids.
    High when evidence is representative.
    """
    if early_emb.size == 0 or late_emb.size == 0:
        return 0.0

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    e_cent = early_emb.mean(axis=0)
    l_cent = late_emb.mean(axis=0)
    if evidence_early_idx:
        e_ev = early_emb[evidence_early_idx].mean(axis=0)
    else:
        idxs = _medoid_indices(early_emb, min(5, len(early_emb)))
        e_ev = early_emb[idxs].mean(axis=0)
    if evidence_late_idx:
        l_ev = late_emb[evidence_late_idx].mean(axis=0)
    else:
        idxs = _medoid_indices(late_emb, min(5, len(late_emb)))
        l_ev = late_emb[idxs].mean(axis=0)

    return float(0.5 * (cos(e_cent, e_ev) + cos(l_cent, l_ev)) + 0.5) / 1.5
    # Keep in ~[0,1]
