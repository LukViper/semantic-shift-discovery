"""Temporal sense induction via clustering of contextual embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class SenseConfig:
    method: str = "kmeans"  # kmeans | hdbscan
    n_clusters: int = 3
    min_cluster_size: int = 2
    random_state: int = 42


@dataclass
class SenseModel:
    labels: np.ndarray
    centers: np.ndarray | None
    n_senses: int
    method: str


def induce_senses(embeddings: np.ndarray, cfg: SenseConfig | None = None) -> SenseModel:
    cfg = cfg or SenseConfig()
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return SenseModel(labels=np.array([], dtype=int), centers=None, n_senses=0, method=cfg.method)

    n = embeddings.shape[0]
    if cfg.method == "hdbscan":
        return _hdbscan_senses(embeddings, cfg)

    # Cap k by unique rows to avoid sklearn ConvergenceWarning on duplicates
    uniq = np.unique(np.round(embeddings, 6), axis=0).shape[0]
    k = max(1, min(cfg.n_clusters, n, uniq))
    if k == 1:
        labels = np.zeros(n, dtype=int)
        centers = embeddings.mean(axis=0, keepdims=True)
        return SenseModel(labels=labels, centers=centers, n_senses=1, method="kmeans")

    km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
    labels = km.fit_predict(embeddings)
    return SenseModel(
        labels=labels.astype(int),
        centers=km.cluster_centers_,
        n_senses=k,
        method="kmeans",
    )


def _hdbscan_senses(embeddings: np.ndarray, cfg: SenseConfig) -> SenseModel:
    try:
        import hdbscan
    except ImportError:
        # Fallback
        return induce_senses(embeddings, SenseConfig(method="kmeans", n_clusters=cfg.n_clusters))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, cfg.min_cluster_size))
    labels = clusterer.fit_predict(embeddings)
    # Map noise (-1) to its own bucket or nearest
    unique = sorted(set(int(x) for x in labels if x >= 0))
    if not unique:
        labels = np.zeros(len(labels), dtype=int)
        unique = [0]
    remap = {old: i for i, old in enumerate(unique)}
    mapped = np.array([remap.get(int(x), len(unique)) for x in labels], dtype=int)
    n_senses = len(set(mapped.tolist()))
    centers = []
    for s in range(n_senses):
        mask = mapped == s
        if mask.any():
            centers.append(embeddings[mask].mean(axis=0))
    return SenseModel(
        labels=mapped,
        centers=np.vstack(centers) if centers else None,
        n_senses=n_senses,
        method="hdbscan",
    )


def sense_distribution(labels: np.ndarray, n_senses: int | None = None) -> np.ndarray:
    if labels.size == 0:
        return np.zeros(1, dtype=np.float64)
    if n_senses is None:
        n_senses = int(labels.max()) + 1
    counts = np.bincount(labels.astype(int), minlength=n_senses).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return np.zeros_like(counts)
    return counts / total


def align_sense_distributions(
    early_labels: np.ndarray,
    late_labels: np.ndarray,
    early_emb: np.ndarray,
    late_emb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align late senses to early senses by nearest center matching.
    Returns aligned early and late probability distributions over a shared sense space.
    """
    early_model = induce_senses(early_emb, SenseConfig(n_clusters=max(1, min(3, len(early_emb)))))
    # Use early centers; assign late points to nearest early center, plus residual cluster if needed
    if early_model.centers is None or early_model.n_senses == 0:
        return np.array([1.0]), np.array([1.0])

    centers = early_model.centers
    early_dist = sense_distribution(early_model.labels, early_model.n_senses)

    # Assign late embeddings to early centers
    if late_emb.size == 0:
        late_dist = np.zeros_like(early_dist)
        return early_dist, late_dist

    dists = np.linalg.norm(late_emb[:, None, :] - centers[None, :, :], axis=2)
    assigned = dists.argmin(axis=1)
    late_dist = sense_distribution(assigned, early_model.n_senses)
    return early_dist, late_dist


def describe_senses(
    contexts: list[str],
    labels: np.ndarray,
    top_n: int = 5,
) -> dict[int, str]:
    """Heuristic sense labels from frequent tokens in each cluster."""
    from collections import Counter

    from ecos.preprocess.pipeline import tokenize

    profiles: dict[int, str] = {}
    for sid in sorted(set(int(x) for x in labels)):
        toks: Counter[str] = Counter()
        for ctx, lab in zip(contexts, labels):
            if int(lab) != sid:
                continue
            toks.update(tokenize(ctx))
        top = [w for w, _ in toks.most_common(top_n)]
        profiles[sid] = ", ".join(top) if top else f"sense-{sid}"
    return profiles


def classify_shift_type(early_dist: np.ndarray, late_dist: np.ndarray) -> str:
    """Classify sense dynamics: expansion, contraction, redistribution, stable."""
    if early_dist.size == 0 or late_dist.size == 0:
        return "unknown"
    # Pad
    n = max(len(early_dist), len(late_dist))
    e = np.zeros(n)
    l = np.zeros(n)
    e[: len(early_dist)] = early_dist
    l[: len(late_dist)] = late_dist
    js = 0.5 * np.sum(e * np.log((e + 1e-12) / ((e + l) / 2 + 1e-12))) + 0.5 * np.sum(
        l * np.log((l + 1e-12) / ((e + l) / 2 + 1e-12))
    )
    early_entropy = -np.sum(e[e > 0] * np.log(e[e > 0]))
    late_entropy = -np.sum(l[l > 0] * np.log(l[l > 0]))
    if js < 0.05:
        return "stable"
    if late_entropy > early_entropy + 0.15:
        return "sense expansion"
    if early_entropy > late_entropy + 0.15:
        return "sense contraction"
    # New mass on previously tiny sense
    if np.any((e < 0.1) & (l > 0.3)):
        return "emerging sense"
    return "sense redistribution"
