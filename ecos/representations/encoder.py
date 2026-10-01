"""Contextual representation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer


@dataclass
class EncoderConfig:
    backend: str = "bag_of_words"  # bag_of_words | transformers
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_features: int = 4096
    device: str = "cpu"


class ContextualEncoder:
    """Encode usage contexts into dense vectors."""

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        self._st_model = None
        self._hasher = HashingVectorizer(
            n_features=self.cfg.max_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self._tfidf = TfidfTransformer()
        self._fitted = False

    def fit(self, texts: Sequence[str]) -> ContextualEncoder:
        if self.cfg.backend == "bag_of_words":
            X = self._hasher.transform(list(texts))
            self._tfidf.fit(X)
            self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1), dtype=np.float64)
        if self.cfg.backend == "transformers":
            return self._encode_transformers(texts)
        return self._encode_bow(texts)

    def _encode_bow(self, texts: list[str]) -> np.ndarray:
        X = self._hasher.transform(texts)
        if not self._fitted:
            self._tfidf.fit(X)
            self._fitted = True
        dense = self._tfidf.transform(X).toarray().astype(np.float64)
        # L2 normalize
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return dense / norms

    def _encode_transformers(self, texts: list[str]) -> np.ndarray:
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Install sentence-transformers for transformers backend: pip install ecos[nlp]"
                ) from exc
            self._st_model = SentenceTransformer(self.cfg.model_name, device=self.cfg.device)
        emb = self._st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float64)


def mean_embedding(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return np.zeros(1, dtype=np.float64)
    return vectors.mean(axis=0)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return float(1.0 - max(min(sim, 1.0), -1.0))
