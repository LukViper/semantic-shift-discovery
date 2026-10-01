"""Confidence calibration for ECSS scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


Method = Literal["platt", "isotonic", "temperature", "identity"]


@dataclass
class Calibrator:
    method: Method = "platt"
    temperature: float = 1.0
    _platt: LogisticRegression | None = field(default=None, repr=False)
    _isotonic: IsotonicRegression | None = field(default=None, repr=False)
    fitted: bool = False

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)
        if len(scores) < 2 or len(np.unique(labels)) < 2:
            self.method = "identity"
            self.fitted = True
            return self

        if self.method == "platt":
            self._platt = LogisticRegression(max_iter=1000)
            self._platt.fit(scores.reshape(-1, 1), labels)
        elif self.method == "isotonic":
            self._isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._isotonic.fit(scores, labels)
        elif self.method == "temperature":
            # Fit temperature by minimizing NLL on logits = score as pseudo-logit
            self.temperature = _fit_temperature(scores, labels)
        self.fitted = True
        return self

    def transform(self, scores: np.ndarray | float) -> np.ndarray | float:
        single = isinstance(scores, (float, int))
        arr = np.asarray([scores] if single else scores, dtype=np.float64).reshape(-1)
        if not self.fitted or self.method == "identity":
            out = np.clip(arr, 0.0, 1.0)
        elif self.method == "platt" and self._platt is not None:
            out = self._platt.predict_proba(arr.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic" and self._isotonic is not None:
            out = self._isotonic.predict(arr)
        elif self.method == "temperature":
            out = 1.0 / (1.0 + np.exp(-arr / max(self.temperature, 1e-3)))
        else:
            out = np.clip(arr, 0.0, 1.0)
        return float(out[0]) if single else out


def _fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 20):
        p = 1.0 / (1.0 + np.exp(-scores / t))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        nll = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def expected_calibration_error(
    confidences: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences) or 1
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1] if i < n_bins - 1 else confidences <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = labels[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def brier_score(confidences: np.ndarray, labels: np.ndarray) -> float:
    confidences = np.asarray(confidences, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    return float(np.mean((confidences - labels) ** 2))


def compute_ecss(
    s_context: float,
    s_sense: float,
    s_temporal: float,
    s_neighborhood: float,
    s_evidence: float,
    c_topic: float,
    c_source: float,
    c_frequency: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Evidence-Calibrated Semantic Shift Score (raw)."""
    w = {
        "w1": 0.30,
        "w2": 0.25,
        "w3": 0.15,
        "w4": 0.15,
        "w5": 0.15,
        "w6": 0.20,
        "w7": 0.15,
        "w8": 0.15,
    }
    if weights:
        w.update(weights)
    raw = (
        w["w1"] * s_context
        + w["w2"] * s_sense
        + w["w3"] * s_temporal
        + w["w4"] * s_neighborhood
        + w["w5"] * s_evidence
        - w["w6"] * c_topic
        - w["w7"] * c_source
        - w["w8"] * c_frequency
    )
    return float(max(0.0, min(1.0, raw)))
