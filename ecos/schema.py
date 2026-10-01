"""Core data schemas for longitudinal corpora and shift reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Document:
    """A single document in a longitudinal corpus."""

    document_id: str
    timestamp: datetime
    text: str
    source: str = "unknown"
    domain: str = "unknown"
    language: str = "en"
    topic: str | None = None
    community: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        ts = data["timestamp"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return cls(
            document_id=str(data["document_id"]),
            timestamp=ts,
            text=str(data["text"]),
            source=str(data.get("source", "unknown")),
            domain=str(data.get("domain", "unknown")),
            language=str(data.get("language", "en")),
            topic=data.get("topic"),
            community=data.get("community"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class TemporalCorpus:
    """Documents grouped into chronological windows."""

    name: str
    documents: list[Document]
    window: str = "year"  # year | quarter | month

    def slices(self) -> dict[str, list[Document]]:
        """Group documents by temporal window key."""
        groups: dict[str, list[Document]] = {}
        for doc in self.documents:
            if self.window == "year":
                key = str(doc.timestamp.year)
            elif self.window == "quarter":
                q = (doc.timestamp.month - 1) // 3 + 1
                key = f"{doc.timestamp.year}-Q{q}"
            else:
                key = f"{doc.timestamp.year}-{doc.timestamp.month:02d}"
            groups.setdefault(key, []).append(doc)
        return dict(sorted(groups.items()))

    def periods(self) -> list[str]:
        return list(self.slices().keys())


@dataclass
class ControlResult:
    name: str
    passed: bool
    score: float
    detail: str = ""


@dataclass
class EvidenceExample:
    period: str
    text: str
    sense_id: int | None = None
    score: float = 0.0


@dataclass
class ShiftReport:
    """Explainable report for one discovered semantic shift."""

    term: str
    time_window: str
    shift_type: str
    original_profile: str
    emerging_profile: str
    contextual_shift: float
    sense_redistribution: float
    neighborhood_shift: float
    temporal_changepoint: str | None
    controls: list[ControlResult] = field(default_factory=list)
    counterfactual_robustness: float = 0.0
    evidence_consistency: float = 0.0
    calibrated_confidence: float = 0.0
    raw_ecss: float = 0.0
    associated_period: str | None = None
    evidence: list[EvidenceExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def controls_summary(self) -> dict[str, bool]:
        return {c.name: c.passed for c in self.controls}

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "time_window": self.time_window,
            "shift_type": self.shift_type,
            "original_profile": self.original_profile,
            "emerging_profile": self.emerging_profile,
            "contextual_shift": self.contextual_shift,
            "sense_redistribution": self.sense_redistribution,
            "neighborhood_shift": self.neighborhood_shift,
            "temporal_changepoint": self.temporal_changepoint,
            "controls": [asdict(c) for c in self.controls],
            "counterfactual_robustness": self.counterfactual_robustness,
            "evidence_consistency": self.evidence_consistency,
            "calibrated_confidence": self.calibrated_confidence,
            "raw_ecss": self.raw_ecss,
            "associated_period": self.associated_period,
            "evidence": [asdict(e) for e in self.evidence],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShiftReport:
        controls = [ControlResult(**c) for c in data.get("controls", [])]
        evidence = [EvidenceExample(**e) for e in data.get("evidence", [])]
        return cls(
            term=data["term"],
            time_window=data.get("time_window", ""),
            shift_type=data.get("shift_type", "unknown"),
            original_profile=data.get("original_profile", ""),
            emerging_profile=data.get("emerging_profile", ""),
            contextual_shift=float(data.get("contextual_shift", 0.0)),
            sense_redistribution=float(data.get("sense_redistribution", 0.0)),
            neighborhood_shift=float(data.get("neighborhood_shift", 0.0)),
            temporal_changepoint=data.get("temporal_changepoint"),
            controls=controls,
            counterfactual_robustness=float(data.get("counterfactual_robustness", 0.0)),
            evidence_consistency=float(data.get("evidence_consistency", 0.0)),
            calibrated_confidence=float(data.get("calibrated_confidence", 0.0)),
            raw_ecss=float(data.get("raw_ecss", 0.0)),
            associated_period=data.get("associated_period"),
            evidence=evidence,
            metadata=dict(data.get("metadata") or {}),
        )
