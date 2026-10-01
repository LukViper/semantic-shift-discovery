"""Shift report generation and markdown rendering."""

from __future__ import annotations

from ecos.schema import ControlResult, EvidenceExample, ShiftReport


def build_report(
    term: str,
    early_period: str,
    late_period: str,
    shift_type: str,
    original_profile: str,
    emerging_profile: str,
    contextual_shift: float,
    sense_redistribution: float,
    neighborhood_shift: float,
    temporal_changepoint: str | None,
    controls: list[ControlResult],
    counterfactual_robustness: float,
    evidence_consistency: float,
    calibrated_confidence: float,
    raw_ecss: float,
    evidence: list[EvidenceExample],
    metadata: dict | None = None,
) -> ShiftReport:
    return ShiftReport(
        term=term,
        time_window=f"{early_period}–{late_period}",
        shift_type=shift_type,
        original_profile=original_profile,
        emerging_profile=emerging_profile,
        contextual_shift=contextual_shift,
        sense_redistribution=sense_redistribution,
        neighborhood_shift=neighborhood_shift,
        temporal_changepoint=temporal_changepoint,
        controls=controls,
        counterfactual_robustness=counterfactual_robustness,
        evidence_consistency=evidence_consistency,
        calibrated_confidence=calibrated_confidence,
        raw_ecss=raw_ecss,
        associated_period=temporal_changepoint,
        evidence=evidence,
        metadata=metadata or {},
    )


def report_to_markdown(report: ShiftReport) -> str:
    lines = [
        "SEMANTIC SHIFT REPORT",
        "",
        f"Term: {report.term}",
        f"Time window: {report.time_window}",
        f"Shift type: {report.shift_type}",
        "",
        f"Original semantic profile: {report.original_profile}",
        f"Emerging semantic profile: {report.emerging_profile}",
        "",
        f"Contextual shift: {report.contextual_shift:.2f}",
        f"Sense redistribution: {report.sense_redistribution:.2f}",
        f"Neighborhood shift: {report.neighborhood_shift:.2f}",
        "",
    ]
    for c in report.controls:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"{c.name.capitalize()}-control: {status} ({c.score:.2f})")
    lines.extend(
        [
            "",
            f"Counterfactual robustness: {report.counterfactual_robustness:.2f}",
            f"Evidence consistency: {report.evidence_consistency:.2f}",
            f"Calibrated confidence: {report.calibrated_confidence:.2f}",
            f"Raw ECSS: {report.raw_ecss:.2f}",
            f"Associated period: {report.associated_period or 'n/a'}",
            "",
            "Representative evidence:",
        ]
    )
    for i, ev in enumerate(report.evidence, 1):
        lines.append(f"[{i}] ({ev.period}) {ev.text}")
    return "\n".join(lines)
