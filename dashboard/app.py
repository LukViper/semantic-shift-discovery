"""Semantic Shift Observatory — Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = ROOT / "data" / "reports" / "demo" / "shift_reports.json"
CANDIDATES = ROOT / "data" / "reports" / "demo" / "candidates.json"
METRICS = ROOT / "data" / "reports" / "demo" / "metrics_hint.json"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Semantic Shift Observatory", layout="wide")
    st.title("Semantic Shift Observatory")
    st.caption("ECOS — Evidence-Calibrated Open-Vocabulary Semantic Shift Detection")

    reports_path = st.sidebar.text_input("Reports JSON", str(DEFAULT_REPORTS))
    reports = load_json(Path(reports_path))
    if not reports:
        st.warning(
            "No reports found. Run: `python scripts/run_pipeline.py --config configs/demo.yaml`"
        )
        return

    corpus_name = "AI Research sample 2020–2025"
    st.subheader(f"Corpus: {corpus_name}")

    metrics = load_json(METRICS) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Discovered shifts", len(reports))
    c2.metric("P@5", f"{metrics.get('P@5', 0):.2f}" if metrics else "—")
    c3.metric("MAP", f"{metrics.get('MAP', 0):.2f}" if metrics else "—")
    c4.metric("FDR@10", f"{metrics.get('FDR@10', 0):.2f}" if metrics else "—")

    st.markdown("### Top Emerging Semantic Shifts")
    rows = []
    for i, r in enumerate(reports[:15], 1):
        controls = r.get("controls", [])
        passed = all(c.get("passed", False) for c in controls) if controls else False
        rows.append(
            {
                "#": i,
                "term": r["term"],
                "confidence": round(r.get("calibrated_confidence", 0), 3),
                "ECSS": round(r.get("raw_ecss", 0), 3),
                "CF robustness": round(r.get("counterfactual_robustness", 0), 3),
                "controls OK": "yes" if passed else "partial",
                "shift type": r.get("shift_type", ""),
            }
        )
    st.dataframe(rows, use_container_width=True)

    terms = [r["term"] for r in reports]
    selected = st.selectbox("Select term", terms)
    report = next(r for r in reports if r["term"] == selected)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Shift profile")
        st.write(
            {
                "time_window": report.get("time_window"),
                "shift_type": report.get("shift_type"),
                "original_profile": report.get("original_profile"),
                "emerging_profile": report.get("emerging_profile"),
                "contextual_shift": report.get("contextual_shift"),
                "sense_redistribution": report.get("sense_redistribution"),
                "neighborhood_shift": report.get("neighborhood_shift"),
                "changepoint": report.get("temporal_changepoint"),
            }
        )
        st.markdown("#### Robustness")
        for c in report.get("controls", []):
            flag = "PASS" if c.get("passed") else "FAIL"
            st.write(f"- **{c.get('name')}**: {flag} (score={c.get('score', 0):.3f})")
        st.write(f"- **Counterfactual**: {report.get('counterfactual_robustness', 0):.3f}")
        st.write(f"- **Evidence consistency**: {report.get('evidence_consistency', 0):.3f}")
        st.write(f"- **Calibrated confidence**: {report.get('calibrated_confidence', 0):.3f}")

    with right:
        st.markdown("#### Score breakdown")
        chart_data = {
            "contextual": report.get("contextual_shift", 0),
            "sense": report.get("sense_redistribution", 0),
            "neighborhood": report.get("neighborhood_shift", 0),
            "ECSS": report.get("raw_ecss", 0),
            "confidence": report.get("calibrated_confidence", 0),
            "CF": report.get("counterfactual_robustness", 0),
        }
        st.bar_chart(chart_data)

        st.markdown("#### Representative evidence")
        for ev in report.get("evidence", []):
            st.markdown(f"**{ev.get('period')}** — {ev.get('text')}")

    cands = load_json(CANDIDATES)
    if cands:
        with st.expander("Candidate ranking (cheap filter)"):
            st.dataframe(cands[:30], use_container_width=True)

    # Paper figures gallery (if regenerated)
    fig_dir = ROOT / "paper" / "figures"
    gallery = [
        ("Architecture", "fig_architecture.png"),
        ("SEM-STREAM ranking", "fig_semstream_ranking.png"),
        ("FDR resistance", "fig_fdr_resistance.png"),
        ("Ablations", "fig_ablation.png"),
        ("Discovery precision", "fig_discovery_precision.png"),
        ("Signal breakdown", "fig_signal_breakdown.png"),
    ]
    available = [(title, fig_dir / name) for title, name in gallery if (fig_dir / name).exists()]
    if available:
        st.markdown("### Experiment figures")
        tabs = st.tabs([t for t, _ in available])
        for tab, (title, path) in zip(tabs, available):
            with tab:
                st.image(str(path), caption=title, use_container_width=True)

    with st.expander("System architecture (Mermaid)"):
        st.markdown(
            """
```mermaid
flowchart TD
  corpus[LongitudinalCorpus] --> prep[Preprocess]
  prep --> cand[CandidateDiscovery]
  cand --> embed[ContextualEmbeddings]
  embed --> sense[SenseInduction]
  sense --> shift[ShiftEngine]
  shift --> controls[ConfounderControls]
  controls --> cf[CounterfactualMatch]
  cf --> evidence[EvidenceRetrieval]
  evidence --> calib[Calibration]
  calib --> report[ShiftReport]
```
"""
        )


if __name__ == "__main__":
    main()
