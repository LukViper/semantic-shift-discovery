"""Unit tests for ECOS core modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def sample_corpus():
    from ecos.data.loaders import load_sample_corpus

    sample = ROOT / "data" / "sample" / "arxiv_sample.json"
    if not sample.exists():
        from importlib.util import module_from_spec, spec_from_file_location

        builder = ROOT / "data" / "sample" / "build_sample.py"
        spec = spec_from_file_location("build_sample", builder)
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.write_sample(sample)
    return load_sample_corpus(sample)


def test_schema_roundtrip():
    from datetime import datetime

    from ecos.schema import ControlResult, Document, EvidenceExample, ShiftReport

    doc = Document("1", datetime(2020, 1, 1), "hello agent world", topic="ai")
    assert Document.from_dict(doc.to_dict()).text == doc.text

    report = ShiftReport(
        term="agent",
        time_window="2020–2025",
        shift_type="emerging sense",
        original_profile="software",
        emerging_profile="ai agent",
        contextual_shift=0.5,
        sense_redistribution=0.4,
        neighborhood_shift=0.3,
        temporal_changepoint="2025",
        controls=[ControlResult("topic", True, 0.1)],
        evidence=[EvidenceExample("2020", "a software agent")],
    )
    assert ShiftReport.from_dict(report.to_dict()).term == "agent"


def test_candidate_discovery(sample_corpus):
    from ecos.candidates.discovery import discover_candidates
    from ecos.preprocess.pipeline import PreprocessConfig

    ranked = discover_candidates(
        sample_corpus,
        "2020",
        "2025",
        prep_cfg=PreprocessConfig(min_token_freq=1),
    )
    assert ranked
    terms = [t for t, _, _ in ranked]
    assert "agent" in terms


def test_metrics():
    from ecos.eval.metrics import ranking_metrics, spearman_correlation

    y_true = {"a": 1, "b": 1, "c": 0, "d": 0}
    ranked = ["a", "c", "b", "d"]
    m = ranking_metrics(y_true, ranked, k_values=[2])
    assert 0 <= m["P@2"] <= 1
    assert spearman_correlation([1, 0, 1], [0.9, 0.1, 0.8]) > 0


def test_calibrator():
    from ecos.calibration.calibrator import Calibrator, compute_ecss, expected_calibration_error

    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    cal = Calibrator(method="platt").fit(scores, labels)
    out = cal.transform(scores)
    assert len(out) == 4
    ece = expected_calibration_error(out, labels.astype(float))
    assert ece >= 0
    assert 0 <= compute_ecss(0.5, 0.5, 0.2, 0.2, 0.5, 0.1, 0.1, 0.1) <= 1


def test_controls_and_counterfactual(sample_corpus):
    from ecos.controls.confounders import evaluate_controls
    from ecos.counterfactual.matching import counterfactual_shift

    slices = sample_corpus.slices()
    controls = evaluate_controls("agent", slices["2020"], slices["2025"], observed_shift=0.7)
    assert {c.name for c in controls} >= {"topic", "source", "frequency"}

    def score_fn(e, l):
        return 0.6 if l else 0.0

    cf = counterfactual_shift(slices["2020"], slices["2025"], 0.7, score_fn)
    assert "robustness" in cf


def test_baselines(sample_corpus):
    from ecos.eval.baselines import contextual_aggregate_shift, static_embedding_shift

    s1 = static_embedding_shift(sample_corpus, "agent", "2020", "2025")
    s2 = contextual_aggregate_shift(sample_corpus, "agent", "2020", "2025")
    assert s1 >= 0 and s2 >= 0


def test_pipeline_smoke():
    from ecos.pipeline import run_from_config_path

    sample = ROOT / "data" / "sample" / "arxiv_sample.json"
    if not sample.exists():
        from importlib.util import module_from_spec, spec_from_file_location

        builder = ROOT / "data" / "sample" / "build_sample.py"
        spec = spec_from_file_location("build_sample", builder)
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.write_sample(sample)

    result = run_from_config_path(ROOT / "configs" / "demo.yaml")
    assert result.reports
    assert result.candidates
    top_terms = {r.term for r in result.reports[:10]}
    # agent should surface among candidates/reports for this narrative corpus
    all_terms = {r.term for r in result.reports} | {t for t, _, _ in result.candidates}
    assert "agent" in all_terms
