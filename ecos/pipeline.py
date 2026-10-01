"""End-to-end ECOS discovery pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ecos.calibration.calibrator import Calibrator, compute_ecss
from ecos.candidates.discovery import CandidateConfig, collect_contexts, discover_candidates
from ecos.config import load_config
from ecos.controls.confounders import ControlThresholds, evaluate_controls
from ecos.counterfactual.matching import CounterfactualConfig, counterfactual_shift
from ecos.data.loaders import load_json_corpus, load_sample_corpus, synthetic_discovery_labels
from ecos.evidence.retrieval import EvidenceConfig, evidence_consistency, retrieve_evidence
from ecos.io import save_json, save_reports
from ecos.logging_utils import setup_logging
from ecos.preprocess.pipeline import PreprocessConfig, preprocess_corpus
from ecos.representations.encoder import ContextualEncoder, EncoderConfig
from ecos.reporting.report import build_report, report_to_markdown
from ecos.reporting.events import associate_change_point
from ecos.schema import ShiftReport, TemporalCorpus
from ecos.sense.induction import describe_senses, induce_senses
from ecos.shift.engine import ShiftWeights, combine_signals, compute_shift_signals


@dataclass
class PipelineResult:
    candidates: list[tuple[str, float, dict[str, float]]]
    reports: list[ShiftReport]
    metrics_hint: dict[str, Any]


def _encode_period_contexts(
    corpus: TemporalCorpus,
    term: str,
    period: str,
    encoder: ContextualEncoder,
    max_contexts: int,
) -> tuple[list[str], np.ndarray, list[dict]]:
    ctxs = collect_contexts(corpus, term, period, max_contexts=max_contexts)
    texts = [c["full_text"] for c in ctxs] or [c["text"] for c in ctxs]
    if not texts:
        return [], np.zeros((0, 1)), []
    emb = encoder.encode(texts)
    return texts, emb, ctxs


def run_ecos_pipeline(cfg: dict[str, Any]) -> PipelineResult:
    logger = setup_logging(cfg.get("project", {}).get("log_level", "INFO"))
    seed = int(cfg.get("project", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    paths = cfg.get("paths", {})
    corpus_path = paths.get("corpus")
    if corpus_path and Path(corpus_path).exists():
        corpus = load_json_corpus(
            corpus_path,
            name=cfg.get("corpus", {}).get("name", "corpus"),
            window=cfg.get("corpus", {}).get("window", "year"),
        )
    else:
        logger.info("Using bundled sample corpus")
        corpus = load_sample_corpus()

    prep_cfg = PreprocessConfig(min_token_freq=int(cfg.get("corpus", {}).get("min_token_freq", 2)))
    corpus, _ = preprocess_corpus(
        corpus,
        prep_cfg,
        languages=cfg.get("corpus", {}).get("languages", ["en"]),
    )

    early = str(cfg.get("pipeline", {}).get("early_period", "2020"))
    late = str(cfg.get("pipeline", {}).get("late_period", "2025"))
    # Auto-pick periods if missing
    periods = corpus.periods()
    if early not in periods and periods:
        early = periods[0]
    if late not in periods and periods:
        late = periods[-1]

    cand_cfg = CandidateConfig(
        top_k=int(cfg.get("candidates", {}).get("top_k", 50)),
        alpha=float(cfg.get("candidates", {}).get("alpha", 0.25)),
        beta=float(cfg.get("candidates", {}).get("beta", 0.35)),
        gamma=float(cfg.get("candidates", {}).get("gamma", 0.20)),
        delta=float(cfg.get("candidates", {}).get("delta", 0.20)),
        min_freq=int(cfg.get("candidates", {}).get("min_freq", 3)),
    )
    candidates = discover_candidates(corpus, early, late, cand_cfg, prep_cfg)
    force_terms = cfg.get("pipeline", {}).get("force_terms") or []
    if force_terms:
        have = {t for t, _, _ in candidates}
        for term in force_terms:
            t = str(term).lower()
            if t not in have:
                candidates.append((t, 0.0, {"F": 0.0, "D": 0.0, "V": 0.0, "T": 0.0}))
                have.add(t)
    logger.info("Discovered %d candidates", len(candidates))

    enc_cfg = EncoderConfig(
        backend=str(cfg.get("representations", {}).get("backend", "bag_of_words")),
        model_name=str(cfg.get("representations", {}).get("model_name", "sentence-transformers/all-MiniLM-L6-v2")),
    )
    encoder = ContextualEncoder(enc_cfg)
    # Fit on all candidate contexts
    all_texts: list[str] = []
    for term, _, _ in candidates:
        for p in (early, late):
            all_texts.extend(
                c["full_text"]
                for c in collect_contexts(
                    corpus,
                    term,
                    p,
                    max_contexts=int(cfg.get("representations", {}).get("max_contexts_per_term", 40)),
                )
            )
    if all_texts:
        encoder.fit(all_texts)

    sw = cfg.get("shift", {}).get("weights", {})
    weights = ShiftWeights(
        context=float(sw.get("context", 0.35)),
        sense=float(sw.get("sense", 0.30)),
        neighborhood=float(sw.get("neighborhood", 0.20)),
        temporal=float(sw.get("temporal", 0.15)),
    )
    thr = ControlThresholds(
        topic=float(cfg.get("controls", {}).get("topic_threshold", 0.35)),
        source=float(cfg.get("controls", {}).get("source_threshold", 0.35)),
        frequency=float(cfg.get("controls", {}).get("frequency_threshold", 0.50)),
    )

    slices = corpus.slices()
    early_docs = slices.get(early, [])
    late_docs = slices.get(late, [])
    max_ctx = int(cfg.get("representations", {}).get("max_contexts_per_term", 40))

    raw_rows: list[dict[str, Any]] = []
    for term, cand_score, parts in candidates:
        e_texts, e_emb, _ = _encode_period_contexts(corpus, term, early, encoder, max_ctx)
        l_texts, l_emb, _ = _encode_period_contexts(corpus, term, late, encoder, max_ctx)
        if e_emb.size == 0 and l_emb.size == 0:
            continue
        # If one side is missing, use a tiny zero-vector placeholder so forced
        # candidates still receive a (low) score instead of disappearing.
        if e_emb.size == 0:
            e_texts = [""]
            e_emb = np.zeros((1, l_emb.shape[1]), dtype=np.float64)
        if l_emb.size == 0:
            l_texts = [""]
            l_emb = np.zeros((1, e_emb.shape[1]), dtype=np.float64)

        period_embs: dict[str, np.ndarray] = {}
        for p in periods:
            _, emb, _ = _encode_period_contexts(corpus, term, p, encoder, max_ctx)
            if emb.size:
                period_embs[p] = emb

        signals = compute_shift_signals(term, e_emb, l_emb, e_texts, l_texts, period_embs)
        observed = combine_signals(signals, weights)

        controls = evaluate_controls(term, early_docs, late_docs, thr, observed_shift=observed)

        def score_fn(edocs, ldocs):
            # Proxy: re-encode contexts from matched late doc texts containing term
            from ecos.schema import TemporalCorpus as TC

            tmp = TC(name="tmp", documents=edocs + ldocs, window=corpus.window)
            et, ee, _ = _encode_period_contexts(tmp, term, early, encoder, max_ctx)
            # For matched late docs, collect from those docs only
            late_only = TC(name="late", documents=ldocs, window=corpus.window)
            # Use a synthetic period key by temporarily tagging — fall back to all late texts
            lt = [d.text for d in ldocs if term.lower() in d.text.lower()]
            if not lt:
                lt = [d.text for d in ldocs]
            if not et or not lt or ee.size == 0:
                return 0.0
            le = encoder.encode(lt)
            return float(
                combine_signals(
                    compute_shift_signals(term, ee, le, et, lt, None),
                    weights,
                )
            )

        cf = {"robustness": 0.0, "counterfactual": observed, "survival_ratio": 1.0}
        if cfg.get("counterfactual", {}).get("enabled", True):
            cf = counterfactual_shift(
                early_docs,
                late_docs,
                observed,
                score_fn,
                CounterfactualConfig(
                    matching=str(cfg.get("counterfactual", {}).get("matching", "reweight")),
                    n_bins=int(cfg.get("counterfactual", {}).get("n_bins", 5)),
                ),
                rng=rng,
            )

        evid = retrieve_evidence(
            e_texts,
            l_texts,
            e_emb,
            l_emb,
            early,
            late,
            EvidenceConfig(n_per_period=3),
        )
        econs = evidence_consistency(e_emb, l_emb)

        ctrl_map = {c.name: c.score for c in controls}
        raw_ecss = compute_ecss(
            signals.contextual,
            signals.sense,
            signals.temporal,
            signals.neighborhood,
            econs,
            ctrl_map.get("topic", 0.0),
            ctrl_map.get("source", 0.0),
            ctrl_map.get("frequency", 0.0),
        )
        # Blend survival into ECSS
        raw_ecss = float(
            max(0.0, min(1.0, 0.7 * raw_ecss + 0.3 * cf["robustness"] * cf.get("survival_ratio", 1.0)))
        )

        e_model = induce_senses(e_emb)
        l_model = induce_senses(l_emb)
        orig = describe_senses(e_texts, e_model.labels)
        emerg = describe_senses(l_texts, l_model.labels)
        assoc = associate_change_point(corpus.documents, signals.changepoint, corpus.window)

        raw_rows.append(
            {
                "term": term,
                "cand_score": cand_score,
                "parts": parts,
                "signals": signals,
                "observed": observed,
                "controls": controls,
                "cf": cf,
                "evid": evid,
                "econs": econs,
                "raw_ecss": raw_ecss,
                "orig": "; ".join(f"{k}:{v}" for k, v in orig.items()) or "n/a",
                "emerg": "; ".join(f"{k}:{v}" for k, v in emerg.items()) or "n/a",
                "association": {
                    "period": assoc.period,
                    "top_topics": assoc.top_topics,
                    "top_communities": assoc.top_communities,
                    "note": assoc.note,
                }
                if assoc
                else None,
            }
        )

    # Calibration on synthetic labels when available
    labels = synthetic_discovery_labels()
    scores = np.array([r["raw_ecss"] for r in raw_rows], dtype=np.float64)
    y = np.array([labels.get(r["term"], 0) for r in raw_rows], dtype=np.int32)
    calibrator = Calibrator(method=str(cfg.get("calibration", {}).get("method", "platt")))
    if cfg.get("calibration", {}).get("enabled", True) and len(scores) >= 4 and len(np.unique(y)) >= 2:
        calibrator.fit(scores, y)
        calibrated = np.asarray(calibrator.transform(scores), dtype=np.float64)
        # Blend with raw ECSS to avoid over-shrinkage on tiny label sets
        calibrated = 0.55 * calibrated + 0.45 * scores
    else:
        calibrated = scores
    calibrated = np.clip(calibrated, 0.0, 1.0)

    reports: list[ShiftReport] = []
    for row, conf in zip(raw_rows, calibrated):
        sig = row["signals"]
        report = build_report(
            term=row["term"],
            early_period=early,
            late_period=late,
            shift_type=sig.shift_type,
            original_profile=row["orig"],
            emerging_profile=row["emerg"],
            contextual_shift=sig.contextual,
            sense_redistribution=sig.sense,
            neighborhood_shift=sig.neighborhood,
            temporal_changepoint=sig.changepoint,
            controls=row["controls"],
            counterfactual_robustness=float(row["cf"]["robustness"]),
            evidence_consistency=float(row["econs"]),
            calibrated_confidence=float(conf),
            raw_ecss=float(row["raw_ecss"]),
            evidence=row["evid"],
            metadata={
                "candidate_score": row["cand_score"],
                "candidate_parts": row["parts"],
                "cf": row["cf"],
                "observed_shift": row["observed"],
                "association": row.get("association"),
            },
        )
        reports.append(report)

    reports.sort(key=lambda r: r.calibrated_confidence, reverse=True)
    top_n = int(cfg.get("pipeline", {}).get("top_reports", 20))
    reports = reports[:top_n]

    out_dir = Path(paths.get("output_dir", "data/reports/demo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_path = Path(paths.get("reports", out_dir / "shift_reports.json"))
    save_reports(reports, reports_path)
    save_json(
        [{"term": t, "score": s, "parts": p} for t, s, p in candidates],
        out_dir / "candidates.json",
    )
    md_path = out_dir / "reports.md"
    md_path.write_text(
        "\n\n---\n\n".join(report_to_markdown(r) for r in reports),
        encoding="utf-8",
    )
    logger.info("Wrote %d reports to %s", len(reports), reports_path)

    from ecos.eval.metrics import ranking_metrics

    ranked = [r.term for r in reports]
    metrics_hint = ranking_metrics(labels, ranked, k_values=[5, 10])
    save_json(metrics_hint, out_dir / "metrics_hint.json")

    return PipelineResult(candidates=candidates, reports=reports, metrics_hint=metrics_hint)


def run_from_config_path(path: str | Path) -> PipelineResult:
    return run_ecos_pipeline(load_config(path))
