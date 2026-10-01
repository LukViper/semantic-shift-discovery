"""Experiment runners: LSC, discovery, FDR resistance, calibration, ablations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ecos.calibration.calibrator import Calibrator, brier_score, expected_calibration_error
from ecos.candidates.discovery import discover_candidates
from ecos.config import load_config
from ecos.data.loaders import (
    load_prepared_discovery_benchmark,
    load_sample_corpus,
    synthetic_discovery_labels,
    try_load_semeval_bundle,
)
from ecos.eval.baselines import score_vocabulary
from ecos.eval.metrics import ranking_metrics, spearman_correlation
from ecos.io import save_json
from ecos.logging_utils import setup_logging
from ecos.pipeline import run_ecos_pipeline
from ecos.preprocess.pipeline import PreprocessConfig
from ecos.schema import Document, TemporalCorpus


def _demo_cfg(base: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {
        "project": {"name": "ecos-exp", "seed": 42, "log_level": "INFO"},
        "paths": {
            "corpus": "data/sample/arxiv_sample.json",
            "output_dir": "data/reports/experiments",
            "reports": "data/reports/experiments/shift_reports.json",
        },
        "corpus": {"name": "arxiv-cs-sample", "window": "year", "languages": ["en"], "min_token_freq": 2},
        "candidates": {"top_k": 40, "min_freq": 2, "alpha": 0.25, "beta": 0.35, "gamma": 0.2, "delta": 0.2},
        "representations": {"backend": "bag_of_words", "max_contexts_per_term": 40},
        "shift": {"weights": {"context": 0.35, "sense": 0.3, "neighborhood": 0.2, "temporal": 0.15}},
        "controls": {"topic_threshold": 0.35, "source_threshold": 0.35, "frequency_threshold": 0.5},
        "counterfactual": {"enabled": True, "matching": "reweight", "n_bins": 5},
        "calibration": {"method": "platt", "enabled": True},
        "pipeline": {"early_period": "2020", "late_period": "2025", "top_reports": 30},
    }
    if base:
        # shallow merge top-level
        for k, v in base.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def experiment_discovery_benchmarks(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """
    Open-vocabulary discovery evaluation on prepared SemEval-EN / LiverpoolFC
    annotation-context corpora (EMNLP 2025 discovery labels).
    """
    results: dict[str, Any] = {}
    datasets = cfg.get("discovery_datasets", ["liverpoolfc", "semeval_en"])
    k_values = cfg.get("metrics", {}).get("k_values", [5, 10, 15, 20])

    for dataset in datasets:
        bundle = load_prepared_discovery_benchmark(dataset)
        if not bundle:
            results[dataset] = {"available": False}
            continue
        corpus: TemporalCorpus = bundle["corpus"]
        labels: dict[str, int] = bundle["labels"]
        graded: dict[str, float] = bundle["graded"]
        early, late = bundle["early"], bundle["late"]
        terms = sorted(set(labels) | set(graded))

        local = _demo_cfg(cfg)
        local["paths"]["corpus"] = bundle["corpus_path"]
        local["paths"]["output_dir"] = str(out_dir / f"discovery_{dataset}")
        local["paths"]["reports"] = str(out_dir / f"discovery_{dataset}" / "shift_reports.json")
        local["pipeline"]["early_period"] = early
        local["pipeline"]["late_period"] = late
        local["pipeline"]["top_reports"] = max(50, len(terms))
        local["pipeline"]["force_terms"] = terms
        local["candidates"]["top_k"] = max(50, len(terms))
        local["candidates"]["min_freq"] = 1
        local["corpus"]["min_token_freq"] = 1
        # Annotation contexts are already word-focused; disable calibration fit on
        # tiny overlapping sets to keep ranking informative.
        local["calibration"]["enabled"] = False

        method_block: dict[str, Any] = {}
        for method in ("static_embedding", "contextual_aggregate", "sense_cluster"):
            scores = score_vocabulary(corpus, terms, early, late, method=method)
            ranked = sorted(scores, key=lambda t: scores[t], reverse=True)
            y_true = [graded.get(t, float(labels.get(t, 0))) for t in terms]
            y_pred = [scores.get(t, 0.0) for t in terms]
            method_block[method] = {
                "spearman_graded": spearman_correlation(y_true, y_pred),
                **ranking_metrics(labels, ranked, k_values=k_values),
            }

        pipe = run_ecos_pipeline(local)
        ecos_scores = {r.term: r.calibrated_confidence for r in pipe.reports}
        # Prefer ECOS scores; fall back to contextual baseline for missing terms
        base_scores = method_block["contextual_aggregate"]
        # re-score map from previous loop
        contextual_scores = score_vocabulary(
            corpus, terms, early, late, method="contextual_aggregate"
        )
        combined = {
            t: ecos_scores.get(t, contextual_scores.get(t, 0.0)) for t in terms
        }
        ranked_ecos = sorted(combined, key=lambda t: combined[t], reverse=True)
        method_block["ecos"] = {
            "spearman_graded": spearman_correlation(
                [graded.get(t, float(labels.get(t, 0))) for t in ranked_ecos],
                [combined[t] for t in ranked_ecos],
            ),
            **ranking_metrics(labels, ranked_ecos, k_values=k_values),
            "top15": ranked_ecos[:15],
            "n_scored_by_pipeline": len(ecos_scores),
        }
        # silence unused
        _ = base_scores

        results[dataset] = {
            "available": True,
            "n_labels": len(labels),
            "n_positive": int(sum(labels.values())),
            "n_docs": len(corpus.documents),
            "early": early,
            "late": late,
            "methods": method_block,
        }

    save_json(results, out_dir / "exp_discovery_benchmarks.json")
    return results


def experiment_1_standard_lsc(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Compare baselines and ECOS against graded/binary labels (sample or SemEval)."""
    labels = synthetic_discovery_labels()
    corpus = load_sample_corpus(cfg.get("paths", {}).get("sample_corpus") or cfg.get("paths", {}).get("corpus"))
    early, late = "2020", "2025"
    terms = list(labels.keys())

    results: dict[str, Any] = {"labels": labels, "methods": {}}
    for method in ("static_embedding", "contextual_aggregate", "sense_cluster"):
        scores = score_vocabulary(corpus, terms, early, late, method=method)
        y_true = [float(labels[t]) for t in terms]
        y_pred = [scores.get(t, 0.0) for t in terms]
        ranked = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        results["methods"][method] = {
            "spearman": spearman_correlation(y_true, y_pred),
            **ranking_metrics(labels, ranked, k_values=[5, 10]),
            "scores": scores,
        }

    # ECOS
    pipe = run_ecos_pipeline(_demo_cfg(cfg))
    ecos_scores = {r.term: r.calibrated_confidence for r in pipe.reports}
    ranked = [r.term for r in pipe.reports]
    y_true = [float(labels.get(t, 0)) for t in ecos_scores]
    y_pred = list(ecos_scores.values())
    results["methods"]["ecos"] = {
        "spearman": spearman_correlation(y_true, y_pred) if y_true else 0.0,
        **ranking_metrics(labels, ranked, k_values=[5, 10]),
        "scores": ecos_scores,
    }

    # Optional SemEval graded if present
    semeval_root = cfg.get("paths", {}).get("semeval_root", "data/raw/semeval2020")
    bundle = try_load_semeval_bundle(semeval_root)
    results["semeval_available"] = bundle is not None
    if bundle and "graded" in bundle:
        results["semeval_targets"] = bundle.get("targets", list(bundle["graded"]))
        results["semeval_note"] = "Graded labels loaded; run full contextual baselines after downloading corpora."

    # Prefer discovery SemEval-EN graded if prepared
    disc = load_prepared_discovery_benchmark("semeval_en")
    if disc:
        results["discovery_semeval_en"] = {
            "n_labels": len(disc["labels"]),
            "n_positive": int(sum(disc["labels"].values())),
            "n_docs": len(disc["corpus"].documents),
        }

    save_json(results, out_dir / "exp1_standard_lsc.json")
    return results

def experiment_2_open_vocab(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    labels = synthetic_discovery_labels()
    pipe = run_ecos_pipeline(_demo_cfg(cfg))
    ranked = [r.term for r in pipe.reports]
    metrics = ranking_metrics(labels, ranked, k_values=cfg.get("metrics", {}).get("k_values", [5, 10, 20]))

    corpus = load_sample_corpus()
    cands = discover_candidates(corpus, "2020", "2025", prep_cfg=PreprocessConfig(min_token_freq=2))
    cand_ranked = [t for t, _, _ in cands]
    cand_metrics = ranking_metrics(labels, cand_ranked, k_values=[5, 10, 20])

    results = {
        "ecos_discovery": metrics,
        "candidate_only": cand_metrics,
        "ranked_ecos": ranked,
        "ranked_candidates": cand_ranked[:20],
    }
    save_json(results, out_dir / "exp2_open_vocab.json")
    return results


def _perturb_topic_only(corpus: TemporalCorpus) -> TemporalCorpus:
    """Create late-period topic skew without changing texts (source/topic metadata only)."""
    docs: list[Document] = []
    for d in corpus.documents:
        nd = Document.from_dict(d.to_dict())
        if nd.timestamp.year >= 2025:
            nd.topic = "agents"
            nd.community = "neurips"
            nd.source = "arxiv-ai-heavy"
        docs.append(nd)
    return TemporalCorpus(name=corpus.name + "-topic-pert", documents=docs, window=corpus.window)


def _perturb_frequency(corpus: TemporalCorpus, term: str = "pipeline") -> TemporalCorpus:
    """Duplicate late docs mentioning term to inflate frequency."""
    docs = list(corpus.documents)
    extras: list[Document] = []
    for d in corpus.documents:
        if d.timestamp.year >= 2025 and term in d.text.lower():
            for i in range(5):
                extras.append(
                    Document(
                        document_id=f"{d.document_id}-freq-{i}",
                        timestamp=d.timestamp,
                        text=d.text,
                        source=d.source,
                        domain=d.domain,
                        language=d.language,
                        topic=d.topic,
                        community=d.community,
                    )
                )
    return TemporalCorpus(name=corpus.name + "-freq-pert", documents=docs + extras, window=corpus.window)


def experiment_3_false_discovery(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """Measure FDR under confounder perturbations vs distance-only baseline."""
    labels = synthetic_discovery_labels()
    corpus = load_sample_corpus()
    early, late = "2020", "2025"

    scenarios = {
        "original": corpus,
        "topic_only": _perturb_topic_only(corpus),
        "frequency_only": _perturb_frequency(corpus),
    }

    results: dict[str, Any] = {}
    for name, corp in scenarios.items():
        # Write temp corpus
        tmp_path = out_dir / f"corpus_{name}.json"
        from ecos.io import save_corpus

        save_corpus(corp, tmp_path)
        local_cfg = _demo_cfg(cfg)
        local_cfg["paths"]["corpus"] = str(tmp_path)
        local_cfg["paths"]["output_dir"] = str(out_dir / name)
        local_cfg["paths"]["reports"] = str(out_dir / name / "shift_reports.json")

        pipe = run_ecos_pipeline(local_cfg)
        ranked_ecos = [r.term for r in pipe.reports]
        # Distance-only baseline on vocabulary
        vocab = list({*labels.keys(), *[r.term for r in pipe.reports]})
        base_scores = score_vocabulary(corp, vocab, early, late, method="contextual_aggregate")
        ranked_base = sorted(base_scores, key=lambda t: base_scores[t], reverse=True)

        results[name] = {
            "ecos": ranking_metrics(labels, ranked_ecos, k_values=[5, 10]),
            "baseline_contextual": ranking_metrics(labels, ranked_base, k_values=[5, 10]),
            "ecos_fdr@10": ranking_metrics(labels, ranked_ecos, k_values=[10]).get("FDR@10"),
            "baseline_fdr@10": ranking_metrics(labels, ranked_base, k_values=[10]).get("FDR@10"),
            "mean_cf_robustness": float(
                np.mean([r.counterfactual_robustness for r in pipe.reports]) if pipe.reports else 0.0
            ),
        }

    # Headline: FDR reduction on topic perturbation
    if "topic_only" in results:
        base_fdr = results["topic_only"]["baseline_fdr@10"] or 0
        ecos_fdr = results["topic_only"]["ecos_fdr@10"] or 0
        results["headline"] = {
            "topic_perturbation_fdr_reduction": float(base_fdr - ecos_fdr),
            "baseline_fdr@10": base_fdr,
            "ecos_fdr@10": ecos_fdr,
        }

    save_json(results, out_dir / "exp3_false_discovery.json")
    return results


def experiment_4_calibration(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    pipe = run_ecos_pipeline(_demo_cfg(cfg))
    labels = synthetic_discovery_labels()
    raw = np.array([r.raw_ecss for r in pipe.reports], dtype=np.float64)
    cal = np.array([r.calibrated_confidence for r in pipe.reports], dtype=np.float64)
    y = np.array([labels.get(r.term, 0) for r in pipe.reports], dtype=np.float64)

    results = {
        "n": len(pipe.reports),
        "raw_ece": expected_calibration_error(raw, y) if len(raw) else None,
        "calibrated_ece": expected_calibration_error(cal, y) if len(cal) else None,
        "raw_brier": brier_score(raw, y) if len(raw) else None,
        "calibrated_brier": brier_score(cal, y) if len(cal) else None,
    }
    # Refit calibrator diagnostics
    if len(raw) >= 2:
        c = Calibrator(method="platt").fit(raw, y.astype(int))
        transformed = c.transform(raw)
        results["refit_ece"] = expected_calibration_error(np.asarray(transformed), y)

    save_json(results, out_dir / "exp4_calibration.json")
    return results


def experiment_ablations(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    labels = synthetic_discovery_labels()
    variants = {
        "full": {},
        "no_counterfactual": {"counterfactual": {"enabled": False}},
        "no_calibration": {"calibration": {"enabled": False}},
        "weak_controls": {
            "controls": {"topic_threshold": 0.99, "source_threshold": 0.99, "frequency_threshold": 0.99}
        },
    }
    results: dict[str, Any] = {}
    for name, override in variants.items():
        local = _demo_cfg(cfg)
        for k, v in override.items():
            local[k] = {**local.get(k, {}), **v} if isinstance(v, dict) else v
        local["paths"]["output_dir"] = str(out_dir / f"ablation_{name}")
        local["paths"]["reports"] = str(out_dir / f"ablation_{name}" / "shift_reports.json")
        pipe = run_ecos_pipeline(local)
        ranked = [r.term for r in pipe.reports]
        results[name] = ranking_metrics(labels, ranked, k_values=[5, 10])

    save_json(results, out_dir / "ablations.json")
    return results


def run_all_experiments(config_path: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config_path) if config_path else {}
    logger = setup_logging("INFO")
    out_dir = Path(cfg.get("paths", {}).get("output_dir", "data/reports/experiments"))
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_flags = cfg.get("experiments", {})
    summary: dict[str, Any] = {}

    if exp_flags.get("exp1_standard_lsc", True):
        logger.info("Running Experiment 1 — standard LSC")
        summary["exp1"] = experiment_1_standard_lsc(cfg, out_dir)
    if exp_flags.get("exp_discovery_benchmarks", True):
        logger.info("Running discovery benchmarks (LiverpoolFC / SemEval-EN)")
        summary["discovery"] = experiment_discovery_benchmarks(cfg, out_dir)
    if exp_flags.get("exp2_open_vocab_discovery", True):
        logger.info("Running Experiment 2 — open-vocabulary discovery")
        summary["exp2"] = experiment_2_open_vocab(cfg, out_dir)
    if exp_flags.get("exp3_false_discovery", True):
        logger.info("Running Experiment 3 — false discovery resistance")
        summary["exp3"] = experiment_3_false_discovery(cfg, out_dir)
    if exp_flags.get("exp4_calibration", True):
        logger.info("Running Experiment 4 — calibration")
        summary["exp4"] = experiment_4_calibration(cfg, out_dir)
    if exp_flags.get("ablations", True):
        logger.info("Running ablations")
        summary["ablations"] = experiment_ablations(cfg, out_dir)

    save_json({k: ("ok" if v else v) for k, v in summary.items()}, out_dir / "summary.json")
    # Save compact summary without huge nested scores
    compact = {}
    for k, v in summary.items():
        if k == "exp1":
            compact[k] = {
                m: {kk: vv for kk, vv in info.items() if kk != "scores"}
                for m, info in v.get("methods", {}).items()
            }
        elif k == "discovery":
            compact[k] = {}
            for ds, info in v.items():
                if not info.get("available"):
                    compact[k][ds] = {"available": False}
                    continue
                methods = {
                    m: {kk: vv for kk, vv in mv.items() if kk != "top15"}
                    for m, mv in info.get("methods", {}).items()
                }
                compact[k][ds] = {
                    "n_labels": info.get("n_labels"),
                    "n_positive": info.get("n_positive"),
                    "n_docs": info.get("n_docs"),
                    "methods": methods,
                    "ecos_top15": info.get("methods", {}).get("ecos", {}).get("top15", []),
                }
        elif k == "exp3":
            compact[k] = {
                sk: {
                    kk: sv[kk]
                    for kk in ("ecos_fdr@10", "baseline_fdr@10", "mean_cf_robustness")
                    if kk in sv
                }
                for sk, sv in v.items()
                if isinstance(sv, dict)
            }
            if "headline" in v:
                compact[k]["headline"] = v["headline"]
        else:
            compact[k] = v
    save_json(compact, out_dir / "summary_compact.json")
    logger.info("Experiments complete → %s", out_dir)
    return summary
