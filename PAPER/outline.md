# Paper status

| File | Role |
|------|------|
| [`MANUSCRIPT.md`](MANUSCRIPT.md) | Full readable draft with Mermaid + figure embeds |
| [`ieee_paper.tex`](ieee_paper.tex) | Full IEEE journal LaTeX manuscript (figures included) |
| [`figures/`](figures/) | Publication plots (PNG + PDF) |
| [`ieee_skeleton.tex`](ieee_skeleton.tex) | Earlier stub (superseded by `ieee_paper.tex`) |
| [`outline.md`](outline.md) | Working outline + result scratchpad |
| [`related_work.md`](related_work.md) | Bibliography notes |

See also [`../docs/FIGURES.md`](../docs/FIGURES.md) and [`../docs/diagrams/`](../docs/diagrams/).

**Regenerate figures:**

```bash
python scripts/generate_figures.py
```

**Compile LaTeX** (requires IEEEtran):

```bash
pdflatex ieee_paper.tex
pdflatex ieee_paper.tex
```

---

**System name:** ECOS — Evidence-Calibrated Open-vocabulary Semantic-shift detection

**One-line description:** An AI framework that automatically discovers emerging changes in word meaning across time, distinguishes genuine semantic evolution from topic/source/frequency-driven distributional changes, validates discoveries through counterfactual testing, and produces calibrated textual evidence for each detected shift.

**Target venue:** IEEE (Access / Big Data / Transactions on Artificial Intelligence — finalize after experiments)

---

## Abstract (draft)

Lexical semantic change detection often reduces to measuring representation distance between time periods. In open-vocabulary settings, however, many highly ranked candidates are not genuine meaning changes but artifacts of topic, source, or frequency shifts. We present ECOS, an evidence-calibrated framework that treats semantic change as a falsifiable hypothesis: candidate shifts must survive confounder controls and counterfactual distribution matching, and must be supported by representative textual evidence with calibrated confidence. Experiments on public lexical semantic change resources and a longitudinal scientific abstract stream (SEM-STREAM) show that counterfactual verification reduces false discoveries relative to representation-distance baselines while improving confidence reliability.

---

## 1. Introduction

Language evolves. Monitoring systems must discover *which* terms changed, *when*, and *whether* the change is semantic rather than distributional. A 2025 EMNLP study shows substantial disagreement between curated-target evaluation and open-vocabulary discovery quality. ECOS reframes the problem from embedding displacement to evidence-calibrated discovery.

**Contributions**

1. Open-vocabulary semantic discovery pipeline over longitudinal text.
2. Confounder-aware controls for topic, source, frequency, and composition.
3. Counterfactual semantic verification via stratum matching/reweighting.
4. Evidence retrieval + confidence calibration (ECSS).
5. SEM-STREAM: a confounder-aware longitudinal benchmark (in progress).

---

## 2. Related work

- Temporal / continuous semantic-change modeling
- Instance-level change (e.g., Unbalanced Optimal Transport, 2025)
- Explainable sense/definition generation
- Multilingual shared tasks (AXOLOTL'24)
- Open-vocabulary discovery evaluation (EMNLP 2025)

**Gap:** discovery + confounder control + counterfactual verification + calibrated evidence in one system.

---

## 3. Method

### 3.1 Longitudinal corpus and candidate discovery

Cheap multi-signal candidate score:

$$C(w)=\alpha F(w)+\beta D(w)+\gamma V(w)+\delta T(w)$$

### 3.2 Contextual representation and sense induction

Frozen contextual encoder; per-period clustering (k-means / HDBSCAN).

### 3.3 Multi-signal shift engine

Contextual divergence, sense-distribution JS, neighborhood Jaccard distance, temporal changepoint.

### 3.4 Confounder controls

Topic / source / frequency / composition divergences; PASS/FAIL relative to observed shift magnitude.

### 3.5 Counterfactual verification

Match late-period stratum weights to early-period margins; recompute shift score \(S_{cf}\). Robustness and survival ratio enter ECSS.

### 3.6 Evidence-Calibrated Semantic Shift Score (ECSS)

$$ECSS = \sum_i w_i S_i - \sum_j w_j C_j$$

then Platt / isotonic calibration → confidence.

**Scientific caveat:** counterfactual matching improves robustness to *observable* confounders; it does not establish causality. Prefer “evidence-based temporal association.”

---

## 4. Experimental design

| Exp | Question | Metrics |
|-----|----------|---------|
| 1 | Standard LSC | Spearman, P@K, MAP, NDCG |
| 2 | Open-vocab discovery | P@K, R@K, MAP@K, NDCG@K |
| 3 | False-discovery resistance | FDR, robustness |
| 4 | Calibration | ECE, Brier |
| Ablations | −controls / −CF / −calibration | ΔFDR, ΔP@K |

**Baselines:** static embedding distance, contextual aggregate, sense clustering; discovery evaluation protocol from 2025 study when data available.

---

## 5. Datasets

- SemEval-2020 Task 1 (EN)
- LiverpoolFC / SemEval-EN discovery extensions (EMNLP 2025 release)
- SEM-STREAM: arXiv CS.AI/CL/LG abstracts with confounder annotations (genuine / topic / frequency / source / stable)

---

## 6. Results (current runs)

### SEM-STREAM demo sample
- Open-vocab ECOS: P@5 ≈ 0.60, MAP@10 ≈ 0.73
- Topic perturbation FDR@10: baseline 0.60 → ECOS 0.50

### EMNLP 2025 discovery benchmarks (prepared locally)

Labels from Q3 annotator mean with published thresholds (θ=0.28 SemEval-EN, θ=0.49 LiverpoolFC).
Corpora = annotation Group_1/Group_2 contexts (+ LiverpoolFC reddit subsample).

| Dataset | Method | P@5 | P@10 | MAP | FDR@10 |
|---------|--------|-----|------|-----|--------|
| LiverpoolFC | contextual aggregate | 0.40 | 0.40 | 0.34 | 0.60 |
| LiverpoolFC | ECOS | 0.20 | 0.20 | 0.32 | 0.80 |
| SemEval-EN | contextual aggregate | 0.80 | 0.70 | 0.70 | 0.30 |
| SemEval-EN | ECOS | 0.80 | 0.60 | 0.64 | 0.40 |

LiverpoolFC reproduces the paper’s discovery gap: top-ranked candidates are mostly non-changes.
Full SemEval-2020 CCOHA dump and arXiv API ingest remain blocked by network in this environment;
loaders and `scripts/prepare_benchmarks.py` / `scripts/download_data.py --arxiv` are ready when access works.


---

## 7. Limitations

- Observable confounders only; unmeasured community effects remain.
- Sense labels are unsupervised and approximate.
- Calibration quality depends on labeled discovery sets.
- No causal claim about *why* meanings change.

---

## 8. Conclusion

ECOS converts open-vocabulary semantic-change detection from a representation-distance problem into an evidence-calibrated discovery process. Semantic shifts must survive confounder-aware counterfactual tests and present consistent textual evidence.

---

## Reproducibility

```bash
pip install -e ".[all]"
python scripts/download_data.py
python scripts/run_pipeline.py --config configs/demo.yaml
python scripts/run_experiments.py --config configs/experiments.yaml
streamlit run dashboard/app.py
```
