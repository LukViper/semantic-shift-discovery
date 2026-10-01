# Raw external datasets

## Prepared automatically

```bash
python scripts/prepare_benchmarks.py
```

Writes:
- `data/raw/discovery/{liverpoolfc,semeval_en}_labels.json`
- `data/raw/discovery/{liverpoolfc,semeval_en}_contexts.json`
- `data/processed/liverpoolfc/liverpoolfc_sample.{json,parquet}`

Sources used:
- EMNLP 2025 discovery annotations (`data/raw/discovery_repo`)
- LiverpoolFC full dumps (`data/raw/liverpoolfc/raw`)

## SemEval-2020 Task 1 full corpora (optional)

IMS Stuttgart download is currently blocked in this environment (proxy/DNS).
When available, place under `data/raw/semeval2020/`:

```text
targets.txt
truth/graded.txt
corpus1/lemma/
corpus2/lemma/
```

Until then, SemEval-EN discovery evaluation uses annotation contexts from the
EMNLP 2025 release (same label protocol, θ=0.28).

## arXiv

```bash
python scripts/download_data.py --arxiv --max-per-year 50
```

Requires outbound access to `export.arxiv.org`. If unavailable, the bundled
SEM-STREAM sample under `data/sample/` remains the longitudinal demo corpus.
