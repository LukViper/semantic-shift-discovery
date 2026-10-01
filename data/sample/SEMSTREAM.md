# SEM-STREAM annotation schema and gold seed for controlled confounder experiments.

## Label classes

| class | meaning |
|-------|---------|
| genuine | Contextual meaning/sense inventory changed |
| topic_only | Apparent shift explained by topic composition |
| frequency_only | Apparent shift explained by frequency burst |
| source_only | Apparent shift explained by source/community skew |
| stable | No meaningful semantic change |

## Files

- `arxiv_sample.json` — longitudinal demo documents
- `semstream_labels.json` — binary discovery labels (1=genuine, 0=other)
- `semstream_classes.json` — fine-grained class labels

## Extension path

1. Download arXiv via `python scripts/download_data.py --arxiv`
2. Run candidate discovery
3. Annotate top-200 candidates into the five classes
4. Build perturbation splits for Experiment 3
