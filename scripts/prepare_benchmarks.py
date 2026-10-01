"""Convert EMNLP 2025 discovery annotations + LiverpoolFC into ECOS formats."""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ecos.io import save_corpus, save_json
from ecos.schema import Document, TemporalCorpus

DISC_ANN = ROOT / "data" / "raw" / "discovery_repo" / "annotations"
LFC_RAW = ROOT / "data" / "raw" / "liverpoolfc" / "raw"
LFC_ANN_CSV = ROOT / "data" / "raw" / "liverpoolfc_src" / "Dataset" / "annotated_words.csv"
OUT_DISC = ROOT / "data" / "raw" / "discovery"
OUT_LFC = ROOT / "data" / "processed" / "liverpoolfc"


def _ann_score(anns: list[dict]) -> float | None:
    """
    Mean of Q3 yes/no across annotators.

    In the EMNLP 2025 discovery protocol, Q1/Q2 are comprehension checks;
    Q3 is the primary meaning-change judgment (θ=0.28 SemEval-EN, θ=0.49 LiverpoolFC).
    """
    vals: list[float] = []
    for a in anns:
        q3 = str(a.get("Q3:", "")).strip().lower()
        if q3 in {"yes", "no"}:
            vals.append(1.0 if q3 == "yes" else 0.0)
    if not vals:
        return None
    return sum(vals) / len(vals)



def annotations_to_labels(
    path: Path,
    theta: float,
    include_extra: Path | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if include_extra and include_extra.exists():
        extra = json.loads(include_extra.read_text(encoding="utf-8"))
        data = {**extra, **data}
    graded: dict[str, float] = {}
    binary: dict[str, int] = {}
    for word, anns in data.items():
        s = _ann_score(anns)
        if s is None:
            continue
        graded[word] = s
        binary[word] = 1 if s >= theta else 0
    return graded, binary


def annotations_to_corpus(
    path: Path,
    name: str,
    early_year: int,
    late_year: int,
    include_extra: Path | None = None,
) -> TemporalCorpus:
    """Build a two-period corpus from Group_1 / Group_2 example sentences."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if include_extra and include_extra.exists():
        extra = json.loads(include_extra.read_text(encoding="utf-8"))
        data = {**extra, **data}

    docs: list[Document] = []
    seen: set[str] = set()
    for word, anns in data.items():
        if not anns:
            continue
        g1 = anns[0].get("Group_1_sents") or []
        g2 = anns[0].get("Group_2_sents") or []
        for i, sent in enumerate(g1):
            text = re.sub(r"\s+", " ", str(sent)).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append(
                Document(
                    document_id=f"{name}-early-{word}-{i}",
                    timestamp=datetime(early_year, 6, 1),
                    text=text,
                    source=name,
                    domain="discovery",
                    language="en",
                    topic=word,
                    community="annotator-context",
                )
            )
        for i, sent in enumerate(g2):
            text = re.sub(r"\s+", " ", str(sent)).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append(
                Document(
                    document_id=f"{name}-late-{word}-{i}",
                    timestamp=datetime(late_year, 6, 1),
                    text=text,
                    source=name,
                    domain="discovery",
                    language="en",
                    topic=word,
                    community="annotator-context",
                )
            )
    return TemporalCorpus(name=name, documents=docs, window="year")


def liverpoolfc_csv_labels(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    graded: dict[str, float] = {}
    binary: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["word"].strip().lower()
            graded[word] = float(row["shift_index"])
            votes = [
                row[k].strip().lower()
                for k in row
                if k.startswith("subject") and row[k].strip() and row[k].strip().upper() != "NA"
            ]
            changes = sum(1 for v in votes if v == "change")
            no_changes = sum(1 for v in votes if v == "no_change")
            if changes + no_changes == 0:
                binary[word] = 1 if graded[word] >= 0.5 else 0
            else:
                binary[word] = 1 if changes > no_changes else 0
    return graded, binary


def subsample_liverpoolfc_corpus(
    early_path: Path,
    late_path: Path,
    focus_terms: set[str],
    max_lines_per_period: int = 2500,
    max_focus_hits: int = 25,
    max_scan_lines: int = 80000,
) -> TemporalCorpus:
    """Build a tractable LiverpoolFC corpus biased toward annotated terms."""
    docs: list[Document] = []
    focus_terms = {t.lower() for t in focus_terms}
    # Single pass: tokenize coarsely; O(1) term membership instead of N regexes/line
    token_re = re.compile(r"[a-zA-Z']+")

    def ingest(path: Path, year: int, period_tag: str) -> None:
        focus_counts: dict[str, int] = defaultdict(int)
        random_kept = 0
        budget_random = max_lines_per_period // 2
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_scan_lines:
                    break
                text = line.strip()
                if len(text) < 20:
                    continue
                toks = {t.lower() for t in token_re.findall(text)}
                hit = next((t for t in focus_terms if t in toks), None)
                keep = False
                if hit is not None and focus_counts[hit] < max_focus_hits:
                    focus_counts[hit] += 1
                    keep = True
                elif hit is None and random_kept < budget_random and (i % 50 == 0):
                    random_kept += 1
                    keep = True
                if not keep:
                    continue
                docs.append(
                    Document(
                        document_id=f"lfc-{period_tag}-{i}",
                        timestamp=datetime(year, 1, 1),
                        text=text[:800],
                        source="liverpoolfc",
                        domain="reddit",
                        language="en",
                        topic=hit or "general",
                        community="r/LiverpoolFC",
                    )
                )
                if (
                    sum(focus_counts.values()) >= min(len(focus_terms), 40) * max_focus_hits
                    and random_kept >= budget_random
                ):
                    break

    ingest(early_path, 2013, "2013")
    ingest(late_path, 2017, "2017")
    return TemporalCorpus(name="liverpoolfc", documents=docs, window="year")


def main() -> None:
    OUT_DISC.mkdir(parents=True, exist_ok=True)
    OUT_LFC.mkdir(parents=True, exist_ok=True)

    # --- Discovery labels (EMNLP 2025) ---
    for dataset, theta, early, late in (
        ("semeval_en", 0.28, 1850, 1980),
        ("liverpoolfc", 0.49, 2013, 2017),
    ):
        main_path = DISC_ANN / f"{dataset}__annotations__main.json"
        extra_path = DISC_ANN / f"{dataset}__annotations__extra.json"
        targets_path = DISC_ANN / f"{dataset}__annotations__targets.json"
        graded, binary = annotations_to_labels(main_path, theta, extra_path)
        # fold targets into label set
        tg, tb = annotations_to_labels(targets_path, theta)
        graded.update(tg)
        binary.update(tb)

        labels_csv = OUT_DISC / f"{dataset}_labels.csv"
        with labels_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["word", "label", "graded"])
            for word in sorted(graded):
                w.writerow([word, binary[word], f"{graded[word]:.4f}"])

        save_json(
            {"graded": graded, "binary": binary, "theta": theta, "n": len(binary)},
            OUT_DISC / f"{dataset}_labels.json",
        )

        corpus = annotations_to_corpus(main_path, dataset, early, late, extra_path)
        # add target contexts
        tcorp = annotations_to_corpus(targets_path, f"{dataset}-targets", early, late)
        merged = TemporalCorpus(
            name=dataset,
            documents=corpus.documents + tcorp.documents,
            window="year",
        )
        save_corpus(merged, OUT_DISC / f"{dataset}_contexts.json")
        print(
            f"{dataset}: {len(binary)} labels "
            f"(pos={sum(binary.values())}) docs={len(merged.documents)} → {labels_csv}"
        )

    # --- Classical LiverpoolFC CSV labels ---
    if LFC_ANN_CSV.exists():
        graded, binary = liverpoolfc_csv_labels(LFC_ANN_CSV)
        save_json(
            {"graded": graded, "binary": binary, "source": "annotated_words.csv"},
            OUT_DISC / "liverpoolfc_classic_labels.json",
        )
        with (OUT_DISC / "liverpoolfc_classic_labels.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["word", "label", "graded"])
            for word in sorted(graded):
                w.writerow([word, binary[word], f"{graded[word]:.4f}"])
        print(f"liverpoolfc classic: {len(binary)} labels")

    # --- Subsampled full LiverpoolFC stream ---
    early = LFC_RAW / "LiverpoolFC_13.txt"
    late = LFC_RAW / "LiverpoolFC_17.txt"
    if early.exists() and late.exists():
        focus = set()
        for name in ("liverpoolfc_labels.json", "liverpoolfc_classic_labels.json"):
            p = OUT_DISC / name
            if p.exists():
                focus |= set(json.loads(p.read_text())["binary"].keys())
        corp = subsample_liverpoolfc_corpus(early, late, focus_terms=focus or {"crew", "toxic"})
        save_corpus(corp, OUT_LFC / "liverpoolfc_sample.json")
        try:
            save_corpus(corp, OUT_LFC / "liverpoolfc_sample.parquet")
        except Exception as exc:  # noqa: BLE001
            print(f"parquet save skipped: {exc}")
        print(f"liverpoolfc sample corpus: {len(corp.documents)} docs → {OUT_LFC}")


if __name__ == "__main__":
    main()
