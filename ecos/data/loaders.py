"""Data loaders for SemEval, discovery benchmarks, arXiv, and sample corpora."""

from __future__ import annotations

import csv
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote_plus

import requests

from ecos.schema import Document, TemporalCorpus

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def load_json_corpus(path: str | Path, name: str = "corpus", window: str = "year") -> TemporalCorpus:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    docs = [Document.from_dict(item) for item in raw]
    return TemporalCorpus(name=name, documents=docs, window=window)


def load_sample_corpus(path: str | Path | None = None) -> TemporalCorpus:
    root = Path(__file__).resolve().parents[2]
    sample = Path(path) if path else root / "data" / "sample" / "arxiv_sample.json"
    if not sample.exists():
        _write_builtin_sample(sample)
    return load_json_corpus(sample, name="arxiv-cs-sample", window="year")


def _write_builtin_sample(path: Path) -> None:
    """Fallback writer if sample JSON is missing."""
    builder = Path(__file__).resolve().parents[2] / "data" / "sample" / "build_sample.py"
    if builder.exists():
        import importlib.util

        spec = importlib.util.spec_from_file_location("build_sample", builder)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.write_sample(path)
        return
    raise FileNotFoundError(f"Sample corpus missing and builder not found: {path}")


# ---------------------------------------------------------------------------
# SemEval-2020 Task 1 (English) — flexible loader for local downloads
# ---------------------------------------------------------------------------


def load_semeval_targets(targets_path: str | Path) -> list[str]:
    path = Path(targets_path)
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line.split("\t")[0].split("_")[0])
    return words


def load_semeval_graded(
    truth_path: str | Path,
) -> dict[str, float]:
    """Load graded change scores: word -> float."""
    path = Path(truth_path)
    scores: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"[\t, ]+", line)
        if len(parts) < 2:
            continue
        word = parts[0].split("_")[0]
        try:
            scores[word] = float(parts[1])
        except ValueError:
            continue
    return scores


def load_semeval_corpus_dir(
    corpus_dir: str | Path,
    period_name: str,
    source: str = "semeval",
) -> list[Document]:
    """Load plain-text files under a SemEval period directory as documents."""
    corpus_dir = Path(corpus_dir)
    docs: list[Document] = []
    year = 2010 if "1" in period_name or "old" in period_name.lower() else 2020
    for i, path in enumerate(sorted(corpus_dir.rglob("*.txt"))):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        docs.append(
            Document(
                document_id=f"{period_name}-{i}",
                timestamp=datetime(year, 1, 1),
                text=text,
                source=source,
                domain="semeval",
                language="en",
                topic=period_name,
            )
        )
    return docs


def try_load_semeval_bundle(root: str | Path) -> dict[str, Any] | None:
    """
    Attempt to load a local SemEval-2020 English bundle.

    Expected flexible layout (any subset):
      root/
        targets.txt
        graded.txt | truth/graded.txt
        corpus1/ or c1/ or old/
        corpus2/ or c2/ or new/
    """
    root = Path(root)
    if not root.exists():
        return None
    targets_path = next(root.rglob("targets*.txt"), None)
    graded_path = next(
        (p for p in root.rglob("*.txt") if "grad" in p.name.lower() or "truth" in p.name.lower()),
        None,
    )
    result: dict[str, Any] = {"root": str(root)}
    if targets_path:
        result["targets"] = load_semeval_targets(targets_path)
    if graded_path:
        result["graded"] = load_semeval_graded(graded_path)
    return result if len(result) > 1 else None


# ---------------------------------------------------------------------------
# Open-vocabulary discovery gold (CSV / TSV)
# ---------------------------------------------------------------------------


def load_discovery_labels(path: str | Path) -> dict[str, int]:
    """
    Load binary change labels for open-vocabulary evaluation.
    Accepts CSV/TSV with columns word,label or term,changed.
    Also accepts JSON with a `binary` mapping.
    """
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        binary = data.get("binary", data)
        return {str(k): int(v) for k, v in binary.items()}

    labels: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            keys = {k.lower(): k for k in row}
            word_key = keys.get("word") or keys.get("term") or keys.get("lemma")
            label_key = (
                keys.get("label")
                or keys.get("changed")
                or keys.get("change")
                or keys.get("y")
            )
            if not word_key or not label_key:
                continue
            word = row[word_key].strip().split("_")[0]
            raw = row[label_key].strip().lower()
            labels[word] = 1 if raw in {"1", "true", "yes", "changed"} else 0
    return labels


def load_discovery_graded(path: str | Path) -> dict[str, float]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    graded = data.get("graded", {})
    return {str(k): float(v) for k, v in graded.items()}


def load_prepared_discovery_benchmark(
    dataset: str = "liverpoolfc",
    root: str | Path | None = None,
) -> dict[str, Any]:
    """
    Load prepared EMNLP-2025 discovery resources written by
    `scripts/prepare_benchmarks.py`.
    """
    root = Path(root) if root else Path(__file__).resolve().parents[2] / "data" / "raw" / "discovery"
    labels_path = root / f"{dataset}_labels.json"
    corpus_path = root / f"{dataset}_contexts.json"
    if not labels_path.exists() or not corpus_path.exists():
        return {}
    binary = load_discovery_labels(labels_path)
    graded = load_discovery_graded(labels_path)
    corpus = load_json_corpus(corpus_path, name=dataset, window="year")
    periods = corpus.periods()
    return {
        "dataset": dataset,
        "labels": binary,
        "graded": graded,
        "corpus": corpus,
        "early": periods[0] if periods else None,
        "late": periods[-1] if periods else None,
        "labels_path": str(labels_path),
        "corpus_path": str(corpus_path),
    }


def synthetic_discovery_labels() -> dict[str, int]:
    """Gold-ish labels aligned with the sample arXiv corpus narrative."""
    return {
        "agent": 1,
        "prompt": 1,
        "alignment": 1,
        "hallucination": 1,
        "reasoning": 1,
        "inference": 0,
        "pipeline": 0,
        "membrane": 0,
        "chemical": 0,
        "multimodal": 1,
    }


# ---------------------------------------------------------------------------
# arXiv API ingestion
# ---------------------------------------------------------------------------


def _parse_arxiv_entry(entry: ET.Element) -> Document | None:
    title_el = entry.find("atom:title", ATOM_NS)
    summary_el = entry.find("atom:summary", ATOM_NS)
    id_el = entry.find("atom:id", ATOM_NS)
    published_el = entry.find("atom:published", ATOM_NS)
    if summary_el is None or published_el is None:
        return None
    categories = [
        c.attrib.get("term", "")
        for c in entry.findall("atom:category", ATOM_NS)
    ]
    primary = categories[0] if categories else "cs"
    published = datetime.fromisoformat(published_el.text.replace("Z", "+00:00"))
    abstract = re.sub(r"\s+", " ", (summary_el.text or "").strip())
    title = re.sub(r"\s+", " ", (title_el.text or "").strip()) if title_el is not None else ""
    text = f"{title}. {abstract}".strip()
    doc_id = (id_el.text or f"arxiv-{published.timestamp()}").split("/")[-1]
    return Document(
        document_id=doc_id,
        timestamp=published.replace(tzinfo=None),
        text=text,
        source="arxiv",
        domain=primary,
        language="en",
        topic=primary,
        community=primary,
        metadata={"categories": categories, "title": title},
    )


def fetch_arxiv_year(
    category: str,
    year: int,
    max_results: int = 100,
    base_url: str = "https://export.arxiv.org/api/query",
    page_size: int = 50,
    sleep_seconds: float = 3.0,
) -> list[Document]:
    """Fetch abstracts for one category and calendar year via arXiv Atom API."""
    docs: list[Document] = []
    start = 0
    query = f"cat:{category} AND submittedDate:[{year}01010000 TO {year}12312359]"
    while len(docs) < max_results:
        n = min(page_size, max_results - len(docs))
        url = (
            f"{base_url}?search_query={quote_plus(query)}"
            f"&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=ascending"
        )
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        entries = root.findall("atom:entry", ATOM_NS)
        if not entries:
            break
        for entry in entries:
            doc = _parse_arxiv_entry(entry)
            if doc is not None:
                docs.append(doc)
        start += len(entries)
        if len(entries) < n:
            break
        time.sleep(sleep_seconds)
    return docs[:max_results]


def build_arxiv_corpus(
    categories: Iterable[str],
    start_year: int,
    end_year: int,
    max_results_per_year: int = 100,
    **fetch_kwargs: Any,
) -> TemporalCorpus:
    documents: list[Document] = []
    for year in range(start_year, end_year + 1):
        for cat in categories:
            documents.extend(
                fetch_arxiv_year(cat, year, max_results=max_results_per_year, **fetch_kwargs)
            )
    return TemporalCorpus(name="arxiv-cs", documents=documents, window="year")
