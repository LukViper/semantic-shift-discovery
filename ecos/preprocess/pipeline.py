"""Text preprocessing for longitudinal corpora."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ecos.schema import Document, TemporalCorpus

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{1,}")

DEFAULT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "we",
    "our",
    "you",
    "your",
    "they",
    "their",
    "he",
    "she",
    "his",
    "her",
    "not",
    "no",
    "yes",
    "can",
    "may",
    "will",
    "would",
    "could",
    "should",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "into",
    "over",
    "under",
    "than",
    "then",
    "also",
    "such",
    "via",
    "using",
    "used",
    "use",
    "based",
    "paper",
    "propose",
    "proposed",
    "method",
    "methods",
    "results",
    "show",
    "shows",
    "across",
    "complete",
    "question",
    "tasks",
    "task",
    "models",
    "model",
    "large",
    "human",
    "systems",
    "system",
    "data",
    "study",
    "studies",
    "approach",
    "approaches",
    "new",
    "high",
    "low",
    "main",
    "first",
    "second",
    "two",
    "one",
    "many",
    "several",
}


@dataclass
class PreprocessConfig:
    language: str = "en"
    lowercase: bool = True
    remove_stopwords: bool = True
    min_token_len: int = 2
    min_token_freq: int = 2
    deduplicate: bool = True


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def preprocess_document(doc: Document, cfg: PreprocessConfig) -> list[str]:
    tokens = tokenize(doc.text, lowercase=cfg.lowercase)
    tokens = [t for t in tokens if len(t) >= cfg.min_token_len]
    if cfg.remove_stopwords:
        tokens = [t for t in tokens if t not in DEFAULT_STOPWORDS]
    return tokens


def dedupe_documents(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        key = re.sub(r"\s+", " ", doc.text.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def filter_by_language(docs: list[Document], languages: list[str]) -> list[Document]:
    allowed = {l.lower() for l in languages}
    return [d for d in docs if d.language.lower() in allowed]


def build_vocab(
    docs: list[Document],
    cfg: PreprocessConfig,
    max_vocab: int | None = None,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(set(preprocess_document(doc, cfg)))
    items = [(t, c) for t, c in counts.items() if c >= cfg.min_token_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab is not None:
        items = items[:max_vocab]
    return {t: i for i, (t, _) in enumerate(items)}


def preprocess_corpus(
    corpus: TemporalCorpus,
    cfg: PreprocessConfig | None = None,
    languages: list[str] | None = None,
) -> tuple[TemporalCorpus, dict[str, list[str]]]:
    """Return filtered corpus and document_id -> tokens map."""
    cfg = cfg or PreprocessConfig()
    docs = list(corpus.documents)
    if languages:
        docs = filter_by_language(docs, languages)
    if cfg.deduplicate:
        docs = dedupe_documents(docs)
    token_map: dict[str, list[str]] = {
        d.document_id: preprocess_document(d, cfg) for d in docs
    }
    return TemporalCorpus(name=corpus.name, documents=docs, window=corpus.window), token_map
