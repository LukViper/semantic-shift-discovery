"""Parquet / JSON I/O for corpora and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ecos.schema import Document, ShiftReport, TemporalCorpus


def documents_to_frame(docs: Iterable[Document]) -> pd.DataFrame:
    rows = [d.to_dict() for d in docs]
    return pd.DataFrame(rows)


def frame_to_documents(df: pd.DataFrame) -> list[Document]:
    return [Document.from_dict(row.to_dict()) for _, row in df.iterrows()]


def save_corpus(corpus: TemporalCorpus, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = documents_to_frame(corpus.documents)
    df["corpus_name"] = corpus.name
    df["window"] = corpus.window
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records", indent=2, date_format="iso")


def load_corpus(path: str | Path, name: str | None = None, window: str = "year") -> TemporalCorpus:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_json(path)
    docs = frame_to_documents(df)
    corpus_name = name or (str(df["corpus_name"].iloc[0]) if "corpus_name" in df else path.stem)
    win = window
    if "window" in df.columns and len(df):
        win = str(df["window"].iloc[0])
    return TemporalCorpus(name=corpus_name, documents=docs, window=win)


def save_reports(reports: list[ShiftReport], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in reports]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_reports(path: str | Path) -> list[ShiftReport]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [ShiftReport.from_dict(item) for item in data]


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
