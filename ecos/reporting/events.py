"""Lightweight event/community association for verified shifts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ecos.schema import Document


@dataclass
class Association:
    period: str
    top_topics: list[str]
    top_communities: list[str]
    note: str


def associate_change_point(
    docs: list[Document],
    changepoint: str | None,
    window: str = "year",
) -> Association | None:
    """
    Summarize topics/communities around a detected changepoint.
    Language is associative, not causal.
    """
    if not changepoint:
        return None
    year = changepoint.split("-")[0]
    nearby = []
    for d in docs:
        key = str(d.timestamp.year) if window == "year" else changepoint
        if key == changepoint or str(d.timestamp.year) == year:
            nearby.append(d)
    if not nearby:
        nearby = docs
    topics = [t for t, _ in Counter(d.topic or "unknown" for d in nearby).most_common(5)]
    communities = [
        c for c, _ in Counter(d.community or "unknown" for d in nearby).most_common(5)
    ]
    return Association(
        period=changepoint,
        top_topics=topics,
        top_communities=communities,
        note="Temporally associated topics/communities near the changepoint (not causal).",
    )
