#!/usr/bin/env python3
"""Download / prepare datasets: sample, discovery benchmarks, optional arXiv."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_sample() -> Path:
    out = ROOT / "data" / "sample" / "arxiv_sample.json"
    builder = ROOT / "data" / "sample" / "build_sample.py"
    spec = importlib.util.spec_from_file_location("build_sample", builder)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.write_sample(out)


def prepare_benchmarks() -> None:
    path = ROOT / "scripts" / "prepare_benchmarks.py"
    spec = importlib.util.spec_from_file_location("prepare_benchmarks", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.main()


def download_arxiv(config: Path, max_per_year: int) -> Path:
    from ecos.config import load_config
    from ecos.data.loaders import build_arxiv_corpus
    from ecos.io import save_corpus

    cfg = load_config(config)
    c = cfg.get("corpus", {})
    api = cfg.get("api", {})
    corpus = build_arxiv_corpus(
        categories=c.get("categories", ["cs.AI"]),
        start_year=int(c.get("start_year", 2020)),
        end_year=int(c.get("end_year", 2025)),
        max_results_per_year=max_per_year,
        base_url=api.get("base_url", "https://export.arxiv.org/api/query"),
        page_size=int(api.get("page_size", 50)),
        sleep_seconds=float(api.get("sleep_seconds", 3.0)),
    )
    out = ROOT / cfg.get("paths", {}).get("processed", "data/processed/arxiv_cs.parquet")
    save_corpus(corpus, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", default=True)
    parser.add_argument("--prepare-benchmarks", action="store_true", default=True)
    parser.add_argument("--arxiv", action="store_true")
    parser.add_argument("--config", default="configs/arxiv.yaml")
    parser.add_argument("--max-per-year", type=int, default=20)
    args = parser.parse_args()

    p = ensure_sample()
    print(f"Sample corpus → {p}")

    if args.prepare_benchmarks:
        prepare_benchmarks()

    if args.arxiv:
        out = download_arxiv(ROOT / args.config, args.max_per_year)
        print(f"arXiv corpus → {out}")


if __name__ == "__main__":
    main()
