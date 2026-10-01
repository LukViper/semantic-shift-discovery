"""CLI entrypoint."""

from __future__ import annotations

import click

from ecos import __version__


@click.group()
@click.version_option(__version__, prog_name="ecos")
def main() -> None:
    """ECOS — Evidence-Calibrated Open-Vocabulary Semantic Shift Detection."""


@main.command("pipeline")
@click.option("--config", "config_path", default="configs/demo.yaml", show_default=True)
def pipeline_cmd(config_path: str) -> None:
    """Run the full ECOS discovery pipeline."""
    from ecos.pipeline import run_from_config_path

    result = run_from_config_path(config_path)
    click.echo(f"Candidates: {len(result.candidates)}")
    click.echo(f"Reports: {len(result.reports)}")
    click.echo(f"Metrics: {result.metrics_hint}")
    if result.reports:
        click.echo("Top shifts:")
        for r in result.reports[:5]:
            click.echo(f"  {r.term:20s} conf={r.calibrated_confidence:.3f} ecss={r.raw_ecss:.3f}")


@main.command("experiments")
@click.option("--config", "config_path", default="configs/experiments.yaml", show_default=True)
def experiments_cmd(config_path: str) -> None:
    """Run the evaluation suite."""
    from ecos.eval.experiments import run_all_experiments

    run_all_experiments(config_path)
    click.echo("Done.")


@main.command("download-arxiv")
@click.option("--config", "config_path", default="configs/arxiv.yaml", show_default=True)
@click.option("--max-per-year", default=None, type=int, help="Override max results per year")
def download_arxiv_cmd(config_path: str, max_per_year: int | None) -> None:
    """Download arXiv abstracts into a Parquet corpus."""
    from pathlib import Path

    from ecos.config import load_config
    from ecos.data.loaders import build_arxiv_corpus
    from ecos.io import save_corpus

    cfg = load_config(config_path)
    c = cfg.get("corpus", {})
    api = cfg.get("api", {})
    max_y = max_per_year or int(c.get("max_results_per_year", 50))
    corpus = build_arxiv_corpus(
        categories=c.get("categories", ["cs.AI"]),
        start_year=int(c.get("start_year", 2019)),
        end_year=int(c.get("end_year", 2026)),
        max_results_per_year=max_y,
        base_url=api.get("base_url", "https://export.arxiv.org/api/query"),
        page_size=int(api.get("page_size", 50)),
        sleep_seconds=float(api.get("sleep_seconds", 3.0)),
    )
    out = Path(cfg.get("paths", {}).get("processed", "data/processed/arxiv_cs.parquet"))
    save_corpus(corpus, out)
    click.echo(f"Saved {len(corpus.documents)} documents → {out}")


if __name__ == "__main__":
    main()
