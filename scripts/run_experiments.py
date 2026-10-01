#!/usr/bin/env python3
"""Run ECOS evaluation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECOS experiments")
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()

    sample = ROOT / "data" / "sample" / "arxiv_sample.json"
    if not sample.exists():
        from importlib.util import module_from_spec, spec_from_file_location

        builder = ROOT / "data" / "sample" / "build_sample.py"
        spec = spec_from_file_location("build_sample", builder)
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.write_sample(sample)

    from ecos.eval.experiments import run_all_experiments

    summary = run_all_experiments(ROOT / args.config)
    print("Completed experiments:", list(summary.keys()))
    if "exp3" in summary and "headline" in summary["exp3"]:
        print("Headline FDR result:", summary["exp3"]["headline"])


if __name__ == "__main__":
    main()
