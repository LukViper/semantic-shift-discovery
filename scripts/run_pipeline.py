#!/usr/bin/env python3
"""Run the ECOS end-to-end pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ECOS pipeline")
    parser.add_argument("--config", default="configs/demo.yaml")
    args = parser.parse_args()

    # Ensure sample exists
    sample = ROOT / "data" / "sample" / "arxiv_sample.json"
    if not sample.exists():
        from importlib.util import module_from_spec, spec_from_file_location

        builder = ROOT / "data" / "sample" / "build_sample.py"
        spec = spec_from_file_location("build_sample", builder)
        mod = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        mod.write_sample(sample)

    from ecos.pipeline import run_from_config_path

    result = run_from_config_path(ROOT / args.config)
    print(f"Candidates: {len(result.candidates)}")
    print(f"Reports: {len(result.reports)}")
    print("Metrics hint:", result.metrics_hint)
    print("\nTop emerging semantic shifts:")
    for i, r in enumerate(result.reports[:10], 1):
        flags = " ".join(
            ("✓" if c.passed else "✗") + c.name[:1].upper() for c in r.controls
        )
        print(
            f"{i:2d}. {r.term:18s}  conf={r.calibrated_confidence:.3f}  "
            f"ecss={r.raw_ecss:.3f}  cf={r.counterfactual_robustness:.3f}  [{flags}]"
        )


if __name__ == "__main__":
    main()
