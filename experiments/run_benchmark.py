from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cib_detector.experiments import run_experiment_suite


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner.

    Returns:
        An ``argparse.Namespace`` containing output paths, seeds, scenarios,
        and optional smoke-test settings.
    """
    parser = argparse.ArgumentParser(description="Run the full benchmark suite for the CIB detector.")
    parser.add_argument("--output-dir", type=str, default="experiment_outputs/benchmark")
    parser.add_argument("--seeds", type=str, default="7,11,19,23")
    parser.add_argument("--scenarios", type=str, default="standard,camouflage_low,camouflage_medium,camouflage_high")
    parser.add_argument("--quick", action="store_true", help="Run a lightweight benchmark smoke test.")
    return parser.parse_args()


def main() -> None:
    """Execute the benchmark experiment suite from the command line."""
    args = parse_args()
    seeds = [7] if args.quick else [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    scenarios = ["standard", "camouflage_low"] if args.quick else [
        item.strip() for item in args.scenarios.split(",") if item.strip()
    ]

    result = run_experiment_suite(
        suite_name="benchmark",
        output_dir=args.output_dir,
        seeds=seeds,
        scenario_names=scenarios,
        variant_names=["full"],
    )

    print(json.dumps(
        {
            "summary_path": result["summary_path"],
            "report_path": result["report_path"],
            "runs": len(result["raw"]),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
