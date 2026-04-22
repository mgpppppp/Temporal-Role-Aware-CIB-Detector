from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cib_detector.experiments import run_experiment_suite


DEFAULT_VARIANTS = [
    "full",
    "no_leader",
    "no_campaign",
    "no_residual",
    "no_mdcs",
    "graphsage_static",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation runner.

    Returns:
        An ``argparse.Namespace`` containing output paths, seeds, scenarios,
        ablation variants, and optional smoke-test settings.
    """
    parser = argparse.ArgumentParser(description="Run ablation studies for the CIB detector.")
    parser.add_argument("--output-dir", type=str, default="experiment_outputs/ablation")
    parser.add_argument("--seeds", type=str, default="7,11,19")
    parser.add_argument("--scenarios", type=str, default="standard,camouflage_low,camouflage_medium,camouflage_high")
    parser.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--quick", action="store_true", help="Run a lightweight smoke-test ablation suite.")
    return parser.parse_args()


def main() -> None:
    """Execute the ablation experiment suite from the command line."""
    args = parse_args()
    seeds = [7] if args.quick else [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    scenarios = ["standard"] if args.quick else [item.strip() for item in args.scenarios.split(",") if item.strip()]
    variants = (
        ["full", "no_leader", "graphsage_static"]
        if args.quick
        else [item.strip() for item in args.variants.split(",") if item.strip()]
    )

    result = run_experiment_suite(
        suite_name="ablation",
        output_dir=args.output_dir,
        seeds=seeds,
        scenario_names=scenarios,
        variant_names=variants,
    )

    print(json.dumps(
        {
            "summary_path": result["summary_path"],
            "delta_path": result["delta_path"],
            "report_path": result["report_path"],
            "runs": len(result["raw"]),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
