from __future__ import annotations

import argparse
import json

from cib_detector.config import DetectorConfig
from cib_detector.pipeline import run_pipeline
from cib_detector.preprocess import load_events
from cib_detector.synthetic import generate_synthetic_events


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the detector entry point.

    Returns:
        An ``argparse.Namespace`` instance containing input paths, runtime
        configuration parameters, and output location settings.
    """
    parser = argparse.ArgumentParser(
        description="Lightweight coordinated inauthentic behavior detector"
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="CSV file containing interaction logs. If omitted, synthetic data is generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory used to save community reports and metrics.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--window-size-minutes", type=int, default=5)
    parser.add_argument("--window-step-minutes", type=int, default=1)
    parser.add_argument("--sync-tolerance-seconds", type=int, default=20)
    parser.add_argument("--edge-quantile", type=float, default=0.90)
    parser.add_argument("--min-edge-weight", type=float, default=0.35)
    parser.add_argument("--null-trials", type=int, default=12)
    parser.add_argument("--min-output-risk", type=float, default=0.55)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> DetectorConfig:
    """Construct a detector configuration from parsed command-line arguments.

    Args:
        args: Parsed command-line arguments returned by :func:`parse_args`.

    Returns:
        A :class:`cib_detector.config.DetectorConfig` instance populated with
        the runtime parameters supplied at the command line.
    """
    return DetectorConfig(
        window_size_minutes=args.window_size_minutes,
        window_step_minutes=args.window_step_minutes,
        sync_tolerance_seconds=args.sync_tolerance_seconds,
        edge_quantile=args.edge_quantile,
        min_edge_weight=args.min_edge_weight,
        null_trials=args.null_trials,
        min_output_risk=args.min_output_risk,
        random_seed=args.seed,
    )


def main() -> None:
    """Run the full coordinated-behavior detection pipeline.

    The entry point loads either a user-provided CSV file or a synthetic event
    dataset, executes the pipeline, and prints a concise summary of detection
    metrics and representative communities for each discovery method.
    """
    args = parse_args()
    config = build_config(args)

    if args.input_csv:
        events = load_events(args.input_csv)
        source = args.input_csv
    else:
        events = generate_synthetic_events(seed=config.random_seed)
        source = "synthetic_generator"

    result = run_pipeline(events, config, output_dir=args.output_dir)
    methods = result["methods"]
    comparison = result["comparison"]
    output_paths = result["output_paths"]

    print(f"data_source={source}")
    print(f"events={len(result['events'])}")
    print("comparison=" + json.dumps(comparison, ensure_ascii=False, indent=2))

    for method_name, method_result in methods.items():
        communities = method_result["communities"]
        metrics = method_result["metrics"]
        print(f"{method_name}_detected_communities={len(communities)}")
        print(f"{method_name}_metrics=" + json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"{method_name}_top_communities=")
        for community in communities[:3]:
            print(
                json.dumps(
                    {
                        "community_id": community.community_id,
                        "members": community.members,
                        "risk_score": round(community.risk_score, 4),
                        "risk_level": community.risk_level,
                        "sync_score": round(community.sync_score, 4),
                        "popularity_score": round(community.popularity_score, 4),
                        "residual_score": round(community.residual_score, 4),
                        "campaign_score": round(community.campaign_score, 4),
                        "leader_score": round(community.leader_score, 4),
                        "centralization_score": round(community.centralization_score, 4),
                        "support_windows": community.support_windows,
                        "window_mdcs": community.window_mdcs,
                        "top_leaders": community.top_leaders,
                        "shared_targets": community.shared_targets,
                    },
                    ensure_ascii=False,
                )
            )
    print("outputs=" + json.dumps(output_paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
