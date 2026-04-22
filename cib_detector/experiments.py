from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DetectorConfig
from .pipeline import run_pipeline
from .synthetic import generate_synthetic_events


SCENARIOS: dict[str, dict[str, object]] = {
    "standard": {
        "description": "Default synthetic benchmark with mixed organic traffic and coordinated botnet bursts.",
        "generator_kwargs": {
            "benign_users": 80,
            "bot_group_sizes": [4, 5, 6],
            "content_count": 180,
        },
        "config_updates": {},
    },
    "camouflage_low": {
        "description": "Low-strength adversarial camouflage with modest lag jitter, decoys, and hot-topic masking.",
        "generator_kwargs": {
            "benign_users": 80,
            "bot_group_sizes": [4, 5, 6],
            "content_count": 180,
            "adversarial_strength": 0.30,
        },
        "config_updates": {
            "edge_quantile": 0.89,
        },
    },
    "camouflage_medium": {
        "description": "Medium-strength camouflage with weaker synchronization, lower overlap, and more organic session rhythm.",
        "generator_kwargs": {
            "benign_users": 90,
            "bot_group_sizes": [4, 5, 6],
            "content_count": 200,
            "adversarial_strength": 0.60,
        },
        "config_updates": {
            "edge_quantile": 0.87,
            "window_size_minutes": 6,
        },
    },
    "camouflage_high": {
        "description": "High-strength camouflage with heavy lag jitter, target dilution, leader disruption, and aggressive hot-topic overlap.",
        "generator_kwargs": {
            "benign_users": 100,
            "bot_group_sizes": [4, 5, 6],
            "content_count": 220,
            "adversarial_strength": 0.85,
        },
        "config_updates": {
            "edge_quantile": 0.85,
            "window_size_minutes": 6,
            "min_output_risk": 0.52,
        },
    },
}


def build_base_config(seed: int) -> DetectorConfig:
    """Create the default detector configuration for a given seed.

    Args:
        seed: Random seed used for synthetic generation and model components.

    Returns:
        A default :class:`DetectorConfig` instance.
    """
    return DetectorConfig(random_seed=seed)


def build_ablation_variants(base_config: DetectorConfig) -> list[dict[str, object]]:
    """Construct configuration variants for ablation studies.

    Args:
        base_config: Baseline detector configuration.

    Returns:
        A list of dictionaries describing ablation names, descriptions, and the
        corresponding derived configurations.
    """
    similarity = base_config.similarity_weights
    risk = base_config.risk_weights
    return [
        {
            "name": "full",
            "description": "Full detector with all temporal, role, campaign, residual, and MDCS components enabled.",
            "config": base_config,
        },
        {
            "name": "no_leader",
            "description": "Remove leader-follower signals from pairwise edges, GraphSAGE role modeling, and community scoring.",
            "config": base_config.clone_with(
                enable_leader_feature=False,
                graphsage_use_role_features=False,
                similarity_weights={**similarity, "leader": 0.0},
                risk_weights={**risk, "leader": 0.0, "centralization": 0.0},
            ),
        },
        {
            "name": "no_campaign",
            "description": "Remove cross-window campaign consistency from graph edges and risk scoring.",
            "config": base_config.clone_with(
                enable_campaign_feature=False,
                similarity_weights={**similarity, "campaign": 0.0},
                risk_weights={**risk, "campaign": 0.0},
            ),
        },
        {
            "name": "no_residual",
            "description": "Remove residual coordination against the background exposure model.",
            "config": base_config.clone_with(
                enable_residual_feature=False,
                similarity_weights={**similarity, "residual": 0.0},
                risk_weights={**risk, "residual": 0.0},
            ),
        },
        {
            "name": "no_mdcs",
            "description": "Disable MDCS adaptive filtering to measure the contribution of null-model significance control.",
            "config": base_config.clone_with(
                enable_mdcs_filter=False,
            ),
        },
        {
            "name": "graphsage_static",
            "description": "Keep the leader signal on graph edges but remove role-aware node features from GraphSAGE.",
            "config": base_config.clone_with(
                graphsage_use_role_features=False,
            ),
        },
    ]


def generate_events_for_scenario(
    scenario_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Generate synthetic events for a named benchmark scenario.

    Args:
        scenario_name: Scenario identifier defined in :data:`SCENARIOS`.
        seed: Random seed used for synthetic generation.

    Returns:
        A tuple containing the generated event table and the scenario metadata.

    Raises:
        ValueError: If the requested scenario name is unknown.
    """
    if scenario_name not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {known}")

    scenario = SCENARIOS[scenario_name]
    generator_kwargs = dict(scenario["generator_kwargs"])
    events = generate_synthetic_events(seed=seed, **generator_kwargs)
    return events, scenario


def _metrics_row(
    suite_name: str,
    scenario_name: str,
    scenario: dict[str, object],
    variant_name: str,
    variant_description: str,
    seed: int,
    method_name: str,
    result: dict[str, object],
) -> dict[str, object]:
    """Convert one experimental run into a flat metrics row.

    Args:
        suite_name: Name of the experiment suite.
        scenario_name: Scenario identifier.
        scenario: Scenario metadata dictionary.
        variant_name: Ablation or configuration variant name.
        variant_description: Human-readable description of the variant.
        seed: Random seed for the run.
        method_name: Community discovery method name.
        result: Method-specific output dictionary from the pipeline.

    Returns:
        A flat dictionary suitable for tabular aggregation.
    """
    communities = result["communities"]
    metrics = result["metrics"]
    return {
        "suite": suite_name,
        "scenario": scenario_name,
        "scenario_description": str(scenario["description"]),
        "variant": variant_name,
        "variant_description": variant_description,
        "seed": int(seed),
        "method": method_name,
        "community_count": int(len(communities)),
        "mean_support_windows": round(
            float(pd.Series([community.support_windows for community in communities], dtype=float).mean())
            if communities
            else 0.0,
            6,
        ),
        "auroc": float(metrics.get("auroc", float("nan"))),
        "precision_at_k": float(metrics.get("precision_at_k", float("nan"))),
        "recall_at_k": float(metrics.get("recall_at_k", float("nan"))),
        "nmi": float(metrics.get("nmi", float("nan"))),
        "ari": float(metrics.get("ari", float("nan"))),
    }


def aggregate_experiment_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw experiment rows into means and standard deviations.

    Args:
        raw_df: Table of raw method-level experiment results.

    Returns:
        A summary table grouped by suite, scenario, variant, and method.
    """
    group_cols = [
        "suite",
        "scenario",
        "scenario_description",
        "variant",
        "variant_description",
        "method",
    ]
    metric_cols = [
        "community_count",
        "mean_support_windows",
        "auroc",
        "precision_at_k",
        "recall_at_k",
        "nmi",
        "ari",
    ]
    aggregated = (
        raw_df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join([part for part in column if part]).rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in aggregated.columns
    ]
    return aggregated


def compute_ablation_delta(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Compute ablation deltas relative to the full model.

    Args:
        summary_df: Aggregated experiment summary table.

    Returns:
        A copy of the summary table augmented with baseline values and
        per-metric deltas relative to the full configuration.
    """
    metric_cols = [
        "community_count_mean",
        "mean_support_windows_mean",
        "auroc_mean",
        "precision_at_k_mean",
        "recall_at_k_mean",
        "nmi_mean",
        "ari_mean",
    ]
    baseline = summary_df[summary_df["variant"] == "full"].copy()
    baseline = baseline.rename(columns={metric: f"baseline_{metric}" for metric in metric_cols})

    merged = summary_df.merge(
        baseline[
            [
                "suite",
                "scenario",
                "method",
                *[f"baseline_{metric}" for metric in metric_cols],
            ]
        ],
        on=["suite", "scenario", "method"],
        how="left",
    )
    for metric in metric_cols:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"baseline_{metric}"]
    return merged


def build_markdown_report(
    suite_name: str,
    seeds: list[int],
    scenarios: list[str],
    raw_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> str:
    """Render an experiment summary as a Markdown report.

    Args:
        suite_name: Name of the experiment suite.
        seeds: Random seeds included in the suite.
        scenarios: Scenario names included in the suite.
        raw_df: Raw experiment result table.
        summary_df: Aggregated summary table.
        delta_df: Ablation-delta summary table.

    Returns:
        A Markdown string summarizing the experiment outcomes.
    """
    lines = [
        f"# {suite_name.replace('_', ' ').title()} Report",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Scenarios: {', '.join(scenarios)}",
        f"- Total runs: {len(raw_df)} method-level records",
        "",
        "## Mean Metrics",
        "",
    ]

    for method_name in ["graphsage", "louvain"]:
        method_rows = summary_df[summary_df["method"] == method_name].copy()
        if method_rows.empty:
            continue
        lines.extend(
            [
                f"### {method_name}",
                "",
                "| variant | scenario | auroc | precision@k | recall@k | nmi | ari |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in method_rows.sort_values(["variant", "scenario"]).itertuples(index=False):
            lines.append(
                f"| {row.variant} | {row.scenario} | {row.auroc_mean:.4f} | "
                f"{row.precision_at_k_mean:.4f} | {row.recall_at_k_mean:.4f} | "
                f"{row.nmi_mean:.4f} | {row.ari_mean:.4f} |"
            )
        lines.append("")

    lines.extend(["## Ablation Deltas vs Full", ""])
    for method_name in ["graphsage", "louvain"]:
        method_rows = delta_df[(delta_df["method"] == method_name) & (delta_df["variant"] != "full")].copy()
        if method_rows.empty:
            continue
        lines.extend(
            [
                f"### {method_name}",
                "",
                "| variant | scenario | delta auroc | delta nmi | delta ari |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in method_rows.sort_values(["variant", "scenario"]).itertuples(index=False):
            lines.append(
                f"| {row.variant} | {row.scenario} | {row.delta_auroc_mean:+.4f} | "
                f"{row.delta_nmi_mean:+.4f} | {row.delta_ari_mean:+.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def run_experiment_suite(
    suite_name: str,
    output_dir: str,
    seeds: list[int],
    scenario_names: list[str],
    variant_names: list[str] | None = None,
) -> dict[str, object]:
    """Execute a benchmark or ablation suite and export all artifacts.

    Args:
        suite_name: Name of the experiment suite.
        output_dir: Root output directory.
        seeds: Random seeds to evaluate.
        scenario_names: Scenario identifiers to generate and evaluate.
        variant_names: Optional subset of ablation variants to run.

    Returns:
        A dictionary containing raw results, aggregated tables, and output
        artifact paths.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for seed in seeds:
        base_config = build_base_config(seed)
        variants = build_ablation_variants(base_config)
        if variant_names is not None:
            variant_lookup = {variant["name"]: variant for variant in variants}
            variants = [variant_lookup[name] for name in variant_names]

        for scenario_name in scenario_names:
            events, scenario = generate_events_for_scenario(scenario_name, seed)
            scenario_config_updates = dict(scenario["config_updates"])
            for variant in variants:
                variant_name = str(variant["name"])
                variant_description = str(variant["description"])
                variant_config: DetectorConfig = variant["config"]  # type: ignore[assignment]
                config = variant_config.clone_with(**scenario_config_updates, random_seed=seed)
                run_output_dir = root / variant_name / scenario_name / f"seed_{seed}"
                result = run_pipeline(events, config, output_dir=str(run_output_dir))
                for method_name, method_result in result["methods"].items():
                    rows.append(
                        _metrics_row(
                            suite_name=suite_name,
                            scenario_name=scenario_name,
                            scenario=scenario,
                            variant_name=variant_name,
                            variant_description=variant_description,
                            seed=seed,
                            method_name=method_name,
                            result=method_result,
                        )
                    )

    raw_df = pd.DataFrame(rows)
    summary_df = aggregate_experiment_rows(raw_df)
    delta_df = compute_ablation_delta(summary_df)
    markdown_report = build_markdown_report(
        suite_name=suite_name,
        seeds=seeds,
        scenarios=scenario_names,
        raw_df=raw_df,
        summary_df=summary_df,
        delta_df=delta_df,
    )

    raw_path = root / f"{suite_name}_raw_runs.csv"
    summary_path = root / f"{suite_name}_summary.csv"
    delta_path = root / f"{suite_name}_delta.csv"
    report_path = root / f"{suite_name}_report.md"
    meta_path = root / f"{suite_name}_meta.json"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    report_path.write_text(markdown_report, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "suite": suite_name,
                "seeds": seeds,
                "scenarios": scenario_names,
                "variants": variant_names or [variant["name"] for variant in build_ablation_variants(build_base_config(seeds[0]))],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "raw": raw_df,
        "summary": summary_df,
        "delta": delta_df,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "delta_path": str(delta_path),
        "raw_path": str(raw_path),
        "meta_path": str(meta_path),
    }
