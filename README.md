# Temporal Role-Aware CIB Detector

This repository contains an academic prototype for coordinated inauthentic
behavior detection. The core pipeline follows the sequence:

`behavioral representation -> temporal coordination graph -> leader-follower role modeling -> Louvain / role-aware GraphSAGE -> risk scoring -> MDCS filtering`

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick Start

Run the detector on the built-in synthetic benchmark:

```bash
python main.py --output-dir outputs
```

Run the detector on a custom event log:

```bash
python main.py --input-csv your_events.csv --output-dir outputs
```

## Input Schema

Required columns:

- `user_id`
- `content_id`
- `action_type`
- `timestamp`

Optional columns:

- `dwell_time`
- `session_id`
- `is_bot`
- `true_group`
- `role_tag`

## Core Signals

- short-window temporal synchronization
- inverse-popularity weighted target overlap
- behavioral sequence similarity via dynamic time warping
- cross-window campaign consistency
- leader-follower lag and role consistency
- community density and leadership centralization

## Exported Outputs

- `outputs/communities.json`: final community detections
- `outputs/community_report.csv`: tabular community summary
- `outputs/account_scores.csv`: account-level risk scores
- `outputs/window_stats.csv`: per-window graph statistics and MDCS thresholds
- `outputs/metrics.json`: AUROC, Precision@k, Recall@k, NMI, and ARI

## Research-Oriented Features

- a role-aware GraphSAGE path rather than a purely static graph embedding
- interpretable community attributes such as `top_leaders`, `leader_score`,
  and `centralization_score`
- synthetic benchmark generation with adversarial camouflage controls
- ablation and benchmark runners for multi-seed evaluation

## Experiment Scripts

Smoke-test the experiment scripts:

```bash
python experiments/run_ablation.py --quick
python experiments/run_benchmark.py --quick
```

Run the full ablation suite:

```bash
python experiments/run_ablation.py --output-dir experiment_outputs/ablation
```

Run the full benchmark suite:

```bash
python experiments/run_benchmark.py --output-dir experiment_outputs/benchmark
```

The experiment runners export:

- `*_raw_runs.csv`: raw method-level metrics for every run
- `*_summary.csv`: aggregated means and standard deviations
- `*_delta.csv`: ablation deltas relative to the full model
- `*_report.md`: Markdown-ready experiment summaries
