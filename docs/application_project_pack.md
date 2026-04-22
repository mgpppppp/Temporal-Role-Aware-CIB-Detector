# Overseas Application Project Pack

## Project Title

Recommended title:

`Temporal Role-Aware Graph Learning for Coordinated Inauthentic Behavior Detection`

Alternative titles:

- `Detecting Coordinated Manipulation via Temporal Role Modeling and Graph Representation Learning`
- `Interpretable Detection of Coordinated Inauthentic Behavior with Role-Aware Temporal Graphs`

## Project Positioning

This project should be presented as a research-oriented graph learning system
rather than a generic anomaly detection implementation. Its main strength is
that it reframes the task from isolated account scoring to structural recovery
of coordinated manipulation networks.

The project is particularly well aligned with:

- graph machine learning
- trustworthy and interpretable AI
- computational social systems
- platform integrity and trust-and-safety research

## Core Research Story

Many harmful online campaigns are not driven by a single obviously malicious
account. Instead, they emerge through groups of accounts that coordinate in
timing, target selection, and latent role division. This project addresses that
problem by converting behavioral logs into temporal coordination graphs and
explicitly modeling leader-follower dynamics, campaign persistence, and
community-level structural risk.

The methodological story can be summarized as:

`event logs -> temporal coordination graph -> role discovery -> community mining -> interpretable risk scoring -> reproducible evaluation`

## Main Contributions

### 1. Temporal Coordination Graph Construction

The system transforms raw event logs into sliding-window user graphs using
temporal synchronization, target overlap, behavioral sequence similarity, and
session rhythm statistics. This step turns unstructured behavioral logs into a
relational representation suitable for graph analysis.

### 2. Leader-Follower Role Modeling

The project introduces directional role modeling through initiation lag,
directional dominance, role consistency, and leadership centralization. This
allows the detector to move beyond pairwise similarity and reason about
initiators and amplifiers inside coordinated groups.

### 3. Dual-Path Detection Framework

The detector compares a classical graph-mining baseline (`Louvain`) with a
learned representation pipeline (`role-aware GraphSAGE`). This design makes the
project stronger academically because it emphasizes method comparison rather
than simply adopting a neural model by default.

### 4. Reproducible Experimental Pipeline

The repository includes a synthetic benchmark generator, ablation study
configuration, multi-seed experiment runners, and standardized result exports.
This turns the project into a research prototype rather than a one-off demo.

## Adversarial Evaluation Narrative

The benchmark includes layered adversarial camouflage settings in which bot
groups intentionally:

- reduce synchronization strength
- increase lag jitter
- dilute shared targets
- disrupt leader-follower structure
- mimic benign session rhythms
- overlap with trending content to create organic-looking exposure

These settings are useful for application materials because they show that the
project does not only perform well on clean synthetic data; it also studies how
the detector degrades under more realistic disguise strategies.

## Representative Experimental Result

In the adversarial benchmark, both the Louvain baseline and the role-aware
GraphSAGE detector degrade as camouflage strength increases. The most important
observation is that community recovery deteriorates more sharply than
account-level ranking, indicating that recovering coordinated structure is more
challenging than identifying suspicious accounts in aggregate.

A concise research-style interpretation is:

> Under adversarial camouflage that weakens synchronization, dilutes shared
> targets, disrupts leader-follower roles, and increases overlap with trending
> content, both detectors degrade substantially, with clustering quality
> dropping more sharply than account-level ranking.

## Why This Project Is Strong for Graduate Applications

- It starts from a concrete and societally meaningful problem.
- It combines temporal modeling, graph learning, interpretability, and
  experimental methodology.
- It includes both a baseline and a learned method.
- It contains ablations, benchmarks, and multi-seed evaluation.
- It demonstrates a full path from hypothesis formation to validation.

## Resume Version (English)

### Concise Version

Temporal Role-Aware Graph Learning for Coordinated Inauthentic Behavior
Detection

- Built an end-to-end detection pipeline for coordinated inauthentic behavior
  from behavioral event logs, including sliding-window graph construction,
  community discovery, risk scoring, and significance-aware filtering.
- Designed a leader-follower temporal modeling module that captures initiation
  lag, directional influence, role consistency, and leadership centralization
  to identify orchestrators and amplifiers within suspicious groups.
- Implemented a role-aware GraphSAGE detector alongside a Louvain baseline and
  developed reproducible ablation and benchmark runners across multiple seeds
  and adversarial camouflage scenarios.

### Research-Oriented Version

- Proposed a graph-based framework for coordinated inauthentic behavior
  detection by integrating temporal synchronization, weighted target overlap,
  behavioral sequence similarity, campaign persistence, and leader-follower
  dynamics.
- Developed a role-aware GraphSAGE model that injects burstiness, content
  focus, action entropy, and role statistics into node representations.
- Built a reproducible experimental pipeline with synthetic benchmarks,
  ablation studies, and adversarial camouflage evaluation using AUROC,
  Precision@k, Recall@k, NMI, and ARI.

## Statement-of-Purpose Paragraph

Among my recent projects, this one best represents my research interests
because it began with a structural question rather than a model choice. Many
platform integrity risks are not caused by a single abnormal account, but by
groups of accounts coordinating through shared timing, shared targets, and
latent role division. To study this problem, I built a temporal role-aware
graph learning system that converts behavioral logs into sliding-window
coordination graphs and explicitly models leader-follower dynamics, community
risk, and structural roles. More importantly, I validated these ideas through
ablations and adversarial camouflage benchmarks instead of relying on a single
best-case run. This project strengthened my interest in graph machine
learning, trustworthy AI, and computational social systems, especially in
settings where predictive performance and interpretability must be developed
together.

## Interview Talking Points

### Main Contribution

My main contribution was not simply applying a graph neural network. I
reframed the task as temporal role discovery inside coordinated networks and
translated that framing into both interpretable graph features and reproducible
experiments.

### Most Difficult Technical Challenge

The hardest part was not training the model itself, but constructing graph
edges that faithfully represented coordinated behavior. If the edge definition
is weak, both community mining and graph representation learning become
unreliable.

### Why Keep Both Louvain and GraphSAGE

I kept Louvain as a strong baseline so that the project would focus on whether
the structural signals were meaningful, rather than on whether a more complex
model merely looked more sophisticated.

## Limitations

- The current evaluation still relies primarily on synthetic benchmarks.
- The GraphSAGE component remains a lightweight prototype rather than a full
  temporal or heterogeneous graph neural architecture.
- Interpretability is currently delivered through exported reports rather than
  an analyst-facing visualization interface.

## Future Work

- Extend the graph into a heterogeneous setting that jointly models users,
  content, sessions, and time.
- Explore temporal GNNs or contrastive graph learning for stronger
  cross-scenario generalization.
- Add case-level visualization and domain-transfer experiments on data that
  better approximates real platform conditions.
