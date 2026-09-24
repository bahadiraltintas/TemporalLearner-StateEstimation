# Adaptive AI Framework — Reproducibility Package v1.1

This repository contains the frozen synthetic dataset, analysis scripts, model artifacts, result tables, uncertainty analyses, intervention-value support diagnostics, an independent synthetic potential-outcome benchmark, simulation sensitivity analyses, risk calibration diagnostics, and figures used for the manuscript:

**From Prediction to Adaptation: A Closed-Loop AI Framework for Temporal Learner-State Estimation and Personalized Learning Decision Support**

## Scope

This is a computational proof-of-concept. All learner records, adaptive labels, expected gains, and longitudinal transitions are synthetic or simulated. The package does **not** establish real-world educational effectiveness or causal intervention effects.

The learner-state representation is a computational abstraction, not a validated psychological, pedagogical, or cognitive model.

## Frozen dataset

`data_frozen_v1/synthetic_learner_state_dataset_v1.0.csv`

- 1,000 synthetic learners
- 36 columns
- 28 raw learner-state predictor variables
- 800/200 deterministic random train/test split, seed 42
- SHA-256: `7593b4561a2b72ff3b9753d28dceaffcd53e426bfa434b3381e16cabf7b5fed5`

The original stochastic generator with every distributional parameter was not retained. Therefore the frozen dataset is the authoritative numerical reproducibility artifact; the retained data-generating specification documents the construction logic without inventing missing parameters.

## Main analyses

1. `src/01_prepare_data.py` — frozen-data checks and train/test split.
2. `src/02_leakage_audit.py` — predictor/target leakage audit.
3. `src/03_train_predictive_models.py` — regression/classification benchmarks and sequential ablation.
4. `src/04_intervention_value.py` — action-conditioned Ridge intervention-value model.
5. `src/05A_state_transition_simulation.py` — Simulation A: state-transition adaptation.
6. `src/05_closed_loop_simulation.py` — Simulation B: intervention-value policy comparison.
7. `src/06_interpretability.py` — permutation-importance analyses.
8. `src/07_generate_report_tables.py` — report tables.
9. `src/08_uncertainty_analysis.py` — student-level bootstrap intervals for principal fixed-test metrics.
10. `src/09_intervention_support_and_potential_outcomes.py` — action distribution/support diagnostics, per-action performance, and independent known-potential-outcome benchmark.
11. `src/10_simulation_sensitivity.py` — sensitivity to state-update, spillover, and diminishing-response parameters.
12. `src/11_risk_calibration.py` — risk-label calibration/Brier diagnostics.
13. `src/12_generate_dag.py` — conceptual DAG of the synthetic target-generating structure.

## Intervention-value interpretation

The action-conditioned model uses an explicit categorical activity representation plus activity × numeric learner-state interactions. Five candidate activities are evaluated; `Guided Review` is excluded from the intervention-value candidate set because it has only three observations.

Each learner has only one observed activity/gain pair. There are no observed counterfactual outcomes for unchosen activities. Consequently, intervention-value estimates are predictive conditional scores, not causal treatment effects.

The support analysis reports two operational overlap diagnostics within each action: training min-max support and a 95th-percentile leave-one-out nearest-neighbour distance threshold.

The potential-outcome benchmark is an **independent synthetic environment** with known action-specific potential outcomes. It is a methodological stress test for action-conditioned prediction, not evidence of educational validity.

## Uncertainty

Principal fixed-test metrics use student-level percentile bootstrap resampling with replacement. The bootstrap does not refit models; it quantifies conditional sampling variability on the fixed 200-student test partition. The current package uses 1,000 bootstrap replicates for the computational reproducibility run (`src/08_uncertainty_analysis.py`, seed 42042). Policy contrasts in Simulation A/B use 10,000 student-level paired bootstrap replicates, seed 42.

## Simulations

Both simulations are model-generated environments rather than independent tests of policy efficacy.

- **Simulation A:** state-transition adaptation; the adaptive path recomputes downstream predictions after each simulated topic update, whereas the static path retains the initial activity.
- **Simulation B:** action-conditioned intervention-value policy comparison; adaptive and static policies are compared using the model-defined cumulative score.

Ten cycles were selected as a fixed computational horizon to demonstrate repeated state updates while keeping the simulation within a bounded, explicitly specified synthetic environment. The horizon is not an empirically optimized value.

Sensitivity analysis perturbs the targeted-topic update coefficient (0.28/0.35/0.42), spillover coefficient (0.04/0.05/0.06), and the complete diminishing-response sequence by multipliers of 0.80/1.00/1.20.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify_dataset.py
python RUN_ALL.py
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python verify_dataset.py
python RUN_ALL.py
```

## Repository/DOI

A persistent public repository and DOI should be inserted here after local verification and Zenodo deposition. Until then, the package is intended as the submission-ready reproducibility archive.
