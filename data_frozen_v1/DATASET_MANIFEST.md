# Frozen Synthetic Dataset Manifest — v1.0

**Dataset:** `synthetic_learner_state_dataset_v1.0.csv`

## Identity

- Rows: 1,000
- Columns: 36
- Learner-state predictor variables: 28
- Missing cells: 0
- Duplicate rows: 0
- Duplicate `student_id`: 0
- Student IDs: 10001–11000
- SHA-256: `7593b4561a2b72ff3b9753d28dceaffcd53e426bfa434b3381e16cabf7b5fed5`

## Provenance

This file is the frozen synthetic dataset used for the computational proof-of-concept. It is the authoritative version-1.0 data artifact for reproduction of the reported experiments.

The original stochastic generator with every distributional parameter and random draw is not retained in the final archive. The dataset itself therefore should be treated as the reproducibility artifact; no missing generator parameters should be inferred or reconstructed from the resulting values.

## Interpretation

The dataset is synthetic and contains rule-/simulation-derived targets. In particular, weakest-topic, recommended activity, expected learning gain, and longitudinal updates should not be interpreted as observations from real learners.
