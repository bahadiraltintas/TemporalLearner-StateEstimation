# Local reproduction — reviewer revision v1.1

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verify the frozen dataset

```bash
python verify_dataset.py
```

Expected SHA-256:

`7593b4561a2b72ff3b9753d28dceaffcd53e426bfa434b3381e16cabf7b5fed5`

## Full run

```bash
python RUN_ALL.py
```

The full run executes predictive benchmarks, both closed-loop simulations, uncertainty analysis, action-support diagnostics, the independent synthetic potential-outcome benchmark, simulation sensitivity, risk calibration, and DAG generation.

## Manuscript

The manuscript source is:

`manuscript/adaptive_ai_cas_sc_reviewer_revision_v5.tex`

The precompiled PDF is:

`manuscript/adaptive_ai_cas_sc_reviewer_revision_v5.pdf`

CAS compilation uses BibTeX8:

```bash
cd manuscript
pdflatex adaptive_ai_cas_sc_reviewer_revision_v5.tex
bibtex8 adaptive_ai_cas_sc_reviewer_revision_v5
pdflatex adaptive_ai_cas_sc_reviewer_revision_v5.tex
pdflatex adaptive_ai_cas_sc_reviewer_revision_v5.tex
```

A public GitHub/Zenodo DOI should be inserted into the manuscript after local verification and repository deposition.
