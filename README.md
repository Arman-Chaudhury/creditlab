# creditlab

**Loan-level mortgage credit risk modeling on real GSE data — stress-tested through the 2008 crisis.**

Most student credit-risk projects train on toy Kaggle CSVs. creditlab trains
probability-of-default (PD) models on the public **Fannie Mae Single-Family Loan
Performance dataset** — millions of real mortgages with origination features and
month-by-month payment outcomes — and asks the question that actually matters in
credit risk: *does a model trained in good times survive a downturn?*

The flagship experiment: train on pre-2007 vintages, validate out-of-time on the
2007–2008 crisis vintages, and report exactly how calibration degrades. That is
the failure mode that broke real mortgage models in 2008, reproduced and
measured on the real data.

## Why this project

- **Real data, real scale.** Loan-level GSE data, not `german_credit.csv`.
  Chunked, memory-bounded parsing of the pipe-delimited performance files.
- **Vintage-based out-of-time validation.** Random train/test splits leak the
  macro environment. creditlab splits by origination vintage, the way model
  risk teams actually validate.
- **Calibration over accuracy.** A PD model's job is honest probabilities:
  reliability curves, Brier score, and expected-vs-realized default rates by
  decile — not just AUC.
- **Regulatory realism.** Adverse-action reason codes (top negative factors per
  applicant, ECOA-style) and a proxy-fairness audit are first-class outputs,
  not afterthoughts.
- **Offline-first.** A deterministic synthetic-fixture generator mirrors the
  Fannie Mae schema so the full test suite runs with zero downloaded data.

## Architecture

```mermaid
flowchart LR
    A[Fannie Mae loan
performance files] --> B[loader
chunked, schema-checked]
    S[synthetic fixture
generator] --> B
    B --> C[labels
D90+ default within horizon,
vintage cohorts]
    C --> D[features
origination: FICO, LTV, DTI,
purpose, occupancy, rate spread]
    D --> E1[logistic baseline]
    D --> E2[gradient boosting]
    E1 --> F[validation
vintage OOT split, AUC,
Brier, reliability, PSI drift]
    E2 --> F
    F --> G[explainability
adverse-action reason codes,
proxy fairness audit]
    G --> H[tearsheet report
HTML/Markdown + CLI demo]
```

## The headline experiment

`creditlab.validation.crisis_validation` runs the same trainer through both
protocols — the naive random split and the honest vintage-based out-of-time
split (train ≤ 2006, validate on 2007–08 originations) — and reports them side
by side. On the bundled synthetic fixtures (800 loans, seed 99; **not real
data** — the real-data table lands with `creditlab demo` once you've
downloaded a Fannie Mae quarter):

| metric | random split (naive) | OOT 2007–2008 vintages (honest) |
|---|---|---|
| realized default rate | 0.1050 | 0.1619 |
| mean predicted PD | 0.1309 | 0.1237 |
| realized / predicted | 0.80 | **1.31** |
| AUC | 0.8742 | 0.8255 |
| Brier skill vs climatology | 0.230 | 0.245 |

The pattern that broke real mortgage models in 2008, reproduced end to end:
**discrimination survives (AUC barely moves) while calibration breaks** — on
unseen crisis vintages the model sees only ~76% of the risk coming
(realized/predicted 1.31), and the naive split shows no warning at all (0.80).
Per-feature PSI attributes the drift to the rate environment (orig_rate 0.20,
rate_spread 0.13) rather than borrower quality (fico 0.04) — the borrowers
didn't get worse; the world did.

## Data access

The Fannie Mae Single-Family Loan Performance data is free but requires
registration at [Fannie Mae Data Dynamics](https://capitalmarkets.fanniemae.com/tools-applications/data-dynamics).
Nothing in this repo redistributes the data. All tests run on the bundled
synthetic fixtures; `creditlab demo` reproduces the README numbers from a
downloaded acquisition/performance file pair.

## Planned CLI

```
creditlab ingest <acq_file> <perf_file>   # parse + label + cache
creditlab train --model logit|gbm         # fit with vintage-aware split
creditlab validate --oot 2007,2008        # crisis stress validation
creditlab explain <loan_id>               # adverse-action reason codes
creditlab tearsheet                       # full HTML report
creditlab demo                            # end-to-end on synthetic data
```

## Relationship to factorlab

[factorlab](https://github.com/Arman-Chaudhury/factorlab) covers market risk
(cross-sectional factor research); creditlab covers credit risk (loan-level PD).
Same philosophy: real data, statistical honesty, walk-forward/out-of-time
validation, reproducible demo numbers.

---

> **Status: scaffolded, not complete.** This repo was scaffolded from a build spec; see `BUILD_PLAN.md` for the milestones.
