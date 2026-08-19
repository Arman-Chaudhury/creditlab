# Build Plan — creditlab

> Loan-level mortgage PD modeling on real Fannie Mae data, stress-tested through the 2008 crisis

## Milestones

- [ ] **Loader and synthetic fixtures** — Chunked, memory-bounded parser for Fannie Mae acquisition/performance file layouts with schema validation; deterministic synthetic fixture generator mirroring the schema so all tests run offline.
- [ ] **Default labels and vintage cohorts** — Construct D90+ (and D180) default-within-horizon labels from monthly performance records; group loans into origination-vintage cohorts.
- [ ] **Feature engineering** — Origination features (FICO, LTV/CLTV, DTI, loan purpose, occupancy, property type, rate spread at origination), missing-value policy, leakage audit (no post-origination information in features).
- [ ] **Logistic baseline with calibration** — Regularized logistic regression; reliability curves, Brier score, AUC, expected-vs-realized default rate by score decile.
- [ ] **Gradient boosting challenger** — LightGBM model benchmarked against the baseline on identical splits; document the accuracy-vs-interpretability trade.
- [ ] **Crisis out-of-time validation** — Train on pre-2007 vintages, validate on 2007-08 vintages; PSI drift metrics; headline README table showing calibration degradation.
- [ ] **Adverse-action reason codes and fairness audit** — Per-applicant top negative factors (ECOA-style reason codes) and a proxy-fairness audit over the available non-protected attributes.
- [ ] **Tearsheet report, CLI, and CI** — creditlab CLI (ingest/train/validate/explain/tearsheet/demo), self-contained HTML report, offline test suite green in CI, `creditlab demo` reproduces the README numbers.

## Architecture

loader (chunked pipe-delimited parsing of Fannie Mae acquisition/performance files, schema-checked; synthetic fixture generator mirrors the schema) -> labels (D90+ default within horizon, vintage cohorts) -> features (origination: FICO, LTV, DTI, purpose, occupancy, rate spread) -> models (logistic baseline; gradient boosting) -> validation (vintage-based out-of-time split incl. 2007-08 crisis vintages; AUC, Brier, reliability curves, PSI drift) -> explainability (adverse-action reason codes, proxy fairness audit) -> tearsheet (HTML/Markdown report; `creditlab demo` reproduces README numbers offline)
