# Elessar 1.2.0rc1 — independent real-data and meta-statistical audit

**Audit date:** 2026-11-03. **Baseline wheel:** `elessar-1.2.0rc1` from GitHub Actions commit
`36e35504c5745a173f82b8b958918527f7afeae2`. Time-index support was then
amended on the PR and confirmed locally; the *latest* PR head needs its own
successful CI and statistical validation before release.

## Real-world datasets and report integrity

An optional local validation (using the datasets bundled with `statsmodels`,
which is **not** a runtime dependency of Elessar) covered 14 distinct datasets
and 3,474 total input rows across 77 column appearances:

- US election survey (ANES 1996), Fair affairs survey, Engel household food
  spending, Scottish local elections, US state crime, education (Spector),
  industrial Stackloss, cancer counts vs population, and RAND healthcare.
- US quarterly macroeconomic records, the Grunfeld firm panel, annual sunspots,
  weekly atmospheric CO2, and annual Nile flow.

**Observed:** 14/14 HTML and Markdown reports generated, 277 figure references,
no broken image references or internal anchors, 61 numeric columns cross-checked
for descriptive moments, and zero reference discrepancies at the preselected
numerical tolerances. All columns were also checked for nonmissing counts,
percent missing and distinct counts. The reports generated 88 alerts. **That is
an alert count, not an estimated detection sensitivity or 88 known defects.**

After the date-index fix, all five intended time-structured cases were
recognized, including DatetimeIndex and PeriodIndex-only sources. The fix
does not inject a pseudo-column into the data. Seven dedicated regression tests
passed locally and are part of this PR.

## Statistical reference agreement

Ten actual numeric column-pairs across the reference datasets were compared
to `scipy.stats`: 10 Pearson, 10 Spearman and 10 Kendall statistics, including
their corresponding p-values.

- **30/30** effects agree, maximum absolute difference `1.3323e-15`.
- **30/30** p-values agree, maximum absolute difference `3.7470e-16`.

These calculations agree with SciPy on those records. Agreement does **not**
establish that IID-based p-values are appropriate for every real observational
sampling structure; time and panel inferential assumptions still apply.

## Meta-statistics of the statistical outputs

All simulation outcomes below use prescribed generators and an observed
real-data predictor distribution. Wilson confidence intervals describe the
finite Monte Carlo measurement precision; they are not universal coverage or
calibration guarantees.

| Experiment | Trials | Observed | 95% Wilson interval | Interpretation |
|---|---:|---:|---:|---|
| Engel food expenditure independently permuted relative to income (Pearson) | 1,000 | 36 rejections: 3.6% | 2.61–4.94% | Null checks against nominal 5% |
| Known linear signal on the empirical Engel income distribution plus Gaussian noise | 500 | 500 detections: 100% | 99.24–100% | Strong-signal detection only |
| Independent stationary AR(1), rho=0.83, fixed Kendall time-shift test | 75 | 2 rejections: 2.67% | 0.73–9.21% | Small-scale time-series null check |
| The *same* AR(1) cases, naïve IID Kendall | 75 | 25 rejections: 33.33% | — | Demonstrates inflation when serial dependence is ignored |
| Strong linked AR(1) signal plus independent noise | 75 | 75 detections | — | Strong-signal benchmark |

The time-shift test's strict stationarity / process-independence assumptions
are satisfied *by the simulator*, not established automatically for arbitrary
observations. The 75 replicates are not sufficient to measure rare-tail
miscalibration precisely; the project's larger predeclared validation suites
remain the primary inferential release gate.

## Permanent regression checks

- `tests/test_datetime_index_v12.py` — seven date/period-index, ordinary-index,
  datetime-column precedence, missingness and HTML/snippet checks.
- `tests/test_public_datasets_additional_v12.py` — three offline public
  scikit-learn dataset report/moment checks (Breast Cancer, Diabetes, Digits)
  and three Pearson/Spearman/Kendall comparisons against SciPy.

These 13 additions passed locally. They must also pass the exact current-head
GitHub Actions runs; do not substitute an earlier run.

## Known limits

Real datasets do not supply ground-truth labels for all leakage, unobserved
errors, causal relations, distribution families, or anomalous values. Therefore
neither 14/14 report generation nor the successful numerical oracles certify
that Elessar detects all real-world anomalies. False-positive and statistical
power estimates apply only to the explicitly simulated designs.

The full optional audit scripts, per-dataset CSV, exact reference comparisons,
Monte Carlo JSON and plots are available separately as the Elessar v1.2
real-data meta-validation audit bundle shared with the reviewer.
