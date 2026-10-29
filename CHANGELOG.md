# Changelog

## 1.0.0 (2026-10-30)

First public release on PyPI.

- **Any input, any size**: pandas, polars, pyarrow, dask, NumPy objects and files (CSV / TSV, JSON, Parquet,
  Feather, ORC, Excel, ODS, SQLite, SAS, Stata, SPSS, HDF5, pickle, XML, compressed or not, globs, folders, URLs).
  Tables bigger than memory are typed progressively on as few rows as the evidence allows, then read once in
  parallel chunks (dask for text files) with exact counts over every row; time series are never cut into
  scattered rows.
- **Semantic typing by an explicit decision tree**, with the path and the certainty (certain, probable,
  uncertain) of every decision: identifiers, booleans in disguise, numbers and dates stored as text in any
  format, emails, URLs, phones, postal codes, JSON, lists of labels, free text, ordinal scales, counts, and
  integers that are really category codes (region 11, store 7, MCC 5411). Types can be set by hand
  (`StoneConfig(column_types=...)`, `--type COL=TYPE`).
- **Dates of birth** told apart from event dates and turned into ages at the right reference date, with their
  own checks.
- **Statistics, distributions and associations**: descriptive statistics with confidence intervals, MLE fits
  ranked by AIC, normality tests, every pairwise association with tests, Benjamini-Hochberg correction and
  confidence intervals, VIF, exact linear dependencies, missingness mechanisms.
- **Target analysis and feature selection**: ranking, leakage detection, informative missingness, five
  cross-validated selection methods with a vote.
- **Multivariate statistics**: PCA verdict (Bartlett, KMO, parallel analysis), a factor analysis run step by step
  (KMO and Bartlett, communalities, number of factors, varimax, cross-loadings, Cronbach's alpha, scale scores),
  redundant groups, robust multivariate outliers.
- **Outliers** by three methods, point by point, with keep / remove / reassign.
- **Time series** recognised on their own (single series, panel, event log): trend with the Hamed-Rao
  correction, seasonality by effect size, ADF and KPSS read together, decomposition, validation folds on the
  real dates.
- **Data quality alerts and a preparation plan**, every step with its reason and its code.
- **Report**: a terminal report, an HTML page in a dark "data observatory" design (sidebar, key figures,
  numbered sections, charts that open full size) and its Markdown twin, report-quality charts, in a dated folder
  per table and target.
- Tested on Linux, macOS and Windows, Python 3.10 to 3.13, on 55 public datasets and 17 reference datasets for
  the factor analysis.
