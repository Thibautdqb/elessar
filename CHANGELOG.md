# Changelog

## 1.1.0 (2026-10-30)

### Statistical fixes

- **Mutual information**: equal values now always share a bin. Ties were ranked by row position, so a constant
  column in a file sorted by the target scored 3.3 bits instead of 0.
- **Pearson or Spearman** for a pair of numeric columns is now chosen from the shape of each column (Pearson
  when both have |skewness| < 1 and |excess kurtosis| < 3, new settings `pearson_max_skew` and
  `pearson_max_kurtosis`), never from which coefficient is larger. Keeping the stronger of the two after the
  fact was two tests in one: at 5% it rejected 6.5% of independent normal pairs and 9.5% of log-normal ones
  (simulation, n = 200, 4,000 runs); the new rule stays at 4.5 to 5.4% and keeps the power of Pearson on
  symmetric data and of Spearman on skewed data. `pairs` gains `other_value` (the other coefficient, also
  shown in the tables) and `method_reason`, and `numeric_shape` gives the skewness and kurtosis of each column.
  On real data the pairs that are no longer significant are tiny effects (|r| < 0.09), such as Pclass and
  SibSp on Titanic, whose Pearson +0.08 came from a few large families (Spearman -0.04).
- **Feature selection on dated rows is validated on their future**, whatever the order of the file: the rows
  are sorted in time (series, panel or event log), the folds are cut between two dates (rows of one date
  are never on both sides, which matters for panels) and the scores come from the most recent rows. An
  unsorted file used to be cross-validated with shuffled folds: two independent random walks scored an R²
  of +0.38 (neighbouring days on both sides of the split); the sorted and unsorted files now give the same,
  negative, scores.
- **Selection scores are measured on held-out rows**: 20% of the rows (the most recent for dated rows) are
  set aside first, every model is tuned by cross-validation on the others, with imputation and scaling
  learnt there too, and scored once on the held-out rows. The best setting used to be scored on the folds
  that chose it (AUC 0.517 on pure noise, now 0.49 to 0.50). `selection` gains `validation` (how the
  scores were measured), `n_train` and `n_holdout`. On a small table the held-out score varies more than a
  5-fold average.
- **Steps learnt from the target** in the preparation plan (supervised bins, category groups, target
  encoding) now say that they must be learnt again on the training rows only for a model.
- **Number of factors set by parallel analysis** (Horn) in the factor analysis, step 5: on 300 simulated
  structures it finds the right number in 98% of them, against 94% for the former rule (two of eigenvalue
  above 1, 60% of the variance and the scree elbow agreeing, else the scree elbow), which overestimated it in
  6%: Kaiser and the 60% target overestimate it alone in 53% and 93% of the structures and often agree on the
  same wrong number. The three classic criteria are still shown beside it, the scree plot draws the threshold
  of random data, and the reference datasets keep their known structure (Holzinger-Swineford: 3 factors on 9
  tests and 4 on 24, Big Five questionnaire: 5, found at once instead of after removing an item).
- **Two-item factors are judged on the Spearman-Brown coefficient** 2r / (1 + r) instead of Cronbach's
  alpha, which understates the reliability of two items that vary unequally (Eisinga, te Grotenhuis &
  Pelzer, 2013): such a factor could be refused a score with an alpha of .57 and a Spearman-Brown of .71.
  Alpha is still shown beside it.
- **Mann-Kendall trend test under autocorrelation**: the variance of S is inflated by the larger of two
  models of the autocorrelation, an AR(1) corrected for its small-sample bias and an AR(p) chosen by AIC. The
  textbook Hamed-Rao correction keeps only the significant autocorrelations, which are biased low: on 200
  points without trend it reported a trend in 15% of AR(1) 0.5 series and 25% of AR(1) 0.8 ones at 5%, now
  6% and 6% (2 to 11% on AR(2), MA(1), ARMA(1,1) noise, against 17 to 58% without any correction). Long
  memory is still mistaken for a trend in about a quarter of the series: documented as a limit.
- **Associations in a table indexed by time** (a series or a panel, sorted or not): the columns that ADF and
  KPSS do not show stationary are replaced by their first differences (within each entity of a panel), and
  every test (numeric pairs, categorical pairs, mixed pairs, links with the target) uses Bartlett's effective
  sample size n (1 - r₁r₂) / (1 + r₁r₂) from the lag-1 autocorrelations of the two columns. Treated as
  independent rows, two independent AR(1) 0.9 series were declared associated in about half of the
  simulations and two random walks in 86%; now 3 to 6%, while real links between series are still found
  (87 to 91%). `see_corr` gains `serial` (autocorrelation and differencing of each column) and `n_effective`.
  When the target of a feature selection is not stationary, its changes are modelled from the changes of
  the other non-stationary columns (`selection["on_differences"]`): a random walk used to be kept as a
  feature of another one in 9 runs out of 10, now in none of 10.
- **U-shaped links with a numeric target**: the test is now Spearman together with a Kruskal-Wallis of the
  target across the deciles of the feature, two tests fixed in advance combined by Bonferroni (type I error
  4%). Spearman alone saw only monotone links: a U-shaped effect was found in 5% of the simulations, now
  97%, at the cost of some power on a weak linear link (69% to 61%). In feature selection the U-shaped effect
  is kept in 90% of the runs (a third before).
- **Feature selection against shuffled shadows** (Boruta): the decision tree, the lasso and ridge are
  refitted with each feature also present with its rows shuffled, and vote for a feature only when it beats
  every shadow. Their former fixed thresholds (half an equal share, any non-zero coefficient, 10% of the
  largest) let 28% of pure-noise features be kept with a binary target, now 2%; real signals are still kept.
- **Outliers advised for removal**: the "more than 10 robust standard deviations" test is made on the log
  scale for a positive, right-skewed column, and removal is advised only when the rest of the column has
  light tails (excess kurtosis below 1 plus three standard errors). Clean columns were advised to drop rows
  in 70% of the simulations for Student t3, 17% for an exponential, 12% for Student t5 and 7% for a
  log-normal, now 0 to 1%; values multiplied by 100 in a light-tailed column are still all found and removed,
  and in a heavy-tailed column they are flagged with the advice to check them.
- **Rows with missing values** on a table profiled from a sample: the number of rows with at least one missing
  cell (overview, KPI card) and the alert on rows with mostly missing cells are now counted over every row
  during the full pass (they were the sample's counts: 150 shown for 851 real). The overview and missing-values
  charts say when their figures are the sample's.

### Reading files

- **CSV lines with more fields than the header** are skipped, counted and reported by a high alert that gives
  their line numbers. A small file used to lose them with only a pandas warning in the console, and a big one
  read in chunks or with dask stopped with a `ParserError`. Each chunk counts its own skipped lines, so the
  count is exact on every reading path (whole file, chunks, dask blocks, parallel processes); dask blocks
  cannot number the lines in the file, so their alert gives only the count.

- **Web files (http, https, ftp) are streamed**: their size comes from the server (`Content-Length`), the
  delimiter, header and size estimate read only the head, and the body is read once, in chunks when the table
  is big or its size unknown. pandas downloads a URL whole into memory before reading it, and elessar did so
  four times: a 55 MB file peaked at 471 MB of memory, now 316 MB. Compressed files (gz, bz2, xz, zst) are
  decompressed as they arrive. A URL that returns a web page (a GitHub `blob/` link instead of the raw file)
  or an HTTP error now says so instead of being profiled as a table.
- **Feather and ORC files are read in chunks** (record batches, stripes) when they are big, with their row
  count from the metadata: they used to be read whole whatever their size.
- **Formats that can only be read whole** (Excel, JSON, XML, pickle, HDF5, SPSS) are checked before reading:
  the memory reading takes is estimated from the file size (measured: Excel about 15 times the file, JSON
  10 times) and, beyond the memory available, elessar stops with advice (convert to Parquet or CSV) instead
  of being killed by the system.
- **Pickles are read only when named**: a folder or a glob pattern no longer loads the pickle files it finds
  (loading a pickle runs any code it holds); a pattern that matches only pickles says to pass the file path.
  A pickle from a URL is rebuilt by a restricted unpickler that knows only the classes and functions that
  rebuild pandas and NumPy objects (pandas 2.2 to 3.x), with every Arrow array validated in full; anything
  else is refused with the advice to download the file and check it.
  A pattern now keeps a single format, like a folder (it used to read a `.pkl` next to a CSV as CSV).
- **SQLite table names** with double quotes are quoted correctly: the biggest table used to be skipped in
  favour of a smaller one, and naming it failed with a syntax error. A `table=` that does not exist gives
  the list of tables.

### Statistical validation

- **`tests/validation`**: 69 checks against a known truth (simulations and public datasets), run apart from
  the unit tests (`python -m pytest -m validation tests/validation`, about 5 minutes; monthly and on demand in
  `.github/workflows/validation.yml`): type I error of every association test on six distributions, false
  discovery rate of Benjamini-Hochberg, coverage of the correlation intervals, mutual information without
  information, associations of autocorrelated and non-stationary columns of dated tables, U-shaped links with
  a target, feature selection on noise, on known signals and on shuffled dated rows, the number of factors
  on 300 simulated structures and on public reference datasets (Holzinger-Swineford, Big Five questionnaire),
  outliers on clean data and with planted errors, seasonality, trend under seven noise structures,
  stationarity verdicts, and ten public datasets (counts against pandas, known facts).
  `ELESSAR_VALIDATION_OUT` writes the measures to a JSON file; the current ones are in
  `docs/validation-results.json` and explained in `VALIDATION.md`.

### Smaller fixes

- `column_outliers` works on a series whose row labels repeat (it used to fail on a label lookup).
- Continuous integration runs `ruff check` (real errors only: undefined names, likely bugs, syntax), configured
  in `pyproject.toml`.

### Preparation code

Every snippet of the preparation plan and of the time-series treatment now runs in plan order on the user's
DataFrame (148 snippets on 8 public datasets: 0 failures, 3 that need a value from the user; tested in CI on
synthetic traps).

- **Box-Cox** is fitted on the observed values and then applied: on a column with missing values it used to
  return NaN everywhere.
- **Order of the steps**: power transforms run before rescaling (Box-Cox needs positive, unscaled values), a
  column is never scaled twice, and a column dropped as mostly missing is no longer encoded or scaled afterwards.
- **Column names** changed by the analysis (non-string, duplicated) are announced by an alert and by a first
  step that renames the DataFrame's columns, so the later snippets find them.
- Smaller fixes: `99,999` read as a number instead of a tuple; `fillna('missing')` on a category column;
  whitespace stripped before the date conversion; outliers of numbers stored as text are parsed first; dummy
  columns prefixed `col=` so they cannot clash with `col_missing`.
- **Time series**: the treatment code names the analysed column (it used a `<value>` placeholder), defines the
  array used by the augmentation, turns a year column into dates before resampling, aggregates an event log
  even without numeric columns, and computes the lags and rolling means of a panel within each entity (a
  global shift handed one entity's last value to the next one).

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
