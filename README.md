<p align="center">
  <img src="docs/logo.png" alt="elessar logo: a marble god-king in sunglasses, wearing an emerald eagle brooch, codes Python on a laptop beside a panda, under a giant green gem in a neon jungle" width="100%">
</p>

**The Elfstone for pandas: see everything about a DataFrame, and how to heal it.**

*In Tolkien's legend, the Elessar is the green Elfstone through which whatever is seen is made whole again.*

<p align="center">
  <img src="docs/terminal.svg" alt="elessar analysing the Titanic dataset in a terminal: install, run, then the report scrolls by (overview, alerts, column summary, correlations, target analysis)" width="766">
</p>

Pass a `DataFrame` (or a `Series`) to `see()`, or a file to the `elessar` command, and get in your terminal:

- the **semantic type** of every column (not just the dtype): id, boolean in disguise, numbers stored as text, dates stored as text, emails, URLs, phones, postal codes, IPs, JSON, free text, ordinal scales (low/medium/high, months, weekdays, 1-5 ratings), counts, mixed types…
- **full descriptive statistics** adapted to each type, with confidence intervals
- the **best-fitting statistical distribution** (MLE, ranked by AIC) and four normality tests
- **associations of every kind with statistical significance**: Pearson, Spearman, Kendall, Cramér's V, correlation ratio, mutual information, AUC, with p-values, Benjamini-Hochberg correction and confidence intervals
- **target analysis**: feature ranking, **leakage detection**, informative missingness
- **data quality** checks and **actionable alerts** (with the code to fix them): placeholders, duplicates, multicollinearity, values missing not at random, sorted rows…
- a set of **plots that actually carry information**, saved as PNG files

It is built for scripts and the terminal: plain-text output that fits your terminal width, PNG files on disk, and no display needed. It only depends on `pandas`, `numpy`, `scipy` and `matplotlib`.

## Install

```bash
pip install git+https://github.com/thibautdqb/elessar
```

Requires Python ≥ 3.10.

## Quickstart

```python
import pandas as pd
import elessar

df = pd.read_csv("titanic.csv")

report = elessar.see(df, target="Survived")    # prints the report, writes PNGs to ./elessar_report/

report.alerts                                # [{severity, column, message}, ...]
report.summary                               # one row per column, one column per metric
report.correlations["target"]                # features ranked, with tests, p-values, AUC, MI, leakage flag
report.to_markdown("elessar_report/report.md") # full report with the plots embedded
```

Or straight from the shell:

```bash
elessar titanic.csv --target Survived --markdown --json
```

Run `python examples/demo.py` to see it on a deliberately messy synthetic dataset.

## The see functions

| Function | Returns | What it does |
|---|---|---|
| `see(df, target=None, config=None, plots=True, out_dir="elessar_report", verbose=True)` | `Vision` | Runs everything below, prints the terminal report, writes the plots. |
| `see_column(series, config=None)` | `dict` | Everything about one column: type, stats, distribution, alerts. |
| `see_types(df)` | `DataFrame` | Semantic type, pandas dtype, confidence, reason, and conversion suggestions (should-be-numeric / datetime / boolean / category, with the code). |
| `see_stats(df)` | `DataFrame` | Descriptive statistics, one row per column. |
| `see_dist(df)` | `DataFrame` | Normality verdict and top-3 fitted distributions for every numeric column. |
| `see_corr(df, target=None)` | `dict[str, DataFrame]` | Every association matrix with its p-values, the ranked pairs, VIF, missingness tests, target analysis (keys below). |
| `see_quality(df)` | `dict` | Every data-quality check, plus `alerts`. |
| `see_plot(df, out_dir, target=None)` | `list[Path]` | Writes only the informative plots. |

Every function accepts a `StoneConfig` to change thresholds (alpha, missing %, correlation and VIF thresholds, rare-category %, sample sizes, candidate distributions, …).

### Command line

```
elessar PATH [-t TARGET] [-o OUT] [--no-plots] [--sep SEP] [--nrows N] [--sample N]
           [--alpha A] [--width W] [--json] [--markdown] [--csv] [-q]
```

Reads CSV / TSV / TXT (separator sniffed) / JSON / JSONL / Parquet / Feather / Pickle / Excel, compressed or not. `python -m elessar` works too.

### `Vision`

| Attribute / method | Content |
|---|---|
| `.overview` | shape, memory, dtype counts, duplicates, overall missing % |
| `.summary` | `DataFrame`, one row per column, one column per key metric |
| `.columns` | column → full detail dict |
| `.correlations` | name → `DataFrame` |
| `.quality` | all quality checks |
| `.alerts` | list of `{severity, column, message}`, sorted high → medium → info |
| `.plots` | paths of the PNG files written |
| `.print()`, `.render(width)` | the terminal report |
| `.to_dict()`, `.to_json(path)`, `.to_csv(path)`, `.to_markdown(path)` | exports (CSV = the summary table) |

## What it computes

**Semantic types**: `empty`, `constant`, `boolean` (including 0/1, yes/no, true/false, y/n, oui/non), `id` (unique or near-unique, consecutive ints, UUIDs), `numeric_continuous`, `numeric_discrete` (counts, few levels), `categorical` (labels and high-cardinality codes), `ordinal-candidate` (low/medium/high, sizes, Likert, months, weekdays, quarters, 1-5 integer scales, ordered categoricals), `datetime`, `datetime_string`, `numeric_string` (`"1 234,5"`, `"12%"`, `"$3.2"`), `text`, `email`, `url`, `phone`, `postal_code`, `ip_address`, `json_like`, `mixed`.

**Statistics**
- *all columns*: count, missing, unique, memory, modes, top-N values
- *numeric*: caps / floors (censoring: values piled up at the max or min), zero inflation, mean (95 % CI), median (95 % CI from order statistics), std, var, sem, CV, min/max/range/sum, quantiles 1–99 %, IQR, MAD, 10 % trimmed mean, geometric and harmonic means, skewness, excess kurtosis, zeros, negatives, infinities, outliers (1.5×IQR, |z| > 3, modified z > 3.5), decimals, heaping (excess of round values, mixed-precision aware), monotonicity, lag-1 autocorrelation in row order with its p-value, bimodality coefficient, number of KDE modes
- *categorical / boolean*: cardinality, entropy, normalized entropy, Gini impurity, imbalance ratio, chi-square test of equal frequencies, share of True with its Wilson CI, rare categories, labels that differ only by case / spaces / accents, string lengths
- *datetime*: range, span, inferred frequency, largest gaps, duplicated timestamps, future dates, trend in rows per period (Kendall tau + p), distribution by year / month / weekday / hour (only when not trivial)
- *text*: length and word counts, empty / whitespace-only strings, share of digits / uppercase / punctuation, most common tokens

**Distributions**: Shapiro-Wilk, D'Agostino K², Jarque-Bera and Anderson-Darling with a majority verdict; MLE fits of `norm, lognorm, expon, gamma, weibull_min, beta, uniform, logistic, t, laplace, pareto, chi2, cauchy` (continuous) or `poisson, nbinom, geom, binom` (discrete), with support handling (shifted / rescaled fits), ranked by AIC, with BIC and KS statistic / p-value (KS p-values are optimistic when the parameters are estimated from the same data: that is reported too). Each candidate has a timeout.

**Associations and significance** (`see_corr` keys)

| Key | Content |
|---|---|
| `pearson`, `spearman`, `kendall` + `*_p`, `n_numeric` | num-num matrices, their p-value matrices and pairwise-complete counts |
| `cramers_v` + `cramers_v_p` + `cramers_v_n` | bias-corrected Cramér's V with chi-square p-values |
| `eta` + `eta_p` + `eta_n` | correlation ratio (categorical rows × numeric columns) with Kruskal-Wallis p-values |
| `association`, `association_p` | one matrix over every column: Spearman, Cramér's V or eta depending on the pair |
| `pairs`, `top_pairs` | every pair tested (best method, n, test, p, **BH-adjusted p**, significant, **Fisher 95 % CI**), and those above `corr_threshold` |
| `vif` | variance inflation factor of each numeric feature (multicollinearity) |
| `linear_dependencies` | exact linear relations recovered as equations, e.g. `Total = 1·HP + 1·Attack + 1·Defense + …` |
| `missingness` | does a column go missing more for some values of another column? (Mann-Whitney / chi-square, BH-adjusted: evidence against MCAR) |
| `target` | each feature vs the target: association, test (Spearman / Kruskal-Wallis / Mann-Whitney / chi-square), p, BH-adjusted p, AUC (binary target), mutual information, leakage flag |
| `missing_target` | is the target different when a feature is missing? (informative missingness) |

**Quality**: duplicate rows and columns, constant and quasi-constant columns, high missing, missing-value patterns (columns that go missing together, rows with many missing, most frequent patterns), mixed types, leading/trailing whitespace, placeholders used as missing (`""`, `"NA"`, `"null"`, `"?"`, `-999`, `99999`…), suspicious values (negative ages or prices, future birth dates, default dates like 1970-01-01), id-like columns, redundant columns, multicollinearity, sorted rows, imbalanced target, dtype optimizations with the estimated memory saving.

**Big data**: expensive steps (correlations, Kendall, KDE, pairwise tests) use at most `sample_size` rows (default 100 000); distribution fits and normality tests use at most `fit_sample_size` values (default 20 000). Sampling is deterministic (`random_state`). 500 000 rows × 20 columns take about 45 s on a laptop, plots included (about 30 s without).

**Robustness**: elessar runs on any DataFrame, whatever the column names (empty, `None`, `NaN`, numbers, tuples / MultiIndex, duplicates, slashes, newlines, emoji, pandas method names like `count`…) and whatever the content: empty frames, a single row, all-NaN columns, `inf` and 1e±300 values, huge cardinality, lists / dicts / bytes in cells, mixed types, sparse, nullable, categorical, period, interval, complex, Decimal, tz-aware datetimes, timedeltas, duplicated or MultiIndex row index. The target can be given by its original label (tuple, number, timestamp…). A randomized test suite generates hostile frames on every CI run. A failing step on one column is recorded as an error entry and an alert; it never stops the report.

**Real data**: tested on 33 public datasets (Titanic, adult census, California housing, diamonds, Pokémon, Vega movies, NYC taxis, penguins, wine quality, a 62-column MultiIndex brain-network table…). Things it finds there unprompted: Pokémon `Total = 1·HP + 1·Attack + 1·Defense + 1·Sp. Atk + 1·Sp. Def + 1·Speed`; housing values capped at 500,001 and ages at 52; adult capital gain top-coded at 99,999 and 92 % zeros; Titanic survival 30 % vs 67 % depending on whether Cabin is missing; taxi tips always 0 when the payment type is missing; movie titles stored as numbers (*300*, *1776*); files sorted by price or year.

## Sample terminal output

The Kaggle Titanic training set (`elessar train.csv --target Survived`), excerpt:

```
════════════════════════════════════════════════════════════════════════════════════════════════════════════════
 elessar report  ·  891 rows x 12 columns
════════════════════════════════════════════════════════════════════════════════════════════════════════════════

OVERVIEW
────────
  rows            891
  columns         12
  memory          285.6 KB  (suggested dtypes save ~124.8 KB)
  pandas dtypes   int64=5, str=5, float64=2
  semantic types  categorical=4, numeric_continuous=2, numeric_discrete=2, id=1, boolean=1, ordinal-candidate=1,
                  text=1
  duplicate rows  0 (0.0%)
  missing cells   8.1%  (708 rows with at least one)
  target          Survived

ALERTS (15: 1 high, 3 medium, 11 info)
──────────────────────────────────────
  HIGH    Cabin        77.1% missing -> drop, or impute and add a missing-indicator
  MEDIUM  Age          missingness is informative about 'Survived': Survived=1: 29% when missing vs 41% when
                       present (p_adj=5.9e-03) -> add a missing-indicator feature
  MEDIUM  Cabin        missingness is informative about 'Survived': Survived=1: 30% when missing vs 67% when
                       present (p_adj=6.2e-21) -> add a missing-indicator feature
  MEDIUM  Name         2 values with leading/trailing whitespace -> .str.strip()
  info    -            dtype optimizations save ~124.8 KB: PassengerId -> uint16, Survived -> uint8, Pclass ->
                       uint8, Sex -> category, Age -> float32, SibSp -> uint8, Parch -> uint8, Fare -> float32
                       (+1 more) (see quality['dtype_optimizations'] for code)
  info    Age          missingness depends on 'Embarked' (+2 other columns): 64% missing when Embarked=Q vs 14%
                       when Embarked=S (p_adj=2.5e-23) -> not missing completely at random
  info    Cabin        high cardinality (147 categories) -> group rare levels or use target/frequency encoding
  info    Cabin        missingness depends on 'Pclass' (+4 other columns): median Pclass = 3 when missing vs 1
                       (p_adj=3.5e-91) -> not missing completely at random
  info    Fare         13.0% outliers by the 1.5xIQR rule (outside [-26.72, 65.63])
  info    Fare         heavily skewed (skewness 4.79)
  info    Parch        heavily skewed (skewness 2.75)
  info    PassengerId  identifier-like column -> exclude from modelling and statistics
  info    SibSp        5.2% outliers by the 1.5xIQR rule (outside [-1.5, 2.5])
  info    SibSp        heavily skewed (skewness 3.70)
  info    Ticket       high cardinality (681 categories) -> group rare levels or use target/frequency encoding

COLUMN SUMMARY
──────────────
  column       type                missing  unique            mean/top  std/top%   min    max   skew
  ───────────  ──────────────────  ───────  ──────  ──────────────────  ────────  ────  ─────  ─────
  PassengerId  id                     0.0%     891                   1      0.1%
  Survived     boolean                0.0%       2                   0     61.6%
  Pclass       ordinal-candidate      0.0%       3               2.309    0.8361     1      3  -0.63
  Name         text                   0.0%     891  Braund, Mr. Owen …      0.1%
  Sex          categorical            0.0%       2                male     64.8%
  Age          numeric_continuous    19.9%      88                29.7     14.53  0.42     80   0.39
  SibSp        numeric_discrete       0.0%       7               0.523     1.103     0      8   3.70
  Parch        numeric_discrete       0.0%       7              0.3816    0.8061     0      6   2.75
  Ticket       categorical            0.0%     681              347082      0.8%
  Fare         numeric_continuous     0.0%     248                32.2     49.69     0  512.3   4.79
  Cabin        categorical           77.1%     147                  G6      2.0%
  Embarked     categorical            0.2%       3                   S     72.4%

COLUMN DETAILS
──────────────
...
▸ Age  ·  numeric_continuous  ·  float64  ·  float64 dtype: 88 distinct real values
    missing 177 (19.9%) · unique 88 (12.3%) · memory 7.0 KB · mode 24.0 ×30
    mean 29.7 ± 14.53 · 95% CI [28.63, 30.77] · median 28 [27, 29] · trimmed 29.27 · IQR 17.88 · MAD 9
    CV 0.4891 · geo mean 24.43
    min 0.42 · p1 1 · p5 4 · p25 20.12 · p75 38 · p95 56 · p99 65.87 · max 80 · sum 21,205.2
    skew 0.389 · kurtosis 0.178 · outliers IQR 1.5% · |z|>3 0.3% · mod-z 0.1% · zeros 0 · negatives 0
    decimals 1 · heaping ×1.0 · KDE modes 1 · bimodality coef 0.36 · lag-1 autocorr 0.05 (p 0.19)
    normality: not normal · Shapiro p 7e-08 · D'Agostino p 1e-04 · Jarque-Bera p 8e-05 · Anderson p 0.01
    shape: roughly symmetric, best fit normal(loc=29.7, scale=14.5)
    #1 norm(loc=29.7, scale=14.5) AIC 5,850.5 KS p 0.005
    #2 t(df=22.9, loc=29.5, scale=13.9) AIC 5,851.3 KS p 0.016
    #3 logistic(loc=29.1, scale=8.15) AIC 5,851.4 KS p 0.046
...

CORRELATIONS
────────────
  28 pairs tested, 24 significant after Benjamini-Hochberg correction (alpha 0.05); strongest shown below
  column A  column B  method      value  95% CI              n   p_adj  sig
  ────────  ────────  ─────────  ──────  ────────────────  ───  ──────  ───
  Pclass    Fare      spearman   -0.688  [-0.725, -0.648]  891  2e-124  *
  Survived  Sex       cramers_v  +0.543                    891   5e-58  *

  multicollinearity (VIF): Pclass 1.7, Fare 1.6, Age 1.3, SibSp 1.3, Parch 1.2

  missingness that depends on other columns (evidence against MCAR):
    Cabin <- Pclass: median Pclass = 3 when missing vs 1 · p_adj 3e-91
    Age <- Embarked: 64% missing when Embarked=Q vs 14% when Embarked=S · p_adj 2e-23

TARGET ANALYSIS
───────────────
  target: Survived  (boolean)
  feature   type                method     association  test            n   p_adj  sig    AUC     MI  flag
  ────────  ──────────────────  ─────────  ───────────  ────────────  ───  ──────  ───  ─────  ─────  ────
  Sex       categorical         cramers_v       +0.543  chi-square    891   3e-58  *           0.218
  Pclass    ordinal-candidate   eta             +0.338  mann-whitney  891   1e-23  *    0.319  0.092
  Fare      numeric_continuous  eta             +0.257  mann-whitney  891   1e-21  *    0.692  0.100
  Embarked  categorical         cramers_v       +0.166  chi-square    889   3e-06  *           0.021
  Parch     numeric_discrete    eta             +0.082  mann-whitney  891   5e-05  *    0.561  0.024
  Age       numeric_continuous  eta             +0.077  mann-whitney  714    0.16       0.469  0.023
  SibSp     numeric_discrete    eta             +0.035  mann-whitney  891  0.0094  *    0.543  0.019
  6/7 features significantly associated (BH-adjusted p < 0.05)

  is the target different when a feature is missing?
    yes  Cabin (77.1% missing): Survived=1: 30% when missing vs 67% when present · p_adj 6e-21
    yes  Age (19.9% missing): Survived=1: 29% when missing vs 41% when present · p_adj 0.0059
```

## Plots

Report-quality PNGs (144 dpi), one consistent design: a header with the column or question as
title and the key numbers as subtitle, the chart, and a footer note explaining the method.
Compact axis labels (`12.5k`, `3.4M`), direct value labels instead of redundant axes, a
colorblind-validated palette, and log axes chosen automatically for heavily skewed data.
Only plots that bring information are drawn: nothing for empty, constant or id columns, no
missing-values matrix when nothing is missing, no heatmap with fewer than 3 columns. Files are
named `out_dir/<NN>_<column>_<kind>.png`.

- **dataset overview**: one row per column with its type, missing share, distinct count, a summary and a sparkline
- **numeric**: histogram in rows with the best fit drawn as expected counts, boxplot with mean and outliers, Q-Q plot against the best fit with a 95 % envelope (points outside it highlighted)
- **categorical / boolean**: top-N bars with shares and counts, sorted by frequency (or in scale order for ordinals), "other" and "missing" shown apart
- **datetime**: rows per period at an automatic resolution, rolling mean, peak annotated
- **text**: lengths in characters and words
- **dataset**: missing-values matrix, missingness correlation, Pearson / Spearman / Cramér's V / all-type association heatmaps (non-significant cells faded)
- **pairs**: joint plots with marginal distributions, hexbin density for large data, least-squares line and a median trend (honest for non-linear relations), levels jittered for discrete variables
- **target**: "what drives the target" ranking (non-significant greyed out), and per-feature plots: joint plot, boxplots sorted by median with individual points, or stacked shares with % labels

| | |
|---|---|
| ![Dataset overview](docs/overview.png) | ![Numeric distribution](docs/numeric_distribution.png) |
| ![Association heatmap](docs/association_heatmap.png) | ![What drives the target](docs/target_ranking.png) |
| ![Joint plot with hexbin density](docs/joint_plot.png) | ![Target by a categorical feature](docs/target_stacked.png) |

## Development

```bash
git clone https://github.com/thibautdqb/elessar && cd elessar
pip install -e ".[dev]"
pytest
python examples/demo.py
```

The animated terminal at the top is generated from a real run of the command line:
`python docs/make_terminal_svg.py train.csv Survived docs/terminal.svg`.

## License

MIT
