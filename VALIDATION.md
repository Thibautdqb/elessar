# Statistical validation

How far elessar's statistics can be trusted, measured against a known truth: simulations whose answer is
known in advance, and public datasets whose content is documented. Every figure of the current version comes
from [`docs/validation-results.json`](docs/validation-results.json), written by the suite in
[`tests/validation`](tests/validation). The "before 1.1" figures come from the audit runs made while preparing
1.1, with the same designs.

```bash
python -m pip install -e ".[dev]"
python -m pytest -m validation tests/validation          # 69 checks, about 5 minutes
ELESSAR_VALIDATION_OUT=results.json python -m pytest -m validation tests/validation   # with the measures
```

The suite also runs once a month and on demand on GitHub (`.github/workflows/validation.yml`), the measures
kept as an artifact. It is left out of the ordinary test run (`python -m pytest`), which stays fast.

**Run of reference**: elessar 1.1.0, Python 3.12, pandas 3.1, NumPy 2.5, SciPy 1.18, scikit-learn 1.9,
2026-10-30. The suite also passes on Python 3.10 with the oldest supported versions (pandas 2.2.0, NumPy 1.26,
SciPy 1.11, scikit-learn 1.3).

## How the checks are built

- **A known truth.** Simulated data are drawn so that the right answer is known: independent columns for a
  type I error, a planted factor structure, a seasonal cycle of a given period, values multiplied by 100.
  Public datasets are checked against facts documented for them, and counts against pandas computed in the
  test itself.
- **Bounds that leave room for chance, not tuned to pass.** A rate measured on *n* runs is compared with
  the claimed rate plus three binomial standard errors. A bound is never moved to make a test pass: when a
  first bound turned out wrong, the claim was restated (and said so in the test).
- **Limits are measured, not hidden.** When a method cannot reach its nominal level on a case, the case is
  still run and its figure reported here.
- **Reproducible.** Every simulation has a fixed seed (no Python hash): two runs give identical figures.
- **Sizes.** 69 checks: 55 on simulations (about 25,000 simulated tables or series in all) and 14 on public
  datasets (12 distinct tables from seaborn-data and Rdatasets).

| Area | Checks | What is measured |
|---|---:|---|
| Associations between columns | 23 | type I error of every test, in independent and in time-ordered rows, false discovery rate, interval coverage, mutual information, power, U-shaped links |
| Feature selection | 4 | scores and kept features on noise, recovery of known signals, dated and non-stationary rows |
| Factor analysis | 4 | number of factors on 300 structures, three reference datasets |
| Outliers | 10 | advice on five clean distributions, planted errors in five kinds of columns |
| Time series | 17 | seasonality, trend under seven noise structures, stationarity |
| Public datasets | 11 | counts against pandas on ten tables, known facts |

## 1. Associations between columns

### Type I error of the numeric pair tests

For a pair of numeric columns, elessar tests Pearson when both columns are roughly symmetric with light
tails (|skewness| < 1, |excess kurtosis| < 3) and Spearman otherwise, the choice being made from each
column's shape before looking at the pair. Design: 4 tables of 30 independent columns of 200 rows per
distribution, 1,740 pairs, share of pairs with a raw p-value below 5%.

| Distribution | Test used | Rejected at 5% |
|---|---|---:|
| normal | Pearson | 5.5% |
| uniform | Pearson | 5.7% |
| discrete 0-9 | Pearson | 5.3% |
| Student t3 | Pearson or Spearman by column | 4.7% |
| log-normal | Spearman | 4.3% |
| zero-inflated exponential | Spearman | 4.8% |

Before 1.1, the stronger of Pearson and Spearman was kept after the fact: 6.5% of independent normal pairs
and 9.5% of log-normal ones were rejected at 5%.

### Rows in time order

In a table indexed by time (a series or a panel, whatever the order of the file), the columns that ADF and
KPSS do not show stationary are replaced by their first differences (within each entity), and every test
uses Bartlett's effective sample size n (1 − r₁r₂) / (1 + r₁r₂) from the lag-1 autocorrelations of the two
columns. Design: 120 tables of three independent columns of 300 dates, rows shuffled in the file.

| Columns | Before 1.1 (rows as if independent) | elessar 1.1 |
|---|---:|---:|
| white noise | 3.8% | 4.7% |
| AR(1) 0.5 | 11% | 3.3% |
| AR(1) 0.9 | 52% | 4.7% |
| random walks | 86% | 5.6% |
| panel of 4 entities × 100 months, random walks | - | 3.3% |

Power, 150 tables: two AR(1) 0.5 series with innovations correlated at 0.25 are found associated in 87%,
two random walks sharing a common walk in 91%.

### Other tests

2,000 runs of 300 independent rows:

| Test | Used for | Rejected at 5% |
|---|---|---:|
| chi-square | categorical with categorical (Cramér's V) | 5.6% |
| Kruskal-Wallis | categorical with numeric (eta) | 5.4% |
| Mann-Whitney | numeric feature with a binary target (AUC) | 5.3% |

### Numeric feature and numeric target: monotone and U-shaped links

The test is Spearman together with a Kruskal-Wallis of the target across the deciles of the feature, two
tests fixed in advance combined by Bonferroni. 400 runs of 300 rows:

| Link | Found (rejected at 5%) |
|---|---:|
| none | 4.3% |
| linear, r = 0.15 | 61% |
| U shape (0.3 x²) | 97% |
| monotone (exp 0.3 x) | 99.8% |

Spearman alone found the U shape in 5% of the runs and the weak linear link in 69%.

### Benjamini-Hochberg

300 runs of 200 tests, 20 of them with a real effect (z = 3): false discovery rate **4.5%**, at the bound the
procedure guarantees (m0 / m × 5% = 4.5%), power 49%. elessar's adjusted p-values match the step-up
procedure written out to 2 × 10⁻¹⁶.

### Confidence intervals of correlations

2,000 bivariate normal samples of 80 rows per true correlation; share of 95% intervals that contain it
(Fisher z for Pearson, Bonett-Wright variance for Spearman):

| True ρ | Pearson | Spearman |
|---:|---:|---:|
| 0.0 | 94.9% | 94.7% |
| 0.3 | 95.2% | 95.4% |
| 0.7 | 94.1% | 95.5% |

### Mutual information

A constant column in a file sorted by the target scores **0 bits** (3.3 bits before 1.1, when ties were
ranked by row position). 200 independent noise columns: 0.003 bits on average, 99th percentile 0.0074. A
binary target shifted by one standard deviation: 0.138 bits, identical in the sorted file.

### Power of the chosen pair test

400 runs of 200 rows: a linear link of r = 0.15 between normal columns is found in 59% of them (Pearson's
power), a monotone link between log-normal columns in 91% (Spearman's power, where Pearson alone found 59%).

## 2. Feature selection

Five methods vote (decision tree, random forest, lasso, ridge, univariate filter); a feature is kept with
three votes. The tree, the lasso and ridge vote only for features that beat every shuffled copy of the
features (shadows); the random forest uses a permutation test on held-out rows, the filter the
Benjamini-Hochberg-adjusted link with the target. Every model is tuned by cross-validation on 80% of the
rows and scored once on the other 20%; dated rows are sorted and validated on their most recent part.

### Pure noise

60 tables of 400 rows and 10 noise features:

| Target | Decision tree | Random forest | Lasso | Ridge | Chance | Noise features kept |
|---|---:|---:|---:|---:|---:|---:|
| binary (AUC) | 0.507 | 0.506 | 0.498 | 0.496 | 0.5 | **1.8%** |
| numeric (R²) | −0.082 | −0.074 | −0.016 | −0.017 | ≤ 0 | **2.7%** |

No method sits above chance. Before 1.1 the decision tree and the lasso reached an AUC of 0.517 on noise (their
best setting scored on the folds that chose it), and with a binary target 28% of noise features were kept
(ridge voted for any coefficient above 10% of the largest, the tree for half an equal share).

### Known signals

30 tables of 600 rows: two linear signals, one U-shaped signal (0.5 x²), eight noise columns.

| | Kept |
|---|---:|
| linear signals | 100% |
| U-shaped signal | 90% (borderline in the other runs) |
| noise columns | 0.4% |

Before 1.1 the U-shaped signal was kept in a third of the runs: the filter was Spearman alone.

### Dated and non-stationary rows

Two independent random walks over 600 days, ten times: nothing to predict. The target is not stationary, so
its changes are modelled from the changes of the other walk (9 runs out of 10; in the tenth, one of the two
walks was not recognised as non-stationary). The best held-out R² is −0.01 on average, the same file with shuffled rows gives
exactly the same scores in 10 runs out of 10, and the other walk is kept in none of them. Before 1.1, the
shuffled file was cross-validated with shuffled folds (R² +0.38), and in levels the other walk was kept in 9
runs out of 10.

## 3. Factor analysis

### Number of factors

300 simulated structures: 1 to 4 factors of 3 to 6 items, loadings .45 to .80, factors correlated 0 to .3,
150, 300 or 800 rows, half of them with 1 to 3 items that load on nothing.

| Criterion | Right number | Overestimated |
|---|---:|---:|
| **elessar** (parallel analysis, then coherence and item removal) | **98.3%** | **1.0%** |
| parallel analysis alone (Horn) | 97.0% | 2.3% |
| scree elbow | 86.0% | 13.0% |
| eigenvalue above 1 (Kaiser) | 39.3% | 60.7% |
| cumulative variance ≥ 60% | 4.7% | 95.3% |

Items that load on nothing are removed in 99% of the structures; 91% of the true items are kept on average
(an item whose communality is below .40 is removed, as the method requires, and with loadings of .45 some
true items fall below it). Before 1.1 the number came from two of the three classic criteria agreeing, else
the scree elbow: 94% right.

### Reference datasets

| Dataset | Rows | Expected | Found | Items kept | Factors |
|---|---:|---:|---:|---:|---|
| Holzinger & Swineford (1939), 9 tests | 301 | 3 | **3** | 9 / 9 | verbal (`paragrap`, `sentence`, `wordm`, α .89), visual (`visual`, `cubes`, `flags`, α .63), speed (`addition`, `counting`, `straight`, α .69) |
| Holzinger & Swineford, 24 tests (Harman) | 301 | 4 | **4** | 16 / 24 | verbal (5 tests, α .91), speed (4, α .75), spatial with `deduct` (4, α .68), memory (`numberr`, `object`, `wordr`, α .62) |
| Big Five questionnaire, BFI (psych) | 2,800 | 5 | **5** | 24 / 25 | neuroticism, extraversion, conscientiousness, openness, agreeableness: each factor gathers exactly the items of one trait (α .61 to .82); A5 removed |

Sources: `sem::HS.data` and `psych::bfi` from Rdatasets, downloaded at test time and cached. A two-item factor
is judged on the Spearman-Brown coefficient (checked by the unit tests), Cronbach's alpha understating the
reliability of two items that vary unequally.

## 4. Outliers

Three methods per column (3 sigma, boxplot fences, modified z); a point flagged by two is a consensus
outlier. The advice is *keep*, *reassign* (cap at the fences) or *remove*: a few values more than 10 robust
standard deviations away (on the log scale for a positive, right-skewed column) from a column whose other
values have light tails. Far values in a heavy-tailed column are flagged with the advice to check them.

### Clean data

300 columns of 1,000 rows without any error; share of columns advised to drop rows:

| Distribution | Consensus points | Removal advised | Before 1.1 |
|---|---:|---:|---:|
| normal | 0.26% | 0% | 0% |
| Student t5 | 1.4% | 1.0% | 12% |
| Student t3 | 2.9% | 0% | 70% |
| log-normal | 7.5% | 0% | 7% |
| exponential | 4.1% | 0.3% | 17% |

The other columns are advised to cap their consensus points (reassign), which changes little on a normal
column that holds a couple of them.

### Planted errors

300 columns per kind, 1 or 2 values multiplied by 100:

| Column | Errors flagged by two methods | Removal advised |
|---|---:|---:|
| normal, 1,000 rows | 100% | 100% |
| uniform, 1,000 rows | 100% | 100% |
| normal, 100 rows | 100% | 99% |
| Student t5 | 100% | 2% (check them) |
| exponential | 94% | 0.3% (check them) |

In a heavy- or long-tailed column an error cannot be told from a genuine extreme value: it is flagged, and
removing it is left to the user.

## 5. Time series

### Seasonality

200 series per case; a cycle is reported when its seasonal strength reaches 0.25.

| Series | Truth | Reported | Right period |
|---|---|---:|---:|
| white noise, 365 days | none | 0% | - |
| AR(1) 0.7, 365 days | none | 0% | - |
| weekly cycle (amplitude 2) + AR(1) 0.3 noise | period 7 | 100% | 100% |
| weak weekly cycle (amplitude 0.6) | period 7 | 18% | 18% |
| monthly series, yearly cycle, 10 years | period 12 | 100% | 100% |

### Trend (Mann-Kendall) under autocorrelation

600 series of 200 points without trend per noise; share in which a trend is reported at 5%.

| Noise | Uncorrected | Hamed-Rao (before 1.1) | **elessar 1.1** |
|---|---:|---:|---:|
| white noise | 5.5% | 4.3% | **5.0%** |
| AR(1) 0.5 | 25.8% | 15% | **6.3%** |
| AR(1) 0.8 | 49.3% | 25% | **5.5%** |
| AR(2) 0.5, 0.3 | 57.7% | 22% | **11.3%** |
| MA(1) 0.7 | 16.5% | 9% | **1.8%** |
| ARMA(1,1) 0.7, 0.4 | 43.2% | 23% | **1.8%** |
| long memory (Hurst 0.8) | 53.8% | 30% | **27.0%** |

The variance of the statistic is inflated by the larger of two models of the autocorrelation (AR(1) with
Kendall's bias correction, AR(p) chosen by AIC). Power: a trend of about one standard deviation of the series
over the period, under AR(1) 0.5 noise, is found in 50% of 500 series.

### Stationarity (ADF and KPSS read together)

300 series of 300 points per kind:

| Series | Expected verdict | Expected | Opposite verdict | Other |
|---|---|---:|---:|---|
| white noise | stationary | 94% | 0% | trend-stationary 6% |
| AR(1) 0.5 | stationary | 94% | 0% | trend-stationary 5%, inconclusive 1% |
| random walk | unit root | 75% | 1.3% | inconclusive 22%, trend-stationary 2% |
| linear trend + noise | trend-stationary | 96% | 3.7% | - |

The associations of a dated table treat "inconclusive" like a unit root (the column is differenced), so a
random walk left undecided is not tested in levels.

## 6. Public datasets

Ten tables from [seaborn-data](https://github.com/mwaskom/seaborn-data), read from their CSV files. Rows,
columns, rows with a missing value and duplicate rows match pandas exactly on all ten:

| Dataset | Rows | Rows with a missing value | Duplicate rows |
|---|---:|---:|---:|
| titanic | 891 | 709 | 107 |
| iris | 150 | 0 | 1 |
| tips | 244 | 0 | 1 |
| penguins | 344 | 11 | 0 |
| mpg | 398 | 6 | 0 |
| flights | 144 | 0 | 0 |
| healthexp | 274 | 0 | 0 |
| diamonds | 53,940 | 0 | 146 |
| planets | 1,035 | 537 | 4 |
| taxis | 6,433 | 92 | 0 |

Known facts found:

- **flights** (monthly airline passengers, 1949-1960): a single monthly series built from the `year` and
  `month` columns, a strong yearly cycle (period 12) and an increasing trend.
- **healthexp**: a panel, one series per `Country`.
- **titanic**, target `survived`: `alive` (the target written as yes / no) is flagged as leakage and left out of
  the selection; the information of `sex` is kept (`sex`, `who` and `adult_male` carry it; the trees pick one of
  a redundant group, so `sex` itself may stay borderline).
- **iris**, target `species`: petal length and width are the two strongest links. **tips**, target `tip`:
  `total_bill` first. **diamonds**, target `price`: `carat` and the length `x` first.

## 7. Remaining limits

What elessar cannot do better, measured above:

1. **Trend under strongly persistent noise.** AR(2) noise (0.5, 0.3) gives 11% of false trends at 5%, at most
   twice the nominal level; long memory (Hurst 0.8) 27%. On 200 points a trend and long memory are hardly
   distinguishable, and no correction built on short memory removes it.
2. **Errors in heavy-tailed columns.** An entry error in a Student or exponential column cannot be told from a
   genuine extreme value: it is flagged, and removal is not advised.
3. **Weak cycles.** A cycle whose seasonal strength is below 0.25 is not reported (by design: the decision rests
   on an effect size, a significance test being invalid under autocorrelation).
4. **Random walks.** ADF and KPSS leave about a fifth of random walks of 300 points "inconclusive" (the
   associations then difference them all the same).
5. **Communality rule.** Removing items whose communality is below .40 (Costello & Osborne, 2005) also removes
   true items with loadings below about .63: 91% of the true items are kept on average.
6. **Power traded for validity.** The Bonferroni test of a numeric target (61% instead of 69% on a weak linear
   link) and the effective sample size of time-ordered rows cost some power, the price of an honest 5%.

## 8. Not covered by this suite

Checked by the unit tests (`python -m pytest`), not by simulation here: the typing decision tree (rules
tested one by one), dates of birth, distribution fitting by AIC, missingness mechanisms, VIF and exact
linear dependencies, robust multivariate outliers, the preparation plan (every snippet executed in order on
the user's DataFrame), exact counts over every row of tables read in chunks, readers of every format, and the
restricted unpickler for pickles fetched from a URL (pandas 2.2 and 3.x pickles rebuilt, code-running and
`getattr` pickles refused).

## 9. Changes made because of this validation

Version 1.1 changed these methods after measuring them (details in the [CHANGELOG](CHANGELOG.md)):

| Feature | Before | After |
|---|---|---|
| mutual information, constant column in a sorted file | 3.3 bits | 0 bits |
| numeric pair tests, log-normal columns, rejected at 5% | 9.5% | 4.3% |
| two independent random walks of a dated table, rejected at 5% | 86% | 5.6% |
| two independent AR(1) 0.9 series of a dated table, rejected at 5% | 52% | 4.7% |
| U-shaped link with a numeric target, found | 5% | 97% |
| selection, noise features kept, binary target | 28% | 1.8% |
| selection, two independent random walks in a shuffled file | R² +0.38 | R² −0.01 |
| selection, AUC on noise (decision tree) | 0.517 | 0.507 |
| factor analysis, right number of factors | 94% | 98% |
| two-item factor judged on | Cronbach's alpha | Spearman-Brown |
| trend under AR(1) 0.8 noise, reported at 5% | 25% | 5.5% |
| clean Student t3 column, removal advised | 70% | 0% |
| clean log-normal column, removal advised | 7% | 0% |
