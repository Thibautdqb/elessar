# Serial inference and independent final consensus — v1.2 candidate

Base `43857e0b7dc15892084fcf80f16f01dc276ff382`. Review branch `fix/v1.2-serial-consensus-validation`. This record supersedes the open serial Kendall/categorical and final-consensus items in the preliminary record; historical measures remain intact.

## Changes and scientific basis

Serial Kendall, Cramér's V and mixed eta tests, including categorical/mixed target links, now use a conservative symmetric truncated time-shift (TTS) test. Effects and p-values use the same fixed central window of original levels. The hypothesis is independence of the whole temporal processes. At least one process must be strictly stationary, with a statistic/type policy fixed in advance; automatic typing or ADF/KPSS cannot certify those assumptions.

The rule is `min(1, max(B_forward, B_reverse)/(r+1))`, including lag zero and all ties. Each orientation is valid when its shifted process is stationary; the maximum is at least the valid orientation's p-value. The radius is fixed from length, `min(99,n_time//4)` by default. No fitted autocorrelation, circular wrapping or search for the best lag/radius is used. Balanced panels shift whole entity vectors synchronously, preserving cross-entity dependence.

This follows the conservative rule of [Harris (2020)](https://arxiv.org/abs/2012.06862) and [Yuan & Shou (2024)](https://doi.org/10.1371/journal.pbio.3002758). The literal reference tests use SciPy Kendall statistics and an independent contingency-table implementation.

Unsupported missing/irregular/unbalanced grids yield no p-value and no significance, with a reason in `serial_inference`. Conditional results expose their windows, counts, radii, exceedances and assumptions. Undefined central effects provide p=1 rather than removing undefined surrogates. Default radius 99 imposes a minimum p-value of 0.01, so multiplicity and weak signals can considerably limit power. The cap retains consecutive dates and full entity vectors.

The selector retains its original 80% training budget. The reserved 20% is split into approximately 10% voting and 10% final-test rows, without target values; chronological cuts keep whole dates. `n_holdout` remains the total reserved from tuning; `n_voting` and `n_final_test` distinguish its uses. Full-report semantic types are learned on training rows, and the target cannot choose the time axis.

`consensus_evaluation` freezes the kept subset and a predefined random forest (100 trees, leaf size 3, fixed feature-sampling rule), refits on training+voting rows only, and scores once on the unused final test. When no feature is kept, the assessment uses a training-prior dummy model. A baseline is reported without using it to choose a winner. Scores are unavailable with a reason when a valid split/metric cannot be formed or an external differencing policy cannot be certified. A chronological unused test is independent of selection decisions, not necessarily IID in its rows.

| Implementation boundary | Exact code reference |
|---|---|
| Conservative TTS rule | [src/elessar/serial_inference.py:87](https://github.com/Thibautdqb/elessar/blob/fix/v1.2-serial-consensus-validation/src/elessar/serial_inference.py#L87) |
| Effect/test/count consistency | [src/elessar/correlations.py:622](https://github.com/Thibautdqb/elessar/blob/fix/v1.2-serial-consensus-validation/src/elessar/correlations.py#L622) |
| Three disjoint partitions | [src/elessar/selection.py:115](https://github.com/Thibautdqb/elessar/blob/fix/v1.2-serial-consensus-validation/src/elessar/selection.py#L115) |
| Fixed final estimator and unused outcomes | [src/elessar/selection.py:218](https://github.com/Thibautdqb/elessar/blob/fix/v1.2-serial-consensus-validation/src/elessar/selection.py#L218) |
| Training-only typing and target-independent time axis | [src/elessar/core.py:152](https://github.com/Thibautdqb/elessar/blob/fix/v1.2-serial-consensus-validation/src/elessar/core.py#L152) |
## Executed verification

- 442 ordinary checks pass; all 416 previous ordinary checks are retained. Six reproductions fail on the base and pass after correction, including full-report type leakage.
- 91 unique statistical/stress checks pass across recorded executions, including all 72 prior checks.
- The original audit's two serial calibration failures and numeric control pass on their unchanged 300-run designs.
- Ruff 0.6.9 and git diff --check pass. Only the recorded Linux/Python 3.12 dependency set was executed.

The statistical result is a consolidation of disjoint/current passing cases, not a claimed single execution: 68 existing/stress cases, 6 valid cases from the initial 7-case consensus run, 15 serial cases, and 2 corrected-signal/policy cases. Per-case provenance is recorded in [the machine-readable results](validation-v12-serial-consensus-results.json). Intermediate failures remain in the raw evidence.

## Serial null calibration

All 500 p-values in every supported null design are finite. Acceptance was fixed at 0.05 + 3 sqrt(0.05*0.95/500) = 0.07924. Wilson intervals below are per-design, not simultaneous.

| Independent stationary design | Rejections | Rate at 5% | Wilson 95% interval |
|---|---:|---:|---|
| positive_ar1 | 3/500 | 0.60% | [0.20%, 1.75%] |
| negative_ar1 | 5/500 | 1.00% | [0.43%, 2.32%] |
| ar2 | 9/500 | 1.80% | [0.95%, 3.39%] |
| ma100 | 3/500 | 0.60% | [0.20%, 1.75%] |
| tied_ar1 | 6/500 | 1.20% | [0.55%, 2.59%] |
| panel | 4/500 | 0.80% | [0.31%, 2.04%] |
| four_state | 5/500 | 1.00% | [0.43%, 2.32%] |
| negative_markov | 9/500 | 1.80% | [0.95%, 3.39%] |
| rotating_markov | 5/500 | 1.00% | [0.43%, 2.32%] |
| mixed | 10/500 | 2.00% | [1.09%, 3.64%] |
| categorical_target | 8/500 | 1.60% | [0.81%, 3.13%] |

On identical 500-run seeds, the earlier Kendall AR(1) result was 48.40% and the four-state categorical result was 27.60%. The new test is conservative; it is not advertised as exactly uniform at the nominal level.

The four predeclared strong-link designs (positive/negative numeric, categorical, mixed) are detected in 200/200 trials each. This does not establish power for weak links or every process.

## Independent consensus scores

| Noise task | Runs | Mean final score | Standard error | Training-prior baseline |
|---|---:|---:|---:|---:|
| regression | 100 | -0.05517 | 0.01038 | -0.02865 |
| binary | 100 | 0.49363 | 0.00613 | 0.50000 |

On 30 independent strong-signal tables (`y=3*x+unit noise`, population correlation .949), the signal is retained in 100.00% and mean independent R² is 0.86994. The original bounds were unchanged.

## Protocol amendments and limits

The initial 64/16/20 partition reduced the original training budget and failed two existing ordinary checks (noise votes and a reserved-date description). The implemented 80/10/10 partition preserves the original tuning sample and clearly labels both reserved uses; those checks pass without relaxing their assertions.

An initial new recovery fixture used `y=3*x+.25*noise`, with population correlation .9965. All 30 true predictors were excluded by the existing .98 leakage-suspect guard. This is retained as an explicit policy-boundary diagnostic: a high association alone cannot establish leakage, and legitimate nearly deterministic predictors can be excluded. Recovery is assessed below that fixed threshold (`3*x+unit noise`, analytic oracle R²=.9), with the same 30 seeds and unchanged >.95 retention/>.8 mean-R² criteria. No existing library test or bound was changed.

Numeric Pearson/Spearman retain a lag-1 approximation. Numeric-target shape and missingness tests still have explicit serial limitations. The unused final test evaluates the outcome of the adaptive selection chain; it does not make every internal screening/stationarity decision fully nested. Reusing final scores to tune a pipeline breaks that separation. Neither two nonstationary processes nor causal/conditional dependence are covered by the TTS guarantee.

## Reproduction

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ELESSAR_VALIDATION_DATA=validation-cache \
  ELESSAR_VALIDATION_OUT=results-local.json python -m pytest -q -m validation tests/validation
ruff check src tests docs examples
```

Public inputs remain pinned to the immutable revisions and hashes in [validation-v12-data.json](validation-v12-data.json). The machine-readable record gives code hashes, exact primary dependencies and per-case provenance. This is a review candidate; main is not modified and no release is published.
