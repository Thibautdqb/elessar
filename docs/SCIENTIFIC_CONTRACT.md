# Scientific contract for the 1.2 candidate

Statistical results are conditional on a sampling model. Executable regression checks, simulation
calibration and reference comparisons support the documented domains; passing them does not prove
universal validity. This contract distinguishes numerical definitions, approximations and unavailable
inference. It does not certify provenance, stationarity or independence from the observed table alone.

| Output | Meaning and conditions | Guard or limitation |
|---|---|---|
| Counts, quantiles, moments, entropy | Descriptions of the observed finite values, subject to float64 precision | Moments use centred magnitude scaling; unrepresentable variance/range/sums are identified. A description is not a population guarantee. |
| Median interval | Exact conservative order-statistic interval for an IID population median | Exact binomial ranks replace the normal approximation. No finite interval is claimed when the requested coverage cannot be attained. Ties can make it conservative. |
| Mean interval, Wilson proportion interval | IID sampling; Student mean interval additionally needs normality or a suitable large-sample approximation | These do not become serially calibrated merely because a date column is present. The numeric profile exposes its IID condition. |
| IID Pearson/Spearman/Kendall | Tests of the prescribed association under their distribution and exchangeability assumptions | Pearson's t reference is exact under bivariate normality; Spearman's t p-value and Fisher intervals are approximations, especially at small sizes. Automatic shape selection is not a distributional certificate. |
| IID categorical/mixed tests | Bias-corrected V with a chi-square approximation; eta with Kruskal-Wallis; binary AUC with Mann-Whitney | Sparse expected cells, small groups and ties affect approximations. Rejection gives evidence of association, not causality. |
| Serial Kendall/categorical/mixed tests | Conservative symmetric truncated time-shift (TTS), on original levels in a fixed central window | Process-independence null; at least one process must be strictly stationary, and statistic/type policy fixed. Irregular/missing/unbalanced inputs are unavailable, without IID fallback. Balanced panels shift entity vectors together. |
| Serial numeric target | TTS of a prescribed maximum of absolute Spearman and eta across feature rank deciles | Recompute the same statistic for every surrogate. The reported signed Spearman effect is descriptive; `serial_inference.statistic` records the actual omnibus statistic. No scalar lag-1 shape correction is used. |
| Serial missingness | TTS of an observed missing indicator versus the comparator; centred AUC or V | The indicator is complete even when the measured feature is missing. Gaps in the comparator are not compressed or imputed to manufacture a p-value. Evidence against MCAR does not distinguish MAR from MNAR. |
| Serial Pearson/Spearman pair p-values and intervals | Existing Bartlett lag-1 effective-size model and Fisher approximation | Explicitly labelled as a model approximation, validated on covered short-memory designs. This is not the TTS guarantee or a general correction for every dependence process. |
| Multiplicity | BH correction over the stated finite test family | Requires valid input p-values and appropriate dependence conditions (independence/PRDS). Automatic EDA does not establish those conditions. No unconditional FDR guarantee is made for arbitrary adaptive analyses. |
| Fitted distributions | Likelihood-based numerical candidates, ranked by AIC/BIC within their declared domains | Global optima and regular-model asymptotics are not certified for every free-location family. Candidate selection and fitting use the same data. CDF gaps are diagnostics; `ks_pvalue` is unavailable, and the unadjusted approximation is retained only as `diagnostic_pvalue`. |
| Normality | Four IID tests and an exploratory majority verdict | A non-rejection does not prove normality. The vote is not itself a calibrated combined alpha-level test. |
| PCA, factors, reliability | Descriptive rotations/loadings and model-based diagnostics; parallel analysis against simulated noise | Bartlett's chi-square reference assumes an adequate IID multivariate-normal model. Factor-count simulations cover specified latent structures, not every possible factor model. Alpha is not unidimensionality or validity. |
| ADF/KPSS | Finite-sample critical surfaces and asymptotic stationarity diagnostics under their assumptions | They do not certify strict stationarity. Lag/bandwidth choices and limited p-value tables remain part of the model; bounded KPSS p-values are flagged. |
| Trend | Mann-Kendall with an estimated short-memory variance-inflation model and Sen slope | `inference_status` identifies the model approximation; long-memory calibration is not guaranteed. A detected trend is not a causal explanation. |
| Outliers and preparation | Exploratory policies and suggested transformations, not automatic data correction | Thresholds are heuristics, not proof of measurement error. Train-only fitting is mandatory for predictive use. Adaptive category cuts are selected using one label-stratified random half and confirmed using independent rows with Welch tests at alpha/7 across at most seven split nodes; the advertised familywise bound is conditional on independent IID rows and a fixed target family, not arbitrary dependence or downstream model validity. |

## Leakage and final evaluation

A high association raises a provenance warning and does not automatically remove a predictor.
`exclude_features` names predictors known to be unavailable at prediction time. By default, observed
exact affine or low-cardinality bijective target aliases are quarantined; this is an explicit precaution,
not proof of temporal provenance. `exclude_target_aliases=False` allows a caller with established
pre-outcome availability to override that precaution. Near-deterministic legitimate signals are retained.
All automatic screening and alias decisions use training rows only.

The 80/10/10 training/voting/final-test partitions are chosen without target values, with whole-date
cuts for temporal tables. All votes and tuning precede the final test. A predefined model is refitted
on the frozen kept subset using development rows and is scored once on unused final outcomes.
An empty subset uses a training-prior baseline; undefined metrics and impossible splits are unavailable.
The final test evaluates the adaptive chain's outcome; it does not make every inner screening decision
fully nested, or make temporal rows IID. Reusing the final score to change the analysis breaks this separation.

## Evidence and reference implementations

- Tests compare coefficients, critical surfaces, order-statistic coverage and new TTS statistics with
  SciPy, statsmodels or independent literal implementations.
- Simulation inputs have specified nulls/alternatives, fixed seeds and recorded acceptance bounds.
  New release designs use two independent seed streams. Replaying earlier fixed benchmarks checks
  reproducibility; it is not counted as new independent Monte Carlo evidence.
- Public datasets retain immutable source revisions and hashes in `validation-v12-data.json`.
- CI executes complete ordinary suites twice across Python/OS combinations, the declared dependency
  floors, two statistical streams, and fresh wheel/sdist installations. Actual run results, not workflow
  definitions, determine which combinations may be reported as verified.
- Historical records remain historical. The release candidate is not published or merged by these checks.

References: [SciPy statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html),
[SciPy fitted-model goodness of fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html),
[Yuan & Shou (2024)](https://doi.org/10.1371/journal.pbio.3002758),
[Harris (2020)](https://arxiv.org/abs/2012.06862).
