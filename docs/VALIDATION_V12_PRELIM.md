# Preliminary v1.2 validation

Based on `2262600bc335e535297bfc64b05b646e6172590f` (1.1.0), branch `fix/v1.2-preliminary-validation`. This is a reviewed-evidence candidate, not a release. No merge into main or publication is authorised by these results.

## Executed checks

| Verification | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Complete ordinary suite, including all 378 original checks | 416 | 0 | 0 |
| Existing statistical validation suite | 69 | 0 | 0 |
| New structural stress checks, 500 cases each | 3 | 0 | 0 |
| Final-code rerun: selection and public known-facts checks | 5 | 0 | 0 |
| Pinned public-data checks, including factor references | 14 | 0 | 0 |

There are **72 unique statistical/stress test cases**, with overlapping verification reruns, not 69+3+5+14 distinct checks. The 15 deterministic reproductions all failed on the base and all pass after the corrections. The ordinary suite adds 38 cases and removes none of the original 378. One expected RuntimeWarning is recorded in the all-missing training-column stress fixture; the fitted column is checked to remain zero. Ruff 0.6.9 and git diff --check pass.

The first full verification found two regressions on huge finite values in the new preprocessing pipeline. A magnitude preconditioner was added before moment calculations, and both original regressions passed again before the final complete ordinary run. The full 69-case statistical run preceded that numerical guard; all affected selection checks and public known facts were subsequently rerun on the final implementation. The consolidated JSON records the source of each measure rather than claiming a single execution.

## Fixed within the reproduced scope

| Audit ID | Change |
|---|---|
| E12-001 | Décisions supervisées sur l'entraînement externe. Les cibles réservées ne modifient plus le design, les paramètres de réglage ou les importances apprises des arbres et modèles linéaires dans les cas testés. |
| E12-002 | Prétraitement dans les plis. Imputation et mise à l'échelle sont ajustées dans chaque pli de réglage ; un préconditionnement des amplitudes évite le débordement rencontré sur 10^300. |
| E12-003 | Dates indivisibles. Le holdout et les plis internes ne partagent plus de date dans les reproductions ; moins de plis sont utilisés ou la sélection est indisponible si la coupe est impossible. |
| E12-006 | Différences au sein de chaque entité. Les trois branches reproduites — racine unitaire, indécision et différence saisonnière — ne soustraient plus la fin d'une autre entité. |
| E12-007 | Booléens inconnus conservés. Les manques deviennent des 0/1 nullables ; ni l'exception de fillna sur boolean, ni la conversion silencieuse de missing en vrai ne réapparaissent dans les contrôles. |
| E12-008 | Action d'outliers cohérente avec le conseil. Le snippet réévalue la même politique et utilise des masques positionnels ; il conserve les manques et fonctionne avec des index répétés. La winsorisation ne modifie que le consensus. |
| E12-009 | Encodage cible multiclasse et reproductible. Des encodeurs binaires un-contre-tous produisent toutes les colonnes multiclasse ; le type de cible est explicite et la graine fixée. Les lignes sans cible sont transformées séparément. |
| E12-010 | Effet et p-value de matrice cohérents. La matrice d'association Spearman prend sa propre p-value Spearman, même lorsque pairs choisit Pearson. |
| E12-011 | Valeurs critiques ADF conformes à la référence. Le nombre d'observations de régression est utilisé et le terme cubique de MacKinnon rétabli ; les seuils concordent avec statsmodels sur la grille testée. |
| E12-012 | Vraisemblance binomiale profilée. n n'est plus fixé au maximum observé. La recherche entière exhaustive est bornée à 100 000 et annoncée comme telle ; un optimum à cette borne est refusé. |
| E12-013 | Bornes exécutables du bucketing. Les coupures utilisent explicitement np.inf et fonctionnent dans l'espace de noms df/np/pd annoncé. |

## Structural checks

- 500 uneven temporal tables: no date shared across training and holdout or across an internal split.
- 500 planted-extreme columns: 486 removal-advice cases executed faithfully; 972 missing rows were preserved in those removal runs. Detection/advice rules otherwise remain unchanged.
- 500 encoding tables: 167 binary, 167 multiclass, 166 integer-regression targets; correct dimensions, preserved duplicate indexes and identical seeded reruns.

These checks validate structure and execution. They do not establish false-positive control for every distribution, or independently validate 500 real datasets per method.

## Open scientific limits

E12-004 and E12-005 remain open: categorical serial corrections and Kendall IID p-values are not generally calibrated for serial rows. Metadata and a high report alert now make this explicit; adding that alert is not a calibration repair. Forest importance still uses the holdout that scores base models, and outer-training screening/stationarity choices are not fully nested in every inner tuning fold. No independent final-subset performance guarantee is made.

Regularisation uses declared finite grids. The binomial profile search is exhaustive within n<=100,000 and rejects a maximum at its upper limit; it is not an unrestricted MLE over all integers. The common estimation sample for ADF lag selection is retained; only nobs and the critical surfaces were corrected. Historical 1.1 numbers are preserved in VALIDATION.md.

## Reference data and reproduction

The 12 public datasets are pinned to immutable Git revisions; all retrieved bytes were checked against those Git blobs, and SHA-256 values are in [validation-v12-data.json](validation-v12-data.json). Cache filenames include the revision, preventing reuse of old floating-reference caches.

Numerical measures, code hashes, package versions and per-measure provenance are in [validation-v12-results.json](validation-v12-results.json). Local execution used Linux, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0 and statsmodels 0.15.0; other OS/minimum-version combinations were not rerun in this task.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ELESSAR_VALIDATION_DATA=validation-cache ELESSAR_VALIDATION_OUT=validation-results-local.json \
  python -m pytest -q -m validation tests/validation
ruff check src tests docs examples
```

No statistical threshold or existing acceptance bound was relaxed to obtain passing checks. The suite remains distinct from a release decision; the open inference issues and final selection-evaluation design must be addressed before claiming v1.2 readiness.
