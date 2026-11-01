# Elessar: Python exploratory data analysis and data-quality profiling

Elessar is an open-source Python library distributed as `elessar` on PyPI.
It automates exploratory data analysis (EDA) and data-quality profiling for tabular data.

- Documented package version: 1.1.0.
- Requires Python >=3.10.
- License: MIT.
- Maintainer: thibautdqb.
- PyPI: https://pypi.org/project/elessar/
- Repository: https://github.com/Thibautdqb/elessar
- Documentation: {{BASE}}
- Versioned source: https://github.com/Thibautdqb/elessar/tree/2262600bc335e535297bfc64b05b646e6172590f
- Statistical validation: https://github.com/Thibautdqb/elessar/blob/main/VALIDATION.md

## Installation

```bash
python -m pip install elessar
python -m elessar --version
```

## Complete minimal example

```python
import pandas as pd
import elessar

df = pd.DataFrame({
    "units": [2, 4, 3, 7, 5, 9, 6, 8, 1, 4, 3, 6],
    "channel": ["web", "store", "web", "store"] * 3,
    "revenue": [20, 45, 31, 76, 53, 91, 64, 85, 12, 43, 32, 65],
})
report = elessar.see(df, target="revenue", out_dir="elessar_example", open_report=False)
print(report.summary)
print(report.alerts)
```

This tiny illustrative dataset demonstrates the API, not statistical performance.
With `plots=True` (the default), `see()` writes charts, `report.html` and `report.md`.
Use `plots=False` to skip charts and automatic HTML/Markdown report generation.
Set `open_report=False` in scripts, notebooks or CI to prevent browser opening.

## Command line

```bash
elessar data.csv --no-target --open none
elessar sales.parquet --target revenue --open none
elessar data.csv --no-target --single-file --open none
```

The input files must exist and a named target must be a column in the input.
`--single-file` additionally creates `report_single.html` with charts inlined.
`python -m elessar` also runs the command-line interface.

## Main API

```python
elessar.see(data, target=None, config=None, plots=True, out_dir=None,
            verbose=True, open_report=None, name=None, **load_options)
```

Returns `elessar.Vision`. The `data` argument accepts a pandas DataFrame or Series,
supported tabular objects, or paths to files, URLs, globs and folders.
`target=None` requests no target; `target="ask"` requests an interactive picker.
Loading options include `sheet`, `table`, `sep`, `encoding`, `columns`, `nrows` and `engine`.

Common public calls:

| Call | Result |
| --- | --- |
| `elessar.load(source, config=None, **options)` | `Source` with `.df`, `.origin` and `.full` when applicable; loads data without analysis |
| `elessar.see_column(series, config=None, fit=True)` | dict: semantic type, stats, distribution where applicable, alerts, errors |
| `elessar.see_types(df, config=None)` | DataFrame: semantic types, confidence, certainty, decision path, suggestions |
| `elessar.see_stats(df, config=None)` | DataFrame of per-column descriptive statistics |
| `elessar.see_dist(df, config=None)` | DataFrame of normality tests and top distribution fits |
| `elessar.see_corr(df, target=None, config=None)` | dict of association DataFrames and target analysis |
| `elessar.see_quality(df, config=None)` | dict of data-quality checks including `alerts` |
| `elessar.see_plot(df, out_dir, target=None, config=None)` | list of chart paths |
| `elessar.parse_dates(values, fmt=None, name=None)` | parsed pandas Series |
| `elessar.suggest_targets(df, n_rows=50_000)` | list of ranked target candidates with metadata |
| `elessar.pick_target(df, how="auto")` | interactively chosen column label |

These are common call forms with uncommon/internal parameters omitted; see the
versioned source for complete signatures. Focused `see_*` functions operate on pandas
DataFrames or Series; `see_column` requires a Series.
Use `see` or `load` for other tabular objects and file paths.

## Vision result and exports

- `report.overview`: dataset facts, source and sampling information.
- `report.summary`: pandas DataFrame, one row per column.
- `report.columns`: full details by column.
- `report.alerts`, `report.quality`: diagnostic alerts and quality checks.
- `report.correlations`: association matrices; `['target']` when a target is provided.
- `report.selection`, `report.multivariate`, `report.outliers`, `report.timeseries`: analyses when applicable.
- `report.preparation`: suggested cleaning/transformation steps and code.
- `report.out_dir`, `report.plots`: output directory and chart paths.
- `report.full`: full-pass column summaries when applicable.
- `report.to_json(path)`, `.to_csv(path)`, `.to_markdown(path)`, `.to_html(path)`: exports.

`to_csv()` exports the column summary. HTML exports reference chart files; use
`StoneConfig(single_file_report=True)` during analysis for an additional HTML report
with inlined charts. Suggested preparation code is not automatically applied to source data.

## Configuration

`elessar.StoneConfig` controls thresholds, sampling, parallelism and type overrides.
Examples of valid fields: `n_jobs=1`, `random_state=42`,
`column_types={"region": "categorical"}`, `single_file_report=True`.
For a fixed random state, sampling is deterministic.

Default limits in version 1.1.0:

- `max_rows=1_000_000`: profile sample limit for large tables.
- `sample_size=100_000`: expensive analyses such as correlations and pairwise tests.
- `fit_sample_size=20_000`: distribution fitting and normality tests.

## Inputs and scale

Supported objects include pandas, Polars, PyArrow, Dask and NumPy tabular objects,
dicts and lists of records. Supported files include CSV/TSV, JSON/JSONL, Parquet,
Feather/Arrow, ORC, Excel/ODS, SQLite, SAS, Stata, SPSS, HDF5 and XML, with some
formats requiring additional readers. See the package README for full options and
pickle-loading restrictions.

Large sources may use progressive typing and streaming, with full-pass summaries
and a sampled profile. Dask is used automatically for supported large text inputs;
`engine="pandas"` selects single-process streaming where supported.
Do not describe all large-data results as exact: costly analyses are sampled,
duplicate counting can be skipped when its memory budget is exceeded, and the
actual behaviour depends on the source format. Inspect the report's run details.
`nrows` deliberately limits the input and therefore the scope of reported results.

## Capabilities and interpretation

Semantic column typing; descriptive statistics; distribution fitting; significance-tested
associations; missingness and duplicate checks; target analysis and leakage flags;
feature selection; PCA and factor analysis; outlier and time-series diagnostics;
preparation suggestions; terminal, HTML and Markdown reports with charts.

Inferred types, outlier flags and leakage flags are diagnostics to review.
Associations do not establish causation. The project publishes statistical
validation and known limitations in `VALIDATION.md`.

## Documentation links

- Overview: {{BASE}}
- Quickstart: {{BASE}}quickstart/
- API: {{BASE}}api/
- Data quality: {{BASE}}guides/data-quality/
- Large datasets: {{BASE}}guides/large-datasets/
- Français: {{BASE}}fr/
