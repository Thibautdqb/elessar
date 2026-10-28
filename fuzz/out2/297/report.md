# elessar report

*1 rows × 1 columns · pandas.DataFrame · read in 0.0s · generated 2026-10-28 11:02 by elessar 0.3.0*

[Key findings](#key-findings) · [Overall statistics](#overall-statistics) · [Target analysis](#target-analysis) · [Data quality](#data-quality) · [Multivariate statistics](#multivariate-statistics) · [Preparation plan](#preparation-plan) · [Attribute statistics](#attribute-statistics) · [Run](#run)

## Key findings

- **MEDIUM** `column`: constant column carries no information -\> drop it

## Overall statistics

|   |   |
|---|---|
| **Rows** | 1 |
| **Columns** | 1 |
| **Memory** | 91 B |
| **Potential saving** | 0 B |
| **Pandas dtypes** | str=1 |
| **Semantic types** | constant=1 |
| **Duplicate rows** | 0 (0.0%) |
| **Missing cells** | 0.0% |
| **Target** | column |

## Target analysis

Target: `column` (constant)

## Data quality

| severity | column | message |
|---|---|---|
| medium | `column` | constant column carries no information -\> drop it |

## Multivariate statistics

### Dimensionality reduction

PCA not applicable: needs at least 3 numeric columns and enough rows (0 columns, 1 rows)

### Associations between columns

## Preparation plan

Nothing to clean or transform.

## Attribute statistics

| column | type | missing | distinct | alerts |
|---|---|---|---|---|
| `column` | constant | 0.0% | 1 | 0 |

<details>
<summary><code>column</code> · constant</summary>

*str · single value: 'delta beta gamma beta alpha alpha'*

**Profile**

| | |
|---|---|
| type | constant |
| dtype | str |
| values | 1 |
| missing | 0 (0.0%) |
| distinct | 1 (100.0%) |
| memory | 41 B |
</details>

## Run

|   |   |
|---|---|
| **Source** | pandas.DataFrame · read in 0.0s |
| **Processes** | 1 for the analysis, 3 for the charts (14 cores available) |
| **Time** | 0.3s (load 0.0s, columns 0.0s, associations 0.0s, quality 0.0s, charts 0.1s) |
| **Significance** | alpha 0.05, Benjamini-Hochberg false discovery rate per family of tests |
| **Versions** | elessar 0.3.0 · Python 3.12.14 · pandas 3.1.0 · numpy 2.5.4 · scipy 1.18.1 |
