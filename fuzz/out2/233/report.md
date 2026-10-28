# elessar report

*0 rows × 1 columns · pandas.DataFrame · read in 0.0s · generated 2026-10-28 11:01 by elessar 0.3.0*

[Key findings](#key-findings) · [Overall statistics](#overall-statistics) · [Data quality](#data-quality) · [Multivariate statistics](#multivariate-statistics) · [Preparation plan](#preparation-plan) · [Attribute statistics](#attribute-statistics) · [Run](#run)

## Key findings

- **HIGH** `nan`: column is entirely missing -\> drop it

## Overall statistics

|   |   |
|---|---|
| **Rows** | 0 |
| **Columns** | 1 |
| **Memory** | 132 B |
| **Potential saving** | 0 B |
| **Pandas dtypes** | object=1 |
| **Semantic types** | empty=1 |
| **Duplicate rows** | 0 (0.0%) |
| **Missing cells** | 0.0% |

## Data quality

| severity | column | message |
|---|---|---|
| high | `nan` | column is entirely missing -\> drop it |

## Multivariate statistics

### Dimensionality reduction

PCA not applicable: needs at least 3 numeric columns and enough rows (0 columns, 0 rows)

### Associations between columns

## Preparation plan

Nothing to clean or transform.

## Attribute statistics

| column | type | missing | distinct | alerts |
|---|---|---|---|---|
| `nan` | empty | 0.0% | 0 | 0 |

<details>
<summary><code>nan</code> · empty</summary>

*object · no non-missing values*

**Profile**

| | |
|---|---|
| type | empty |
| dtype | object |
| values | 0 |
| missing | 0 (0.0%) |
| distinct | 0 (0.0%) |
| memory | 0 B |
</details>

## Run

|   |   |
|---|---|
| **Source** | pandas.DataFrame · read in 0.0s |
| **Processes** | 1 for the analysis, 3 for the charts (14 cores available) |
| **Time** | 0.1s (load 0.0s, columns 0.0s, associations 0.0s, quality 0.0s, charts 0.1s) |
| **Significance** | alpha 0.05, Benjamini-Hochberg false discovery rate per family of tests |
| **Versions** | elessar 0.3.0 · Python 3.12.14 · pandas 3.1.0 · numpy 2.5.4 · scipy 1.18.1 |
