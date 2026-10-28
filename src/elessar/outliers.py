"""Outlier points of every numeric column, found three ways and listed row by row.

- **3 sigma**: more than 3 standard deviations from the mean (assumes a roughly normal column);
- **boxplot (Tukey)**: outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR], the points a boxplot draws;
- **modified z (Hampel)**: |0.6745 (x - median) / MAD| > 3.5, robust to the outliers themselves.

A point flagged by at least two methods is a consensus outlier. For each column elessar then
proposes to **remove** (values so far out they are likely errors) or **reassign** them (cap at
the boxplot fences: the tail is real but too influential).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._utils import quiet
from .config import StoneConfig
from .types import NUMERIC_TYPES, as_numeric

METHODS = ("3 sigma", "boxplot", "modified z")


def column_outliers(x: pd.Series, cfg: StoneConfig) -> dict[str, Any] | None:
    """Flags of the three methods for one numeric column, with the listed points."""
    v = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(v) < 8 or v.nunique() < 3:
        return None
    mean, std = float(v.mean()), float(v.std())
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    iqr = q3 - q1
    med = float(v.median())
    mad = float((v - med).abs().median())
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (v - mean) / std if std > 0 else pd.Series(0.0, index=v.index)
    mz = 0.6745 * (v - med) / mad if mad > 0 else pd.Series(0.0, index=v.index)
    flags = pd.DataFrame({"3 sigma": z.abs() > 3, "boxplot": (v < lo) | (v > hi) if iqr > 0 else False,
                          "modified z": mz.abs() > 3.5})
    n_methods = flags.sum(axis=1)
    consensus = n_methods >= 2
    robust = (v - med) / (iqr / 1.349 if iqr > 0 else (mad * 1.4826 or std or 1.0))
    picked = v[n_methods >= 1]
    points = pd.DataFrame({
        "row": picked.index, "value": picked.to_numpy(), "z": z[picked.index].to_numpy(),
        "robust_z": robust[picked.index].to_numpy(),
        "methods": [", ".join(m for m in METHODS if flags.at[i, m]) for i in picked.index],
        "n_methods": n_methods[picked.index].to_numpy(),
    })
    points = points.reindex(points["robust_z"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    extreme = int((robust.abs() > 10).sum())
    n_cons = int(consensus.sum())
    if n_cons == 0:
        action, why = "keep", "no point is flagged by two methods"
    elif extreme and extreme <= max(3, 0.002 * len(v)):
        action, why = "remove", (f"{extreme} value{'s are' if extreme > 1 else ' is'} more than 10 robust standard "
                                 "deviations away: likely entry or unit errors")
    else:
        action, why = "reassign", (f"{n_cons} consensus outliers form a real but influential tail: cap them at the "
                                   "boxplot fences (winsorize) or use robust models")
    return {
        "n": len(v), "mean": mean, "std": std, "median": med, "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi,
        "sigma_low": mean - 3 * std, "sigma_high": mean + 3 * std,
        "counts": {m: int(flags[m].sum()) for m in METHODS}, "n_consensus": n_cons,
        "pct_consensus": 100.0 * n_cons / len(v), "n_extreme": extreme, "action": action, "why": why,
        "points": points, "robust": robust, "consensus": consensus,
    }


def outliers(df: pd.DataFrame, types: dict[str, str], cfg: StoneConfig) -> dict[str, Any]:
    """Per-column outlier analysis of the numeric columns, plus one table of every flagged point."""
    per_column: dict[str, Any] = {}
    with quiet():
        for col in df.columns:
            stype = types.get(col)
            if stype not in NUMERIC_TYPES and stype != "numeric_string":
                continue
            try:
                res = column_outliers(as_numeric(df[col], stype), cfg)
            except (TypeError, ValueError):
                res = None
            if res is not None:
                per_column[col] = res
    frames = [r["points"].assign(column=c) for c, r in per_column.items() if len(r["points"])]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["row", "value", "z", "robust_z", "methods", "n_methods", "column"])
    summary = pd.DataFrame([{
        "column": c, "3 sigma": r["counts"]["3 sigma"], "boxplot": r["counts"]["boxplot"],
        "modified z": r["counts"]["modified z"], "consensus": r["n_consensus"], "pct_consensus": r["pct_consensus"],
        "lower_fence": r["lower_fence"], "upper_fence": r["upper_fence"], "action": r["action"],
    } for c, r in per_column.items()])
    return {"columns": per_column, "points": table[["column", "row", "value", "z", "robust_z", "methods", "n_methods"]],
            "summary": summary}


__all__ = ["METHODS", "column_outliers", "outliers"]
