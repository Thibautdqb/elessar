"""The see functions: ``see`` (everything) and ``see_column`` (one column)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._utils import dedupe_names, finite, prepare, quiet, resolve_target
from .config import StoneConfig
from .correlations import PAIR_COLUMNS, compute_correlations
from .distributions import _fmt_params, analyze_distribution, is_fittable
from .plots import make_plots
from .quality import alert, check_quality, column_alerts, correlation_alerts, row_order_alert, sort_alerts
from .report import Vision
from .stats import column_stats
from .types import as_numeric, infer_type


def see_column(series: pd.Series, config: StoneConfig | None = None, fit: bool = True) -> dict[str, Any]:
    """Everything about one column: type, statistics, distribution and alerts.

    Never raises on bad data: a failing step is recorded under ``errors``.
    """
    cfg = config or StoneConfig()
    if not isinstance(series, pd.Series):
        raise TypeError(f"expected a pandas Series, got {type(series).__name__}")
    name = str(series.name) if series.name is not None else "value"
    detail: dict[str, Any] = {"name": name, "errors": {}}
    with quiet():
        try:
            info = infer_type(series, cfg)
        except Exception as exc:  # noqa: BLE001
            info = {"semantic_type": "mixed", "pandas_dtype": str(series.dtype), "confidence": 0.0,
                    "reason": "type inference failed", "suggestions": []}
            detail["errors"]["types"] = repr(exc)
        detail.update(info)
        stype = info["semantic_type"]
        try:
            detail["stats"] = column_stats(series, stype, cfg)
        except Exception as exc:  # noqa: BLE001
            detail["stats"] = {}
            detail["errors"]["stats"] = repr(exc)
        if fit and is_fittable(stype):
            try:
                detail["distribution"] = analyze_distribution(as_numeric(series, stype), cfg)
            except Exception as exc:  # noqa: BLE001
                detail["errors"]["distribution"] = repr(exc)
    detail["alerts"] = column_alerts(name, detail, cfg)
    return detail


def analyze(
    df: pd.DataFrame, target: str | None, config: StoneConfig
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame]]:
    """Per-column details and correlations for a prepared DataFrame."""
    details = {col: see_column(df[col], config) for col in df.columns}
    types = {c: d["semantic_type"] for c, d in details.items()}
    try:
        correlations = compute_correlations(df, types, target, config)
    except Exception as exc:  # noqa: BLE001
        correlations = {"top_pairs": pd.DataFrame(columns=PAIR_COLUMNS), "error": pd.DataFrame({"error": [repr(exc)]})}
    return details, correlations


def _summary_row(detail: dict[str, Any]) -> dict[str, Any]:
    st = detail.get("stats", {})
    dist = detail.get("distribution") or {}
    fits = (dist.get("fit") or {}).get("fits") or []
    top = st.get("top_values") or []
    return {
        "column": detail["name"],
        "semantic_type": detail.get("semantic_type"),
        "pandas_dtype": detail.get("pandas_dtype"),
        "count": st.get("count"),
        "pct_missing": st.get("pct_missing"),
        "n_unique": st.get("n_unique"),
        "pct_unique": st.get("pct_unique"),
        "top_value": top[0]["value"] if top else None,
        "top_pct": top[0]["pct"] if top else None,
        "mean": finite(st.get("mean")),
        "std": finite(st.get("std")),
        "min": st.get("min"),
        "p25": st.get("p25"),
        "median": st.get("median"),
        "p75": st.get("p75"),
        "max": st.get("max"),
        "skewness": finite(st.get("skewness")),
        "kurtosis": finite(st.get("kurtosis")),
        "pct_outliers_iqr": st.get("pct_outliers_iqr"),
        "normality": (dist.get("normality") or {}).get("verdict"),
        "best_fit": f"{fits[0]['distribution']}({_fmt_params(fits[0]['params'])})" if fits else None,
        "entropy": st.get("normalized_entropy"),
        "memory_bytes": st.get("memory_bytes"),
        "n_alerts": len(detail.get("alerts", [])),
        "errors": "; ".join(f"{k}: {v}" for k, v in detail.get("errors", {}).items()) or None,
    }


def see(
    df: pd.DataFrame | pd.Series,
    target: str | None = None,
    config: StoneConfig | None = None,
    plots: bool = True,
    out_dir: str | Path = "elessar_report",
    verbose: bool = True,
) -> Vision:
    """Analyse everything about ``df``: types, stats, distributions, correlations, quality, plots.

    Prints a terminal report when ``verbose`` and writes PNG plots to ``out_dir`` when ``plots``.
    """
    cfg = config or StoneConfig()
    original = list(df.columns) if isinstance(df, pd.DataFrame) else []
    target = resolve_target(df, target)
    df = prepare(df)

    details, correlations = analyze(df, target, cfg)
    quality = check_quality(df, details, correlations, cfg, target)
    alerts = list(quality["alerts"])
    for d in details.values():
        alerts.extend(d["alerts"])
    if original and dedupe_names(original) != [str(c) for c in original]:
        alerts.append(alert("medium", None, "duplicate or non-string column names were renamed (a, a.1, ...)"))
    if "error" in correlations:
        alerts.append(alert("high", None, f"correlations failed: {correlations['error'].iloc[0, 0]}"))
    alerts.extend(correlation_alerts(correlations, target, cfg))
    alerts.extend(row_order_alert(details, cfg))
    if target is not None:
        top = details[target].get("stats", {}).get("top_values") or []
        stype = details[target]["semantic_type"]
        if stype in ("boolean", "categorical", "ordinal-candidate") and len(top) >= 2 and top[-1]["pct"] < 10:
            alerts = [a for a in alerts if not (a["column"] == target and a["message"].startswith("imbalanced:"))]
            alerts.append(alert("medium", target, f"imbalanced target: '{top[-1]['value']}' is only "
                                f"{top[-1]['pct']:.1f}% of rows -> stratify splits, use PR-AUC / class weights"))

    plot_paths: list[Path] = []
    plot_errors: list[str] = []
    if plots and df.shape[1]:
        plot_paths = make_plots(df, details, correlations, out_dir, target, cfg, plot_errors)
    for err in plot_errors:
        alerts.append(alert("info", None, f"plot skipped after an error: {err[:100]}"))

    summary = pd.DataFrame([_summary_row(d) for d in details.values()])
    summary = summary.set_index("column") if not summary.empty else summary
    n_cells = df.size
    overview = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "memory_bytes": quality["memory_bytes"],
        "potential_saving_bytes": quality["potential_saving_bytes"],
        "pandas_dtypes": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "semantic_types": {str(k): int(v) for k, v in pd.Series([d["semantic_type"] for d in details.values()],
                                                                 dtype=object).value_counts().items()},
        "n_duplicate_rows": quality["duplicate_rows"]["count"],
        "pct_duplicate_rows": quality["duplicate_rows"]["pct"],
        "pct_missing_cells": 100.0 * float(df.isna().to_numpy().sum()) / n_cells if n_cells else 0.0,
        "n_rows_with_missing": int(df.isna().any(axis=1).sum()) if df.shape[1] else 0,
        "target": target,
    }
    report = Vision(
        overview=overview,
        summary=summary,
        columns=details,
        correlations=correlations,
        quality=quality,
        alerts=sort_alerts(alerts),
        plots=plot_paths,
        target=target,
        alpha=cfg.alpha,
    )
    if verbose:
        report.print()
    return report
