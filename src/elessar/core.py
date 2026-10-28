"""The see functions: ``see`` (everything) and ``see_column`` (one column)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ._utils import dedupe_names, finite, prepare, quiet, resolve_target
from .config import StoneConfig
from .correlations import PAIR_COLUMNS, compute_correlations
from .distributions import _fmt_params, analyze_distribution, is_fittable
from .parallel import TaskError, n_workers, run_tasks, shared
from .picker import pick_target
from .plots import make_plots
from .quality import alert, check_quality, column_alerts, column_quality, correlation_alerts, group_alerts, row_order_alert, sort_alerts
from . import recap
from .report import Vision
from .sources import describe_origin, load
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


def _column_task(col: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Everything about one column of the shared frame: details, then quality findings."""
    st = shared()
    s, cfg = st["df"][col], st["cfg"]
    detail = see_column(s, cfg)
    try:
        found = column_quality(s, detail, cfg)
    except Exception as exc:  # noqa: BLE001
        found = {"error": repr(exc)}
    return detail, found


def analyze(
    df: pd.DataFrame, target: str | None, config: StoneConfig, workers: int = 1,
    timings: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Per-column details, correlations and per-column quality findings for a prepared DataFrame."""
    started = time.perf_counter()
    details: dict[str, dict[str, Any]] = {}
    found: dict[str, dict[str, Any]] = {}
    cols = list(df.columns)
    for col, res in zip(cols, run_tasks(_column_task, cols, workers, {"df": df, "cfg": config})):
        if isinstance(res, TaskError):
            details[col] = {"name": col, "semantic_type": "mixed", "pandas_dtype": str(df[col].dtype),
                            "confidence": 0.0, "reason": "analysis failed", "suggestions": [], "stats": {},
                            "errors": {"column": res.error}, "alerts": []}
            found[col] = {"error": res.error}
        else:
            details[col], found[col] = res
    types = {c: d["semantic_type"] for c, d in details.items()}
    corr_started = time.perf_counter()
    try:
        correlations = compute_correlations(df, types, target, config, workers)
    except Exception as exc:  # noqa: BLE001
        correlations = {"top_pairs": pd.DataFrame(columns=PAIR_COLUMNS), "error": pd.DataFrame({"error": [repr(exc)]})}
    if timings is not None:
        timings["columns"] = corr_started - started
        timings["associations"] = time.perf_counter() - corr_started
    return details, correlations, found


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
    data: Any,
    target: str | None = None,
    config: StoneConfig | None = None,
    plots: bool = True,
    out_dir: str | Path = "elessar_report",
    verbose: bool = True,
    open_report: bool | str | None = None,
    name: str | None = None,
    **load_options: Any,
) -> Vision:
    """Analyse everything about ``data``: types, stats, distributions, correlations, quality, plots.

    ``data`` is a DataFrame or anything :func:`elessar.load` reads: a path, URL, glob or folder
    (CSV, Parquet, Excel, JSON, ...), or a polars / pyarrow / dask / NumPy object. Tables bigger
    than ``config.max_rows`` are streamed: exact counts over every row, the profile on a uniform
    sample. ``load_options`` (``sheet``, ``table``, ``sep``, ``encoding``, ``columns``, ``nrows``,
    ``engine``) are passed to :func:`elessar.load`.

    ``target``: the column to explain, or ``"ask"`` to pick it in a window (or at the terminal).

    Prints a terminal report when ``verbose``. When ``plots``, writes the charts and a recap
    (``report.html`` and ``report.md``, charts embedded) to ``out_dir``.

    ``open_report``: ``None`` opens the HTML recap only for a person at a terminal (never in CI,
    tests or pipes); ``True`` or ``"html"`` always opens it, ``"md"`` opens the Markdown one,
    ``False`` never opens. ``name`` titles the recap (default: the file name).
    """
    cfg = config or StoneConfig()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    src = load(data, cfg, **load_options)
    timings["load"] = time.perf_counter() - started
    df, full = src.df, src.full
    name = name or src.name
    if isinstance(target, str) and target == "ask" and "ask" not in map(str, df.columns):
        target = pick_target(df)
    original = list(df.columns) if isinstance(df, pd.DataFrame) else []
    target = resolve_target(df, target)
    df = prepare(df)
    workers = n_workers(cfg, df.size, df.shape[1])

    details, correlations, found = analyze(df, target, cfg, workers, timings)
    step = time.perf_counter()
    exact_dups = full.duplicate_rows if full is not None else None
    quality = check_quality(df, details, correlations, cfg, target, found, exact_dups)
    alerts = list(quality["alerts"])
    for d in details.values():
        alerts.extend(d["alerts"])
    if original and dedupe_names(original) != [str(c) for c in original]:
        alerts.append(alert("medium", None, "duplicate or non-string column names were renamed (a, a.1, ...)"))
    if "error" in correlations:
        alerts.append(alert("high", None, f"correlations failed: {correlations['error'].iloc[0, 0]}"))
    alerts.extend(correlation_alerts(correlations, target, cfg, quality["duplicate_columns"]))
    alerts.extend(row_order_alert(details, cfg))
    if target is not None:
        top = details[target].get("stats", {}).get("top_values") or []
        stype = details[target]["semantic_type"]
        if stype in ("boolean", "categorical", "ordinal-candidate") and len(top) >= 2 and top[-1]["pct"] < 10:
            alerts = [a for a in alerts if not (a["column"] == target and a["message"].startswith("imbalanced:"))]
            alerts.append(alert("medium", target, f"imbalanced target: '{top[-1]['value']}' is only "
                                f"{top[-1]['pct']:.1f}% of rows -> stratify splits, use PR-AUC / class weights"))
    if full is not None:
        alerts.append(alert("info", None, f"profiled on a uniform random sample of {len(df):,} of {full.n_rows:,} rows "
                            f"({100 * len(df) / full.n_rows:.1f}%); row counts, missing values, min/max/mean/std, "
                            "distinct counts and duplicate rows are exact over every row", kind="sampled"))
    timings["quality"] = time.perf_counter() - step

    step = time.perf_counter()
    plot_paths: list[Path] = []
    plot_errors: list[str] = []
    plot_index: list[dict[str, Any]] = []
    plot_workers = 0
    if plots and df.shape[1]:
        n_charts = len(df.columns) + 8  # about one per column, plus the dataset and target charts
        # a chart costs ~0.2 s whatever the table size: processes pay off as soon as there are a dozen
        plot_workers = n_workers(cfg, cfg.parallel_min_cells if n_charts >= 12 else df.size, n_charts)
        plot_paths = make_plots(df, details, correlations, out_dir, target, cfg, plot_errors, plot_index, plot_workers)
    for err in plot_errors:
        alerts.append(alert("info", None, f"plot skipped after an error: {err[:100]}"))
    timings["charts"] = time.perf_counter() - step

    summary = pd.DataFrame([_summary_row(d) for d in details.values()])
    summary = summary.set_index("column") if not summary.empty else summary
    n_cells = df.size
    pct_missing_cells = 100.0 * float(df.isna().to_numpy().sum()) / n_cells if n_cells else 0.0
    if full is not None and not full.columns.empty and full.n_rows:
        pct_missing_cells = 100.0 * float(full.columns["n_missing"].sum()) / (full.n_rows * len(full.columns))
    overview = {
        "n_rows": full.n_rows if full is not None else len(df),
        "n_columns": df.shape[1],
        "n_rows_profiled": len(df),
        "sampled": full is not None,
        "memory_bytes": quality["memory_bytes"],
        "potential_saving_bytes": quality["potential_saving_bytes"],
        "pandas_dtypes": {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()},
        "semantic_types": {str(k): int(v) for k, v in pd.Series([d["semantic_type"] for d in details.values()],
                                                                 dtype=object).value_counts().items()},
        "n_duplicate_rows": quality["duplicate_rows"]["count"],
        "pct_duplicate_rows": quality["duplicate_rows"]["pct"],
        "pct_missing_cells": pct_missing_cells,
        "n_rows_with_missing": int(df.isna().any(axis=1).sum()) if df.shape[1] else 0,
        "target": target,
        "source": describe_origin(src),
        "workers": workers,
        "chart_workers": plot_workers,
    }
    report = Vision(
        overview=overview,
        summary=summary,
        columns=details,
        correlations=correlations,
        quality=quality,
        alerts=sort_alerts(group_alerts(alerts, cfg)),
        plots=plot_paths,
        plot_index=plot_index,
        target=target,
        alpha=cfg.alpha,
        name=name,
        full=full.columns if full is not None else None,
        timings=timings,
    )
    if plots and df.shape[1]:
        step = time.perf_counter()
        report.recap = recap.write(report, out_dir, single_file=cfg.single_file_report)
        timings["report"] = time.perf_counter() - step
    timings["total"] = time.perf_counter() - started
    if verbose:
        report.print()
    recap.maybe_open(report, open_report, sys.stdout if verbose else None)
    return report
