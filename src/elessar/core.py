"""The see functions: ``see`` (everything) and ``see_column`` (one column)."""

from __future__ import annotations

import datetime as dt
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ._utils import dedupe_names, finite, prepare, quiet, resolve_target, safe_name
from .multivariate import multivariate
from .outliers import outliers as find_outliers
from .preparation import plan as preparation_plan
from .selection import feature_selection
from .timeseries import detect as detect_time, timeseries
from .config import StoneConfig
from .correlations import PAIR_COLUMNS, compute_correlations
from .distributions import _fmt_params, analyze_distribution, is_fittable
from .parallel import TaskError, n_workers, run_tasks, shared
from .picker import pick_target
from .plots import make_plots
from .quality import advanced_alerts, alert, duplicates_text, check_quality, column_alerts, column_quality, correlation_alerts, group_alerts, row_order_alert, sort_alerts
from . import recap
from .report import Vision
from .readers import COMPRESSIONS, FORMATS
from .sources import Source, describe_origin, load
from .stats import column_stats
from .types import as_numeric, infer_type


def see_column(series: pd.Series, config: StoneConfig | None = None, fit: bool = True,
               decided: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything about one column: type, statistics, distribution and alerts.

    ``decided``: a type decision already made (e.g. by progressive typing on a big table), used
    instead of walking the decision tree again. Never raises on bad data: a failing step is
    recorded under ``errors``.
    """
    cfg = config or StoneConfig()
    if not isinstance(series, pd.Series):
        raise TypeError(f"expected a pandas Series, got {type(series).__name__}")
    name = str(series.name) if series.name is not None else "value"
    detail: dict[str, Any] = {"name": name, "errors": {}}
    with quiet():
        try:
            info = dict(decided) if decided is not None else infer_type(series, cfg)
        except Exception as exc:  # noqa: BLE001
            info = {"semantic_type": "mixed", "pandas_dtype": str(series.dtype), "confidence": 0.0,
                    "reason": "type inference failed", "suggestions": [], "path": [], "certainty": "uncertain",
                    "certainty_reason": "type inference failed"}
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
    detail = see_column(s, cfg, decided=(st.get("decided") or {}).get(col))
    try:
        found = column_quality(s, detail, cfg)
    except Exception as exc:  # noqa: BLE001
        found = {"error": repr(exc)}
    return detail, found


_QUANTITY_KINDS = {"outliers", "skewed", "heaping", "zero_inflated", "capped"}


def verify_types(df: pd.DataFrame, details: dict[str, dict[str, Any]], found: dict[str, dict[str, Any]],
                 src: Source, cfg: StoneConfig) -> list[dict[str, Any]]:
    """Confront the types decided on part of a streamed table with the full pass over every row.

    Identifiers must be unique over every row, empty and constant columns must stay so, and every
    value must fit its type; a decision that does not survive is corrected and reported.
    """
    full, plan = src.full, src.plan
    out: list[dict[str, Any]] = []
    if full is None or plan is None:
        return out
    n_rows = full.n_rows
    for col, d in details.items():
        stype = d["semantic_type"]
        stats_full = full.columns.loc[col] if col in full.columns.index else None
        redo = None
        if stype == "id":
            n_dup = full.id_duplicates.get(col)
            if n_dup:
                redo = {**d, "semantic_type": "categorical", "certainty": "certain",
                        "reason": f"typed as an identifier on {plan.rows_used:,} rows, but {n_dup:,} values repeat over "
                                  f"all {n_rows:,} rows: a grouping key",
                        "certainty_reason": "checked on every row", "path": d.get("path", []) + ["unique over every row? no"]}
                out.append(alert("medium", col, f"not an identifier after all: {n_dup:,} repeated values over all rows "
                                 "(it looked unique on the rows used for typing)", kind="type_check"))
            elif n_dup == 0:
                d.update(certainty="certain", certainty_reason=f"unique over all {n_rows:,} rows (checked)")
        elif stype in ("empty", "constant") and stats_full is not None:
            ok = (stats_full["count"] == 0) if stype == "empty" else (stats_full["n_distinct"] <= 1)
            if ok:
                d.update(certainty="certain", certainty_reason=f"{stype} over all {n_rows:,} rows (checked)")
            else:
                redo = infer_type(df[col], cfg, sampled=True)
                redo["path"] = d.get("path", []) + [f"{stype} over every row? no"]
                out.append(alert("medium", col, f"not {stype} after all: it has values further in the file "
                                 "(typed again on the sample)", kind="type_check"))
        if redo is not None:
            details[col] = see_column(df[col], cfg, decided=redo)
            found[col] = column_quality(df[col], details[col], cfg)
        bad = full.violations.get(col)
        if bad:
            ex = ", ".join(f"row {r:,}: {v!r}" for r, v in bad["examples"][:3])
            out.append(alert("medium" if bad["pct"] >= 0.1 else "info", col,
                             f"{bad['count']:,} values ({bad['pct']:.2g}%) over all rows do not fit the type "
                             f"{details[col]['semantic_type']} (e.g. {ex})", kind="type_check"))
    rounds = plan.rounds
    left = plan.uncertain
    out.append(alert("info", None, f"types decided progressively on {plan.rows_used:,} rows in {len(rounds)} round"
                     f"{'s' if len(rounds) > 1 else ''} ({plan.stopped})"
                     + (f"; still uncertain: {', '.join(left[:6])}" if left else ""), kind="typing"))
    return out


def _advanced_task(name: str) -> Any:
    """One of the dataset-level analyses that only need the column types (runs in a worker)."""
    st = shared()
    df, types, target, cfg, correlations = st["df"], st["types"], st["target"], st["cfg"], st["correlations"]
    with quiet():
        if name == "multivariate":
            return multivariate(df, types, target, cfg)
        if name == "outliers":
            return find_outliers(df, types, cfg)
        if name == "timeseries":
            exact, info, panel = st.get("exact"), st.get("ts_info"), st.get("panel_df")
            if exact is not None and info is not None:
                frame = panel if info["kind"] == "panel" and panel is not None else df
                return timeseries(frame, types, target, cfg, exact=exact, info=info)
            return timeseries(df, types, target, cfg)
        if name == "selection":
            info = detect_time(df, types)
            ordered = info is not None and info["kind"] != "event log" and info["sorted"]
            return feature_selection(df, target, types, correlations, cfg, time_ordered=ordered)
    raise ValueError(name)


def advanced(
    df: pd.DataFrame, details: dict[str, dict[str, Any]], correlations: dict[str, pd.DataFrame], target: str | None,
    cfg: StoneConfig, workers: int = 1, src: Source | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Multivariate statistics, outliers, time series and feature selection; errors per analysis.

    For a streamed table (``src.full``), the time series are analysed on the exact per-period series
    computed over every row, and a panel on whole entities drawn at random.
    """
    types = {c: d["semantic_type"] for c, d in details.items()}
    names = ["multivariate", "outliers", "timeseries"] + (["selection"] if target is not None else [])
    state = {"df": df, "types": types, "target": target, "cfg": cfg, "correlations": correlations}
    full, plan = (src.full, src.plan) if src is not None else (None, None)
    if full is not None and plan is not None and plan.timeseries is not None and full.series is not None:
        state.update(exact={"series": full.series, "unit": full.time_unit, "entity_fraction": full.panel_fraction},
                     ts_info=plan.timeseries,
                     panel_df=prepare(full.panel_rows) if full.panel_rows is not None else None)
    results, errors = {}, {}
    for name, res in zip(names, run_tasks(_advanced_task, names, min(workers, len(names)), state)):
        if isinstance(res, TaskError):
            errors[name] = res.error
            results[name] = None
        else:
            results[name] = res
    return results, errors


def analyze(
    df: pd.DataFrame, target: str | None, config: StoneConfig, workers: int = 1,
    timings: dict[str, float] | None = None, decided: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    """Per-column details, correlations and per-column quality findings for a prepared DataFrame.

    ``decided``: column types already decided (progressive typing), used instead of the tree.
    """
    started = time.perf_counter()
    details: dict[str, dict[str, Any]] = {}
    found: dict[str, dict[str, Any]] = {}
    cols = list(df.columns)
    state = {"df": df, "cfg": config, "decided": decided}
    for col, res in zip(cols, run_tasks(_column_task, cols, workers, state)):
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


DEFAULT_ROOT = "elessar_report"


def dataset_stem(name: str | None) -> str:
    """A folder-safe name for a dataset: the file name without its extensions (``trans.asc`` -> ``trans``,
    ``sales.csv.gz`` -> ``sales``), or ``dataframe`` when there is no name."""
    if not name:
        return "dataframe"
    base = Path(str(name)).name
    known = set(FORMATS) | set(COMPRESSIONS) | {".asc"}
    while Path(base).suffix.lower() in known and Path(base).stem:
        base = Path(base).stem
    return safe_name(base) or "dataframe"


def report_dir(name: str | None, out_dir: str | Path | None = None, root: str | Path = DEFAULT_ROOT,
               day: dt.date | None = None) -> Path:
    """Where a run writes its report: ``out_dir`` when given, else ``<root>/<name>_<YYYY-MM-DD>``.

    One folder per dataset and per day, so reports of different tables never overwrite each other;
    running the same table again the same day refreshes that day's report.
    """
    if out_dir is not None:
        return Path(out_dir)
    return Path(root) / f"{dataset_stem(name)}_{(day or dt.date.today()):%Y-%m-%d}"


def _variable_name(obj: Any, frame: Any) -> str | None:
    """The caller's variable name for ``obj`` (``see(sales)`` -> ``"sales"``), when it is a table object."""
    if isinstance(obj, (str, Path, Source)) or frame is None or frame.f_back is None:
        return None
    try:
        for key, value in frame.f_back.f_locals.items():
            if value is obj and not key.startswith("_"):
                return key
    except Exception:  # noqa: BLE001 - only a nicety
        return None
    return None


def see(
    data: Any,
    target: str | None = None,
    config: StoneConfig | None = None,
    plots: bool = True,
    out_dir: str | Path | None = None,
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
    (``report.html`` and ``report.md``, charts embedded) to ``out_dir``; by default a folder of
    its own, ``elessar_report/<name>_<YYYY-MM-DD>/`` (see :func:`report_dir`).

    ``open_report``: ``None`` opens the HTML recap only for a person at a terminal (never in CI,
    tests or pipes); ``True`` or ``"html"`` always opens it, ``"md"`` opens the Markdown one,
    ``False`` never opens. ``name`` titles the recap and names its folder (default: the file name,
    or the name of the variable holding the DataFrame).
    """
    cfg = config or StoneConfig()
    variable = _variable_name(data, inspect.currentframe())
    started = time.perf_counter()
    timings: dict[str, float] = {}
    src = load(data, cfg, **load_options)
    timings["load"] = time.perf_counter() - started
    df, full = src.df, src.full
    name = name or src.name or variable  # the file name first; for a DataFrame, its variable name
    out_dir = report_dir(name, out_dir)
    if isinstance(target, str) and target == "ask" and "ask" not in map(str, df.columns):
        target = pick_target(df)
    original = list(df.columns) if isinstance(df, pd.DataFrame) else []
    target = resolve_target(df, target)
    df = prepare(df)
    workers = n_workers(cfg, df.size, df.shape[1])

    decided = src.plan.types if src.plan is not None else None
    details, correlations, found = analyze(df, target, cfg, workers, timings, decided)
    type_alerts = verify_types(df, details, found, src, cfg) if src.plan is not None else []
    step = time.perf_counter()
    exact_dups = full.duplicate_rows if full is not None else None
    if full is not None and exact_dups is None:  # too many rows to hash: say so instead of showing the sample's count
        exact_dups = {"not_counted": True}
    quality = check_quality(df, details, correlations, cfg, target, found, exact_dups)
    alerts = list(quality["alerts"]) + type_alerts
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
    if src.origin.get("headerless"):
        alerts.append(alert("medium", None, "the file has no header row (its first line is data): columns are numbered "
                            "0, 1, ... -> pass names=[...] when reading it", kind="headerless"))
    if full is not None:
        alerts.append(alert("info", None, f"profiled on a uniform random sample of {len(df):,} of {full.n_rows:,} rows "
                            f"({100 * len(df) / full.n_rows:.1f}%); row counts, missing values, min/max/mean/std, "
                            "distinct counts and duplicate rows are exact over every row", kind="sampled"))
    timings["quality"] = time.perf_counter() - step

    step = time.perf_counter()
    adv_workers = n_workers(cfg, df.size * 4, 4)  # each analysis is heavy whatever the table size
    extra, adv_errors = advanced(df, details, correlations, target, cfg, adv_workers, src)
    for step_name, err in adv_errors.items():
        alerts.append(alert("info", None, f"{step_name} analysis skipped after an error: {err[:120]}"))
    ts = extra.get("timeseries")
    axis = [c for c in (ts["time_column"].split(" + ") if ts else []) if c in details and c != target]
    for c in axis:  # a year is a position in time, not a quantity: no fit, no outliers, no transformation
        details[c].pop("distribution", None)
        details[c]["alerts"] = [a for a in details[c].get("alerts", []) if a.get("kind") not in _QUANTITY_KINDS]
        (extra.get("outliers") or {}).get("columns", {}).pop(c, None)
        out = extra.get("outliers") or {}
        for key in ("points", "summary"):
            frame = out.get(key)
            if frame is not None and "column" in frame:
                out[key] = frame[frame["column"] != c]
    alerts = [a for a in alerts if not (a["column"] in axis and a.get("kind") in _QUANTITY_KINDS)]
    try:
        prep = preparation_plan(df, details, quality, correlations, target, cfg, extra.get("outliers"),
                                extra.get("multivariate"), extra.get("timeseries"))
    except Exception as exc:  # noqa: BLE001 - the plan is advice: never let it stop the report
        prep = {"steps": [], "counts": {}}
        alerts.append(alert("info", None, f"preparation plan skipped after an error: {exc!r}"[:140]))
    if axis:
        quantity = {"Bucketing", "Transformation", "Rescale", "Normalization", "Remove outliers", "Reassign outliers"}
        prep["steps"] = [st for st in prep["steps"] if not (st["column"] in axis and st["action"] in quantity)]
    alerts.extend(advanced_alerts(extra, prep, target))
    if ts is not None and ts["kind"] != "event log":
        alerts = [a for a in alerts if not a["message"].startswith("rows are ordered")]  # expected: it is a time series
    timings["advanced"] = time.perf_counter() - step

    step = time.perf_counter()
    plot_paths: list[Path] = []
    plot_errors: list[str] = []
    plot_index: list[dict[str, Any]] = []
    plot_workers = 0
    if plots and df.shape[1]:
        n_charts = len(df.columns) + 8  # about one per column, plus the dataset and target charts
        # a chart costs ~0.2 s whatever the table size: processes pay off as soon as there are a dozen
        plot_workers = n_workers(cfg, cfg.parallel_min_cells if n_charts >= 12 else df.size, n_charts)
        plot_paths = make_plots(df, details, correlations, out_dir, target, cfg, plot_errors, plot_index, plot_workers,
                                advanced=extra)
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
        "duplicates_text": duplicates_text(quality["duplicate_rows"]),
        "pct_missing_cells": pct_missing_cells,
        "n_rows_with_missing": int(df.isna().any(axis=1).sum()) if df.shape[1] else 0,
        "target": target,
        "source": describe_origin(src),
        "typing": None if src.plan is None else {
            "rows_used": src.plan.rows_used, "rounds": src.plan.rounds, "stopped": src.plan.stopped,
            "sampling": src.plan.sampling},
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
        out_dir=Path(out_dir),
        timings=timings,
        multivariate=extra.get("multivariate") or {},
        outliers=extra.get("outliers") or {},
        timeseries=extra.get("timeseries"),
        selection=extra.get("selection"),
        preparation=prep,
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
