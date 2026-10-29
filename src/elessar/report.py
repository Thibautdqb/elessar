"""``Vision``: the result object, its terminal rendering and its exports."""

from __future__ import annotations

import json
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd

from ._utils import fmt_bytes, fmt_num, fmt_p, jsonable, truncate
from .distributions import _fmt_params

_SEVERITY_LABEL = {"high": "HIGH", "medium": "MEDIUM", "info": "info"}


def _pct(x: Any) -> str:
    return "-" if x is None or pd.isna(x) else f"{x:.1f}%"


def _str(x: Any) -> str:
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)


def _kv(counts: dict[Any, Any], limit: int = 8) -> str:
    items = list(counts.items())
    text = ", ".join(f"{k}={v}" for k, v in items[:limit])
    return text + (f", … (+{len(items) - limit})" if len(items) > limit else "")


def _join_wrap(parts: list[str], width: int, indent: str) -> list[str]:
    """Join ``parts`` with ' · ', wrapping between parts to stay within ``width``."""
    lines, cur = [], ""
    for part in (p for p in parts if p):
        part = truncate(part, width - len(indent))
        candidate = f"{cur} · {part}" if cur else part
        if cur and len(indent) + len(candidate) > width:
            lines.append(indent + cur)
            cur = part
        else:
            cur = candidate
    if cur:
        lines.append(indent + cur)
    return lines


def _table(headers: list[str], rows: list[list[str]], width: int, drop_order: tuple[str, ...] = (), indent: str = "  ") -> list[str]:
    """Fixed-width text table that drops low-priority columns, then truncates, to fit ``width``."""
    headers, rows = list(headers), [list(r) for r in rows]
    right = {i for i, h in enumerate(headers) if h in _NUMERIC_HEADERS}

    def widths() -> list[int]:
        return [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def total(ws: list[int]) -> int:
        return len(indent) + sum(ws) + 2 * (len(ws) - 1)

    for name in drop_order:
        if total(widths()) <= width or name not in headers:
            continue
        i = headers.index(name)
        headers.pop(i)
        for r in rows:
            r.pop(i)
        right = {j - (j > i) for j in right if j != i}
    ws = widths()
    while total(ws) > width and max(ws) > 8:
        ws[ws.index(max(ws))] -= 1

    def line(cells: list[str]) -> str:
        out = []
        for i, (c, w) in enumerate(zip(cells, ws)):
            c = truncate(c, w)
            out.append(c.rjust(w) if i in right else c.ljust(w))
        return (indent + "  ".join(out)).rstrip()

    return [line(headers), indent + "  ".join("─" * w for w in ws)] + [line(r) for r in rows]


_NUMERIC_HEADERS = {"missing", "unique", "mean/top", "std/top%", "min", "max", "skew", "value", "association", "%",
                    "n", "p_adj", "AUC", "MI"}


def _pair_rows(pairs: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    rows = [
        [r.col_a, r.col_b, r.method, f"{r.value:+.3f}",
         f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]" if pd.notna(r.ci_low) else "", f"{r.n:,}", fmt_p(r.p_adjusted),
         "*" if r.significant else ""]
        for r in pairs.itertuples(index=False)
    ]
    return ["column A", "column B", "method", "value", "95% CI", "n", "p_adj", "sig"], rows


def _target_rows(assoc: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    rows = [
        [r.feature, r.feature_type, r.method, f"{r.association:+.3f}", r.test, f"{r.n:,}", fmt_p(r.p_adjusted),
         "*" if r.significant else "", f"{r.auc:.3f}" if pd.notna(r.auc) else "",
         f"{r.mutual_info:.3f}" if pd.notna(r.mutual_info) else "", "LEAKAGE?" if r.leakage_suspect else ""]
        for r in assoc.itertuples(index=False)
    ]
    return ["feature", "type", "method", "association", "test", "n", "p_adj", "sig", "AUC", "MI", "flag"], rows


_ASCII = str.maketrans({"═": "=", "─": "-", "│": "|", "▸": ">", "·": ".", "→": "->", "←": "<-", "×": "x", "≥": ">=",
                        "≤": "<=", "±": "+/-", "…": "...", "²": "2", "χ": "chi", "≈": "~", "—": "-", "–": "-"})


def _ascii(text: str) -> str:
    """The terminal report with box-drawing and typographic characters spelled in ASCII; anything else
    the console cannot show becomes ``?``."""
    return text.translate(_ASCII).encode("ascii", "replace").decode("ascii")


@dataclass
class Vision:
    """Everything elessar found about a DataFrame.

    Attributes:
        overview: dataset-level facts (shape, memory, dtypes, duplicates, missing share).
        summary: one row per column, one column per key metric.
        columns: column name -> full detail dict (type, stats, distribution, alerts, errors).
        correlations: name -> DataFrame (``pearson``, ``spearman``, ``kendall``, ``cramers_v``,
            ``eta``, ``top_pairs``, ``target``).
        quality: every data-quality check result.
        alerts: list of ``{severity, column, message}`` sorted by severity.
        plots: paths of the PNG files written.
        plot_index: one ``{column, other, kind, path}`` record per chart.
        recap: paths of the recap files written after the run (``{"md": ..., "html": ...}``).
        target: name of the target column, if any.
        alpha: significance level used for the tests.
        name: dataset name shown in the recap title (e.g. the file name).
        full: when the profile ran on a sample of a bigger table, exact figures over every row
            (one row per column: count, missing, distinct, mean, std, min, max, ...); else None.
        out_dir: the folder the charts and the report page were written to.
        timings: seconds spent in each step (load, columns, associations, quality, advanced, charts, report).
        multivariate: PCA (``pca``), factor analysis (``factor``), groups of redundant columns (``groups``),
            rows that are outliers on several columns (``outliers``), covariance matrix.
        outliers: per numeric column the three methods (3 sigma, boxplot, modified z), the action
            (keep / reassign / remove) and every flagged point (``points``: column, row, value, ...).
        timeseries: the time structure (single series, panel, event log), trend, stationarity,
            seasonality, treatment and validation folds; None when the table is not time-indexed.
        selection: feature selection against the target (decision tree, random forest, lasso,
            ridge, filter): ``table`` of importances and votes, ``keep`` / ``borderline`` / ``drop``.
        preparation: the cleaning and transformation plan (``steps``: kind, action, column, why, code).
    """

    overview: dict[str, Any]
    summary: pd.DataFrame
    columns: dict[str, dict[str, Any]]
    correlations: dict[str, pd.DataFrame]
    quality: dict[str, Any]
    alerts: list[dict[str, Any]]
    plots: list[Path] = field(default_factory=list)
    plot_index: list[dict[str, Any]] = field(default_factory=list)
    recap: dict[str, Path] = field(default_factory=dict)
    target: str | None = None
    alpha: float = 0.05
    name: str | None = None
    full: pd.DataFrame | None = None
    timings: dict[str, float] = field(default_factory=dict)
    out_dir: Path | None = None
    multivariate: dict[str, Any] = field(default_factory=dict)
    outliers: dict[str, Any] = field(default_factory=dict)
    timeseries: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    preparation: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        o = self.overview
        return (f"Vision({o['n_rows']:,} rows x {o['n_columns']} columns, "
                f"{len(self.alerts)} alerts, {len(self.plots)} plots)")

    def __str__(self) -> str:
        return self.render()

    # ------------------------------------------------------------------ terminal
    def print(self, file: TextIO | None = None, width: int | None = None) -> None:
        """Print the full plain-text report (fits the terminal width by default)."""
        stream = file or sys.stdout
        text = self.render(width)
        try:
            print(text, file=stream)
        except UnicodeEncodeError:  # a console without UTF-8 (Windows cp1252): plain ASCII look-alikes
            print(_ascii(text), file=stream)

    def render(self, width: int | None = None) -> str:
        """Return the plain-text report as a string."""
        w = width or min(max(shutil.get_terminal_size((120, 40)).columns, 60), 160)
        o = self.overview
        out = ["═" * w, f" elessar report  ·  {o['n_rows']:,} rows x {o['n_columns']} columns", "═" * w]
        for title, lines in (
            ("OVERVIEW", self._overview_lines(w)),
            (self._alerts_title(), self._alert_lines(w)),
            ("COLUMN SUMMARY", self._summary_lines(w)),
            ("COLUMN DETAILS", self._detail_lines(w)),
            ("CORRELATIONS", self._corr_lines(w)),
            ("TARGET ANALYSIS", self._target_lines(w)),
            ("FEATURE SELECTION", self._selection_lines(w)),
            ("TIME SERIES", self._timeseries_lines(w)),
            ("MULTIVARIATE", self._multivariate_lines(w)),
            ("PREPARATION PLAN", self._preparation_lines(w)),
            ("PLOTS", self._plot_lines(w)),
        ):
            if lines:
                out += ["", title, "─" * len(title), *lines]
        return "\n".join(out)

    @staticmethod
    def _wrap(rows: list[tuple[str, str]], w: int) -> list[str]:
        return [line for k, v in rows for line in textwrap.wrap(v, w, initial_indent=f"  {k:<16}",
                                                                 subsequent_indent=" " * 18) or [f"  {k}"]]

    def _selection_lines(self, w: int) -> list[str]:
        sel = self.selection
        if not sel:
            return []
        rows = [("task", f"{sel['task']}, {sel['n_rows']:,} rows, {sel['cv']} CV {sel['scoring']}: "
                         + ", ".join(f"{m} {v:.3f}" for m, v in sel["scores"].items())),
                ("keep", ", ".join(map(str, sel["keep"])) or "-"),
                ("borderline", ", ".join(map(str, sel["borderline"])) or "-"),
                ("drop", ", ".join(map(str, sel["drop"])) or "-")]
        if sel.get("excluded_leaks"):
            rows.append(("set aside", "leakage suspects: " + ", ".join(sel["excluded_leaks"])))
        return self._wrap(rows, w)

    def _timeseries_lines(self, w: int) -> list[str]:
        ts = self.timeseries
        if not ts:
            return []
        rows = [("structure", f"{ts['kind']} on {ts['time_column']}, {ts['frequency']} steps, "
                              f"{ts['start']:%Y-%m-%d} -> {ts['end']:%Y-%m-%d}, {ts['n_steps']:,} steps"
                              + (f", entity {ts['entity']}" if ts.get("entity") else ""))]
        for c, a in ts["analyses"].items():
            rows.append((truncate(c, 16), ", ".join(x for x in (
                f"trend {a.get('trend')}", (a.get("stationarity") or {}).get("verdict", ""),
                f"season {a['period']} ({a['seasonality']}, {a['seasonal_strength']:.2f})" if a.get("period") else "no season",
            ) if x)))
        rows += [(st["step"].lower(), st["advice"]) for st in ts["treatment"]]
        return self._wrap(rows, w)

    def _multivariate_lines(self, w: int) -> list[str]:
        mv = self.multivariate or {}
        pca, fa, mo = mv.get("pca") or {}, mv.get("factor") or {}, mv.get("outliers") or {}
        rows = []
        if pca:
            rows.append(("PCA", f"{pca.get('verdict')}: {pca.get('reason')}"))
        if fa:
            rows.append(("factors", f"{fa.get('verdict')}: {fa.get('reason')}"))
            rel = fa.get("reliability") or {}
            for f, cols in (fa.get("defined_by") or {}).items():
                a = (rel.get(f) or {}).get("alpha")
                alpha = f" (alpha {a:.2f}, {rel[f]['label']})" if a is not None and a == a else ""
                rows.append(("", f"{f}: {', '.join(map(str, cols))}{alpha}"))
        for g in mv.get("groups") or []:
            rows.append(("redundant group", f"{', '.join(map(str, g['columns']))} (|rho| {g['mean_abs_spearman']:.2f})"))
        if mo.get("verdict") == "computed":
            rows.append(("outlier rows", f"{mo['n_both']:,} flagged by robust Mahalanobis and isolation forest"
                                         + ("" if mo.get("reliable", True) else " (data far from elliptical: indicative)")))
        out = self.outliers or {}
        if out.get("summary") is not None and len(out["summary"]):
            sm = out["summary"]
            rows.append(("outlier points", ", ".join(f"{r['column']} {r['consensus']:,} ({r['action']})"
                                                     for _, r in sm[sm["consensus"] > 0].iterrows()) or "none"))
        return self._wrap(rows, w)

    def _preparation_lines(self, w: int) -> list[str]:
        steps = (self.preparation or {}).get("steps") or []
        if not steps:
            return []
        lines = []
        for st in steps[:40]:
            head = f"{st['action']}" + (f" [{truncate(st['column'], 24)}]" if st["column"] else "")
            lines += textwrap.wrap(f"{head}: {st['why']}", w, initial_indent="  ", subsequent_indent="      ")
        if len(steps) > 40:
            lines.append(f"  ... {len(steps) - 40} more steps in the HTML report (with the code)")
        return lines

    def _overview_lines(self, w: int) -> list[str]:
        o = self.overview
        saving = o.get("potential_saving_bytes") or 0
        rows = [("source", o["source"])] if o.get("source") else []
        rows += [
            ("rows", f"{o['n_rows']:,}" + (f"  (profiled on a uniform sample of {o['n_rows_profiled']:,})"
                                          if o.get("sampled") else "")),
            ("columns", f"{o['n_columns']:,}"),
            ("memory", fmt_bytes(o["memory_bytes"]) + (f"  (suggested dtypes save ~{fmt_bytes(saving)})" if saving else "")),
            ("pandas dtypes", _kv(o["pandas_dtypes"])),
            ("semantic types", _kv(o["semantic_types"])),
            ("duplicate rows", o.get("duplicates_text") or f"{o['n_duplicate_rows']:,} ({o['pct_duplicate_rows']:.1f}%)"),
            ("missing cells", f"{o['pct_missing_cells']:.1f}%  ({o['n_rows_with_missing']:,} rows with at least one)"),
        ]
        if self.target:
            rows.append(("target", self.target))
        return [line for k, v in rows for line in textwrap.wrap(v, w, initial_indent=f"  {k:<16}",
                                                                 subsequent_indent=" " * 18) or [f"  {k}"]]

    def _alerts_title(self) -> str:
        counts = {s: sum(a["severity"] == s for a in self.alerts) for s in _SEVERITY_LABEL}
        detail = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        return f"ALERTS ({len(self.alerts)}{': ' + detail if detail else ''})"

    def _alert_lines(self, w: int) -> list[str]:
        if not self.alerts:
            return ["  no issue found"]
        col_w = min(max((len(str(a["column"] or "-")) for a in self.alerts), default=1), 24)
        lines = []
        for a in self.alerts:
            head = f"  {_SEVERITY_LABEL[a['severity']]:<7} {truncate(a['column'] or '-', col_w):<{col_w}}  "
            body = textwrap.wrap(a["message"], max(w - len(head), 30)) or [""]
            lines.append(head + body[0])
            lines += [" " * len(head) + b for b in body[1:]]
        return lines

    def _summary_lines(self, w: int) -> list[str]:
        if self.summary.empty:
            return ["  (no columns)"]
        headers = ["column", "type", "dtype", "missing", "unique", "mean/top", "std/top%", "min", "max", "skew", "best fit · normality"]
        rows = []
        for col, r in self.summary.iterrows():
            numeric = r["mean"] is not None and pd.notna(r["mean"])
            fit = " · ".join(x for x in (_str(r["best_fit"]).split("(")[0], _str(r["normality"])) if x)
            rows.append([
                str(col),
                str(r["semantic_type"]),
                str(r["pandas_dtype"]),
                _pct(r["pct_missing"]),
                fmt_num(r["n_unique"]) if r["n_unique"] is not None and pd.notna(r["n_unique"]) else "-",
                fmt_num(r["mean"]) if numeric else truncate(r["top_value"] if r["top_value"] is not None else "-", 18),
                fmt_num(r["std"]) if numeric else _pct(r["top_pct"]),
                fmt_num(r["min"]) if numeric else "",
                fmt_num(r["max"]) if numeric else "",
                f"{r['skewness']:.2f}" if r["skewness"] is not None and pd.notna(r["skewness"]) else "",
                fit,
            ])
        return _table(headers, rows, w, drop_order=("dtype", "best fit · normality", "skew", "min", "max"))

    def _detail_lines(self, w: int) -> list[str]:
        lines: list[str] = []
        for name, d in self.columns.items():
            lines.append(truncate(f"▸ {name}  ·  {d.get('semantic_type')}  ·  {d.get('pandas_dtype')}  ·  {d.get('reason')}", w))
            for parts in _detail_parts(d):
                lines += _join_wrap(parts, w, "    ")
            lines.append("")
        return lines[:-1]

    def _corr_lines(self, w: int) -> list[str]:
        c = self.correlations
        pairs, top = c.get("pairs"), c.get("top_pairs")
        if top is None:
            return []
        lines = []
        if pairs is not None and not pairs.empty:
            n_sig = int(pairs["significant"].sum())
            lines += textwrap.wrap(f"{len(pairs):,} pairs tested, {n_sig:,} significant after Benjamini-Hochberg "
                                   f"correction (alpha {self.alpha:g}); strongest shown below", w,
                                   initial_indent="  ", subsequent_indent="  ")
        if top.empty:
            lines.append("  no pair above the association threshold")
        else:
            headers, rows = _pair_rows(top.head(15))
            lines += _table(headers, rows, w, drop_order=("95% CI", "n"))
            if len(top) > 15:
                lines.append(f"  … {len(top) - 15} more in report.correlations['top_pairs']")
        vif = c.get("vif")
        if vif is not None and not vif.empty:
            lines += [""] + _join_wrap(["multicollinearity (VIF): " + ", ".join(
                f"{k} {'inf' if np.isinf(v) else f'{v:.1f}'}" for k, v in vif["vif"].head(10).items())], w, "  ")
        deps = c.get("linear_dependencies")
        if deps is not None and not deps.empty:
            lines += ["", "  exact linear dependencies:"]
            for r in deps.itertuples(index=False):
                lines += _join_wrap([r.equation], w, "    ")
        miss = c.get("missingness")
        if miss is not None and not miss.empty and miss["significant"].any():
            lines += ["", "  missingness that depends on other columns (evidence against MCAR):"]
            for r in miss[miss["significant"]].drop_duplicates("column").head(10).itertuples(index=False):
                lines += _join_wrap([f"{r.column} <- {r.depends_on}: {r.detail}", f"p_adj {fmt_p(r.p_adjusted)}"], w, "    ")
        return lines

    def _target_lines(self, w: int) -> list[str]:
        assoc = self.correlations.get("target")
        if not self.target or assoc is None:
            return []
        lines = [f"  target: {self.target}  ({self.columns[self.target]['semantic_type']})"]
        if assoc.empty:
            return lines + [f"  no usable feature association with '{self.target}'"]
        headers, rows = _target_rows(assoc.head(20))
        lines += _table(headers, rows, w, drop_order=("test", "type", "MI", "n"))
        n_sig = int(assoc["significant"].sum())
        lines += textwrap.wrap(f"{n_sig}/{len(assoc)} features significantly associated (BH-adjusted p < "
                               f"{self.alpha:g})", w, initial_indent="  ", subsequent_indent="  ")
        mt = self.correlations.get("missing_target")
        if mt is not None and not mt.empty:
            lines += ["", "  is the target different when a feature is missing?"]
            for r in mt.head(10).itertuples(index=False):
                flag = "yes" if r.significant else "no "
                lines += _join_wrap([f"{flag}  {r.feature} ({r.pct_missing:.1f}% missing): {r.detail}",
                                     f"p_adj {fmt_p(r.p_adjusted)}"], w, "    ")
        return lines

    def _plot_lines(self, w: int) -> list[str]:
        if not self.plots:
            return []
        folder = self.plots[0].parent
        names = [p.name for p in self.plots]
        lines = [f"  {len(names)} PNG files in {folder}/"]
        return lines + _join_wrap(names, w, "    ")

    # ------------------------------------------------------------------ exports
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict of the whole report."""
        return jsonable({
            "overview": self.overview,
            "summary": self.summary,
            "columns": self.columns,
            "correlations": {k: _matrix_dict(v) for k, v in self.correlations.items()},
            "quality": self.quality,
            "alerts": self.alerts,
            "plots": [str(p) for p in self.plots],
        })

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def to_csv(self, path: str | Path) -> Path:
        """Write the per-column ``summary`` table as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.summary.to_csv(path)
        return path

    def to_markdown(self, path: str | Path) -> Path:
        """The recap as Markdown (key findings, target, quality, associations, columns, with charts)."""
        from .recap import build, render_markdown

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(build(self, path.parent)), encoding="utf-8")
        return path

    def to_html(self, path: str | Path) -> Path:
        """The same recap as a standalone HTML page (charts referenced next to it)."""
        from .recap import build, render_html

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_html(build(self, path.parent), self.name or "elessar report"), encoding="utf-8")
        return path


def _matrix_dict(df: pd.DataFrame) -> Any:
    if df.index.equals(pd.RangeIndex(len(df))):
        return jsonable(df.to_dict(orient="records"))
    return jsonable(df.to_dict(orient="index"))


def _overview_md(o: dict[str, Any]) -> dict[str, str]:
    return {
        "rows": f"{o['n_rows']:,}" + (f" ({o['n_rows_profiled']:,} profiled, uniform random sample)"
                                       if o.get("sampled") else ""),
        "columns": f"{o['n_columns']:,}",
        "memory": fmt_bytes(o["memory_bytes"]),
        "potential saving": fmt_bytes(o.get("potential_saving_bytes") or 0),
        "pandas dtypes": _kv(o["pandas_dtypes"], 50),
        "semantic types": _kv(o["semantic_types"], 50),
        "duplicate rows": o.get("duplicates_text") or f"{o['n_duplicate_rows']:,} ({o['pct_duplicate_rows']:.1f}%)",
        "missing cells": f"{o['pct_missing_cells']:.1f}%",
        **({"target": o["target"]} if o.get("target") else {}),
    }


def _detail_parts(d: dict[str, Any]) -> list[list[str]]:
    """Compact human lines (as lists of parts) describing one column."""
    st = d.get("stats", {})
    stype = d.get("semantic_type")
    lines: list[list[str]] = []
    unique_ish = (st.get("pct_unique") or 0) > 90
    if st:
        lines.append([
            f"missing {st['n_missing']:,} ({st['pct_missing']:.1f}%)",
            f"unique {st['n_unique']:,} ({st['pct_unique']:.1f}%)",
            f"memory {fmt_bytes(st['memory_bytes'])}",
            f"mode {truncate(st['modes'][0], 30)!s} ×{st['mode_freq']:,}" if st.get("modes") and not unique_ish else "",
        ])
    if "mean" in st:
        ci = (f"95% CI [{fmt_num(st['ci95_mean_low'])}, {fmt_num(st['ci95_mean_high'])}]"
              if st.get("ci95_mean_low") is not None else "")
        mci = (f"[{fmt_num(st['ci95_median_low'])}, {fmt_num(st['ci95_median_high'])}]"
               if st.get("ci95_median_low") is not None else "")
        lines.append([f"mean {fmt_num(st['mean'])} ± {fmt_num(st['std'])}", ci, f"median {fmt_num(st['median'])} {mci}".strip(),
                      f"trimmed {fmt_num(st['trimmed_mean'])}", f"IQR {fmt_num(st['iqr'])}", f"MAD {fmt_num(st['mad'])}",
                      f"CV {fmt_num(st['cv'])}" if st.get("cv") is not None else "",
                      f"geo mean {fmt_num(st['geometric_mean'])}" if st.get("geometric_mean") else ""])
        lines.append([f"min {fmt_num(st['min'])}", *(f"p{q} {fmt_num(st[f'p{q}'])}" for q in (1, 5, 25, 75, 95, 99)),
                      f"max {fmt_num(st['max'])}", f"sum {fmt_num(st['sum'])}"])
        lines.append([
            f"skew {fmt_num(st['skewness'], 3)}", f"kurtosis {fmt_num(st['kurtosis'], 3)}",
            f"outliers IQR {st['pct_outliers_iqr']:.1f}% · |z|>3 {st['pct_outliers_z']:.1f}% · mod-z {st['pct_outliers_mz']:.1f}%",
            f"zeros {st['n_zeros']:,}", f"negatives {st['n_negative']:,}", f"inf {st['n_inf']:,}" if st["n_inf"] else "",
            f"decimals {st['n_decimals']}" if st.get("n_decimals") is not None else "",
            f"heaping ×{st['heaping_index']:.1f}" if st.get("heaping_index") else "",
            f"KDE modes {st['n_modes_kde']}" if st.get("n_modes_kde") else "",
            f"bimodality coef {st['bimodality_coefficient']:.2f}" if st.get("bimodality_coefficient") is not None else "",
            "monotonic ↑" if st.get("monotonic_increasing") else "monotonic ↓" if st.get("monotonic_decreasing") else "",
            f"lag-1 autocorr {st['autocorr_lag1']:.2f} (p {fmt_p(st['autocorr_lag1_p'])})" if st.get("autocorr_lag1") is not None else "",
        ])
    dist = d.get("distribution") or {}
    norm = dist.get("normality") or {}
    if norm.get("verdict"):
        lines.append([f"normality: {norm['verdict']}", f"Shapiro p {fmt_p(norm.get('shapiro_p'))}",
                      f"D'Agostino p {fmt_p(norm.get('dagostino_p'))}", f"Jarque-Bera p {fmt_p(norm.get('jarque_bera_p'))}",
                      f"Anderson p {fmt_p(norm.get('anderson_p'))}" if norm.get("anderson_p") is not None else ""])
    fit = dist.get("fit") or {}
    if fit.get("fits"):
        lines.append([f"shape: {fit['description']}"])
        lines.append([
            f"#{i + 1} {f['distribution']}({_fmt_params(f['params'])}) AIC {fmt_num(f['aic'], 6)} KS p {fmt_p(f['ks_pvalue'])}"
            + (f" [{f['note']}]" if f["note"] else "")
            for i, f in enumerate(fit["fits"])
        ])
    if "cardinality" in st:
        lines.append([f"cardinality {st['cardinality']:,}", f"entropy {st['entropy']:.2f} bits ({st['normalized_entropy']:.2f} norm.)",
                      f"gini {st['gini_impurity']:.2f}", f"imbalance ×{fmt_num(st['imbalance_ratio'])}",
                      f"rare {st['n_rare_categories']} ({st['pct_rows_rare']:.1f}% rows)" if st["n_rare_categories"] else "",
                      f"true {st['pct_true']:.1f}% [{st['ci95_pct_true_low']:.1f}, {st['ci95_pct_true_high']:.1f}]"
                      if st.get("pct_true") is not None else "",
                      f"uniform? chi² p {fmt_p(st['uniform_chi2_p'])}" if st.get("uniform_chi2_p") is not None else ""])
    if st.get("top_values") and not unique_ish and stype not in ("numeric_continuous", "datetime", "text", "multi_label"):
        lines.append(["top: " + ", ".join(f"{truncate(v['value'], 25)} {v['pct']:.1f}%" for v in st["top_values"][:6])])
    if st.get("inconsistent_labels"):
        lines.append(["inconsistent labels: " + "; ".join(" / ".join(map(repr, g)) for g in st["inconsistent_labels"][:3])])
    if "span" in st:
        lines.append([f"{st['min']} → {st['max']}", f"span {st['span']}", f"freq {st['inferred_freq']}",
                      f"dup timestamps {st['n_duplicated_timestamps']:,}", f"future {st['n_future']:,}" if st["n_future"] else "",
                      "monotonic ↑" if st.get("monotonic_increasing") else "",
                      f"trend in rows per {st['counts_resolution']}: tau {st['counts_trend_tau']:+.2f} "
                      f"(p {fmt_p(st['counts_trend_p'])})" if st.get("counts_trend_tau") is not None else ""])
        if st.get("largest_gaps"):
            g = st["largest_gaps"][0]
            lines.append([f"largest gap {g['gap']} ({g['start']} → {g['end']})"])
        for key in ("by_weekday", "by_month", "by_hour"):
            if key in st:
                lines.append([f"{key.replace('_', ' ')}: " + _kv(st[key], 24)])
    if "words_mean" in st:
        lines.append([f"length {st['len_min']}–{st['len_max']} (mean {st['len_mean']:.0f})",
                      f"words mean {st['words_mean']:.1f} (max {st['words_max']})",
                      f"empty {st['n_empty']}" if st["n_empty"] else "",
                      f"digits {st['pct_digits']:.1f}% · upper {st['pct_uppercase']:.1f}% · punct {st['pct_punctuation']:.1f}%"])
        if st.get("top_tokens"):
            lines.append(["top tokens: " + ", ".join(f"{t['token']} ({t['count']})" for t in st["top_tokens"][:8])])
    elif "len_mean" in st and "cardinality" not in st:
        lines.append([f"length {st['len_min']}–{st['len_max']} (mean {st['len_mean']:.1f})",
                      f"duplicates {st['n_duplicates']:,}" if "n_duplicates" in st else ""])
    if "labels_per_row_mean" in st:
        lines.append([f"{st['n_labels']:,} distinct labels", f"{st['labels_per_row_mean']:.1f} labels per row "
                      f"(max {st['labels_per_row_max']})", f"single label {st['pct_single_label']:.0f}%",
                      f"separator {st['separator']!r}"])
        lines.append(["top labels: " + ", ".join(f"{truncate(x['label'], 25)} {x['pct_rows']:.0f}%"
                                                 for x in st["top_labels"][:8])])
    if st.get("top_domains"):
        lines.append([f"{st['n_domains']} domains: " + ", ".join(f"{x['domain']} ({x['count']})" for x in st["top_domains"][:5])])
    if st.get("python_kinds"):
        lines.append(["python types: " + _kv(st["python_kinds"])])
    if st.get("top_keys"):
        lines.append(["top keys: " + ", ".join(f"{k['key']} ({k['count']})" for k in st["top_keys"][:8])])
    for sug in d.get("suggestions", []):
        lines.append([f"{sug['flag']}: {sug['code']}"])
    for step, err in d.get("errors", {}).items():
        lines.append([f"ERROR in {step}: {err}"])
    return lines
