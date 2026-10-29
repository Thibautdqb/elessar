"""The recap: one document (Markdown and HTML) with key findings first and every chart in place.

The recap is built once as a list of blocks, then rendered twice. Inline text is a list of
``(style, text)`` spans (``""``, ``"code"``, ``"b"`` or ``"i"``), never pre-formatted markup, so column
names and values are escaped correctly for each format whatever characters they contain.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, Union
from urllib.parse import quote

import numpy as np
import pandas as pd

from . import __version__
from ._utils import fmt_bytes, fmt_num, fmt_p, truncate
from .distributions import _fmt_params
from .preparation import CLEANING, TRANSFORMATION
from .report import _overview_md, _pair_rows, _target_rows

if TYPE_CHECKING:
    from .report import Vision

Span = tuple[str, str]
Inline = Union[str, list[Span]]  # a plain string is one plain span
Block = tuple[Any, ...]
SECTIONS = {"findings": "Key findings", "overall": "Overall statistics", "target": "Target analysis",
            "selection": "Feature selection", "timeseries": "Time series", "distributions": "Distributions",
            "outliers": "Outliers", "quality": "Data quality", "multivariate": "Multivariate statistics",
            "preparation": "Preparation plan", "columns": "Attribute statistics", "run": "Run"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


def _strength(value: float) -> str:
    """Words for an association on a 0-1 scale (Cohen-style rule of thumb)."""
    a = abs(value)
    return "strongly" if a >= 0.5 else "moderately" if a >= 0.3 else "weakly" if a >= 0.1 else "very weakly"


def code(text: Any) -> Span:
    return ("code", str(text))


def bold(text: Any) -> Span:
    return ("b", str(text))


def _spans(inline: Inline) -> list[Span]:
    if isinstance(inline, str):
        return [("", inline)]
    return inline if isinstance(inline, list) else [("", _txt(inline, 200))]


def _num(x: Any, digits: int = 4) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x) else fmt_num(x, digits)


def _txt(x: Any, width: int = 60) -> str:
    try:
        if x is None or pd.isna(x):
            return "-"
    except (TypeError, ValueError):
        pass
    return truncate(x, width)


def _int(x: Any) -> str:
    return "-" if x is None or pd.isna(x) else f"{int(x):,}"


def _ks(d: float) -> str:
    return f"{d:.3f} ({'close' if d < 0.03 else 'fair' if d < 0.08 else 'poor'})"


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception:  # noqa: BLE001
        return "?"


def _pct(n: Any, pct: Any) -> str:
    return f"{n:,} ({pct:.1f}%)" if n is not None and pct is not None else "-"


def _cards(d: dict[str, Any]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Key figures of one column grouped in small titled tables."""
    st = d.get("stats", {})
    if not st:
        return []
    unique_ish = (st.get("pct_unique") or 0) > 90
    profile = [("type", str(d.get("semantic_type"))), ("dtype", str(d.get("pandas_dtype"))),
               ("values", f"{st.get('count', 0):,}"), ("missing", _pct(st.get("n_missing"), st.get("pct_missing"))),
               ("distinct", _pct(st.get("n_unique"), st.get("pct_unique"))),
               ("memory", fmt_bytes(st.get("memory_bytes") or 0))]
    if st.get("modes") and not unique_ish:
        profile.append(("mode", f"{truncate(st['modes'][0], 30)} ×{st['mode_freq']:,}"))
    cards = [("Profile", profile)]
    if "mean" in st:
        ci = (f"[{_num(st['ci95_mean_low'])}, {_num(st['ci95_mean_high'])}]" if st.get("ci95_mean_low") is not None
              else "-")
        mci = (f"[{_num(st['ci95_median_low'])}, {_num(st['ci95_median_high'])}]"
               if st.get("ci95_median_low") is not None else "-")
        cards.append(("Center", [("mean", _num(st["mean"])), ("95% CI of mean", ci), ("median", _num(st["median"])),
                                 ("95% CI of median", mci), ("trimmed mean", _num(st.get("trimmed_mean"))),
                                 ("geometric mean", _num(st.get("geometric_mean"))),
                                 ("harmonic mean", _num(st.get("harmonic_mean")))]))
        cards.append(("Spread", [("std", _num(st["std"])), ("variance", _num(st.get("var"))), ("sem", _num(st.get("sem"))),
                                 ("IQR", _num(st.get("iqr"))), ("MAD", _num(st.get("mad"))), ("CV", _num(st.get("cv"))),
                                 ("range", _num(st.get("range")))]))
        cards.append(("Quantiles", [("min", _num(st["min"]))] + [(f"p{q}", _num(st.get(f"p{q}"))) for q in
                                                                 (1, 5, 25, 50, 75, 95, 99) if st.get(f"p{q}") is not None]
                      + [("max", _num(st["max"])), ("sum", _num(st.get("sum")))]))
        shape = [("skewness", _num(st.get("skewness"), 3)), ("excess kurtosis", _num(st.get("kurtosis"), 3)),
                 ("KDE modes", str(st.get("n_modes_kde") or "-")),
                 ("bimodality coef.", _num(st.get("bimodality_coefficient"), 3))]
        if st.get("n_decimals") is not None:
            shape.append(("decimals", str(st["n_decimals"])))
        if st.get("heaping_index"):
            shape.append(("heaping", f"×{st['heaping_index']:.1f}"))
        cards.append(("Shape", shape))
        outl = [("IQR fences", f"[{_num(st.get('iqr_lower'))}, {_num(st.get('iqr_upper'))}]"),
                ("outside fences", _pct(st.get("n_outliers_iqr"), st.get("pct_outliers_iqr"))),
                ("|z| > 3", _pct(st.get("n_outliers_z"), st.get("pct_outliers_z"))),
                ("modified z > 3.5", _pct(st.get("n_outliers_mz"), st.get("pct_outliers_mz"))),
                ("zeros", f"{st.get('n_zeros', 0):,}"), ("negatives", f"{st.get('n_negative', 0):,}"),
                ("infinite", f"{st.get('n_inf', 0):,}")]
        cards.append(("Outliers & counts", outl))
        order = [("monotonic", "increasing" if st.get("monotonic_increasing") else "decreasing"
                  if st.get("monotonic_decreasing") else "no")]
        if st.get("autocorr_lag1") is not None:
            order.append(("lag-1 autocorrelation", f"{st['autocorr_lag1']:.3f} (p {fmt_p(st.get('autocorr_lag1_p'))})"))
        cards.append(("Row order", order))
    if "cardinality" in st:
        labels = [("cardinality", f"{st['cardinality']:,}"),
                  ("entropy", f"{st['entropy']:.2f} bits ({st['normalized_entropy']:.2f} of max)"),
                  ("Gini impurity", f"{st['gini_impurity']:.3f}"), ("imbalance", f"×{_num(st['imbalance_ratio'])}"),
                  ("rare labels", f"{st['n_rare_categories']} ({st['pct_rows_rare']:.1f}% of rows)")]
        if st.get("uniform_chi2_p") is not None:
            labels.append(("uniform? chi²", f"p {fmt_p(st['uniform_chi2_p'])}"))
        if st.get("pct_true") is not None:
            labels.append(("true", f"{st['pct_true']:.1f}% [{st['ci95_pct_true_low']:.1f}, {st['ci95_pct_true_high']:.1f}]"))
        cards.append(("Labels", labels))
    if "span" in st:
        tm = [("first", str(st["min"])), ("last", str(st["max"])), ("span", str(st["span"])),
              ("frequency", str(st.get("inferred_freq") or "irregular")),
              ("duplicated timestamps", f"{st.get('n_duplicated_timestamps', 0):,}"),
              ("in the future", f"{st.get('n_future', 0):,}"), ("before 1900", f"{st.get('n_before_1900', 0):,}")]
        if st.get("largest_gaps"):
            g = st["largest_gaps"][0]
            tm.append(("largest gap", f"{g['gap']} ({g['start']} → {g['end']})"))
        if st.get("counts_trend_tau") is not None:
            tm.append((f"trend of rows per {st['counts_resolution']}",
                       f"tau {st['counts_trend_tau']:+.2f} (p {fmt_p(st['counts_trend_p'])})"))
        cards.append(("Time", tm))
    if "words_mean" in st:
        cards.append(("Text", [("length", f"{st['len_min']}–{st['len_max']} (mean {st['len_mean']:.0f})"),
                               ("words", f"mean {st['words_mean']:.1f}, max {st['words_max']}"),
                               ("empty", f"{st.get('n_empty', 0):,}"),
                               ("digits / upper / punct.", f"{st['pct_digits']:.1f}% / {st['pct_uppercase']:.1f}% / "
                                                          f"{st['pct_punctuation']:.1f}%")]))
    if "labels_per_row_mean" in st:
        cards.append(("Lists", [("separator", repr(st["separator"])), ("distinct labels", f"{st['n_labels']:,}"),
                                ("labels per row", f"{st['labels_per_row_mean']:.1f} (max {st['labels_per_row_max']})"),
                                ("single label", f"{st['pct_single_label']:.0f}%")]))
    return cards


def _extra_lines(d: dict[str, Any]) -> list[Inline]:
    """What the cards and tables do not show: label hygiene, time patterns, tokens, fixes, errors."""
    st = d.get("stats", {})
    out: list[Inline] = []
    if st.get("inconsistent_labels"):
        out.append("Inconsistent labels: " + "; ".join(" / ".join(map(repr, g)) for g in st["inconsistent_labels"][:5]))
    for key in ("by_weekday", "by_month", "by_hour"):
        if key in st:
            out.append(f"Rows {key.replace('_', ' ')}: " + ", ".join(f"{k} {v}" for k, v in st[key].items()))
    if st.get("top_tokens"):
        out.append("Top tokens: " + ", ".join(f"{t['token']} ({t['count']})" for t in st["top_tokens"][:12]))
    if st.get("top_labels"):
        out.append("Top labels: " + ", ".join(f"{truncate(x['label'], 30)} {x['pct_rows']:.0f}% of rows"
                                              for x in st["top_labels"][:12]))
    if st.get("top_domains"):
        out.append(f"{st['n_domains']} domains: " + ", ".join(f"{x['domain']} ({x['count']})" for x in st["top_domains"][:8]))
    if st.get("python_kinds"):
        out.append("Python types: " + ", ".join(f"{k} {v}" for k, v in st["python_kinds"].items()))
    if st.get("top_keys"):
        out.append("Top keys: " + ", ".join(f"{k['key']} ({k['count']})" for k in st["top_keys"][:10]))
    if (st.get("rare_categories") or []) and isinstance(st["rare_categories"], list):
        out.append("Rare labels: " + ", ".join(truncate(r, 24) for r in st["rare_categories"][:12]))
    for sug in d.get("suggestions", []):
        out.append([bold(sug["flag"]), ("", ": "), code(sug["code"])] + ([("", f" ({sug['note']})")] if sug.get("note") else []))
    for step, err in d.get("errors", {}).items():
        out.append([bold(f"Error in {step}"), ("", f": {err}")])
    return out


# ------------------------------------------------------------------------------------------- building
class _Recap:
    """Collects blocks and knows where each chart is (paths relative to the recap file)."""

    def __init__(self, vision: Vision, base: Path) -> None:
        self.v = vision
        self.base = base.resolve()
        self.blocks: list[Block] = []

    def rel(self, path: Path) -> str:
        return Path(os.path.relpath(Path(path).resolve(), self.base)).as_posix()

    def charts(self, kinds: tuple[str, ...], column: str | None = None, any_column: bool = False) -> list[tuple[str, str]]:
        out = []
        for rec in self.v.plot_index:
            if rec["kind"] in kinds and (any_column or rec["column"] == column):
                label = rec["kind"] if rec["column"] is None else " vs ".join(
                    str(c) for c in (rec["column"], rec.get("other")) if c is not None)
                out.append((self.rel(rec["path"]), label))
        return out

    def add(self, *block: Any) -> None:
        self.blocks.append(block)

    # sections ---------------------------------------------------------------------------------
    def build(self, title: str) -> list[Block]:
        v, o = self.v, self.v.overview
        self.add("h", 1, title, "top")
        meta = f"{o['n_rows']:,} rows × {o['n_columns']} columns"
        if o.get("source"):
            meta += f" · {o['source']}"
        self.add("p", [("i", f"{meta} · generated {dt.datetime.now():%Y-%m-%d %H:%M} by elessar {__version__}")])
        if o.get("sampled"):
            self.add("callout", [bold("Sampled profile. "), ("", f"The table has {o['n_rows']:,} rows: statistics, tests "
                                 f"and charts use a uniform random sample of {o['n_rows_profiled']:,} rows "
                                 f"({100 * o['n_rows_profiled'] / o['n_rows']:.1f}%). Row counts, missing values, "
                                 "min / max / mean / std, distinct counts, duplicate rows and type checks are exact over "
                                 "every row ("), ("link", "exact"), ("", "exact figures"),
                                 ("", "); time series are analysed on their exact per-period aggregates, never on the "
                                      "sample. Row-order statistics (lag-1 autocorrelation, monotonicity) compare "
                                      "consecutive sampled rows.")])
        toc = (["findings", "overall"] + (["target"] if v.target else []) + (["selection"] if v.selection else [])
               + (["timeseries"] if v.timeseries else []) + (["distributions"] if self.fitted() else [])
               + (["outliers"] if (v.outliers or {}).get("columns") else [])
               + ["quality", "multivariate", "preparation", "columns", "run"])
        self.add("toc", [(key, SECTIONS[key]) for key in toc])
        self.findings()
        self.overall_section()
        if v.target:
            self.target_section()
        if v.selection:
            self.selection_section()
        if v.timeseries:
            self.timeseries_section()
        if self.fitted():
            self.distribution_section()
        if (v.outliers or {}).get("columns"):
            self.outlier_section()
        self.quality_section()
        self.association_section()
        self.preparation_section()
        self.column_section()
        self.run_section()
        return self.blocks

    def overall_section(self) -> None:
        v, o = self.v, self.v.overview
        self.add("h", 2, SECTIONS["overall"], "overall")
        self.add("table", ["", ""], [[[bold(k.capitalize())], v_] for k, v_ in _overview_md(o).items()], "kv")
        if v.full is not None:
            self.exact_section()
        self.figures(self.charts(("overview",)))
        summary = v.summary
        if summary is not None and not summary.empty:
            num = summary[summary["mean"].notna()] if "mean" in summary else summary.iloc[:0]
            if not num.empty:
                self.add("h", 3, "Numeric columns at a glance", None)
                self.add("table", ["column", "count", "missing", "mean", "std", "min", "median", "max", "skew", "kurtosis",
                                   "outliers", "best fit"],
                         [[[("link", f"col-{list(v.columns).index(c)}"), code(c)], _int(r["count"]), f"{r['pct_missing']:.1f}%",
                           _num(r["mean"]), _num(r["std"]), _num(r["min"]), _num(r["median"]), _num(r["max"]),
                           _num(r["skewness"], 3), _num(r["kurtosis"], 3),
                           "-" if pd.isna(r["pct_outliers_iqr"]) else f"{r['pct_outliers_iqr']:.1f}%", _txt(r["best_fit"])]
                          for c, r in num.iterrows()], "data")
            cat = summary[summary["semantic_type"].isin(["categorical", "boolean", "ordinal-candidate", "multi_label"])]
            if not cat.empty:
                self.add("h", 3, "Categorical columns at a glance", None)
                self.add("table", ["column", "type", "count", "missing", "distinct", "most frequent", "share", "entropy"],
                         [[[("link", f"col-{list(v.columns).index(c)}"), code(c)], _txt(r["semantic_type"]), _int(r["count"]),
                           f"{r['pct_missing']:.1f}%", _int(r["n_unique"]), _txt(r["top_value"], 30),
                           "-" if pd.isna(r["top_pct"]) else f"{r['top_pct']:.1f}%", _num(r["entropy"], 3)]
                          for c, r in cat.iterrows()], "data")
        self.typing_table()

    def typing_table(self) -> None:
        v, o = self.v, self.v.overview
        self.add("h", 3, "How each column was typed", "typing")
        typing = o.get("typing")
        if typing:
            steps = " → ".join(f"{r['rows']:,}" for r in typing["rounds"])
            self.add("p", f"Big table: the types were decided before reading every row, on {typing['rows_used']:,} rows "
                          f"drawn progressively ({steps} rows; "
                          + ("contiguous blocks, so that time order is kept" if typing["sampling"] == "blocks"
                             else "many short blocks at random positions") + f"). Stopped: {typing['stopped']}. "
                          "The full pass then checked every row against these types.")
        self.add("p", [("i", "Each column goes down a decision tree of explicit questions. Certain: every threshold "
                              "test is decided at 95% confidence (Wilson interval of the observed share) and no label "
                              "is likely unseen (Good-Turing unseen mass below 1%). Probable: an identifier, empty or "
                              "constant column, which only every row can prove. Uncertain: a test too close to its "
                              "threshold, too few values, or an ambiguous date order.")])
        rows = []
        for i, (name, d) in enumerate(v.columns.items()):
            path = d.get("path") or []
            rows.append([[("link", f"col-{i}"), code(name)], d.get("semantic_type", ""),
                         [(f"cert-{d.get('certainty', 'certain')}", d.get("certainty", "-"))],
                         d.get("certainty_reason", ""), path[-1] if path else "-", str(len(path))])
        self.add("table", ["column", "type", "certainty", "why", "decided by", "questions"], rows, "data")

    def selection_section(self) -> None:
        v, sel = self.v, self.v.selection
        self.add("h", 2, SECTIONS["selection"], "selection")
        scores = ", ".join(f"{m} {sel['scores'][m]:.3f}" for m in sel["scores"])
        self.add("p", f"{sel['task'].capitalize()} of {v.target} on {sel['n_rows']:,} rows and {sel['n_features']} "
                      f"features ({sel['n_design_columns']} model columns after one-hot encoding). Cross-validated "
                      f"{sel['scoring']} ({sel['cv']}): {scores}.")
        self.add("p", [bold("Keep: "), ("", ", ".join(map(str, sel["keep"])) or "-"), ("", "  ·  "),
                       bold("Borderline: "), ("", ", ".join(map(str, sel["borderline"])) or "-"), ("", "  ·  "),
                       bold("Drop: "), ("", ", ".join(map(str, sel["drop"])) or "-")])
        if sel.get("excluded_leaks"):
            self.add("callout", [bold("Left out as leakage suspects: "), ("", ", ".join(sel["excluded_leaks"])),
                                 ("", " (they predict the target almost perfectly, probably because they are derived "
                                      "from it).")])
        methods = [m for m in ("decision tree", "random forest", "lasso", "ridge", "filter") if m in sel["table"].columns]
        votes = sel["votes"]
        rows = []
        for f, r in sel["table"].iterrows():
            cells: list[Inline] = [[code(f)]]
            for m in methods:
                mark = " ✓" if m in votes.columns and votes.loc[f, m] else ""
                cells.append(f"{r[m]:.2f}{mark}")
            cells += [str(int(r["votes"])), r["decision"]]
            rows.append(cells)
        self.add("table", ["feature"] + methods + ["votes", "decision"], rows, "data")
        self.add("p", [("i", "Importances scaled to the strongest feature (1); ✓ = selected by that method. Decision "
                              f"tree depth {sel.get('tree_depth')} chosen by cross-validation; random forest: permutation "
                              "importance on held-out rows (all one-hot columns of a feature shuffled together); lasso: "
                              "non-zero coefficient at the penalty chosen by cross-validation; ridge: |coefficient| at "
                              "least 10% of the largest; filter: significant univariate association.")])
        self.figures(self.charts(("selection", "lasso_path"), v.target), grid=False)

    def timeseries_section(self) -> None:
        ts = self.v.timeseries
        self.add("h", 2, SECTIONS["timeseries"], "timeseries")
        what = {"single series": "a single time series (one row per timestamp)",
                "panel": f"a panel: one series per {ts.get('entity')} ({ts.get('n_entities', '?')} series)",
                "event log": f"timestamped events, aggregated per {ts.get('aggregated_to')} for the analysis"}[ts["kind"]]
        self.add("p", [("", "The table is "), bold(what), ("", ", indexed by "), code(ts["time_column"]),
                       ("", f": {ts['frequency']} steps, {100 * ts['regularity']:.0f}% regular, "
                            f"{ts['start']:%Y-%m-%d} → {ts['end']:%Y-%m-%d}, {ts['n_steps']:,} steps"
                            + (f", {ts['n_missing_steps']:,} missing steps" if ts.get("n_missing_steps") else "")
                            + ("" if ts["sorted"] else "; rows are not in time order") + ".")])
        basis = (f"every row, aggregated per {ts.get('frequency_analysed')}" if ts.get("exact") else "the profiled rows")
        notes = []
        if ts.get("n_trimmed_periods"):
            notes.append(f"{ts['n_trimmed_periods']} incomplete edge period(s) left out (their counts are biased low)")
        self.add("p", [("i", f"Computed on {basis}. Trend: Mann-Kendall test with the Hamed-Rao correction for "
                              "autocorrelation (the uncorrected p-value of an autocorrelated series is far too small), "
                              "slope by Sen's estimator. Stationarity: ADF (H0 unit root) and KPSS (H0 stationary), "
                              "with and without a linear trend, read together, on the seasonally adjusted series when there "
                              "is a seasonality. Seasonality: a period natural to the frequency is screened on the "
                              "autocorrelation of the detrended series, then kept when its seasonal strength "
                              "(1 - var(remainder) / var(seasonal + remainder)) reaches 0.25: an effect size, since a "
                              "test would be invalidated by the autocorrelation." + (" " + "; ".join(notes) + "." if notes else ""))])
        rows = []
        for c, a in ts["analyses"].items():
            st = a.get("stationarity") or {}
            adf, kpss = st.get("adf") or {}, st.get("kpss") or {}
            mk = a.get("mann_kendall") or {}
            trend = a.get("trend", "-")
            if mk.get("p_value") is not None and np.isfinite(mk.get("p_value", np.nan)):
                trend += (f" (tau {mk['tau']:+.2f}, p {fmt_p(mk['p_value'])}"
                          + (f"; uncorrected {fmt_p(mk['p_value_naive'])}" if mk.get("variance_inflation", 1) > 1.05 else "")
                          + f", slope {_num(mk['slope'])}/step)")
            stat_txt = st.get("verdict", "-")
            if "statistic" in adf and "statistic" in kpss:
                stat_txt += f" (ADF {adf['statistic']:.2f} vs {adf['critical']['5%']:.2f}; KPSS {kpss['statistic']:.3f}, " \
                            f"p {'≤' if kpss['statistic'] >= 0.739 else '≥' if kpss['statistic'] <= 0.347 else '≈'} " \
                            f"{kpss['p_value']:.2f})"
            rows.append([[code(c)], f"{a.get('n', 0):,}", trend, stat_txt,
                         "-" if a.get("acf_lag1") is None else f"{a['acf_lag1']:.2f}",
                         f"{a['period']} ({a['seasonality']}, {a['seasonal_strength']:.2f})" if a.get("period")
                         else a.get("seasonality", "-")])
        self.add("table", ["series", "steps", "trend", "stationarity", "lag-1 autocorr.", "seasonality"], rows, "data")
        panel = ts.get("panel")
        if panel:
            drawn = (f"about {100 * ts['entity_fraction']:.0f}% of the {ts.get('entity')} values kept by hashing their key"
                     if ts.get("entity_fraction") not in (None, 1.0) else "")
            self.add("h", 3, f"Per {ts.get('entity')} ({panel['n_analysed']} of {panel['n_entities']} drawn at random, "
                             "each with its whole history)", None)
            if drawn:
                self.add("p", [("i", f"The table is streamed: {drawn} (a Bernoulli sample of entities), every row of "
                                     "each kept one.")])
            self.add("p", f"Series length per {ts.get('entity')}: median {panel['length']['median']:.0f} steps "
                          f"(min {panel['length']['min']}, max {panel['length']['max']}). Trend shares are after a "
                          "Benjamini-Hochberg correction over the entities; the mean series above mixes each "
                          "entity's dynamics with changes in which entities are present.")
            self.add("table", ["value", "series", "significant upward trend", "significant downward trend",
                               "stationarity verdicts", "median lag-1 autocorr."],
                     [[[code(c)], str(p["n_series"]), f"{p['pct_increasing']:.0f}%", f"{p['pct_decreasing']:.0f}%",
                       ", ".join(f"{k} {v:.0f}%" for k, v in p["stationarity"].items()), f"{p['median_lag1']:.2f}"]
                      for c, p in panel["per_value"].items()], "data")
        self.figures(self.charts(("timeseries",)) + self.charts(("acf", "decomposition"), ts["main"]))
        self.add("h", 3, "Treatment", None)
        self.add("table", ["step", "what to do", "code"],
                 [[st["step"], st["advice"], [code(st["code"])]] for st in ts["treatment"]], "data")
        if ts.get("folds"):
            self.add("h", 3, "Validation folds (expanding window, on the real dates)", None)
            self.add("table", ["fold", "train", "validate", "train steps", "validation steps"],
                     [[str(f["fold"]), f["train"], f["validate"], f"{f['n_train']:,}", f"{f['n_validate']:,}"]
                      for f in ts["folds"]], "data")

    def outlier_section(self) -> None:
        v, out = self.v, self.v.outliers
        self.add("h", 2, SECTIONS["outliers"], "outliers")
        self.add("p", "Three methods per numeric column: 3 sigma (|value - mean| > 3 standard deviations), boxplot "
                      "(outside the Tukey fences Q1 - 1.5 IQR, Q3 + 1.5 IQR) and modified z (Hampel: |0.6745 (value "
                      "- median) / MAD| > 3.5). A consensus outlier is flagged by at least two.")
        summary = out["summary"]
        self.add("table", ["column", "3 sigma", "boxplot", "modified z", "consensus", "fences", "action"],
                 [[[code(r["column"])], f"{r['3 sigma']:,}", f"{r['boxplot']:,}", f"{r['modified z']:,}",
                   f"{r['consensus']:,} ({r['pct_consensus']:.1f}%)",
                   f"[{_num(r['lower_fence'])}, {_num(r['upper_fence'])}]", r["action"]]
                  for _, r in summary.iterrows()], "data")
        self.figures(self.charts(("outliers",)))
        points = out["points"]
        cons = points[points["n_methods"] >= 2]
        if not cons.empty:
            self.add("h", 3, f"Outlier points ({len(cons):,} consensus; the most extreme 40)", None)
            top = cons.reindex(cons["robust_z"].abs().sort_values(ascending=False).index).head(40)
            self.add("table", ["column", "row", "value", "z", "robust z", "flagged by"],
                     [[[code(r["column"])], str(r["row"]), _num(r["value"]), f"{r['z']:+.2f}", f"{r['robust_z']:+.2f}",
                       r["methods"]] for _, r in top.iterrows()], "data")
            self.add("p", [("", "Every flagged point, with the methods that flag it: "), ("file", "outliers.csv"),
                           code("outliers.csv")])
        mo = (v.multivariate or {}).get("outliers") or {}
        if mo.get("verdict") == "computed":
            self.add("h", 3, "Rows that are outliers on several columns at once", None)
            self.add("p", f"Robust Mahalanobis distance ({mo['method']}) over {mo['n_columns']} numeric columns: "
                          f"{mo['n_mahalanobis']:,} rows ({mo['pct_mahalanobis']:.1f}%) beyond the chi-square 99.9% "
                          f"cutoff ({mo['cutoff_distance']:.2f}); {mo['n_both']:,} also isolated by an isolation forest.")
            if mo.get("note"):
                self.add("callout", [bold("Caution: "), ("", mo["note"] + ".")])
            if not mo["top"].empty:
                self.add("table", ["row", "robust distance", "isolation forest", "values driving it"],
                         [[str(r["row"]), f"{r['distance']:.2f}", "yes" if r["isolation_forest"] else "no", r["drivers"]]
                          for _, r in mo["top"].head(20).iterrows()], "data")
            self.figures(self.charts(("multivariate_outliers",)))

    def preparation_section(self) -> None:
        prep = self.v.preparation or {}
        steps = prep.get("steps") or []
        self.add("h", 2, SECTIONS["preparation"], "preparation")
        if not steps:
            self.add("p", "Nothing to clean or transform.")
            return
        counts = prep.get("counts", {})
        self.add("p", "Cleaning: " + ", ".join(f"{a.lower()} {counts[a]}" for a in CLEANING if counts.get(a)) +
                 ".  Transformation: " + ", ".join(f"{a.lower()} {counts[a]}" for a in TRANSFORMATION if counts.get(a)) + ".")
        for kind, title in (("cleaning", "Cleaning"), ("transformation", "Transformation")):
            rows = [[s_["action"], [code(s_["column"])] if s_["column"] else "-", s_["why"],
                     [code(s_["code"])] if s_["code"] else ""] for s_ in steps if s_["kind"] == kind]
            if rows:
                self.add("h", 3, title, None)
                self.add("table", ["action", "column", "why", "code"], rows, "plan")

    def fitted(self) -> list[str]:
        return [c for c, d in self.v.columns.items() if (d.get("distribution") or {}).get("normality")]

    def exact_section(self) -> None:
        full, o = self.v.full, self.v.overview
        self.add("h", 3, f"Exact figures over all {o['n_rows']:,} rows", "exact")
        rows = []
        for col, r in full.iterrows():
            distinct = f"{int(r['n_distinct']):,}" if r["distinct_exact"] else f"≈ {int(r['n_distinct']):,}"
            top = "" if r["top_value"] is None or pd.isna(r["top_count"]) else (
                f"{truncate(r['top_value'], 24)} ({100 * r['top_count'] / o['n_rows']:.1f}%)")
            rows.append([[code(col)], f"{int(r['count']):,}", f"{int(r['n_missing']):,} ({r['pct_missing']:.1f}%)", distinct,
                         _num(r["mean"]), _num(r["std"]), _num(r["min"]), _num(r["max"]), _int(r["n_zeros"]),
                         _int(r["n_negative"]), top])
        self.add("table", ["column", "count", "missing", "distinct", "mean", "std", "min", "max", "zeros", "negatives",
                           "most frequent"], rows, "data")
        self.add("p", [("i", "Distinct counts marked ≈ are estimated (k-minimum-values sketch, about 1.6% relative "
                              "error); every other figure is exact. Means and standard deviations are combined across "
                              "chunks with Chan's parallel algorithm.")])

    def findings(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["findings"], "findings")
        items: list[Inline] = []
        assoc = v.correlations.get("target")
        if v.target and assoc is not None and not assoc.empty:
            sig = assoc[assoc["significant"]]
            for r in sig.head(3).itertuples(index=False):
                extra = f", AUC {r.auc:.2f}" if pd.notna(r.auc) else ""
                items.append([code(r.feature), ("", f" is {_strength(r.association)} associated with "), code(v.target),
                              ("", f" ({r.method} {r.association:.3f}{extra}, p_adj {fmt_p(r.p_adjusted)})")])
            if len(assoc) > len(sig):
                names = ", ".join(assoc[~assoc["significant"]]["feature"].astype(str).head(6))
                n_none = len(assoc) - len(sig)
                items.append([("", f"{n_none} feature{' shows' if n_none == 1 else 's show'} no significant link "
                                   "with "), code(v.target), ("", f": {names}")])
        ranked = sorted(v.alerts, key=lambda a: _SEVERITY_ORDER.get(a["severity"], 9))
        for a in [a for a in ranked if a["severity"] != "info"][:12]:
            prefix = [(f"sev-{a['severity']}", a["severity"].upper()), ("", " ")] + (
                [code(a["column"]), ("", ": ")] if a["column"] else [])
            items.append(prefix + [("", a["message"])])
        if not items:
            items.append("No issue found, and nothing stands out.")
        self.add("ul", items)

    def target_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["target"], "target")
        self.add("p", [("", "Target: "), code(v.target), ("", f" ({v.columns[v.target]['semantic_type']})")])
        self.figures(self.charts(("target_ranking",), v.target))
        assoc = v.correlations.get("target")
        if assoc is not None and not assoc.empty:
            headers, rows = _target_rows(assoc)
            self.add("table", headers, rows, "data")
        mt = v.correlations.get("missing_target")
        if mt is not None and not mt.empty:
            self.add("h", 3, "Is the target different when a feature is missing?", None)
            self.add("table", ["feature", "% missing", "significant", "p_adj", "detail"],
                     [[[code(r.feature)], f"{r.pct_missing:.1f}", "yes" if r.significant else "no", fmt_p(r.p_adjusted),
                       r.detail] for r in mt.itertuples(index=False)], "data")
        self.figures(self.charts(("target",), v.target), grid=True)

    def quality_section(self) -> None:
        v, q = self.v, self.v.quality
        self.add("h", 2, SECTIONS["quality"], "quality")
        if v.alerts:
            ranked = sorted(v.alerts, key=lambda a: _SEVERITY_ORDER.get(a["severity"], 9))
            self.add("table", ["severity", "column", "message"],
                     [[a["severity"], [code(a["column"])] if a["column"] else "-", a["message"]] for a in ranked], "alerts")
        else:
            self.add("p", "No data-quality issue found.")
        self.figures(self.charts(("missing", "nullity_corr")))
        miss = v.correlations.get("missingness")
        if miss is not None and not miss.empty and miss["significant"].any():
            self.add("h", 3, "Missing values that are not random", None)
            sig = miss[miss["significant"]].drop_duplicates("column").head(20)
            self.add("table", ["column", "% missing", "depends on", "detail", "p_adj"],
                     [[[code(r.column)], f"{r.pct_missing:.1f}", [code(r.depends_on)], r.detail, fmt_p(r.p_adjusted)]
                      for r in sig.itertuples(index=False)], "data")
        opts = q.get("dtype_optimizations") or []
        if opts:
            self.add("h", 3, f"Memory: suggested dtypes save {fmt_bytes(q.get('potential_saving_bytes') or 0)}", None)
            self.add("table", ["column", "current", "suggested", "saving", "code"],
                     [[[code(o_["column"])], o_["current_dtype"], o_["suggested_dtype"],
                       f"{fmt_bytes(o_['saving_bytes'])} ({o_['saving_pct']:.0f}%)", [code(o_["code"])]] for o_ in opts],
                     "data")

    def association_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["multivariate"], "multivariate")
        self.reduction()
        self.add("h", 3, "Associations between columns", None)
        pairs, top = v.correlations.get("pairs"), v.correlations.get("top_pairs")
        if pairs is not None and not pairs.empty:
            self.add("p", f"{len(pairs):,} pairs tested, {int(pairs['significant'].sum()):,} significant after "
                          f"Benjamini-Hochberg correction (alpha {v.alpha:g}).")
        self.figures(self.charts(("association", "corr_pearson", "corr_spearman", "corr_cramers_v")), grid=True)
        if top is not None and not top.empty:
            self.add("h", 3, "Strongest associations", None)
            headers, rows = _pair_rows(top.head(30))
            self.add("table", headers, rows, "data")
        deps = v.correlations.get("linear_dependencies")
        if deps is not None and not deps.empty:
            self.add("h", 3, "Exact linear dependencies", None)
            self.add("ul", [[code(r.equation)] for r in deps.itertuples(index=False)])
        vif = v.correlations.get("vif")
        if vif is not None and not vif.empty:
            self.add("h", 3, "Multicollinearity (VIF)", None)
            self.add("table", ["column", "VIF"],
                     [[[code(k)], "inf" if val == float("inf") else f"{val:.2f}"] for k, val in vif["vif"].items()], "data")
        self.figures(self.charts(("scatter",), any_column=True), grid=True)

    def reduction(self) -> None:
        mv = self.v.multivariate or {}
        pca, fa, groups = mv.get("pca") or {}, mv.get("factor") or {}, mv.get("groups") or []
        if not pca:
            return
        self.add("h", 3, "Dimensionality reduction", None)
        if "eigenvalues" in pca:
            self.add("table", ["", ""], [
                [[bold("Numeric columns")], f"{pca['n_variables']} (standardised; target, identifiers and constants left out)"],
                [[bold("Bartlett sphericity")], f"chi² {pca['bartlett_chi2']:,.1f}, p {fmt_p(pca['bartlett_p'])} "
                                                "(significant = the columns share information)"],
                [[bold("KMO adequacy")], f"{pca['kmo']:.2f} ({pca['kmo_label']})"],
                [[bold("Components to keep")], f"{pca['k_parallel']} by parallel analysis, {pca['k_kaiser']} by Kaiser "
                                               f"(eigenvalue > 1), {pca['k80']} / {pca['k90']} / {pca['k95']} for 80 / 90 / "
                                               "95% of the variance"],
                [[bold("PCA")], [bold(pca["verdict"]), ("", f": {pca['reason']}")]],
                [[bold("Factor analysis")], [bold(fa.get("verdict", "-")), ("", f": {fa.get('reason', '')}")]],
            ], "kv")
            pcs = pca.get("top_variables") or {}
            if pcs:
                self.add("ul", [[bold(pc), ("", f" ({100 * pca['explained'][int(pc[2:]) - 1]:.1f}%): "),
                                 ("", ", ".join(map(str, cols)))] for pc, cols in list(pcs.items())[:pca["recommended"] + 1]])
            if pca.get("verdict") == "recommended":
                self.add("code", pca["code"])
        else:
            self.add("p", f"PCA {pca.get('verdict')}: {pca.get('reason')}")
        self.figures(self.charts(("pca", "factors")))
        self.factor_analysis(fa)
        if groups:
            self.add("h", 3, "Groups of redundant columns", None)
            self.add("table", ["columns", "mean |Spearman|", "keep"],
                     [[", ".join(map(str, g["columns"])), f"{g['mean_abs_spearman']:.2f}", [code(g["representative"])]]
                      for g in groups], "data")

    def factor_analysis(self, fa: dict[str, Any]) -> None:
        """The factor analysis step by step: checklist, matrices, reliability, the new factors."""
        if not fa.get("steps"):
            return
        self.add("h", 3, "Factor analysis", None)
        self.add("p", "Principal components of the correlation matrix, varimax rotation, then Cronbach's alpha per "
                      "factor; items are examined and removed one at a time (communality < .40, highest loading "
                      "< .40 or cross-loading), and the analysis is run again after each removal.")
        chip = {True: ("cert-certain", "yes"), False: ("cert-uncertain", "no"), None: ("sev-info", "done"),
                "removed": ("sev-medium", "removed")}
        self.add("table", ["step", "question", "answer", ""],
                 [[str(st["step"]), st["question"], st["answer"], [chip[st["ok"]]]] for st in fa["steps"]], "data")
        load = fa.get("loadings")
        if load is None:
            return
        unrot, h = fa["unrotated"], fa["communality"]
        assign = load.abs().idxmax(axis=1)
        self.add("p", [bold("Component matrices"), ("", " (sorted by size; each item goes to its highest rotated "
                                                         "loading; communality = share of the item explained)")])
        self.add("table", ["item"] + [f"{f} unrotated" for f in unrot.columns] + [f"{f} rotated" for f in load.columns]
                 + ["communality", "factor"],
                 [[[code(c)]] + [f"{x:+.2f}" for x in unrot.loc[c]] + [f"{x:+.2f}" for x in load.loc[c]]
                  + [f"{h[c]:.2f}", assign[c]] for c in load.index], "data")
        rel = fa.get("reliability") or {}
        rows = []
        for f, r in rel.items():
            if len(r["items"]) < 2:
                rows.append([f, ", ".join(map(str, r["items"])), "-", "single item", "-"])
                continue
            deleted = ", ".join(f"{c}: {a:.2f}" + (" (+)" if a - r["alpha"] >= 0.05 else "") for c, a in r["if_deleted"].items())
            rows.append([f, ", ".join(f"{c} (reversed)" if c in r["reversed"] else str(c) for c in r["items"]),
                         f"{r['alpha']:.2f}", r["label"], deleted or "-"])
        if rows:
            self.add("p", [bold("Reliability"), ("", ": Cronbach's alpha per factor (excellent >= .90, very good >= .80, "
                                                     "good >= .70, acceptable >= .60, poor >= .50). (+) marks a deletion "
                                                     "that would raise alpha by .05 or more: delete only if it also makes "
                                                     "sense for the construct; marginal gains are not worth an item.")])
            self.add("table", ["factor", "items", "alpha", "", "alpha if item deleted"], rows, "data")
        desc = fa.get("descriptives") or {}
        if desc:
            self.add("p", [bold("New factors"), ("", " (created only for factors with an acceptable alpha: the mean of "
                                                      "their items)")])
            self.add("table", ["factor", "items", "n", "mean", "median", "mode", "std"],
                     [[[code(k)], ", ".join(map(str, d["items"])), f"{d['n']:,}", f"{d['mean']:.2f}", f"{d['median']:.2f}",
                       f"{d['mode']:.2f}", f"{d['std']:.2f}"] for k, d in desc.items()], "data")
        if fa.get("code"):
            self.add("code", fa["code"])

    def distribution_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["distributions"], "distributions")
        self.add("p", "Normality is tested four ways (Shapiro-Wilk on up to 5,000 values, D'Agostino K², Jarque-Bera, "
                      "Anderson-Darling); the verdict is 'not normal' when at least half of them reject at "
                      f"alpha {v.alpha:g}. Every candidate distribution is fitted by maximum likelihood and ranked by AIC; "
                      "the Akaike weight is its relative support among the candidates.")
        self.add("p", [("i", "With thousands of values every test rejects even a good fit, so read KS D (the largest gap "
                              "between the fitted and the observed cumulative distribution) as the size of the misfit: "
                              "below 0.03 close, below 0.08 fair, above poor. KS p-values are optimistic since the "
                              "parameters come from the same data. For integer columns the gap is measured on the "
                              "integer support and the p-value comes from a chi-square goodness-of-fit test.")])
        rows = []
        for i, name in enumerate(v.columns):
            d = v.columns[name]
            dist = d.get("distribution") or {}
            norm, fit = dist.get("normality") or {}, dist.get("fit") or {}
            if not norm:
                continue
            fits = fit.get("fits") or []
            best = fits[0] if fits else None
            rejecting = f"{norm['n_reject']} of {norm['n_tests']}" if norm.get("n_tests") else ""
            second = fits[1] if len(fits) > 1 else None
            rows.append([
                [("link", f"col-{i}"), code(name)], f"{norm.get('n', 0):,}", norm.get("verdict", ""), rejecting,
                best["label"] if best else "-", _fmt_params(best["params"]) if best else "",
                f"{best['akaike_weight']:.2f}" if best and "akaike_weight" in best else "",
                f"{second['delta_aic']:.1f} ({second['label']})" if second and "delta_aic" in second else "",
                _ks(best["ks_stat"]) if best else "", fmt_p(best["ks_pvalue"]) if best else "",
                (fit.get("description") or "").split(", best fit")[0],
            ])
        self.add("table", ["column", "n", "normality", "tests rejecting", "best fit", "parameters", "weight",
                           "ΔAIC to 2nd", "KS D", "KS p", "shape"], rows, "data")
        self.figures(self.charts(("dist",), any_column=True), grid=True)

    def column_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["columns"], "columns")
        rows = []
        for i, (name, d) in enumerate(v.columns.items()):
            st = d.get("stats", {})
            rows.append([[("link", f"col-{i}"), code(name)], d.get("semantic_type", ""),
                         f"{st.get('pct_missing', 0):.1f}%", f"{st.get('n_unique', 0):,}",
                         str(len(d.get("alerts", [])))])
        self.add("table", ["column", "type", "missing", "distinct", "alerts"], rows, "data")
        for i, (name, d) in enumerate(v.columns.items()):
            summary = [code(name), ("", f" · {d.get('semantic_type')}")]
            n_alerts = len(d.get("alerts", []))
            if n_alerts:
                summary.append(("", f" · {n_alerts} alert{'s' if n_alerts > 1 else ''}"))
            self.add("details", summary, self.column_body(name, d), f"col-{i}")

    def column_body(self, name: str, d: dict[str, Any]) -> list[Block]:
        st = d.get("stats", {})
        body: list[Block] = [("p", [("i", f"{d.get('pandas_dtype')} · {d.get('reason')}")])]
        path = d.get("path") or []
        if path:
            body.append(("p", [bold("Decision path: "), ("", " → ".join(path[:-1]) + (" → " if len(path) > 1 else "")),
                               bold(path[-1] + f" yes: {d.get('semantic_type')}"),
                               ("", f"  ({d.get('certainty', '-')}: {d.get('certainty_reason', '')})")]))
        if d.get("role"):
            b = d.get("birth") or {}
            ages = (f" Ages {b['reference']}: median {b['median_age']:.0f}, {b['min_age']:.0f} to {b['max_age']:.0f}; "
                    f"see the derived column {name}_age." if "median_age" in b else "")
            body.append(("callout", [bold(f"Role: {d['role']} "), ("", f"({d['role_certainty']}: {d['role_reason']}).{ages}")]))
            body += [("ul", [[(f"sev-{sev}", sev.upper()), ("", " " + msg)] for sev, msg in b.get("checks", [])])]
        elif d.get("role_note"):
            body.append(("callout", [bold("Not used as a date of birth: "), ("", d["role_note"] + ".")]))
        if d.get("derived_from"):
            body.append(("p", [("i", f"Derived column: {d['reason']}.")]))
        alerts = d.get("alerts") or []
        if alerts:
            body.append(("ul", [[(f"sev-{a['severity']}", a["severity"].upper()), ("", " " + a["message"])]
                                for a in alerts]))
        cards = _cards(d)
        if cards:
            body.append(("kvgrid", cards))
        dist = d.get("distribution") or {}
        norm, fit = dist.get("normality") or {}, dist.get("fit") or {}
        if norm.get("shapiro_p") is not None:
            tests = [("Shapiro-Wilk", "shapiro"), ("D'Agostino K²", "dagostino"), ("Jarque-Bera", "jarque_bera"),
                     ("Anderson-Darling", "anderson")]
            trows = []
            for label, key in tests:
                stat, p = norm.get(f"{key}_stat"), norm.get(f"{key}_p")
                if stat is None:
                    continue
                verdict = "-" if p is None else ("rejects normality" if p < self.v.alpha else "consistent with normal")
                trows.append([label, _num(stat), fmt_p(p) if p is not None else "critical values only", verdict])
            body += [("h", 4, f"Normality: {norm.get('verdict')} (n = {norm.get('n', 0):,})", None),
                     ("table", ["test", "statistic", "p-value", "at alpha " + f"{self.v.alpha:g}"], trows, "data")]
        if fit.get("fits"):
            frows = [[f"{j + 1}", f["label"], _fmt_params(f["params"]), _num(f["aic"], 6),
                      f"{f.get('delta_aic', 0):.1f}", f"{f.get('akaike_weight', float('nan')):.2f}", _num(f["bic"], 6),
                      _ks(f["ks_stat"]), fmt_p(f["ks_pvalue"])] for j, f in enumerate(fit["fits"])]
            n_tried = fit.get("n_candidates_fitted", len(fit["fits"]))
            body += [("h", 4, f"Best fits ({fit.get('kind', '')}, {n_tried} candidates fitted on "
                              f"{fit.get('n_used', 0):,} values)", None),
                     ("table", ["#", "distribution", "parameters", "AIC", "ΔAIC", "weight", "BIC", "KS D", "KS p"],
                      frows, "data")]
        top = st.get("top_values") or []
        if top and (st.get("pct_unique") or 0) <= 90 and d.get("semantic_type") not in ("numeric_continuous", "text"):
            body += [("h", 4, "Most frequent values", None),
                     ("table", ["value", "count", "share"],
                      [[[code(truncate(t["value"], 60))], f"{t['count']:,}", f"{t['pct']:.1f}%"] for t in top], "data")]
        extra = _extra_lines(d)
        if extra:
            body.append(("ul", extra))
        for path, label in self.charts(("dist", "bar", "time", "text"), name):
            body.append(("img", path, f"{name}: {label}"))
        return body

    def run_section(self) -> None:
        v, o, t = self.v, self.v.overview, self.v.timings
        self.add("h", 2, SECTIONS["run"], "run")
        rows: list[list[Inline]] = [[[bold("Source")], o.get("source") or "DataFrame"]]
        if o.get("sampled"):
            rows.append([[bold("Rows profiled")], f"{o['n_rows_profiled']:,} of {o['n_rows']:,} (uniform random sample)"])
        procs = f"{o.get('workers', 1)} for the analysis"
        if o.get("chart_workers"):
            procs += f", {o['chart_workers']} for the charts"
        rows.append([[bold("Processes")], procs + f" ({os.cpu_count() or 1} cores available)"])
        if t:
            steps = ", ".join(f"{k} {t[k]:.1f}s" for k in ("load", "columns", "associations", "quality", "charts", "report")
                              if k in t)
            rows.append([[bold("Time")], f"{t.get('total', sum(t.values())):.1f}s ({steps})"])
        rows.append([[bold("Significance")], f"alpha {v.alpha:g}, Benjamini-Hochberg false discovery rate per family of tests"])
        rows.append([[bold("Versions")], f"elessar {__version__} · Python {platform.python_version()} · pandas {pd.__version__}"
                                         f" · numpy {np.__version__} · scipy {_version('scipy')}"])
        self.add("table", ["", ""], rows, "kv")

    def figures(self, charts: list[tuple[str, str]], grid: bool = False) -> None:
        if charts:
            self.add("figures", charts, grid)


# ------------------------------------------------------------------------------------------- markdown
def _md_escape(text: str, table: bool = False) -> str:
    text = re.sub(r"([\\`*_\[\]<>#])", r"\\\1", text)
    return text.replace("|", "\\|").replace("\n", " ") if table else text


def _md_code(text: str, table: bool = False) -> str:
    fence = "`" * (max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    body = text.replace("|", "\\|").replace("\n", " ") if table else text
    pad = " " if body.startswith("`") or body.endswith("`") or not body.strip() else ""
    return f"{fence}{pad}{body}{pad}{fence}"


def _md_inline(inline: Inline, table: bool = False) -> str:
    out: list[str] = []
    target = None
    for style, text in _spans(inline):
        if style == "file":
            target = _md_url(text)
            continue
        if target is not None:
            out.append(f"[{_md_inline([(style, text)], table)}]({target})")
            target = None
            continue
        if style == "code":
            out.append(_md_code(text, table))
        elif style == "b":
            out.append(f"**{_md_escape(text, table)}**")
        elif style == "i":
            out.append(f"*{_md_escape(text, table)}*")
        elif style.startswith(("sev-", "cert-")):
            out.append(f"**{_md_escape(text, table)}**")
        elif style not in ("link", "file"):
            out.append(_md_escape(text, table))
    return "".join(out)


def _md_url(path: str) -> str:
    return quote(path, safe="/._-~")


def render_markdown(blocks: list[Block]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            lines += ["#" * block[1] + " " + _md_escape(block[2]), ""]
        elif kind == "p":
            lines += [_md_inline(block[1]), ""]
        elif kind == "toc":
            lines += [" · ".join(f"[{_md_escape(label)}](#{key_slug(label)})" for _, label in block[1]), ""]
        elif kind == "ul":
            lines += [f"- {_md_inline(item)}" for item in block[1]] + [""]
        elif kind == "table":
            headers, rows = block[1], block[2]
            lines.append("| " + " | ".join(_md_escape(h, True) or " " for h in headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")
            lines += ["| " + " | ".join(_md_inline(c, True) for c in row) + " |" for row in rows]
            lines.append("")
        elif kind == "callout":
            lines += ["> " + _md_inline(block[1]), ""]
        elif kind == "code":
            fence = "`" * max(3, max((len(m) for m in re.findall(r"`+", block[1])), default=0) + 1)
            lines += [fence + "python", block[1], fence, ""]
        elif kind == "kvgrid":
            for title, pairs in block[1]:
                lines += [f"**{_md_escape(title)}**", "", "| | |", "|---|---|"]
                lines += [f"| {_md_escape(k, True)} | {_md_escape(v, True)} |" for k, v in pairs]
                lines.append("")
        elif kind == "img":
            lines += [f"![{_md_escape(block[2])}]({_md_url(block[1])})", ""]
        elif kind == "figures":
            lines += [line for path, label in block[1] for line in (f"![{_md_escape(label)}]({_md_url(path)})", "")]
        elif kind == "details":
            lines += ["<details>", f"<summary>{_html_inline(block[1])}</summary>", ""]
            lines += render_markdown(block[2]).splitlines()
            lines += ["</details>", ""]
    return "\n".join(lines).rstrip() + "\n"


def key_slug(title: str) -> str:
    """GitHub's heading anchor for the fixed section titles."""
    return re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")


# ------------------------------------------------------------------------------------------- html
def _html_inline(inline: Inline) -> str:
    out, link = [], None
    for style, text in _spans(inline):
        if style in ("link", "file"):
            link = ("#" if style == "link" else "") + (text if style == "link" else quote(text, safe="/._-~"))
            continue
        t = html.escape(text)
        piece = {"code": f"<code>{t}</code>", "b": f"<strong>{t}</strong>", "i": f"<em>{t}</em>"}.get(style, t)
        if style.startswith(("sev-", "cert-")):
            piece = f'<span class="chip {html.escape(style)}">{t}</span>'
        if link:
            piece, link = f'<a href="{html.escape(link)}">{piece}</a>', None
        out.append(piece)
    return "".join(out)


def render_html(blocks: list[Block], title: str, embed_from: Path | None = None) -> str:
    """The recap page; with ``embed_from`` (the folder chart paths are relative to) the charts are
    inlined, so the page is one file that can be sent or attached on its own."""
    _EMBED["base"] = embed_from
    try:
        return _PAGE.format(title=html.escape(title), body=_html_body(blocks))
    finally:
        _EMBED["base"] = None


def _html_body(blocks: list[Block]) -> str:
    body: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            anchor = f' id="{block[3]}"' if block[3] else ""
            body.append(f"<h{block[1]}{anchor}>{html.escape(block[2])}</h{block[1]}>")
        elif kind == "p":
            body.append(f"<p>{_html_inline(block[1])}</p>")
        elif kind == "toc":
            body.append('<nav class="toc">' + "".join(f'<a href="#{key}">{html.escape(label)}</a>'
                                                     for key, label in block[1]) + "</nav>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{_html_inline(i)}</li>" for i in block[1]) + "</ul>")
        elif kind == "table":
            headers, rows, cls = block[1], block[2], block[3]
            head = "" if cls == "kv" else "<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead>"
            trs = []
            for row in rows:
                attrs = f' class="sev-{row[0]}"' if cls == "alerts" else ""
                trs.append(f"<tr{attrs}>" + "".join(f"<td>{_html_inline(c)}</td>" for c in row) + "</tr>")
            body.append(f'<div class="table-wrap"><table class="{cls}">{head}<tbody>{"".join(trs)}</tbody></table></div>')
        elif kind == "callout":
            body.append(f'<div class="callout">{_html_inline(block[1])}</div>')
        elif kind == "code":
            body.append(f"<pre><code>{html.escape(block[1])}</code></pre>")
        elif kind == "kvgrid":
            cards = []
            for title, pairs in block[1]:
                trs = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in pairs)
                cards.append(f'<div class="card"><div class="card-title">{html.escape(title)}</div>'
                             f"<table>{trs}</table></div>")
            body.append(f'<div class="kvgrid">{"".join(cards)}</div>')
        elif kind == "img":
            body.append(_figure(block[1], block[2]))
        elif kind == "figures":
            cls = "figures grid" if block[2] and len(block[1]) > 1 else "figures"
            body.append(f'<div class="{cls}">' + "".join(_figure(p, label) for p, label in block[1]) + "</div>")
        elif kind == "details":
            body.append(f'<details id="{block[3]}"><summary>{_html_inline(block[1])}</summary>'
                        f"{_html_body(block[2])}</details>")
    return "\n".join(body)


_EMBED: dict[str, Path | None] = {"base": None}


def _figure(path: str, label: str) -> str:
    base = _EMBED["base"]
    if base is not None:  # one self-contained file: the PNG inlined as a data URI
        data = base64.b64encode((base / path).read_bytes()).decode("ascii")
        return (f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(label)}" '
                'loading="lazy"></figure>')
    src = html.escape(quote(path, safe="/._-~"))
    return f'<figure><a href="{src}"><img src="{src}" alt="{html.escape(label)}" loading="lazy"></a></figure>'


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#f4f3ef; --surface:#fcfcfb; --ink:#1b1b19; --ink2:#52514e; --muted:#8f8e88; --line:#e6e4de;
  --blue:#2a78d6; --high:#c9302c; --medium:#b7791f; --info:#2a78d6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 Inter, Lato, "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 20px 80px; }}
h1 {{ font-size:30px; margin:8px 0 2px; letter-spacing:-0.01em; }}
h2 {{ font-size:22px; margin:44px 0 12px; padding-top:8px; border-top:1px solid var(--line); }}
h3 {{ font-size:16px; margin:26px 0 8px; }}
p {{ color:var(--ink2); }}
code {{ font:13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#ebe9e3; padding:1px 5px; border-radius:4px; }}
a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
nav.toc {{ position:sticky; top:0; z-index:5; display:flex; gap:18px; flex-wrap:wrap; padding:10px 14px; margin:14px 0 18px;
  background:var(--surface); border:1px solid var(--line); border-radius:10px; font-weight:600; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
.table-wrap {{ overflow-x:auto; margin:10px 0 18px; border:1px solid var(--line); border-radius:10px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th, td {{ text-align:left; padding:7px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); background:#f7f6f2; }}
tr:last-child td {{ border-bottom:none; }}
table.kv td:first-child {{ width:190px; color:var(--ink2); }}
tr.sev-high td:first-child {{ color:var(--high); font-weight:700; }}
tr.sev-medium td:first-child {{ color:var(--medium); font-weight:700; }}
tr.sev-info td:first-child {{ color:var(--info); }}
.chip {{ display:inline-block; font-size:11px; font-weight:700; letter-spacing:.04em; padding:1px 7px; border-radius:99px; }}
.chip.sev-high {{ background:#fde2e1; color:var(--high); }} .chip.sev-medium {{ background:#fdf0d5; color:var(--medium); }}
.chip.sev-info {{ background:#dbe9fb; color:var(--info); }}
.chip.cert-certain {{ background:#d3f1e5; color:#0f7a52; }} .chip.cert-probable {{ background:#dbe9fb; color:var(--info); }}
.chip.cert-uncertain {{ background:#fdf0d5; color:var(--medium); }}
figure {{ margin:14px 0; }}
.figures:not(.grid) figure img, details figure img {{ width:auto; max-width:100%; max-height:760px; margin:0 auto; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:10px; background:#fcfcfb; }}
.figures.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(480px, 1fr)); gap:14px; }}
.figures.grid figure {{ margin:0; }}
details {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; margin:8px 0; padding:0 16px; }}
details[open] {{ padding-bottom:12px; }}
summary {{ cursor:pointer; padding:11px 0; font-weight:600; }}
details > ul {{ font-size:13.5px; color:var(--ink2); }}
pre {{ background:#1f2430; color:#e6e4de; border-radius:10px; padding:12px 16px; overflow-x:auto; margin:10px 0 16px; }}
pre code {{ background:none; color:inherit; padding:0; font-size:12.5px; line-height:1.5; }}
h4 {{ font-size:14px; margin:20px 0 6px; color:var(--ink2); }}
.callout {{ background:#fdf6e3; border:1px solid #f1dfb0; border-radius:10px; padding:10px 14px; margin:10px 0 16px; color:var(--ink2); }}
.kvgrid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:10px; margin:10px 0 14px; }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:8px 4px 4px; }}
.card-title {{ font-size:11.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); padding:0 10px 4px; }}
.card table {{ font-size:13px; }} .card td {{ padding:4px 10px; border-bottom:1px solid #f0efea; }}
.card td:last-child {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); }}
.card td:first-child {{ color:var(--ink2); }}
table.data td {{ font-variant-numeric:tabular-nums; }}
table.plan td:first-child {{ white-space:nowrap; font-weight:600; }}
table.plan td:last-child code, table.data td:last-child code {{ white-space:pre-wrap; display:inline-block; }}
table.plan td:nth-child(3) {{ min-width:260px; }}
@media (max-width: 640px) {{ main {{ padding:16px; }} .figures.grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
{body}
</main>
<script>
function openTarget() {{ var el = document.getElementById(location.hash.slice(1)); if (el && el.tagName === "DETAILS") el.open = true; }}
window.addEventListener("hashchange", openTarget); openTarget();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------------------- entry points
def build(vision: Vision, base: Path, title: str | None = None) -> list[Block]:
    return _Recap(vision, base).build(title or vision.name or "elessar report")


def write(vision: Vision, out_dir: str | Path, title: str | None = None, single_file: bool = False) -> dict[str, Path]:
    """Write ``report.html`` and ``report.md`` in ``out_dir``; returns their paths.

    ``single_file`` also writes ``report_single.html``, with every chart inlined.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = build(vision, out, title)
    name = title or vision.name or "elessar report"
    points = (vision.outliers or {}).get("points")
    if points is not None and len(points):
        points.to_csv(out / "outliers.csv", index=False)
    md, page = out / "report.md", out / "report.html"
    md.write_text(render_markdown(blocks), encoding="utf-8")
    page.write_text(render_html(blocks, name), encoding="utf-8")
    paths = {"md": md, "html": page}
    if single_file:
        paths["single"] = out / "report_single.html"
        paths["single"].write_text(render_html(blocks, name, embed_from=out.resolve()), encoding="utf-8")
    return paths


def maybe_open(vision: Vision, open_report: bool | str | None, stream: TextIO | None = None) -> Path | None:
    """Open the recap per ``open_report`` (see :func:`elessar.see`) and say where it is on ``stream``."""
    if not vision.recap or open_report is False or (open_report is None and not should_open()):
        return None
    path = vision.recap["md" if open_report == "md" else "html"].resolve()
    opened = open_file(path)
    if stream is not None:
        print(f"\nRecap: {path}" + ("" if opened else "  (open it in a browser)"), file=stream)
    return path


def should_open() -> bool:
    """Open automatically only for a person at a terminal: never in CI, tests or pipes."""
    return sys.stdout.isatty() and not os.environ.get("CI")


def open_file(path: Path) -> bool:
    """Open ``path`` with the system default application; False when that is not possible."""
    path = Path(path).resolve()
    try:
        if sys.platform == "darwin":
            cmd = ["open", str(path)]
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "microsoft" in platform.release().lower():
            cmd = ["xdg-open", str(path)]
        else:
            return False
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return True
    except (OSError, AttributeError):
        return False
