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
from .report import _overview_md, _pair_rows, _target_rows

if TYPE_CHECKING:
    from .report import Vision

Span = tuple[str, str]
Inline = Union[str, list[Span]]  # a plain string is one plain span
Block = tuple[Any, ...]
SECTIONS = {"findings": "Key findings", "target": "Target analysis", "distributions": "Distributions",
            "quality": "Data quality", "associations": "Associations", "columns": "Columns", "run": "Run"}
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
    return [("", inline)] if isinstance(inline, str) else inline


def _num(x: Any, digits: int = 4) -> str:
    return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x) else fmt_num(x, digits)


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
                                 "min / max / mean / std, distinct counts and duplicate rows are exact over every row ("),
                                 ("link", "exact"), ("", "exact figures"), ("", ").")])
        toc = (["findings"] + (["target"] if v.target else []) + (["distributions"] if self.fitted() else [])
               + ["quality", "associations", "columns", "run"])
        self.add("toc", [(key, SECTIONS[key]) for key in toc])
        self.add("table", ["", ""], [[[bold(k.capitalize())], v_] for k, v_ in _overview_md(o).items()], "kv")
        if v.full is not None:
            self.exact_section()
        self.findings()
        if v.target:
            self.target_section()
        if self.fitted():
            self.distribution_section()
        self.quality_section()
        self.association_section()
        self.column_section()
        self.run_section()
        return self.blocks

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
        self.figures(self.charts(("overview",)))

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
        self.add("h", 2, SECTIONS["associations"], "associations")
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
    out = []
    for style, text in _spans(inline):
        if style == "code":
            out.append(_md_code(text, table))
        elif style == "b":
            out.append(f"**{_md_escape(text, table)}**")
        elif style == "i":
            out.append(f"*{_md_escape(text, table)}*")
        elif style.startswith("sev-"):
            out.append(f"**{_md_escape(text, table)}**")
        elif style != "link":
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
        if style == "link":
            link = text
            continue
        t = html.escape(text)
        piece = {"code": f"<code>{t}</code>", "b": f"<strong>{t}</strong>", "i": f"<em>{t}</em>"}.get(style, t)
        if style.startswith("sev-"):
            piece = f'<span class="chip {html.escape(style)}">{t}</span>'
        if link:
            piece, link = f'<a href="#{html.escape(link)}">{piece}</a>', None
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
figure {{ margin:14px 0; }}
.figures:not(.grid) figure img, details figure img {{ width:auto; max-width:100%; max-height:760px; margin:0 auto; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:10px; background:#fcfcfb; }}
.figures.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(480px, 1fr)); gap:14px; }}
.figures.grid figure {{ margin:0; }}
details {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; margin:8px 0; padding:0 16px; }}
details[open] {{ padding-bottom:12px; }}
summary {{ cursor:pointer; padding:11px 0; font-weight:600; }}
details > ul {{ font-size:13.5px; color:var(--ink2); }}
h4 {{ font-size:14px; margin:20px 0 6px; color:var(--ink2); }}
.callout {{ background:#fdf6e3; border:1px solid #f1dfb0; border-radius:10px; padding:10px 14px; margin:10px 0 16px; color:var(--ink2); }}
.kvgrid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(250px, 1fr)); gap:10px; margin:10px 0 14px; }}
.card {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:8px 4px 4px; }}
.card-title {{ font-size:11.5px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); padding:0 10px 4px; }}
.card table {{ font-size:13px; }} .card td {{ padding:4px 10px; border-bottom:1px solid #f0efea; }}
.card td:last-child {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--ink); }}
.card td:first-child {{ color:var(--ink2); }}
table.data td {{ font-variant-numeric:tabular-nums; }}
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
