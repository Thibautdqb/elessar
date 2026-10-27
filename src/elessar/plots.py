"""Informative plots only, rendered off-screen on the Agg canvas and saved as PNG files.

Figures are built with the object-oriented :class:`matplotlib.figure.Figure` API and an explicit
``FigureCanvasAgg``: no display is ever needed and pyplot's global state is never touched.

Every chart shares one layout: a header (bold title + muted subtitle carrying the key numbers),
the plot body, and an optional footer note explaining the method.
"""

from __future__ import annotations

import math
import re
import textwrap
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.figure import Figure, SubFigure
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
from scipy import stats

from ._utils import fmt_bytes, fmt_num, fmt_p, hashable, prepare, quiet, resolve_target, safe_name, sample, sample_array, truncate
from .config import StoneConfig
from .correlations import bh_adjust
from .distributions import _fmt_params, frozen_best
from .quality import duplicate_rows
from .stats import split_labels, time_resolution
from .types import CATEGORY_TYPES, DATETIME_TYPES, MULTI_LABEL, NUMERIC_TYPES, as_datetime, as_numeric, ordinal_order

# ---------------------------------------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
PANEL = "#f4f3ef"
INK = "#1b1b19"
INK_2 = "#52514e"
MUTED = "#8f8e88"
LIGHT = "#c9c7bf"
GRID = "#e6e4de"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
BLUE, ORANGE = SERIES[0], SERIES[1]
BLUE_FILL = "#9ec5f4"
BLUE_PALE = "#dbe9fb"
HIGHLIGHT = "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("elessar_div", ["#104281", "#3987e5", "#cde2fb", "#f3f2ee", "#fbd3bf",
                                                             "#eb6834", "#a23d12"])
SEQUENTIAL = LinearSegmentedColormap.from_list("elessar_seq", ["#f3f2ee", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])
DENSITY = LinearSegmentedColormap.from_list("elessar_density", ["#dbe9fb", "#86b6ef", "#2a78d6", "#104281"])
MISSING_CMAP = ListedColormap([PANEL, "#2a78d6"])
TYPE_COLORS = {"numeric": "#dbe9fb", "category": "#fde3d6", "datetime": "#d3f1e5", "text": "#e6e1fa", "other": "#ebe9e3"}

FONTS = ["Inter", "Lato", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": FONTS,
    "text.parse_math": False,  # column names and values may contain "$"
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": LIGHT,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9.5,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "font.size": 9.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.constrained_layout.h_pad": 0.06,
    "figure.constrained_layout.w_pad": 0.08,
}
_PLOT_FILE = re.compile(r"^\d{2,}_.+\.png$")


# ---------------------------------------------------------------------------------------------- infrastructure
class _Writer:
    """Numbers and saves figures as ``out_dir/<NN>_<column>_<kind>.png``."""

    def __init__(self, out_dir: str | Path, dpi: int) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for old in self.out_dir.iterdir():
            if old.is_file() and _PLOT_FILE.match(old.name):
                old.unlink()
        self.dpi = dpi
        self.paths: list[Path] = []
        self.index: list[dict[str, Any]] = []  # {"column", "other", "kind", "path"} per chart written
        self.errors: list[str] = []

    def safe(self, what: str, step: Callable[..., Any], *args: Any) -> None:
        """Run one plotting step; a failure is recorded, never raised."""
        try:
            step(self, *args)
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{what}: {exc!r}")

    def save(self, fig: Figure, column: str | None, kind: str, other: str | None = None) -> None:
        """Write the figure; ``column`` is None for dataset-level charts, ``other`` is the second column of a pair."""
        stem = "dataset" if column is None else safe_name(column) + (f"__{safe_name(other)}" if other is not None else "")
        path = self.out_dir / f"{len(self.paths):02d}_{stem}_{kind}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.22)
        fig.clear()
        self.paths.append(path)
        self.index.append({"column": column, "other": other, "kind": kind, "path": path})


def _figure(w: float, h: float, title: str, subtitle: str = "", note: str = "") -> tuple[Figure, SubFigure]:
    """Figure with a header (title + subtitle), a body sub-figure to draw in, and an optional footer note."""
    fig = Figure(figsize=(w, h), layout="constrained")
    FigureCanvasAgg(fig)
    note_lines = textwrap.wrap(note, int(w * 16)) if note else []
    head_h, foot_h = 0.5 if subtitle else 0.36, (0.14 + 0.16 * len(note_lines)) if note else 0.0
    ratios = [head_h, max(h - head_h - foot_h, 0.5)] + ([foot_h] if note else [])
    subs = fig.subfigures(len(ratios), 1, height_ratios=ratios)
    for sub in subs:
        sub.set_facecolor(SURFACE)
    chars = int(w * 12.5)
    subs[0].text(0.0, 1.0 if subtitle else 0.5, truncate(title, int(w * 8)), fontsize=14, fontweight="bold", color=INK,
                 va="top" if subtitle else "center", ha="left")
    if subtitle:
        subs[0].text(0.0, 0.02, truncate(subtitle, chars), fontsize=9.5, color=INK_2, va="bottom", ha="left")
    if note:
        subs[2].text(0.0, 0.5, "\n".join(note_lines), fontsize=8, color=MUTED, va="center", ha="left", linespacing=1.4)
    return fig, subs[1]


def _compact(x: float, _pos: Any = None) -> str:
    """Axis label: 12.5k, 3.4M, 0.05 (years such as 2020 stay as they are)."""
    if not np.isfinite(x):
        return ""
    a = abs(x)
    if a == 0:
        return "0"
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e4, "k")):
        if a >= div:
            return f"{x / (div if suffix != 'k' else 1e3):.3g}{suffix}"
    if a < 1e-3:
        return f"{x:.0e}"
    return f"{x:.4g}"


def _format_axis(ax: Any, which: str = "xy") -> None:
    """Compact tick labels on every scale (log axes included: matplotlib's defaults there need mathtext)."""
    for axis in which:
        target = getattr(ax, f"{axis}axis")
        target.set_major_formatter(FuncFormatter(_compact))
        if target.get_scale() != "linear":
            target.set_minor_formatter(NullFormatter())


def _value_axis_only(ax: Any, axis: str) -> None:
    """Keep the grid on the value axis only and drop the spine that duplicates the category labels."""
    ax.grid(False, axis="y" if axis == "x" else "x")
    ax.spines["left" if axis == "x" else "bottom"].set_visible(False)


def _text_color(fill: str) -> str:
    r, g, b = to_rgb(fill)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


def _type_family(stype: str) -> str:
    if stype in NUMERIC_TYPES or stype == "numeric_string":
        return "numeric"
    if stype in CATEGORY_TYPES or stype == MULTI_LABEL:
        return "category"
    if stype in DATETIME_TYPES:
        return "datetime"
    if stype in ("text", "email", "url", "phone", "postal_code", "ip_address"):
        return "text"
    return "other"


def _boxplot(ax: Any, data: list[np.ndarray], labels: list[str] | None = None, widths: float = 0.5) -> None:
    kw: dict[str, Any] = dict(
        widths=widths, patch_artist=True, showfliers=True, showmeans=True,
        boxprops=dict(facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=1.1),
        medianprops=dict(color=INK, linewidth=1.8),
        whiskerprops=dict(color=BLUE, linewidth=1.0), capprops=dict(color=BLUE, linewidth=1.0),
        meanprops=dict(marker="D", markersize=4.5, markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.9),
        flierprops=dict(marker="o", markersize=2.6, markerfacecolor=HIGHLIGHT, markeredgecolor="none", alpha=0.45),
    )
    if labels is not None:
        kw["tick_labels" if mpl.__version_info__ >= (3, 9) else "labels"] = labels
    try:
        ax.boxplot(data, orientation="horizontal", **kw)
    except TypeError:  # matplotlib < 3.10
        ax.boxplot(data, vert=False, **kw)


def _log_x(v: np.ndarray, skew: float | None) -> bool:
    """Heavily right-skewed, non-negative data spanning > 1.3 orders of magnitude: plot on a log axis."""
    pos = v[v > 0]
    if len(pos) < 10 or v.min() < 0 or skew is None or not np.isfinite(skew) or skew < 2:
        return False
    lo, hi = np.percentile(pos, [1, 99])
    return hi / lo > 20


def _set_log(ax: Any, v: np.ndarray, axes: str = "x") -> None:
    """Log scale, or symlog (linear around 0) when the data contains zeros."""
    for axis in axes:
        setter = getattr(ax, f"set_{axis}scale")
        if v.min() > 0:
            setter("log")
        else:
            setter("symlog", linthresh=float(v[v > 0].min()))
            if axis == "x":
                ax.set_xlim(left=0)
            else:
                ax.set_ylim(bottom=0)
    _format_axis(ax, axes)


def _bins(v: np.ndarray, discrete: bool, logx: bool) -> np.ndarray:
    if discrete and v.max() - v.min() <= 80:
        return np.arange(v.min() - 0.5, v.max() + 1.5)
    if logx:
        edges = np.geomspace(v[v > 0].min(), v.max(), 45)
        return np.r_[0.0, edges] if v.min() == 0 else edges
    edges = np.histogram_bin_edges(v, bins="auto")
    return np.linspace(v.min(), v.max(), 61) if len(edges) > 61 else edges


# ---------------------------------------------------------------------------------------------- numeric
def _qq(ax: Any, v: np.ndarray, ref: Any, ref_name: str, logx: bool, cfg: StoneConfig) -> None:
    """Q-Q plot with a 95 % pointwise envelope (order statistics ~ Beta) under the reference distribution."""
    q = np.sort(sample_array(v, 1_500, cfg.random_state))
    n = len(q)
    i = np.arange(1, n + 1)
    with quiet():
        theo = ref.ppf((i - 0.5) / n)
        lo_band = ref.ppf(stats.beta.ppf(0.025, i, n - i + 1))
        hi_band = ref.ppf(stats.beta.ppf(0.975, i, n - i + 1))
    ok = np.isfinite(theo) & np.isfinite(lo_band) & np.isfinite(hi_band)
    theo, q, lo_band, hi_band = theo[ok], q[ok], lo_band[ok], hi_band[ok]
    if not len(q):
        return
    ax.fill_between(theo, lo_band, hi_band, color=BLUE_PALE, linewidth=0, label="95% envelope")
    outside = (q < lo_band) | (q > hi_band)
    ax.scatter(theo[~outside], q[~outside], s=7, color=BLUE, alpha=0.75, edgecolors="none", label="data")
    if outside.any():
        ax.scatter(theo[outside], q[outside], s=9, color=HIGHLIGHT, alpha=0.85, edgecolors="none", label="outside envelope")
    lo, hi = min(q.min(), theo.min()), max(q.max(), theo.max())
    lo, hi = max(lo, q.min() - (q.max() - q.min())), min(hi, q.max() + (q.max() - q.min()))
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=1, ls=(0, (4, 3)))
    pad = (hi - lo) * 0.04
    if logx:
        positive = q[q > 0]
        _set_log(ax, positive, "xy")
        ax.set_xlim(max(lo, positive.min()), hi)
        ax.set_ylim(max(lo, positive.min()), hi)
    else:
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        _format_axis(ax)
    ax.set_title(f"Q-Q plot vs {ref_name}")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("observed quantiles")
    ax.text(0.98, 0.03, f"{100 * outside.mean():.1f}% of points outside the envelope", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=HIGHLIGHT if outside.mean() > 0.05 else INK_2)


def plot_numeric(writer: _Writer, col: str, x: pd.Series, detail: dict[str, Any], cfg: StoneConfig) -> None:
    """Histogram (row counts) with the best fit as expected counts, boxplot, and Q-Q plot with envelope."""
    v = x.to_numpy(dtype="float64")
    v = sample_array(v[np.isfinite(v)], cfg.sample_size, cfg.random_state)
    if len(v) < 3 or np.unique(v).size < 2:
        return
    st = detail.get("stats", {})
    fit = (detail.get("distribution") or {}).get("fit") or {}
    norm = (detail.get("distribution") or {}).get("normality") or {}
    best = frozen_best(fit)
    best_info = fit["fits"][0] if fit.get("fits") else None
    discrete = fit.get("kind") == "discrete"
    logx = _log_x(v, st.get("skewness"))

    units = list((detail.get("units") or {}).keys())
    n_values = int(np.isfinite(x.to_numpy(dtype="float64")).sum())
    parts = [detail.get("semantic_type", "")]
    if len(units) > 1:
        parts.append(f"{units[0]} only: {n_values:,} of {st.get('count', n_values):,} values (mixed units)")
    else:
        parts.append(f"n = {st.get('count', len(v)):,}")
    if st.get("pct_missing"):
        parts.append(f"{st['pct_missing']:.1f}% missing")
    if st.get("mean") is not None:
        parts.append(f"mean {fmt_num(st['mean'])} · median {fmt_num(st['median'])} · sd {fmt_num(st['std'])}")
    if st.get("skewness") is not None and np.isfinite(st["skewness"]):
        parts.append(f"skew {st['skewness']:.2f}")
    if norm.get("verdict") in ("normal", "not normal"):
        parts.append(norm["verdict"])
    shape = (fit.get("description") or "").split(", best fit")[0]
    n_zero = int(np.sum(v <= 0)) if logx else 0
    note = (f"{n_zero:,} zero values ({100 * n_zero / len(v):.1f}%) are not shown on the log axis. " if n_zero else "")
    note += "Histogram in rows; the curve is the count expected under the best fit (MLE, lowest AIC). "
    note += "Q-Q envelope: 95% pointwise band for a sample of this size."
    fig, body = _figure(12, 5.6, col, "  ·  ".join(p for p in parts if p), note)
    gs = body.add_gridspec(2, 2, height_ratios=[4.2, 1], width_ratios=[1.75, 1])
    ax_h = body.add_subplot(gs[0, 0])
    ax_b = body.add_subplot(gs[1, 0], sharex=ax_h)
    ax_q = body.add_subplot(gs[:, 1])

    full = v
    if n_zero:  # a log axis cannot show zeros: plot the positive values and say so in the note
        v = v[v > 0]
    bins = _bins(v, discrete, logx)
    counts, edges, _ = ax_h.hist(v, bins=bins, color=BLUE_FILL, edgecolor=SURFACE, linewidth=0.7, label="observed")
    n = len(v)
    if best is not None and best_info is not None:
        label = f"{best_info['label']} fit ({_fmt_params(best_info['params'])})"
        with quiet():
            if discrete:
                k = np.arange(math.floor(v.min()), math.ceil(v.max()) + 1)
                ax_h.plot(k, best.pmf(k) * n, "o-", color=ORANGE, lw=1.8, ms=3.5, label=label)
            elif logx:
                centers = np.sqrt(np.clip(edges[:-1], edges[1] / 10, None) * edges[1:])
                ax_h.plot(centers, n * np.diff(best.cdf(edges)), color=ORANGE, lw=2, label=label)
            else:
                grid = np.linspace(v.min(), v.max(), 400)
                ax_h.plot(grid, best.pdf(grid) * n * np.diff(edges).mean(), color=ORANGE, lw=2, label=label)
    med = float(np.median(full))
    ax_h.axvline(med, color=INK, lw=1, ls=(0, (3, 2)))
    ax_h.annotate(f"median {fmt_num(med)}", xy=(med, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                  textcoords="offset points", fontsize=8, color=INK, va="top")
    ax_h.set_ylabel("rows" + (" (sample)" if len(full) < n_values else ""))
    ax_h.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2, handlelength=1.4, borderaxespad=0.2)

    ax_h.tick_params(labelbottom=False)
    ax_h.set_title(shape or "distribution")
    ax_h.grid(False, axis="x")
    ax_h.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    _format_axis(ax_h, "y")
    if logx:
        _set_log(ax_h, v)

    _boxplot(ax_b, [v], widths=0.55)
    ax_b.set_yticks([])
    ax_b.grid(False, axis="y")
    ax_b.spines["left"].set_visible(False)
    n_out = st.get("n_outliers_iqr")
    ax_b.set_xlabel(col + (f" ({units[0]})" if units else "") + (" · log scale" if logx else "") + (f"   ·   {n_out:,} outliers (1.5×IQR)" if n_out else ""))
    if not logx:
        _format_axis(ax_b, "x")

    ref = best if best is not None else stats.norm(np.mean(v), np.std(v) or 1)
    _qq(ax_q, full, ref, best_info["label"] if best is not None else "normal", logx, cfg)
    writer.save(fig, col, "dist")


# ---------------------------------------------------------------------------------------------- categorical
def _hbar(ax: Any, labels: list[str], counts: np.ndarray, total: int, colors: list[str]) -> None:
    """Bars with direct labels (share and count): the value axis is redundant and hidden."""
    y = np.arange(len(labels))
    ax.barh(y, counts, color=colors, height=0.7, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, [truncate(lbl, 34) for lbl in labels])
    ax.invert_yaxis()
    peak = counts.max()
    for yi, c in zip(y, counts):
        ax.text(c + peak * 0.012, yi, f"{100 * c / total:.1f}%", va="center", fontsize=8.5, color=INK, fontweight="bold")
        ax.text(c + peak * 0.012 + peak * 0.085, yi, f"{c:,}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, peak * 1.25)
    ax.set_xticks([])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def plot_multilabel(writer: _Writer, col: str, s: pd.Series, detail: dict[str, Any], cfg: StoneConfig) -> None:
    """Most frequent labels of a list column, as the share of rows carrying each label."""
    st = detail.get("stats", {})
    sep = st.get("separator") or detail.get("sep") or ", "
    labels = split_labels(s, sep)
    vc = labels.explode().dropna().value_counts()
    if len(vc) < 2:
        return
    top = vc.head(cfg.plot_top_n)
    names = [str(v) for v in top.index]
    counts = top.to_numpy(dtype="int64")
    colors = [BLUE] * len(names)
    rest = len(vc) - len(top)
    sub = (f"multi-label  ·  {len(vc):,} distinct labels  ·  {st.get('labels_per_row_mean', 0):.1f} labels per row  ·  "
           f"n = {len(labels):,}")
    note = "Share of rows carrying each label; a row can carry several, so shares add up to more than 100%."
    if rest:
        note += f" {rest:,} less frequent labels not shown."
    fig, body = _figure(9.5, 0.34 * len(names) + 1.6, col, sub, note)
    ax = body.add_subplot()
    _hbar(ax, names, counts, len(labels), colors)
    writer.save(fig, col, "bar")


def plot_categorical(writer: _Writer, col: str, s: pd.Series, detail: dict[str, Any], cfg: StoneConfig) -> None:
    """Top-N categories (by frequency, or in scale order for ordinals), the rest grouped, missing shown apart."""
    vc = hashable(s.dropna()).value_counts()
    if len(vc) < 2 or vc.iloc[0] == 1:
        return
    stype = detail.get("semantic_type")
    order = detail.get("order") or (ordinal_order(vc.index) if stype == "ordinal-candidate" else None)
    if order is None and stype == "ordinal-candidate":
        vc = vc.sort_index()
    elif order:
        vc = vc.reindex([o for o in order if o in vc.index])
    top = vc.head(cfg.plot_top_n)
    labels = [str(v) for v in top.index]
    counts = top.to_numpy(dtype="int64")
    colors = [BLUE] * len(labels)
    rest = vc.iloc[cfg.plot_top_n:]
    if len(rest):
        labels.append(f"{len(rest):,} other categories")
        counts = np.append(counts, rest.sum())
        colors.append(LIGHT)
    n_missing = int(s.isna().sum())
    if n_missing:
        labels.append("missing")
        counts = np.append(counts, n_missing)
        colors.append(GRID)
    st = detail.get("stats", {})
    parts = [stype, f"{len(vc):,} distinct", f"n = {len(s):,}"]
    if st.get("normalized_entropy") is not None:
        parts.append(f"normalized entropy {st['normalized_entropy']:.2f}")
    if st.get("pct_true") is not None:
        parts.append(f"{st['pct_true']:.1f}% true [{st['ci95_pct_true_low']:.1f}, {st['ci95_pct_true_high']:.1f}]")
    ordering = "scale order" if order or stype == "ordinal-candidate" else "most frequent first"
    fig, body = _figure(9.5, 0.34 * len(labels) + 1.5, col, "  ·  ".join(parts),
                        f"Shares of all rows, {ordering}" + (f"; top {cfg.plot_top_n} shown." if len(rest) else "."))
    ax = body.add_subplot()
    _hbar(ax, labels, counts, len(s), colors)
    writer.save(fig, col, "bar")


# ---------------------------------------------------------------------------------------------- datetime & text
def plot_datetime(writer: _Writer, col: str, s: pd.Series, detail: dict[str, Any], cfg: StoneConfig) -> None:
    """Rows per period at an automatic resolution, with a rolling mean and the peak annotated."""
    d = as_datetime(s).dropna()
    if d.nunique() < 2:
        return
    span = (d.max() - d.min()).total_seconds() / 86_400
    freq, unit = time_resolution(span)
    counts = pd.Series(1, index=pd.DatetimeIndex(d)).sort_index().resample(freq).sum()
    if len(counts) < 2:
        return
    st = detail.get("stats", {})
    parts = [detail.get("semantic_type", ""), f"{d.min():%Y-%m-%d} → {d.max():%Y-%m-%d}", f"n = {len(d):,}",
             f"rows per {unit}"]
    if st.get("counts_trend_tau") is not None:
        parts.append(f"trend τ = {st['counts_trend_tau']:+.2f} (p {fmt_p(st['counts_trend_p'])})")
    fig, body = _figure(12, 4.4, col, "  ·  ".join(parts))
    ax = body.add_subplot()
    x, y = counts.index, counts.to_numpy(dtype="float64")
    ax.fill_between(x, y, step="mid", color=BLUE_PALE, linewidth=0)
    ax.plot(x, y, drawstyle="steps-mid", color=BLUE, lw=1.2, label=f"rows per {unit}")
    window = max(3, len(counts) // 15)
    if len(counts) >= 20:
        ax.plot(x, counts.rolling(window, center=True, min_periods=1).mean(), color=ORANGE, lw=2,
                label=f"rolling mean ({window} {unit}s)")
        ax.legend(loc="upper left", ncols=2)
    peak = int(np.argmax(y))
    ax.annotate(f"peak {int(y[peak]):,}", xy=(x[peak], y[peak]), xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, y.max() * 1.15)
    ax.set_ylabel(f"rows per {unit}")
    ax.grid(False, axis="x")
    _format_axis(ax, "y")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    writer.save(fig, col, "time")


def plot_text(writer: _Writer, col: str, s: pd.Series, detail: dict[str, Any], cfg: StoneConfig) -> None:
    """Distribution of text lengths (characters and words)."""
    strs = sample(s.dropna().astype(str), cfg.sample_size, cfg.random_state)
    if strs.str.len().nunique() < 2:
        return
    st = detail.get("stats", {})
    sub = f"text  ·  n = {len(s.dropna()):,}  ·  {st.get('n_unique', 0):,} distinct"
    if st.get("top_tokens"):
        sub += "  ·  top words: " + ", ".join(t["token"] for t in st["top_tokens"][:6])
    fig, body = _figure(12, 4.2, col, sub)
    for i, (values, label) in enumerate(((strs.str.len(), "characters"), (strs.str.split().str.len(), "words"))):
        ax = body.add_subplot(1, 2, i + 1)
        lo, hi = int(values.min()), int(values.max())
        ax.hist(values, bins=np.arange(lo - 0.5, hi + 1.5) if hi - lo <= 80 else 50, color=BLUE_FILL,
                edgecolor=SURFACE, linewidth=0.7)
        med = float(values.median())
        ax.axvline(med, color=INK, lw=1, ls=(0, (3, 2)))
        ax.annotate(f"median {med:g}", xy=(med, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
                    textcoords="offset points", fontsize=8, va="top")
        ax.set_title(f"{label} per value")
        ax.set_ylabel("rows")
        ax.grid(False, axis="x")
        _format_axis(ax, "y")
    writer.save(fig, col, "text")


# ---------------------------------------------------------------------------------------------- dataset level
def plot_missing(writer: _Writer, df: pd.DataFrame, cfg: StoneConfig) -> None:
    """Missing-values matrix (rows x columns) with the per-column missing share on top."""
    miss = df.isna()
    pct = 100 * miss.mean()
    if not pct.any():
        return
    cols = list(df.columns) if df.shape[1] <= 60 else list(pct.sort_values(ascending=False).index[:60])
    m = sample(miss[cols], 1_000, cfg.random_state).to_numpy(dtype="int8")
    subtitle = (f"{100 * miss.to_numpy().mean():.1f}% of cells missing  ·  {int(miss.any(axis=1).sum()):,} of "
                f"{len(df):,} rows have at least one  ·  {int((pct > 0).sum())} of {df.shape[1]} columns affected")
    fig, body = _figure(max(8, 0.32 * len(cols) + 2.5), 6.6, "Missing values", subtitle,
                        "Each line of the matrix is a row (in file order" + (", sampled" if len(df) > len(m) else "")
                        + "); blue = missing.")
    gs = body.add_gridspec(2, 1, height_ratios=[1, 3.2])
    ax_bar = body.add_subplot(gs[0])
    ax_mat = body.add_subplot(gs[1], sharex=ax_bar)
    x = np.arange(len(cols))
    values = pct[cols].to_numpy()
    ax_bar.bar(x, values, color=[BLUE if p > 0 else GRID for p in values], width=0.7)
    for xi, p in zip(x, values):
        if p > 0:
            ax_bar.text(xi, p + 2, f"{p:.0f}%" if p >= 1 else "<1%", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax_bar.set_ylim(0, 118)
    ax_bar.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    ax_bar.tick_params(labelbottom=False)
    ax_bar.grid(False, axis="x")
    ax_mat.imshow(m, aspect="auto", interpolation="nearest", cmap=MISSING_CMAP, vmin=0, vmax=1,
                  extent=(-0.5, len(cols) - 0.5, len(m), 0))
    ax_mat.set_xticks(x, [truncate(c, 24) for c in cols], rotation=55, ha="right", rotation_mode="anchor")
    ax_mat.set_ylabel("rows")
    ax_mat.grid(False)
    for side in ("left", "bottom"):
        ax_mat.spines[side].set_visible(False)
    writer.save(fig, None, "missing")


def plot_heatmap(
    writer: _Writer,
    mat: pd.DataFrame,
    p_values: pd.DataFrame | None,
    name: str,
    title: str,
    cfg: StoneConfig,
    signed: bool = True,
    subtitle: str = "",
) -> None:
    """Strictly-lower-triangle heatmap (>= 3 columns). Cells not significant after BH correction are faded."""
    if mat.shape[0] < 3:
        return
    vals = mat.to_numpy(dtype="float64")[1:, :-1].copy()
    k = vals.shape[0]
    lower = np.tril(np.ones((k, k), dtype=bool))
    vals[~lower] = np.nan
    weak = np.zeros_like(lower)
    if p_values is not None:
        p = p_values.reindex(index=mat.index, columns=mat.columns).to_numpy(dtype="float64")[1:, :-1]
        adj = np.full(p.shape, np.nan)
        adj[lower] = bh_adjust(p[lower])
        weak = lower & ~(adj < cfg.alpha)
    size = min(max(5.8, 0.5 * k + 3), 24)
    note = (f"Faded cells: not significant after Benjamini-Hochberg correction (α = {cfg.alpha:g})."
            if p_values is not None and weak.any() else "")
    n_sig = int((lower & ~weak).sum())
    sub = subtitle or f"{mat.shape[0]} columns  ·  {int(lower.sum())} pairs"
    if p_values is not None:
        sub += f"  ·  {n_sig} significant"
    fig, body = _figure(size + 1.4, size + (0.9 if note else 0.6), title, sub, note)
    ax = body.add_subplot()
    gap = 1.6 if k <= 30 else 0.4
    im = ax.pcolormesh(np.ma.masked_invalid(vals), cmap=DIVERGING if signed else SEQUENTIAL, vmin=-1 if signed else 0,
                       vmax=1, edgecolors=SURFACE, linewidth=gap)
    if weak.any():
        ax.pcolormesh(np.ma.masked_where(~weak, np.ones_like(vals)), cmap=ListedColormap([SURFACE]), alpha=0.72,
                      edgecolors=SURFACE, linewidth=gap)
    ax.set_xlim(0, k)
    ax.set_ylim(k, 0)
    ax.set_aspect("equal")
    labels = [truncate(c, 24) for c in mat.columns]
    ax.set_xticks(np.arange(k) + 0.5, labels[:-1], rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(k) + 0.5, labels[1:])
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    if k <= 25:
        fontsize = 8.5 if k <= 10 else (7.5 if k <= 16 else 6.5)
        cmap = DIVERGING if signed else SEQUENTIAL
        for i in range(k):
            for j in range(i + 1):
                v = vals[i, j]
                if np.isfinite(v):
                    fill = cmap((v + 1) / 2 if signed else v)
                    color = MUTED if weak[i, j] else _text_color(mpl.colors.to_hex(fill))
                    ax.text(j + 0.5, i + 0.5, f"{v:.2f}".replace("0.", ".", 1) if abs(v) < 1 else f"{v:.0f}",
                            ha="center", va="center", fontsize=fontsize, color=color)
    cbar = body.colorbar(im, ax=ax, fraction=0.035, pad=0.02, aspect=30)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=8, length=0)
    writer.save(fig, None, name)


def plot_nullity(writer: _Writer, df: pd.DataFrame, cfg: StoneConfig) -> None:
    """Correlation of missing-value indicators: which columns go missing together."""
    miss = df.isna()
    share = miss.mean()
    cols = [c for c in df.columns if 0 < share[c] < 1]
    if len(cols) < 3:
        return
    corr = sample(miss[cols], cfg.sample_size, cfg.random_state).astype("float64").corr()
    plot_heatmap(writer, corr, None, "nullity_corr", "Missingness correlation", cfg, True,
                 f"{len(cols)} partly-missing columns  ·  1 = always missing together, −1 = never together")


def plot_overview(writer: _Writer, df: pd.DataFrame, details: dict[str, dict[str, Any]], cfg: StoneConfig) -> None:
    """One-page profile: one row per column with type, missing share, distinct count, summary and sparkline."""
    cols = list(df.columns)[:40]
    if len(cols) < 2:
        return
    n_rows = len(df)
    row_h, width = 0.36, 13.0
    pct_missing = 100 * float(df.isna().to_numpy().mean()) if df.size else 0.0
    subtitle = (f"{n_rows:,} rows × {df.shape[1]} columns  ·  {pct_missing:.1f}% of cells missing  ·  "
                f"{duplicate_rows(df)['count']:,} duplicate rows  ·  "
                f"{fmt_bytes(float(df.memory_usage(deep=True).sum()))} in memory")
    more = f"Showing the first 40 of {df.shape[1]} columns. " if df.shape[1] > 40 else ""
    fig, body = _figure(width, row_h * (len(cols) + 1) + 1.3, "Dataset overview", subtitle,
                        more + "Sparklines: histogram for numbers, top categories for labels, rows over time for dates.")
    ax = body.add_subplot()
    ax.set_xlim(0, 1)
    ax.set_ylim(len(cols) + 0.2, -0.9)
    ax.axis("off")
    x_name, x_type, x_miss, x_dist, x_summary, x_spark = 0.0, 0.25, 0.39, 0.55, 0.63, 0.80
    for x0, header in ((x_name, "column"), (x_type, "type"), (x_miss, "missing"), (x_dist, "distinct"),
                       (x_summary, "summary"), (x_spark, "distribution")):
        ax.text(x0, -0.55, header.upper(), fontsize=7.5, color=MUTED, fontweight="bold", va="center")
    ax.plot([0, 1], [-0.25, -0.25], color=LIGHT, lw=0.8)
    for i, col in enumerate(cols):
        d = details.get(col, {})
        st = d.get("stats", {})
        stype = d.get("semantic_type", "?")
        if i % 2:
            ax.add_patch(mpl.patches.Rectangle((-0.005, i - 0.5), 1.01, 1, color=PANEL, lw=0, zorder=0))
        ax.text(x_name, i, truncate(col, 30), fontsize=9, color=INK, fontweight="bold", va="center")
        ax.text(x_type, i, stype, fontsize=8, color=INK, va="center",
                bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.6", fc=TYPE_COLORS[_type_family(stype)], ec="none"))
        miss = float(st.get("pct_missing") or 0.0)
        ax.add_patch(mpl.patches.Rectangle((x_miss, i - 0.12), 0.09, 0.24, color=GRID, lw=0))
        if miss:
            ax.add_patch(mpl.patches.Rectangle((x_miss, i - 0.12), 0.09 * miss / 100, 0.24,
                                               color=HIGHLIGHT if miss >= cfg.high_missing_pct else BLUE, lw=0))
        ax.text(x_miss + 0.097, i, f"{miss:.0f}%", fontsize=8, color=INK_2, va="center")
        ax.text(x_dist, i, f"{st.get('n_unique', 0):,}", fontsize=8.5, color=INK_2, va="center")
        ax.text(x_summary, i, truncate(_summary(st, stype), 26), fontsize=8, color=INK_2, va="center")
        _sparkline(ax, df[col], stype, st, (x_spark, i - 0.32, 0.2, 0.64), cfg)
    writer.save(fig, None, "overview")


def _summary(st: dict[str, Any], stype: str) -> str:
    if st.get("mean") is not None and stype in NUMERIC_TYPES | {"numeric_string"}:
        return f"{fmt_num(st['mean'])} ± {fmt_num(st['std'])}"
    if st.get("span") is not None:
        return f"{str(st['min'])[:10]} → {str(st['max'])[:10]}"
    if st.get("words_mean") is not None:
        return f"{st['words_mean']:.1f} words on average"
    if st.get("pct_true") is not None:
        return f"{st['pct_true']:.0f}% true"
    if st.get("top_labels"):
        return f"{st['top_labels'][0]['label']} {st['top_labels'][0]['pct_rows']:.0f}% of rows"
    top = st.get("top_values") or []
    if top and stype in CATEGORY_TYPES:
        return f"{top[0]['value']} {top[0]['pct']:.0f}%"
    return ""


def _sparkline(ax: Any, s: pd.Series, stype: str, st: dict[str, Any], box: tuple[float, float, float, float],
               cfg: StoneConfig) -> None:
    """Tiny chart drawn inside ``box`` (x, y, width, height in data units of the table axes)."""
    x0, y0, w, h = box
    family = _type_family(stype)
    heights: np.ndarray | None = None
    color = BLUE
    if family == "numeric" and stype != "id":
        v = as_numeric(s, stype).to_numpy(dtype="float64")
        v = sample_array(v[np.isfinite(v)], cfg.sample_size, cfg.random_state)
        if np.unique(v).size >= 2:
            integer_like = bool(np.all(np.mod(v, 1) == 0)) and v.max() - v.min() <= 40
            edges = _bins(v, integer_like, not integer_like and _log_x(v, st.get("skewness")))
            heights = np.histogram(v, bins=edges if len(edges) <= 30 else 24)[0].astype(float)
    elif family == "category":
        values = split_labels(s, st.get("separator") or ", ").explode() if stype == MULTI_LABEL else hashable(s.dropna())
        top = values.value_counts().head(12)
        if len(top) >= 2:
            heights, color = top.to_numpy(dtype=float), ORANGE
    elif family == "datetime":
        dt = as_datetime(s).dropna()
        if dt.nunique() >= 2:
            freq, _ = time_resolution((dt.max() - dt.min()).total_seconds() / 86_400)
            counts = pd.Series(1, index=pd.DatetimeIndex(dt)).sort_index().resample(freq).sum().to_numpy(dtype=float)
            if len(counts) >= 2:
                xs = x0 + w * np.linspace(0, 1, len(counts))
                ys = y0 + h - h * counts / counts.max()
                ax.fill_between(xs, y0 + h, ys, color="#bfe9d8", lw=0)
                ax.plot(xs, ys, color=SERIES[2], lw=1)
                return
    if heights is None or not heights.max():
        return
    k = len(heights)
    bar_w = w / max(k, 12 if family == "category" else k)
    for j, val in enumerate(heights):
        bh = h * val / heights.max()
        ax.add_patch(mpl.patches.Rectangle((x0 + j * bar_w + bar_w * 0.08, y0 + h - bh), bar_w * 0.84, bh,
                                           color=color, lw=0, alpha=0.9))


# ---------------------------------------------------------------------------------------------- relations
def plot_target_ranking(writer: _Writer, assoc: pd.DataFrame, target: str, cfg: StoneConfig) -> None:
    """Features ranked by association with the target; grey = not significant after BH correction."""
    top = assoc.head(25).iloc[::-1]
    if top.empty:
        return
    n_sig = int(assoc["significant"].sum())
    fig, body = _figure(10, 0.36 * len(top) + 1.7, f"What drives {target}?",
                        f"{len(assoc)} features tested  ·  {n_sig} significant  ·  ranked by strength of association",
                        f"Association: Spearman (numbers), eta (numbers vs labels), Cramér's V (labels). Grey = not "
                        f"significant after Benjamini-Hochberg correction (α = {cfg.alpha:g}).")
    ax = body.add_subplot()
    y = np.arange(len(top))
    vals = top["association"].abs().to_numpy()
    colors = [BLUE if sig else LIGHT for sig in top["significant"]]
    ax.barh(y, vals, color=colors, height=0.66, edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, [truncate(f, 30) for f in top["feature"]])
    xmax = min(1.0, float(vals.max())) * 1.5 + 0.06
    for yi, r in zip(y, top.itertuples(index=False)):
        extra = f"  AUC {r.auc:.2f}" if pd.notna(r.auc) else ""
        ax.text(abs(r.association) + xmax * 0.01, yi, f"{abs(r.association):.2f}", va="center", fontsize=8.5,
                color=INK, fontweight="bold")
        ax.text(abs(r.association) + xmax * 0.075, yi, f"{r.method}{extra}  ·  p_adj {fmt_p(r.p_adjusted)}",
                va="center", fontsize=7.5, color=MUTED)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("|association|")
    _value_axis_only(ax, "x")
    writer.save(fig, target, "target_ranking")


def _few_levels(v: np.ndarray) -> np.ndarray | None:
    """The sorted levels when a variable has at most 15 distinct values (plotted as categories), else None."""
    levels = np.unique(v)
    return levels if len(levels) <= 15 else None


def _binned_median(x: np.ndarray, y: np.ndarray, levels: np.ndarray | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Non-parametric trend: median of y per level of x, or per x-quantile bin (20 bins)."""
    if len(x) < 30:
        return None
    if levels is not None:
        keys = x
    else:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 21)))
        if len(edges) < 3:
            return None
        keys = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    frame = pd.DataFrame({"k": keys, "x": x, "y": y}).groupby("k").agg(x=("x", "median"), y=("y", "median"), n=("y", "size"))
    frame = frame[frame["n"] >= 5]
    return (frame["x"].to_numpy(), frame["y"].to_numpy()) if len(frame) >= 2 else None


def plot_pair(writer: _Writer, df: pd.DataFrame, a: str, b: str, label: str, cfg: StoneConfig, kind: str) -> None:
    """Joint plot: scatter (hexbin density for many points), least-squares line, marginal distributions.

    Variables with few distinct values are jittered with one tick per level; heavily right-skewed
    positive variables get a log axis.
    """
    pair = pd.DataFrame({"a": as_numeric(df[a]), "b": as_numeric(df[b])}).replace([np.inf, -np.inf], np.nan).dropna()
    n_all = len(pair)
    if n_all < 3:
        return
    pair = sample(pair, 20_000, cfg.random_state)
    x, y = pair["a"].to_numpy(), pair["b"].to_numpy()
    with quiet():
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else float("nan")
        rho = float(stats.spearmanr(x, y)[0]) if np.std(x) and np.std(y) else float("nan")
    x_levels, y_levels = _few_levels(x), _few_levels(y)
    log_x = x_levels is None and _log_x(x, float(stats.skew(x)) if len(x) > 2 else None)
    log_y = y_levels is None and _log_x(y, float(stats.skew(y)) if len(y) > 2 else None)
    dense = len(x) > 2_000 and x_levels is None and y_levels is None
    fig, body = _figure(8.4, 7.6, f"{b} vs {a}", label + f"  ·  n = {n_all:,}",
                        "Darker hexagons = more rows." if dense else ("Points jittered within each level."
                                                                      if x_levels is not None or y_levels is not None
                                                                      else ""))
    gs = body.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])
    ax = body.add_subplot(gs[1, 0])
    ax_top = body.add_subplot(gs[0, 0], sharex=ax)
    ax_right = body.add_subplot(gs[1, 1], sharey=ax)
    rng = np.random.default_rng(cfg.random_state)
    xj = x + rng.uniform(-0.18, 0.18, len(x)) * (np.diff(x_levels).min() if x_levels is not None and len(x_levels) > 1 else 0)
    yj = y + rng.uniform(-0.18, 0.18, len(y)) * (np.diff(y_levels).min() if y_levels is not None and len(y_levels) > 1 else 0)
    if dense:
        scales = {"xscale": "log" if log_x and x.min() > 0 else "linear", "yscale": "log" if log_y and y.min() > 0 else "linear"}
        ax.hexbin(x, y, gridsize=55, cmap=DENSITY, mincnt=1, linewidths=0, bins="log", **scales)
    else:
        ax.scatter(xj, yj, s=14, alpha=float(np.clip(400 / len(x), 0.25, 0.8)), color=BLUE, edgecolors="none")
        if log_x:
            _set_log(ax, x, "x")
        if log_y:
            _set_log(ax, y, "y")
    if len(x) > 2 and np.ptp(x) > 0 and not (log_x or log_y):
        try:
            with quiet():
                slope, intercept = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 50)
            ax.plot(grid, slope * grid + intercept, color=INK_2, lw=1.2, ls=(0, (4, 3)), label="least-squares line")
        except (np.linalg.LinAlgError, ValueError):  # extreme magnitudes: no straight line
            pass
    trend = _binned_median(x, y, x_levels)
    if trend is not None:
        ax.plot(*trend, color=ORANGE, lw=2.2, marker="o", ms=3.5, label="median trend")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")
    ax.text(0.02, 0.98, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}", transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID, alpha=0.9))
    ax.set_xlabel(a + (" · log scale" if log_x else ""))
    ax.set_ylabel(b + (" · log scale" if log_y else ""))
    _format_axis(ax)
    for values, levels, marginal, horizontal in ((x, x_levels, ax_top, False), (y, y_levels, ax_right, True)):
        if levels is not None:
            counts = np.array([np.sum(values == lv) for lv in levels])
            width = 0.6 * (np.diff(levels).min() if len(levels) > 1 else 1)
            (marginal.barh if horizontal else marginal.bar)(levels, counts, width, color=BLUE_FILL)
        else:
            on_log = log_y if horizontal else log_x
            if on_log:  # zeros sit on the symlog edge of the scatter; the margin shows the positive values
                values = values[values > 0]
            edges = _bins(values, False, on_log)
            marginal.hist(values, bins=edges, color=BLUE_FILL, edgecolor=SURFACE, linewidth=0.5,
                          orientation="horizontal" if horizontal else "vertical")
        marginal.axis("off")
    if x_levels is not None:
        ax.set_xticks(x_levels, [f"{lv:g}" for lv in x_levels])
        ax.grid(False, axis="x")
    if y_levels is not None:
        ax.set_yticks(y_levels, [f"{lv:g}" for lv in y_levels])
        ax.grid(False, axis="y")
    writer.save(fig, a, kind, other=b)


def _groups(cat: pd.Series, values: pd.Series, max_groups: int = 10) -> tuple[list[str], list[np.ndarray]]:
    """Values per category (largest groups), ordered by median."""
    mask = cat.notna() & values.notna() & np.isfinite(values)
    keys = hashable(cat[mask]).astype(str)
    order = keys.value_counts().index[:max_groups]
    data = [values[mask][keys == k].to_numpy(dtype="float64") for k in order]
    ranked = sorted(zip(order, data), key=lambda kv: np.median(kv[1]), reverse=True)
    return [str(k) for k, _ in ranked], [d for _, d in ranked]


def plot_target(
    writer: _Writer, df: pd.DataFrame, target: str, assoc: pd.DataFrame, types: dict[str, str], cfg: StoneConfig
) -> None:
    """Target vs the most associated features: joint plot, grouped boxplots, or stacked proportions."""
    t_type = types[target]
    t_numeric = t_type in NUMERIC_TYPES or t_type == "numeric_string"
    rng = np.random.default_rng(cfg.random_state)
    for row in assoc.head(cfg.n_target_plots).itertuples(index=False):
        feat, f_type = row.feature, row.feature_type
        f_numeric = row.method == "spearman" or (row.method == "eta" and not t_numeric)
        auc = f"  ·  AUC {row.auc:.2f}" if pd.notna(row.auc) else ""
        label = f"{row.method} = {row.association:.3f}  ·  p_adj {fmt_p(row.p_adjusted)}{auc}"
        if t_numeric and f_numeric:
            plot_pair(writer, df, feat, target, label, cfg, "target")
            continue
        if t_numeric or f_numeric:
            cat, num, num_name = ((df[feat], as_numeric(df[target], t_type), target) if t_numeric
                                  else (df[target], as_numeric(df[feat], f_type), feat))
            labels, data = _groups(cat, num.replace([np.inf, -np.inf], np.nan))
            if len(data) < 2:
                continue
            group_name = feat if t_numeric else target
            fig, body = _figure(10, 0.62 * len(data) + 2.2, f"{num_name} by {group_name}", label,
                                "Box = interquartile range, line = median, diamond = mean, dots = individual rows "
                                "(sampled); groups sorted by median.")
            ax = body.add_subplot()
            _boxplot(ax, data, [f"{truncate(lbl, 24)}  (n={len(d):,})" for lbl, d in zip(labels, data)], widths=0.55)
            for gi, d in enumerate(data, start=1):
                pts = d if len(d) <= 200 else rng.choice(d, 200, replace=False)
                ax.scatter(pts, gi + rng.uniform(-0.18, 0.18, len(pts)), s=5, color=INK_2, alpha=0.25,
                           edgecolors="none", zorder=1)
            ax.invert_yaxis()
            pooled = np.concatenate(data)
            if _log_x(pooled, float(stats.skew(pooled)) if len(pooled) > 2 else None):
                _set_log(ax, pooled)
                num_name += " · log scale"
            ax.set_xlabel(num_name)
            _value_axis_only(ax, "x")
            _format_axis(ax, "x")
        else:
            mask = df[feat].notna() & df[target].notna()
            f = hashable(df[feat][mask]).astype(str)
            t = hashable(df[target][mask]).astype(str)
            f_top = f.value_counts().index[:12]
            t_top = list(t.value_counts().index[:7])
            t = t.where(t.isin(t_top), "other")
            tab = pd.crosstab(f[f.isin(f_top)], t, normalize="index").reindex(f_top)
            sizes = f[f.isin(f_top)].value_counts()
            cols = [c for c in t_top + ["other"] if c in tab.columns]
            if len(cols) > 1:  # rank feature levels by the share of the second most frequent target class
                tab = tab.sort_values(cols[1], ascending=False)
            fig, body = _figure(10, 0.5 * len(tab) + 2.4, f"{target} by {feat}", label,
                                f"Share of each {target} value within each {feat} level.")
            ax = body.add_subplot()
            left = np.zeros(len(tab))
            yy = np.arange(len(tab))
            for i, c in enumerate(cols):
                vals = tab[c].to_numpy()
                fill = SERIES[i] if c != "other" else LIGHT
                ax.barh(yy, vals, left=left, color=fill, height=0.68, edgecolor=SURFACE, linewidth=1.5,
                        label=truncate(c, 20))
                for yi, (lft, v) in enumerate(zip(left, vals)):
                    if v >= 0.07:
                        ax.text(lft + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8,
                                color=_text_color(fill))
                left += vals
            ax.set_yticks(yy, [f"{truncate(v, 24)}  (n={sizes[v]:,})" for v in tab.index])
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
            _value_axis_only(ax, "x")
            ax.legend(title=target, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=min(len(cols), 6),
                      handlelength=1, columnspacing=1.2, title_fontsize=8.5, alignment="left")
        writer.save(fig, target, "target", other=feat)


# ---------------------------------------------------------------------------------------------- entry points
def make_plots(
    df: pd.DataFrame,
    details: dict[str, dict[str, Any]],
    correlations: dict[str, pd.DataFrame],
    out_dir: str | Path,
    target: str | None = None,
    config: StoneConfig | None = None,
    errors: list[str] | None = None,
    index: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Write every informative plot given precomputed details and correlations.

    Empty, constant and id columns get no individual plot (they carry nothing to see); failing
    plots are appended to ``errors`` instead of raising, and ``index`` receives one
    ``{column, other, kind, path}`` record per chart.
    """
    cfg = config or StoneConfig()
    writer = _Writer(out_dir, cfg.dpi)
    types = {c: d.get("semantic_type", "mixed") for c, d in details.items()}
    with mpl.rc_context(STYLE), quiet():
        writer.safe("overview", plot_overview, df, details, cfg)
        writer.safe("missing matrix", plot_missing, df, cfg)
        writer.safe("nullity heatmap", plot_nullity, df, cfg)
        for key, title, signed in (
            ("pearson", "Pearson correlation", True),
            ("spearman", "Spearman rank correlation", True),
            ("cramers_v", "Cramér's V between categorical columns", False),
        ):
            if key in correlations:
                writer.safe(f"{key} heatmap", plot_heatmap, correlations[key], correlations.get(f"{key}_p"),
                            f"corr_{key}", title, cfg, signed)
        assoc_all = correlations.get("association")
        if assoc_all is not None and "eta" in correlations:
            writer.safe("association heatmap", plot_heatmap, assoc_all, correlations["association_p"], "association",
                        "All associations", cfg, True,
                        f"{assoc_all.shape[0]} columns  ·  Spearman for numbers, Cramér's V for labels, eta across")
        for col in df.columns:
            stype, detail = types[col], details[col]
            if stype in NUMERIC_TYPES or stype == "numeric_string":
                writer.safe(col, plot_numeric, col, as_numeric(df[col], stype), detail, cfg)
            elif stype in CATEGORY_TYPES:
                writer.safe(col, plot_categorical, col, df[col], detail, cfg)
            elif stype in DATETIME_TYPES:
                writer.safe(col, plot_datetime, col, df[col], detail, cfg)
            elif stype == "text":
                writer.safe(col, plot_text, col, df[col], detail, cfg)
            elif stype == MULTI_LABEL:
                writer.safe(col, plot_multilabel, col, df[col], detail, cfg)
        pairs = correlations.get("top_pairs")
        if pairs is not None and not pairs.empty:
            num_pairs = pairs[pairs["method"].isin(["pearson", "spearman"])]
            if target is not None:
                num_pairs = num_pairs[(num_pairs["col_a"] != target) & (num_pairs["col_b"] != target)]
            for r in num_pairs.head(cfg.n_scatter).itertuples(index=False):
                ci = f" [{r.ci_low:.3f}, {r.ci_high:.3f}]" if pd.notna(r.ci_low) else ""
                label = f"{r.method} = {r.value:.3f}{ci}  ·  p_adj {fmt_p(r.p_adjusted)}"
                writer.safe(f"{r.col_a}/{r.col_b}", plot_pair, df, r.col_a, r.col_b, label, cfg, "scatter")
        assoc = correlations.get("target")
        if target is not None and assoc is not None and not assoc.empty:
            writer.safe("target ranking", plot_target_ranking, assoc, target, cfg)
            writer.safe("target", plot_target, df, target, assoc, types, cfg)
    if errors is not None:
        errors.extend(writer.errors)
    if index is not None:
        index.extend(writer.index)
    return writer.paths


def see_plot(
    df: pd.DataFrame | pd.Series,
    out_dir: str | Path = "elessar_report",
    target: str | None = None,
    config: StoneConfig | None = None,
) -> list[Path]:
    """Analyse ``df`` and write all informative plots to ``out_dir``. Returns the PNG paths."""
    from .core import analyze

    cfg = config or StoneConfig()
    target = resolve_target(df, target)
    df = prepare(df)
    details, correlations = analyze(df, target, cfg)
    return make_plots(df, details, correlations, out_dir, target, cfg)
