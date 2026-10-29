"""Charts of the advanced analyses: PCA, factor analysis, feature selection, outliers, time series.

Same visual system as :mod:`elessar.plots` (header, body, footer note; muted grid on a dark surface;
cyan for data, red for flagged points, neon pink for fitted or smoothed lines, mint for what is kept).
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ._utils import fmt_num, fmt_p, truncate
from .config import StoneConfig
from .plots import (AMBER, BLUE, BLUE_FILL, BLUE_PALE, DIVERGING, GRID, HIGHLIGHT, INK, INK_2, LIGHT, MINT, MUTED, ORANGE,
                    SEQUENTIAL, SURFACE, _figure, _format_axis, _text_color, _value_axis_only)


def _matrix(ax: Any, values: np.ndarray, rows: list[str], cols: list[str], signed: bool, strong: float | None = None,
            fmt: str = "{:.2f}") -> Any:
    cmap = DIVERGING if signed else SEQUENTIAL
    im = ax.pcolormesh(values, cmap=cmap, vmin=-1 if signed else 0, vmax=1, edgecolors=SURFACE, linewidth=1.5)
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5, cols)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, [truncate(r, 28) for r in rows])
    ax.grid(False)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_visible(False)
    small = values.shape[0] > 18
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            fill = mpl.colors.to_hex(cmap((v + 1) / 2 if signed else v))
            bold = strong is not None and abs(v) >= strong
            ax.text(j + 0.5, i + 0.5, fmt.format(v), ha="center", va="center", fontsize=7 if small else 8.2,
                    color=_text_color(fill) if bold or strong is None else MUTED,
                    fontweight="bold" if bold else "normal")
    return im


# ------------------------------------------------------------------------------------------- PCA / factors
def plot_pca(writer: Any, pca: dict[str, Any], cfg: StoneConfig) -> None:
    """Scree plot (variance per component, cumulative, parallel-analysis threshold) and loadings."""
    if "eigenvalues" not in pca:
        return
    eig = np.array(pca["eigenvalues"])
    p = len(eig)
    shown = min(p, 15)
    k = pca["recommended"]
    ratio = 100 * np.array(pca["explained"])[:shown]
    cum = 100 * np.array(pca["cumulative"])[:shown]
    thr = 100 * np.array(pca["parallel_threshold"])[:shown] / p
    load = pca["loadings"]
    n_comp = min(load.shape[1], max(k, 2), 6)
    top = load.iloc[:, :n_comp].pow(2).sum(axis=1).sort_values(ascending=False).index[:22]
    height = max(5.2, 0.3 * len(top) + 2.4)
    verdict = {"recommended": "PCA recommended", "weak": "PCA: little to gain", "not useful": "PCA not useful"}.get(
        pca["verdict"], "PCA")
    fig, body = _figure(13, height, f"Principal components  ·  {verdict}",
                        f"{p} numeric columns  ·  keep {k} component{'s' if k > 1 else ''} = {pca['recommended_variance'] * 100:.0f}% "
                        f"of the variance  ·  KMO {pca['kmo']:.2f} ({pca['kmo_label']})  ·  Bartlett p {fmt_p(pca['bartlett_p'])}",
                        "Bars: variance explained by each component of the standardised columns; line: cumulative. "
                        "Grey ticks: what random data of the same size would give (parallel analysis): components "
                        "above them carry real structure. Right: correlation of each column with each kept component.")
    gs = body.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = body.add_subplot(gs[0, 0])
    x = np.arange(1, shown + 1)
    ax.bar(x, ratio, color=[BLUE if i < k else BLUE_FILL for i in range(shown)], width=0.66, edgecolor=SURFACE,
           linewidth=1.2, label="variance explained")
    ax.plot(x, cum, color=ORANGE, lw=2, marker="o", ms=4, label="cumulative")
    ax.scatter(x, thr, marker="_", s=260, color=INK_2, linewidths=2, label="random data (95th pct)", zorder=4)
    for q in (80, 90):
        ax.axhline(q, color=GRID, lw=1, ls=(0, (3, 2)), zorder=0)
        ax.text(shown + 0.45, q, f"{q}%", fontsize=7.5, color=MUTED, va="center")
    ax.axvline(k + 0.5, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(k + 0.6, 50, f"keep {k}", fontsize=8.5, color=INK, rotation=90, va="center")
    ax.set_xticks(x, [f"PC{i}" for i in x], rotation=45 if shown > 8 else 0)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_title("Variance explained")
    ax.legend(loc="center right", fontsize=8)
    ax2 = body.add_subplot(gs[0, 1])
    vals = load.loc[top].iloc[:, :n_comp].to_numpy(dtype="float64")
    _matrix(ax2, np.clip(vals, -1, 1), list(top), list(load.columns[:n_comp]), True, strong=0.5)
    ax2.set_title("Loadings (correlation with the component)", pad=24)
    writer.save(fig, None, "pca")


def plot_factors(writer: Any, fa: dict[str, Any], cfg: StoneConfig) -> None:
    """Scree plot with the three criteria for the number of factors, and the rotated component matrix."""
    load, sol = fa.get("loadings"), fa.get("solution")
    if load is None or load.empty or sol is None:
        return
    eig = np.asarray(sol["eigenvalues"])
    shown = min(len(eig), 15)
    k = fa["n_factors"]
    rows = list(load.index)
    alphas = {f: r["alpha"] for f, r in (fa.get("reliability") or {}).items() if len(r["items"]) > 1}
    fig, body = _figure(13, max(5.2, 0.32 * len(rows) + 2.4), "Factor analysis (principal components, varimax)",
                        f"{k} factor{'s' if k > 1 else ''}  ·  KMO {fa['kmo']:.2f}  ·  Bartlett p {fmt_p(fa['bartlett_p'])}"
                        + "".join(f"  ·  {f} α {a:.2f}" for f, a in alphas.items())[:120],
                        "Left: share of the variance of each component (scree plot) - the dashed line is the Kaiser "
                        "criterion (eigenvalue 1) - and the cumulative share against the 60% target. Right: rotated loadings, items "
                        "grouped by the factor they load on most; bold when |loading| ≥ 0.4."
                        + (f" Removed before this solution: {', '.join(str(r['item']) for r in fa['removed'])}."
                           if fa.get("removed") else ""))
    gs = body.add_gridspec(1, 2, width_ratios=[1.1, 1])
    ax = body.add_subplot(gs[0, 0])
    x = np.arange(1, shown + 1)
    p = len(eig)
    share = 100 * eig[:shown] / p  # one axis: eigenvalue 1 = 100 / p % of the variance
    ax.plot(x, share, color=BLUE, lw=2, marker="o", ms=5, label="variance of the component (scree)")
    ax.plot(x, 100 * np.asarray(sol["cumulative"])[:shown], color=ORANGE, lw=1.6, marker="o", ms=3,
            label="cumulative")
    ax.axhline(100 / p, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.text(shown, 100 / p + 1.5, "eigenvalue 1", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.axhline(60, color=GRID, lw=1, ls=(0, (3, 2)), zorder=0)
    ax.text(shown, 61.5, "60%", fontsize=7.5, color=MUTED, ha="right", va="bottom")
    ax.axvline(k + 0.5, color=INK, lw=1, ls=(0, (4, 3)))
    ax.text(k + 0.6, 35, f"keep {k}", fontsize=8.5, color=INK, rotation=90, va="center")
    ax.set_xticks(x, [f"C{i}" for i in x])
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="center right", fontsize=8)
    ax.set_title("Scree plot")
    ax2 = body.add_subplot(gs[0, 1])
    _matrix(ax2, np.clip(load.loc[rows].to_numpy(dtype="float64"), -1, 1), rows, list(load.columns), True, strong=0.4)
    ax2.set_title("Rotated component matrix", pad=24)
    writer.save(fig, None, "factors")


# ------------------------------------------------------------------------------------------- selection
def plot_selection(writer: Any, sel: dict[str, Any], target: str, cfg: StoneConfig) -> None:
    """Every method's normalised importance per feature, and the vote."""
    table = sel["table"].head(30)
    methods = [m for m in ("decision tree", "random forest", "lasso", "ridge", "filter") if m in table.columns]
    if table.empty:
        return
    scores = sel["scores"]
    sub = f"{sel['task']}  ·  {sel['n_rows']:,} rows  ·  {sel['cv']} CV {sel['scoring']}: " + ", ".join(
        f"{m} {scores[m]:.2f}" for m in methods if m in scores)
    fig, body = _figure(12, 0.34 * len(table) + 2.6, f"Feature selection for {target}", sub,
                        "Cells: importance by each method, scaled to the strongest feature (1). Decision tree: impurity "
                        "importance; random forest: drop in held-out score when the feature is shuffled; lasso / ridge: "
                        "size of the standardised coefficient (0 in lasso = dropped); filter: univariate association. "
                        "Keep = selected by 3+ methods, borderline = 2.")
    gs = body.add_gridspec(1, 2, width_ratios=[len(methods), 1.6])
    ax = body.add_subplot(gs[0, 0])
    vals = table[methods].to_numpy(dtype="float64")
    _matrix(ax, np.clip(vals, 0, 1), list(table.index), methods, False, strong=None)
    votes = sel["votes"].reindex(table.index)
    for i, f in enumerate(table.index):
        for j, m in enumerate(methods):
            if m in votes.columns and votes.loc[f, m]:
                ax.add_patch(mpl.patches.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fill=False, edgecolor=INK, lw=1.1))
    ax2 = body.add_subplot(gs[0, 1], sharey=ax)
    colors = {"keep": MINT, "borderline": AMBER, "drop": LIGHT}
    for i, (_, r) in enumerate(table.iterrows()):
        ax2.barh(i + 0.5, r["votes"], color=colors[r["decision"]], height=0.66, edgecolor=SURFACE)
        ax2.text(r["votes"] + 0.1, i + 0.5, r["decision"], va="center", fontsize=8, color=INK_2)
    ax2.set_xlim(0, len(methods) + 1.6)
    ax2.set_xticks(range(len(methods) + 1))
    ax2.set_title("votes", loc="left")
    ax2.tick_params(labelleft=False)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    _value_axis_only(ax2, "x")
    writer.save(fig, target, "selection")


def plot_lasso_path(writer: Any, path: pd.DataFrame, target: str, cfg: StoneConfig) -> None:
    if path is None or path.empty or path.shape[0] < 3:
        return
    fig, body = _figure(11, 5.4, "Lasso path", f"{path.shape[1]} strongest coefficients as the penalty decreases",
                        "Each line is a standardised coefficient; features entering early (left) matter most. "
                        "Categorical features appear as their one-hot levels.")
    ax = body.add_subplot()
    alphas = path.index.to_numpy(dtype="float64")
    for i, col in enumerate(path.columns):
        color = mpl.colormaps["tab10"](i % 10) if i < 10 else LIGHT
        ax.plot(alphas, path[col], lw=1.8, color=color)
        ax.text(alphas[-1], path[col].iloc[-1], f"  {truncate(col, 22)}", fontsize=7.5, color=color, va="center")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("penalty alpha (log scale, strong → weak)")
    ax.set_ylabel("coefficient")
    _format_axis(ax, "y")
    writer.save(fig, target, "lasso_path")


# ------------------------------------------------------------------------------------------- outliers
def plot_outliers(writer: Any, outl: dict[str, Any], cfg: StoneConfig) -> None:
    """Every numeric column on one robust scale, its boxplot fences, and the flagged points named by row."""
    cols = [c for c, r in outl["columns"].items() if r["counts"]["boxplot"] or r["counts"]["3 sigma"]]
    cols = sorted(cols, key=lambda c: -outl["columns"][c]["n_consensus"])[:20]
    if not cols:
        return
    n_points = int(sum(outl["columns"][c]["n_consensus"] for c in cols))
    fig, body = _figure(12, 0.55 * len(cols) + 2.4, "Outlier points",
                        f"{len(cols)} columns with flagged values  ·  {n_points:,} consensus outliers (2+ methods)",
                        "Each column on a robust scale: (value − median) / (IQR / 1.349). Box = interquartile range, "
                        "whiskers = boxplot fences. Red: flagged by 2-3 methods (3 sigma, boxplot, modified z); "
                        "amber: by one. The most extreme points are labelled with their row.")
    ax = body.add_subplot()
    rng = np.random.default_rng(cfg.random_state)
    lim = 1.0
    for i, c in enumerate(cols):
        r = outl["columns"][c]
        z = r["robust"]
        q1, q3 = np.percentile(z, [25, 75])
        scale = (r["q3"] - r["q1"]) / 1.349 or 1.0
        lo, hi = (r["lower_fence"] - r["median"]) / scale, (r["upper_fence"] - r["median"]) / scale
        ax.add_patch(mpl.patches.Rectangle((q1, i - 0.22), q3 - q1, 0.44, facecolor=BLUE_PALE, edgecolor=BLUE, lw=1))
        ax.plot([lo, q1], [i, i], color=BLUE, lw=1)
        ax.plot([q3, hi], [i, i], color=BLUE, lw=1)
        inside = z[(z >= lo) & (z <= hi)]
        if len(inside) > 300:
            inside = inside.sample(300, random_state=cfg.random_state)
        ax.scatter(inside, i + rng.uniform(-0.15, 0.15, len(inside)), s=3, color=LIGHT, edgecolors="none", zorder=2)
        pts = r["points"].head(600)  # sorted by |robust z|: the most extreme
        for mask, color in ((pts["n_methods"] < 2, AMBER), (pts["n_methods"] >= 2, HIGHLIGHT)):
            sel = pts[mask]
            ax.scatter(sel["robust_z"], np.full(len(sel), i) + rng.uniform(-0.1, 0.1, len(sel)), s=16, color=color,
                       edgecolors=SURFACE, linewidths=0.4, zorder=3)
        if len(pts):
            pt = pts.iloc[0]
            ax.annotate(f"row {truncate(pt['row'], 12)}: {fmt_num(pt['value'])}", (pt["robust_z"], i), xytext=(0, 8),
                        textcoords="offset points", fontsize=7, color=INK_2, ha="right" if pt["robust_z"] > 0 else "left")
        lim = max(lim, float(np.abs(z).max()))
    ax.set_yticks(range(len(cols)), [truncate(c, 26) for c in cols])
    ax.set_ylim(len(cols) - 0.4, -0.7)
    if lim > 30:
        ax.set_xscale("symlog", linthresh=3)
        ticks = [t for t in (-100, -30, -10, -3, 0, 3, 10, 30, 100, 300, 1000) if abs(t) <= lim * 1.2]
        ax.set_xticks(ticks)
    ax.axvline(0, color=GRID, lw=1)
    ax.set_xlabel("robust z")
    _format_axis(ax, "x")
    _value_axis_only(ax, "x")
    writer.save(fig, None, "outliers")


def plot_multivariate_outliers(writer: Any, mo: dict[str, Any], cfg: StoneConfig) -> None:
    if mo.get("verdict") != "computed":
        return
    d = mo["distance"].reset_index(drop=True)
    both = mo["flagged_both"].reset_index(drop=True).to_numpy()
    cut = mo["cutoff_distance"]
    fig, body = _figure(12, 4.6, "Multivariate outliers",
                        f"{mo['n_columns']} numeric columns  ·  {mo['n_mahalanobis']:,} rows beyond the cutoff "
                        f"({mo['pct_mahalanobis']:.1f}%)  ·  {mo['n_both']:,} also isolated by the isolation forest",
                        f"Robust Mahalanobis distance of each row ({mo['method']}, columns Yeo-Johnson-normalised); dashed line: "
                        "chi-square 99.9% cutoff. "
                        "Red: confirmed by an isolation forest. Rows labelled are the most extreme.")
    ax = body.add_subplot()
    over = d > cut
    ax.scatter(np.flatnonzero(~over), d[~over], s=4, color=BLUE_FILL, edgecolors="none")
    ax.scatter(np.flatnonzero(over & ~both), d[over & ~both], s=9, color=ORANGE, edgecolors="none")
    ax.scatter(np.flatnonzero(both), d[both], s=12, color=HIGHLIGHT, edgecolors="none")
    ax.axhline(cut, color=INK, lw=1, ls=(0, (4, 3)))
    for _, r in mo["top"].head(5).iterrows():
        pos = int(np.flatnonzero(mo["distance"].index == r["row"])[0]) if (mo["distance"].index == r["row"]).any() else None
        if pos is not None:
            ax.annotate(f"row {truncate(r['row'], 12)}", (pos, r["distance"]), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=INK_2)
    if d.max() / max(d.median(), 1e-9) > 30:
        ax.set_yscale("log")
    ax.set_xlabel("row (position in the analysed rows)")
    ax.set_ylabel("robust distance")
    _format_axis(ax, "xy")
    writer.save(fig, None, "multivariate_outliers")


# ------------------------------------------------------------------------------------------- time series
def _time_axis(ax: Any) -> None:
    loc = mdates.AutoDateLocator(maxticks=8)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def plot_series(writer: Any, ts: dict[str, Any], cfg: StoneConfig) -> None:
    series = ts["series"]
    values = [v for v in ts["values"] if v in series][:4]
    if not values:
        return
    main = ts["analyses"].get(ts["main"], {})
    window = main.get("period") or max(3, len(series) // 50)
    fig, body = _figure(12, 2.0 * len(values) + 1.8, f"Time series  ·  {ts['kind']}",
                        f"{ts['time_column']}  ·  {ts['frequency']} steps  ·  {ts['start']:%Y-%m-%d} → {ts['end']:%Y-%m-%d}  ·  "
                        f"{ts['n_steps']:,} steps" + (f"  ·  mean over {ts['n_entities']} {ts['entity']}" if ts.get("entity") else ""),
                        f"Pink: rolling mean over {window} steps. " + ("Rows per period for an event log. "
                                                                         if ts["kind"] == "event log" else ""))
    axes = body.subplots(len(values), 1, sharex=True, squeeze=False)[:, 0]
    for ax, v in zip(axes, values):
        y = series[v]
        ax.plot(series.index, y, color=BLUE, lw=0.8, alpha=0.8)
        ax.plot(series.index, y.rolling(window, center=True, min_periods=1).mean(), color=ORANGE, lw=1.8)
        a = ts["analyses"].get(v, {})
        tag = ", ".join(x for x in (f"trend {a.get('trend')}" if a.get("trend") not in (None, "none") else "",
                                    f"season {a['period']} ({a['seasonality']})" if a.get("period") else "",
                                    (a.get("stationarity") or {}).get("verdict", "")) if x)
        ax.set_title(f"{truncate(v, 40)}" + (f"   ·   {tag}" if tag else ""), fontsize=9.5)
        _format_axis(ax, "y")
        _time_axis(ax)
    writer.save(fig, None, "timeseries")


def plot_acf(writer: Any, ts: dict[str, Any], cfg: StoneConfig) -> None:
    main = ts["analyses"].get(ts["main"], {})
    r = main.get("acf")
    if r is None or len(r) < 3:
        return
    period = main.get("period")
    max_lag = min(len(r) - 1, max(2 * period if period else 0, 40))
    fig, body = _figure(12, 3.8, f"Autocorrelation of {truncate(ts['main'], 40)}",
                        f"lag 1 = {main.get('acf_lag1', 0):.2f}" + (f"  ·  seasonal peak at lag {period}" if period else ""),
                        "Shaded: 95% band from Bartlett's formula (bars outside it are autocorrelations beyond those "
                        "explained by the shorter lags); a peak at a seasonal lag reveals the period.")
    ax = body.add_subplot()
    lags = np.arange(1, max_lag + 1)
    ax.bar(lags, r[1:max_lag + 1], color=[HIGHLIGHT if period and lag == period else BLUE for lag in lags],
           width=0.8 if max_lag < 120 else 1.0)
    se = 1.96 * np.asarray(main["acf_se"])[1:max_lag + 1]
    ax.fill_between(lags, -se, se, color=BLUE_PALE, zorder=0, step="mid")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("lag (steps)")
    ax.set_ylim(min(-0.3, float(r[1:max_lag + 1].min()) - 0.05), 1.02)
    writer.save(fig, ts["main"], "acf")


def plot_decomposition(writer: Any, ts: dict[str, Any], cfg: StoneConfig) -> None:
    main = ts["analyses"].get(ts["main"], {})
    parts = main.get("decomposition")
    if not parts:
        return
    idx = ts["series"].index[-len(parts["observed"]):]
    fig, body = _figure(12, 7.2, f"Seasonal decomposition of {truncate(ts['main'], 40)}",
                        f"period {main['period']}  ·  seasonal strength {main['seasonal_strength']:.2f}  ·  "
                        f"trend strength {main['trend_strength']:.2f}",
                        "Classical additive decomposition: centred moving-average trend, average seasonal profile, "
                        "residual. Strengths: 1 − var(residual) / var(component + residual).")
    axes = body.subplots(4, 1, sharex=True)
    for ax, key, color in zip(axes, ("observed", "trend", "seasonal", "resid"), (BLUE, ORANGE, MINT, MUTED)):
        if key == "resid":
            ax.scatter(idx, parts[key], s=2, color=color)
            ax.axhline(0, color=INK_2, lw=0.6)
        else:
            ax.plot(idx, parts[key], color=color, lw=1)
        ax.set_ylabel(key, fontsize=8.5)
        _format_axis(ax, "y")
        _time_axis(ax)
    writer.save(fig, ts["main"], "decomposition")


__all__ = ["plot_acf", "plot_decomposition", "plot_factors", "plot_lasso_path", "plot_multivariate_outliers",
           "plot_outliers", "plot_pca", "plot_selection", "plot_series"]
