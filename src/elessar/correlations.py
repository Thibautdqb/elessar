"""Associations between columns, with significance: num-num, cat-cat, num-cat, target, missingness.

Every association comes with a test p-value, a Benjamini-Hochberg adjusted p-value (false
discovery rate over all the pairs tested together) and a ``significant`` flag at ``alpha``.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import hashable, prepare, quiet, resolve_target, sample
from .config import StoneConfig
from .types import CATEGORY_TYPES, NUMERIC_TYPES, as_numeric, infer_type

PAIR_COLUMNS = [
    "col_a", "col_b", "kind", "method", "value", "abs_value", "n", "test", "p_value", "p_adjusted",
    "significant", "ci_low", "ci_high",
]
TARGET_COLUMNS = [
    "feature", "feature_type", "method", "association", "n", "test", "p_value", "p_adjusted", "significant",
    "auc", "mutual_info", "leakage_suspect",
]
MISSING_TARGET_COLUMNS = ["feature", "pct_missing", "test", "effect", "p_value", "p_adjusted", "significant", "detail"]
MISSINGNESS_COLUMNS = ["column", "pct_missing", "depends_on", "test", "effect", "p_value", "p_adjusted", "significant", "detail"]


# --------------------------------------------------------------------------- statistics helpers
def bh_adjust(p_values: Any) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept as NaN)."""
    p = np.asarray(p_values, dtype="float64")
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = stats.false_discovery_control(np.clip(p[ok], 0, 1))
    return out


def corr_pvalue(r: float, n: int) -> float:
    """Two-sided p-value of a correlation coefficient (t approximation, exact for Pearson)."""
    if n < 3 or not np.isfinite(r):
        return float("nan")
    if abs(r) >= 1:
        return 0.0
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    return float(2 * stats.t.sf(t, n - 2))


def fisher_ci(r: float, n: int, alpha: float = 0.05, spearman: bool = False) -> tuple[float, float]:
    """Confidence interval of a correlation via Fisher's z (Bonett-Wright variance for Spearman)."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return (float("nan"), float("nan"))
    se = math.sqrt((1 + r * r / 2) / (n - 3)) if spearman else 1 / math.sqrt(n - 3)
    z, zc = math.atanh(r), stats.norm.ppf(1 - alpha / 2)
    return (math.tanh(z - zc * se), math.tanh(z + zc * se))


def _labels(s: pd.Series) -> pd.Series:
    return hashable(s).astype(str)


def cramers_v_test(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Bias-corrected Cramér's V (Bergsma 2013) with the chi-square test p-value and n."""
    mask = a.notna() & b.notna()
    table = pd.crosstab(_labels(a[mask]), _labels(b[mask]))
    n = int(table.to_numpy().sum())
    r, k = table.shape
    if n < 2 or r < 2 or k < 2:
        return float("nan"), float("nan"), n
    with quiet():
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    phi2c = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    v = float(math.sqrt(phi2c / denom)) if denom > 0 else float("nan")
    return v, float(p), n


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Bias-corrected Cramér's V (0..1) between two categorical Series."""
    return cramers_v_test(a, b)[0]


def correlation_ratio_test(categories: pd.Series, values: pd.Series) -> tuple[float, float, int]:
    """Correlation ratio eta with the Kruskal-Wallis p-value (robust one-way test) and n."""
    mask = categories.notna() & values.notna() & np.isfinite(values)
    y = values[mask].astype("float64")
    n = len(y)
    if n < 3:
        return float("nan"), float("nan"), n
    keys = _labels(categories[mask]).to_numpy()
    groups = y.groupby(keys)
    ss_total = float(((y - y.mean()) ** 2).sum())
    if ss_total == 0 or groups.ngroups < 2:
        return float("nan"), float("nan"), n
    ss_between = float((groups.count() * (groups.mean() - y.mean()) ** 2).sum())
    with quiet():
        p = float(stats.kruskal(*[g.to_numpy() for _, g in groups]).pvalue)
    return math.sqrt(ss_between / ss_total), p, n


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Correlation ratio eta (0..1): share of the numeric variance explained by the categories."""
    return correlation_ratio_test(categories, values)[0]


def _discretize(s: pd.Series, numeric: bool, bins: int = 10) -> pd.Series:
    if numeric:
        x = s.replace([np.inf, -np.inf], np.nan)
        return pd.qcut(x.rank(method="first"), q=min(bins, max(x.notna().sum(), 1)), labels=False, duplicates="drop")
    labels = _labels(s.where(s.notna()))
    top = labels.value_counts().index[:50]
    return labels.where(labels.isin(top) | s.isna(), "__other__").where(s.notna())


def mutual_information(x: pd.Series, y: pd.Series, x_numeric: bool, y_numeric: bool) -> float:
    """Mutual information in bits (numeric variables binned into deciles)."""
    xd, yd = _discretize(x, x_numeric), _discretize(y, y_numeric)
    mask = xd.notna() & yd.notna()
    if mask.sum() < 3:
        return float("nan")
    joint = pd.crosstab(xd[mask], yd[mask]).to_numpy(dtype="float64")
    p = joint / joint.sum()
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def auc_binary(values: pd.Series, labels: pd.Series) -> tuple[float, float]:
    """AUC of ``values`` for separating the two classes of ``labels`` (positive = minority class), with p."""
    mask = values.notna() & np.isfinite(values) & labels.notna()
    lab = _labels(labels[mask])
    classes = lab.value_counts()
    if len(classes) != 2:
        return float("nan"), float("nan")
    pos = classes.index[-1]
    a, b = values[mask][lab == pos], values[mask][lab != pos]
    with quiet():
        res = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(res.statistic / (len(a) * len(b))), float(res.pvalue)


# --------------------------------------------------------------------------- matrices
def _roles(df: pd.DataFrame, types: dict[str, str], cfg: StoneConfig) -> tuple[list[str], list[str]]:
    """Split columns into numeric and (low-cardinality) categorical roles."""
    numeric, categorical = [], []
    for col in df.columns:
        stype = types[col]
        if stype in NUMERIC_TYPES or stype == "numeric_string" or (
            stype == "ordinal-candidate" and pd.api.types.is_numeric_dtype(df[col].dtype)
        ):
            numeric.append(col)
        elif stype in CATEGORY_TYPES and hashable(df[col].dropna()).nunique() <= cfg.max_categories:
            categorical.append(col)
    return numeric, categorical


def _numeric_frame(df: pd.DataFrame, cols: list[str], types: dict[str, str]) -> pd.DataFrame:
    num = pd.DataFrame({c: as_numeric(df[c], types[c]) for c in cols}, index=df.index)
    return num.replace([np.inf, -np.inf], np.nan)


def _pairwise_n(frame: pd.DataFrame) -> pd.DataFrame:
    present = frame.notna().to_numpy(dtype="float64")
    return pd.DataFrame(present.T @ present, index=frame.columns, columns=frame.columns)


def _p_matrix(r: pd.DataFrame, n: pd.DataFrame) -> pd.DataFrame:
    p = np.vectorize(corr_pvalue, otypes=[float])(r.to_numpy(), n.to_numpy().astype(int))
    np.fill_diagonal(p, 0.0)
    return pd.DataFrame(p, index=r.index, columns=r.columns)


def _kendall(num: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(num.columns)
    tau = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    pv = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pair = num[[a, b]].dropna()
            t, p = stats.kendalltau(pair[a], pair[b]) if len(pair) > 2 else (np.nan, np.nan)
            tau.loc[a, b] = tau.loc[b, a] = t
            pv.loc[a, b] = pv.loc[b, a] = p
    return tau, pv


def variance_inflation(num: pd.DataFrame, config: StoneConfig | None = None) -> pd.DataFrame:
    """Variance inflation factor 1 / (1 - R²) of each numeric column regressed on the others (complete cases).

    VIF > 10 signals strong multicollinearity; ``inf`` means exact linear dependence.
    """
    cfg = config or StoneConfig()
    usable = num.loc[:, num.std() > 0].dropna()
    k = usable.shape[1]
    if k < 2 or len(usable) < k + 2:
        return pd.DataFrame(columns=["vif"])
    x = sample(usable, cfg.sample_size, cfg.random_state).to_numpy(dtype="float64")
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    vif = []
    for j in range(k):
        others = np.column_stack([np.ones(len(x)), np.delete(x, j, axis=1)])
        coef, *_ = np.linalg.lstsq(others, x[:, j], rcond=None)
        resid = x[:, j] - others @ coef
        r2 = 1 - resid.var() / x[:, j].var()
        vif.append(np.inf if r2 >= 1 - 1e-10 else 1 / (1 - r2))
    return pd.DataFrame({"vif": vif}, index=usable.columns).sort_values("vif", ascending=False)


def _equation(target: str, coefs: list[tuple[float, str]], intercept: float) -> str:
    """``y = 1·a - 2·b + 3`` built term by term (column names are never parsed or rewritten)."""
    def quote(name: str) -> str:
        return name if re.fullmatch(r"[\w. ]+", name) else f"`{name}`"

    terms = [(c, f"{abs(c):.4g}·{quote(name)}") for c, name in coefs]
    if intercept:
        terms.append((intercept, f"{abs(intercept):.4g}"))
    if not terms:
        return f"{quote(target)} = 0"
    first_coef, first = terms[0]
    text = ("-" if first_coef < 0 else "") + first
    text += "".join(f" {'-' if c < 0 else '+'} {t}" for c, t in terms[1:])
    return f"{quote(target)} = {text}"


def linear_dependencies(num: pd.DataFrame, vif: pd.DataFrame, config: StoneConfig | None = None) -> pd.DataFrame:
    """Recover exact linear relations among columns with infinite VIF, e.g. ``Total = 1·HP + 1·Attack``.

    The widest column (a sum usually is) is regressed on the others; an exact fit is recorded and
    the column removed, an inexact one is just removed, until fewer than two columns remain.
    """
    cfg = config or StoneConfig()
    remaining = [c for c, v in vif["vif"].items() if np.isinf(v)]
    if len(remaining) < 2:
        return pd.DataFrame(columns=["column", "equation", "columns"])
    data = sample(num[remaining].dropna(), cfg.sample_size, cfg.random_state)
    rows = []
    while len(remaining) >= 2:
        col = max(remaining, key=lambda c: float(data[c].var()))
        others = [c for c in remaining if c != col]
        y = data[col].to_numpy(dtype="float64")
        x = np.column_stack([np.ones(len(data)), data[others].to_numpy(dtype="float64")])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        if np.std(y - x @ coef) <= 1e-8 * max(float(np.std(y)), 1e-300):
            scale = max(float(np.abs(y).max()), 1.0)
            used = [(float(c), o) for c, o in zip(coef[1:], others) if abs(c) > 1e-9]
            intercept = float(coef[0]) if abs(coef[0]) > 1e-9 * scale else 0.0
            rows.append({"column": col, "equation": _equation(col, used, intercept), "columns": [col, *(o for _, o in used)]})
        remaining = others
    return pd.DataFrame(rows, columns=["column", "equation", "columns"])


def _pairs(out: dict[str, pd.DataFrame], cfg: StoneConfig) -> pd.DataFrame:
    """One row per column pair with its best association, test p-value, BH-adjusted p and CI."""
    rows = []
    if "pearson" in out:
        cols = list(out["pearson"].columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                n = int(out["n_numeric"].loc[a, b])
                rp, rs = out["pearson"].loc[a, b], out["spearman"].loc[a, b]
                if pd.isna(rp) and pd.isna(rs):
                    continue
                use_s = pd.isna(rp) or (pd.notna(rs) and abs(rs) > abs(rp))
                method, r = ("spearman", rs) if use_s else ("pearson", rp)
                p = out[f"{method}_p"].loc[a, b]
                lo, hi = fisher_ci(r, n, cfg.alpha, spearman=use_s)
                rows.append([a, b, "num-num", method, r, n, f"{method} t-test", p, lo, hi])
    for key, kind, test in (("cramers_v", "cat-cat", "chi-square"), ("eta", "cat-num", "kruskal-wallis")):
        if key not in out:
            continue
        mat, pmat, nmat = out[key], out[f"{key}_p"], out[f"{key}_n"]
        for i, a in enumerate(mat.index):
            for j, b in enumerate(mat.columns):
                if key == "cramers_v" and j <= i:
                    continue
                v = mat.loc[a, b]
                if pd.notna(v):
                    rows.append([a, b, kind, key, v, int(nmat.loc[a, b]), test, pmat.loc[a, b], np.nan, np.nan])
    frame = pd.DataFrame(rows, columns=["col_a", "col_b", "kind", "method", "value", "n", "test", "p_value",
                                        "ci_low", "ci_high"])
    frame["abs_value"] = frame["value"].abs()
    frame["p_adjusted"] = bh_adjust(frame["p_value"])
    frame["significant"] = frame["p_adjusted"] < cfg.alpha
    return frame[PAIR_COLUMNS].sort_values("abs_value", ascending=False, ignore_index=True)


def _association_matrix(out: dict[str, pd.DataFrame], pairs: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    """Unified matrix over numeric + categorical columns (Spearman / Cramér's V / eta) and the raw test p-values."""
    assoc = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    padj = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    adjusted = {(r.col_a, r.col_b): r.p_value for r in pairs.itertuples(index=False)}
    adjusted.update({(b, a): p for (a, b), p in adjusted.items()})
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if "spearman" in out and a in out["spearman"].index and b in out["spearman"].index:
                v = out["spearman"].loc[a, b]
            elif "cramers_v" in out and a in out["cramers_v"].index and b in out["cramers_v"].index:
                v = out["cramers_v"].loc[a, b]
            elif "eta" in out and a in out["eta"].index and b in out["eta"].columns:
                v = out["eta"].loc[a, b]
            elif "eta" in out and b in out["eta"].index and a in out["eta"].columns:
                v = out["eta"].loc[b, a]
            else:
                v = np.nan
            assoc.loc[a, b] = assoc.loc[b, a] = v
            padj.loc[a, b] = padj.loc[b, a] = adjusted.get((a, b), np.nan)
    return {"association": assoc, "association_p": padj}


# --------------------------------------------------------------------------- target & missingness
def target_associations(
    df: pd.DataFrame, target: str, types: dict[str, str], config: StoneConfig | None = None
) -> pd.DataFrame:
    """Association of every feature with ``target``: effect size, test, adjusted p, AUC, mutual information."""
    cfg = config or StoneConfig()
    numeric, categorical = _roles(df, types, cfg)
    t_type = types[target]
    t_numeric = target in numeric
    t_cat = not t_numeric and (target in categorical or t_type in CATEGORY_TYPES)
    if not (t_numeric or t_cat):
        return pd.DataFrame(columns=TARGET_COLUMNS)
    y = as_numeric(df[target], t_type).replace([np.inf, -np.inf], np.nan) if t_numeric else df[target]
    binary = t_cat and _labels(y.dropna()).nunique() == 2
    rows = []
    with quiet():
        for col in df.columns:
            if col == target or (col not in numeric and col not in categorical):
                continue
            f_numeric = col in numeric
            x = _numeric_frame(df, [col], types)[col] if f_numeric else df[col]
            auc = np.nan
            if f_numeric and t_numeric:
                pair = pd.concat([x, y], axis=1).dropna()
                if len(pair) < 3:
                    continue
                value, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                method, test, n = "spearman", "spearman t-test", len(pair)
            elif f_numeric:
                value, p, n = correlation_ratio_test(y, x)
                method, test = "eta", "kruskal-wallis"
                if binary:
                    auc, p = auc_binary(x, y)
                    test = "mann-whitney"
            elif t_numeric:
                value, p, n = correlation_ratio_test(x, y)
                method, test = "eta", "kruskal-wallis"
            else:
                value, p, n = cramers_v_test(x, y)
                method, test = "cramers_v", "chi-square"
            if not np.isfinite(value):
                continue
            mi = mutual_information(x, y, f_numeric, t_numeric)
            leak = abs(value) >= cfg.leakage_threshold or (np.isfinite(auc) and abs(auc - 0.5) >= cfg.leakage_threshold / 2)
            rows.append([col, types[col], method, float(value), int(n), test, float(p), np.nan, False, auc, mi, leak])
    out = pd.DataFrame(rows, columns=TARGET_COLUMNS)
    if out.empty:
        return out
    out["p_adjusted"] = bh_adjust(out["p_value"])
    out["significant"] = out["p_adjusted"] < cfg.alpha
    order = out["association"].abs().sort_values(ascending=False).index
    return out.loc[order].reset_index(drop=True)


def _fmt_level(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def missing_vs_target(
    df: pd.DataFrame, target: str, types: dict[str, str], config: StoneConfig | None = None
) -> pd.DataFrame:
    """Is the target different when a feature is missing? (informative missingness)"""
    cfg = config or StoneConfig()
    numeric, _ = _roles(df, types, cfg)
    t_numeric = target in numeric
    y = as_numeric(df[target], types[target]).replace([np.inf, -np.inf], np.nan) if t_numeric else df[target]
    rows = []
    with quiet():
        for col in df.columns:
            miss = df[col].isna()
            n_miss = int(miss[y.notna()].sum())
            if col == target or n_miss < 5 or (y.notna() & ~miss).sum() < 5:
                continue
            ok = y.notna()
            if t_numeric:
                a, b = y[ok & miss], y[ok & ~miss]
                res = stats.mannwhitneyu(a, b, alternative="two-sided")
                effect = float(res.statistic / (len(a) * len(b)))
                detail = f"median {target} = {_fmt_level(float(a.median()))} when missing vs {_fmt_level(float(b.median()))}"
                rows.append([col, 100 * miss.mean(), "mann-whitney (AUC)", effect, float(res.pvalue), detail])
            else:
                v, p, _ = cramers_v_test(miss.map({True: "missing", False: "present"}), y)
                if not np.isfinite(p):
                    continue
                labels = _labels(y[ok])
                top = labels.value_counts().index[-1]
                ra = (labels[miss[ok]] == top).mean()
                rb = (labels[~miss[ok]] == top).mean()
                detail = f"{target}={top}: {100 * ra:.0f}% when missing vs {100 * rb:.0f}% when present"
                rows.append([col, 100 * miss.mean(), "chi-square (V)", v, p, detail])
    out = pd.DataFrame(rows, columns=["feature", "pct_missing", "test", "effect", "p_value", "detail"])
    out["p_adjusted"] = bh_adjust(out["p_value"])
    out["significant"] = out["p_adjusted"] < cfg.alpha
    return out[MISSING_TARGET_COLUMNS].sort_values("p_value", ignore_index=True)


def missingness_dependence(
    df: pd.DataFrame, types: dict[str, str], config: StoneConfig | None = None
) -> pd.DataFrame:
    """Does a column go missing more often for some values of other columns? (evidence against MCAR)

    The missing indicator of every partly-missing column is tested against every other usable
    column (Mann-Whitney for numeric, chi-square for categorical), BH-adjusted over all tests.
    Effects: AUC for numeric (0.5 = no dependence), Cramér's V for categorical.
    """
    cfg = config or StoneConfig()
    df = sample(df, cfg.sample_size, cfg.random_state)
    numeric, categorical = _roles(df, types, cfg)
    num = _numeric_frame(df, numeric, types)
    rows = []
    with quiet():
        for col in df.columns:
            miss = df[col].isna()
            if miss.sum() < 5 or (~miss).sum() < 5:
                continue
            for other in numeric + categorical:
                if other == col:
                    continue
                if other in numeric:
                    x = num[other]
                    a, b = x[miss & x.notna()], x[~miss & x.notna()]
                    if len(a) < 5 or len(b) < 5 or x.nunique() < 2:
                        continue
                    res = stats.mannwhitneyu(a, b, alternative="two-sided")
                    effect = float(res.statistic / (len(a) * len(b)))
                    detail = f"median {other} = {_fmt_level(float(a.median()))} when missing vs {_fmt_level(float(b.median()))}"
                    rows.append([col, other, "mann-whitney (AUC)", effect, float(res.pvalue), detail])
                else:
                    v, p, n = cramers_v_test(miss, df[other])
                    if not np.isfinite(p):
                        continue
                    rates = miss[df[other].notna()].groupby(_labels(df[other].dropna())).mean()
                    detail = (f"{100 * rates.max():.0f}% missing when {other}={rates.idxmax()} vs "
                              f"{100 * rates.min():.0f}% when {other}={rates.idxmin()}")
                    rows.append([col, other, "chi-square (V)", v, p, detail])
    out = pd.DataFrame(rows, columns=["column", "depends_on", "test", "effect", "p_value", "detail"])
    out.insert(1, "pct_missing", out["column"].map(100 * df.isna().mean()))
    out["p_adjusted"] = bh_adjust(out["p_value"])
    out["significant"] = out["p_adjusted"] < cfg.alpha
    strength = np.where(out["test"].str.startswith("mann"), (out["effect"] - 0.5).abs() * 2, out["effect"])
    out = out.assign(_s=strength).sort_values(["significant", "_s"], ascending=False, ignore_index=True)
    return out.drop(columns="_s")[MISSINGNESS_COLUMNS]


# --------------------------------------------------------------------------- entry points
def compute_correlations(
    df: pd.DataFrame, types: dict[str, str], target: str | None = None, config: StoneConfig | None = None
) -> dict[str, pd.DataFrame]:
    """All association matrices and tests given precomputed semantic types (see :func:`see_corr`)."""
    cfg = config or StoneConfig()
    df = sample(df, cfg.sample_size, cfg.random_state)
    numeric, categorical = _roles(df, types, cfg)
    num = _numeric_frame(df, numeric, types)
    out: dict[str, pd.DataFrame] = {}
    with quiet():
        if len(numeric) >= 2:
            out["n_numeric"] = _pairwise_n(num)
            for method in ("pearson", "spearman"):
                out[method] = num.corr(method=method)
                out[f"{method}_p"] = _p_matrix(out[method], out["n_numeric"])
            out["kendall"], out["kendall_p"] = _kendall(num)
            features = num.drop(columns=[target], errors="ignore")
            out["vif"] = variance_inflation(features, cfg)
            out["linear_dependencies"] = linear_dependencies(features, out["vif"], cfg)
        if len(categorical) >= 2:
            shape = (len(categorical), len(categorical))
            frames = [pd.DataFrame(np.full(shape, fill), index=categorical, columns=categorical) for fill in (1.0, 0.0, 0.0)]
            for i, a in enumerate(categorical):
                frames[2].loc[a, a] = int(df[a].notna().sum())
                for b in categorical[i + 1:]:
                    v, p, n = cramers_v_test(df[a], df[b])
                    for frame, val in zip(frames, (v, p, n)):
                        frame.loc[a, b] = frame.loc[b, a] = val
            out["cramers_v"], out["cramers_v_p"], out["cramers_v_n"] = frames
        if numeric and categorical:
            frames = [pd.DataFrame(index=categorical, columns=numeric, dtype="float64") for _ in range(3)]
            for c in categorical:
                for n_col in numeric:
                    for frame, val in zip(frames, correlation_ratio_test(df[c], num[n_col])):
                        frame.loc[c, n_col] = val
            out["eta"], out["eta_p"], out["eta_n"] = frames
        pairs = _pairs(out, cfg)
        out["pairs"] = pairs
        out["top_pairs"] = pairs[pairs["abs_value"] >= cfg.corr_threshold].reset_index(drop=True)
        if len(numeric) + len(categorical) >= 2:
            out.update(_association_matrix(out, pairs, numeric + categorical))
        out["missingness"] = missingness_dependence(df, types, cfg)
        if target is not None:
            out["target"] = target_associations(df, target, types, cfg)
            out["missing_target"] = missing_vs_target(df, target, types, cfg)
    return out


def see_corr(
    df: pd.DataFrame | pd.Series, target: str | None = None, config: StoneConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Association matrices, significance tests and ranked pairs.

    Keys (present when applicable):

    - ``pearson``, ``spearman``, ``kendall`` and their ``*_p`` p-value matrices, ``n_numeric``
      (pairwise complete counts), ``vif`` (variance inflation factors)
    - ``cramers_v`` (cat-cat, bias-corrected) + ``cramers_v_p`` (chi-square) + ``cramers_v_n``
    - ``eta`` (rows = categorical, columns = numeric) + ``eta_p`` (Kruskal-Wallis) + ``eta_n``
    - ``association`` / ``association_p``: one matrix over every column (Spearman, Cramér's V or
      eta depending on the pair) with the raw test p-values
    - ``pairs`` (every pair tested, with p, BH-adjusted p, significance and Fisher CI) and
      ``top_pairs`` (those with ``|value| >= corr_threshold``)
    - ``missingness``: which columns' missingness depends on other columns
    - with ``target``: ``target`` (feature ranking with tests, AUC, mutual information, leakage) and
      ``missing_target`` (is the target different when a feature is missing?)
    """
    cfg = config or StoneConfig()
    target = resolve_target(df, target)
    df = prepare(df)
    types: dict[str, Any] = {}
    for col in df.columns:
        try:
            types[col] = infer_type(df[col], cfg)["semantic_type"]
        except Exception:  # noqa: BLE001 - an un-typable column is just left out
            types[col] = "mixed"
    return compute_correlations(df, types, target, cfg)


__all__ = [
    "auc_binary", "bh_adjust", "compute_correlations", "correlation_ratio", "correlation_ratio_test", "corr_pvalue",
    "cramers_v", "cramers_v_test", "fisher_ci", "see_corr", "missing_vs_target", "missingness_dependence",
    "mutual_information", "target_associations", "variance_inflation",
]
