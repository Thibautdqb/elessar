"""Multivariate statistics: is dimensionality reduction worth it (PCA, factor analysis), which
columns form redundant groups, and which rows are outliers on several columns at once.

Everything runs on the numeric columns (identifiers, constants and the target left out),
standardised, with missing values replaced by the column median.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from ._utils import fmt_p, quiet, sample
from .config import StoneConfig
from .correlations import _numeric_frame, _roles

_SKIP = {"id", "constant", "empty", "boolean"}


def numeric_block(df: pd.DataFrame, types: dict[str, str], target: str | None, cfg: StoneConfig) -> pd.DataFrame:
    """Numeric features as floats (sampled rows), without the target, ids and near-constant columns."""
    df = sample(df, cfg.sample_size, cfg.random_state)
    numeric, _ = _roles(df, types, cfg)
    cols = [c for c in numeric if c != target and types.get(c) not in _SKIP]
    num = _numeric_frame(df, cols, types)
    keep = [c for c in num.columns if num[c].notna().mean() >= 0.5 and num[c].nunique() > 2 and num[c].std() > 0]
    return num[keep]


def _standardize(num: pd.DataFrame) -> np.ndarray:
    filled = num.fillna(num.median())
    return ((filled - filled.mean()) / filled.std(ddof=0)).to_numpy(dtype="float64", copy=True)


def _gaussianize(num: pd.DataFrame) -> np.ndarray:
    """Yeo-Johnson then standardise each column: Mahalanobis cutoffs assume roughly normal margins,
    and skewed columns would otherwise flag their whole long tail."""
    filled = num.fillna(num.median())
    cols = []
    with quiet():
        for c in filled.columns:
            v = filled[c].to_numpy(dtype="float64", copy=True)
            try:
                v, _ = stats.yeojohnson(v)
            except (ValueError, FloatingPointError):
                pass
            sd = v.std()
            cols.append((v - v.mean()) / (sd if sd > 0 else 1.0))
    return np.column_stack(cols)


def bartlett_sphericity(corr: np.ndarray, n: int) -> tuple[float, float]:
    """Bartlett's test that the correlation matrix is the identity (nothing to reduce)."""
    p = corr.shape[0]
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:
        return float("inf"), 0.0  # singular: exact linear dependencies, maximal redundancy
    chi2 = -(n - 1 - (2 * p + 5) / 6) * logdet
    return float(chi2), float(stats.chi2.sf(chi2, p * (p - 1) / 2))


def kmo(corr: np.ndarray) -> tuple[float, np.ndarray]:
    """Kaiser-Meyer-Olkin sampling adequacy: overall and per variable (0.5 poor, 0.7 middling, 0.8+ good)."""
    inv = np.linalg.pinv(corr)
    d = np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
    partial = -inv / np.where(d == 0, 1, d)
    np.fill_diagonal(partial, 0)
    r2 = corr.copy()
    np.fill_diagonal(r2, 0)
    r2, p2 = r2**2, partial**2
    per_var = r2.sum(0) / np.where(r2.sum(0) + p2.sum(0) == 0, 1, r2.sum(0) + p2.sum(0))
    overall = r2.sum() / (r2.sum() + p2.sum()) if r2.sum() + p2.sum() else 0.0
    return float(overall), per_var


def parallel_analysis(n: int, p: int, seed: int, n_iter: int = 30) -> np.ndarray:
    """95th percentile of the correlation eigenvalues of random normal data of the same shape (Horn)."""
    rng = np.random.default_rng(seed)
    n = min(n, 5_000)
    eig = np.empty((n_iter, p))
    for i in range(n_iter):
        x = rng.standard_normal((n, p))
        eig[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False)))[::-1]
    return np.percentile(eig, 95, axis=0)


def _strength(kmo_value: float) -> str:
    for cut, word in ((0.9, "marvellous"), (0.8, "meritorious"), (0.7, "middling"), (0.6, "mediocre"), (0.5, "miserable")):
        if kmo_value >= cut:
            return word
    return "unacceptable"


def pca_analysis(num: pd.DataFrame, cfg: StoneConfig) -> dict[str, Any]:
    """Principal components of the standardised numeric columns, and whether PCA is worth proposing."""
    p, n = num.shape[1], len(num)
    out: dict[str, Any] = {"n_variables": p, "n_rows": n, "columns": list(num.columns)}
    if p < 3 or n < max(20, 3 * p):
        out.update(verdict="not applicable", reason=f"needs at least 3 numeric columns and enough rows ({p} columns, {n} rows)")
        return out
    x = _standardize(num)
    corr = np.corrcoef(x, rowvar=False)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]
    for j in range(eigvec.shape[1]):  # deterministic signs: largest loading positive
        if eigvec[np.argmax(np.abs(eigvec[:, j])), j] < 0:
            eigvec[:, j] *= -1
    ratio = eigval / eigval.sum()
    cum = np.cumsum(ratio)
    threshold = parallel_analysis(n, p, cfg.random_state)
    k_parallel = int(np.argmax(eigval <= threshold)) if (eigval <= threshold).any() else p
    k_kaiser = int((eigval > 1).sum())
    k = {f"k{int(q * 100)}": int(np.searchsorted(cum, q) + 1) for q in (0.8, 0.9, 0.95)}
    chi2, bart_p = bartlett_sphericity(corr, n)
    kmo_all, kmo_var = kmo(corr)
    recommended = max(1, k_parallel)
    loadings = pd.DataFrame(eigvec * np.sqrt(eigval), index=num.columns,
                            columns=[f"PC{i + 1}" for i in range(p)])
    out.update(
        eigenvalues=eigval.tolist(), explained=ratio.tolist(), cumulative=cum.tolist(),
        parallel_threshold=threshold.tolist(), k_parallel=k_parallel, k_kaiser=k_kaiser, **k,
        bartlett_chi2=chi2, bartlett_p=bart_p, kmo=kmo_all, kmo_label=_strength(kmo_all),
        kmo_per_variable=dict(zip(num.columns, map(float, kmo_var))),
        recommended=recommended, recommended_variance=float(cum[recommended - 1]),
        loadings=loadings.iloc[:, :max(recommended, min(p, 3))],
        top_variables={f"PC{j + 1}": [c for c in loadings.iloc[:, j].abs().sort_values(ascending=False).index[:4]]
                       for j in range(min(p, max(recommended, 3)))},
    )
    if bart_p >= cfg.alpha:
        out.update(verdict="not useful", reason=f"the columns are close to uncorrelated (Bartlett p {fmt_p(bart_p)}): "
                                                "every component carries its own information")
    elif kmo_all < 0.5:
        out.update(verdict="not useful", reason=f"sampling adequacy too low (KMO {kmo_all:.2f}, {_strength(kmo_all)})")
    elif recommended >= 0.75 * p:
        out.update(verdict="weak", reason=f"{recommended} components are needed for {p} columns: little to gain")
    else:
        out.update(verdict="recommended", reason=f"{recommended} components (parallel analysis) keep "
                                                 f"{100 * cum[recommended - 1]:.0f}% of the variance of {p} columns "
                                                 f"(KMO {kmo_all:.2f} {_strength(kmo_all)}, Bartlett p {fmt_p(bart_p)})")
    out["code"] = (f"from sklearn.pipeline import make_pipeline\nfrom sklearn.impute import SimpleImputer\n"
                   f"from sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n"
                   f"cols = {list(num.columns)!r}\n"
                   f"pca = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), PCA(n_components={recommended}))\n"
                   f"scores = pca.fit_transform(df[cols])")
    return out


def varimax(loadings: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Varimax rotation with Kaiser normalisation: each column loads strongly on few factors."""
    h = np.sqrt((loadings**2).sum(axis=1, keepdims=True))
    h[h == 0] = 1
    a = loadings / h
    p, k = a.shape
    rot = np.eye(k)
    crit = 0.0
    for _ in range(max_iter):
        b = a @ rot
        u, sv, vt = np.linalg.svd(a.T @ (b**3 - b @ np.diag((b**2).sum(axis=0)) / p))
        rot = u @ vt
        new = sv.sum()
        if new - crit < tol * max(new, 1e-12):
            break
        crit = new
    return (a @ rot) * h


def factor_analysis(num: pd.DataFrame, pca: dict[str, Any], cfg: StoneConfig) -> dict[str, Any]:
    """Varimax-rotated factor analysis when the correlations support latent factors."""
    p = num.shape[1]
    out: dict[str, Any] = {}
    if pca.get("verdict") in (None, "not applicable") or p < 4:
        out.update(verdict="not applicable", reason="needs at least 4 correlated numeric columns")
        return out
    k = int(min(max(1, pca["k_parallel"]), p // 2))
    if pca["bartlett_p"] >= cfg.alpha or pca["kmo"] < 0.6 or k < 1:
        out.update(verdict="not useful", reason=f"KMO {pca['kmo']:.2f} ({pca['kmo_label']}) and Bartlett p "
                                                f"{fmt_p(pca['bartlett_p'])}: no clear shared factors")
        return out
    from sklearn.decomposition import FactorAnalysis

    x = _standardize(num)
    if len(x) > 5_000:  # loadings are stable long before that; the iterative fit is linear in rows
        x = x[np.random.default_rng(cfg.random_state).choice(len(x), 5_000, replace=False)]
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fa = FactorAnalysis(n_components=k, svd_method="lapack", random_state=cfg.random_state)
        fa.fit(x)
    raw = fa.components_.T
    load = pd.DataFrame(varimax(raw) if k > 1 else raw, index=num.columns, columns=[f"F{i + 1}" for i in range(k)])
    for c in load.columns:  # sign: the strongest loading positive
        if load[c].iloc[int(np.argmax(load[c].abs().to_numpy()))] < 0:
            load[c] *= -1
    communality = (load**2).sum(axis=1)
    names = {}
    for c in load.columns:
        strong = load[c][load[c].abs() >= 0.4].abs().sort_values(ascending=False)
        names[c] = list(strong.index[:4]) or [load[c].abs().idxmax()]
    out.update(
        verdict="recommended", n_factors=k, loadings=load, communality=communality.to_dict(),
        variance=((load**2).sum(axis=0) / p).to_dict(), defined_by=names,
        weak_columns=[c for c, h in communality.items() if h < 0.2],
        reason=f"{k} latent factor{'s' if k > 1 else ''} (varimax) explain "
               f"{100 * float((load**2).to_numpy().sum()) / p:.0f}% of the shared variance of {p} columns",
        code=(f"from sklearn.decomposition import FactorAnalysis\nfrom sklearn.preprocessing import StandardScaler\n"
              f"cols = {list(num.columns)!r}\nx = StandardScaler().fit_transform(df[cols].fillna(df[cols].median()))\n"
              f"factors = FactorAnalysis(n_components={k}, rotation='varimax').fit_transform(x)"),
    )
    return out


def feature_groups(num: pd.DataFrame, threshold: float = 0.7) -> list[dict[str, Any]]:
    """Groups of columns whose pairwise |Spearman| is high (average linkage): candidates to merge or thin out."""
    if num.shape[1] < 2:
        return []
    with quiet():
        rho = num.corr(method="spearman").abs().fillna(0).to_numpy(copy=True)
    np.fill_diagonal(rho, 1)
    dist = np.clip(1 - rho, 0, None)
    np.fill_diagonal(dist, 0)
    labels = fcluster(linkage(squareform(dist, checks=False), "average"), t=1 - threshold, criterion="distance")
    groups = []
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        if len(idx) < 2:
            continue
        sub = rho[np.ix_(idx, idx)]
        cols = [num.columns[i] for i in idx]
        keep = cols[int(np.argmax(sub.mean(axis=1)))]
        groups.append({"columns": cols, "mean_abs_spearman": float(sub[np.triu_indices(len(idx), 1)].mean()),
                       "representative": keep})
    return sorted(groups, key=lambda g: -len(g["columns"]))


def multivariate_outliers(num: pd.DataFrame, cfg: StoneConfig) -> dict[str, Any]:
    """Rows far from the bulk on several columns at once: robust Mahalanobis distance (minimum covariance
    determinant, on Yeo-Johnson-normalised columns) against the chi-square 99.9% quantile, cross-checked
    with an isolation forest."""
    # continuous columns only (discrete counts give the robust core zero variance), one per collinear group
    num = num[[c for c in num.columns if num[c].nunique() >= 20]]
    with quiet():
        corr = num.corr().abs()
    keep: list[str] = []
    for c in num.columns:
        if all(corr.loc[c, k] < 0.98 for k in keep):
            keep.append(c)
    num = num[keep]
    while num.shape[1] > 2:  # drop exact or near linear combinations (Total = sum of parts, x / y / z)
        with quiet():
            r = num.fillna(num.median()).corr().to_numpy()
            vif = np.diag(np.linalg.inv(r + 1e-9 * np.eye(len(r))))  # pinv would hide exact dependencies
        worst = int(np.argmax(vif))
        if vif[worst] <= 30:
            break
        num = num.drop(columns=num.columns[worst])
    p, n = num.shape[1], len(num)
    if p < 2 or n < max(50, 10 * p):
        return {"verdict": "not applicable", "reason": f"needs 2+ continuous numeric columns and {max(50, 10 * p)}+ rows"}
    from sklearn.covariance import MinCovDet
    from sklearn.ensemble import IsolationForest

    x = _gaussianize(num)
    fit_rows = x if n <= 20_000 else x[np.random.default_rng(cfg.random_state).choice(n, 20_000, replace=False)]
    method = "minimum covariance determinant"
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mcd = MinCovDet(random_state=cfg.random_state).fit(fit_rows)
            d2 = mcd.mahalanobis(x)
        except (ValueError, np.linalg.LinAlgError):
            method = "classical covariance (robust fit failed)"
            inv = np.linalg.pinv(np.cov(x, rowvar=False))
            d2 = np.einsum("ij,jk,ik->i", x, inv, x)
        forest = IsolationForest(n_estimators=200, contamination="auto", random_state=cfg.random_state)
        forest_flag = forest.fit_predict(x) == -1
        forest_score = -forest.score_samples(x)
    cutoff = float(stats.chi2.ppf(0.999, p))
    flag = d2 > cutoff
    both = flag & forest_flag
    median = num.median()
    scale = (num.quantile(0.75) - num.quantile(0.25)).replace(0, np.nan)
    rows = []
    for i in np.argsort(-d2)[:25]:
        if not flag[i]:
            break
        r = num.iloc[i]
        z = ((r - median) / scale).abs().sort_values(ascending=False)
        drivers = ", ".join(f"{c}={r[c]:.4g}" for c in z.index[:3] if pd.notna(r[c]))
        rows.append({"row": num.index[i], "distance": float(np.sqrt(d2[i])), "isolation_score": float(forest_score[i]),
                     "isolation_forest": bool(forest_flag[i]), "drivers": drivers})
    reliable = flag.mean() <= 0.15
    return {
        "verdict": "computed", "reliable": bool(reliable), "columns": list(num.columns),
        "note": "" if reliable else (f"{100 * flag.mean():.0f}% of rows beyond the cutoff: the columns are far from "
                                     "elliptical (clusters, heavy tails), so read only the rows confirmed by both methods"),
        "method": method, "n_rows": n, "n_columns": p, "cutoff_distance": float(np.sqrt(cutoff)),
        "n_mahalanobis": int(flag.sum()), "pct_mahalanobis": 100.0 * flag.mean(),
        "n_isolation_forest": int(forest_flag.sum()), "n_both": int(both.sum()), "pct_both": 100.0 * both.mean(),
        "top": pd.DataFrame(rows), "distance": pd.Series(np.sqrt(d2), index=num.index),
        "flagged_both": pd.Series(both, index=num.index),
    }


def multivariate(df: pd.DataFrame, types: dict[str, str], target: str | None, cfg: StoneConfig) -> dict[str, Any]:
    """PCA, factor analysis, redundant groups and multivariate outliers, in one dictionary."""
    num = numeric_block(df, types, target, cfg)
    out: dict[str, Any] = {"columns": list(num.columns)}
    if num.shape[1]:
        cov = num.cov()
        out["covariance"] = cov
        out["generalized_variance"] = float(np.linalg.det(np.corrcoef(_standardize(num), rowvar=False))) \
            if num.shape[1] > 1 else None
    out["pca"] = pca_analysis(num, cfg)
    out["factor"] = factor_analysis(num, out["pca"], cfg)
    out["groups"] = feature_groups(num)
    out["outliers"] = multivariate_outliers(num, cfg)
    return out


__all__ = ["bartlett_sphericity", "factor_analysis", "feature_groups", "kmo", "multivariate", "multivariate_outliers",
           "parallel_analysis", "pca_analysis"]
