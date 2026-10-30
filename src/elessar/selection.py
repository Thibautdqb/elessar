"""Feature selection against the target, by several embedded methods and one filter, then a vote.

Methods (all cross-validated, on standardised features; categorical columns one-hot encoded and
scored as one feature):

- **decision tree**: impurity importance of a tree whose depth is chosen by cross-validation;
- **random forest**: permutation importance on held-out rows (robust to the cardinality bias
  of impurity importance), with its standard deviation;
- **lasso** (L1): coefficients that survive the penalty chosen by cross-validation; zero = dropped.
  For a categorical target, L1-penalised logistic regression;
- **ridge** (L2): size of the standardised coefficients (L2 logistic regression for classes);
- **filter**: the univariate association with the target and its BH-adjusted p-value.

A feature is **kept** when at least three methods select it, **borderline** with two, **dropped**
otherwise. Each model is tuned by cross-validation on 80% of the rows (imputation and scaling learnt
there too) and scored once on the other 20%: that score says how much to trust its ranking. Tables
with dates are sorted in time and validated on their most recent rows, never shuffled.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._utils import hashable, quiet, sample
from .config import StoneConfig
from .correlations import _numeric_frame, _roles
from .types import NUMERIC_TYPES

METHODS = ("decision tree", "random forest", "lasso", "ridge", "filter")
_MAX_ROWS = 20_000
HOLDOUT = 0.2  # share of the rows set aside to score the tuned models


def _design(df: pd.DataFrame, features: list[str], numeric: list[str], types: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Numeric columns (as numbers, missing values kept) and one-hot categoricals; owner of every design column.
    Imputation and scaling are learnt later, on the training rows only."""
    blocks, owners = [], []
    num_cols = [c for c in features if c in numeric]
    if num_cols:
        blocks.append(_numeric_frame(df, num_cols, types))
        owners += num_cols
    for c in [c for c in features if c not in numeric]:
        lab = hashable(df[c]).astype("string").fillna("(missing)")
        dummies = pd.get_dummies(lab, prefix=str(c), prefix_sep="=", dtype="float64")
        blocks.append(dummies)
        owners += [c] * dummies.shape[1]
    design = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
    return design, owners


def _standardise(x: np.ndarray, train: np.ndarray, n_numeric: int) -> np.ndarray:
    """Median-impute and standardise the first ``n_numeric`` columns with figures of the training rows."""
    x = x.copy()
    if n_numeric:
        num = x[:, :n_numeric]
        with quiet():
            med = np.nanmedian(num[train], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        num = np.where(np.isnan(num), med, num)
        mean, std = num[train].mean(axis=0), num[train].std(axis=0)
        x[:, :n_numeric] = (num - mean) / np.where(std > 0, std, 1.0)
    return x


def _time_folds(codes: np.ndarray, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Expanding-window folds over rows sorted in time, cut between two dates (``codes``: rank of each row's
    date), so that rows of one date are never on both sides; None when there are too few dates."""
    from sklearn.model_selection import TimeSeriesSplit

    k = int(codes.max()) + 1 if len(codes) else 0
    if k < n_splits + 1:
        return None
    folds = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(np.arange(k)):
        folds.append((np.flatnonzero(codes <= tr[-1]), np.flatnonzero((codes >= te[0]) & (codes <= te[-1]))))
    return folds


def _by_feature(values: np.ndarray, owners: list[str], features: list[str]) -> pd.Series:
    return pd.Series(values, index=owners).groupby(level=0).sum().reindex(features).fillna(0.0)


def feature_selection(
    df: pd.DataFrame, target: str, types: dict[str, str], correlations: dict[str, pd.DataFrame],
    cfg: StoneConfig, time_ordered: bool = False, times: pd.Series | None = None,
    differences: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Rank and select the features of ``target`` with every method; None when there is nothing to model.

    A share of the rows (``HOLDOUT``) is set aside first: every model is tuned by cross-validation on the
    others, with imputation and scaling learnt there too, and scored once on the held-out rows, so the scores
    do not carry the optimism of picking the best setting on the folds that score it. With ``times`` (the
    date of each row) or ``time_ordered`` (rows already in time order), the rows are sorted in time, the
    held-out rows are the most recent ones and the folds are cut between two dates: a model is always
    validated on the future of what it learnt from. ``differences`` (first differences of the columns with a
    unit root, see :func:`elessar.timeseries.serial_structure`): when the target is one of them, the
    selection models its changes from the changes of those columns, since a random walk is "explained" by any
    other trending column.
    """
    from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    on_differences = differences is not None and target in differences.columns
    if on_differences:
        df = df.copy()
        for c in differences.columns:
            if c in df.columns:
                df[c] = differences[c].reindex(df.index)
    df = df[df[target].notna()]
    stamp = None
    if times is not None:
        stamp = pd.to_datetime(times, errors="coerce")
        if getattr(stamp.dt, "tz", None) is not None:
            stamp = stamp.dt.tz_convert(None)
        t = stamp.reindex(df.index)
        df = df.loc[t.notna()]
        order = np.argsort(t[t.notna()].to_numpy(), kind="stable")
        df = df.iloc[order]
        time_ordered = True
    ordered = time_ordered
    df = df.iloc[-_MAX_ROWS:] if ordered else sample(df, _MAX_ROWS, cfg.random_state)
    when = stamp.reindex(df.index) if stamp is not None else pd.Series(np.arange(len(df)), index=df.index)
    numeric, categorical = _roles(df, types, cfg)
    t_type = types[target]
    regression = target in numeric and t_type in NUMERIC_TYPES and df[target].nunique() > 10
    assoc = correlations.get("target")
    leaks = set(assoc.loc[assoc["leakage_suspect"].astype(bool), "feature"]) if assoc is not None and not assoc.empty else set()
    features = [c for c in numeric + categorical
                if c != target and c not in leaks and types.get(c) not in ("id", "constant", "empty")]
    if not features or len(df) < 30:
        return None
    x, owners = _design(df, features, numeric, types)
    n_numeric = sum(c in numeric for c in features)
    if regression:
        y = _numeric_frame(df, [target], types)[target].to_numpy(dtype="float64")
        ok = np.isfinite(y)
    else:
        y_lab = hashable(df[target]).astype("string")
        counts = y_lab.value_counts()
        ok = y_lab.isin(counts[counts >= 5].index).to_numpy()
        y = y_lab.to_numpy()
    x, y, when = x[ok], y[ok], when[ok]
    if regression:
        task = "regression"
    else:
        if len(np.unique(y)) < 2:
            return None
        task = "binary classification" if len(np.unique(y)) == 2 else f"{len(np.unique(y))}-class classification"
    raw = x.to_numpy(dtype="float64")
    n = len(raw)
    if n < 30:
        return None
    rs = cfg.random_state
    codes = pd.factorize(when.to_numpy(), sort=True)[0]
    if ordered:  # the most recent dates, whole: a date is never on both sides
        cut = codes[int(np.floor((1 - HOLDOUT) * n))]
        train = codes < cut
        if train.sum() < 20 or (~train).sum() < 10:
            train = np.arange(n) < int((1 - HOLDOUT) * n)
    else:
        strat = None if regression or min(np.unique(y, return_counts=True)[1]) < 2 else y
        tr_idx, _ = train_test_split(np.arange(n), test_size=HOLDOUT, random_state=rs, stratify=strat)
        train = np.zeros(n, bool)
        train[tr_idx] = True
    xv = _standardise(raw, train, n_numeric)
    x_tr, y_tr, x_te, y_te = xv[train], y[train], xv[~train], y[~train]
    n_tr = len(x_tr)

    def folds(k: int, rows: slice = slice(None)) -> Any:
        if ordered:
            sub = codes[train][rows]
            f = _time_folds(sub - sub.min(), k)
            if f is not None:
                return f
            from sklearn.model_selection import TimeSeriesSplit

            return TimeSeriesSplit(n_splits=k)
        return KFold(k, shuffle=True, random_state=rs) if regression else StratifiedKFold(k, shuffle=True, random_state=rs)

    cv = folds(5)
    scoring = "r2" if regression else ("roc_auc" if task.startswith("binary") else "balanced_accuracy")
    test_from = when[~train].min() if ordered else None
    out: dict[str, Any] = {"task": task, "n_rows": n, "n_features": len(features), "n_design_columns": x.shape[1],
                           "excluded_leaks": sorted(map(str, leaks)), "scoring": scoring,
                           "cv": "time-ordered 5-fold" if ordered else "5-fold",
                           "n_train": int(n_tr), "n_holdout": int(n - n_tr),
                           "validation": (f"tuned by {'time-ordered ' if ordered else ''}5-fold CV on {n_tr:,} rows, "
                                          f"scored on {n - n_tr:,} held-out "
                                          + (f"later rows (from {_fmt_time(test_from)})" if ordered else "rows")
                                          + ("; on differences: the target is not stationary, so its changes are "
                                             "modelled from the changes of the other non-stationary columns"
                                             if on_differences else "")),
                           "on_differences": sorted(map(str, differences.columns)) if on_differences else [],
                           "scores": {}, "importance": {}, "selected": {}}
    imp, sel, scores = out["importance"], out["selected"], out["scores"]
    # shadows (Boruta): each feature again, its rows shuffled - same distribution, no link with the target. A
    # method votes for a feature only when it ranks it above every shadow, refitted with the setting it tuned
    shadow_rng = np.random.default_rng(rs)
    owner_arr = np.array(owners, dtype=object)
    shadows = np.empty_like(x_tr)
    for f in features:
        cols = np.flatnonzero(owner_arr == f)
        shadows[:, cols] = x_tr[shadow_rng.permutation(n_tr)][:, cols]
    with_shadows = np.hstack([x_tr, shadows])

    def beats_shadows(importance: np.ndarray) -> pd.Series:
        real = _by_feature(np.abs(importance[:len(owners)]), owners, features)
        fake = _by_feature(np.abs(importance[len(owners):]), owners, features)
        return real > fake.max()

    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # decision tree: depth by cross-validation on the training rows
        tree_cls = DecisionTreeRegressor if regression else DecisionTreeClassifier
        leaf = max(5, n_tr // 200)
        depth_scores = {d: float(np.mean(cross_val_score(tree_cls(max_depth=d, min_samples_leaf=leaf, random_state=rs),
                                                         x_tr, y_tr, cv=cv, scoring=scoring))) for d in (2, 3, 4, 6, 8)}
        depth = max(depth_scores, key=lambda d: (np.nan_to_num(depth_scores[d], nan=-np.inf), -d))
        tree = tree_cls(max_depth=depth, min_samples_leaf=leaf, random_state=rs).fit(x_tr, y_tr)
        scores["decision tree"] = _held_out(tree, x_te, y_te, scoring)
        out["tree_depth"] = depth
        imp["decision tree"] = _by_feature(tree.feature_importances_, owners, features)
        sel["decision tree"] = beats_shadows(tree_cls(max_depth=depth, min_samples_leaf=leaf, random_state=rs)
                                             .fit(with_shadows, y_tr).feature_importances_)

        # random forest: permutation importance on the held-out rows
        forest_cls = RandomForestRegressor if regression else RandomForestClassifier
        forest = forest_cls(n_estimators=100, min_samples_leaf=3, max_features="sqrt" if not regression else 0.5,
                            random_state=rs, n_jobs=1)
        forest.fit(x_tr[-10_000:], y_tr[-10_000:])
        scores["random forest"] = _held_out(forest, x_te, y_te, scoring)
        try:
            pm, ps = _grouped_permutation(forest, x_te[:2_000], y_te[:2_000], owners, features, scoring, rs)
        except ValueError:  # the score is undefined on the held-out rows (a single class)
            pm = ps = pd.Series(0.0, index=features)
        imp["random forest"] = pm.clip(lower=0)
        out["forest_std"] = ps
        sel["random forest"] = (pm > 2 * ps) & (pm > 0.002)

        # lasso / L1 logistic
        if regression:
            grid = {"n_alphas": 40} if "n_alphas" in LassoCV().get_params() else {"alphas": 40}  # renamed in 1.7
            lasso = LassoCV(cv=cv, random_state=rs, max_iter=5_000, **grid).fit(x_tr, y_tr)
            coef = np.abs(lasso.coef_)
            out["lasso_alpha"] = float(lasso.alpha_)
            scores["lasso"] = _held_out(lasso, x_te, y_te, scoring)
            out["lasso_path"] = _lasso_path(x_tr, y_tr, owners, x.columns)
        else:
            binary = task.startswith("binary")
            m = min(n_tr, 5_000 if binary else 3_000)  # logistic paths are the slow part: fit on a subset, 3 folds
            rows = slice(n_tr - m, None)  # the most recent training rows when ordered
            small_cv = folds(3, rows)
            l1 = LogisticRegressionCV(Cs=6, cv=small_cv, penalty="l1", solver="liblinear" if binary else "saga",
                                      scoring=scoring, max_iter=500, tol=1e-3, random_state=rs).fit(x_tr[rows], y_tr[rows])
            coef = np.abs(np.atleast_2d(l1.coef_)).sum(axis=0)
            out["lasso_alpha"] = float(1 / np.atleast_1d(l1.C_)[0])
            scores["lasso"] = _held_out(l1, x_te, y_te, scoring)
        imp["lasso"] = _by_feature(coef, owners, features)
        if regression:
            refit = Lasso(alpha=lasso.alpha_, max_iter=5_000).fit(with_shadows, y_tr)
        else:
            refit = LogisticRegression(C=float(np.atleast_1d(l1.C_)[0]), penalty="l1", solver="liblinear" if binary
                                       else "saga", max_iter=500, tol=1e-3, random_state=rs).fit(with_shadows[rows], y_tr[rows])
        sel["lasso"] = beats_shadows(np.abs(np.atleast_2d(refit.coef_)).sum(axis=0))

        # ridge / L2 logistic
        if regression:
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv).fit(x_tr, y_tr)
            rcoef = np.abs(ridge.coef_)
            scores["ridge"] = _held_out(ridge, x_te, y_te, scoring)
            out["ridge_alpha"] = float(ridge.alpha_)
        else:
            l2 = LogisticRegressionCV(Cs=6, cv=small_cv, penalty="l2", scoring=scoring, max_iter=500,
                                      tol=1e-3).fit(x_tr[rows], y_tr[rows])
            rcoef = np.abs(np.atleast_2d(l2.coef_)).sum(axis=0)
            scores["ridge"] = _held_out(l2, x_te, y_te, scoring)
        imp["ridge"] = _by_feature(rcoef, owners, features)
        if regression:
            refit = Ridge(alpha=float(ridge.alpha_)).fit(with_shadows, y_tr)
        else:
            refit = LogisticRegression(C=float(np.atleast_1d(l2.C_)[0]), max_iter=500, tol=1e-3).fit(with_shadows[rows],
                                                                                                    y_tr[rows])
        sel["ridge"] = beats_shadows(np.abs(np.atleast_2d(refit.coef_)).sum(axis=0))

    if assoc is not None and not assoc.empty:
        a = assoc.set_index("feature")
        imp["filter"] = a["association"].abs().reindex(features).fillna(0.0)
        sel["filter"] = a["significant"].reindex(features).fillna(False).astype(bool)
    table = pd.DataFrame({m: imp[m] / imp[m].max() if imp[m].max() > 0 else imp[m] for m in imp})
    votes = pd.DataFrame(sel).astype(int)
    table["votes"] = votes.sum(axis=1)
    table["score"] = table[list(imp)].mean(axis=1)
    n_methods = len(imp)
    table["decision"] = np.where(table["votes"] >= min(3, n_methods), "keep",
                                 np.where(table["votes"] == 2, "borderline", "drop"))
    out["table"] = table.sort_values(["votes", "score"], ascending=False)
    out["votes"] = votes
    out["n_methods"] = n_methods
    out["keep"] = list(out["table"].index[out["table"]["decision"] == "keep"])
    out["borderline"] = list(out["table"].index[out["table"]["decision"] == "borderline"])
    out["drop"] = list(out["table"].index[out["table"]["decision"] == "drop"])
    return out


def _grouped_permutation(model: Any, x: np.ndarray, y: np.ndarray, owners: list[str], features: list[str],
                         scoring: str, seed: int, repeats: int = 5) -> tuple[pd.Series, pd.Series]:
    """Drop in held-out score when all the design columns of one feature are shuffled together."""
    rng = np.random.default_rng(seed)
    base = _score(model, x, y, scoring)
    owner = np.array(owners, dtype=object)
    means, stds = {}, {}
    for f in features:
        cols = np.flatnonzero(owner == f)
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, cols] = x[rng.permutation(len(x))][:, cols]
            drops.append(base - _score(model, shuffled, y, scoring))
        means[f], stds[f] = float(np.mean(drops)), float(np.std(drops))
    return pd.Series(means), pd.Series(stds)


def _held_out(model: Any, x: np.ndarray, y: np.ndarray, scoring: str) -> float:
    """Score on the held-out rows; NaN when it is undefined there (one class only for an AUC)."""
    try:
        return float(_score(model, x, y, scoring))
    except ValueError:
        return float("nan")


def _fmt_time(t: Any) -> str:
    if isinstance(t, pd.Timestamp):
        return str(t.date()) if t == t.normalize() else str(t)
    return f"row {t:,}"


def _score(model: Any, x: np.ndarray, y: np.ndarray, scoring: str) -> float:
    from sklearn.metrics import get_scorer

    return get_scorer(scoring)(model, x, y)


def _lasso_path(x: np.ndarray, y: np.ndarray, owners: list[str], names: Any) -> pd.DataFrame:
    """Coefficients along the lasso path (rows = penalty, columns = the design columns that ever enter)."""
    from sklearn.linear_model import lasso_path

    try:
        alphas, coefs, _ = lasso_path(x, y - y.mean(), n_alphas=60)
    except TypeError:  # scikit-learn >= 1.7
        alphas, coefs, _ = lasso_path(x, y - y.mean(), alphas=60)
    frame = pd.DataFrame(coefs.T, index=alphas, columns=list(names))
    frame = frame.loc[:, (frame.abs() > 1e-10).any()]
    order = frame.abs().max().sort_values(ascending=False).index[:12]
    return frame[order]


__all__ = ["METHODS", "feature_selection"]
