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
otherwise. Each model's cross-validated score says how much to trust its ranking. Time-indexed
tables are split in time order, never shuffled.
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


def _design(df: pd.DataFrame, features: list[str], numeric: list[str], types: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Numeric columns (median-imputed, standardised) and one-hot categoricals; owner of every design column."""
    blocks, owners = [], []
    num_cols = [c for c in features if c in numeric]
    if num_cols:
        num = _numeric_frame(df, num_cols, types)
        num = num.fillna(num.median())
        std = num.std(ddof=0).replace(0, 1)
        blocks.append((num - num.mean()) / std)
        owners += num_cols
    for c in [c for c in features if c not in numeric]:
        lab = hashable(df[c]).astype("string").fillna("(missing)")
        dummies = pd.get_dummies(lab, prefix=str(c), prefix_sep="=", dtype="float64")
        blocks.append(dummies)
        owners += [c] * dummies.shape[1]
    design = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
    return design, owners


def _by_feature(values: np.ndarray, owners: list[str], features: list[str]) -> pd.Series:
    return pd.Series(values, index=owners).groupby(level=0).sum().reindex(features).fillna(0.0)


def feature_selection(
    df: pd.DataFrame, target: str, types: dict[str, str], correlations: dict[str, pd.DataFrame],
    cfg: StoneConfig, time_ordered: bool = False,
) -> dict[str, Any] | None:
    """Rank and select the features of ``target`` with every method; None when there is nothing to model."""
    from sklearn.linear_model import LassoCV, LogisticRegressionCV, RidgeCV
    from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    df = df[df[target].notna()]
    if not time_ordered:
        df = sample(df, _MAX_ROWS, cfg.random_state)
    else:
        df = df.iloc[-_MAX_ROWS:]
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
    if regression:
        y = _numeric_frame(df, [target], types)[target].to_numpy(dtype="float64")
        ok = np.isfinite(y)
        x, y = x[ok], y[ok]
        task = "regression"
    else:
        y_lab = hashable(df[target]).astype("string")
        counts = y_lab.value_counts()
        keep = y_lab.isin(counts[counts >= 5].index)
        x, y = x[keep.to_numpy()], y_lab[keep].to_numpy()
        if len(np.unique(y)) < 2:
            return None
        task = "binary classification" if len(np.unique(y)) == 2 else f"{len(np.unique(y))}-class classification"
    xv = x.to_numpy(dtype="float64")
    n = len(xv)
    if n < 30:
        return None
    if time_ordered:
        cv: Any = TimeSeriesSplit(n_splits=5)
    elif regression:
        cv = KFold(5, shuffle=True, random_state=cfg.random_state)
    else:
        cv = StratifiedKFold(5, shuffle=True, random_state=cfg.random_state)
    scoring = "r2" if regression else ("roc_auc" if task.startswith("binary") else "balanced_accuracy")
    rs = cfg.random_state
    out: dict[str, Any] = {"task": task, "n_rows": n, "n_features": len(features), "n_design_columns": x.shape[1],
                           "excluded_leaks": sorted(map(str, leaks)),
                           "scoring": scoring, "cv": "time-ordered 5-fold" if time_ordered else "5-fold",
                           "scores": {}, "importance": {}, "selected": {}}
    imp, sel, scores = out["importance"], out["selected"], out["scores"]
    with quiet(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # decision tree: depth by cross-validation
        tree_cls = DecisionTreeRegressor if regression else DecisionTreeClassifier
        leaf = max(5, n // 200)
        depth_scores = {d: float(np.mean(cross_val_score(tree_cls(max_depth=d, min_samples_leaf=leaf, random_state=rs),
                                                         xv, y, cv=cv, scoring=scoring))) for d in (2, 3, 4, 6, 8)}
        depth = max(depth_scores, key=depth_scores.get)
        tree = tree_cls(max_depth=depth, min_samples_leaf=leaf, random_state=rs).fit(xv, y)
        scores["decision tree"] = depth_scores[depth]
        out["tree_depth"] = depth
        imp["decision tree"] = _by_feature(tree.feature_importances_, owners, features)
        sel["decision tree"] = imp["decision tree"] >= 0.5 / len(features)  # half an equal share

        # random forest: permutation importance on held-out rows
        forest_cls = RandomForestRegressor if regression else RandomForestClassifier
        forest = forest_cls(n_estimators=100, min_samples_leaf=3, max_features="sqrt" if not regression else 0.5,
                            random_state=rs, n_jobs=1)
        split = dict(test_size=0.25, random_state=rs, shuffle=not time_ordered)
        if not regression and not time_ordered:
            split["stratify"] = y
        x_tr, x_te, y_tr, y_te = train_test_split(xv, y, **split)
        forest.fit(x_tr[:10_000], y_tr[:10_000])
        scores["random forest"] = float(_score(forest, x_te, y_te, scoring))
        pm, ps = _grouped_permutation(forest, x_te[:2_000], y_te[:2_000], owners, features, scoring, rs)
        imp["random forest"] = pm.clip(lower=0)
        out["forest_std"] = ps
        sel["random forest"] = (pm > 2 * ps) & (pm > 0.002)

        # lasso / L1 logistic
        if regression:
            grid = {"n_alphas": 40} if "n_alphas" in LassoCV().get_params() else {"alphas": 40}  # renamed in 1.7
            lasso = LassoCV(cv=cv, random_state=rs, max_iter=5_000, **grid).fit(xv, y)
            coef = np.abs(lasso.coef_)
            out["lasso_alpha"] = float(lasso.alpha_)
            scores["lasso"] = float(np.mean(cross_val_score(lasso.__class__(alphas=[lasso.alpha_], max_iter=5_000),
                                                            xv, y, cv=cv, scoring=scoring)))
            out["lasso_path"] = _lasso_path(xv, y, owners, x.columns)
        else:
            binary = task.startswith("binary")
            m = min(n, 5_000 if binary else 3_000)  # logistic paths are the slow part: fit on a subset, 3 folds
            small_cv = TimeSeriesSplit(n_splits=3) if time_ordered else StratifiedKFold(3, shuffle=True, random_state=rs)
            l1 = LogisticRegressionCV(Cs=6, cv=small_cv, penalty="l1", solver="liblinear" if binary else "saga",
                                      scoring=scoring, max_iter=500, tol=1e-3, random_state=rs).fit(xv[:m], y[:m])
            coef = np.abs(np.atleast_2d(l1.coef_)).sum(axis=0)
            out["lasso_alpha"] = float(1 / np.atleast_1d(l1.C_)[0])
            scores["lasso"] = float(np.max(np.mean(next(iter(l1.scores_.values())), axis=0)))
        imp["lasso"] = _by_feature(coef, owners, features)
        sel["lasso"] = imp["lasso"] > 1e-8

        # ridge / L2 logistic
        if regression:
            ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(xv, y)
            rcoef = np.abs(ridge.coef_)
            scores["ridge"] = float(np.mean(cross_val_score(RidgeCV(alphas=[ridge.alpha_]), xv, y, cv=cv, scoring=scoring)))
            out["ridge_alpha"] = float(ridge.alpha_)
        else:
            l2 = LogisticRegressionCV(Cs=6, cv=small_cv, penalty="l2", scoring=scoring, max_iter=500,
                                      tol=1e-3).fit(xv[:m], y[:m])
            rcoef = np.abs(np.atleast_2d(l2.coef_)).sum(axis=0)
            scores["ridge"] = float(np.max(np.mean(next(iter(l2.scores_.values())), axis=0)))
        imp["ridge"] = _by_feature(rcoef, owners, features)
        sel["ridge"] = imp["ridge"] >= 0.1 * imp["ridge"].max() if imp["ridge"].max() > 0 else imp["ridge"] > 0

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
