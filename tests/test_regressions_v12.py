"""Regression checks reproduced as failures on audited 1.1.0 before the v1.2 fixes.

These tests express the required invariants. Run with:
python -m pytest -q tests/test_regressions_v12.py
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import train_test_split

from elessar import StoneConfig, see
from elessar.correlations import compute_correlations, target_associations
from elessar.outliers import column_outliers
from elessar.distributions import _fit_discrete
from elessar.preparation import plan
from elessar.selection import feature_selection
from elessar.timeseries import adf_test, treatment

CFG = StoneConfig(n_jobs=1, candidates_continuous=("norm",))


class Captured(Exception):
    """Stop immediately after observing the production boundary being tested."""


def selection_frame(n=150):
    rng = np.random.default_rng(314)
    df = pd.DataFrame({"x": np.arange(n, dtype=float), "y": rng.normal(size=n)})
    types = dict.fromkeys(df.columns, "numeric_continuous")
    return df, types


def test_p0_cv_preprocessing_is_fitted_inside_each_fold():
    df, types = selection_frame()
    dates = pd.Series(pd.date_range("2020-01-01", periods=len(df)))
    observed = {}

    def capture_tree_fit(model, x, y, **kwargs):
        observed["first_fold_train_mean"] = float(x[:, 0].mean())
        observed["first_fold_train_std"] = float(x[:, 0].std())
        observed["n_train"] = len(x)
        raise Captured

    def capture(model, x, y, *, cv, scoring):
        splits = cv.split(x, y) if hasattr(cv, "split") else cv
        tr, va = next(iter(splits))
        # Fit the estimator passed by production: if it becomes a Pipeline, its
        # preprocessor will run inside this fold before the tree boundary.
        with patch("sklearn.tree.DecisionTreeRegressor.fit", capture_tree_fit):
            model.fit(x[tr], y[tr])

    with patch("sklearn.model_selection.cross_val_score", capture), pytest.raises(Captured):
        feature_selection(df, "y", types, {}, CFG, times=dates)
    print(observed)
    # Each nonconstant training fold must be centred using its own observations.
    assert abs(observed["first_fold_train_mean"]) < 1e-12
    assert abs(observed["first_fold_train_std"] - 1) < 1e-12


def test_p0_small_panel_holdout_keeps_whole_dates():
    df, types = selection_frame(36)
    dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=12), 3))
    observed = {}

    def capture(x, train, n_numeric):
        observed["last_train"] = dates[train].max()
        observed["first_test"] = dates[~train].min()
        observed["overlap"] = sorted(set(dates[train]) & set(dates[~train]))
        raise Captured

    with patch("elessar.selection._standardise", capture), pytest.raises(Captured):
        feature_selection(df, "y", types, {}, CFG, times=dates)
    print(observed)
    assert observed["last_train"] < observed["first_test"]


def test_p0_inner_cv_with_few_dates_keeps_whole_dates():
    df, types = selection_frame(100)
    counts = [25, 25, 25, 5, 3, 3, 3, 3, 2, 2, 2, 2]
    dates = pd.Series(np.repeat(pd.date_range("2020-01-01", periods=12), counts))
    observed = {}

    def capture(model, x, y, *, cv, scoring):
        splits = cv.split(x, y) if hasattr(cv, "split") else cv
        tr, va = next(iter(splits))
        train_dates = dates.iloc[:len(x)]
        observed["last_train"] = train_dates.iloc[tr].max()
        observed["first_validation"] = train_dates.iloc[va].min()
        raise Captured

    with patch("sklearn.model_selection.cross_val_score", capture), pytest.raises(Captured):
        feature_selection(df, "y", types, {}, CFG, times=dates)
    print(observed)
    assert observed["last_train"] < observed["first_validation"]


def test_p0_holdout_targets_do_not_change_training_feature_design():
    rng = np.random.default_rng(8)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = df["x"]
    train, test = train_test_split(np.arange(n), test_size=.2, random_state=CFG.random_state)
    altered = df.copy()
    altered.loc[test, "y"] = 10 * rng.normal(size=len(test))
    pd.testing.assert_frame_equal(df.iloc[train], altered.iloc[train])
    types = dict.fromkeys(df.columns, "numeric_continuous")
    designs = []

    def capture(frame, features, numeric, semantic_types):
        designs.append(list(features))
        raise Captured

    for frame in (df, altered):
        corr = {"target": target_associations(frame, "y", types, CFG)}
        with patch("elessar.selection._design", capture), pytest.raises(Captured):
            feature_selection(frame, "y", types, corr, CFG)
    print({"original_training_features": designs[0], "altered_holdout_training_features": designs[1]})
    assert designs[0] == designs[1]


def boolean_report(values, tmp_path):
    df = pd.DataFrame({"flag": values, "x": np.random.default_rng(4).normal(size=len(values))})
    vision = see(df, plots=False, verbose=False, config=CFG, out_dir=tmp_path, open_report=False)
    return df, vision


def run_column_steps(df, vision, column):
    import elessar
    ns = {"df": df.copy(), "np": np, "pd": pd, "elessar": elessar}
    for step in vision.preparation["steps"]:
        if step["column"] == column:
            exec(step["code"], ns)
    return ns["df"]


def test_p1_nullable_boolean_preparation_runs(tmp_path):
    df, vision = boolean_report(pd.Series([True, False, pd.NA] * 40, dtype="boolean"), tmp_path)
    out = run_column_steps(df, vision, "flag")
    assert len(out) == len(df)


def test_p1_unknown_boolean_is_not_silently_true(tmp_path):
    df, vision = boolean_report(pd.Series(["yes", "no", None] * 40), tmp_path)
    out = run_column_steps(df, vision, "flag")
    print({"encoded_counts": out["flag"].value_counts(dropna=False).to_dict()})
    missing_preserved = pd.isna(out.loc[2, "flag"]) or out.loc[2, "flag"] != out.loc[0, "flag"]
    missing_indicated = "flag_missing" in out and bool(out.loc[2, "flag_missing"])
    assert missing_preserved or missing_indicated


@pytest.mark.parametrize("operation", ["unit root", "inconclusive", "seasonal"])
def test_p1_panel_differences_do_not_cross_entity_boundaries(operation):
    n = 40
    df = pd.DataFrame({"day": np.tile(pd.date_range("2020-01-01", periods=n), 2),
                       "shop": np.repeat(["a", "b"], n), "value": np.r_[np.arange(n), 1000 + np.arange(n)]})
    info = {"kind": "panel", "entity": "shop", "time_column": "day", "frequency": "day", "sorted": True}
    main = {"stationarity": {"verdict": operation}, "period": 7 if operation == "seasonal" else None,
            "seasonal_strength": .5, "seasonality": "moderate"}
    ns = {"df": df.copy(), "np": np, "pd": pd}
    for step in treatment(info, main, n, value="value"):
        if step["step"] in ("Panel", "Stationarity", "Seasonality"):
            exec(step["code"], ns)
    col = "value_sdiff" if operation == "seasonal" else "value_diff"
    k = 7 if operation == "seasonal" else 1
    heads = ns["df"].groupby("shop").head(k)
    print({"operation": operation, "entity_b_first_values": heads.loc[heads.shop == "b", col].tolist()})
    assert heads[col].isna().all()


def test_p1_outlier_removal_uses_the_same_scale_as_the_verdict():
    rng = np.random.default_rng(51)
    x = pd.Series(np.exp(rng.normal(0, .5, 1000)), name="x")
    x.iloc[0] = 1e-12
    result = column_outliers(x, CFG)
    assert result["action"] == "remove" and result["n_extreme"] == 1
    details = {"x": {"semantic_type": "numeric_continuous", "stats": {}}}
    prep = plan(x.to_frame(), details, {}, {}, None, CFG, outliers={"columns": {"x": result}})
    step = next(s for s in prep["steps"] if s["action"] == "Remove outliers")
    ns = {"df": x.to_frame(), "np": np, "pd": pd}
    exec(step["code"], ns)
    print({"action": result["action"], "n_extreme": result["n_extreme"],
           "tiny_extreme_retained": 0 in ns["df"].index, "removed_rows": len(x) - len(ns["df"])})
    assert 0 not in ns["df"].index


def test_p1_multiclass_target_encoding_snippet_runs():
    n = 450
    rng = np.random.default_rng(71)
    df = pd.DataFrame({"category": [f"c{i % 15}" for i in range(n)], "target": rng.choice(["a", "b", "c"], n)})
    details = {"category": {"semantic_type": "categorical", "stats": {"cardinality": 15}},
               "target": {"semantic_type": "categorical", "stats": {"cardinality": 3}}}
    prep = plan(df, details, {}, {}, "target", CFG)
    step = next(s for s in prep["steps"] if s["action"] == "Encode" and s["column"] == "category")
    ns = {"df": df.copy(), "np": np, "pd": pd}
    exec(step["code"], ns)
    assert len(ns["df"]) == n


def test_p1_association_matrix_pvalue_matches_its_spearman_effect():
    rng = np.random.default_rng(19)
    x = rng.normal(size=120)
    df = pd.DataFrame({"x": x, "y": .3 * x + rng.normal(size=len(x))})
    types = dict.fromkeys(df.columns, "numeric_continuous")
    out = compute_correlations(df, types, config=CFG)
    r, p = stats.spearmanr(df.x, df.y)
    assert out["association"].loc["x", "y"] == pytest.approx(r)
    print({"pair_method": out["pairs"].iloc[0]["method"],
           "matrix_effect": r, "matrix_p": out["association_p"].loc["x", "y"], "spearman_p": p})
    assert out["association_p"].loc["x", "y"] == pytest.approx(p, abs=1e-12)


def test_p1_adf_finite_sample_critical_values_use_regression_observation_count():
    from statsmodels.tsa.adfvalues import mackinnoncrit
    y = np.random.default_rng(3).normal(size=40)
    maxlag = 8
    out = adf_test(y, max_lag=maxlag)
    # Every candidate in Elessar is fitted on n-1-maxlag observations.
    nobs = len(y) - 1 - maxlag
    expected = mackinnoncrit(N=1, regression="c", nobs=nobs)[1]
    print({"original_n": len(y), "regression_n": nobs, "critical_5pct": out["critical"]["5%"],
           "mackinnon_5pct": expected})
    assert out["critical"]["5%"] == pytest.approx(expected, abs=2e-5)


def test_p1_binomial_candidate_really_maximizes_likelihood():
    x = np.random.default_rng(1234).binomial(20, .2, 2000)
    params, _, note = _fit_discrete("binom", x)
    claimed_ll = float(stats.binom.logpmf(x, *params).sum())
    # At each integer n, the exact conditional MLE is p = sample_mean / n.
    # Finding even one admissible n with larger likelihood disproves maximality.
    candidates = range(int(x.max()), 101)
    ll = {n: float(stats.binom.logpmf(x, n, x.mean() / n).sum()) for n in candidates}
    best_n = max(ll, key=ll.get)
    print({"reported_n": params[0], "sample_max": int(x.max()), "reference_n_in_0_100": best_n,
           "log_likelihood_gain": ll[best_n] - claimed_ll, "fit_note": note})
    assert claimed_ll >= max(ll.values()) - 1e-8


def test_p1_supervised_binning_snippet_defines_its_infinite_bounds():
    x = np.arange(200, dtype=float)
    df = pd.DataFrame({"x": x, "y": (x > 100).astype(int)})
    details = {"x": {"semantic_type": "numeric_continuous", "stats": {"heaping_index": 1}},
               "y": {"semantic_type": "boolean", "stats": {}}}
    prep = plan(df, details, {}, {}, "y", CFG)
    step = next(s for s in prep["steps"] if s["action"] == "Bucketing")
    print({"generated_code": step["code"]})
    ns = {"df": df.copy(), "np": np, "pd": pd}
    exec(step["code"], ns)
    assert ns["df"]["x_bin"].notna().all()
