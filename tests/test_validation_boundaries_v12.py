"""Additional acceptance checks for the corrected training and generated-code boundaries."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from statsmodels.tsa.adfvalues import mackinnoncrit
from statsmodels.tsa.stattools import adfuller

from elessar import StoneConfig, see_corr
from elessar.distributions import BINOM_MAX_TRIALS, _fit_discrete
from elessar.outliers import column_outliers
from elessar.preparation import plan
from elessar.quality import correlation_alerts
from elessar.selection import _cv_pipeline, _holdout_mask, _time_folds, feature_selection
from elessar.timeseries import adf_test

CFG = StoneConfig(n_jobs=1)


@pytest.mark.parametrize("classification", [False, True])
def test_holdout_targets_leave_fitted_training_choices_unchanged(classification):
    rng = np.random.default_rng(412)
    n = 180
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["target"] = (df.x + .7 * rng.normal(size=n) > 0).astype(int) if classification else df.x + rng.normal(size=n)
    types = dict.fromkeys(df.columns, "numeric_continuous")
    if classification:
        types["target"] = "boolean"
    train, holdout = train_test_split(np.arange(n), test_size=.2, random_state=CFG.random_state)
    altered = df.copy()
    altered.loc[holdout, "target"] = 1 - df.loc[holdout, "target"] if classification else 50 * rng.normal(size=len(holdout))
    pd.testing.assert_frame_equal(df.iloc[train], altered.iloc[train])
    a = feature_selection(df, "target", types, {}, CFG)
    b = feature_selection(altered, "target", types, {}, CFG)
    for key in ("task", "training_features", "excluded_leaks", "tree_depth", "lasso_alpha", "n_design_columns"):
        assert a[key] == b[key], key
    for key in ("decision tree", "lasso", "ridge"):
        pd.testing.assert_series_equal(a["importance"][key], b["importance"][key])
        pd.testing.assert_series_equal(a["selected"][key], b["selected"][key])
    assert "before consensus" in a["score_scope"]


def test_unseen_holdout_categories_do_not_expand_the_training_design():
    rng = np.random.default_rng(413)
    n = 160
    df = pd.DataFrame({"x": rng.normal(size=n), "category": rng.choice(["a", "b"], n), "y": rng.normal(size=n)})
    types = {"x": "numeric_continuous", "category": "categorical", "y": "numeric_continuous"}
    _, holdout = train_test_split(np.arange(n), test_size=.2, random_state=CFG.random_state)
    a = feature_selection(df, "y", types, {}, CFG)
    df.loc[holdout, "category"] = [f"new_level_{i}" for i in range(len(holdout))]
    b = feature_selection(df, "y", types, {}, CFG)
    assert a["n_design_columns"] == b["n_design_columns"] == 3
    assert a["tree_depth"] == b["tree_depth"] and a["lasso_alpha"] == b["lasso_alpha"]
    for key in ("decision tree", "lasso", "ridge"):
        pd.testing.assert_series_equal(a["importance"][key], b["importance"][key])


def test_fold_imputation_has_no_access_to_future_observations():
    x = np.c_[np.r_[np.full(30, np.nan), np.arange(30, dtype=float)], np.arange(60, dtype=float)]
    y = np.random.default_rng(414).normal(size=60)
    pipeline = _cv_pipeline(Ridge(), 2).fit(x[:30], y[:30])
    transformed = pipeline.named_steps["preprocess"].transform(x[:30])
    assert np.all(transformed[:, 0] == 0)  # all missing in train, not filled from future values
    np.testing.assert_allclose(transformed[:, 1].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(transformed[:, 1].std(), 1, atol=1e-12)


def test_unavailable_time_split_never_falls_back_to_rows():
    codes = np.zeros(100, dtype=int)
    assert _holdout_mask(100, codes, True, 42) is None
    assert _time_folds(codes, 5) is None


@pytest.mark.parametrize("trend", [False, True])
@pytest.mark.parametrize("n,maxlag", [(40, 0), (40, 8), (80, 2), (200, 12)])
def test_adf_critical_surface_matches_reference(n, maxlag, trend):
    rng = np.random.default_rng(415 + n)
    y = rng.normal(size=n) + (.01 * np.arange(n) if trend else 0)
    actual = adf_test(y, max_lag=maxlag, trend=trend)
    expected = mackinnoncrit(N=1, regression="ct" if trend else "c", nobs=actual["nobs"])
    np.testing.assert_allclose(list(actual["critical"].values()), expected, rtol=0, atol=1e-12)
    if maxlag == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = adfuller(y, maxlag=0, autolag=None, regression="ct" if trend else "c")
        assert actual["statistic"] == pytest.approx(reference[0], abs=1e-10)


@pytest.mark.parametrize("n,p,seed", [(10, .7, 1), (20, .2, 2), (50, .4, 3)])
def test_binomial_profile_matches_an_independent_small_grid(n, p, seed):
    x = np.random.default_rng(seed).binomial(n, p, 500)
    params, k, note = _fit_discrete("binom", x)
    assert k == 2 and "profile MLE" in note
    assert params[0] >= x.max() and params[0] < BINOM_MAX_TRIALS
    assert params[0] * params[1] == pytest.approx(x.mean())
    obtained = stats.binom.logpmf(x, *params).sum()
    oracle = max(stats.binom.logpmf(x, nn, x.mean() / nn).sum() for nn in range(int(x.max()), 101))
    assert obtained >= oracle - 1e-8


def test_binomial_search_limit_is_reported_instead_of_claiming_an_unbounded_mle():
    x = np.r_[np.zeros(90), np.full(10, 20)]
    with pytest.raises(ValueError, match="search limit"):
        _fit_discrete("binom", x)


def test_outlier_removal_preserves_missing_rows_and_duplicate_labels():
    x = pd.Series(np.r_[np.random.default_rng(416).normal(size=100), 1000, np.nan], index=[0] * 102)
    result = column_outliers(x)
    assert result["action"] == "remove"
    assert np.flatnonzero(result["remove_mask"]).tolist() == [100]
    assert not result["remove_mask"][-1]
    details = {"x": {"semantic_type": "numeric_continuous", "stats": {}}}
    prep = plan(x.to_frame("x"), details, {}, {}, None, CFG, outliers={"columns": {"x": result}})
    code = next(s["code"] for s in prep["steps"] if s["action"] == "Remove outliers")
    ns = {"df": x.to_frame("x"), "np": np, "pd": pd}
    exec(code, ns)
    assert len(ns["df"]) == 101 and ns["df"].x.isna().sum() == 1


@pytest.mark.parametrize("task", ["binary", "multiclass", "integer regression"])
def test_target_encoding_preserves_rows_and_is_reproducible(task):
    rng = np.random.default_rng(417)
    n = 180
    y = rng.integers(0, 2 if task == "binary" else 3, n) if task != "integer regression" else np.arange(n)
    df = pd.DataFrame({"category": rng.choice([f"c{i}" for i in range(15)], n), "target": y}, index=np.arange(n) // 2)
    df.iloc[-3:, df.columns.get_loc("target")] = np.nan
    details = {"category": {"semantic_type": "categorical", "stats": {"cardinality": 15}},
               "target": {"semantic_type": "numeric_discrete" if task == "integer regression" else "categorical", "stats": {}}}
    prep = plan(df, details, {}, {}, "target", CFG)
    code = next(s["code"] for s in prep["steps"] if s["action"] == "Encode")
    outputs = []
    for _ in range(2):
        ns = {"df": df.copy(), "np": np, "pd": pd}
        exec(code, ns)
        outputs.append(ns["df"])
    pd.testing.assert_frame_equal(outputs[0], outputs[1])
    assert len(outputs[0]) == n and outputs[0].index.equals(df.index)
    encoded = outputs[0].drop(columns="target")
    assert encoded.shape[1] == (3 if task == "multiclass" else 1)
    assert np.isfinite(encoded.to_numpy()).all()


@pytest.mark.parametrize("categorical", [False, True])
def test_serial_inference_limits_are_explicit_and_raise_a_report_alert(categorical):
    rng = np.random.default_rng(418)
    n = 120
    df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=n),
                       "x": rng.choice(list("abcd"), n) if categorical else rng.normal(size=n),
                       "y": rng.choice(list("wxyz"), n) if categorical else rng.normal(size=n)})
    out = see_corr(df, config=CFG)
    limits = out.get("inference_limits", pd.DataFrame(columns=["output", "status"]))
    expected = "cramers_v_p" if categorical else "kendall_p"
    assert expected in set(limits["output"])
    assert "unvalidated" in limits.set_index("output").loc[expected, "status"]
    alerts = correlation_alerts(out)
    assert any(a.get("kind") == "serial_inference_limit" and expected in a["message"] for a in alerts)
