"""Scientific and partition boundaries beyond the original reproductions."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import train_test_split

from elessar import StoneConfig, see
from elessar.correlations import compute_correlations, target_associations
from elessar.selection import _evaluate_consensus, _holdout_mask, feature_selection

CFG = StoneConfig(n_jobs=1)


def series_frame(n=120, categorical=False):
    rng = np.random.default_rng(636001)
    df = pd.DataFrame({"a": rng.choice(list("abcd"), n) if categorical else rng.normal(size=n),
                       "b": rng.choice(list("wxyz"), n) if categorical else rng.normal(size=n)})
    types = dict.fromkeys(df.columns, "categorical" if categorical else "numeric_continuous")
    serial = {"time": pd.Series(pd.date_range("2020-01-01", periods=n)), "entity": None, "differences": None}
    return df, types, serial


@pytest.mark.parametrize("categorical", [False, True])
def test_tts_is_invariant_to_file_order_column_order_and_label_names(categorical):
    df, types, serial = series_frame(categorical=categorical)
    key = "cramers_v_p" if categorical else "kendall_p"
    original = compute_correlations(df, types, config=CFG, serial=serial)
    changed = df.sample(frac=1, random_state=73)[["b", "a"]]
    if categorical:
        changed = changed.replace({"a": "A9", "b": "A0", "c": "A7", "d": "A2",
                                   "w": "B9", "x": "B0", "y": "B7", "z": "B2"})
    other = compute_correlations(changed, types, config=CFG, serial=serial)
    assert original[key].loc["a", "b"] == other[key].loc["a", "b"]


@pytest.mark.parametrize("kind", ["irregular", "missing", "unbalanced", "duplicate"])
def test_unsupported_temporal_grids_abstain_and_do_not_mark_significance(kind):
    df, types, serial = series_frame(categorical=True)
    if kind == "irregular":
        serial["time"].iloc[30:] += pd.Timedelta(days=1)
    elif kind == "missing":
        df.loc[30, "a"] = None
    elif kind == "duplicate":
        serial["time"].iloc[30] = serial["time"].iloc[29]
    else:
        df = pd.concat([df, df], ignore_index=True).iloc[:-1]
        serial = {"time": pd.concat([serial["time"], serial["time"]], ignore_index=True).iloc[:-1],
                  "entity": pd.Series(["e0"] * 120 + ["e1"] * 119), "differences": None}
    out = compute_correlations(df, types, target="b", config=CFG, serial=serial)
    assert np.isnan(out["cramers_v_p"].loc["a", "b"])
    assert np.isnan(out["target"].iloc[0]["p_value"])
    assert not out["pairs"]["significant"].any() and not out["target"]["significant"].any()
    assert (out["serial_inference"]["status"] == "unavailable").all()


def test_balanced_panel_shifts_are_synchronous_and_preserve_row_order_invariance():
    df, types, serial = series_frame()
    panel = pd.DataFrame({"a": np.repeat(df.a, 2), "b": np.repeat(df.b, 2)}).reset_index(drop=True)
    serial = {"time": pd.Series(np.repeat(serial["time"], 2)).reset_index(drop=True),
              "entity": pd.Series(np.tile(["e0", "e1"], len(df))), "differences": None}
    first = compute_correlations(panel, types, config=CFG, serial=serial)
    shuffled = compute_correlations(panel.sample(frac=1, random_state=74), types, config=CFG, serial=serial)
    assert first["kendall_p"].loc["a", "b"] == shuffled["kendall_p"].loc["a", "b"]
    record = first["serial_inference"].iloc[0]
    assert record["n_entities"] == 2 and record["n_time"] == 120 and record["n"] == 120


def test_sampling_uses_a_fixed_contiguous_window_for_tts():
    df, types, serial = series_frame()
    out = compute_correlations(df, types, config=StoneConfig(n_jobs=1, sample_size=80), serial=serial)
    row = out["serial_inference"].iloc[0]
    assert row["n_time"] == 80 and row["radius"] == 20 and row["n"] == 40
    expected = stats.kendalltau(df.a.iloc[-60:-20], df.b.iloc[-60:-20]).statistic
    assert out["kendall"].loc["a", "b"] == pytest.approx(expected)


@pytest.mark.parametrize("categorical", [False, True])
def test_undefined_central_effect_provides_no_evidence(categorical):
    df, types, serial = series_frame(categorical=categorical)
    df.loc[30:89, "a"] = "constant" if categorical else 0.
    out = compute_correlations(df, types, config=CFG, serial=serial)
    key = "cramers_v" if categorical else "kendall"
    assert np.isnan(out[key].loc["a", "b"])
    assert out[key + "_p"].loc["a", "b"] == 1.


def test_tts_never_adapts_its_scale_from_the_stationarity_pretest():
    df, types, serial = series_frame()
    first = compute_correlations(df, types, config=CFG, serial=serial)
    serial["differences"] = df.diff()
    second = compute_correlations(df, types, config=CFG, serial=serial)
    pd.testing.assert_frame_equal(first["kendall"], second["kendall"])
    pd.testing.assert_frame_equal(first["kendall_p"], second["kendall_p"])


def test_single_lag_summary_without_a_time_grid_does_not_create_a_categorical_pvalue():
    df, types, _ = series_frame(categorical=True)
    out = target_associations(df, "b", types, CFG, rho={"a": .9, "b": .9})
    assert np.isnan(out.iloc[0]["p_value"]) and not out.iloc[0]["significant"]
    assert "unavailable" in out.iloc[0]["test"]


def test_tts_categorical_target_and_pair_test_use_the_same_window_and_pvalue():
    df, types, serial = series_frame(categorical=True)
    out = compute_correlations(df, types, target="b", config=CFG, serial=serial)
    assert out["target"].iloc[0]["p_value"] == out["cramers_v_p"].loc["a", "b"]
    assert out["target"].iloc[0]["association"] == out["cramers_v"].loc["a", "b"]
    assert out["target"].iloc[0]["n"] == out["cramers_v_n"].loc["a", "b"]


def test_final_model_never_fits_on_test_outcomes():
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(636002)
    df = pd.DataFrame({"x": rng.normal(size=180), "y": rng.normal(size=180)})
    mask = _holdout_mask(len(df), np.arange(len(df)), False, CFG.random_state)
    _, final_rows = train_test_split(np.flatnonzero(~mask), test_size=.5, random_state=CFG.random_state + 1)
    forbidden = set(df.y.iloc[final_rows])
    fit = RandomForestRegressor.fit
    sizes = []

    def checked(model, x, y, **kwargs):
        assert not forbidden.intersection(y)
        sizes.append(len(y))
        return fit(model, x, y, **kwargs)

    with patch.object(RandomForestRegressor, "fit", checked):
        out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, CFG)
    assert out["consensus_evaluation"]["status"] == "ok"
    assert sizes and max(sizes) <= len(df) - len(final_rows)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_empty_consensus_has_an_explicit_training_prior_baseline(task):
    rng = np.random.default_rng(636003)
    df = pd.DataFrame({"x": rng.normal(size=100),
                       "y": rng.integers(2, size=100) if task == "binary" else rng.normal(size=100)})
    mask = _holdout_mask(len(df), np.arange(len(df)), False, CFG.random_state)
    selection = {"keep": [], "task": "regression" if task == "regression" else "binary classification",
                 "scoring": "r2" if task == "regression" else "roc_auc", "on_differences": []}
    out = _evaluate_consensus(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), mask, selection, CFG,
                              pd.Series(np.arange(len(df))))
    assert out["status"] == "ok" and out["score"] == out["baseline_score"]
    assert out["features"] == [] and "no feature kept" in out["model"]
    if task == "binary":
        assert out["score"] == .5


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_final_consensus_classification_scores_are_finite(task):
    rng = np.random.default_rng(636004)
    df = pd.DataFrame(rng.normal(size=(180, 2)), columns=["x", "z"])
    df["y"] = (df.x > 0).astype(int) if task == "binary" else pd.cut(df.x, [-np.inf, -.5, .5, np.inf], labels=list("abc"))
    types = {"x": "numeric_continuous", "z": "numeric_continuous", "y": "boolean" if task == "binary" else "categorical"}
    out = feature_selection(df, "y", types, {}, CFG)
    assert out["consensus_evaluation"]["status"] == "ok"
    assert np.isfinite(out["consensus_evaluation"]["score"])


def test_small_tables_retain_selection_with_an_explicit_unavailable_final_evaluation():
    rng = np.random.default_rng(636005)
    df = pd.DataFrame(rng.normal(size=(36, 2)), columns=["x", "y"])
    out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, CFG)
    assert out is not None and out["consensus_evaluation"]["status"] == "unavailable"
    assert out["consensus_evaluation"]["n_test"] == 0


def test_dated_final_consensus_is_chronological_and_invariant_to_file_order():
    rng = np.random.default_rng(636006)
    df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=180), "x": rng.normal(size=180), "y": rng.normal(size=180)})
    types = {"day": "datetime", "x": "numeric_continuous", "y": "numeric_continuous"}
    first = feature_selection(df, "y", types, {}, CFG, times=df.day)
    second = feature_selection(df.sample(frac=1, random_state=75), "y", types, {}, CFG, times=df.day)
    assert first["consensus_evaluation"]["status"] == "ok"
    assert first["consensus_evaluation"]["score"] == second["consensus_evaluation"]["score"]
    assert first["consensus_evaluation"]["test_from"] == str(df.day.iloc[162])
    pd.testing.assert_frame_equal(first["votes"], second["votes"])


def test_full_report_learns_target_type_without_final_test_values(tmp_path):
    rng = np.random.default_rng(636007)
    df = pd.DataFrame({"x": rng.normal(size=180), "z": rng.normal(size=180)})
    df["y"] = (df.x + .5 * rng.normal(size=len(df)) > 0).astype(int)
    train = _holdout_mask(len(df), np.arange(len(df)), False, CFG.random_state)
    _, final_rows = train_test_split(np.flatnonzero(~train), test_size=.5, random_state=CFG.random_state + 1)
    changed = df.copy()
    changed.loc[final_rows, "y"] = np.arange(1000, 1000 + len(final_rows))
    outputs = [see(frame, target="y", config=StoneConfig(n_jobs=1, candidates_continuous=("norm",)),
                   plots=False, verbose=False, out_dir=tmp_path / str(i), open_report=False).selection
               for i, frame in enumerate((df, changed))]
    a, b = outputs
    assert a["task"] == b["task"] == "binary classification"
    pd.testing.assert_series_equal(a["importance"]["filter"], b["importance"]["filter"])
    pd.testing.assert_frame_equal(a["table"], b["table"])
    assert a["scores"] == b["scores"]
    assert b["consensus_evaluation"]["status"] == "unavailable"  # a test-only class cannot be scored as binary AUC
