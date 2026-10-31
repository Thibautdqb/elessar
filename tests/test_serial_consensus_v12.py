"""Acceptance boundaries for serial inference and independent consensus assessment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.model_selection import train_test_split

from elessar import StoneConfig
from elessar.correlations import compute_correlations, cramers_v_test
from elessar.selection import _holdout_mask, feature_selection

CFG = StoneConfig(n_jobs=1)


def ar1(rng, n, phi=.9):
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(1 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def markov(rng, n):
    state = np.empty(n, dtype=int)
    state[0] = rng.integers(2)
    for i in range(1, n):
        state[i] = state[i - 1] ^ int(rng.random() < .05)
    return (2 * state + rng.integers(2, size=n)).astype(str)


def serial(n):
    return {"time": pd.Series(pd.date_range("2020-01-01", periods=n)), "entity": None, "differences": None}


def oracle_tts(a, b, statistic, radius):
    """Literal two-orientation implementation of the published B/(r+1) rule."""
    center = slice(radius, len(a) - radius)
    value = statistic(a[center], b[center])
    observed = abs(value)
    counts = []
    for x, y in ((a, b), (b, a)):
        null = [abs(statistic(x[center], y[radius + s:len(y) - radius + s]))
                for s in range(-radius, radius + 1)]
        counts.append(sum(z >= observed - 1e-12 for z in null))
    return value, min(1., max(counts) / (radius + 1))


def test_serial_kendall_uses_the_truncated_shift_reference():
    rng = np.random.default_rng(631001)
    a, b = ar1(rng, 300), ar1(rng, 300)
    out = compute_correlations(pd.DataFrame({"a": a, "b": b}), dict.fromkeys(["a", "b"], "numeric_continuous"),
                               config=CFG, serial=serial(300))
    value, p = oracle_tts(a, b, lambda x, y: stats.kendalltau(x, y).statistic, 75)
    assert out["kendall_p"].loc["a", "b"] == pytest.approx(p)
    assert out["kendall"].loc["a", "b"] == pytest.approx(value)


def test_serial_categorical_uses_the_truncated_shift_reference():
    rng = np.random.default_rng(631002)
    a, b = markov(rng, 600), markov(rng, 600)
    out = compute_correlations(pd.DataFrame({"a": a, "b": b}), dict.fromkeys(["a", "b"], "categorical"),
                               config=CFG, serial=serial(600))
    def metric(x, y):
        return cramers_v_test(pd.Series(x), pd.Series(y))[0]
    value, p = oracle_tts(a, b, metric, 99)
    assert out["cramers_v_p"].loc["a", "b"] == pytest.approx(p)
    assert out["cramers_v"].loc["a", "b"] == pytest.approx(value)
    assert out["cramers_v_n"].loc["a", "b"] == 402


def test_serial_missing_data_does_not_get_an_iid_pvalue_by_fallback():
    rng = np.random.default_rng(631003)
    df = pd.DataFrame({"a": ar1(rng, 120), "b": ar1(rng, 120)})
    df.loc[30, "a"] = np.nan
    out = compute_correlations(df, dict.fromkeys(df.columns, "numeric_continuous"), config=CFG, serial=serial(120))
    assert np.isnan(out["kendall_p"].loc["a", "b"])
    assert "missing" in out["serial_inference"].iloc[0]["detail"]


def selection_frame():
    rng = np.random.default_rng(631004)
    df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["x", "z", "w"])
    df["y"] = df.x + rng.normal(size=len(df))
    return df, dict.fromkeys(df.columns, "numeric_continuous")


def test_consensus_has_a_separate_final_test_set():
    df, types = selection_frame()
    out = feature_selection(df, "y", types, {}, CFG)
    evaluation = out.get("consensus_evaluation", {})
    assert evaluation.get("status") == "ok"
    assert evaluation["n_test"] == 20
    assert evaluation["n_development"] == out["n_train"] + out["n_voting"] == 180
    assert out["n_holdout"] == out["n_voting"] + evaluation["n_test"]
    assert evaluation["features"] == out["keep"]
    assert np.isfinite(evaluation["score"])


def test_final_test_targets_cannot_change_votes_or_learned_importances():
    df, types = selection_frame()
    train = _holdout_mask(len(df), np.arange(len(df)), False, CFG.random_state)
    _, final_rows = train_test_split(np.flatnonzero(~train), test_size=.5, random_state=CFG.random_state + 1)
    altered = df.copy()
    altered.loc[final_rows, "y"] = 100 * np.random.default_rng(631005).normal(size=len(final_rows))
    a, b = (feature_selection(frame, "y", types, {}, CFG) for frame in (df, altered))
    for name in ("table", "votes"):
        pd.testing.assert_frame_equal(a[name], b[name])
    for name in a["importance"]:
        pd.testing.assert_series_equal(a["importance"][name], b["importance"][name])
    assert a["scores"] == b["scores"]
    assert a["consensus_evaluation"]["score"] != b["consensus_evaluation"]["score"]
