"""Independent final scores of the whole selector, with fixed simulation seeds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig
from elessar.selection import feature_selection

pytestmark = pytest.mark.validation
CFG = StoneConfig(n_jobs=1)


@pytest.mark.parametrize("task", ["regression", "binary"])
def test_final_consensus_score_is_not_optimistic_on_100_noise_tables(task, record):
    values, baselines, kept = [], [], []
    runs = 100
    for r in range(runs):
        rng = np.random.default_rng(634000 + r)
        df = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"x{i}" for i in range(6)])
        df["y"] = rng.integers(2, size=len(df)) if task == "binary" else rng.normal(size=len(df))
        types = dict.fromkeys(df.columns, "numeric_continuous")
        if task == "binary":
            types["y"] = "boolean"
        out = feature_selection(df, "y", types, {}, CFG)
        final = out["consensus_evaluation"]
        assert final["status"] == "ok" and final["n_test"] == 30
        assert final["features"] == out["keep"]
        values.append(final["score"])
        baselines.append(final["baseline_score"])
        kept.append(len(final["features"]))
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(runs))
    chance = .5 if task == "binary" else 0.
    record(task=task, runs=runs, first_seed=634000, truth="features independent of outcomes", chance=chance,
           mean_final_score=mean, standard_error=se, mean_baseline_score=float(np.mean(baselines)),
           mean_features_kept=float(np.mean(kept)), min_final_score=float(min(values)), max_final_score=float(max(values)))
    assert np.isfinite(values).all()
    assert mean <= chance + 3 * se


def test_final_consensus_recovers_a_predeclared_signal(record):
    # Population correlation 3/sqrt(10) ~= .949, below the existing .98
    # leakage-suspect exclusion. A nearly deterministic copy is a separate boundary case below.
    scores, retained = [], []
    for r in range(30):
        rng = np.random.default_rng(635000 + r)
        df = pd.DataFrame(rng.normal(size=(400, 4)), columns=["signal", "n0", "n1", "n2"])
        df["y"] = 3 * df.signal + rng.normal(size=len(df))
        out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, CFG)
        final = out["consensus_evaluation"]
        assert final["status"] == "ok"
        scores.append(final["score"])
        retained.append("signal" in final["features"])
    record(runs=30, first_seed=635000, truth="y=3*signal+independent unit noise; population correlation .949 below .98 guard",
           mean_final_r2=float(np.mean(scores)),
           signal_retained=float(np.mean(retained)))
    assert np.mean(retained) > .95 and np.mean(scores) > .8


def test_existing_leakage_guard_can_exclude_a_legitimate_near_deterministic_signal(record):
    rng = np.random.default_rng(635000)
    df = pd.DataFrame(rng.normal(size=(400, 4)), columns=["signal", "n0", "n1", "n2"])
    df["y"] = 3 * df.signal + .25 * rng.normal(size=len(df))
    out = feature_selection(df, "y", dict.fromkeys(df.columns, "numeric_continuous"), {}, CFG)
    record(seed=635000, truth="legitimate near-deterministic signal, population correlation .9965",
           excluded_leaks=out["excluded_leaks"], final_features=out["consensus_evaluation"]["features"],
           limitation="association threshold alone cannot establish data leakage; existing exclusion policy retained")
    assert "signal" in out["excluded_leaks"] and "signal" not in out["consensus_evaluation"]["features"]
