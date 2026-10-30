"""Feature selection: honest scores on noise, recovery of known signals, validation on the future of dated rows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig, see
from elessar.correlations import compute_correlations
from elessar.selection import feature_selection

pytestmark = pytest.mark.validation
CFG = StoneConfig()


def _select(df: pd.DataFrame, target: str, target_type: str) -> dict:
    types = {c: "numeric_continuous" for c in df}
    types[target] = target_type
    return feature_selection(df, target, types, compute_correlations(df, types, target, CFG), CFG)


@pytest.mark.parametrize("task", ["binary", "regression"])
def test_scores_on_pure_noise_are_not_optimistic(task: str, record) -> None:  # noqa: ANN001
    runs = 60
    scores, kept = [], []
    for r in range(runs):
        rng = np.random.default_rng(1000 + r)
        df = pd.DataFrame(rng.normal(size=(400, 10)), columns=[f"x{i}" for i in range(10)])
        df["y"] = rng.integers(0, 2, 400) if task == "binary" else rng.normal(size=400)
        out = _select(df, "y", "boolean" if task == "binary" else "numeric_continuous")
        scores.append(out["scores"])
        kept.append(len(out["keep"]) / 10)
    s = pd.DataFrame(scores)
    mean, se = s.mean(), s.std() / np.sqrt(runs)
    record(task=task, runs=runs, rows=400, features=10, scoring="roc_auc" if task == "binary" else "r2",
           mean_scores=mean.round(4).to_dict(), noise_features_kept=float(np.mean(kept)))
    if task == "binary":  # chance is an AUC of 0.5: no method may sit significantly above it
        assert (mean - 3 * se < 0.5).all()
    else:  # no model may explain noise
        assert (mean - 3 * se < 0).all()
    # a method votes only for features that beat their shuffled shadows: noise is rarely kept (it was 28% with a
    # binary target when ridge voted for every coefficient above 10% of the largest)
    assert np.mean(kept) < 0.05


def test_known_signals_are_kept_and_noise_dropped(record) -> None:  # noqa: ANN001
    """Two linear signals, one U-shaped signal, eight noise columns.

    Lasso and ridge only see linear links; the trees and the filter (Spearman and a Kruskal-Wallis across
    deciles) see the U shape, so it reaches three votes. Before 1.1 the filter was Spearman alone and the U
    shape was kept in a third of the runs.
    """
    runs = 30
    linear, u_keep, u_seen, false_keep = [], [], [], []
    for r in range(runs):
        rng = np.random.default_rng(2000 + r)
        n = 600
        df = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"noise{i}" for i in range(8)])
        df["x1"], df["x2"], df["x3"] = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        df["y"] = 1.0 * df["x1"] - 0.7 * df["x2"] + 0.5 * df["x3"] ** 2 + rng.normal(size=n)
        out = _select(df, "y", "numeric_continuous")
        decision = out["table"]["decision"]
        linear.append((decision[["x1", "x2"]] == "keep").mean())
        u_keep.append(decision["x3"] == "keep")
        u_seen.append(decision["x3"] != "drop")
        false_keep.append((decision[[f"noise{i}" for i in range(8)]] == "keep").mean())
    record(runs=runs, rows=600, linear_signals_kept=np.mean(linear), u_shaped_signal_kept=np.mean(u_keep),
           u_shaped_signal_kept_or_borderline=np.mean(u_seen), noise_kept=np.mean(false_keep))
    assert np.mean(linear) > 0.95 and np.mean(false_keep) < 0.05
    assert np.mean(u_keep) > 0.8


def test_dated_rows_are_validated_on_their_future_whatever_their_order(record) -> None:  # noqa: ANN001
    """Two independent random walks: nothing to predict. A shuffled cross-validation found an R2 of about +0.4, and
    in levels the walk x "explained" y: it was kept in 9 runs out of 10. The target is not stationary, so the
    selection models its changes from the changes of x."""
    runs = 10
    rows = []
    for r in range(runs):
        rng = np.random.default_rng(3000 + r)
        n = 600
        df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=n, freq="D"), "x": np.cumsum(rng.normal(size=n)),
                           "noise": rng.normal(size=n), "y": np.cumsum(rng.normal(size=n))})
        ordered = see(df, target="y", plots=False, verbose=False).selection
        shuffled = see(df.sample(frac=1, random_state=r), target="y", plots=False, verbose=False).selection
        rows.append({"best_score": max(ordered["scores"].values()), "same": ordered["scores"] == shuffled["scores"],
                     "x_kept": "x" in ordered["keep"], "differences": ordered["on_differences"] == ["x", "y"]})
    t = pd.DataFrame(rows)
    record(runs=runs, mean_best_r2=t["best_score"].mean(), shuffled_file_gives_same_scores=float(t["same"].mean()),
           random_walk_kept=float(t["x_kept"].mean()), modelled_on_differences=float(t["differences"].mean()))
    assert t["same"].all() and (t["best_score"] < 0.05).all() and t["x_kept"].mean() <= 0.2
