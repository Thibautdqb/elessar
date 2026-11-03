"""Predeclared null calibration and power of the conservative serial test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, signal, stats

from elessar import StoneConfig
from elessar.correlations import compute_correlations

pytestmark = pytest.mark.validation
CFG = StoneConfig(n_jobs=1)
RUNS = 500
NULLS = ["positive_ar1", "negative_ar1", "ar2", "ma100", "tied_ar1", "panel",
         "four_state", "negative_markov", "rotating_markov", "mixed", "categorical_target"]


def ar1(rng, n, phi=.9):
    x = np.empty(n)
    x[0] = rng.normal() / np.sqrt(1 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


def ar2(rng, n):
    a = np.array([[1.2, -.65], [1., 0.]])
    covariance = linalg.solve_discrete_lyapunov(a, np.diag([1., 0.]))
    state = rng.multivariate_normal(np.zeros(2), covariance)
    out = []
    for _ in range(n):
        state = a @ state + [rng.normal(), 0.]
        out.append(state[0])
    return np.array(out)


def chain(rng, n, kind="four_state"):
    if kind == "rotating_markov":
        transition = .3 * np.eye(3) + .6 * np.roll(np.eye(3), 1, axis=1) + .1 / 3
        x = np.empty(n, dtype=int)
        x[0] = rng.integers(3)
        for i in range(1, n):
            x[i] = rng.choice(3, p=transition[x[i - 1]])
        return x.astype(str)
    x = np.empty(n, dtype=int)
    x[0] = rng.integers(2)
    switch = .9 if kind == "negative_markov" else .05
    for i in range(1, n):
        x[i] = x[i - 1] ^ int(rng.random() < switch)
    return ((2 * x + rng.integers(2, size=n)) if kind == "four_state" else x).astype(str)


def make_pair(rng, design):
    n = 600 if design in ("four_state", "negative_markov", "rotating_markov", "mixed", "categorical_target") else 300
    categorical = design in ("four_state", "negative_markov", "rotating_markov", "categorical_target")
    if categorical:
        kind = "four_state" if design == "categorical_target" else design
        a, b = chain(rng, n, kind), chain(rng, n, kind)
    elif design == "mixed":
        a, b = chain(rng, n), ar1(rng, n)
    elif design == "ar2":
        a, b = ar2(rng, n), ar2(rng, n)
    elif design == "ma100":
        weights = np.arange(1, 101, dtype=float) ** -.6
        a, b = (signal.convolve(rng.normal(size=n + 99), weights, mode="valid") for _ in range(2))
    elif design == "panel":
        a = np.column_stack([ar1(rng, n)] * 3) + .25 * rng.normal(size=(n, 3))
        b = np.column_stack([ar1(rng, n)] * 3) + .25 * rng.normal(size=(n, 3))
    else:
        a, b = ar1(rng, n, -.9 if design == "negative_ar1" else .9), ar1(rng, n, -.9 if design == "negative_ar1" else .9)
        if design == "tied_ar1":
            a, b = np.digitize(a, [-1., 1.]), np.digitize(b, [-1., 1.])
    types = {"a": "categorical" if categorical or design == "mixed" else "numeric_continuous",
             "b": "categorical" if categorical else "numeric_continuous"}
    dates = pd.date_range("2020-01-01", periods=n)
    panel = design == "panel"
    serial = {"time": pd.Series(np.repeat(dates, 3) if panel else dates),
              "entity": pd.Series(np.tile(["e0", "e1", "e2"], n)) if panel else None, "differences": None}
    return pd.DataFrame({"a": np.asarray(a).ravel(), "b": np.asarray(b).ravel()}), types, serial


def pvalue(out, design):
    if design == "categorical_target":
        return float(out["target"].iloc[0]["p_value"])
    matrix = "eta_p" if design == "mixed" else "cramers_v_p" if design in ("four_state", "negative_markov", "rotating_markov") else "kendall_p"
    return float(out[matrix].loc["a", "b"])


def summary(p):
    values = np.asarray(p)
    count = int((values < .05).sum())
    interval = stats.binomtest(count, len(values)).proportion_ci(.95, method="wilson")
    return {"runs": len(values), "finite": int(np.isfinite(values).sum()), "rejected": count,
            "rate": count / len(values), "wilson95": [float(interval.low), float(interval.high)]}


@pytest.mark.parametrize("design", NULLS)
def test_serial_null_is_controlled_on_500_known_independent_processes(design, record):
    seed = 632000 + NULLS.index(design)
    rng = np.random.default_rng(seed)
    p = []
    for _ in range(RUNS):
        df, types, serial = make_pair(rng, design)
        out = compute_correlations(df, types, target="b" if design == "categorical_target" else None,
                                   config=CFG, serial=serial)
        p.append(pvalue(out, design))
    measured = summary(p)
    upper = .05 + 3 * np.sqrt(.05 * .95 / RUNS)
    record(design=design, seed=seed, truth="independent stationary processes", nominal=.05, upper_bound=float(upper), **measured)
    assert measured["finite"] == RUNS
    assert measured["rate"] < upper


@pytest.mark.parametrize("design", ["positive", "negative", "categorical", "mixed"])
def test_serial_power_on_predeclared_strong_links(design, record):
    seed = 633000 + ["positive", "negative", "categorical", "mixed"].index(design)
    rng = np.random.default_rng(seed)
    p = []
    for _ in range(200):
        a = ar1(rng, 600)
        if design in ("positive", "negative"):
            b = (1 if design == "positive" else -1) * a + .25 * rng.normal(size=len(a))
            types = {"a": "numeric_continuous", "b": "numeric_continuous"}
        elif design == "categorical":
            a = chain(rng, 600)
            b = a.copy()
            use_noise = rng.random(len(a)) < .1
            b[use_noise] = rng.integers(4, size=use_noise.sum()).astype(str)
            types = {"a": "categorical", "b": "categorical"}
        else:
            b = a + .25 * rng.normal(size=len(a))
            a = np.where(a > 0, "high", "low")
            types = {"a": "categorical", "b": "numeric_continuous"}
        serial = {"time": pd.Series(pd.date_range("2020-01-01", periods=600)), "entity": None, "differences": None}
        out = compute_correlations(pd.DataFrame({"a": a, "b": b}), types, config=CFG, serial=serial)
        p.append(pvalue(out, "four_state" if design == "categorical" else "mixed" if design == "mixed" else "positive_ar1"))
    measured = summary(p)
    record(design=design, seed=seed, truth="strong synchronous link", **measured)
    assert measured["finite"] == 200
    assert measured["rate"] > .8
