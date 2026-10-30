"""Outliers: how many clean points the three methods flag, and whether planted errors are found."""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig
from elessar.outliers import column_outliers

from .sim import upper

pytestmark = pytest.mark.validation
CFG = StoneConfig()


@pytest.mark.parametrize("shape", ["normal", "student t5", "student t3", "lognormal", "exponential"])
def test_clean_data_flags_and_actions(shape: str, record) -> None:  # noqa: ANN001
    """No error in the data: the share of points flagged by two methods (consensus) and the advice given.

    Removal is advised only for values 10 robust standard deviations away (on the log scale for a positive,
    right-skewed column) from a column whose other values have light tails: a heavy or long tail is never
    mistaken for errors. Known and harmless: a clean normal column of 1,000 rows almost always holds a few
    consensus points, so capping them (reassign) is advised for most such columns, which changes little."""
    rng = np.random.default_rng({"normal": 41, "student t5": 42, "student t3": 45, "lognormal": 43, "exponential": 46}[shape])
    draw = {"normal": lambda: rng.normal(size=1_000), "student t5": lambda: rng.standard_t(5, 1_000),
            "student t3": lambda: rng.standard_t(3, 1_000), "lognormal": lambda: rng.lognormal(0, 1, 1_000),
            "exponential": lambda: rng.exponential(1, 1_000)}[shape]
    runs = 300
    consensus, actions = [], []
    for _ in range(runs):
        r = column_outliers(pd.Series(draw()), CFG)
        consensus.append(r["pct_consensus"] / 100)
        actions.append(r["action"])
    share = pd.Series(actions).value_counts(normalize=True).to_dict()
    record(shape=shape, runs=runs, rows=1_000, consensus_share=np.mean(consensus), actions=share)
    assert share.get("remove", 0) < upper(0.005, runs)  # before 1.1: t3 70%, exponential 17%, t5 12%, log-normal 7%
    if shape == "normal":  # two of 3 sigma (0.27%), boxplot (0.70%) and modified z (0.05%) agree on few points
        assert np.mean(consensus) < 0.005


@pytest.mark.parametrize("column", ["normal", "uniform", "normal, 100 rows", "student t5", "exponential"])
def test_planted_errors_are_found(column: str, record) -> None:  # noqa: ANN001
    """Entry or unit errors (1 or 2 values multiplied by 100) planted in a column.

    In a light-tailed column they are all found and removal is advised. In a heavy-tailed or long-tailed column
    an error cannot be told from a genuine extreme value: it is still flagged by the methods, and the advice is
    to check it rather than to remove it."""
    rng = np.random.default_rng(zlib.crc32(column.encode()))
    draw = {"normal": lambda: rng.normal(50, 5, 1_000), "uniform": lambda: rng.uniform(10, 20, 1_000),
            "normal, 100 rows": lambda: rng.normal(50, 5, 100), "student t5": lambda: rng.standard_t(5, 1_000) + 10,
            "exponential": lambda: rng.exponential(1, 1_000)}[column]
    runs = 300
    found, removed = [], []
    for _ in range(runs):
        x = draw()
        bad = rng.choice(len(x), int(rng.integers(1, 3)), replace=False)
        x[bad] *= 100
        r = column_outliers(pd.Series(x), CFG)
        top = set(r["points"].loc[r["points"]["n_methods"] >= 2, "row"])
        found.append(set(bad) <= top)
        removed.append(r["action"] == "remove")
    record(column=column, runs=runs, planted_errors="1 or 2 values x100", all_flagged=np.mean(found),
           advice_remove=np.mean(removed))
    assert np.mean(found) > 0.9
    if column in ("normal", "uniform", "normal, 100 rows"):
        assert np.mean(removed) > 0.95
