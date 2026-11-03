"""Fixed-seed structural stress checks; not claims of nominal statistical calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig
from elessar.outliers import column_outliers
from elessar.preparation import plan
from elessar.selection import _holdout_mask, _time_folds

pytestmark = pytest.mark.validation
RUNS = 500
CFG = StoneConfig(n_jobs=1)


def test_whole_date_splits_over_500_unbalanced_time_tables(record):
    rng = np.random.default_rng(511001)
    rows = []
    for _ in range(RUNS):
        n_dates = int(rng.integers(3, 41))
        counts = rng.integers(10, 41, size=n_dates)
        dates = np.repeat(np.arange(n_dates), counts)
        train = _holdout_mask(len(dates), dates, True, CFG.random_state)
        assert train is not None
        assert train.sum() >= 20 and (~train).sum() >= 10
        assert dates[train].max() < dates[~train].min()
        folds = _time_folds(dates[train], 5)
        if folds is None:
            assert np.unique(dates[train]).size < 3
        else:
            for tr, va in folds:
                assert dates[train][tr].max() < dates[train][va].min()
        rows.append(len(dates))
    record(runs=RUNS, seed=511001, min_rows=min(rows), max_rows=max(rows),
           holdout_date_overlaps=0, cv_date_overlaps=0, tested="partition integrity, not predictive performance")


def test_outlier_actions_over_500_columns_preserve_missing_and_row_identity(record):
    rng = np.random.default_rng(511002)
    removed, missing_preserved = 0, 0
    for i in range(RUNS):
        n = int(rng.integers(128, 513))
        scale = 10 ** rng.uniform(-2, 2)
        values = rng.normal(size=n) * scale if i % 2 else np.exp(rng.normal(0, .5, n)) * scale
        values[0] = 1000 * scale if i % 2 else 1e-12 * scale
        values[-2:] = np.nan
        df = pd.DataFrame({"x": values}, index=np.arange(n) // 2)
        result = column_outliers(df.x)
        assert result is not None
        assert result["remove_mask"].shape == result["consensus_mask"].shape == (n,)
        assert not result["remove_mask"][-2:].any() and not result["consensus_mask"][-2:].any()
        if result["action"] == "remove":
            assert result["remove_mask"][0]
            assert result["remove_mask"].sum() == 1
            details = {"x": {"semantic_type": "numeric_continuous", "stats": {}}}
            prep = plan(df, details, {}, {}, None, CFG, outliers={"columns": {"x": result}})
            code = next(s["code"] for s in prep["steps"] if s["action"] == "Remove outliers")
            ns = {"df": df.copy(), "np": np, "pd": pd}
            exec(code, ns)
            assert len(ns["df"]) == n - 1 and ns["df"].x.isna().sum() == 2
            removed += 1
            missing_preserved += 2
    # This checks faithful execution of the policy, not that every error is diagnosed.
    assert removed > RUNS // 2
    record(runs=RUNS, seed=511002, removal_advised=removed, missing_rows_preserved_in_removal_runs=missing_preserved,
           tested="action identity and missing preservation; detection policy otherwise unchanged")


def test_target_encoding_over_500_tables_has_stable_shapes_and_seed(record):
    rng = np.random.default_rng(511003)
    counts = {"binary": 0, "multiclass": 0, "integer regression": 0}
    for i in range(RUNS):
        task = list(counts)[i % 3]
        n = 150
        y = rng.integers(0, 2 if task == "binary" else 3, n) if task != "integer regression" else np.arange(n)
        df = pd.DataFrame({"category": rng.choice([f"c{k}" for k in range(15)], n), "target": y},
                          index=np.arange(n) // 2)
        details = {"category": {"semantic_type": "categorical", "stats": {"cardinality": 15}},
                   "target": {"semantic_type": "numeric_discrete" if task == "integer regression" else "categorical",
                              "stats": {}}}
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
        counts[task] += 1
    record(runs=RUNS, seed=511003, target_types=counts, repeated_outputs_identical=True,
           tested="dimensions, duplicate-index preservation and deterministic encoding, not unbiased model evaluation")
