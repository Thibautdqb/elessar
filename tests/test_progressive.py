from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig, see
from elessar.progressive import Coverage, CsvProbe, FrameProbe
from elessar.sources import _csv_kwargs, load
from elessar.types import good_turing_unseen, infer_type, wilson

BIG = StoneConfig(max_rows=4_000, chunk_rows=3_000, probe_max_rows=30_000)


def _csv(tmp_path: Path, df: pd.DataFrame, name: str = "t.csv") -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_wilson_and_good_turing() -> None:
    lo, hi = wilson(95, 100)
    assert 0.88 < lo < 0.9 < hi < 0.98
    assert wilson(50, 50)[0] > 0.92 and wilson(20, 20)[0] < 0.9  # 20 matches out of 20 cannot prove 90%
    assert good_turing_unseen(pd.Series(list("aabbcc"))) == 0
    assert good_turing_unseen(pd.Series(list("abcdef"))) == 1


def test_certainty_reflects_the_evidence() -> None:
    dates = pd.Series(["2020-01-0%d" % (i % 9 + 1) for i in range(19)] + ["oops"])
    small = infer_type(dates, sampled=True)  # 95% of 20 values: the interval crosses the 90% threshold
    assert small["semantic_type"] == "datetime_string" and small["certainty"] == "uncertain"
    big = infer_type(pd.concat([dates] * 50, ignore_index=True), sampled=True)
    assert big["certainty"] == "certain"
    assert infer_type(dates)["certainty"] == "certain"  # every value seen: nothing left to infer
    ids = infer_type(pd.Series(np.arange(5_000)), sampled=True)
    assert ids["semantic_type"] == "id" and ids["certainty"] == "probable"
    rare = pd.Series(list("ab" * 200) + [f"x{i}" for i in range(30)])
    assert "unseen" in infer_type(rare, sampled=True)["certainty_reason"]
    assert infer_type(pd.Series([1, 2, 3]))["path"][-1] == "integers?"


def test_probes_never_read_a_row_twice(tmp_path: Path) -> None:
    df = pd.DataFrame({"k": np.arange(5_000), "x": np.random.default_rng(0).normal(size=5_000)})
    path = _csv(tmp_path, df)
    for probe in (CsvProbe(str(path), _csv_kwargs(str(path), {})), FrameProbe(df)):
        probe.take_head(2_000)
        for p in (0.1, 0.3, 0.3, 0.9, 0.999):
            probe.take_block(p, 1_500)
        rows = probe.rows()
        assert rows["k"].is_unique and rows["k"].max() == 4_999
    cov = Coverage()
    cov.add(0, 10)
    cov.add(20, 30)
    assert cov.free_from(5) == 10 and cov.room(12) == 8 and cov.free_from(25) == 30


def test_late_values_and_late_formats_are_seen(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 30_000
    late = np.full(n, np.nan)
    late[22_000:] = rng.normal(size=n - 22_000)  # empty for the first 73% of the file
    code = np.where(np.arange(n) < 27_000, rng.integers(0, 100, n).astype(str), "n/a?")  # junk only at the end
    df = pd.DataFrame({"late": late, "code": code, "x": rng.normal(size=n)})
    path = _csv(tmp_path, df)
    src = load(path, BIG)
    assert src.plan is not None and src.plan.types["late"]["semantic_type"] == "numeric_continuous"
    assert src.full.columns.loc["late", "n_missing"] == 22_000
    v = see(path, config=BIG, plots=False, verbose=False)
    assert any(a.get("kind") == "typing" for a in v.alerts)


def test_identifier_checked_on_every_row(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 40_000
    ids = np.arange(n) + 1_000_000
    ids[rng.choice(n, 30, replace=False)] = ids[rng.choice(n, 30, replace=False)]  # 30 repeats, far apart
    df = pd.DataFrame({"order_id": ids, "amount": rng.lognormal(3, 1, n).round(2)})
    v = see(_csv(tmp_path, df), config=BIG, plots=False, verbose=False)
    assert v.columns["order_id"]["semantic_type"] == "categorical"
    assert "repeat over all" in v.columns["order_id"]["reason"]
    clean = df.assign(order_id=np.arange(n) + 1_000_000)
    v = see(_csv(tmp_path, clean, "clean.csv"), config=BIG, plots=False, verbose=False)
    assert v.columns["order_id"]["semantic_type"] == "id" and v.columns["order_id"]["certainty"] == "certain"


def test_values_that_break_the_type_are_reported(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 30_000
    amount = rng.normal(100, 10, n).round(2).astype(str)
    bad = rng.choice(np.arange(5_000, n), 12, replace=False)
    amount[bad] = "error"
    df = pd.DataFrame({"amount": amount, "y": rng.normal(size=n)})
    v = see(_csv(tmp_path, df), config=BIG, plots=False, verbose=False)
    checks = [a["message"] for a in v.alerts if a.get("kind") == "type_check"]
    assert any("12 values" in m and "'error'" in m for m in checks)
    assert v.full.loc["amount", "mean"] == pytest.approx(pd.to_numeric(df["amount"], errors="coerce").mean())


def test_time_series_keep_their_order_and_are_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    t = pd.date_range("2015-01-01", periods=60_000, freq="h")
    y = 10 + 5 * np.sin(2 * np.pi * np.arange(len(t)) / 24) + rng.normal(0, 1, len(t))
    df = pd.DataFrame({"when": t, "load": y.round(3)})
    path = _csv(tmp_path, df)
    src = load(path, BIG)
    assert src.plan.sampling == "blocks" and src.plan.timeseries["kind"] == "single series"
    assert src.full.series is not None and len(src.full.series) == len(t)
    v = see(path, config=BIG, plots=False, verbose=False)
    ts = v.timeseries
    assert ts["exact"] and ts["n_steps"] >= 50_000 - 1
    assert ts["analyses"]["load"]["period"] == 24


def test_panel_entities_are_whole(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    t = pd.date_range("2018-01-01", periods=400, freq="D")
    stores = [f"S{i:03d}" for i in range(120)]
    df = pd.DataFrame({"day": np.tile(t, len(stores)), "store": np.repeat(stores, len(t))})
    df["sales"] = rng.gamma(5, 10, len(df)).round(1)
    src = load(_csv(tmp_path, df), BIG)
    assert src.plan.timeseries["kind"] == "panel"
    rows = src.full.panel_rows
    assert rows is not None and 0 < rows["store"].nunique() < len(stores)
    assert (rows.groupby("store").size() == len(t)).all()  # every kept store has its whole history
    assert len(rows) <= BIG.max_rows
    assert src.full.series["active entities"].max() == len(stores)
