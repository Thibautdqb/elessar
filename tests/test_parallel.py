from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar import StoneConfig, see
from elessar.parallel import TaskError, can_fork, n_workers, run_tasks, shared


def _square(i: int) -> int:
    if i == 3:
        raise ValueError("boom")
    return i * i + shared()["offset"]


def test_run_tasks_keeps_order_and_reports_failures() -> None:
    for workers in (1, 3):
        out = run_tasks(_square, range(6), workers, {"offset": 1})
        assert [o for i, o in enumerate(out) if i != 3] == [1, 2, 5, 17, 26]
        assert isinstance(out[3], TaskError) and "boom" in out[3].error


def test_worker_count() -> None:
    cfg = StoneConfig()
    assert n_workers(cfg, 10, 50) == 1  # small table: one process
    assert n_workers(StoneConfig(n_jobs=1), 10**9, 50) == 1
    if can_fork():
        assert n_workers(StoneConfig(n_jobs=2), 10, 50) == 2
        assert n_workers(cfg, 10**9, 3) <= 3


@pytest.mark.skipif(not can_fork(), reason="process pool needs fork")
def test_parallel_and_serial_reports_are_identical(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 3_000
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) * (i + 1) for i in range(5)})
    df["c"] = rng.choice(list("abc"), n)
    df["y"] = (df["x0"] + rng.normal(size=n) > 0).astype(int)
    df.loc[rng.random(n) < 0.1, "x1"] = np.nan
    a = see(df, target="y", config=StoneConfig(n_jobs=1), verbose=False, out_dir=tmp_path / "a")
    b = see(df, target="y", config=StoneConfig(n_jobs=4), verbose=False, out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a.summary, b.summary)
    pd.testing.assert_frame_equal(a.correlations["target"], b.correlations["target"])
    pd.testing.assert_frame_equal(a.correlations["pairs"], b.correlations["pairs"])
    assert [p.name for p in a.plots] == [p.name for p in b.plots]
    assert [x["message"] for x in a.alerts] == [x["message"] for x in b.alerts]
    assert b.overview["workers"] == 4
