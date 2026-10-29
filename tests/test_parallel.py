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


def test_spawned_pool_keeps_order_and_reports_failures() -> None:
    import operator

    from elessar.parallel import run_spawned

    out = run_spawned(operator.truediv, [(6, 3), (1, 0), (9, 3)], 2)
    assert out[0] == 2 and out[2] == 3 and isinstance(out[1], TaskError) and "ZeroDivisionError" in out[1].error


def test_big_file_chunks_on_spawned_processes_match_one_process(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """What macOS and Windows do (no safe fork): chunks of a big file on spawned processes, same figures."""
    import elessar.parallel as par
    from elessar import load

    rng = np.random.default_rng(5)
    n = 30_000
    df = pd.DataFrame({"id": np.arange(n), "amount": rng.lognormal(3, 1, n).round(2),
                       "shop": rng.choice(list("abcdef"), n), "qty": rng.poisson(3, n)})
    path = tmp_path / "big.parquet"
    df.to_parquet(path, row_group_size=3_000)
    monkeypatch.setattr(par, "can_fork", lambda: False)
    spawned = load(path, StoneConfig(max_rows=5_000, n_jobs=3))
    serial = load(path, StoneConfig(max_rows=5_000, n_jobs=1))
    assert "3 processes" in spawned.full.engine and "processes" not in serial.full.engine
    assert spawned.full.n_rows == serial.full.n_rows == n
    pd.testing.assert_frame_equal(spawned.full.columns, serial.full.columns)
    pd.testing.assert_frame_equal(spawned.full.sample, serial.full.sample)
