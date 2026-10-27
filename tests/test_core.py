from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import elessar
from elessar import StoneConfig, Vision, see, see_column


def test_public_api() -> None:
    for name in ("see", "see_column", "see_types", "see_stats", "see_dist", "see_corr", "see_quality", "see_plot",
                 "Vision", "StoneConfig"):
        assert hasattr(elessar, name)


def test_see_full(messy_df: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = see(messy_df, target="target", out_dir=tmp_path / "out")
    printed = capsys.readouterr().out
    for section in ("OVERVIEW", "ALERTS", "COLUMN SUMMARY", "COLUMN DETAILS", "CORRELATIONS", "TARGET ANALYSIS", "PLOTS"):
        assert section in printed
    assert isinstance(report, Vision)
    assert list(report.summary.index) == list(messy_df.columns)
    assert set(report.columns) == set(messy_df.columns)
    assert report.columns["income"]["distribution"]["fit"]["best"] == "lognorm"
    assert report.overview["n_duplicate_rows"] == 5
    assert report.plots and all(p.exists() for p in report.plots)
    assert "target" in report.correlations
    assert {a["severity"] for a in report.alerts} <= {"high", "medium", "info"}

    data = json.loads(report.to_json(tmp_path / "r.json").read_text())
    assert set(data) == {"overview", "summary", "columns", "correlations", "quality", "alerts", "plots"}
    csv = pd.read_csv(report.to_csv(tmp_path / "s.csv"), index_col=0)
    assert list(csv.index) == list(messy_df.columns)
    md = report.to_markdown(tmp_path / "r.md").read_text()
    assert "## Data quality" in md and "![" in md and "out/" in md
    assert "Vision(" in repr(report)


def test_see_edge_cases_never_crash(edge_df: pd.DataFrame, tmp_path: Path) -> None:
    report = see(edge_df, out_dir=tmp_path, verbose=False)
    assert report.render(width=80)
    json.dumps(report.to_dict())
    assert len(report.summary) == edge_df.shape[1]


def test_see_series_and_column() -> None:
    s = pd.Series(np.random.default_rng(0).exponential(size=500), name="wait")
    report = see(s, plots=False, verbose=False)
    assert list(report.summary.index) == ["wait"]
    detail = see_column(s)
    assert detail["semantic_type"] == "numeric_continuous"
    assert detail["distribution"]["fit"]["best"] in ("expon", "weibull_min", "gamma")  # expon is nested in both
    assert detail["errors"] == {}
    with pytest.raises(TypeError):
        see_column([1, 2, 3])  # type: ignore[arg-type]


def test_see_column_records_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("elessar.core.column_stats", boom)
    detail = see_column(pd.Series([1.0, 2.0, 3.0]))
    assert "kaboom" in detail["errors"]["stats"]
    assert any(a["severity"] == "high" for a in detail["alerts"])


def test_sampling_big_frame() -> None:
    n = 300_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n)})
    cfg = StoneConfig(sample_size=5_000, fit_sample_size=2_000)
    report = see(df, config=cfg, plots=False, verbose=False)
    assert report.columns["a"]["distribution"]["fit"]["n_used"] == 2_000
    again = see(df, config=cfg, plots=False, verbose=False)
    assert report.correlations["kendall"].equals(again.correlations["kendall"])


def test_target_validation() -> None:
    with pytest.raises(KeyError):
        see(pd.DataFrame({"a": [1]}), target="nope", verbose=False, plots=False)


def test_render_respects_width(messy_df: pd.DataFrame) -> None:
    text = see(messy_df, plots=False, verbose=False).render(width=90)
    assert max(len(line) for line in text.splitlines()) <= 90
