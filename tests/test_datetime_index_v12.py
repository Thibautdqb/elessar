"""Regression tests for datetime/period row-index temporal inference."""
import numpy as np
import pandas as pd
import pytest
import elessar


@pytest.fixture
def config():
    return elessar.StoneConfig(
        n_jobs=1, random_state=11, fit_timeout=.2,
        candidates_continuous=("norm",), candidates_discrete=("poisson",)
    )


def test_datetime_index_is_detected_and_kept_virtual(config):
    rng = np.random.default_rng(4)
    idx = pd.date_range("2020-01-01", periods=120, freq="D")
    df = pd.DataFrame({"x": rng.normal(size=len(idx)), "y": rng.normal(size=len(idx))}, index=idx)
    out = elessar.see(df, plots=False, config=config, verbose=False, open_report=False)
    assert out.timeseries and out.timeseries["kind"] == "single series"
    assert out.timeseries["time_column"] == "__datetime_index__"
    assert len(out.summary) == 2 and "__datetime_index__" not in out.summary.index
    assert out.correlations.get("serial") is not None


def test_period_index_is_detected(config):
    idx = pd.period_range("1990Q1", periods=80, freq="Q")
    df = pd.DataFrame({"sales": np.arange(80, dtype=float) + np.sin(np.arange(80))}, index=idx)
    out = elessar.see(df, plots=False, config=config, verbose=False, open_report=False)
    assert out.timeseries and out.timeseries["time_column"] == "__datetime_index__"
    assert out.timeseries["frequency"] == "quarter"


def test_rangeindex_is_not_a_time_axis(config):
    df = pd.DataFrame({"x": np.arange(80), "y": np.arange(80) ** 2})
    out = elessar.see(df, plots=False, config=config, verbose=False, open_report=False)
    assert not out.timeseries


def test_explicit_datetime_column_is_preferred(config):
    idx = pd.date_range("2020-01-01", periods=80, freq="D")
    df = pd.DataFrame({"when": idx, "x": np.arange(80, dtype=float)}, index=idx)
    out = elessar.see(df, plots=False, config=config, verbose=False, open_report=False)
    assert out.timeseries and out.timeseries["time_column"] == "when"


def test_weekly_index_with_missing_values(config):
    idx = pd.date_range("2010-01-01", periods=105, freq="W-FRI")
    df = pd.DataFrame({"co2": np.linspace(300, 340, 105)}, index=idx)
    df.iloc[10:14, 0] = np.nan
    out = elessar.see(df, plots=False, config=config, verbose=False, open_report=False)
    assert out.timeseries and out.timeseries["time_column"] == "__datetime_index__"
    assert out.timeseries["kind"] == "single series"
    assert out.timeseries["frequency"] == "week"


def test_datetime_index_html_and_snippets(config, tmp_path):
    idx = pd.date_range("2020-01-01", periods=85, freq="D")
    df = pd.DataFrame({"value": np.random.default_rng(5).normal(size=85)}, index=idx)
    out = elessar.see(df, plots=True, config=config, verbose=False,
                      open_report=False, out_dir=tmp_path)
    assert out.timeseries
    assert "Time series" in (tmp_path / "report.html").read_text(encoding="utf-8")
    steps = out.timeseries.get("treatment", [])
    code = "\n".join(str(s.get("code")) for s in steps)
    assert "df['__datetime_index__']" not in code
    assert "set_index('__datetime_index__')" not in code


def test_time_index_does_not_add_data_columns(config):
    idx = pd.date_range("2022-01-01", periods=100, freq="D")
    x = np.random.default_rng(3).normal(size=100)
    df = pd.DataFrame({"a": x, "b": 2 * x + 1}, index=idx)
    out = elessar.see(df, plots=False, config=config, verbose=False, open_report=False)
    assert out.timeseries
    assert len(out.summary) == 2
