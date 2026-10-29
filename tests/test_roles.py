from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import see
from elessar.roles import birth_columns, birth_report, is_birth_name, is_event_name

N = 600
RNG = np.random.default_rng(11)
BIRTHS = pd.Timestamp("1940-01-01") + pd.to_timedelta(RNG.integers(0, 60 * 365, N), unit="D")  # 1940 - 2000


def _types(df: pd.DataFrame) -> dict[str, str]:
    return {c: ("datetime" if pd.api.types.is_datetime64_any_dtype(df[c]) else "categorical") for c in df.columns}


def _people(**cols: object) -> pd.DataFrame:
    return pd.DataFrame({"client_id": np.arange(N), "sex": RNG.choice(["F", "M"], N), **cols})


@pytest.mark.parametrize("name", ["birth_date", "DateOfBirth", "dob", "DOB", "date_naissance", "Geburtsdatum",
                                  "fecha_nacimiento", "birth_number", "client_birthdate", "born_on", "ddn"])
def test_birth_names(name: str) -> None:
    assert is_birth_name(name)


@pytest.mark.parametrize("name", ["adobe_date", "stubborn", "reborn_at", "date", "created", "order_date",
                                  "anniversary", "release_date", "dobby"])
def test_not_birth_names(name: str) -> None:
    assert not is_birth_name(name)


def test_event_names_never_births() -> None:
    assert is_event_name("hire_date") and is_event_name("DateCreated") and is_event_name("death_date")
    assert not is_event_name("birth_date") and not is_event_name("date")


def test_named_birth_date_is_certain_and_aged() -> None:
    df = _people(birth_date=BIRTHS)
    found = birth_columns(df, _types(df))
    assert found["birth_date"]["certainty"] == "certain"
    rep = birth_report(df, "birth_date", _types(df))
    assert 25 <= rep["min_age"] and rep["max_age"] <= 90
    assert rep["reference"].startswith("on ")  # no other date in the table: ages on the day of the analysis


def test_unnamed_birth_dates_in_a_people_table_are_probable() -> None:
    df = _people(date=BIRTHS)
    assert birth_columns(df, _types(df))["date"]["certainty"] == "probable"
    no_people = pd.DataFrame({"title_id": np.arange(N), "date": BIRTHS})  # films, not people
    assert birth_columns(no_people, _types(no_people)) == {}


@pytest.mark.parametrize("label,dates", [
    ("signup dates (too young)", pd.Timestamp("2015-01-01") + pd.to_timedelta(RNG.integers(0, 3000, N), unit="D")),
    ("timestamps with a time of day", BIRTHS + pd.to_timedelta(RNG.integers(1, 86_000, N), unit="s")),
    ("sorted event dates", pd.Series(np.sort(BIRTHS.to_numpy()))),
    ("future expiry dates", pd.Timestamp("2027-01-01") + pd.to_timedelta(RNG.integers(0, 2000, N), unit="D")),
    ("narrow range", pd.Timestamp("1980-01-01") + pd.to_timedelta(RNG.integers(0, 3 * 365, N), unit="D")),
    ("month-level dates (first appearance, 1935-2013)",
     pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(RNG.integers(1935, 2014, N), RNG.integers(1, 13, N))])),
])
def test_dates_that_must_not_be_taken_for_births(label: str, dates: pd.Series) -> None:
    df = _people(date=pd.Series(dates).to_numpy())
    assert birth_columns(df, _types(df)) == {}, label
    for event in ("hire_date", "first_appearance", "term_start", "since"):  # plausible values, event names
        df = _people(**{event: BIRTHS})
        assert birth_columns(df, _types(df)) == {}, event


def test_birth_named_but_implausible_is_reported_not_used() -> None:
    df = _people(birth_date=pd.Timestamp("2030-01-01") + pd.to_timedelta(RNG.integers(0, 900, N), unit="D"))
    assert "rejected" in birth_columns(df, _types(df))["birth_date"]


def test_age_at_the_event_and_the_checks() -> None:
    births = BIRTHS.to_series(index=range(N)).copy()
    births.iloc[:40] = pd.Timestamp("1950-01-01")  # unknown day and month recorded as 01-01
    admission = pd.Timestamp("2010-01-01") + pd.to_timedelta(RNG.integers(0, 3000, N), unit="D")
    births.iloc[40:43] = pd.Timestamp("2020-06-01")  # born after admission: data error
    df = _people(birth_date=births.to_numpy(), admission_date=admission)
    df["age"] = ((admission - births).dt.days / 365.2425).round() + 3  # a stale age, 3 years off
    types = {**_types(df), "age": "numeric_continuous"}
    rep = birth_report(df, "birth_date", types)
    assert "at admission_date" in rep["reference"]
    text = " ".join(m for _, m in rep["checks"])
    assert "after the reference date" in text and "1 January" in text and "personal data" in text
    assert "constant" in text or "disagrees" in text


def test_see_derives_age_and_never_uses_a_birth_date_as_time_axis() -> None:
    df = _people(birth_date=BIRTHS, income=RNG.lognormal(10, 0.5, N).round())
    v = see(df, plots=False, verbose=False)
    d = v.columns["birth_date"]
    assert d["role"] == "date of birth" and d["path"][-1] == "a date of birth? yes"
    assert "birth_date_age" in v.columns and v.columns["birth_date_age"]["derived_from"] == "birth_date"
    assert v.timeseries is None  # a column of births is not a time series
    assert any(a.get("kind") == "birth" and "personal data" in a["message"] for a in v.alerts)
    steps = [s for s in v.preparation["steps"] if s["column"] == "birth_date" and s["action"] == "Decomposition"]
    assert steps and "age" in steps[0]["why"]
    years = _people(birth_year=BIRTHS.year.to_numpy())
    assert see(years, plots=False, verbose=False).timeseries is None  # birth_year is not a year axis


def test_czech_birth_numbers_are_dates_of_birth() -> None:
    nums = [f"{d:%y}{d.month + (50 if i % 2 else 0):02d}{d:%d}" for i, d in enumerate(BIRTHS)]
    v = see(pd.DataFrame({"client_id": np.arange(N), "birth_number": nums}), plots=False, verbose=False)
    assert v.columns["birth_number"]["role"] == "date of birth"
    assert v.columns["birth_number_age"]["stats"]["min"] > 20


def test_zero_dates_are_missing_and_dates_are_never_phones() -> None:
    died = pd.Series([f"{y}-0{m}-1{d}" for y, m, d in zip(RNG.integers(1900, 2020, N), RNG.integers(1, 10, N),
                                                          RNG.integers(0, 10, N))])
    died[RNG.random(N) < 0.32] = "0000-00-00"  # still alive
    df = pd.DataFrame({"name": [f"p{i}" for i in range(N)], "died": died})
    v = see(df, plots=False, verbose=False)
    assert v.columns["died"]["semantic_type"] == "datetime_string"
    assert "'0000-00-00'" in v.quality["placeholder_values"]["died"]


def test_age_at_a_year_column() -> None:
    df = pd.DataFrame({"gender": RNG.choice(["female", "male"], N), "born": BIRTHS,
                       "year": (BIRTHS.year + RNG.integers(30, 70, N)).to_numpy()})
    rep = birth_report(df, "born", {"born": "datetime", "gender": "categorical", "year": "numeric_discrete"})
    assert rep["reference"].startswith("at year") and 29 <= rep["median_age"] <= 71
