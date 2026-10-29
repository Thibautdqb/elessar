"""Dates of birth: recognised among the date columns, then turned into what analyses need.

A date of birth is a date, but not a time axis: it says *who* a row is about, not *when* it
happened. Treated like any date it would be offered as the axis of a time series and decomposed
into year / month / weekday; what a model needs is the **age**, ideally at the date of the event
the row records.

Recognition, decided per column and explained:

- **by name**, on whole name tokens (``birth_date``, ``dob``, ``date_naissance``, ``geburtsdatum``,
  ``birth_number``...; ``adobe`` or ``stubborn`` never match), when the ages it implies are
  plausible: *certain*;
- **by values alone**, only when every one of these holds: the table is about people (a sex,
  gender, first name, age... column), dates recorded to the day (month-level dates, all on the
  1st, are periods or publications), no time of day, not sorted in row order (event logs are),
  at most 1% of dates in the future, ages between 0 and 110 for 98% of rows, a median age between
  18 and 85, and a 5-95% age spread of at least 20 years: *probable*, said as such. A column whose
  name says it records an event (``hire``, ``created``, ``order``, ``death``, ``release``...)
  is never taken for a birth date, nor is one whose name announces an occurrence (``first``,
  ``last``, ``appearance``, ``since``, ``term``...).

Age is computed row by row at the date of another event column of the table when there is one
(age at admission, at order, at death), else on the day of the analysis - which the report says,
since an old extract then gives today's ages.

Checks on a date of birth: births after the reference (negative ages), ages above 120, an excess
of 1 January or of first days of the month (day or month unknown, recorded as 01: one-sided
binomial test against the uniform calendar share), agreement with an ``age`` column when the table
has one (a constant gap dates the extract), the share of minors, and the fact that a date of birth
is personal data and a strong quasi-identifier.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from scipy import stats

from ._utils import quiet
from .types import DATETIME_TYPES, NUMERIC_TYPES, as_datetime, as_numeric

_TOKENS = re.compile(r"[^0-9a-zà-ÿ]+")
BIRTH_NAME = re.compile(
    r"(^|_)(dob|ddn|bday|born|birth\w*|\w+birth|birthdate|date_?of_?birth|naiss\w*|naissance|geburt\w*|"
    r"nacimiento|nacim\w*|fecha_?nac\w*|nascita|data_?nasc\w*|rodne\w*|narozen\w*)(_|$)")
EVENT_NAME = re.compile(
    r"(^|_)(death|deces|décès|died|dod|created?|creation|updated?|modified|modif\w*|registr\w*|inscri\w*|order\w*|"
    r"commande|transac\w*|payment|paiement|paid|ship\w*|deliver\w*|livr\w*|start\w*|debut|début|end|fin|expir\w*|due|"
    r"issued?|open\w*|ouverture|clos\w*|release\w*|sortie|hire\w*|hired|embauche|join\w*|signup|login|event|visit\w*|"
    r"admission|admit\w*|discharge\w*|sale|vente|purchase\w*|achat|valid\w*|launch\w*|publi\w*|posted|sent|received|"
    r"first|last|appear\w*|since|until|award\w*|elect\w*|term\w*|season|premiere|aired?)"
    r"(_|$)")
PERSON_NAME = re.compile(
    r"(^|_)(sex|sexe|gender|genre|first_?name|firstname|last_?name|lastname|surname|given_?name|prenom|prénom|"
    r"patient|client|customer|employee|employe|salarie|member|person|people|user|student|eleve|age|marital|"
    r"married|nationality|nationalite|citizen)(_|$)")
_AGE_NAME = re.compile(r"(^|_)(age|âge|age_?years|ans)($|_)")
DAYS_PER_YEAR = 365.2425


def tokens(name: Any) -> str:
    """``"DateOfBirth"`` -> ``"date_of_birth"``; accents kept, every other separator becomes ``_``."""
    text = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).lower()
    return _TOKENS.sub("_", text).strip("_")


def is_birth_name(name: Any) -> bool:
    return bool(BIRTH_NAME.search(tokens(name)))


def is_event_name(name: Any) -> bool:
    return bool(EVENT_NAME.search(tokens(name))) and not is_birth_name(name)


def _reference(df: pd.DataFrame, parsed: dict[str, pd.Series], birth: str,
               types: dict[str, str] | None = None) -> tuple[pd.Series | pd.Timestamp, str]:
    """The date each age is computed at: the best-filled other event date of the same row (a date
    column, or a year column such as an award year), else today."""
    from .timeseries import _time_columns  # year and year + month columns as dates

    candidates = {c: d for c, d in parsed.items() if c != birth and not is_birth_name(c)}
    with quiet():
        years = {c: d for c, d in _time_columns(df, types or {}).items() if c not in candidates and c != birth}
    candidates.update(years)
    events = [c for c, d in candidates.items() if d.notna().mean() >= 0.5]
    if events:
        best = max(events, key=lambda c: candidates[c].notna().sum())
        precision = " (a year only: ages within one year)" if best in years and "+" not in best else ""
        return candidates[best], f"at {best} (row by row){precision}"
    today = pd.Timestamp.today().normalize()
    return today, f"on {today:%Y-%m-%d} (the table has no other date: an old extract gives today's ages)"


def ages(births: pd.Series, reference: pd.Series | pd.Timestamp) -> pd.Series:
    return (reference - births).dt.days / DAYS_PER_YEAR


def _evidence(df: pd.DataFrame, col: str, d: pd.Series, age: pd.Series, person: bool) -> dict[str, Any] | None:
    """Decide whether ``col`` is a date of birth; None when it is not (or cannot be told)."""
    ok = d.dropna()
    a = age.dropna()
    if len(ok) < 10 or len(a) < 10:
        return None
    q = a.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    future = float((a < 0).mean())
    facts = f"median age {q[0.5]:.0f}, {ok.min():%Y} to {ok.max():%Y}"
    if is_birth_name(col):
        if -1 <= q[0.5] <= 110 and future <= 0.05:
            return {"certainty": "certain", "reason": f"named like a date of birth, and the ages are plausible ({facts})"}
        return {"rejected": f"named like a date of birth, but the ages are not plausible ({facts}, "
                            f"{100 * future:.0f}% in the future)"}
    if is_event_name(col) or not person:
        return None
    no_time = float((ok == ok.dt.normalize()).mean()) >= 0.99
    to_the_day = float((ok.dt.day == 1).mean()) < 0.5  # births carry their day; month-level dates sit on the 1st
    in_order = ok.is_monotonic_increasing or ok.is_monotonic_decreasing
    plausible = (len(ok) >= 30 and no_time and to_the_day and not in_order and future <= 0.01 and q[0.01] >= 0
                 and q[0.99] <= 110 and 18 <= q[0.5] <= 85 and q[0.95] - q[0.05] >= 20)
    if not plausible:
        return None
    return {"certainty": "probable",
            "reason": f"the values look like dates of birth in a table about people ({facts}, no time of day, "
                      "not sorted); the name does not say so"}


def birth_columns(df: pd.DataFrame, types: dict[str, str]) -> dict[str, dict[str, Any]]:
    """The date columns that are dates of birth, with the evidence; rejected candidates carry ``rejected``."""
    dates = [c for c in df.columns if types.get(c) in DATETIME_TYPES]
    if not dates:
        return {}
    with quiet():
        parsed = {c: as_datetime(df[c]) for c in dates}
    person = any(PERSON_NAME.search(tokens(c)) for c in df.columns)
    out = {}
    for col in dates:
        if is_event_name(col) and not is_birth_name(col):
            continue
        reference, _ = _reference(df, parsed, col, types)
        with quiet():
            evidence = _evidence(df, col, parsed[col], ages(parsed[col], reference), person)
        if evidence is not None:
            out[col] = evidence
    return out


def _binomial_excess(k: int, n: int, p0: float) -> tuple[float, float]:
    """Ratio observed / expected, and the one-sided binomial p-value of an excess."""
    if n == 0:
        return 0.0, 1.0
    return (k / n) / p0, float(stats.binomtest(k, n, p0, alternative="greater").pvalue)


def birth_report(df: pd.DataFrame, col: str, types: dict[str, str], alpha: float = 0.05) -> dict[str, Any]:
    """Ages and checks of one date-of-birth column."""
    with quiet():
        parsed = {c: as_datetime(df[c]) for c in df.columns if types.get(c) in DATETIME_TYPES}
        reference, ref_text = _reference(df, parsed, col, types)
        d = parsed[col]
        age = ages(d, reference)
    ok = d.dropna()
    a = age.dropna()
    out: dict[str, Any] = {"reference": ref_text, "n": int(len(a)), "checks": []}
    if a.empty:
        return out
    out.update(age=age, median_age=float(a.median()), min_age=float(a.min()), max_age=float(a.max()))
    checks = out["checks"]
    neg, old = int((a < 0).sum()), int((a > 120).sum())
    if neg:
        checks.append(("medium", f"{neg:,} births after the reference date (negative ages {ref_text})"))
    if old and not isinstance(reference, pd.Timestamp):
        checks.append(("medium", f"{old:,} ages above 120 {ref_text}: impossible dates"))
    elif old:
        checks.append(("info", f"{old:,} people would be over 120 today: the table includes people no longer alive "
                               "(compute the age at an event of theirs), or these dates are wrong"))
    n = len(ok)
    jan1 = int(((ok.dt.month == 1) & (ok.dt.day == 1)).sum())
    ratio, p = _binomial_excess(jan1, n, 1 / DAYS_PER_YEAR)
    if jan1 >= 5 and ratio >= 3 and p < alpha:
        checks.append(("medium", f"{jan1:,} births on 1 January, {ratio:.0f}x the calendar share (binomial p "
                                 f"{p:.1e}): day and month probably unknown and recorded as 01-01"))
    first = int(((ok.dt.day == 1) & (ok.dt.month != 1)).sum())
    ratio, p = _binomial_excess(first, n, 11 / DAYS_PER_YEAR)
    if first >= 10 and ratio >= 3 and p < alpha:
        checks.append(("info", f"{first:,} births on the 1st of a month, {ratio:.1f}x the calendar share (binomial p "
                               f"{p:.1e}): the day is probably often unknown"))
    minors = float((a < 18).mean())
    if minors >= 0.01:
        checks.append(("info", f"{100 * minors:.1f}% of rows are minors {ref_text}"))
    for other in df.columns:
        if other == col or types.get(other) not in NUMERIC_TYPES or not _AGE_NAME.search(tokens(other)):
            continue
        stated = as_numeric(df[other], types.get(other))
        gap = (stated - age).dropna()
        if len(gap) < 10:
            continue
        offset, spread = float(gap.median()), float(gap.std())
        bad = float((gap.abs() > 1.5).mean())
        if spread <= 1.5 and abs(offset) > 1.5:
            year = (pd.Timestamp.today() + pd.Timedelta(days=offset * DAYS_PER_YEAR)).year if isinstance(
                reference, pd.Timestamp) else None
            checks.append(("info", f"'{other}' agrees with {col} up to a constant {offset:+.1f} years"
                                   + (f": the ages were probably computed around {year}" if year else "")))
        elif bad > 0.05:
            checks.append(("medium", f"'{other}' disagrees with {col} by more than 1.5 years on {100 * bad:.0f}% of rows"))
    checks.append(("info", "a date of birth is personal data and a strong quasi-identifier (with a postal code and "
                           "the sex it singles out most people): keep only the age or an age band when sharing"))
    return out


def age_column_name(col: str, columns: list[str]) -> str:
    name = f"{col}_age"
    while name in columns:
        name += "_"
    return name


__all__ = ["BIRTH_NAME", "EVENT_NAME", "age_column_name", "ages", "birth_columns", "birth_report", "is_birth_name",
           "is_event_name", "tokens"]
