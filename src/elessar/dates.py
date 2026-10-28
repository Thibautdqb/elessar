"""Recognise dates whatever the way they are written, and parse them with the format found.

Handled, as text or as numbers:

- ISO 8601 and its variants, with or without time, fractions and time zones;
- day-first / month-first with any separator (``05/01/2020``, ``5.1.20``, ``2020/1/5``): the order is
  the one that parses every value (ambiguous columns where both work are reported as such);
- month names and abbreviations in English, French, German, Spanish, Italian, Portuguese, Dutch
  and Czech (``5 janvier 2020``, ``Jan 5, 2020``, ``März 2021``);
- compact digits: ``YYYYMMDD``, ``YYMMDD``, ``DDMMYYYY``, ``YYYYMM``, ``YYYYMMDDHHMM(SS)``, including the
  Czech / Slovak birth number where 50 is added to the month for women (``YYMMDD`` / ``YY(MM+50)DD``);
- quarters and ISO weeks (``2020Q1``, ``Q1 2020``, ``2020-W05``), year-months (``2020-01``, ``01/2020``);
- Unix timestamps in seconds, milliseconds, microseconds or nanoseconds;
- Excel serial day numbers (only when the column name says it is a date).

Two-digit years never land in the future: ``45`` is 1945, not 2045.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_MONTHS = {
    # French
    "janvier": "January", "janv": "Jan", "février": "February", "fevrier": "February", "févr": "Feb", "fevr": "Feb",
    "mars": "March", "avril": "April", "avr": "Apr", "mai": "May", "juin": "June", "juillet": "July", "juil": "Jul",
    "août": "August", "aout": "August", "septembre": "September", "sept": "Sep", "octobre": "October",
    "novembre": "November", "décembre": "December", "decembre": "December", "déc": "Dec",
    # German
    "januar": "January", "jänner": "January", "februar": "February", "märz": "March", "maerz": "March", "mär": "Mar",
    "juni": "June", "juli": "July", "oktober": "October", "okt": "Oct", "dezember": "December", "dez": "Dec",
    # Spanish
    "enero": "January", "ene": "Jan", "febrero": "February", "marzo": "March", "abril": "April", "mayo": "May",
    "junio": "June", "julio": "July", "agosto": "August", "ago": "Aug", "septiembre": "September", "setiembre": "September",
    "octubre": "October", "noviembre": "November", "diciembre": "December", "dic": "Dec",
    # Italian
    "gennaio": "January", "gen": "Jan", "febbraio": "February", "aprile": "April", "maggio": "May", "mag": "May",
    "giugno": "June", "giu": "Jun", "luglio": "July", "lug": "Jul", "settembre": "September", "set": "Sep",
    "ottobre": "October", "ott": "Oct", "dicembre": "December",
    # Portuguese
    "janeiro": "January", "fevereiro": "February", "março": "March", "marco": "March", "maio": "May", "junho": "June",
    "julho": "July", "setembro": "September", "outubro": "October", "out": "Oct", "dezembro": "December",
    # Dutch
    "januari": "January", "februari": "February", "maart": "March", "mei": "May", "augustus": "August",
    # Czech
    "leden": "January", "únor": "February", "unor": "February", "březen": "March", "brezen": "March", "duben": "April",
    "květen": "May", "kveten": "May", "červen": "June", "cerven": "June", "červenec": "July", "cervenec": "July",
    "srpen": "August", "září": "September", "zari": "September", "říjen": "October", "rijen": "October",
    "listopad": "November", "prosinec": "December",
    "ledna": "January", "února": "February", "unora": "February", "března": "March", "brezna": "March",
    "dubna": "April", "května": "May", "kvetna": "May", "června": "June", "cervna": "June", "července": "July",
    "cervence": "July", "srpna": "August", "října": "October", "rijna": "October", "listopadu": "November",
    "prosince": "December",
}
_MONTH_RX = re.compile(r"(?<![^\W\d_])(" + "|".join(sorted(map(re.escape, _MONTHS), key=len, reverse=True)) + r")\.?(?![^\W\d_])",
                       re.IGNORECASE)
_HINT = re.compile(r"date|time|day|jour|datum|fecha|data|_at$|^at$|_on$|created|updated|modified|timestamp|^ts$|_ts$|"
                   r"epoch|born|birth|naiss|dob|expir|start|end|debut|fin|when|period", re.I)
_QUARTER = [re.compile(r"^(?P<y>\d{4})\s*[-/ ]?\s*Q(?P<q>[1-4])$", re.I), re.compile(r"^Q(?P<q>[1-4])\s*[-/ ]?\s*(?P<y>\d{4})$", re.I),
            re.compile(r"^(?P<y>\d{4})\s*[-/ ]?\s*T(?P<q>[1-4])$", re.I)]
_WEEK = re.compile(r"^(?P<y>\d{4})-?W(?P<w>\d{1,2})(?:-?(?P<d>[1-7]))?$", re.I)
_EPOCH = {"s": (1e8, 4.2e9), "ms": (1e11, 4.2e12), "us": (1e14, 4.2e15), "ns": (1e17, 4.2e18)}
_COMPACT = {  # digits -> candidate formats, most usual first
    8: ["%Y%m%d", "%d%m%Y", "%m%d%Y"],
    6: ["%y%m%d", "%Y%m", "%d%m%y", "%m%d%y"],
    12: ["%Y%m%d%H%M"],
    14: ["%Y%m%d%H%M%S"],
}
_SEPARATED = ["%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y", "%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y",
              "%Y/%m/%d", "%Y.%m.%d", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
              "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S", "%m/%Y", "%m-%Y", "%Y/%m", "%m.%Y"]
_LABELS = {"%Y%m%d": "YYYYMMDD", "%y%m%d": "YYMMDD", "%d%m%Y": "DDMMYYYY", "%m%d%Y": "MMDDYYYY", "%d%m%y": "DDMMYY",
           "%m%d%y": "MMDDYY", "%Y%m": "YYYYMM", "%Y%m%d%H%M": "YYYYMMDDHHMM", "%Y%m%d%H%M%S": "YYYYMMDDHHMMSS"}


@dataclass
class DateFormat:
    """How a column writes its dates."""

    kind: str  # "format", "compact", "birth_number", "epoch", "excel", "quarter", "week", "mixed", "iso"
    format: str | None = None
    unit: str | None = None
    ratio: float = 0.0
    ambiguous: list[str] = field(default_factory=list)
    translated: bool = False

    @property
    def label(self) -> str:
        if self.kind == "epoch":
            return f"Unix timestamps in {self.unit}"
        if self.kind == "excel":
            return "Excel serial day numbers"
        if self.kind == "birth_number":
            return "Czech / Slovak birth numbers (YYMMDD, month + 50 for women)"
        if self.kind == "quarter":
            return "quarters (2020Q1)"
        if self.kind == "week":
            return "ISO weeks (2020-W05)"
        if self.kind == "compact":
            return f"compact {_LABELS.get(self.format or '', self.format)}"
        if self.kind == "format":
            return "dates in " + (self.format or "").replace("%Y", "YYYY").replace("%y", "YY").replace("%m", "MM").replace(
                "%d", "DD").replace("%H", "hh").replace("%M", "mm").replace("%S", "ss")
        return "dates" + (" with month names" if self.translated else "")

    def code(self, col: str) -> str:
        """pandas code that converts the column."""
        c = repr(col)
        if self.kind == "epoch":
            return f"df[{c}] = pd.to_datetime(df[{c}], unit={self.unit!r}, errors='coerce')"
        if self.kind == "excel":
            return f"df[{c}] = pd.to_datetime(df[{c}], unit='D', origin='1899-12-30', errors='coerce')"
        if self.kind in ("compact", "format") and "%y" not in (self.format or ""):
            src = f"df[{c}].astype('Int64').astype(str)" if self.kind == "compact" else f"df[{c}]"
            return f"df[{c}] = pd.to_datetime({src}, format={self.format!r}, errors='coerce')"
        return f"df[{c}] = elessar.parse_dates(df[{c}])"


def _strings(values: pd.Series) -> pd.Series:
    """Values as stripped strings; integral floats lose their '.0' (930101.0 -> '930101')."""
    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
        v = pd.to_numeric(values, errors="coerce")
        whole = v.dropna()
        if len(whole) and (whole % 1 == 0).all():
            return v.dropna().astype("int64").astype(str)
        return v.dropna().astype(str)
    return values.dropna().astype(str).str.strip()


def _translate(t: pd.Series) -> tuple[pd.Series, bool]:
    if not t.str.contains(r"[^\W\d_]{3,}", regex=True).any():
        return t, False
    out = t.str.replace(_MONTH_RX, lambda m: _MONTHS[m.group(1).lower()], regex=True)
    return out, bool((out != t).any())


def _fix_century(d: pd.Series, past_only: bool = False) -> pd.Series:
    """Two-digit years: one century split for the whole column, the one that keeps the dates the most
    tightly grouped without putting any in the future (more than a year ahead; at all for birth dates).

    1911-1987 birth years stay in the 1900s; '95' and '20' in recent data become 1995 and 2020.
    """
    ok = d.dropna()
    if ok.empty:
        return d
    limit = pd.Timestamp.now() + pd.Timedelta(days=0 if past_only else 366)
    yy = np.unique(ok.dt.year.to_numpy() % 100)
    best, best_span = None, None
    for pivot in range(0, 101):  # yy >= pivot -> 1900s, below -> 2000s
        years = np.where(yy >= pivot, 1900 + yy, 2000 + yy)
        if years.max() > limit.year:
            continue
        span = years.max() - years.min()
        if best_span is None or span < best_span or (span == best_span and pivot > best):
            best, best_span = pivot, span
    if best is None:
        best = 0
    shift = np.where(d.dt.year % 100 >= best, 1900, 2000) + d.dt.year % 100 - d.dt.year
    out = d + pd.to_timedelta(0, unit="D")
    for delta in np.unique(shift.dropna()):
        mask = shift == delta
        out[mask] = d[mask] + pd.DateOffset(years=int(delta))
    return out.where(out <= limit, out - pd.DateOffset(years=100))


def _plausible(d: pd.Series) -> bool:
    ok = d.dropna()
    return bool(len(ok)) and ok.min() >= pd.Timestamp("1800-01-01") and ok.max() <= pd.Timestamp("2200-12-31")


def _parse_format(t: pd.Series, fmt: str) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(t, format=fmt, errors="coerce")
    return _fix_century(d) if "%y" in fmt else d


def _birth_number(t: pd.Series) -> pd.Series:
    digits = t.str.extract(r"^(\d{2})(\d{2})(\d{2})$")
    month = pd.to_numeric(digits[1], errors="coerce")
    month = month.where(month <= 12, month - 50)
    text = digits[0] + month.astype("Int64").astype("string").str.zfill(2) + digits[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = pd.to_datetime(text, format="%y%m%d", errors="coerce")
    return _fix_century(d, past_only=True)  # people are born in the past


def _quarters(t: pd.Series) -> pd.Series:
    out = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns]")
    for rx in _QUARTER:
        m = t.str.extract(rx)
        ok = m["y"].notna()
        if ok.any():
            out[ok] = pd.to_datetime(m.loc[ok, "y"] + "-" + ((m.loc[ok, "q"].astype(int) - 1) * 3 + 1).astype(str) + "-01",
                                     errors="coerce")
    return out


def _weeks(t: pd.Series) -> pd.Series:
    m = t.str.extract(_WEEK)
    text = m["y"] + "-W" + m["w"].str.zfill(2) + "-" + m["d"].fillna("1")
    return pd.to_datetime(text, format="%G-W%V-%u", errors="coerce")


def _ratio(parsed: pd.Series, n: int) -> float:
    return float(parsed.notna().sum()) / n if n else 0.0


_PLACEHOLDERS = {"", "-", "--", "?", ".", "na", "n/a", "nan", "null", "none", "nat", "0", "00000000", "000000"}


def detect(values: pd.Series, name: Any = None, min_ratio: float = 0.9, sample: int = 3_000) -> DateFormat | None:
    """The date format of a column, or None when it does not hold dates.

    ``min_ratio`` of the non-placeholder values must parse; numbers are only read as epochs or
    compact dates when the values make it unambiguous (or the column name says it is a date).
    """
    s = values.dropna()
    if s.empty or pd.api.types.is_bool_dtype(s.dtype):
        return None
    if len(s) > sample:
        s = s.sample(sample, random_state=0)
    hint = bool(_HINT.search(str(name or "")))
    numeric = pd.api.types.is_numeric_dtype(s.dtype)
    t = _strings(s)
    t = t[~t.str.lower().isin(_PLACEHOLDERS)]
    n = len(t)
    if n < 3 or t.nunique() < 2:
        return None

    if numeric:
        v = pd.to_numeric(s, errors="coerce").dropna()
        for unit, (lo, hi) in _EPOCH.items():
            inside = float(v.between(lo, hi).mean())
            if inside >= 0.99 and (unit != "s" or hint) and v.nunique() > 1:
                return DateFormat("epoch", unit=unit, ratio=inside)
        if hint and float(v.between(18_000, 73_000).mean()) >= 0.99 and v.nunique() > 1:  # 1949 .. 2099
            return DateFormat("excel", ratio=1.0)

    digits = t.str.fullmatch(r"\d+")
    if float(digits.mean()) >= 0.99:
        lengths = t.str.len()
        length = int(lengths.mode().iloc[0])
        if float((lengths == length).mean()) >= 0.98:
            best: DateFormat | None = None
            fits = []
            for fmt in _COMPACT.get(length, []):
                d = _parse_format(t, fmt)
                r = _ratio(d, n)
                if r >= min_ratio and _plausible(d) and d.dt.normalize().nunique() >= 2:
                    fits.append(fmt)
                    if best is None or r > best.ratio + 0.01:
                        best = DateFormat("compact", format=fmt, ratio=r)
            if best is not None:
                best.ambiguous = [f for f in fits if f != best.format]
                return best
            if length == 6:
                d = _birth_number(t)
                r = _ratio(d, n)
                if r >= min_ratio and _plausible(d):
                    return DateFormat("birth_number", format="%y%m%d", ratio=r)
        return None  # other plain numbers are numbers
    if numeric:
        return None

    if not t.str.contains(r"\d", regex=True).mean() >= min_ratio:
        return None
    for kind, parser in (("quarter", _quarters), ("week", _weeks)):
        r = _ratio(parser(t), n)
        if r >= min_ratio:
            return DateFormat(kind, ratio=r)
    text, translated = _translate(t)
    # explicit separated formats first: they decide day-first vs month-first
    fits = []
    for fmt in _SEPARATED:
        d = _parse_format(text, fmt)
        r = _ratio(d, n)
        if r >= min_ratio and _plausible(d):
            fits.append((r, fmt))
    if fits:
        top = max(r for r, _ in fits)
        best_fmts = [f for r, f in fits if r >= top - 1e-9]
        return DateFormat("format", format=best_fmts[0], ratio=top, ambiguous=best_fmts[1:], translated=translated)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = pd.to_datetime(text, format="ISO8601", errors="coerce", utc=True)
        r = _ratio(iso, n)
        if r >= min_ratio and _plausible(iso.dt.tz_localize(None)):
            return DateFormat("iso", ratio=r, translated=translated)
        mixed = pd.to_datetime(text, format="mixed", errors="coerce", utc=True)
    long_enough = text.str.len() >= 6  # "T1", "Q3", "12" are not dates
    r = _ratio(mixed.where(long_enough), n)
    if r >= min_ratio and _plausible(mixed.dt.tz_localize(None)):
        return DateFormat("mixed", ratio=r, translated=translated)
    return None


def parse_dates(values: pd.Series, fmt: DateFormat | None = None, name: Any = None) -> pd.Series:
    """Parse a column of dates written in any supported way (naive datetimes; unparsable -> NaT)."""
    s = values
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        out = s
    else:
        label = name if name is not None else s.name
        fmt = fmt or detect(s, label) or detect(s, label, min_ratio=0.5)  # the strict reading first
        if fmt is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = pd.to_datetime(s.astype("object"), errors="coerce", format="mixed", utc=True)
        else:
            out = _parse_with(s, fmt)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out


def _parse_with(s: pd.Series, fmt: DateFormat) -> pd.Series:
    full = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    ok = s.notna()
    if not ok.any():
        return full
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt.kind == "epoch":
            out = pd.to_datetime(pd.to_numeric(s[ok], errors="coerce"), unit=fmt.unit, errors="coerce")
        elif fmt.kind == "excel":
            out = pd.to_datetime(pd.to_numeric(s[ok], errors="coerce"), unit="D", origin="1899-12-30", errors="coerce")
        else:
            t = _strings(s[ok])
            if fmt.kind == "compact":
                out = _parse_format(t, fmt.format)
            elif fmt.kind == "birth_number":
                out = _birth_number(t)
            elif fmt.kind == "quarter":
                out = _quarters(t)
            elif fmt.kind == "week":
                out = _weeks(t)
            else:
                text = _translate(t)[0] if fmt.translated else t
                if fmt.kind == "format":
                    out = _parse_format(text, fmt.format)
                else:
                    out = pd.to_datetime(text, format="ISO8601" if fmt.kind == "iso" else "mixed", errors="coerce",
                                         utc=True)
                    if getattr(out.dt, "tz", None) is not None:
                        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    out = pd.Series(np.asarray(out, dtype="datetime64[ns]"), index=out.index)
    full.loc[out.index] = out
    return full


__all__ = ["DateFormat", "detect", "parse_dates"]
