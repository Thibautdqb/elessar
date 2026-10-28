"""Semantic type inference: what a column *means*, not just how pandas stores it."""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from ._utils import hashable, is_text_dtype, prepare, quiet, sample, truncate
from .config import StoneConfig
from .dates import _HINT as DATE_HINT, detect as detect_date, parse_dates

SEMANTIC_TYPES: tuple[str, ...] = (
    "empty", "constant", "boolean", "id", "numeric_continuous", "numeric_discrete",
    "categorical", "ordinal-candidate", "datetime", "datetime_string", "numeric_string",
    "text", "email", "url", "phone", "postal_code", "ip_address", "json_like", "mixed", "multi_label",
)
NUMERIC_TYPES = frozenset({"numeric_continuous", "numeric_discrete"})
CATEGORY_TYPES = frozenset({"categorical", "boolean", "ordinal-candidate"})
MULTI_LABEL = "multi_label"
DATETIME_TYPES = frozenset({"datetime", "datetime_string"})
STRING_PATTERN_TYPES = frozenset({"email", "url", "phone", "postal_code", "ip_address"})

TRUE_TOKENS = frozenset({"true", "t", "yes", "y", "1", "1.0", "oui", "o", "on"})
FALSE_TOKENS = frozenset({"false", "f", "no", "n", "0", "0.0", "non", "off"})

ORDINAL_SCALES: tuple[tuple[str, ...], ...] = (
    ("very low", "low", "medium", "high", "very high"),
    ("low", "mid", "high"),
    ("xs", "s", "m", "l", "xl", "xxl"),
    ("small", "medium", "large", "extra large"),
    ("never", "rarely", "sometimes", "often", "always"),
    ("very poor", "poor", "fair", "average", "good", "very good", "excellent"),
    ("bad", "average", "good"),
    ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    ("cold", "warm", "hot"),
    ("beginner", "intermediate", "advanced", "expert"),
    ("none", "mild", "moderate", "severe"),
    ("first", "second", "third", "fourth"),
    ("bronze", "silver", "gold", "platinum"),
    ("fair", "good", "very good", "premium", "ideal"),
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"),
    ("mon", "tue", "wed", "thu", "fri", "sat", "sun"),
    ("mon", "tues", "wed", "thur", "fri", "sat", "sun"),
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
    ("q1", "q2", "q3", "q4"),
)

_EMAIL = re.compile(r"^[\w.+'-]+@[\w-]+(\.[\w-]+)*\.[a-z]{2,}$", re.I)
_URL = re.compile(r"^(?:(?:https?|ftp)://[^\s/$.?#][^\s]*|www\.[^\s]+\.[a-z]{2,}\S*|//[\w-]+(?:\.[\w-]+)+/\S*)$", re.I)
_UUID = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$", re.I)
_PHONE = re.compile(r"^\+?[\d\s().-]{6,24}$")
_POSTAL_STRICT = re.compile(
    r"^(?:\d{5}-\d{4}"  # US ZIP+4
    r"|[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}"  # UK
    r"|[A-Z]\d[A-Z] ?\d[A-Z]\d"  # Canada
    r"|\d{4} ?[A-Z]{2})$",  # Netherlands
    re.I,
)
_POSTAL_DIGITS = re.compile(r"^\d{4,6}$")
_PLAIN_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_GROUPED_NUM = re.compile(r"[+-]?\d{1,3}(?P<sep>[ ,.'’])\d{3}(?:(?P=sep)\d{3})*(?:(?P<dec>[.,])\d+)?")
_COMMA_DEC = re.compile(r"[+-]?\d+,\d+")
_CURRENCY = re.compile(r"[$€£¥₹]|\b(?:usd|eur|gbp)\b", re.I)

_HINT_ID = re.compile(r"(^|[_\s.-])(id|uuid|guid|pk)($|[_\s.-])|^#$|_key$|^key_|_(number|code|no|num)$|"
                      r"^(account|acct|iban|bic|swift|siren|siret|ean|isbn|sku|imei)(_?(no|num|number|to|from))?$|"
                      r"_(account|acct|iban)$", re.I)
_UNIT_NUM = re.compile(r"(?P<num>[+-]?\d[\d,.]*)(?P<space>\s*)(?P<unit>[°µ]?[^\W\d_][^\d]{0,14}?)\.?")
_NOT_UNITS = frozenset({"st", "nd", "rd", "th", "e", "x"})  # ordinals, exponents, multipliers
LIST_SEPARATORS: tuple[str, ...] = (", ", ",", "|", ";", " / ")
_HINT_PHONE = re.compile(r"phone|tel|mobile|cell|fax")
_HINT_POSTAL = re.compile(r"zip|postal|post_?code|postcode|code_?postal|(^|_)cp($|_)|plz")


def _name_tokens(name: Any) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name)).lower()


def parse_number(value: Any) -> float:
    """Parse one number written as text: ``"1 234,5"``, ``"12%"``, ``"$3.2"``, ``"1,234.56"``.

    Returns NaN when the value is not a number.
    """
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if not isinstance(value, str):
        return np.nan
    t = value.strip().replace(" ", " ").replace(" ", " ")
    t = _CURRENCY.sub("", t).strip()
    if t.endswith("%"):
        t = t[:-1].strip()
    if t.startswith("(") and t.endswith(")"):  # accounting negatives
        t = "-" + t[1:-1].strip()
    if _PLAIN_NUM.fullmatch(t):
        return float(t)
    m = _GROUPED_NUM.fullmatch(t)
    if m and m.group("dec") != m.group("sep"):
        num = t.replace(m.group("sep"), "")
        return float(num.replace(m.group("dec"), ".") if m.group("dec") else num)
    if _COMMA_DEC.fullmatch(t):
        return float(t.replace(",", "."))
    number, _ = split_unit(t)
    return number


def normalize_unit(unit: str) -> str:
    """``"Seasons"`` -> ``"season"``, ``"mins"`` -> ``"min"``, ``"KG"`` -> ``"kg"`` (plural ``s`` dropped past 3 letters)."""
    u = unit.strip().rstrip(".").lower()
    return u[:-1] if len(u) > 3 and u.endswith("s") and not u.endswith("ss") else u


def split_unit(value: Any) -> tuple[float, str | None]:
    """``"90 min"`` -> ``(90.0, "min")``; anything else -> ``(nan, None)``."""
    if not isinstance(value, str):
        return np.nan, None
    m = _UNIT_NUM.fullmatch(value.strip())
    if not m:
        return np.nan, None
    unit = normalize_unit(m.group("unit"))
    if unit in _NOT_UNITS or (len(unit) == 1 and not m.group("space")):  # "5A" is a code, "100 m" a length
        return np.nan, None
    number = parse_number(m.group("num"))
    return (number, unit) if np.isfinite(number) else (np.nan, None)


def unit_counts(s: pd.Series) -> dict[str, int]:
    """How many values carry each (normalized) unit, most frequent first."""
    counts: dict[str, int] = {}
    for value, n in hashable(s.dropna()).value_counts().items():
        unit = split_unit(value)[1]
        if unit is not None:
            counts[unit] = counts.get(unit, 0) + int(n)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def parse_numeric(s: pd.Series, dominant_unit_only: bool = True) -> pd.Series:
    """Vectorised :func:`parse_number` over a Series (parses each distinct value once).

    Numbers written with units (``"90 min"``) are parsed too. When several units are mixed
    (``"90 min"`` and ``"2 Seasons"``), only values in the most frequent unit (or without unit)
    are kept unless ``dominant_unit_only`` is False: numbers in different units are not comparable.
    """
    s = hashable(s)
    uniques = s.dropna().unique()
    mapping = {u: parse_number(u) for u in uniques}
    if dominant_unit_only:
        units = unit_counts(s)
        if len(units) > 1:
            main = next(iter(units))
            mapping = {u: v if split_unit(u)[1] in (None, main) else np.nan for u, v in mapping.items()}
    return s.map(mapping).astype("float64")


def list_separator(t: pd.Series) -> str | None:
    """The separator of a multi-label column (``"Comedies, Dramas"``), or None.

    A real list has a varying number of short labels per row, labels recur across rows, and a
    label shows up both first and later in lists (order carries no meaning). That rejects composite
    values (``"Paris, France"``), names containing the separator (``"Washed / Wet"``,
    ``"Acme Co., Ltd"``: the second part is never first) and prose with commas.
    """
    for sep in LIST_SEPARATORS:
        has = t.str.contains(sep, regex=False)
        if has.mean() < 0.15:
            continue
        parts = t.str.split(sep, regex=False)
        counts = parts.str.len()
        if counts.nunique() < 2:
            continue
        labels = parts.explode().str.strip()
        labels = labels[labels != ""]
        if labels.str.split().str.len().mean() > 4 or labels.str.len().mean() > 40:
            continue
        freq = labels.map(labels.value_counts())
        if (freq >= 2).mean() >= 0.4 and _position_mixing(parts) >= 0.1:
            return sep
    return None


def _position_mixing(parts: pd.Series) -> float:
    """Share of recurring labels seen both first and later in their lists."""
    first, later, counts = set(), set(), {}
    for labels in parts:
        for i, label in enumerate(x.strip() for x in labels if x.strip()):
            (first if i == 0 else later).add(label)
            counts[label] = counts.get(label, 0) + 1
    recurring = [label for label, n in counts.items() if n >= 2]
    return sum(label in first and label in later for label in recurring) / max(len(recurring), 1)


def _bool_token(v: Any) -> bool | None:
    t = str(v).strip().lower()
    if t in TRUE_TOKENS:
        return True
    if t in FALSE_TOKENS:
        return False
    return None


def as_boolean(s: pd.Series) -> pd.Series:
    """Boolean view of a boolean-like column (``True/False/NaN`` as float 1/0/NaN)."""
    if pd.api.types.is_bool_dtype(s.dtype):
        return s.astype("float64")
    if pd.api.types.is_numeric_dtype(s.dtype):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    mapped = s.map(lambda v: _bool_token(v) if pd.notna(v) else None)
    return mapped.map({True: 1.0, False: 0.0}).astype("float64")


def as_numeric(s: pd.Series, semantic_type: str | None = None) -> pd.Series:
    """Float view of any numeric-like column (bool, timedelta, numeric strings, ...)."""
    if pd.api.types.is_bool_dtype(s.dtype) or semantic_type == "boolean":
        return as_boolean(s)
    if pd.api.types.is_timedelta64_dtype(s.dtype):
        return s.dt.total_seconds()
    if pd.api.types.is_numeric_dtype(s.dtype):
        return pd.Series(
            pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan), index=s.index, name=s.name
        )
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.Series(np.nan, index=s.index, name=s.name)
    return parse_numeric(s)


def as_datetime(s: pd.Series) -> pd.Series:
    """Datetime view whatever the way the dates are written (see :mod:`elessar.dates`); naive UTC."""
    with quiet():
        return parse_dates(hashable(s) if pd.api.types.is_object_dtype(s.dtype) else s)


def ordinal_order(values: Any) -> list[str] | None:
    """Return the values sorted along a known ordered scale (low < medium < high ...), else None."""
    vals = {str(v) for v in values}
    lowered = {v.strip().lower(): v for v in vals}
    for scale in ORDINAL_SCALES:
        if len(lowered) >= 2 and set(lowered) <= set(scale):
            return [lowered[k] for k in scale if k in lowered]
    return None


def _kind(v: Any) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "bool"
    if isinstance(v, (int, float, np.number)):
        return "number"
    if isinstance(v, str):
        return "str"
    if isinstance(v, (dict,)):
        return "dict"
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return "list"
    if isinstance(v, (dt.date, dt.datetime, pd.Timestamp, np.datetime64)):
        return "datetime"
    return type(v).__name__


def python_kinds(s: pd.Series) -> dict[str, int]:
    """Count Python value kinds in a column (``str``, ``number``, ``list``...)."""
    if not pd.api.types.is_object_dtype(s.dtype):
        return {}
    return {str(k): int(v) for k, v in s.dropna().map(_kind).value_counts().items()}


def _result(stype: str, s: pd.Series, confidence: float, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "semantic_type": stype,
        "pandas_dtype": str(s.dtype),
        "confidence": round(float(confidence), 3),
        "reason": reason,
        "suggestions": [],
        **extra,
    }


def _ratio(mask: pd.Series | np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def _numeric_type(x: pd.Series, s: pd.Series, name: str, cfg: StoneConfig, how: str) -> dict[str, Any]:
    """Classify a float view ``x`` (no NaN) of the original column ``s``."""
    finite = x[np.isfinite(x)]
    n = len(finite)
    if n == 0:
        return _result("numeric_continuous", s, 0.5, f"{how}: only infinite values")
    nunique = finite.nunique()
    is_int = bool(np.all(np.mod(finite, 1) == 0))
    uniq_ratio = nunique / n
    lo, hi = float(finite.min()), float(finite.max())
    if nunique == 2 and set(np.unique(finite)) <= {0.0, 1.0}:
        return _result("boolean", s, 0.9, f"{how}: only 0/1 values")
    hinted = bool(_HINT_ID.search(name))
    if (not hinted or DATE_HINT.search(name)) and nunique >= 2 and (is_int or lo >= 1e8):
        with quiet():
            fmt = detect_date(finite, name)
        if fmt is not None:
            return _result("datetime_string", s, fmt.ratio, f"{how}: dates stored as numbers ({fmt.label})", date_format=fmt)
    if is_int and hinted and uniq_ratio < cfg.id_unique_ratio:
        if uniq_ratio >= 0.8 and (hi - lo + 1) <= 1.5 * nunique:
            return _result("id", s, 0.7, f"{how}: name looks like an id, mostly unique consecutive integers")
        return _result("categorical", s, 0.7, f"{how}: repeated identifier ({nunique} ids), a grouping key")
    if is_int and uniq_ratio >= cfg.id_unique_ratio:
        consecutive = n >= 20 and (hi - lo + 1) <= 1.5 * nunique
        monotonic = n >= 20 and nunique == n and (finite.is_monotonic_increasing or finite.is_monotonic_decreasing)
        if hinted or consecutive or monotonic:
            why = "name looks like an id" if hinted else "unique, (nearly) consecutive integers"
            return _result("id", s, 0.9 if hinted else 0.8, f"{how}: {why}")
    consecutive_small = 2 <= nunique <= cfg.ordinal_max_int_levels and lo in (0.0, 1.0) and hi - lo + 1 == nunique
    if is_int and consecutive_small and n >= 20:
        skew = float(finite.skew())
        if abs(skew) < 1:  # counts (children, passengers...) are right-skewed; rating scales are not
            return _result(
                "ordinal-candidate", s, 0.6, f"{how}: {nunique} consecutive small integers ({lo:g}..{hi:g}), maybe a scale"
            )
        return _result("numeric_discrete", s, 0.8, f"{how}: small right-skewed integers ({lo:g}..{hi:g}), a count")
    if is_int:
        if nunique > cfg.discrete_max_levels:
            return _result("numeric_continuous", s, 0.7, f"{how}: integer-valued with many levels ({nunique})")
        return _result("numeric_discrete", s, 0.85, f"{how}: integer values ({nunique} distinct)")
    if nunique <= min(cfg.discrete_max_levels, 20) and uniq_ratio < 0.05:
        return _result("numeric_discrete", s, 0.7, f"{how}: only {nunique} distinct non-integer values")
    return _result("numeric_continuous", s, 0.9, f"{how}: {nunique} distinct real values")


def _string_type(strs: pd.Series, s: pd.Series, name: str, cfg: StoneConfig, n_nonnull: int, nunique: int) -> dict:
    """Classify a column whose non-null values are all strings."""
    t = strs.str.strip()
    r = cfg.pattern_match_ratio
    uniq_ratio = nunique / n_nonnull

    def match(rx: re.Pattern) -> float:
        return _ratio(t.str.match(rx).fillna(False).to_numpy(dtype=bool))

    lowered = set(t.str.lower().unique())
    if len(lowered) == 2 and any(v in TRUE_TOKENS for v in lowered) and any(v in FALSE_TOKENS for v in lowered):
        return _result("boolean", s, 0.95, f"two boolean-like labels: {sorted(lowered)}")

    starts = t.str[:1].isin(["{", "["])
    if _ratio(starts) >= r:
        ok = _ratio([_is_json(v) for v in t[starts]]) * _ratio(starts)
        if ok >= r:
            return _result("json_like", s, ok, f"{ok:.0%} of values parse as JSON")
    for stype, rx in (("email", _EMAIL), ("url", _URL)):
        m = match(rx)
        if m >= r:
            return _result(stype, s, m, f"{m:.0%} of values look like {stype}s")
    ip = _ratio([_is_ip(v) for v in t])
    if ip >= r:
        return _result("ip_address", s, ip, f"{ip:.0%} of values are IP addresses")
    m = match(_UUID)
    if m >= r:
        return _result("id", s, m, f"{m:.0%} of values are UUIDs")

    postal_hint = bool(_HINT_POSTAL.search(name))
    m = match(_POSTAL_STRICT)
    if m >= r or (postal_hint and _ratio(t.str.match(_POSTAL_STRICT) | t.str.match(_POSTAL_DIGITS)) >= r):
        return _result("postal_code", s, max(m, 0.8), "postal-code pattern" + (" and column name" if postal_hint else ""))

    if _ratio(t.str.fullmatch(r"\d+|-|\?|").fillna(False)) >= 0.99 and (not _HINT_ID.search(name) or DATE_HINT.search(name)):
        with quiet():
            fmt = detect_date(t, name)
        if fmt is not None:
            return _result("datetime_string", s, fmt.ratio, f"dates written as {fmt.label}", date_format=fmt)

    digits_only = t.str.fullmatch(r"\d+")
    leading_zero_code = _ratio(digits_only) >= r and _ratio(t.str.match(r"0\d")) >= 0.05 and t.str.len().nunique() == 1
    phone_hint = bool(_HINT_PHONE.search(name))
    num = parse_numeric(t, dominant_unit_only=False)
    m = _ratio(num.notna())
    if m >= r and not leading_zero_code and not phone_hint:
        units = unit_counts(t)
        reason = f"{m:.0%} of values are numbers stored as text"
        if units:
            total = sum(units.values())
            listed = ", ".join(f"{u} {100 * c / total:.0f}%" for u, c in list(units.items())[:4])
            reason = f"numbers with units ({listed})" + (": mixed units, not comparable" if len(units) > 1 else "")
        out = _result("numeric_string", s, m, reason, units=units)
        out["numeric"] = _numeric_type(parse_numeric(t).dropna(), s, name, cfg, "parsed numbers")["semantic_type"]
        return out

    has_digit = t.str.contains(r"\d", regex=True)
    if _ratio(has_digit) >= r and not leading_zero_code:
        with quiet():
            fmt = detect_date(t, name, min_ratio=r)
        if fmt is not None:
            amb = f"; also valid as {', '.join(fmt.ambiguous)}" if fmt.ambiguous else ""
            return _result("datetime_string", s, fmt.ratio, f"{fmt.ratio:.0%} of values are {fmt.label}{amb}",
                           date_format=fmt)

    digit_count = t.str.count(r"\d")
    phone_like = t.str.match(_PHONE) & digit_count.between(7, 15)
    phone_like &= t.str.contains(r"^\+|\d[\s().-]+\d", regex=True) | phone_hint
    m = _ratio(phone_like)
    if m >= r:
        return _result("phone", s, m, f"{m:.0%} of values look like phone numbers")

    sep = list_separator(t)
    if sep is not None:
        return _result(MULTI_LABEL, s, 0.8, f"lists of labels separated by {sep.strip() or repr(sep)!s} "
                       f"(e.g. {truncate(str(t[t.str.contains(sep, regex=False)].iloc[0]), 40)!r})", sep=sep)

    words = t.str.split().str.len()
    avg_words, avg_len = float(words.mean()), float(t.str.len().mean())
    if uniq_ratio >= cfg.id_unique_ratio and (n_nonnull >= 20 or _HINT_ID.search(name)):
        if avg_words <= 2.5 and avg_len <= 40:
            return _result("id", s, 0.8, f"{uniq_ratio:.0%} unique short labels (names / codes)")
    if (avg_words >= cfg.text_min_avg_words or avg_len >= cfg.text_min_avg_length) and uniq_ratio > 0.3:
        return _result("text", s, 0.85, f"free text (avg {avg_words:.1f} words, {avg_len:.0f} chars)")
    order = ordinal_order(t.unique()) if nunique <= cfg.ordinal_max_levels else None
    if order:
        return _result("ordinal-candidate", s, 0.75, f"ordered-looking labels: {' < '.join(order)}", order=order)
    if nunique <= cfg.max_categories or uniq_ratio < 0.5:
        return _result("categorical", s, 0.85, f"{nunique} distinct labels")
    if avg_words < 2:
        return _result("categorical", s, 0.6, f"high-cardinality codes ({nunique} distinct short labels)")
    return _result("text", s, 0.6, f"high-cardinality strings ({nunique} distinct)")


def _is_json(v: str) -> bool:
    try:
        return isinstance(json.loads(v), (dict, list))
    except (ValueError, TypeError):
        return False


def _is_ip(v: str) -> bool:
    try:
        ipaddress.ip_address(v)
    except ValueError:
        return False
    return True


def _suggest(info: dict[str, Any], s: pd.Series, name: str) -> list[dict[str, str]]:
    """Conversion suggestions (should-be-category / numeric / datetime / boolean)."""
    stype, col, out = info["semantic_type"], repr(name), []
    texty = is_text_dtype(s)
    units = info.get("units") or {}
    if stype == "numeric_string" and len(units) > 1:
        total = sum(units.values())
        out.append({
            "flag": "mixed-units",
            "note": ", ".join(f"{u} {100 * c / total:.0f}%" for u, c in list(units.items())[:4]),
            "code": f"df[[{name + '_value'!r}, {name + '_unit'!r}]] = df[{col}].str.extract(r'([\\d.,]+)\\s*(\\D+)')",
        })
    elif stype == "numeric_string":
        out.append({
            "flag": "should-be-numeric",
            "code": f"df[{col}] = elessar.types.parse_numeric(df[{col}])",
        })
    elif texty and stype in NUMERIC_TYPES:
        out.append({"flag": "should-be-numeric", "code": f"df[{col}] = pd.to_numeric(df[{col}], errors='coerce')"})
    if stype == "datetime_string":
        fmt = info.get("date_format")
        out.append({
            "flag": "should-be-datetime",
            "note": fmt.label if fmt is not None else "",
            "code": fmt.code(str(name)) if fmt is not None else f"df[{col}] = elessar.parse_dates(df[{col}])",
        })
    if stype == "boolean" and texty:
        labels = {str(v).strip().lower() for v in s.dropna().unique()}
        mapping = {v: v in TRUE_TOKENS for v in sorted(labels)}
        out.append({
            "flag": "should-be-boolean",
            "code": f"df[{col}] = df[{col}].str.strip().str.lower().map({mapping!r})",
        })
    if stype == MULTI_LABEL:
        out.append({"flag": "multi-label", "code": f"labels = df[{col}].str.get_dummies(sep={info['sep']!r})"})
    if stype in ("categorical", "ordinal-candidate") and texty:
        nonnull = max(int(s.notna().sum()), 1)
        if hashable(s).nunique() / nonnull <= 0.5:
            if info.get("order"):
                code = (
                    f"df[{col}] = pd.Categorical(df[{col}], categories={info['order']!r}, ordered=True)"
                )
            else:
                code = f"df[{col}] = df[{col}].astype('category')"
            out.append({"flag": "should-be-category", "code": code})
    return out


def infer_type(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Infer the semantic type of one column.

    Returns a dict with ``semantic_type``, ``pandas_dtype``, ``confidence`` (0-1),
    ``reason`` and ``suggestions`` (list of ``{flag, code}``).
    """
    cfg = config or StoneConfig()
    info = _infer(s, cfg)
    info["suggestions"] = _suggest(info, s, str(s.name))
    return info


def _infer(s: pd.Series, cfg: StoneConfig) -> dict[str, Any]:
    name = _name_tokens(s.name)
    nonnull = s.dropna()
    if len(nonnull) == 0:
        return _result("empty", s, 1.0, "no non-missing values")
    hs = hashable(nonnull)
    nunique = int(hs.nunique())
    if nunique == 1:
        return _result("constant", s, 1.0, f"single value: {str(hs.iloc[0])[:40]!r}")
    dtype = s.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return _result("boolean", s, 1.0, "bool dtype")
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return _result("datetime", s, 1.0, "datetime dtype")
    if isinstance(dtype, pd.CategoricalDtype):
        if dtype.ordered:
            return _result("ordinal-candidate", s, 0.95, "ordered categorical dtype", order=[str(c) for c in dtype.categories])
        sub = _infer(nonnull.astype(object), cfg)
        if sub["semantic_type"] in NUMERIC_TYPES | {"text", "id", "email", "url", "phone", "postal_code", "ip_address"}:
            sub["pandas_dtype"] = str(dtype)
            return sub
        return _result("categorical", s, 1.0, f"category dtype ({nunique} levels)")
    if pd.api.types.is_complex_dtype(dtype):
        return _result("mixed", s, 0.5, "complex numbers")
    if pd.api.types.is_timedelta64_dtype(dtype):
        out = _numeric_type(nonnull.dt.total_seconds(), s, name, cfg, "timedelta (seconds)")
        out["semantic_type"] = "numeric_continuous" if out["semantic_type"] in ("id", "boolean") else out["semantic_type"]
        return out
    if pd.api.types.is_numeric_dtype(dtype):
        return _numeric_type(as_numeric(nonnull), s, name, cfg, f"{dtype} dtype")

    if not is_text_dtype(s):
        return _result("categorical", s, 0.5, f"unrecognised dtype {dtype}, treated as labels")

    sampled = sample(nonnull, 5_000, cfg.random_state)
    kinds = python_kinds(sampled) if pd.api.types.is_object_dtype(dtype) else {"str": len(sampled)}
    if set(kinds) <= {"dict", "list"}:
        return _result("json_like", s, 1.0, "cells contain Python " + "/".join(sorted(kinds)) + " objects")
    if len(kinds) > 1 and not set(kinds) <= {"number", "bool"}:
        mix = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        return _result("mixed", s, 1.0, f"several Python types: {mix}", kinds=kinds)
    only = next(iter(kinds))
    if only == "bool":
        return _result("boolean", s, 1.0, "Python bool objects")
    if only == "number":
        return _numeric_type(as_numeric(nonnull), s, name, cfg, "numbers in object column")
    if only == "datetime":
        return _result("datetime", s, 0.95, "Python date/datetime objects")
    if only != "str":
        return _result("categorical", s, 0.5, f"{only} objects, treated as labels")
    return _string_type(sampled.astype(str), s, name, cfg, len(nonnull), nunique)


def see_types(df: pd.DataFrame | pd.Series, config: StoneConfig | None = None) -> pd.DataFrame:
    """Semantic type of every column: one row per column.

    Columns: ``semantic_type``, ``pandas_dtype``, ``confidence``, ``reason``, ``flags``, ``suggestion``.
    """
    df = prepare(df)
    rows = []
    for col in df.columns:
        try:
            info = infer_type(df[col], config)
        except Exception as exc:  # noqa: BLE001 - one bad column must not stop the rest
            info = _result("mixed", df[col], 0.0, f"type inference failed: {exc!r}")
        rows.append({
            "column": col,
            "semantic_type": info["semantic_type"],
            "pandas_dtype": info["pandas_dtype"],
            "confidence": info["confidence"],
            "reason": info["reason"],
            "flags": ", ".join(x["flag"] for x in info["suggestions"]),
            "suggestion": "; ".join(x["code"] for x in info["suggestions"]),
        })
    cols = ["column", "semantic_type", "pandas_dtype", "confidence", "reason", "flags", "suggestion"]
    return pd.DataFrame(rows, columns=cols).set_index("column")


__all__ = [
    "SEMANTIC_TYPES", "as_boolean", "as_datetime", "as_numeric", "see_types", "infer_type",
    "ordinal_order", "parse_number", "parse_numeric", "python_kinds",
]
