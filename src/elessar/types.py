"""Semantic type inference: what a column *means*, not just how pandas stores it."""

from __future__ import annotations

import datetime as dt
import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

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
FORCEABLE_TYPES = ("numeric_continuous", "numeric_discrete", "categorical", "ordinal-candidate", "id", "text")
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
# a category written with numbers: the first or last word of the name says what the integers are
# (region 11, store 7, payment_type 3, mcc 5411); "code" alone is too vague (a test called Code) and
# only counts with another word (code_region, mcc_code)
_CODE_WORDS = frozenset((
    "region regions dept department departement store shop branch agency agence zip zipcode postal postcode cp plz "
    "mcc naf ape sic naics nace isic icd cim atc country pays state province district county commune insee sex sexe "
    "gender genre status statut type kind category categorie cat segment cluster group groupe team mode method "
    "channel canal payment paiement brand marque race ethnicity religion marital nationality nationalite language "
    "langue occupation profession job industry sector secteur color colour couleur origin vendor carrier airline "
    "station site zone ward tract").split())
# ...unless a word of the name says it measures or counts something
_QUANTITY_WORDS = frozenset((
    "n nb nbr num number count counts cnt nombre total sum qty quantity amount montant price prix cost cout salary "
    "salaire income revenue weight poids height taille length size age duration duree time score scores rate ratio "
    "pct percent temp temperature distance km kg year years annee month day days hour hours minutes seconds "
    "population pop votes rank rating level niveau grade defects defect errors faults items units visits calls cases "
    "deaths births points goals wins losses sales orders clicks views stars reviews words pages children members "
    "employees rooms beds seats").split())


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


# ------------------------------------------------------------------------------------------- the decision tree
#
# Typing is an explicit decision tree. The root routes a column by its storage (empty, constant,
# bool, datetime, category, number, text...), then the column goes down an ordered list of
# yes / no questions (``NUMERIC_RULES``, ``TEXT_RULES``) until one answers yes. Every column keeps
# its path through the tree, so the report can say *why* it got its type.
#
# Certainty. When the tree only sees part of the column (a sample), a decision such as "92% of
# the values are dates, threshold 90%" may not hold on every row. Each threshold test therefore
# records the Wilson 95% interval of the observed share: a decision whose interval crosses its
# threshold, on any question along the path, is *uncertain* and is re-examined on more rows.
# Shares of distinct values are not proportions of independent draws (they shrink as the sample
# grows): a decision is uncertain when such a share sits within 5 points of its threshold.
# Decisions that rest on the set of distinct values (labels, levels) are uncertain while the
# Good-Turing estimate of the unseen probability mass (values seen once / values seen) is above
# 1%: some labels have probably not been seen yet. Uniqueness (identifiers) cannot be proven on
# a sample at all: such decisions are *probable* and checked on every row by the full pass. So are
# readings of what values mean (integers taken for category codes), which no number of rows can prove.

Z95 = 1.959963984540054
UNSEEN_MASS = 0.01
MIN_VALUES = 30


def wilson(k: float, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval of a proportion k / n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def good_turing_unseen(values: pd.Series) -> float:
    """Estimated probability that the next value is one never seen so far (Good-Turing: f1 / n)."""
    counts = values.value_counts()
    n = int(counts.sum())
    return float((counts == 1).sum()) / n if n else 1.0


@dataclass
class Test:
    """One threshold question answered on the way down the tree."""

    rule: str
    share: float
    n: int
    threshold: float

    @property
    def passed(self) -> bool:
        return self.share >= self.threshold

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.share * self.n, self.n)

    @property
    def close_call(self) -> bool:
        lo, hi = self.interval
        return lo < self.threshold <= hi


class Walk:
    """One column going down the tree: the data it carries, the questions asked, the answers."""

    def __init__(self, s: pd.Series, cfg: StoneConfig, sampled: bool) -> None:
        self.s, self.cfg, self.sampled = s, cfg, sampled
        self.name = _name_tokens(s.name)
        self.path: list[str] = []
        self.tests: list[Test] = []
        self.ambiguity: str | None = None
        self.cardinality: pd.Series | None = None  # values whose label set must be complete
        self.identifier = False
        self.verify: str | None = None  # "empty" / "constant": only all the rows can prove it
        self.inferred: str | None = None  # a reading of what the values mean, which no count of rows can prove

    def ask(self, question: str) -> None:
        self.path.append(question)

    def near(self, rule: str, value: float, threshold: float, margin: float = 0.05) -> None:
        """A sample statistic that is not a proportion of independent draws (the share of distinct values
        shrinks as the sample grows): only flag it when it sits within ``margin`` of its threshold."""
        if self.sampled and abs(value - threshold) < margin and not self.ambiguity:
            self.ambiguity = f"{rule}: {value:.1%} on the sample, too close to the {threshold:.0%} threshold to decide"

    def share(self, rule: str, share: float, n: int, threshold: float) -> bool:
        test = Test(rule, float(share), int(n), float(threshold))
        self.tests.append(test)
        return test.passed

    def leaf(self, stype: str, confidence: float, reason: str, **extra: Any) -> dict[str, Any]:
        out = {"semantic_type": stype, "pandas_dtype": str(self.s.dtype), "confidence": round(float(confidence), 3),
               "reason": reason, "suggestions": [], "path": list(self.path), **extra}
        out["certainty"], out["certainty_reason"] = self.certainty()
        return out

    def certainty(self) -> tuple[str, str]:
        if self.ambiguity:
            return "uncertain", self.ambiguity
        if self.inferred:
            return "probable", self.inferred
        if not self.sampled:
            return "certain", "decided on every value"
        close = [t for t in self.tests if t.close_call]
        if close:
            t = close[0]
            lo, hi = t.interval
            return "uncertain", (f"{t.rule}: {t.share:.1%} of {t.n:,} values, 95% CI [{lo:.1%}, {hi:.1%}] crosses the "
                                 f"{t.threshold:.0%} threshold")
        if self.cardinality is not None:
            unseen = good_turing_unseen(self.cardinality)
            if unseen > UNSEEN_MASS:
                return "uncertain", (f"labels may be missing: Good-Turing unseen mass {unseen:.1%} "
                                     f"(> {UNSEEN_MASS:.0%})")
        if self.identifier:
            return "probable", "uniqueness can only be proven on every row: checked by the full pass"
        if self.verify:
            return "probable", f"{self.verify} on the sample: checked on every row by the full pass"
        n = int(self.s.notna().sum())
        if n < MIN_VALUES:
            return "uncertain", f"only {n} non-missing values"
        return "certain", "every test is decided at 95% confidence"


Rule = tuple[str, Callable[[Walk, Any], "dict[str, Any] | None"]]


# ---------------------------------------------------------------------------------------- numeric branch
class _Num:
    """What the numeric rules look at: the finite values of a float view of the column."""

    def __init__(self, x: pd.Series, how: str) -> None:
        self.how = how
        self.finite = x[np.isfinite(x)]
        self.n = len(self.finite)
        self.nunique = int(self.finite.nunique())
        self.is_int = bool(np.all(np.mod(self.finite, 1) == 0))
        self.uniq_ratio = self.nunique / self.n if self.n else 0.0
        self.lo = float(self.finite.min()) if self.n else 0.0
        self.hi = float(self.finite.max()) if self.n else 0.0


def _n_binary(w: Walk, v: _Num) -> dict[str, Any] | None:
    if v.nunique == 2 and set(np.unique(v.finite)) <= {0.0, 1.0}:
        return w.leaf("boolean", 0.9, f"{v.how}: only 0/1 values")
    return None


def _n_dates(w: Walk, v: _Num) -> dict[str, Any] | None:
    hinted = bool(_HINT_ID.search(w.name))
    if (hinted and not DATE_HINT.search(w.name)) or v.nunique < 2 or not (v.is_int or v.lo >= 1e8):
        return None
    with quiet():
        fmt = detect_date(v.finite, w.name)
    if fmt is None:
        return None
    w.share("dates written as numbers", fmt.ratio, v.n, 0.9)
    if fmt.ambiguous:
        w.ambiguity = f"also readable as {', '.join(fmt.ambiguous)}"
    return w.leaf("datetime_string", fmt.ratio, f"{v.how}: dates stored as numbers ({fmt.label})", date_format=fmt)


def _n_id_named(w: Walk, v: _Num) -> dict[str, Any] | None:
    if not (v.is_int and _HINT_ID.search(w.name) and v.uniq_ratio < w.cfg.id_unique_ratio):
        return None
    if v.uniq_ratio >= 0.8 and (v.hi - v.lo + 1) <= 1.5 * v.nunique:
        w.identifier = True
        return w.leaf("id", 0.7, f"{v.how}: name looks like an id, mostly unique consecutive integers")
    # a grouping key needs no complete set of values (unlike labels): no Good-Turing check here
    return w.leaf("categorical", 0.7, f"{v.how}: repeated identifier ({v.nunique} ids), a grouping key")


def _n_id_values(w: Walk, v: _Num) -> dict[str, Any] | None:
    if not v.is_int:
        return None
    w.near("share of distinct values", v.uniq_ratio, w.cfg.id_unique_ratio)
    if v.uniq_ratio < w.cfg.id_unique_ratio:
        return None
    hinted = bool(_HINT_ID.search(w.name))
    consecutive = v.n >= 20 and (v.hi - v.lo + 1) <= 1.5 * v.nunique
    monotonic = v.n >= 20 and v.nunique == v.n and (v.finite.is_monotonic_increasing or v.finite.is_monotonic_decreasing)
    if hinted or consecutive or monotonic:
        w.identifier = True
        why = "name looks like an id" if hinted else "unique, (nearly) consecutive integers"
        return w.leaf("id", 0.9 if hinted else 0.8, f"{v.how}: {why}")
    return None


def _code_tokens(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name).strip("_")


def _n_codes(w: Walk, v: _Num) -> dict[str, Any] | None:
    """Integers that label a category (region 11, store 7, payment type 3, MCC 5411, postal code 75011):
    their mean, spread or correlation mean nothing. Values alone cannot prove it, so the decision is
    at best *probable*: by the name, or by the shape of codes (a few fixed-width integers of 3+ digits,
    not years, not round amounts)."""
    if not v.is_int or v.nunique < 2 or v.uniq_ratio >= w.cfg.id_unique_ratio:
        return None
    words = [t for t in _code_tokens(w.name).split("_") if t]
    if not words or _QUANTITY_WORDS.intersection(words):
        return None
    uniq = np.unique(v.finite)
    if v.lo == 0 and v.nunique >= 3 and bool(np.all(np.mod(uniq, 5) == 0)):
        return None  # a graded scale from 0 in steps of 5 (0 / 5 / 10, 0 / 25 / ... / 100) is a score
    named = words[0] in _CODE_WORDS or words[-1] in _CODE_WORDS or (len(words) > 1 and "code" in (words[0], words[-1]))
    if named:
        w.inferred = ("the name says these integers label categories; if they are quantities, set "
                      "StoneConfig(column_types={...: 'numeric_discrete'})")
        return w.leaf("categorical", 0.75, f"{v.how}: {v.nunique} integer codes, the name says a category",
                      codes="name")
    widths = np.floor(np.log10(np.abs(uniq[uniq != 0]))) + 1 if (uniq != 0).any() else np.array([1.0])
    fixed = v.lo >= 100 and len(set(widths.tolist())) == 1
    yearly = 1800 <= v.lo and v.hi <= 2100
    rounded = bool(np.all(np.mod(uniq, 5) == 0))  # price tiers, grams, scores: rounded quantities
    if (fixed and not yearly and not rounded and v.n >= 20 and v.nunique <= w.cfg.discrete_max_levels
            and v.uniq_ratio < 0.5):
        w.inferred = (f"{v.nunique} distinct {int(widths[0])}-digit integers far from 0 look like codes (postal, "
                      "merchant, activity...); if they are quantities, set StoneConfig(column_types={...: "
                      "'numeric_discrete'})")
        return w.leaf("categorical", 0.6, f"{v.how}: {v.nunique} fixed-width integers, probably codes", codes="values")
    return None


def _n_scale(w: Walk, v: _Num) -> dict[str, Any] | None:
    cfg = w.cfg
    small = 2 <= v.nunique <= cfg.ordinal_max_int_levels and v.lo in (0.0, 1.0) and v.hi - v.lo + 1 == v.nunique
    if not (v.is_int and small and v.n >= 20):
        return None
    w.cardinality = v.finite
    if abs(float(v.finite.skew())) < 1:  # counts (children, passengers...) are right-skewed; rating scales are not
        return w.leaf("ordinal-candidate", 0.6,
                      f"{v.how}: {v.nunique} consecutive small integers ({v.lo:g}..{v.hi:g}), maybe a scale")
    return w.leaf("numeric_discrete", 0.8, f"{v.how}: small right-skewed integers ({v.lo:g}..{v.hi:g}), a count")


def _n_integers(w: Walk, v: _Num) -> dict[str, Any] | None:
    if not v.is_int:
        return None
    if v.nunique > w.cfg.discrete_max_levels:
        return w.leaf("numeric_continuous", 0.7, f"{v.how}: integer-valued with many levels ({v.nunique})")
    w.cardinality = v.finite
    return w.leaf("numeric_discrete", 0.85, f"{v.how}: integer values ({v.nunique} distinct)")


def _n_few_reals(w: Walk, v: _Num) -> dict[str, Any] | None:
    if v.nunique <= min(w.cfg.discrete_max_levels, 20) and v.uniq_ratio < 0.05:
        w.cardinality = v.finite
        return w.leaf("numeric_discrete", 0.7, f"{v.how}: only {v.nunique} distinct non-integer values")
    return None


def _n_continuous(w: Walk, v: _Num) -> dict[str, Any] | None:
    if v.n == 0:
        return w.leaf("numeric_continuous", 0.5, f"{v.how}: only infinite values")
    return w.leaf("numeric_continuous", 0.9, f"{v.how}: {v.nunique} distinct real values")


NUMERIC_RULES: tuple[Rule, ...] = (
    ("only 0 and 1?", _n_binary),
    ("dates written as numbers (YYYYMMDD, Unix time...)?", _n_dates),
    ("an identifier by its name?", _n_id_named),
    ("unique, consecutive or sorted integers?", _n_id_values),
    ("integers that label categories (codes)?", _n_codes),
    ("a small scale of consecutive integers?", _n_scale),
    ("integers?", _n_integers),
    ("a handful of distinct real values?", _n_few_reals),
    ("real numbers", _n_continuous),
)


def _numeric_branch(w: Walk, x: pd.Series, how: str) -> dict[str, Any]:
    v = _Num(x, how)
    for question, rule in NUMERIC_RULES:
        w.ask(question)
        out = rule(w, v)
        if out is not None:
            return out
    raise AssertionError("the last numeric rule always answers")


# ------------------------------------------------------------------------------------------- text branch
class _Text:
    """What the text rules look at: stripped strings (a sample of them) and the column's cardinality."""

    def __init__(self, strs: pd.Series, n_nonnull: int, nunique: int) -> None:
        self.t = strs.str.strip()
        self.n = len(self.t)
        self.n_nonnull, self.nunique = n_nonnull, nunique
        self.uniq_ratio = nunique / n_nonnull
        self._cache: dict[str, Any] = {}

    def match(self, rx: re.Pattern) -> float:
        return _ratio(self.t.str.match(rx).fillna(False).to_numpy(dtype=bool))

    def get(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def leading_zero_code(self) -> bool:
        t = self.t
        return self.get("lzc", lambda: _ratio(t.str.fullmatch(r"\d+")) >= 0.9
                        and _ratio(t.str.match(r"0\d")) >= 0.05 and t.str.len().nunique() == 1)


def _t_boolean(w: Walk, v: _Text) -> dict[str, Any] | None:
    lowered = set(v.t.str.lower().unique())
    if len(lowered) == 2 and any(x in TRUE_TOKENS for x in lowered) and any(x in FALSE_TOKENS for x in lowered):
        return w.leaf("boolean", 0.95, f"two boolean-like labels: {sorted(lowered)}")
    return None


def _t_json(w: Walk, v: _Text) -> dict[str, Any] | None:
    starts = v.t.str[:1].isin(["{", "["])
    if _ratio(starts) < w.cfg.pattern_match_ratio:
        return None
    ok = _ratio([_is_json(x) for x in v.t[starts]]) * _ratio(starts)
    if w.share("valid JSON", ok, v.n, w.cfg.pattern_match_ratio):
        return w.leaf("json_like", ok, f"{ok:.0%} of values parse as JSON")
    return None


def _t_pattern(stype: str, rx: re.Pattern) -> Callable[[Walk, _Text], dict[str, Any] | None]:
    def rule(w: Walk, v: _Text) -> dict[str, Any] | None:
        m = v.match(rx)
        if m > 0 and w.share(f"{stype} pattern", m, v.n, w.cfg.pattern_match_ratio):
            return w.leaf(stype, m, f"{m:.0%} of values look like {stype}s")
        return None
    return rule


def _t_ip(w: Walk, v: _Text) -> dict[str, Any] | None:
    ip = _ratio([_is_ip(x) for x in v.t])
    if ip > 0 and w.share("IP addresses", ip, v.n, w.cfg.pattern_match_ratio):
        return w.leaf("ip_address", ip, f"{ip:.0%} of values are IP addresses")
    return None


def _t_uuid(w: Walk, v: _Text) -> dict[str, Any] | None:
    m = v.match(_UUID)
    if m > 0 and w.share("UUIDs", m, v.n, w.cfg.pattern_match_ratio):
        w.identifier = True
        return w.leaf("id", m, f"{m:.0%} of values are UUIDs")
    return None


def _t_postal(w: Walk, v: _Text) -> dict[str, Any] | None:
    r = w.cfg.pattern_match_ratio
    hint = bool(_HINT_POSTAL.search(w.name))
    m = v.match(_POSTAL_STRICT)
    if m >= r or (hint and _ratio(v.t.str.match(_POSTAL_STRICT) | v.t.str.match(_POSTAL_DIGITS)) >= r):
        w.share("postal-code pattern", max(m, r), v.n, r)
        return w.leaf("postal_code", max(m, 0.8), "postal-code pattern" + (" and column name" if hint else ""))
    return None


def _t_compact_dates(w: Walk, v: _Text) -> dict[str, Any] | None:
    digits = _ratio(v.t.str.fullmatch(r"\d+|-|\?|").fillna(False))
    if digits < 0.99 or (_HINT_ID.search(w.name) and not DATE_HINT.search(w.name)):
        return None
    with quiet():
        fmt = detect_date(v.t, w.name)
    if fmt is None:
        return None
    w.share("dates written as digits", fmt.ratio, v.n, 0.9)
    if fmt.ambiguous:
        w.ambiguity = f"also readable as {', '.join(fmt.ambiguous)}"
    return w.leaf("datetime_string", fmt.ratio, f"dates written as {fmt.label}", date_format=fmt)


def _t_numbers(w: Walk, v: _Text) -> dict[str, Any] | None:
    if v.leading_zero_code or _HINT_PHONE.search(w.name):
        return None
    num = parse_numeric(v.t, dominant_unit_only=False)
    m = _ratio(num.notna())
    if not w.share("numbers written as text", m, v.n, w.cfg.pattern_match_ratio):
        return None
    units = unit_counts(v.t)
    reason = f"{m:.0%} of values are numbers stored as text"
    if units:
        total = sum(units.values())
        listed = ", ".join(f"{u} {100 * c / total:.0f}%" for u, c in list(units.items())[:4])
        reason = f"numbers with units ({listed})" + (": mixed units, not comparable" if len(units) > 1 else "")
    out = w.leaf("numeric_string", m, reason, units=units)
    out["numeric"] = _numeric_branch(Walk(w.s, w.cfg, w.sampled), parse_numeric(v.t).dropna(),
                                     "parsed numbers")["semantic_type"]
    return out


def _t_dates(w: Walk, v: _Text) -> dict[str, Any] | None:
    r = w.cfg.pattern_match_ratio
    if _ratio(v.t.str.contains(r"\d", regex=True)) < r or v.leading_zero_code:
        return None
    with quiet():
        fmt = detect_date(v.t, w.name, min_ratio=r)
    if fmt is None:
        return None
    w.share("dates", fmt.ratio, v.n, r)
    if fmt.ambiguous:
        w.ambiguity = f"day / month order ambiguous: also readable as {', '.join(fmt.ambiguous)}"
    amb = f"; also valid as {', '.join(fmt.ambiguous)}" if fmt.ambiguous else ""
    return w.leaf("datetime_string", fmt.ratio, f"{fmt.ratio:.0%} of values are {fmt.label}{amb}", date_format=fmt)


def _t_phone(w: Walk, v: _Text) -> dict[str, Any] | None:
    hint = bool(_HINT_PHONE.search(w.name))
    digit_count = v.t.str.count(r"\d")
    like = v.t.str.match(_PHONE) & digit_count.between(7, 15)
    like &= v.t.str.contains(r"^\+|\d[\s().-]+\d", regex=True) | hint
    m = _ratio(like)
    if m == 0:
        return None
    with quiet():  # 1923-02-10 has the shape of a phone number: dates are never phones
        dated = pd.to_datetime(v.t[like.to_numpy()], format="mixed", errors="coerce").notna().mean()
    if dated > 0.5:
        return None
    if w.share("phone numbers", m, v.n, w.cfg.pattern_match_ratio):
        return w.leaf("phone", m, f"{m:.0%} of values look like phone numbers")
    return None


def _t_lists(w: Walk, v: _Text) -> dict[str, Any] | None:
    sep = list_separator(v.t)
    if sep is None:
        return None
    example = truncate(str(v.t[v.t.str.contains(sep, regex=False)].iloc[0]), 40)
    w.cardinality = v.t.str.split(sep, regex=False).explode().str.strip()
    return w.leaf(MULTI_LABEL, 0.8, f"lists of labels separated by {sep.strip() or repr(sep)!s} (e.g. {example!r})",
                  sep=sep)


def _words(v: _Text) -> tuple[float, float]:
    return v.get("words", lambda: (float(v.t.str.split().str.len().mean()), float(v.t.str.len().mean())))


def _t_unique_labels(w: Walk, v: _Text) -> dict[str, Any] | None:
    avg_words, avg_len = _words(v)
    w.near("share of distinct labels", v.uniq_ratio, w.cfg.id_unique_ratio)
    if v.uniq_ratio >= w.cfg.id_unique_ratio and (v.n_nonnull >= 20 or _HINT_ID.search(w.name)):
        if avg_words <= 2.5 and avg_len <= 40:
            w.identifier = True
            return w.leaf("id", 0.8, f"{v.uniq_ratio:.0%} unique short labels (names / codes)")
    return None


def _t_free_text(w: Walk, v: _Text) -> dict[str, Any] | None:
    avg_words, avg_len = _words(v)
    if (avg_words >= w.cfg.text_min_avg_words or avg_len >= w.cfg.text_min_avg_length) and v.uniq_ratio > 0.3:
        return w.leaf("text", 0.85, f"free text (avg {avg_words:.1f} words, {avg_len:.0f} chars)")
    return None


def _t_ordered(w: Walk, v: _Text) -> dict[str, Any] | None:
    order = ordinal_order(v.t.unique()) if v.nunique <= w.cfg.ordinal_max_levels else None
    if order:
        w.cardinality = v.t
        return w.leaf("ordinal-candidate", 0.75, f"ordered-looking labels: {' < '.join(order)}", order=order)
    return None


def _t_categorical(w: Walk, v: _Text) -> dict[str, Any] | None:
    if v.nunique <= w.cfg.max_categories or v.uniq_ratio < 0.5:
        w.cardinality = v.t
        return w.leaf("categorical", 0.85, f"{v.nunique} distinct labels")
    if _words(v)[0] < 2:
        return w.leaf("categorical", 0.6, f"high-cardinality codes ({v.nunique} distinct short labels)")
    return None


def _t_fallback(w: Walk, v: _Text) -> dict[str, Any] | None:
    return w.leaf("text", 0.6, f"high-cardinality strings ({v.nunique} distinct)")


TEXT_RULES: tuple[Rule, ...] = (
    ("two boolean-like labels?", _t_boolean),
    ("JSON objects or arrays?", _t_json),
    ("e-mail addresses?", _t_pattern("email", _EMAIL)),
    ("URLs?", _t_pattern("url", _URL)),
    ("IP addresses?", _t_ip),
    ("UUIDs?", _t_uuid),
    ("postal codes?", _t_postal),
    ("dates written as digits (YYMMDD...)?", _t_compact_dates),
    ("numbers written as text?", _t_numbers),
    ("dates?", _t_dates),
    ("phone numbers?", _t_phone),
    ("lists of labels?", _t_lists),
    ("unique short labels (an identifier)?", _t_unique_labels),
    ("free text?", _t_free_text),
    ("labels on a known ordered scale?", _t_ordered),
    ("labels (categories)?", _t_categorical),
    ("high-cardinality strings", _t_fallback),
)


def _text_branch(w: Walk, strs: pd.Series, n_nonnull: int, nunique: int) -> dict[str, Any]:
    v = _Text(strs, n_nonnull, nunique)
    for question, rule in TEXT_RULES:
        w.ask(question)
        out = rule(w, v)
        if out is not None:
            return out
    raise AssertionError("the last text rule always answers")


def _ratio(mask: pd.Series | np.ndarray | list) -> float:
    return float(np.mean(mask)) if len(mask) else 0.0


def _result(stype: str, s: pd.Series, confidence: float, reason: str, **extra: Any) -> dict[str, Any]:
    """A decision made outside the tree (e.g. after an error)."""
    return {"semantic_type": stype, "pandas_dtype": str(s.dtype), "confidence": round(float(confidence), 3),
            "reason": reason, "suggestions": [], "path": [], "certainty": "uncertain", "certainty_reason": reason,
            **extra}


# ------------------------------------------------------------------------------------------- the root
def _root(s: pd.Series, cfg: StoneConfig, sampled: bool) -> dict[str, Any]:
    """Route by storage, then down the numeric or the text branch."""
    w = Walk(s, cfg, sampled)
    nonnull = s.dropna()
    w.ask("any non-missing value?")
    if len(nonnull) == 0:
        w.verify = "empty"
        return w.leaf("empty", 1.0, "no non-missing values")
    hs = hashable(nonnull)
    nunique = int(hs.nunique())
    w.ask("a single distinct value?")
    if nunique == 1:
        w.verify = "constant"
        return w.leaf("constant", 1.0, f"single value: {str(hs.iloc[0])[:40]!r}")
    dtype = s.dtype
    w.ask(f"how is it stored? ({dtype})")
    if pd.api.types.is_bool_dtype(dtype):
        return w.leaf("boolean", 1.0, "bool dtype")
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return w.leaf("datetime", 1.0, "datetime dtype")
    if isinstance(dtype, pd.CategoricalDtype):
        if dtype.ordered:
            return w.leaf("ordinal-candidate", 0.95, "ordered categorical dtype", order=[str(c) for c in dtype.categories])
        sub = _root(nonnull.astype(object), cfg, sampled)
        if sub["semantic_type"] in NUMERIC_TYPES | {"text", "id", "email", "url", "phone", "postal_code", "ip_address"}:
            sub["pandas_dtype"] = str(dtype)
            sub["path"] = w.path + sub["path"]
            return sub
        w.cardinality = hs
        return w.leaf("categorical", 1.0, f"category dtype ({nunique} levels)")
    if pd.api.types.is_complex_dtype(dtype):
        return w.leaf("mixed", 0.5, "complex numbers")
    if pd.api.types.is_timedelta64_dtype(dtype):
        out = _numeric_branch(w, nonnull.dt.total_seconds(), "timedelta (seconds)")
        if out["semantic_type"] in ("id", "boolean", "datetime_string"):
            out["semantic_type"] = "numeric_continuous"
        return out
    if pd.api.types.is_numeric_dtype(dtype):
        return _numeric_branch(w, as_numeric(nonnull), f"{dtype} dtype")
    if not is_text_dtype(s):
        return w.leaf("categorical", 0.5, f"unrecognised dtype {dtype}, treated as labels")

    seen = sample(nonnull, 5_000, cfg.random_state)
    if len(seen) < len(nonnull):
        w.sampled = True  # patterns are checked on a sample of the values
    kinds = python_kinds(seen) if pd.api.types.is_object_dtype(dtype) else {"str": len(seen)}
    w.ask("what Python objects are in the cells?")
    if set(kinds) <= {"dict", "list"}:
        return w.leaf("json_like", 1.0, "cells contain Python " + "/".join(sorted(kinds)) + " objects")
    if len(kinds) > 1 and not set(kinds) <= {"number", "bool"}:
        mix = ", ".join(f"{k}={v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]))
        return w.leaf("mixed", 1.0, f"several Python types: {mix}", kinds=kinds)
    only = next(iter(kinds))
    if only == "bool":
        return w.leaf("boolean", 1.0, "Python bool objects")
    if only == "number":
        return _numeric_branch(w, as_numeric(nonnull), "numbers in object column")
    if only == "datetime":
        return w.leaf("datetime", 0.95, "Python date/datetime objects")
    if only != "str":
        return w.leaf("categorical", 0.5, f"{only} objects, treated as labels")
    return _text_branch(w, seen.astype(str), len(nonnull), nunique)


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
    if info.get("codes"):
        out.append({"flag": "integer-codes", "note": info.get("certainty_reason", ""),
                    "code": f"df[{col}] = df[{col}].astype('category')"})
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


def infer_type(s: pd.Series, config: StoneConfig | None = None, sampled: bool = False) -> dict[str, Any]:
    """Infer the semantic type of one column by walking the decision tree.

    Returns a dict with ``semantic_type``, ``pandas_dtype``, ``confidence`` (0-1), ``reason``,
    ``suggestions`` (list of ``{flag, code}``), ``path`` (the questions asked, in order) and
    ``certainty`` (``certain`` / ``probable`` / ``uncertain``) with ``certainty_reason``.
    ``sampled``: the series is a sample of a bigger column (certainty then accounts for it).
    """
    cfg = config or StoneConfig()
    forced = (cfg.column_types or {}).get(str(s.name))
    if forced is not None:
        if forced not in FORCEABLE_TYPES:
            raise ValueError(f"column_types[{str(s.name)!r}] = {forced!r}: choose one of {', '.join(FORCEABLE_TYPES)}")
        w = Walk(s, cfg, sampled)
        w.ask("set by the user (StoneConfig.column_types)?")
        info = w.leaf(forced, 1.0, "type set by the user")
        info["certainty"], info["certainty_reason"] = "certain", "set by the user"
    else:
        info = _root(s, cfg, sampled)
    info["suggestions"] = _suggest(info, s, str(s.name))
    return info


def see_types(df: pd.DataFrame | pd.Series, config: StoneConfig | None = None) -> pd.DataFrame:
    """Semantic type of every column: one row per column.

    Columns: ``semantic_type``, ``pandas_dtype``, ``confidence``, ``certainty`` (certain / probable / uncertain),
    ``path`` (the questions of the decision tree, in order), ``reason``, ``flags``, ``suggestion``.
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
            "certainty": info.get("certainty", "certain"),
            "path": " > ".join(map(str, info.get("path") or [])),
            "reason": info["reason"],
            "flags": ", ".join(x["flag"] for x in info["suggestions"]),
            "suggestion": "; ".join(x["code"] for x in info["suggestions"]),
        })
    cols = ["column", "semantic_type", "pandas_dtype", "confidence", "certainty", "path", "reason", "flags",
            "suggestion"]
    return pd.DataFrame(rows, columns=cols).set_index("column")


__all__ = [
    "NUMERIC_RULES", "SEMANTIC_TYPES", "TEXT_RULES", "as_boolean", "as_datetime", "as_numeric", "good_turing_unseen",
    "infer_type", "ordinal_order", "parse_number", "parse_numeric", "python_kinds", "see_types", "wilson",
]
