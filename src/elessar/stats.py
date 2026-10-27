"""Descriptive statistics, adapted to each semantic type."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import signal, stats

from ._utils import hashable, prepare, quiet, sample, sample_array
from .config import StoneConfig
from .types import (
    CATEGORY_TYPES,
    DATETIME_TYPES,
    LIST_SEPARATORS,
    MULTI_LABEL,
    NUMERIC_TYPES,
    STRING_PATTERN_TYPES,
    as_boolean,
    as_datetime,
    as_numeric,
    infer_type,
    list_separator,
    parse_numeric,
    python_kinds,
)

QUANTILES: tuple[int, ...] = (1, 5, 10, 25, 50, 75, 90, 95, 99)
_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it this that at by from as not but "
    "le la les un une des et ou de du en au aux pour par sur est sont pas il elle".split()
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _value(v: Any) -> Any:
    """Readable, JSON-friendly representation of one cell value."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, (bool, int, float)) or v is None:
        return v
    text = str(v)
    return text if len(text) <= 80 else text[:79] + "…"


def _f(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def common_stats(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Statistics shared by every column: counts, missing, uniques, memory, modes, top values."""
    cfg = config or StoneConfig()
    n = len(s)
    nonnull = hashable(s.dropna())
    count = len(nonnull)
    vc = nonnull.value_counts(dropna=True)
    top = vc.iloc[0] if len(vc) else 0
    modes = [_value(v) for v in vc.index[vc == top][:3]] if top > 1 or len(vc) == 1 else []
    return {
        "count": count,
        "n_missing": n - count,
        "pct_missing": 100.0 * (n - count) / n if n else 0.0,
        "n_unique": int(len(vc)),
        "pct_unique": 100.0 * len(vc) / count if count else 0.0,
        "memory_bytes": int(s.memory_usage(deep=True, index=False)),
        "modes": modes,
        "mode_freq": int(top) if modes else None,
        "mode_pct": 100.0 * top / count if modes else None,
        "top_values": [
            {"value": _value(v), "count": int(c), "pct": 100.0 * c / count} for v, c in vc.head(cfg.top_n).items()
        ],
    }


def _decimals(v: np.ndarray) -> int | None:
    """Smallest number of decimals that represents >= 99 % of values exactly (None if > 10)."""
    for d in range(11):
        if np.mean(np.isclose(np.round(v, d), v, rtol=1e-10, atol=1e-12)) >= 0.99:
            return d
    return None


def _heaping(v: np.ndarray) -> dict[str, Any]:
    """Excess of non-zero values ending in 0 or 5 at the data's own precision (index 1 = no heaping).

    Skipped (None) when the precision is mixed (e.g. fares with 0 to 4 decimals) or the range is tiny.
    """
    none: dict[str, Any] = {"pct_round_values": None, "heaping_index": None}
    v = v[v != 0]
    if len(v) < 20:
        return none
    shares = [float(np.mean(np.isclose(np.round(v, d), v, rtol=1e-10, atol=1e-12))) for d in range(7)]
    d = next((i for i, sh in enumerate(shares) if sh >= 0.9), None)
    if d is None or (d > 0 and shares[d - 1] >= 0.5):
        return none
    exact = v[np.isclose(np.round(v, d), v, rtol=1e-10, atol=1e-12)]
    scaled = np.round(exact * 10**d)
    while scaled.any() and np.mean(scaled % 10 == 0) >= 0.9:  # data rounded to tens, hundreds... (caps aside)
        scaled = scaled[scaled % 10 == 0] / 10
    if np.ptp(scaled) < 20:
        return none
    share = float(np.mean(np.isin(np.abs(scaled) % 10, (0, 5))))
    return {"pct_round_values": 100.0 * share, "heaping_index": share / 0.2}


def _pile_up(v: np.ndarray, at: float, side: str) -> dict[str, Any]:
    """Excess of values exactly at the min or max: a cap / floor (censoring) when far above its neighbours."""
    n_at = int(np.sum(v == at))
    others = v[v != at]
    if n_at < 10 or len(others) < 20:
        return {f"pct_at_{side}": 100.0 * n_at / len(v), f"capped_{side}": False}
    near = np.sort(np.unique(others))
    near = near[-10:] if side == "max" else near[:10]
    counts = np.array([np.sum(others == x) for x in near])
    return {f"pct_at_{side}": 100.0 * n_at / len(v), f"capped_{side}": bool(n_at >= 5 * max(np.median(counts), 1))}


def _median_ci(v: np.ndarray, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Distribution-free CI of the median from order statistics."""
    n = len(v)
    if n < 6:
        return None, None
    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = int(np.floor(n / 2 - z * np.sqrt(n) / 2)), int(np.ceil(n / 2 + z * np.sqrt(n) / 2))
    srt = np.sort(v)
    return float(srt[max(lo - 1, 0)]), float(srt[min(hi - 1, n - 1)])


def _autocorr(v: np.ndarray) -> tuple[float | None, float | None]:
    """Lag-1 autocorrelation in row order and its p-value (non-zero = rows are sorted / not independent)."""
    if len(v) < 10 or np.std(v) == 0:
        return None, None
    r = float(np.corrcoef(v[:-1], v[1:])[0, 1])
    return r, float(2 * stats.norm.sf(abs(r) * np.sqrt(len(v))))


def _kde_modes(v: np.ndarray, random_state: int) -> int | None:
    """Number of modes of a Gaussian KDE (peaks with prominence >= 5 % of the max)."""
    v = sample_array(v, 3_000, random_state)
    if len(v) < 10 or np.unique(v).size < 3:
        return None
    v = v / np.abs(v).max()  # the mode count is scale-free; this avoids overflow at 1e300
    v = (v - v.mean()) / v.std() if v.std() > 0 else v
    if not np.all(np.isfinite(v)) or v.std() == 0:
        return None
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi <= lo:
        return 1
    grid = np.linspace(lo, hi, 512)
    dens = stats.gaussian_kde(v)(grid)
    peaks, _ = signal.find_peaks(np.r_[0, dens, 0], prominence=0.05 * dens.max())
    return max(int(len(peaks)), 1)


def numeric_stats(x: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Full numeric profile of a float Series (NaN = missing, inf counted separately)."""
    cfg = config or StoneConfig()
    values = x.dropna().to_numpy(dtype="float64")
    is_inf = np.isinf(values)
    v = values[~is_inf]
    n = len(v)
    out: dict[str, Any] = {"n_inf": int(is_inf.sum())}
    if n == 0:
        return out
    with quiet():
        mean, std = float(np.mean(v)), float(np.std(v, ddof=1)) if n > 1 else float("nan")
        q = np.percentile(v, QUANTILES)
        iqr = q[5] - q[3]
        sem = std / math.sqrt(n) if n > 1 else float("nan")
        skew = float(stats.skew(v, bias=False)) if n > 2 else float("nan")
        kurt = float(stats.kurtosis(v, bias=False)) if n > 3 else float("nan")
        mad = float(stats.median_abs_deviation(v))
        z = (v - mean) / std if std and std > 0 else np.zeros_like(v)
        mz = 0.6745 * (v - q[4]) / mad if mad > 0 else np.zeros_like(v)
        lo_iqr, hi_iqr = q[3] - 1.5 * iqr, q[5] + 1.5 * iqr
        n_iqr = int(np.sum((v < lo_iqr) | (v > hi_iqr)))
        n_z, n_mz = int(np.sum(np.abs(z) > 3)), int(np.sum(np.abs(mz) > 3.5))
        positive = bool(np.all(v > 0))
        t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else float("nan")
        bc = None
        if n > 3 and math.isfinite(skew) and math.isfinite(kurt):
            bc = (skew**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        dec = _decimals(sample_array(v, 10_000, cfg.random_state))
        finite_series = pd.Series(v)
        out.update({
            "mean": mean,
            "std": std,
            "var": std**2,
            "sem": sem,
            "cv": std / abs(mean) if mean else None,
            "min": float(v.min()),
            "max": float(v.max()),
            "range": float(v.max() - v.min()),
            "sum": float(v.sum()),
            **{f"p{p}": float(val) for p, val in zip(QUANTILES, q)},
            "median": float(q[4]),
            "iqr": float(iqr),
            "mad": mad,
            "trimmed_mean": float(stats.trim_mean(v, cfg.trim_fraction)),
            "geometric_mean": float(stats.gmean(v)) if positive else None,
            "harmonic_mean": float(stats.hmean(v)) if positive else None,
            "skewness": skew,
            "kurtosis": kurt,
            "n_zeros": int(np.sum(v == 0)),
            "n_negative": int(np.sum(v < 0)),
            "iqr_lower": float(lo_iqr),
            "iqr_upper": float(hi_iqr),
            "n_outliers_iqr": n_iqr,
            "pct_outliers_iqr": 100.0 * n_iqr / n,
            "n_outliers_z": n_z,
            "pct_outliers_z": 100.0 * n_z / n,
            "n_outliers_mz": n_mz,
            "pct_outliers_mz": 100.0 * n_mz / n,
            "n_decimals": dec,
            **_heaping(v),
            "monotonic_increasing": bool(n > 1 and finite_series.is_monotonic_increasing),
            "monotonic_decreasing": bool(n > 1 and finite_series.is_monotonic_decreasing),
            "bimodality_coefficient": bc,
            "n_modes_kde": _kde_modes(v, cfg.random_state),
            "ci95_mean_low": mean - t_crit * sem if n > 1 else None,
            "ci95_mean_high": mean + t_crit * sem if n > 1 else None,
        })
        out["ci95_median_low"], out["ci95_median_high"] = _median_ci(v)
        if np.unique(v).size > 10:
            out.update(_pile_up(v, out["max"], "max"))
            out.update(_pile_up(v, out["min"], "min"))
        out["autocorr_lag1"], out["autocorr_lag1_p"] = _autocorr(v)
    return out


def _normalize_label(v: str) -> str:
    v = unicodedata.normalize("NFKD", v)
    v = "".join(c for c in v if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", v).strip().casefold()


def _length_stats(strs: pd.Series) -> dict[str, Any]:
    lens = strs.str.len()
    if lens.empty:
        return {}
    return {
        "len_min": int(lens.min()),
        "len_mean": float(lens.mean()),
        "len_median": float(lens.median()),
        "len_max": int(lens.max()),
    }


def categorical_stats(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Cardinality, entropy, Gini impurity, imbalance, rare categories and label inconsistencies."""
    cfg = config or StoneConfig()
    nonnull = hashable(s.dropna())
    vc = nonnull.value_counts()
    k, total = len(vc), int(vc.sum())
    if k == 0:
        return {}
    p = vc.to_numpy(dtype="float64") / total
    entropy = float(-(p * np.log2(p)).sum())
    rare = vc[100.0 * vc / total < cfg.rare_category_pct]
    groups: dict[str, set[str]] = {}
    for label in vc.index:
        if isinstance(label, str):
            groups.setdefault(_normalize_label(label), set()).add(label)
    inconsistent = [sorted(g) for g in groups.values() if len(g) > 1]
    strs = nonnull[nonnull.map(lambda v: isinstance(v, str))]
    return {
        "cardinality": k,
        "entropy": entropy,
        "normalized_entropy": entropy / math.log2(k) if k > 1 else 0.0,
        "gini_impurity": float(1 - (p**2).sum()),
        "imbalance_ratio": float(vc.iloc[0] / vc.iloc[-1]),
        "n_rare_categories": int(len(rare)),
        "pct_rows_rare": 100.0 * float(rare.sum()) / total,
        "rare_categories": [_value(v) for v in rare.index[:20]],
        "uniform_chi2_p": float(stats.chisquare(vc.to_numpy()).pvalue) if k > 1 else None,
        "n_inconsistent_groups": len(inconsistent),
        "inconsistent_labels": inconsistent[:10],
        **_length_stats(strs.astype(str)),
    }


def boolean_stats(s: pd.Series) -> dict[str, Any]:
    """Share of True values with its Wilson 95 % confidence interval."""
    b = as_boolean(s).dropna()
    n = len(b)
    if not n:
        return {"n_true": 0, "pct_true": None}
    p, z = float(b.mean()), stats.norm.ppf(0.975)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"n_true": int(b.sum()), "pct_true": 100 * p, "ci95_pct_true_low": 100 * (center - half),
            "ci95_pct_true_high": 100 * (center + half)}


def time_resolution(span_days: float) -> tuple[str, str]:
    """Pandas resample alias and its name for a time span: hour / day / week / month / year."""
    if span_days <= 2:
        return "h", "hour"
    if span_days <= 120:
        return "D", "day"
    if span_days <= 3 * 365:
        return "W", "week"
    if span_days <= 25 * 365:
        return "MS", "month"
    return "YS", "year"


def _ts(t: pd.Timestamp) -> str:
    """Readable timestamp: date only at midnight, otherwise rounded to the second."""
    return str(t.date()) if t == t.normalize() else str(t.round("s"))


def _td(td: pd.Timedelta) -> str:
    return str(td.round("s") if td >= pd.Timedelta(seconds=1) else td)


def datetime_stats(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Range, frequency, gaps, duplicates, future dates and calendar distributions."""
    cfg = config or StoneConfig()
    d = as_datetime(s).dropna()
    n_unparsed = int(s.notna().sum() - len(d))
    if d.empty:
        return {"n_unparsed": n_unparsed}
    uniq = pd.DatetimeIndex(d.unique()).sort_values()
    diffs = pd.Series(uniq[1:] - uniq[:-1])
    freq = None
    if len(uniq) >= 3:
        try:
            with quiet():
                inferred = pd.infer_freq(uniq)
            freq = getattr(inferred, "freqstr", inferred)
        except (TypeError, ValueError):
            freq = None
        if freq is None:
            freq = f"~{_td(diffs.median())} (irregular)"
    gaps = []
    if len(diffs):
        for i in diffs.nlargest(3).index:
            gaps.append({"start": _ts(uniq[i]), "end": _ts(uniq[i + 1]), "gap": _td(diffs[i])})
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    out: dict[str, Any] = {
        "min": _ts(d.min()),
        "max": _ts(d.max()),
        "span": _td(d.max() - d.min()),
        "span_days": (d.max() - d.min()).total_seconds() / 86_400,
        "inferred_freq": freq,
        "largest_gaps": gaps,
        "n_duplicated_timestamps": int(len(d) - len(uniq)),
        "n_future": int((d > now).sum()),
        "n_before_1900": int((d < pd.Timestamp("1900-01-01")).sum()),
        "n_unparsed": n_unparsed,
        "monotonic_increasing": bool(len(d) > 1 and d.is_monotonic_increasing),
        "monotonic_decreasing": bool(len(d) > 1 and d.is_monotonic_decreasing),
    }
    freq_alias, unit = time_resolution(out["span_days"])
    counts = pd.Series(1, index=pd.DatetimeIndex(d)).sort_index().resample(freq_alias).sum()
    if len(counts) >= 4 and counts.nunique() > 1:
        tau, p = stats.kendalltau(np.arange(len(counts)), counts.to_numpy())
        out.update(counts_resolution=unit, counts_trend_tau=float(tau), counts_trend_p=float(p))
    d = sample(d, cfg.sample_size, cfg.random_state)
    if d.dt.year.nunique() > 1:
        out["by_year"] = {int(k): int(v) for k, v in d.dt.year.value_counts().sort_index().tail(30).items()}
    if d.dt.month.nunique() > 1 and out["span_days"] >= 60:
        out["by_month"] = {_MONTHS[k - 1]: int(v) for k, v in d.dt.month.value_counts().sort_index().items()}
    if d.dt.day_of_week.nunique() > 1:
        out["by_weekday"] = {_WEEKDAYS[k]: int(v) for k, v in d.dt.day_of_week.value_counts().sort_index().items()}
    if d.dt.hour.nunique() > 1:
        out["by_hour"] = {int(k): int(v) for k, v in d.dt.hour.value_counts().sort_index().items()}
    return out


def text_stats(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Length / word counts, empty strings, character mix and most common tokens."""
    cfg = config or StoneConfig()
    strs = s.dropna().astype(str)
    if strs.empty:
        return {}
    words = strs.str.split().str.len()
    sub = sample(strs, 20_000, cfg.random_state)
    joined = "".join(sub.tolist())
    n_chars = max(len(joined), 1)
    tokens = Counter(t for t in re.findall(r"[^\W\d_]{2,}", " ".join(sub).lower()) if t not in _STOPWORDS)
    return {
        **_length_stats(strs),
        "words_mean": float(words.mean()),
        "words_median": float(words.median()),
        "words_max": int(words.max()),
        "n_empty": int((strs == "").sum()),
        "n_whitespace_only": int(((strs.str.strip() == "") & (strs != "")).sum()),
        "pct_digits": 100.0 * sum(c.isdigit() for c in joined) / n_chars,
        "pct_uppercase": 100.0 * sum(c.isupper() for c in joined) / n_chars,
        "pct_punctuation": 100.0 * len(re.findall(r"[^\w\s]", joined)) / n_chars,
        "top_tokens": [{"token": t, "count": c} for t, c in tokens.most_common(cfg.top_n)],
    }


def pattern_stats(s: pd.Series, semantic_type: str, config: StoneConfig | None = None) -> dict[str, Any]:
    """Stats for emails, URLs, phones, postal codes, IPs: lengths plus top domains when relevant."""
    cfg = config or StoneConfig()
    strs = s.dropna().astype(str).str.strip()
    out = _length_stats(strs)
    if semantic_type == "email":
        domains = strs.str.split("@").str[-1].str.lower()
    elif semantic_type == "url":
        domains = sample(strs, cfg.sample_size, cfg.random_state).map(
            lambda u: urlparse(u if "://" in u else "http://" + u).netloc.lower()
        )
    else:
        return out
    vc = domains.value_counts()
    out["n_domains"] = int(len(vc))
    out["top_domains"] = [{"domain": k, "count": int(v)} for k, v in vc.head(cfg.top_n).items()]
    return out


def structure_stats(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Python kinds (for mixed / JSON-like columns) and most frequent top-level keys."""
    cfg = config or StoneConfig()
    sub = sample(s.dropna(), cfg.sample_size, cfg.random_state)
    keys: Counter[str] = Counter()
    for v in sub.head(5_000):
        if isinstance(v, str) and v.lstrip().startswith("{"):
            try:
                v = json.loads(v)
            except ValueError:
                continue
        if isinstance(v, dict):
            keys.update(map(str, v.keys()))
    out: dict[str, Any] = {"python_kinds": python_kinds(sub)}
    if keys:
        out["top_keys"] = [{"key": k, "count": c} for k, c in keys.most_common(cfg.top_n)]
    return out


def id_stats(s: pd.Series) -> dict[str, Any]:
    nonnull = hashable(s.dropna())
    out: dict[str, Any] = {"n_duplicates": int(len(nonnull) - nonnull.nunique())}
    if pd.api.types.is_numeric_dtype(s.dtype):
        x = as_numeric(nonnull)
        out.update({"min": _f(x.min()), "max": _f(x.max()), "monotonic_increasing": bool(x.is_monotonic_increasing)})
    else:
        out.update(_length_stats(nonnull.astype(str)))
    return out


def column_stats(s: pd.Series, semantic_type: str | None = None, config: StoneConfig | None = None) -> dict[str, Any]:
    """Common + type-specific statistics for one column (type inferred when not given)."""
    cfg = config or StoneConfig()
    stype = semantic_type or infer_type(s, cfg)["semantic_type"]
    out = common_stats(s, cfg)
    with quiet():
        if stype in NUMERIC_TYPES or (stype == "ordinal-candidate" and pd.api.types.is_numeric_dtype(s.dtype)):
            out.update(numeric_stats(as_numeric(s), cfg))
        elif stype == "numeric_string":
            out.update(numeric_stats(as_numeric(s), cfg))
            out["n_unparsable"] = int(s.notna().sum() - parse_numeric(s, dominant_unit_only=False).notna().sum())
        if stype in CATEGORY_TYPES or stype == "ordinal-candidate":
            out.update(categorical_stats(s, cfg))
        if stype == "boolean":
            out.update(boolean_stats(s))
        elif stype in DATETIME_TYPES:
            out.update(datetime_stats(s, cfg))
        elif stype == "text":
            out.update(text_stats(s, cfg))
        elif stype in STRING_PATTERN_TYPES:
            out.update(pattern_stats(s, stype, cfg))
        elif stype in ("mixed", "json_like"):
            out.update(structure_stats(s, cfg))
        elif stype == "id":
            out.update(id_stats(s))
        elif stype == MULTI_LABEL:
            out.update(multilabel_stats(s, cfg))
    return out


def split_labels(s: pd.Series, sep: str) -> pd.Series:
    """Each non-missing cell as a list of stripped, non-empty labels."""
    parts = s.dropna().astype(str).str.split(sep, regex=False)
    return parts.map(lambda labels: [x.strip() for x in labels if x.strip()])


def multilabel_stats(s: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Lists of labels per cell: separator, labels per row, distinct labels and the most frequent ones."""
    cfg = config or StoneConfig()
    strs = s.dropna().astype(str)
    sep = list_separator(sample(strs, 5_000, cfg.random_state).str.strip()) or LIST_SEPARATORS[0]
    parts = split_labels(s, sep)
    counts = parts.str.len()
    vc = parts.explode().dropna().value_counts()
    n = max(len(parts), 1)
    return {
        "separator": sep,
        "n_labels": int(len(vc)),
        "labels_per_row_mean": float(counts.mean()) if len(counts) else 0.0,
        "labels_per_row_max": int(counts.max()) if len(counts) else 0,
        "pct_single_label": 100.0 * float((counts == 1).mean()) if len(counts) else 0.0,
        "top_labels": [{"label": _value(k), "count": int(v), "pct_rows": 100.0 * v / n}
                       for k, v in vc.head(cfg.top_n).items()],
    }


def flatten(stats_dict: dict[str, Any]) -> dict[str, Any]:
    """Keep only scalar entries (for tabular summaries)."""
    return {k: v for k, v in stats_dict.items() if v is None or isinstance(v, (str, int, float, bool, np.generic))}


def see_stats(df: pd.DataFrame | pd.Series, config: StoneConfig | None = None) -> pd.DataFrame:
    """Descriptive statistics for every column: one row per column, one column per scalar metric."""
    df = prepare(df)
    rows = []
    for col in df.columns:
        try:
            info = infer_type(df[col], config)
            row = {"semantic_type": info["semantic_type"], **flatten(column_stats(df[col], info["semantic_type"], config))}
        except Exception as exc:  # noqa: BLE001 - one bad column must not stop the rest
            row = {"error": repr(exc)}
        rows.append({"column": col, **row})
    if not rows:
        return pd.DataFrame(columns=["semantic_type"]).rename_axis("column")
    return pd.DataFrame(rows).set_index("column")
