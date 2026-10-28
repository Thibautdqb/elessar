"""Time series: recognise them quickly, then describe and prepare them.

Detection looks for a time column (a datetime column, dates stored as text, or a datetime row
index) and classifies the table as

- a **single series**: one row per timestamp;
- a **panel**: several series, one per entity (e.g. one per store), unique per (time, entity);
- an **event log**: timestamps that repeat without an entity; it becomes a series once
  aggregated per period.

Analysis of the main series (the target when numeric, else the most variable columns): sampling
frequency and gaps, trend (Kendall tau, Theil-Sen slope), stationarity (augmented Dickey-Fuller),
seasonality (autocorrelation peak among the periods natural to the frequency, strength from a
classical decomposition), then the treatment: aggregation, differencing, calendar features,
window sizes, a time-ordered validation split computed on the actual dates, and augmentation
recipes for the training folds only.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import hashable, quiet
from .config import StoneConfig
from .types import CATEGORY_TYPES, DATETIME_TYPES, NUMERIC_TYPES, as_datetime, as_numeric

# natural seasonal periods (in steps) per sampling frequency
_PERIODS = {"second": [60], "minute": [60, 1440], "hour": [24, 168], "day": [7, 30, 365], "week": [52],
            "month": [12], "quarter": [4], "year": []}
_STEP_SECONDS = [(1, "second"), (60, "minute"), (3600, "hour"), (86400, "day"), (7 * 86400, "week"),
                 (30.44 * 86400, "month"), (91.31 * 86400, "quarter"), (365.25 * 86400, "year")]
_PANDAS_FREQ = {"second": "s", "minute": "min", "hour": "h", "day": "D", "week": "W", "month": "MS", "quarter": "QS",
                "year": "YS"}
_ADF_CRIT = {"1%": (-3.43035, -6.5393, -16.786), "5%": (-2.86154, -2.8903, -4.234), "10%": (-2.56677, -1.5384, -2.809)}


def _unit(seconds: float) -> tuple[str, float]:
    """Closest named frequency to a step in seconds, and the step expressed in that unit."""
    best = min(_STEP_SECONDS, key=lambda s: abs(np.log(max(seconds, 1e-9) / s[0])))
    return best[1], seconds / best[0]


_YEAR = re.compile(r"^(year|yr|annee|année|jahr|año|ano|anno)$|[_ ](year|yr)$", re.I)


_MONTH = re.compile(r"^(month|mon|mois|monat|mes)$", re.I)


def _month_number(s: pd.Series) -> pd.Series | None:
    """1-12 from month numbers or names (Jan, January), or None."""
    if pd.api.types.is_numeric_dtype(s.dtype):
        m = pd.to_numeric(s, errors="coerce")
        return m if m.dropna().between(1, 12).all() and (m.dropna() % 1 == 0).all() else None
    text = s.astype("string").str.strip().str[:3].str.title()
    m = text.map({n: i for i, n in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
                                              "Nov", "Dec"], start=1)})
    return m.astype("float64") if m.notna().mean() >= 0.95 else None


def _time_columns(df: pd.DataFrame, types: dict[str, str]) -> dict[str, pd.Series]:
    """Candidate time axes: datetime columns, dates as text, year columns (with their month column if any)."""
    out = {}
    month_cols = [c for c in df.columns if _MONTH.search(str(c))]
    for col in df.columns:
        if _YEAR.search(str(col)) and pd.api.types.is_numeric_dtype(df[col].dtype):
            y = pd.to_numeric(df[col], errors="coerce")
            ok = y.dropna()
            if len(ok) and (ok % 1 == 0).all() and ok.between(1700, 2200).all() and ok.nunique() >= 2:
                month = next((m for m in map(lambda c: (c, _month_number(df[c])), month_cols) if m[1] is not None), None)
                if month is not None:
                    out[f"{col} + {month[0]}"] = pd.to_datetime(
                        pd.DataFrame({"year": y, "month": month[1], "day": 1}), errors="coerce")
                elif ok.nunique() >= 5:
                    out[col] = pd.to_datetime(y.astype("Int64").astype("string"), format="%Y", errors="coerce")
            continue
        if types.get(col) in DATETIME_TYPES:
            with quiet():
                t = as_datetime(df[col])
            if t.notna().mean() >= 0.8 and t.nunique() >= 10:
                out[col] = t
    return out


def detect(df: pd.DataFrame, types: dict[str, str]) -> dict[str, Any] | None:
    """The time structure of the table, or None when it is not time-indexed."""
    best: dict[str, Any] | None = None
    entities = [c for c in df.columns if types.get(c) in CATEGORY_TYPES or types.get(c) == "id"]
    for col, t in _time_columns(df, types).items():
        tt = t.dropna()
        if getattr(tt.dt, "tz", None) is not None:
            tt = tt.dt.tz_convert(None)
        unique = tt.drop_duplicates().sort_values()
        diffs = unique.diff().dropna().dt.total_seconds().to_numpy()
        if not len(diffs):
            continue
        step = float(np.median(diffs))
        regular = float(np.mean(np.isclose(diffs, step, rtol=0.05))) if step > 0 else 0.0
        unit, n_units = _unit(step)
        if unit == "month" or unit == "quarter" or unit == "year":  # calendar months vary in length
            regular = max(regular, float(np.mean(np.abs(diffs / step - 1) < 0.12)))
        info: dict[str, Any] = {"time_column": col, "step_seconds": step, "frequency": unit,
                                "step_in_unit": round(n_units, 2), "regularity": regular,
                                "start": unique.iloc[0], "end": unique.iloc[-1], "n_timestamps": len(unique),
                                "sorted": bool(t.dropna().is_monotonic_increasing), "entity": None}
        ratio = len(unique) / len(tt)
        if ratio >= 0.95 and regular >= 0.6:
            info["kind"] = "single series"
        else:
            info["kind"] = "event log"  # irregular or repeated timestamps: events, a series once aggregated
            for e in entities if regular >= 0.6 and ratio < 0.95 else []:
                if e == col or e in col.split(" + "):
                    continue
                key = pd.DataFrame({"t": t, "e": hashable(df[e])}).dropna()
                n_ent = key["e"].nunique()
                if 2 <= n_ent <= max(2, len(key) // 5) and not key.duplicated().mean() > 0.05:
                    info.update(kind="panel", entity=e, n_entities=int(n_ent))
                    break
        score = {"single series": 2, "panel": 2, "event log": 1}[info["kind"]] + regular
        if best is None or score > best["_score"]:
            best = {**info, "_score": score}
    if best is None:
        return None
    best.pop("_score")
    if best["n_timestamps"] < 12 or (best["kind"] == "event log" and len(df) < 100):
        return None
    return best


# ------------------------------------------------------------------------------------------- statistics
def adf_test(y: np.ndarray, max_lag: int | None = None) -> dict[str, Any]:
    """Augmented Dickey-Fuller with a constant, lag chosen by AIC; MacKinnon critical values.

    Rejecting (statistic below the critical value) means no unit root: the series is stationary.
    """
    y = np.asarray(y, dtype="float64")
    y = y[np.isfinite(y)][-5_000:]
    n = len(y)
    if n < 20 or np.ptp(y) == 0:
        return {"verdict": "insufficient data"}
    max_lag = max_lag if max_lag is not None else int(min(12 * (n / 100) ** 0.25, n // 4))
    dy = np.diff(y)
    best: tuple[float, float, int] | None = None
    for lag in range(max_lag + 1):
        rows = len(dy) - max_lag
        if rows < 10:
            break
        cols = [np.ones(rows), y[max_lag:-1]] + [dy[max_lag - i:len(dy) - i] for i in range(1, lag + 1)]
        x = np.column_stack(cols)
        target = dy[max_lag:]
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
        resid = target - x @ beta
        sigma2 = resid @ resid / (rows - x.shape[1])
        aic = rows * np.log(resid @ resid / rows) + 2 * x.shape[1]
        cov = sigma2 * np.linalg.pinv(x.T @ x)
        stat = beta[1] / np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.nan
        if best is None or aic < best[0]:
            best = (aic, float(stat), lag)
    if best is None or not np.isfinite(best[1]):
        return {"verdict": "insufficient data"}
    crit = {k: b + b1 / n + b2 / n**2 for k, (b, b1, b2) in _ADF_CRIT.items()}
    stationary = best[1] < crit["5%"]
    return {"statistic": best[1], "lag": best[2], "critical": crit, "stationary": bool(stationary),
            "verdict": "stationary" if stationary else "not stationary (unit root)"}


def acf(y: np.ndarray, max_lag: int) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    y = y - np.nanmean(y)
    y = np.where(np.isfinite(y), y, 0.0)
    n = len(y)
    max_lag = min(max_lag, n - 2)
    f = np.fft.rfft(y, n=2 ** int(np.ceil(np.log2(2 * n))))
    r = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return r / r[0] if r[0] else np.zeros(max_lag + 1)


def decompose(y: np.ndarray, period: int) -> dict[str, np.ndarray]:
    """Classical additive decomposition: centred moving-average trend, mean seasonal profile, residual."""
    s = pd.Series(y, dtype="float64")
    window = period if period % 2 else period + 1
    trend = s.rolling(window, center=True, min_periods=max(2, window // 2)).mean()
    if period % 2 == 0:
        trend = s.rolling(period, center=True, min_periods=period // 2).mean().rolling(2, center=True).mean()
    detrended = s - trend
    phase = np.arange(len(s)) % period
    profile = detrended.groupby(phase).mean()
    seasonal = profile.reindex(phase).to_numpy() - profile.mean()
    resid = s.to_numpy() - trend.to_numpy() - seasonal
    return {"observed": s.to_numpy(), "trend": trend.to_numpy(), "seasonal": seasonal, "resid": resid}


def _strength(part: np.ndarray, resid: np.ndarray) -> float:
    ok = np.isfinite(part) & np.isfinite(resid)
    if ok.sum() < 10:
        return 0.0
    den = np.var(part[ok] + resid[ok])
    return float(max(0.0, 1 - np.var(resid[ok]) / den)) if den > 0 else 0.0


def analyze_series(y: pd.Series, frequency: str, cfg: StoneConfig) -> dict[str, Any]:
    """Trend, stationarity, autocorrelation and seasonality of one regularly spaced series."""
    v = y.astype("float64").interpolate(limit_direction="both").to_numpy()
    n = len(v)
    out: dict[str, Any] = {"n": n}
    if n < 12 or np.nanstd(v) == 0:
        out["verdict"] = "too short or constant"
        return out
    idx = np.arange(n)
    with quiet():
        tau, p = stats.kendalltau(idx[-20_000:], v[-20_000:])
        step = max(1, n // 3_000)
        slope = stats.theilslopes(v[::step], idx[::step])[0]
    out.update(trend_tau=float(tau), trend_p=float(p), trend_slope=float(slope),
               trend="increasing" if p < cfg.alpha and tau > 0 else "decreasing" if p < cfg.alpha and tau < 0 else "none")
    out["adf"] = adf_test(v)
    candidates = [c for c in _PERIODS.get(frequency, []) if 2 <= c <= n // 2]
    max_lag = int(min(n // 2, max(candidates + [40]) * 1.2))
    r = acf(v, max_lag)
    out["acf"] = r
    out["acf_lag1"] = float(r[1]) if len(r) > 1 else None
    band = 1.96 / np.sqrt(n)
    out["acf_band"] = band
    period, best = None, 0.0
    detr = acf(v - np.polyval(np.polyfit(idx, v, 1), idx), max_lag)  # a trend would inflate every lag
    for c in candidates:
        w = max(1, c // 20)
        if c + w >= len(detr):
            continue
        peak = float(detr[c - w:c + w + 1].max())
        dip = float(detr[max(1, c // 2):c].min())  # a real cycle dips between lag 0 and its period
        if peak > max(best, band) and peak - dip >= 0.15:
            period, best = c, peak
    out["period"] = period
    if period:
        parts = decompose(v, period)
        out["seasonal_strength"] = _strength(parts["seasonal"], parts["resid"])
        out["trend_strength"] = _strength(parts["trend"], parts["resid"])
        out["decomposition"] = parts
        out["seasonality"] = "strong" if out["seasonal_strength"] >= 0.6 else "moderate" \
            if out["seasonal_strength"] >= 0.3 else "weak"
    else:
        out["seasonality"] = "none detected"
    return out


# ------------------------------------------------------------------------------------------- treatment
def _fmt(t: Any) -> str:
    t = pd.Timestamp(t)
    return f"{t:%Y-%m-%d}" if t == t.normalize() else f"{t:%Y-%m-%d %H:%M}"


def _splits(times: pd.Series, n_steps: int, period: int | None) -> list[dict[str, Any]]:
    """Expanding-window folds on the real dates: train on the past, validate on the next block."""
    test = max(period or 0, n_steps // 10, 1)
    n_splits = int(min(5, max(2, (n_steps - test) // max(test, 1) - 1)))
    if n_steps < 3 * test:
        return []
    folds = []
    for k in range(n_splits):
        test_end = n_steps - (n_splits - 1 - k) * test
        test_start = test_end - test
        if test_start < test:
            continue
        folds.append({"fold": k + 1, "train": f"{_fmt(times.iloc[0])} → {_fmt(times.iloc[test_start - 1])}",
                      "validate": f"{_fmt(times.iloc[test_start])} → {_fmt(times.iloc[test_end - 1])}",
                      "n_train": test_start, "n_validate": test})
    return folds


def treatment(info: dict[str, Any], main: dict[str, Any], n_steps: int) -> list[dict[str, str]]:
    """What to do with the series, as (step, advice, code) records."""
    t, freq = info["time_column"], info["frequency"]
    period = main.get("period")
    steps: list[dict[str, str]] = []
    if " + " in t:
        y, m = t.split(" + ")
        steps.append({"step": "Time column", "advice": f"the time is spread over {y} and {m}: build one date column",
                      "code": f"df['date'] = pd.to_datetime(df[{y!r}].astype(str) + '-' + df[{m!r}].astype(str), "
                              "format='mixed')"})
        t = "date"
    if not info["sorted"]:
        steps.append({"step": "Order", "advice": "the rows are not in time order: sort before anything else",
                      "code": f"df = df.sort_values({t!r})"})
    if info["kind"] == "event log":
        steps.append({"step": "Aggregation", "advice": f"timestamps repeat: aggregate per {info['aggregated_to']} to get "
                                                       "a regular series (counts, sums or means)",
                      "code": f"series = df.set_index({t!r}).resample({_PANDAS_FREQ[info['aggregated_to']]!r})"
                              ".agg(['count', 'mean'])"})
    elif info.get("n_missing_steps"):
        steps.append({"step": "Gaps", "advice": f"{info['n_missing_steps']:,} missing time steps: reindex on the full "
                                                "range, then interpolate short gaps",
                      "code": f"df = df.set_index({t!r}).asfreq({_PANDAS_FREQ[freq]!r})\n"
                              "df = df.interpolate(limit=3)"})
    if freq in ("second", "minute") and n_steps > 50_000:
        steps.append({"step": "Aggregation", "advice": "very fine sampling: resample to hours to remove noise and size",
                      "code": f"hourly = df.set_index({t!r}).resample('h').mean(numeric_only=True)"})
    if info["kind"] == "panel":
        e = info["entity"]
        steps.append({"step": "Panel", "advice": f"one series per {e}: compute lags and windows within each {e}, "
                                                 "never across them",
                      "code": f"df = df.sort_values([{e!r}, {t!r}])\n"
                              f"df['lag_1'] = df.groupby({e!r})['<value>'].shift(1)"})
    adf = main.get("adf") or {}
    if adf.get("stationary") is False:
        steps.append({"step": "Stationarity", "advice": "unit root (ADF): model the differences, or detrend",
                      "code": "df['<value>_diff'] = df['<value>'].diff()"})
    if period and main.get("seasonal_strength", 0) >= 0.3:
        steps.append({"step": "Seasonality", "advice": f"{main['seasonality']} seasonality of period {period} "
                                                       f"(strength {main['seasonal_strength']:.2f}): seasonal "
                                                       "differencing, or calendar / Fourier features",
                      "code": f"df['<value>_sdiff'] = df['<value>'].diff({period})\n"
                              f"k = np.arange(len(df))\ndf['sin_{period}'] = np.sin(2 * np.pi * k / {period})\n"
                              f"df['cos_{period}'] = np.cos(2 * np.pi * k / {period})"})
    windows = sorted({w for w in (period, 2 * period if period else None, 3, 7 if freq == "day" else None,
                                  24 if freq == "hour" else None, 12 if freq == "month" else None) if w and w < n_steps // 4})
    lags = sorted({lag for lag in (1, 2, period) if lag and lag < n_steps // 4})
    if windows or lags:
        parts = ([f"rolling windows of {', '.join(map(str, windows))} steps"] if windows else []) + (
            [f"lags {', '.join(map(str, lags))}"] if lags else [])
        steps.append({"step": "Windows", "advice": " and ".join(parts) + " as features, shifted by one step so they "
                                                                         "only see the past",
                      "code": "\n".join([f"df['lag_{lag}'] = df['<value>'].shift({lag})" for lag in lags]
                                        + [f"df['roll_mean_{w}'] = df['<value>'].shift(1).rolling({w}).mean()"
                                           for w in windows])})
    steps.append({"step": "Validation", "advice": "never shuffle: use expanding-window folds (below), with a gap "
                                                  "equal to the forecast horizon",
                  "code": f"from sklearn.model_selection import TimeSeriesSplit\n"
                          f"cv = TimeSeriesSplit(n_splits=5, test_size={max(period or 0, n_steps // 10, 1)}, gap=0)"})
    steps.append({"step": "Augmentation", "advice": "on the training folds only: jittering (noise at 3-5% of the "
                                                    "std), scaling, window slicing, magnitude warping",
                  "code": "rng = np.random.default_rng(0)\n"
                          "jittered = y + rng.normal(0, 0.03 * y.std(), len(y))\n"
                          "scaled = y * rng.normal(1, 0.1)\n"
                          "start = rng.integers(0, len(y) // 10); sliced = y[start:start + int(0.9 * len(y))]\n"
                          "warp = np.interp(np.arange(len(y)), np.linspace(0, len(y) - 1, 6), rng.normal(1, 0.1, 6))\n"
                          "warped = y * warp"})
    return steps


def timeseries(df: pd.DataFrame, types: dict[str, str], target: str | None, cfg: StoneConfig) -> dict[str, Any] | None:
    """Detect the time structure, then analyse the main series and propose its treatment."""
    info = detect(df, types)
    if info is None:
        return None
    t = info["time_column"]
    time = _time_columns(df, types)[t]
    if getattr(time.dt, "tz", None) is not None:
        time = time.dt.tz_convert(None)
    numeric = [c for c in df.columns if c not in t.split(" + ") and c != info.get("entity")
               and (types.get(c) in NUMERIC_TYPES or types.get(c) == "numeric_string")]
    values = ([target] if target in numeric else []) + sorted(
        [c for c in numeric if c != target],
        key=lambda c: -float(as_numeric(df[c], types[c]).std() / (abs(as_numeric(df[c], types[c]).mean()) + 1e-12)))
    values = values[:6]
    frame = pd.DataFrame({c: as_numeric(df[c], types[c]) for c in values}, index=df.index)
    frame.insert(0, "__time", time)
    frame = frame.dropna(subset=["__time"]).sort_values("__time")
    freq = info["frequency"]
    if info["kind"] == "event log":
        span = (info["end"] - info["start"]).total_seconds()
        unit = next((u for s, u in _STEP_SECONDS if span / s <= 400 and span / s >= 30), "day")
        info["aggregated_to"] = unit
        grouped = frame.set_index("__time").resample(_PANDAS_FREQ[unit])
        series = grouped.mean() if values else pd.DataFrame(index=grouped.size().index)
        series["rows"] = grouped.size()
        values = ["rows"] + values
        freq = unit
    elif info["kind"] == "panel":
        series = frame.groupby("__time")[values].mean() if values else frame.groupby("__time").size().to_frame("rows")
        if not values:
            values = ["rows"]
    else:
        series = frame.drop_duplicates("__time").set_index("__time")[values]
    if info["kind"] != "event log" and info["regularity"] >= 0.8:
        try:
            full = series.asfreq(_PANDAS_FREQ[freq]) if freq not in ("month", "quarter", "year") else series
            info["n_missing_steps"] = int(len(full) - len(series)) if len(full) >= len(series) else 0
            if len(full) <= 3 * len(series):
                series = full
        except (ValueError, TypeError):
            info["n_missing_steps"] = 0
    series = series.iloc[-50_000:]
    analyses = {c: analyze_series(series[c], freq, cfg) for c in values if c in series}

    def temporal(c: str) -> float:  # how much of a time series this column is
        a = analyses[c]
        return max(a.get("seasonal_strength", 0.0), abs(a.get("acf_lag1") or 0.0) * 0.8)

    ranked = sorted(analyses, key=temporal, reverse=True)
    if target in analyses:
        ranked = [target] + [c for c in ranked if c != target]
    values = ranked[:4]
    analyses = {c: analyses[c] for c in values}
    main_col = values[0] if values else None
    main = analyses.get(main_col, {}) if main_col else {}
    info.update(series=series, values=values, main=main_col, analyses=analyses, n_steps=len(series),
                folds=_splits(series.index.to_series(), len(series), main.get("period")),
                treatment=treatment(info, main, len(series)))
    return info


__all__ = ["acf", "adf_test", "analyze_series", "decompose", "detect", "timeseries"]
