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
from .roles import is_birth_name
from .types import CATEGORY_TYPES, DATETIME_TYPES, NUMERIC_TYPES, as_datetime, as_numeric

# natural seasonal periods (in steps) per sampling frequency
_PERIODS = {"second": [60], "minute": [60, 1440], "hour": [24, 168], "day": [7, 30, 365], "week": [52],
            "month": [12], "quarter": [4], "year": []}
_STEP_SECONDS = [(1, "second"), (60, "minute"), (3600, "hour"), (86400, "day"), (7 * 86400, "week"),
                 (30.44 * 86400, "month"), (91.31 * 86400, "quarter"), (365.25 * 86400, "year")]
_PANDAS_FREQ = {"second": "s", "minute": "min", "hour": "h", "day": "D", "week": "W", "month": "MS", "quarter": "QS",
                "year": "YS"}
SEASONAL_STRENGTH_MIN = 0.25
"""Seasonal strength 1 - var(remainder) / var(seasonal + remainder) needed to report a seasonality."""
# MacKinnon (2010), N=1 response surfaces, including the cubic finite-sample term.
# Checked against statsmodels.tsa.adfvalues.mackinnoncrit (BSD-licensed reference).
_ADF_CRIT = {"1%": (-3.43035, -6.5393, -16.786, -79.433),
             "5%": (-2.86154, -2.8903, -4.234, -40.04), "10%": (-2.56677, -1.5384, -2.809, 0.0)}


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
        if is_birth_name(col):  # who a row is about, not when it happened: never a time axis
            continue
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
                                "sorted": bool(t.dropna().is_monotonic_increasing), "entity": None,
                                "time_dtype": "year" if " + " not in col and types.get(col) in NUMERIC_TYPES else "date"}
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
                coverage = len(key) / n_ent / max(len(unique), 1)  # share of the time steps each entity has
                if 2 <= n_ent <= max(2, len(key) // 5) and not key.duplicated().mean() > 0.05 and coverage >= 0.2:
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
_ADF_CRIT_TREND = {"1%": (-3.95877, -9.0531, -28.428, -134.155),
                   "5%": (-3.41049, -4.3904, -9.036, -45.374), "10%": (-3.12705, -2.5856, -3.925, -22.38)}
_KPSS_CRIT = {"c": {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739},
              "ct": {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216}}


def adf_test(y: np.ndarray, max_lag: int | None = None, trend: bool = False) -> dict[str, Any]:
    """Augmented Dickey-Fuller (H0: unit root) with a constant, plus a linear trend when ``trend``;
    lag order by AIC, MacKinnon (2010) finite-sample critical values.
    """
    y = np.asarray(y, dtype="float64")
    y = y[np.isfinite(y)][-5_000:]
    n = len(y)
    if n < 20 or np.ptp(y) == 0:
        return {"verdict": "insufficient data"}
    max_lag = max_lag if max_lag is not None else int(min(12 * (n / 100) ** 0.25, n // 4))
    dy = np.diff(y)
    best: tuple[float, float, int] | None = None
    rows = len(dy) - max_lag
    if rows < 10:
        return {"verdict": "insufficient data"}
    for lag in range(max_lag + 1):  # same estimation sample for every lag, so the AICs are comparable
        cols = [np.ones(rows), y[max_lag:-1]] + ([np.arange(rows, dtype="float64")] if trend else [])
        cols += [dy[max_lag - i:len(dy) - i] for i in range(1, lag + 1)]
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
    table = _ADF_CRIT_TREND if trend else _ADF_CRIT
    crit = {k: b + b1 / rows + b2 / rows**2 + b3 / rows**3 for k, (b, b1, b2, b3) in table.items()}
    reject = best[1] < crit["5%"]
    return {"statistic": best[1], "lag": best[2], "nobs": rows, "critical": crit,
            "reject_unit_root": bool(reject), "trend": trend}


def kpss_test(y: np.ndarray, trend: bool = False) -> dict[str, Any]:
    """KPSS (H0: level-stationary, or trend-stationary with ``trend``), Bartlett-kernel long-run variance
    with Schwert's lag rule; p-value interpolated in the Kwiatkowski et al. (1992) table (0.01-0.10)."""
    y = np.asarray(y, dtype="float64")
    y = y[np.isfinite(y)][-5_000:]
    n = len(y)
    if n < 20 or np.ptp(y) == 0:
        return {"verdict": "insufficient data"}
    t = np.arange(n, dtype="float64")
    e = y - (np.polyval(np.polyfit(t, y, 1), t) if trend else y.mean())
    s = np.cumsum(e)
    lags = min(int(12 * (n / 100) ** 0.25), n - 1)
    lrv = e @ e / n
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    stat = float(s @ s / (n * n * lrv)) if lrv > 0 else np.nan
    table = _KPSS_CRIT["ct" if trend else "c"]
    levels, crits = np.array(list(table)), np.array(list(table.values()))
    p = float(np.interp(stat, crits, levels, left=0.10, right=0.01)) if np.isfinite(stat) else np.nan  # crits increase
    return {"statistic": stat, "p_value": p, "p_bounded": stat <= crits[0] or stat >= crits[-1],
            "reject_stationarity": bool(np.isfinite(stat) and stat > table[0.05]), "lags": lags, "trend": trend}


def stationarity(y: np.ndarray) -> dict[str, Any]:
    """ADF and KPSS read together (each has the other's null): the usual 2 x 2 decision."""
    adf_c, kpss_c = adf_test(y), kpss_test(y)
    adf_t, kpss_t = adf_test(y, trend=True), kpss_test(y, trend=True)
    if "statistic" not in adf_c or "statistic" not in kpss_c:
        return {"verdict": "insufficient data", "adf": adf_c, "kpss": kpss_c}
    if adf_c["reject_unit_root"] and not kpss_c["reject_stationarity"]:
        verdict = "stationary"
    elif adf_t.get("reject_unit_root") and not kpss_t.get("reject_stationarity", True):
        verdict = "trend-stationary"
    elif not adf_c["reject_unit_root"] and kpss_c["reject_stationarity"]:
        verdict = "unit root"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "adf": adf_c, "kpss": kpss_c, "adf_trend": adf_t, "kpss_trend": kpss_t,
            "stationary": verdict == "stationary"}


def _ar_acf(v: np.ndarray, max_lag: int, max_order: int = 8) -> np.ndarray:
    """Autocorrelation at lags 1..max_lag of an AR(p) fitted by Yule-Walker, the order chosen by AIC."""
    from scipy.linalg import solve_toeplitz

    n = len(v)
    r = acf(v, max_order)
    best_aic, phi = 0.0, np.empty(0)  # order 0: white noise
    for p in range(1, min(max_order, len(r) - 1) + 1):
        coef = solve_toeplitz(r[:p], r[1:p + 1])
        aic = np.log(max(1.0 - coef @ r[1:p + 1], 1e-12)) + 2 * p / n
        if aic < best_aic:
            best_aic, phi = aic, coef
    rho = np.zeros(max_lag + 1)
    rho[0] = 1.0
    p = len(phi)
    rho[1:p + 1] = r[1:p + 1][:max_lag]
    for k in range(p + 1, max_lag + 1):
        rho[k] = phi @ rho[k - p:k][::-1]
    return rho[1:]


def mann_kendall(y: np.ndarray, alpha: float = 0.05, max_n: int = 5_000) -> dict[str, Any]:
    """Mann-Kendall trend test with a correction for autocorrelation, and the Sen (Theil) slope.

    The variance of S is inflated as in Hamed & Rao (1998), by the autocorrelation of the detrended series,
    taken from two models: an AR(1) whose lag-1 coefficient is corrected for its small-sample bias (Kendall),
    and an AR(p) fitted to the ranks with its order chosen by AIC; the larger inflation is kept. Hamed and Rao
    keep only the significant sample autocorrelations, which are biased low: on 200 points with AR(1) 0.8
    noise and no trend, that version still rejected 25% of the series at 5%, this one 6% (strong long
    memory remains out of reach: trend and long memory are hardly distinguishable on a short series).
    """
    v = np.asarray(y, dtype="float64")
    v = v[np.isfinite(v)][-max_n:]
    n = len(v)
    if n < 10 or np.ptp(v) == 0:
        return {"n": n, "trend": "none", "p_value": np.nan}
    s = 0.0
    for i in range(n - 1):  # S = sum of sign(x_j - x_i), j > i (O(n^2), vectorised by row)
        s += np.sign(v[i + 1:] - v[i]).sum()
    _, ties = np.unique(v, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0
    idx = np.arange(n, dtype="float64")
    with quiet():
        slope = float(stats.theilslopes(v, idx)[0]) if n <= 2_000 else float(stats.theilslopes(v[::n // 2_000 + 1],
                                                                                              idx[::n // 2_000 + 1])[0])
    resid = v - slope * idx
    k = np.arange(1, max(1, n // 3) + 1)
    weight = 2 / (n * (n - 1) * (n - 2)) * (n - k) * (n - k - 1) * (n - k - 2.0)
    r1 = float(acf(resid, 1)[1]) if np.ptp(resid) > 0 else 0.0
    r1 = min(r1 + (1 + 4 * r1) / n, 0.99)  # Kendall's correction of the downward bias of r1
    inflation_ar1 = 1 + np.sum(weight * np.power(r1, k))
    inflation_arp = 1 + np.sum(weight * _ar_acf(stats.rankdata(resid), len(k)))
    factor = max(inflation_ar1, inflation_arp, 1.0)  # never claim more information than independent data

    def z_of(var: float) -> float:
        return (s - np.sign(s)) / np.sqrt(var) if var > 0 and s != 0 else 0.0

    z, z_naive = z_of(var_s * factor), z_of(var_s)
    p = float(2 * stats.norm.sf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    return {"n": n, "tau": float(tau), "s": float(s), "slope": slope, "z": float(z), "p_value": p,
            "p_value_naive": float(2 * stats.norm.sf(abs(z_naive))), "variance_inflation": float(factor),
            "trend": "increasing" if p < alpha and s > 0 else "decreasing" if p < alpha and s < 0 else "none"}


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


def bartlett_se(r: np.ndarray, n: int) -> np.ndarray:
    """Standard error of each autocorrelation under H0 "no correlation beyond lag k - 1" (Bartlett):
    sqrt((1 + 2 sum_{j<k} r_j^2) / n). The white-noise band 1.96 / sqrt(n) is its lag-1 case."""
    cum = np.concatenate([[0.0, 0.0], np.cumsum(r[1:-1] ** 2)])[:len(r)]  # sum over lags 1 .. k - 1
    return np.sqrt((1 + 2 * cum) / n)


def analyze_series(y: pd.Series, frequency: str, cfg: StoneConfig) -> dict[str, Any]:
    """Trend, stationarity, autocorrelation and seasonality of one regularly spaced series.

    - trend: Mann-Kendall with a correction for autocorrelation (see :func:`mann_kendall`), Sen slope;
    - stationarity: ADF and KPSS, with and without a linear trend, read together;
    - seasonality: a candidate period is screened on the autocorrelation of the linearly detrended
      series (a peak above the white-noise band at a period natural to the sampling frequency, a dip
      before it, two full cycles), then kept when the seasonal strength of a classical decomposition
      reaches 0.25. The decision rests on this effect size: a significance test of seasonality
      would itself be invalidated by the autocorrelation;
    - stationarity is then tested on the seasonally adjusted series.
    """
    raw = y.astype("float64")
    n_gaps = int(raw.isna().sum())
    v = raw.interpolate(limit_direction="both").to_numpy()
    n = len(v)
    out: dict[str, Any] = {"n": n, "n_interpolated": n_gaps}
    if n < 12 or np.nanstd(v) == 0:
        out["verdict"] = "too short or constant"
        return out
    mk = mann_kendall(v, cfg.alpha)
    out.update(mann_kendall=mk, trend=mk["trend"], trend_tau=mk.get("tau"), trend_p=mk.get("p_value"),
               trend_slope=mk.get("slope"))
    idx = np.arange(n)
    candidates = [c for c in _PERIODS.get(frequency, []) if 2 <= c <= n // 2]
    max_lag = int(min(n // 2, max(candidates + [40]) * 1.2))
    r = acf(v, max_lag)
    out["acf"] = r
    out["acf_lag1"] = float(r[1]) if len(r) > 1 else None
    out["acf_band"] = 1.96 / np.sqrt(n)
    out["acf_se"] = bartlett_se(r, n)  # the band drawn on the autocorrelation chart
    detr = acf(v - np.polyval(np.polyfit(idx, v, 1), idx), max_lag)  # a trend would inflate every lag
    white = 1.96 / np.sqrt(n)
    period, best, parts = None, 0.0, None
    for c in candidates:  # screening: a peak at a natural period, and a real cycle (a dip before it)
        w = max(1, c // 20)
        if c + w >= len(detr) or n < 2 * c:
            continue
        peak = float(detr[c - w:c + w + 1].max())
        dip = float(detr[max(1, c // 2):c].min())
        if peak > max(best, white) and peak - dip >= 0.15:
            trial = decompose(v, c)
            strength = _strength(trial["seasonal"], trial["resid"])
            if strength >= SEASONAL_STRENGTH_MIN:  # decision on the effect size
                period, best, parts = c, peak, trial
    out["period"] = period
    if period:
        out["seasonal_acf"] = best
        out["seasonal_strength"] = _strength(parts["seasonal"], parts["resid"])
        out["trend_strength"] = _strength(parts["trend"], parts["resid"])
        out["decomposition"] = parts
        out["seasonality"] = "strong" if out["seasonal_strength"] >= 0.6 else "moderate" \
            if out["seasonal_strength"] >= 0.4 else "weak"
        adjusted = v - parts["seasonal"]  # unit-root tests assume no seasonal pattern left
    else:
        out["seasonality"] = "none detected"
        adjusted = v
    st = stationarity(adjusted)
    st["seasonally_adjusted"] = period is not None
    out["stationarity"] = st
    out["adf"] = st.get("adf", {})
    return out


def panel_summary(df: pd.DataFrame, time: pd.Series, entity: str, values: list[str], types: dict[str, str],
                  frequency: str, cfg: StoneConfig, max_entities: int = 200) -> dict[str, Any]:
    """Per-entity analysis of a panel: each entity's own series, then the share of entities with a
    significant trend (Benjamini-Hochberg over the entities), stationarity verdicts, seasonality.

    A mean across entities would mix the entities' dynamics with the change in which entities are
    present; per-entity statistics do not.
    """
    from .correlations import bh_adjust

    frame = pd.DataFrame({"t": time, "e": hashable(df[entity])})
    for c in values:
        frame[c] = as_numeric(df[c], types[c])
    frame = frame.dropna(subset=["t", "e"])
    ents = frame["e"].unique()
    rng = np.random.default_rng(cfg.random_state)
    chosen = rng.choice(ents, size=min(len(ents), max_entities), replace=False)
    out: dict[str, Any] = {"n_entities": int(len(ents)), "n_analysed": int(len(chosen)), "per_value": {}}
    lengths = frame.groupby("e").size()
    out["length"] = {"median": float(lengths.median()), "min": int(lengths.min()), "max": int(lengths.max())}
    for c in values:
        rows = []
        for e in chosen:
            ys = frame.loc[frame["e"] == e].sort_values("t")[c].dropna().to_numpy()
            if len(ys) < 12 or np.ptp(ys) == 0:
                continue
            mk = mann_kendall(ys, cfg.alpha)
            st = stationarity(ys) if len(ys) >= 20 else {"verdict": "insufficient data"}
            r = acf(ys, 1)
            rows.append({"p": mk["p_value"], "s": mk.get("s", 0.0), "verdict": st["verdict"],
                         "lag1": float(r[1]) if len(r) > 1 else np.nan})
        if not rows:
            continue
        t = pd.DataFrame(rows)
        t["p_adj"] = bh_adjust(t["p"])
        sig = t["p_adj"] < cfg.alpha
        out["per_value"][c] = {
            "n_series": int(len(t)),
            "pct_increasing": 100.0 * float((sig & (t["s"] > 0)).mean()),
            "pct_decreasing": 100.0 * float((sig & (t["s"] < 0)).mean()),
            "stationarity": {k: 100.0 * v for k, v in t["verdict"].value_counts(normalize=True).items()},
            "median_lag1": float(t["lag1"].median()),
        }
    return out


def _trim_edges(series: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, freq: str) -> tuple[pd.DataFrame, int]:
    """Drop the first / last period when the data only covers part of it (their counts are biased low)."""
    if len(series) < 3:
        return series, 0
    off = pd.tseries.frequencies.to_offset(_PANDAS_FREQ[freq])
    first, last = series.index[0], series.index[-1]
    drop = []
    if start > first + (first + off - first) * 0.1:
        drop.append(first)
    if end < last + (last + off - last) * 0.9:
        drop.append(last)
    return series.drop(index=drop), len(drop)


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


def treatment(info: dict[str, Any], main: dict[str, Any], n_steps: int, value: str | None = None,
              values: list[str] | None = None) -> list[dict[str, str]]:
    """What to do with the series, as (step, advice, code) records. The code names the analysed column; an
    event log is aggregated first into ``series`` (rows per period and the mean of each value), which the
    later steps then use."""
    t, freq = info["time_column"], info["frequency"]
    frame = "series" if info["kind"] == "event log" else "df"
    value = value or "value"
    v = f"{frame}[{value!r}]"
    period = main.get("period")
    steps: list[dict[str, str]] = []
    if " + " in t:
        y, m = t.split(" + ")
        steps.append({"step": "Time column", "advice": f"the time is spread over {y} and {m}: build one date column",
                      "code": f"df['date'] = pd.to_datetime(df[{y!r}].astype(str) + '-' + df[{m!r}].astype(str), "
                              "format='mixed')"})
        t = "date"
    elif info.get("time_dtype") == "year":
        steps.append({"step": "Time column", "advice": f"{t} holds years: turn it into dates first",
                      "code": f"df[{t!r}] = pd.to_datetime(df[{t!r}].astype(int).astype(str), format='%Y')"})
    if not info["sorted"]:
        steps.append({"step": "Order", "advice": "the rows are not in time order: sort before anything else",
                      "code": f"df = df.sort_values({t!r})"})
    if info["kind"] == "event log":
        steps.append({"step": "Aggregation", "advice": f"timestamps repeat: aggregate per {info['aggregated_to']} to get "
                                                       "a regular series (counts, sums or means)",
                      "code": "\n".join(
                          [f"s = df.set_index(pd.to_datetime(df[{t!r}])).sort_index()",
                           f"series = s.resample({_PANDAS_FREQ[info['aggregated_to']]!r}).size().to_frame('rows')"]
                          + [f"series[{c!r}] = elessar.types.as_numeric(s[{c!r}])"
                             f".resample({_PANDAS_FREQ[info['aggregated_to']]!r}).mean()"
                             for c in (values or []) if c != "rows"])})
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
                              f"df['lag_1'] = df.groupby({e!r})[{value!r}].shift(1)"})
    diff_value = f"df.groupby({info['entity']!r})[{value!r}]" if info["kind"] == "panel" else v
    verdict = (main.get("stationarity") or {}).get("verdict")
    if verdict == "unit root":
        steps.append({"step": "Stationarity", "advice": "unit root (ADF does not reject it, KPSS rejects stationarity): "
                                                        "model the first differences",
                      "code": f"{frame}[{value + '_diff'!r}] = {diff_value}.diff()"})
    elif verdict == "trend-stationary":
        steps.append({"step": "Stationarity", "advice": "stationary around a linear trend (ADF and KPSS with a trend "
                                                        "agree): remove the trend, do not difference (that would add "
                                                        "noise)",
                      "code": f"y = {v}\nk = np.arange(len(y))\nok = y.notna().to_numpy()\n"
                              f"{frame}[{value + '_detrended'!r}] = y - np.polyval(np.polyfit(k[ok], y[ok], 1), k)"})
    elif verdict == "inconclusive":
        steps.append({"step": "Stationarity", "advice": "ADF and KPSS disagree (both reject, or neither): a structural "
                                                        "break, changing variance or long memory; compare models on "
                                                        "levels and on differences",
                      "code": f"{frame}[{value + '_diff'!r}] = {diff_value}.diff()  # try both versions in validation"})
    if period and main.get("seasonal_strength", 0) >= 0.3:
        steps.append({"step": "Seasonality", "advice": f"{main['seasonality']} seasonality of period {period} "
                                                       f"(strength {main['seasonal_strength']:.2f}): seasonal "
                                                       "differencing, or calendar / Fourier features",
                      "code": f"{frame}[{value + '_sdiff'!r}] = {diff_value}.diff({period})\n"
                              f"k = np.arange(len({frame}))\n{frame}['sin_{period}'] = np.sin(2 * np.pi * k / {period})\n"
                              f"{frame}['cos_{period}'] = np.cos(2 * np.pi * k / {period})"})
    windows = sorted({w for w in (period, 2 * period if period else None, 3, 7 if freq == "day" else None,
                                  24 if freq == "hour" else None, 12 if freq == "month" else None) if w and w < n_steps // 4})
    lags = sorted({lag for lag in (1, 2, period) if lag and lag < n_steps // 4})
    if windows or lags:
        parts = ([f"rolling windows of {', '.join(map(str, windows))} steps"] if windows else []) + (
            [f"lags {', '.join(map(str, lags))}"] if lags else [])
        # a panel shifts within each entity: a global shift would hand one entity's last value to the next one
        g = f"df.groupby({info['entity']!r})[{value!r}]" if info["kind"] == "panel" and info.get("entity") else None
        steps.append({"step": "Windows", "advice": " and ".join(parts) + " as features, shifted by one step so they "
                                                                         "only see the past"
                                                  + (f", within each {info['entity']}" if g else ""),
                      "code": "\n".join([f"{frame}['lag_{lag}'] = {g or v}.shift({lag})" for lag in lags]
                                        + [f"{frame}['roll_mean_{w}'] = "
                                           + (f"{g}.transform(lambda s: s.shift(1).rolling({w}).mean())" if g
                                              else f"{v}.shift(1).rolling({w}).mean()")
                                           for w in windows])})
    steps.append({"step": "Validation", "advice": "never shuffle: use expanding-window folds (below), with a gap "
                                                  "equal to the forecast horizon",
                  "code": f"from sklearn.model_selection import TimeSeriesSplit\n"
                          f"cv = TimeSeriesSplit(n_splits=5, test_size={max(period or 0, n_steps // 10, 1)}, gap=0)"})
    steps.append({"step": "Augmentation", "advice": "on the training folds only: jittering (noise at 3-5% of the "
                                                    "std), scaling, window slicing, magnitude warping",
                  "code": f"y = {v}.dropna().to_numpy()  # the training part only\n"
                          "rng = np.random.default_rng(0)\n"
                          "jittered = y + rng.normal(0, 0.03 * y.std(), len(y))\n"
                          "scaled = y * rng.normal(1, 0.1)\n"
                          "start = rng.integers(0, len(y) // 10); sliced = y[start:start + int(0.9 * len(y))]\n"
                          "warp = np.interp(np.arange(len(y)), np.linspace(0, len(y) - 1, 6), rng.normal(1, 0.1, 6))\n"
                          "warped = y * warp"})
    return steps


def _value_columns(df: pd.DataFrame, info: dict[str, Any], types: dict[str, str], target: str | None) -> list[str]:
    """The numeric columns worth following in time (the target first), at most six."""
    t = info["time_column"]
    numeric = [c for c in df.columns if c not in t.split(" + ") and c != info.get("entity")
               and (types.get(c) in NUMERIC_TYPES or types.get(c) == "numeric_string")]
    def spread(c: str) -> float:
        x = as_numeric(df[c], types[c])
        return -float(x.std() / (abs(x.mean()) + 1e-12))
    return (([target] if target in numeric else []) + sorted([c for c in numeric if c != target], key=spread))[:6]


def finest_unit(frequency: str, start: pd.Timestamp, end: pd.Timestamp, max_periods: int = 200_000) -> str:
    """The sampling frequency itself, or the finest coarser unit that keeps at most ``max_periods``."""
    span = max((end - start).total_seconds(), 1.0)
    names = [u for _, u in _STEP_SECONDS]
    for s, u in _STEP_SECONDS[names.index(frequency) if frequency in names else 0:]:
        if span / s <= max_periods:
            return u
    return "year"


def serial_structure(df: pd.DataFrame, types: dict[str, str], columns: list[str] | None = None,
                     max_points: int = 5_000) -> dict[str, Any] | None:
    """The time order of a table indexed by time, for tests that must account for it.

    Returns ``{"time", "entity", "differences"}``: the time of each row, its entity for a panel (else None),
    and the first differences (in time order, within each entity) of the numeric columns that are not shown to
    be stationary: ADF and KPSS read together on their series (the longest entities of a panel) say "unit
    root" or cannot decide. Two random walks look associated whatever they are; their changes do not.
    ``columns`` skips the tests and differences these. None for a table without a regular time axis.
    """
    info = detect(df, types)
    if info is None or info["kind"] == "event log":
        return None
    t = info["time_column"]
    time = _time_columns(df, types).get(t)
    if time is None:
        return None
    entity = info.get("entity") if info["kind"] == "panel" else None
    parts = [c.strip() for c in t.split(" + ")]
    numeric = [c for c in df.columns if types.get(c) in NUMERIC_TYPES and c not in parts and c != entity
               and types.get(c) != "boolean"]
    key = pd.DataFrame({"t": time.to_numpy()}, index=df.index)
    if entity is not None:
        key["e"] = hashable(df[entity]).astype("string").to_numpy()
    key = key[key["t"].notna()]
    order = key.sort_values(["e", "t"] if entity is not None else ["t"], kind="stable").index
    groups = key.loc[order, "e"] if entity is not None else pd.Series(0, index=order)
    longest = groups.value_counts().index[:3]
    diffs = {}
    with quiet():
        for c in numeric if columns is None else [c for c in columns if c in df.columns]:
            v = pd.Series(as_numeric(df.loc[order, c], types.get(c)).to_numpy(dtype="float64"), index=order)
            if columns is None:
                verdicts = []
                for g in longest:
                    series = v[(groups == g).to_numpy()].dropna().to_numpy()[-max_points:]
                    if len(series) >= 30 and np.ptp(series) > 0:
                        verdicts.append(stationarity(series)["verdict"] in ("unit root", "inconclusive"))
                if not verdicts or sum(verdicts) * 2 <= len(verdicts):
                    continue
            diffs[c] = v.groupby(groups.to_numpy()).diff()
    return {"time": time, "entity": hashable(df[entity]).astype("string") if entity is not None else None,
            "differences": pd.DataFrame(diffs).reindex(df.index) if diffs else None}


def aggregation_unit(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """The calendar unit that turns events into a series of 30 to 400 periods."""
    span = (end - start).total_seconds()
    return next((u for s, u in _STEP_SECONDS if 30 <= span / s <= 400), "day")


def timeseries(df: pd.DataFrame, types: dict[str, str], target: str | None, cfg: StoneConfig,
               exact: dict[str, Any] | None = None, info: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Detect the time structure, then analyse the series and propose its treatment.

    ``exact``: for a table streamed in chunks, the series aggregated per period over every row
    (``{"series": frame, "unit": ...}``). Temporal statistics are then computed on it, never on a
    thinned random sample (thinning destroys autocorrelation); ``df`` is only used for the
    per-entity analysis of a panel (whole entities drawn at random).
    """
    info = dict(info) if info is not None else detect(df, types)
    if info is None:
        return None
    t = info["time_column"]
    values = _value_columns(df, info, types, target)
    freq = info["frequency"]
    if exact is not None:
        series = exact["series"].copy()
        freq = exact["unit"]
        if info["kind"] == "event log":
            info["aggregated_to"] = freq
        values = [c for c in ["rows"] + values if c in series] if info["kind"] == "event log" else \
            [c for c in values if c in series] or ["rows"]
        info["exact"] = True
        info["entity_fraction"] = exact.get("entity_fraction")
        if len(series):  # the true range, over every row (the detection only saw part of the table)
            info["start"], info["end"] = series.index.min(), series.index.max()
    else:
        time = _time_columns(df, types)[t]
        if getattr(time.dt, "tz", None) is not None:
            time = time.dt.tz_convert(None)
        frame = pd.DataFrame({c: as_numeric(df[c], types[c]) for c in values}, index=df.index)
        frame.insert(0, "__time", time)
        frame = frame.dropna(subset=["__time"]).sort_values("__time")
        if info["kind"] == "event log":
            freq = aggregation_unit(info["start"], info["end"])
            info["aggregated_to"] = freq
            grouped = frame.set_index("__time").resample(_PANDAS_FREQ[freq])
            series = grouped.mean() if values else pd.DataFrame(index=grouped.size().index)
            series["rows"] = grouped.size()
            values = ["rows"] + values
        elif info["kind"] == "panel":
            grouped = frame.groupby("__time")
            series = grouped[values].mean() if values else grouped.size().to_frame("rows")
            series["active entities"] = frame.assign(e=hashable(df[info["entity"]])).groupby("__time")["e"].nunique()
            values = values or ["rows"]
        else:
            series = frame.drop_duplicates("__time").set_index("__time")[values]
        info["exact"] = True  # every row was used
    if info["kind"] == "event log":
        series, info["n_trimmed_periods"] = _trim_edges(series, info["start"], info["end"], freq)
    if info["kind"] != "event log" and info["regularity"] >= 0.8 and exact is None:
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
    if info["kind"] == "panel" and info.get("entity") in df.columns:
        try:
            time = _time_columns(df, types)[t]
            info["panel"] = panel_summary(df, time, info["entity"], [v for v in values if v in df.columns], types,
                                          freq, cfg)
        except Exception:  # noqa: BLE001 - the per-entity summary is a complement, never a blocker
            info["panel"] = None
    info.update(series=series, values=values, main=main_col, analyses=analyses, n_steps=len(series),
                frequency_analysed=freq, folds=_splits(series.index.to_series(), len(series), main.get("period")),
                treatment=treatment(info, main, len(series), main_col, values))
    return info


__all__ = ["acf", "adf_test", "aggregation_unit", "analyze_series", "bartlett_se", "decompose", "detect",
           "kpss_test", "mann_kendall", "panel_summary", "stationarity", "timeseries"]
