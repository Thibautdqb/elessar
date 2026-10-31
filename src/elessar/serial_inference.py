"""Conservative truncated time-shift tests for regular series and balanced panels.

Yuan & Shou (2024), doi:10.1371/journal.pbio.3002758, and Harris (2020),
arXiv:2012.06862. For a fixed statistic and radius r, B/(r+1), clipped at 1,
is superuniform under process independence if the shifted process is strictly
stationary. We take the maximum of the two orientations, hence require at least
one process to be stationary without choosing an orientation from the results.
No circular shifts, fitted autocorrelation model or stationarity pretest is used.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import hashable
from .config import StoneConfig


def serial_layout(df: pd.DataFrame, serial: dict[str, Any], cfg: StoneConfig) -> dict[str, Any]:
    """Fixed chronological grid; shift whole entity vectors together, preserving cross-entity dependence."""
    if not isinstance(cfg.serial_shift_radius, int) or cfg.serial_shift_radius < 1:
        raise ValueError("serial_shift_radius must be a positive integer")
    if not isinstance(cfg.serial_max_points, int) or cfg.serial_max_points < 8:
        raise ValueError("serial_max_points must be an integer >= 8")
    out: dict[str, Any] = {"df": df, "detail": "", "n_time": 0, "n_entities": 0}

    def aligned(values: pd.Series) -> pd.Series:
        return values if values.index.equals(df.index) else values.reindex(df.index)

    time = pd.to_datetime(aligned(serial["time"]), errors="coerce")
    key = pd.DataFrame({"time": time.to_numpy(), "position": np.arange(len(df))})
    entity = serial.get("entity")
    key["entity"] = hashable(aligned(entity)).astype("string").to_numpy() if entity is not None else "single"
    if key[["time", "entity"]].isna().any().any():
        out["detail"] = "missing timestamps or entities; no IID fallback"
        return out
    if key.duplicated(["time", "entity"]).any():
        out["detail"] = "duplicate entity/time observations; no IID fallback"
        return out
    dates = pd.DatetimeIndex(key["time"].unique()).sort_values()
    n_entities = int(key["entity"].nunique())
    out.update(n_time=len(dates), n_entities=n_entities)
    if len(key) != len(dates) * n_entities:
        out["detail"] = "unbalanced panel: synchronous complete entity vectors required; no IID fallback"
        return out
    if len(dates) < 8 or pd.infer_freq(dates) is None:
        out["detail"] = "at least 8 regular time points required; no IID fallback"
        return out
    cap = min(cfg.serial_max_points, cfg.sample_size // n_entities)
    if cap < 8:
        out["detail"] = "complete entity vectors exceed the sampling budget; no IID fallback"
        return out
    order = key.sort_values(["time", "entity"], kind="stable")["position"].to_numpy().reshape(-1, n_entities)[-cap:]
    out.update(order=order, n_time=len(order), first_time=dates[-len(order)], last_time=dates[-1])
    return out


def _measure(a: np.ndarray, b: np.ndarray, kind: str, n_b: int) -> float:
    if kind == "kendall":
        return float(stats.kendalltau(a, b, method="asymptotic").statistic)
    if kind == "auc":
        present = a == 1  # positive class is the missing indicator, fixed independently of class frequency
        n1, n0 = int(present.sum()), int((~present).sum())
        if not n1 or not n0:
            return float("nan")
        u = stats.rankdata(b)[present].sum() - n1 * (n1 + 1) / 2
        return float(2 * u / (n1 * n0) - 1)  # centred strength; two-sided via abs in the TTS rule
    if kind == "target_numeric":
        ra, rb = stats.rankdata(a), stats.rankdata(b)
        ca, cb = ra - ra.mean(), rb - rb.mean()
        denominator = np.sqrt((ca @ ca) * (cb @ cb))
        mono = abs(float((ca @ cb) / denominator)) if denominator > 0 else np.nan
        # Recompute the prescribed rank-decile rule for every surrogate. Ties share a bin.
        edges = np.unique(np.quantile(ra, np.linspace(0., 1., 11)))[1:-1]
        groups = np.searchsorted(edges, ra, side="left")
        shape = _measure(groups, b, "eta", 0)
        available = [v for v in (mono, shape) if np.isfinite(v)]
        return max(available) if available else float("nan")
    if kind == "cramers_v":
        table = np.bincount(a * n_b + b, minlength=(int(a.max()) + 1) * n_b).reshape(-1, n_b)
        table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0].astype(float)
        r, k = table.shape
        n = table.sum()
        if r < 2 or k < 2:
            return float("nan")
        expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / n
        chi2 = float(((table - expected) ** 2 / expected).sum())
        denominator = min(k - 1 - (k - 1) ** 2 / (n - 1), r - 1 - (r - 1) ** 2 / (n - 1))
        return float(np.sqrt(max(0., chi2 / n - (k - 1) * (r - 1) / (n - 1)) / denominator)) if denominator > 0 else float("nan")
    # eta: a is categorical, b numeric. Scaling b first prevents squared-moment overflow.
    b = b / (float(np.abs(b).max()) or 1.)
    counts = np.bincount(a)
    sums = np.bincount(a, weights=b)
    mean = b.mean()
    total = float(((b - mean) ** 2).sum())
    present = counts > 0
    if total == 0 or present.sum() < 2:
        return float("nan")
    between = float((counts[present] * (sums[present] / counts[present] - mean) ** 2).sum())
    return float(np.sqrt(np.clip(between / total, 0., 1.)))


def serial_test(layout: dict[str, Any], a: pd.Series, b: pd.Series, kind: str, cfg: StoneConfig) -> dict[str, Any]:
    """Effect and test use the same central window on original levels, never an adaptively differenced series."""
    result = {"test": "symmetric truncated time-shift", "status": "unavailable", "value": np.nan, "p_value": np.nan,
              "n": 0, "n_time": layout["n_time"], "n_entities": layout["n_entities"], "radius": 0,
              "B_forward": np.nan, "B_reverse": np.nan, "minimum_p": np.nan,
              "scale": "original levels; fixed central window", "detail": layout["detail"]}
    if result["detail"]:
        return result
    order = layout["order"]
    x, y = a.to_numpy()[order], b.to_numpy()[order]
    if pd.isna(x).any() or pd.isna(y).any():
        result["detail"] = "missing observations: do not compress the temporal grid; no IID fallback"
        return result
    if kind in ("kendall", "target_numeric"):
        x, y = x.astype(float), y.astype(float)
    elif kind == "cramers_v":
        x = pd.factorize(hashable(pd.Series(x.ravel())).astype("string"), sort=True)[0].reshape(x.shape)
        y = pd.factorize(hashable(pd.Series(y.ravel())).astype("string"), sort=True)[0].reshape(y.shape)
    elif kind == "eta":
        x = pd.factorize(hashable(pd.Series(x.ravel())).astype("string"), sort=True)[0].reshape(x.shape)
        y = y.astype(float)
    elif kind == "auc":
        if not np.isin(x, [0, 1]).all():
            raise ValueError("serial AUC requires a fixed binary 0/1 indicator")
        x, y = x.astype(np.int64), y.astype(float)
    else:
        raise ValueError(f"unknown serial statistic {kind!r}")
    if (kind in ("kendall", "target_numeric") and not np.isfinite(x).all()) or (kind != "cramers_v" and not np.isfinite(y).all()):
        result["detail"] = "non-finite observations; no IID fallback"
        return result
    n = len(x)
    radius = min(cfg.serial_shift_radius, n // 4)
    center = slice(radius, n - radius)
    n_b = int(y.max()) + 1 if kind == "cramers_v" else 0

    def measure(u: np.ndarray, v: np.ndarray) -> float:
        return _measure(u.ravel(), v.ravel(), kind, n_b)

    value = measure(x[center], y[center])
    # A constant-window Kendall statistic is undefined; define its test strength as zero.
    # This yields no evidence, instead of discarding non-finite surrogates from the denominator.
    strength = abs(value) if np.isfinite(value) else 0.
    counts = [0, 0]
    tolerance = 1e-12 * max(1., strength)
    for shift in range(-radius, radius + 1):
        window = slice(radius + shift, n - radius + shift)
        for j, v in enumerate((measure(x[center], y[window]), measure(x[window], y[center]))):
            surrogate = abs(v) if np.isfinite(v) else 0.
            counts[j] += int(surrogate >= strength - tolerance)
    descriptive = value
    if kind == "auc":
        descriptive = (value + 1) / 2
    elif kind == "target_numeric":
        descriptive = float(stats.spearmanr(x[center].ravel(), y[center].ravel()).statistic)
    result.update(status="conditional", value=descriptive, statistic=value,
                  statistic_name=("max(abs(Spearman), eta across feature rank deciles)" if kind == "target_numeric" else
                                  "2*AUC(missing indicator)-1" if kind == "auc" else kind),
                  p_value=min(1., max(counts) / (radius + 1)),
                  n=(n - 2 * radius) * layout["n_entities"], radius=radius,
                  B_forward=counts[0], B_reverse=counts[1], minimum_p=1 / (radius + 1),
                  detail="process-independence null; at least one process must be strictly stationary; "
                         "this assumption is unvalidated by automatic stationarity tests; statistic/type policy must be fixed")
    if not np.isfinite(value):
        result["detail"] += "; central effect undefined, p=1 provides no evidence"
    return result
