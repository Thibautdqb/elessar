"""Normality tests and best-fitting distribution (MLE, ranked by AIC)."""

from __future__ import annotations

import math
import signal
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ._utils import prepare, quiet, sample_array
from .config import StoneConfig
from .types import NUMERIC_TYPES, as_numeric, infer_type

PRETTY_NAMES: dict[str, str] = {
    "norm": "normal",
    "lognorm": "lognormal",
    "expon": "exponential",
    "gamma": "gamma",
    "weibull_min": "Weibull",
    "beta": "beta",
    "uniform": "uniform",
    "logistic": "logistic",
    "t": "Student t",
    "laplace": "Laplace",
    "pareto": "Pareto",
    "chi2": "chi-squared",
    "cauchy": "Cauchy",
    "poisson": "Poisson",
    "nbinom": "negative binomial",
    "geom": "geometric",
    "binom": "binomial",
}
KS_NOTE = "KS p-values are optimistic: the parameters were estimated from the same data."
_POSITIVE_SUPPORT = frozenset({"lognorm", "gamma", "weibull_min", "chi2", "pareto"})


class FitTimeout(Exception):
    """Raised when a single candidate fit exceeds ``StoneConfig.fit_timeout``."""


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Abort the block after ``seconds`` (POSIX main thread only; no-op elsewhere)."""
    if not seconds or not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def _raise(signum: int, frame: Any) -> None:
        raise FitTimeout(f"fit exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _anderson(x: np.ndarray, alpha: float) -> tuple[float, float | None, bool]:
    """Anderson-Darling against the normal: (statistic, p-value or None, rejected)."""
    try:
        res = stats.anderson(x, dist="norm", method="interpolate")
        return float(res.statistic), float(res.pvalue), bool(res.pvalue < alpha)
    except TypeError:  # SciPy < 1.17: critical values only
        res = stats.anderson(x, dist="norm")
        levels = np.asarray(res.significance_level) / 100
        crit = float(res.critical_values[int(np.argmin(np.abs(levels - alpha)))])
        return float(res.statistic), None, bool(res.statistic > crit)


def normality_tests(x: np.ndarray | pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Shapiro-Wilk, D'Agostino K², Jarque-Bera and Anderson-Darling, plus a majority verdict.

    The verdict is ``"not normal"`` when at least half of the tests reject normality at ``alpha``.
    """
    cfg = config or StoneConfig()
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    out: dict[str, Any] = {"n": n, "alpha": cfg.alpha}
    if n < 8 or np.ptp(x) == 0:
        out["verdict"] = "insufficient data" if n < 8 else "constant"
        return out
    x = sample_array(x, cfg.fit_sample_size, cfg.random_state)
    rejects: list[bool] = []
    with quiet():
        sw = stats.shapiro(sample_array(x, 5_000, cfg.random_state))
        out.update(shapiro_stat=float(sw.statistic), shapiro_p=float(sw.pvalue))
        rejects.append(sw.pvalue < cfg.alpha)
        k2 = stats.normaltest(x)
        out.update(dagostino_stat=float(k2.statistic), dagostino_p=float(k2.pvalue))
        rejects.append(k2.pvalue < cfg.alpha)
        jb = stats.jarque_bera(x)
        out.update(jarque_bera_stat=float(jb.statistic), jarque_bera_p=float(jb.pvalue))
        rejects.append(jb.pvalue < cfg.alpha)
        ad_stat, ad_p, ad_reject = _anderson(x, cfg.alpha)
        out.update(anderson_stat=ad_stat, anderson_p=ad_p)
        rejects.append(ad_reject)
    out["n_tests"], out["n_reject"] = len(rejects), int(sum(rejects))
    out["verdict"] = "not normal" if 2 * sum(rejects) >= len(rejects) else "normal"
    return out


def _fit_continuous(name: str, x: np.ndarray) -> tuple[tuple[float, ...], int, str]:
    """MLE fit of one continuous candidate: (params, n_params, note)."""
    dist = getattr(stats, name)
    lo, hi = float(x.min()), float(x.max())
    eps = (hi - lo) * 1e-3 or 1e-9
    if name in _POSITIVE_SUPPORT:
        if lo > 0:
            params = dist.fit(x, floc=0.0)
            return params, len(params) - 1, ""
        loc = lo - eps
        params = dist.fit(x, floc=loc)
        return params, len(params), f"shifted (loc={loc:.4g})"
    if name == "beta":
        if lo > 0 and hi < 1:
            return dist.fit(x, floc=0.0, fscale=1.0), 2, ""
        loc, scale = lo - eps, (hi - lo) + 2 * eps
        return dist.fit(x, floc=loc, fscale=scale), 4, f"rescaled to [{loc:.4g}, {loc + scale:.4g}]"
    params = dist.fit(x)
    return params, len(params), ""


def _fit_discrete(name: str, x: np.ndarray) -> tuple[tuple[float, ...], int, str]:
    """Closed-form / 1-D MLE fits for discrete candidates: (params, n_params, note)."""
    mean, var = float(x.mean()), float(x.var())
    if name == "poisson":
        return (mean,), 1, ""
    if name == "geom":
        loc = -1.0 if x.min() == 0 else 0.0
        return (1.0 / (mean - loc), loc), 1, "support starts at 0" if loc else ""
    if name == "nbinom":
        if var <= mean:
            raise ValueError("no overdispersion (variance <= mean)")

        def nll(log_r: float) -> float:
            r = math.exp(log_r)
            return -float(stats.nbinom.logpmf(x, r, r / (r + mean)).sum())

        r = math.exp(optimize.minimize_scalar(nll, bounds=(-6, 12), method="bounded").x)
        return (r, r / (r + mean)), 2, ""
    if name == "binom":
        n = int(x.max())
        if n < 1:
            raise ValueError("needs a positive upper bound")
        return (float(n), mean / n), 2, f"n fixed to max = {n}"
    raise ValueError(f"unknown discrete candidate {name!r}")


def _param_names(name: str, n_params: int) -> list[str]:
    dist = getattr(stats, name)
    shapes = [p.strip() for p in dist.shapes.split(",")] if dist.shapes else []
    extra = ["loc"] if isinstance(dist, stats.rv_discrete) else ["loc", "scale"]
    return (shapes + extra)[:n_params]


def _fmt_params(params: dict[str, float]) -> str:
    shown = {k: v for k, v in params.items() if not (k == "loc" and v == 0)}
    return ", ".join(f"{k}={v:.3g}" for k, v in shown.items())


def fit_distributions(
    x: np.ndarray | pd.Series, discrete: bool | None = None, config: StoneConfig | None = None
) -> dict[str, Any]:
    """Fit every candidate by MLE and rank by AIC.

    ``discrete=None`` auto-detects: non-negative integers with few distinct values use the
    discrete candidates. Returns ``fits`` (top ``n_best_fits``), ``best``, ``description``,
    ``failed`` candidates and a note on KS p-values.
    """
    cfg = config or StoneConfig()
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    out: dict[str, Any] = {"n_used": 0, "fits": [], "best": None, "failed": {}, "ks_note": KS_NOTE}
    if len(x) < 8 or np.ptp(x) == 0:
        out["description"] = "not enough distinct values to fit a distribution"
        return out
    x = sample_array(x, cfg.fit_sample_size, cfg.random_state)
    if discrete is None:
        discrete = bool(
            np.all(np.mod(x, 1) == 0) and x.min() >= 0 and np.unique(x).size <= cfg.discrete_max_levels
        )
    out["n_used"], out["kind"] = len(x), "discrete" if discrete else "continuous"
    candidates = cfg.candidates_discrete if discrete else cfg.candidates_continuous
    n = len(x)
    fits = []
    for name in candidates:
        try:
            with quiet(), _time_limit(cfg.fit_timeout):
                params, k, note = (_fit_discrete if discrete else _fit_continuous)(name, x)
                dist = getattr(stats, name)
                logp = dist.logpmf(x, *params) if discrete else dist.logpdf(x, *params)
                loglik = float(np.sum(logp))
                if not math.isfinite(loglik):
                    raise ValueError("data outside the fitted support")
                ks = stats.kstest(x, dist.cdf, args=params)
        except Exception as exc:  # noqa: BLE001 - a failing candidate is skipped, never fatal
            out["failed"][name] = str(exc) or type(exc).__name__
            continue
        named = dict(zip(_param_names(name, len(params)), map(float, params)))
        fits.append({
            "distribution": name,
            "label": PRETTY_NAMES.get(name, name),
            "params": named,
            "n_params": k,
            "loglik": loglik,
            "aic": 2 * k - 2 * loglik,
            "bic": k * math.log(n) - 2 * loglik,
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "note": note,
        })
    fits.sort(key=lambda f: f["aic"])
    out["fits"] = fits[: cfg.n_best_fits]
    out["n_candidates_fitted"] = len(fits)
    if fits:
        best = fits[0]
        out["best"] = best["distribution"]
        out["description"] = f"{shape_description(x)}, best fit {best['label']}({_fmt_params(best['params'])})"
    else:
        out["description"] = f"{shape_description(x)}, no candidate could be fitted"
    return out


def shape_description(x: np.ndarray) -> str:
    """Words for skewness and tails: 'heavily right-skewed, heavy-tailed'."""
    with quiet():
        skew = float(stats.skew(x, bias=False)) if len(x) > 2 else 0.0
        kurt = float(stats.kurtosis(x, bias=False)) if len(x) > 3 else 0.0
    side = "right" if skew > 0 else "left"
    if not math.isfinite(skew) or abs(skew) < 0.5:
        words = ["roughly symmetric"]
    elif abs(skew) < 1:
        words = [f"moderately {side}-skewed"]
    else:
        words = [f"heavily {side}-skewed"]
    if math.isfinite(kurt) and kurt > 1:
        words.append("heavy-tailed")
    elif math.isfinite(kurt) and kurt < -1:
        words.append("light-tailed")
    return ", ".join(words)


def frozen_best(fit: dict[str, Any]) -> Any:
    """Return the frozen scipy distribution of the best fit (or None)."""
    if not fit.get("fits"):
        return None
    best = fit["fits"][0]
    return getattr(stats, best["distribution"])(*best["params"].values())


def is_fittable(semantic_type: str) -> bool:
    return semantic_type in NUMERIC_TYPES or semantic_type == "numeric_string"


def analyze_distribution(x: pd.Series, config: StoneConfig | None = None) -> dict[str, Any]:
    """Normality tests + distribution fit for one float Series."""
    values = x.dropna().to_numpy(dtype="float64")
    return {"normality": normality_tests(values, config), "fit": fit_distributions(values, None, config)}


def see_dist(df: pd.DataFrame | pd.Series, config: StoneConfig | None = None) -> pd.DataFrame:
    """Normality verdict and top-3 fitted distributions for every numeric column."""
    cfg = config or StoneConfig()
    df = prepare(df)
    rows = []
    for col in df.columns:
        try:
            stype = infer_type(df[col], cfg)["semantic_type"]
            if not is_fittable(stype):
                continue
            res = analyze_distribution(as_numeric(df[col]), cfg)
            norm, fit = res["normality"], res["fit"]
            fits = fit["fits"]
            row: dict[str, Any] = {
                "column": col,
                "n": norm["n"],
                "normality": norm["verdict"],
                "shapiro_p": norm.get("shapiro_p"),
                "dagostino_p": norm.get("dagostino_p"),
                "jarque_bera_p": norm.get("jarque_bera_p"),
                "anderson_p": norm.get("anderson_p"),
                "best_fit": fits[0]["distribution"] if fits else None,
                "best_params": _fmt_params(fits[0]["params"]) if fits else None,
                "aic": fits[0]["aic"] if fits else None,
                "bic": fits[0]["bic"] if fits else None,
                "ks_pvalue": fits[0]["ks_pvalue"] if fits else None,
                "second_fit": fits[1]["distribution"] if len(fits) > 1 else None,
                "third_fit": fits[2]["distribution"] if len(fits) > 2 else None,
                "description": fit["description"],
            }
        except Exception as exc:  # noqa: BLE001
            row = {"column": col, "error": repr(exc)}
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["normality", "best_fit", "description"]).rename_axis("column")
    return pd.DataFrame(rows).set_index("column")
