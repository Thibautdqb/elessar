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
KS_NOTE = ("KS p-values are optimistic: the parameters were estimated from the same data. For integer data the "
           "gap is measured on the integer support and the p-value is a chi-square goodness-of-fit test.")
POINT_MASS_SHARE = 0.3
"""A single value holding this share of a continuous column is reported apart and left out of the fit."""
_POSITIVE_SUPPORT = frozenset({"lognorm", "gamma", "weibull_min", "chi2", "pareto"})
BINOM_MAX_TRIALS = 100_000
"""Finite search domain for a binomial profile MLE; a maximum at the upper bound is not accepted."""


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
        lower = int(x.max())
        if lower < 1:
            raise ValueError("needs a positive upper bound")
        if x.min() < 0 or not np.all(x == np.floor(x)):
            raise ValueError("binomial data must be non-negative integers")
        if lower >= BINOM_MAX_TRIALS:
            raise ValueError(f"binomial support exceeds the search limit {BINOM_MAX_TRIALS}")
        # At every admissible integer n, p_hat = mean/n exactly. Compress repeated
        # observations, and examine the whole declared range rather than assuming
        # that the sample maximum is the distribution's upper support bound.
        values, counts = np.unique(x, return_counts=True)
        best_n, best_ll = lower, -np.inf
        for start in range(lower, BINOM_MAX_TRIALS + 1, 2_048):
            ns = np.arange(start, min(start + 2_048, BINOM_MAX_TRIALS + 1))
            likelihood = counts @ stats.binom.logpmf(values[:, None], ns, mean / ns)
            i = int(np.argmax(likelihood))
            if likelihood[i] > best_ll:
                best_n, best_ll = int(ns[i]), float(likelihood[i])
        if best_n == BINOM_MAX_TRIALS:
            raise ValueError("binomial profile maximum is at the search limit; use the Poisson candidate "
                             "or supply a scientifically justified trial count")
        return (float(best_n), mean / best_n), 2, f"profile MLE over n in [{lower}, {BINOM_MAX_TRIALS}]"
    raise ValueError(f"unknown discrete candidate {name!r}")


def _param_names(name: str, n_params: int) -> list[str]:
    dist = getattr(stats, name)
    shapes = [p.strip() for p in dist.shapes.split(",")] if dist.shapes else []
    extra = ["loc"] if isinstance(dist, stats.rv_discrete) else ["loc", "scale"]
    return (shapes + extra)[:n_params]


def _fmt_params(params: dict[str, float]) -> str:
    shown = {k: v for k, v in params.items() if not (k == "loc" and v == 0)}
    return ", ".join(f"{k}={v:.3g}" for k, v in shown.items())


def _discrete_gof(x: np.ndarray, dist: Any, params: tuple[float, ...], n_fitted: int) -> tuple[float, float]:
    """Largest gap between the observed and fitted CDF on the integer support, and a chi-square
    goodness-of-fit p-value (cells merged until each expects at least 5 values).

    The KS test assumes a continuous distribution: on integers its ties inflate the statistic.
    """
    values, counts = np.unique(x, return_counts=True)
    n = len(x)
    gap = float(np.max(np.abs(np.cumsum(counts) / n - dist.cdf(values, *params))))
    support = np.arange(values.min(), values.max() + 1)
    expected = dist.pmf(support, *params) * n
    expected[0] += dist.cdf(support[0] - 1, *params) * n  # tails folded into the end cells
    expected[-1] += dist.sf(support[-1], *params) * n
    observed = np.zeros(len(support))
    observed[(values - support[0]).astype(int)] = counts
    obs_cells, exp_cells, o_acc, e_acc = [], [], 0.0, 0.0
    for o, e in zip(observed, expected):
        o_acc, e_acc = o_acc + o, e_acc + e
        if e_acc >= 5:
            obs_cells.append(o_acc)
            exp_cells.append(e_acc)
            o_acc = e_acc = 0.0
    if exp_cells:
        obs_cells[-1] += o_acc
        exp_cells[-1] += e_acc
    dof = len(exp_cells) - 1 - n_fitted
    if dof < 1:
        return gap, float("nan")
    chi2 = float(np.sum((np.array(obs_cells) - np.array(exp_cells)) ** 2 / np.array(exp_cells)))
    return gap, float(stats.chi2.sf(chi2, dof))


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
    if not discrete:
        values, counts = np.unique(x, return_counts=True)
        top = int(np.argmax(counts))
        if counts[top] / len(x) >= POINT_MASS_SHARE and len(values) >= 10:
            # e.g. rain: 58% exact zeros. A continuous density cannot carry that mass, and MLE then
            # collapses onto the spike; describe the spike, fit the rest.
            out["point_mass"] = {"value": float(values[top]), "pct": 100.0 * counts[top] / len(x)}
            x = x[x != values[top]]
            if len(x) < 8 or np.ptp(x) == 0:
                out["description"] = f"point mass at {values[top]:g} ({out['point_mass']['pct']:.0f}% of values)"
                return out
            out["n_used"] = len(x)
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
                if not discrete and params[-1] < 1e-6 * (np.std(x) or 1.0):
                    raise ValueError("degenerate fit: the density collapsed onto repeated values")
                ks_stat, ks_p = (_discrete_gof(x, dist, params, k) if discrete
                                 else tuple(stats.kstest(x, dist.cdf, args=params))[:2])
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
            "ks_stat": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "note": note,
        })
    fits.sort(key=lambda f: f["aic"])
    if fits:  # Akaike weights: relative support of each candidate among those fitted
        rel = np.exp(-0.5 * np.array([f["aic"] - fits[0]["aic"] for f in fits]))
        for f, w in zip(fits, rel / rel.sum()):
            f["delta_aic"] = f["aic"] - fits[0]["aic"]
            f["akaike_weight"] = float(w)
    out["fits"] = fits[: cfg.n_best_fits]
    out["n_candidates_fitted"] = len(fits)
    pm = out.get("point_mass")
    prefix = f"point mass at {pm['value']:g} ({pm['pct']:.0f}% of values); the rest: " if pm else ""
    if fits:
        best = fits[0]
        out["best"] = best["distribution"]
        out["description"] = f"{prefix}{shape_description(x)}, best fit {best['label']}({_fmt_params(best['params'])})"
    else:
        out["description"] = f"{prefix}{shape_description(x)}, no candidate could be fitted"
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
