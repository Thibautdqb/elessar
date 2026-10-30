"""Exploratory factor analysis, step by step, the way it is taught and reported.

1. **Preparing the variables**: quantitative or Likert items only (ordered labels become their
   ranks; identifiers, booleans, dates and free labels are left out); declared missing codes
   (``-999``...) removed; same scale for every item (a Likert battery) or not (then items are
   standardised for the reliability analysis and the scores); listwise deletion; a sample of at
   least 100 complete rows and about 10 per variable.
2. **Extraction**: principal components of the correlation matrix, unrotated solution, scree
   plot; varimax rotation with Kaiser normalisation; loadings sorted by size.
3. **Is it worth it?** Kaiser-Meyer-Olkin above .50 and Bartlett's test of sphericity below .05.
4. **Communalities** (extraction): an item explained below .40 (Costello & Osborne, 2005) is examined, removed, and the
   analysis is run again - one item at a time, the weakest first.
5. **How many factors?** Horn's parallel analysis decides: the components whose eigenvalue exceeds the
   95th percentile of those of random data of the same size, reduced if needed so that no factor rests
   on a single item (coherence). Eigenvalue above 1 (Kaiser), cumulative variance of at least 60% and
   the elbow of the scree plot are shown beside it: on 300 simulated structures the parallel analysis
   found the right number in 98% of them, the former rule (two of these three agree, else the scree
   plot) in 94%, Kaiser and the 60% rule alone overestimating it in most cases.
6. **Rotated component matrix**: each item goes to the factor of its highest loading; an item whose
   highest loading is below .40, or that loads at least .40 on a second factor within .20 of the
   first (a cross-loading), is examined, removed, and the analysis is run again. Factors are named
   after their items.
7. **Cronbach's alpha, one per factor**, on that factor's items (negatively loading items reversed),
   with the alpha if each item is deleted; a two-item factor is judged on the Spearman-Brown coefficient,
   which alpha understates for two items. Excellent >= .90, very good >= .80, good >= .70,
   acceptable >= .60, poor >= .50, else unacceptable. Deleting an item that would raise alpha by .05
   or more is *proposed*, never done: it must also make sense for the construct.
8. **Creating the factors**, only for factors whose reliability is at least acceptable: the mean of their
   items (raw for a common scale, z-scores otherwise; reversed items flipped).
9. **Descriptives** of the new factors: mean, median, mode, standard deviation.
"""

from __future__ import annotations

from typing import Any

import re

import numpy as np
import pandas as pd

from ._utils import fmt_num, fmt_p, quiet, sample
from .config import StoneConfig
from .multivariate import bartlett_sphericity, kmo, parallel_analysis, varimax
from .types import NUMERIC_TYPES, as_numeric, ordinal_order

KMO_MIN = 0.50
COMMUNALITY_MIN = 0.40
LOADING_MIN = 0.40
CROSS_LOADING = 0.40
CROSS_GAP = 0.20
VARIANCE_TARGET = 0.60
ALPHA_GAIN = 0.05
ALPHA_LABELS = ((0.90, "excellent"), (0.80, "very good"), (0.70, "good"), (0.60, "acceptable"), (0.50, "poor"))


def alpha_label(alpha: float) -> str:
    return next((label for cut, label in ALPHA_LABELS if alpha >= cut), "unacceptable")


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha: k / (k - 1) * (1 - sum of item variances / variance of the total)."""
    k = items.shape[1]
    if k < 2:
        return float("nan")
    total = items.sum(axis=1).var(ddof=1)
    if total <= 0:
        return float("nan")
    return float(k / (k - 1) * (1 - items.var(ddof=1).sum() / total))


def scree_elbow(eigenvalues: np.ndarray) -> int:
    """Factors before the elbow of the scree plot: the eigenvalue farthest from the straight line
    joining the first and the last one marks where the curve flattens."""
    lam = np.asarray(eigenvalues, dtype="float64")
    p = len(lam)
    if p <= 2:
        return 1
    x = np.arange(p, dtype="float64")
    start, end = np.array([0.0, lam[0]]), np.array([p - 1.0, lam[-1]])
    direction = (end - start) / np.linalg.norm(end - start)
    points = np.column_stack([x, lam]) - start
    distance = np.abs(points[:, 0] * direction[1] - points[:, 1] * direction[0])
    return max(1, int(np.argmax(distance)))  # the elbow's index = how many components come before it


def _items(df: pd.DataFrame, types: dict[str, str], target: str | None, cfg: StoneConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Step 1: the analysable items as numbers, declared missing codes removed."""
    from .quality import placeholder_values

    from .roles import tokens

    cols, kinds, declared, demographic = {}, {}, {}, []
    for c in df.columns:
        stype = types.get(c)
        if c == target or stype in ("id", "boolean", "constant", "empty"):
            continue
        if DEMOGRAPHIC_NAME.fullmatch(tokens(c)):
            demographic.append(c)
            continue
        if stype in NUMERIC_TYPES or (stype == "ordinal-candidate" and pd.api.types.is_numeric_dtype(df[c].dtype)):
            x = as_numeric(df[c], stype)
        elif stype == "ordinal-candidate" and (order := ordinal_order(df[c].dropna().astype(str).str.strip().unique())):
            rank = {v: i + 1 for i, v in enumerate(order)}
            x = df[c].astype("string").str.strip().map(rank).astype("float64")
            kinds[c] = "Likert labels, ranked " + " < ".join(order[:5])
        else:
            continue
        with quiet():
            codes = placeholder_values(df[c], stype or "", cfg)
        numeric_codes = [p for p in cfg.placeholder_numbers if fmt_num(p) in codes]
        if numeric_codes:
            x = x.where(~x.isin(numeric_codes))
            declared[c] = numeric_codes
        x = x.replace([np.inf, -np.inf], np.nan)
        if c not in kinds:
            v = x.dropna()
            likert = stype == "ordinal-candidate" or (v.nunique() <= 11 and bool(np.all(np.mod(v, 1) == 0)))
            kinds[c] = f"Likert-like, {v.min():g} to {v.max():g}" if likert and len(v) else "quantitative"
        if x.notna().mean() >= 0.5 and x.nunique() >= 3 and x.std() > 0:
            cols[c] = x
    frame = pd.DataFrame(cols, index=df.index)
    return frame, {"kinds": {c: kinds[c] for c in frame.columns}, "declared_missing": declared,
                   "demographic": demographic}


DEMOGRAPHIC_NAME = re.compile(  # a whole column name: "age" is left out, "housing_median_age" is a measure
    r"(id|age|ages|age_?[ymd]|age_?years|sex|sexe|gender|genre|educ\w*|diploma\w*|income|revenu\w*|salary|salaire|wage|year|annee|"
    r"grade|school|class|married|marital|children|kids|enfants|family\w*|household\w*|race|ethnic\w*|religion|"
    r"region|country|pays|urban|wave|group|condition|cohort|engnat|voted|drug)")


def _likert(x: pd.Series) -> bool:
    v = x.dropna()
    return len(v) > 0 and v.nunique() <= 11 and bool(np.all(np.mod(v, 1) == 0))


def _battery(items: pd.DataFrame) -> tuple[list[str], tuple[float, float]] | None:
    """The questionnaire, when the table holds one: at least 3 integer items with at most 11 levels on the
    most common response scale (an item may leave one end unused)."""
    cand = [c for c in items if _likert(items[c])]
    if len(cand) < 3:
        return None
    ranges = pd.Series([(float(items[c].min()), float(items[c].max())) for c in cand], index=cand)
    lo, hi = ranges.value_counts().index[0]
    members = [c for c, (a, b) in ranges.items() if lo <= a and b <= hi and b - a >= hi - lo - 1]
    return (members, (lo, hi)) if len(members) >= 3 else None


def _same_scale(items: pd.DataFrame) -> bool:
    """Every item a Likert-type integer on one common response scale (an item may leave one end unused)."""
    if not all(_likert(items[c]) for c in items):
        return False
    lo, hi = float(items.min().min()), float(items.max().max())
    return all(float(items[c].max()) - float(items[c].min()) >= hi - lo - 1 for c in items)


def _solution(x: pd.DataFrame, k: int | None, seed: int = 0) -> dict[str, Any]:
    """Principal components of the correlation matrix, the number of factors, unrotated and varimax loadings."""
    corr = np.corrcoef(x.to_numpy(dtype="float64"), rowvar=False)
    eigval, eigvec = np.linalg.eigh(corr)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = np.clip(eigval[order], 0, None), eigvec[:, order]
    p = len(eigval)
    cum = np.cumsum(eigval) / p
    k_kaiser = max(1, int((eigval > 1).sum()))
    k_var = int(np.searchsorted(cum, VARIANCE_TARGET) + 1)
    k_scree = scree_elbow(eigval)
    threshold = parallel_analysis(len(x), p, seed)
    above = eigval > threshold
    k_parallel = max(1, int(np.argmin(above)) if not above.all() else p)
    chosen = k if k is not None else k_parallel
    chosen = int(min(max(chosen, 1), p - 1))
    unrot = eigvec[:, :chosen] * np.sqrt(eigval[:chosen])
    for j in range(chosen):  # deterministic signs: the largest loading positive
        if unrot[np.argmax(np.abs(unrot[:, j])), j] < 0:
            unrot[:, j] *= -1
    rot = varimax(unrot) if chosen > 1 else unrot.copy()
    for j in range(chosen):
        if rot[np.argmax(np.abs(rot[:, j])), j] < 0:
            rot[:, j] *= -1
    # order the rotated factors by the variance they explain, as SPSS does
    order = np.argsort(-(rot**2).sum(axis=0))
    rot = rot[:, order]
    names = [f"F{i + 1}" for i in range(chosen)]
    return {
        "corr": corr, "eigenvalues": eigval, "explained": eigval / p, "cumulative": cum,
        "k_kaiser": k_kaiser, "k_variance": k_var, "k_scree": k_scree, "k_parallel": k_parallel, "k": chosen,
        "parallel_threshold": threshold,
        "unrotated": pd.DataFrame(unrot, index=x.columns, columns=names),
        "rotated": pd.DataFrame(rot, index=x.columns, columns=names),
        "communality": pd.Series((unrot**2).sum(axis=1), index=x.columns),
    }


def _problems(sol: dict[str, Any]) -> list[tuple[str, str]]:
    """Items to examine (steps 4 and 6), the one to remove first at the head: the worst communality,
    else the item whose two highest loadings are closest."""
    h = sol["communality"].round(2)  # decide on the value the report shows
    out = [(c, f"communality {h[c]:.2f} < {COMMUNALITY_MIN:.2f}: poorly explained by the factors")
           for c in h[h < COMMUNALITY_MIN].sort_values().index]
    loading = []
    for c, row in sol["rotated"].abs().iterrows():
        if c in h[h < COMMUNALITY_MIN].index:
            continue
        top = row.sort_values(ascending=False).round(2).to_numpy()
        second = top[1] if len(top) > 1 else 0.0
        if top[0] < LOADING_MIN:
            loading.append((top[0] - second, c, f"highest loading {top[0]:.2f} < {LOADING_MIN:.2f}"))
        elif second >= CROSS_LOADING and top[0] - second < CROSS_GAP:
            loading.append((top[0] - second, c, f"cross-loading: {top[0]:.2f} and {second:.2f} on two factors "
                                                f"(gap < {CROSS_GAP:.2f})"))
    return out + [(c, why) for _, c, why in sorted(loading, key=lambda t: t[0])]


def _reliability(items: pd.DataFrame, loadings: pd.Series, same_scale: bool,
                 scale: tuple[float, float]) -> dict[str, Any]:
    """Step 7 for one factor: alpha on its items (negative loadings reversed), alpha if item deleted. A two-item
    factor is judged on the Spearman-Brown coefficient 2r / (1 + r) instead: alpha understates the reliability
    of a two-item scale whenever the two variances differ (Eisinga, te Grotenhuis & Pelzer, 2013)."""
    x = items.copy()
    reversed_items = [c for c in x if loadings[c] < 0]
    lo, hi = scale
    if same_scale:
        for c in reversed_items:
            x[c] = lo + hi - x[c]
    else:
        x = (x - x.mean()) / x.std(ddof=1)
        for c in reversed_items:
            x[c] = -x[c]
    alpha = cronbach_alpha(x)
    deleted = {c: cronbach_alpha(x.drop(columns=c)) for c in x} if x.shape[1] > 2 else {}
    proposals = [(c, a) for c, a in deleted.items() if np.isfinite(alpha) and a - alpha >= ALPHA_GAIN]
    r = sb = None
    if x.shape[1] == 2:
        r = float(x.iloc[:, 0].corr(x.iloc[:, 1]))
        sb = 2 * r / (1 + r) if np.isfinite(r) and r > -1 else float("nan")
    value = sb if sb is not None else alpha
    return {"items": list(x.columns), "alpha": alpha, "r": r, "spearman_brown": sb,
            "coefficient": "Spearman-Brown" if sb is not None else "Cronbach's alpha", "reliability": value,
            "label": alpha_label(value) if np.isfinite(value) else "n/a",
            "if_deleted": deleted, "reversed": reversed_items, "proposals": proposals, "scored": x, "low": lo, "high": hi,
            "standardised": not same_scale}


def _mode(x: pd.Series) -> float:
    m = x.round(2).mode()
    return float(m.iloc[0]) if len(m) else float("nan")


def factor_analysis(df: pd.DataFrame, types: dict[str, str], cfg: StoneConfig, target: str | None = None) -> dict[str, Any]:
    """Run the nine steps (see the module docstring); every step records its question and answer."""
    out: dict[str, Any] = {"steps": [], "removed": [], "coherence": []}
    steps = out["steps"]

    def step(n: int, question: str, answer: str, ok: bool | str | None = None) -> None:
        steps.append({"step": n, "question": question, "answer": answer, "ok": ok})

    data = sample(df, cfg.sample_size, cfg.random_state)
    items, prep = _items(data, types, target, cfg)
    out.update(prep)
    p = items.shape[1]
    step(1, "Quantitative or Likert variables?", f"{p} usable: " + ", ".join(f"{c} ({k})" for c, k in prep["kinds"].items())
         if p else "none", p >= 3)
    if prep["demographic"]:
        step(1, "Quantitative or Likert variables?", "left out as describing the respondent, not measuring a construct: "
             + ", ".join(map(str, prep["demographic"])), True)
    if p < 3:
        out.update(verdict="not applicable", reason="factor analysis needs at least 3 quantitative or Likert variables")
        return out
    battery = _battery(items)
    if battery is not None and p / 2 <= len(battery[0]) < p:  # a questionnaire is most of the table, not a few indices
        members, (lo, hi) = battery
        left = [c for c in items if c not in members]
        step(1, "Which variables form the scale?", f"a questionnaire of {len(members)} items answered on {lo:g} to {hi:g}: "
             f"analysed alone; left out (not items of that scale): {', '.join(map(str, left))}", True)
        items = items[members]
        out["kinds"] = {c: prep["kinds"][c] for c in members}
        out["left_out"] = left
        p = len(members)
    if prep["declared_missing"]:
        step(1, "Missing values correctly declared?",
             "codes declared missing before the analysis: " + "; ".join(f"{c}: {v}" for c, v in prep["declared_missing"].items()),
             True)
    else:
        step(1, "Missing values correctly declared?", "yes: no missing-value code among the items", True)
    complete = items.dropna()
    n = len(complete)
    redundant = []
    while complete.shape[1] > 3 and n > complete.shape[1]:
        # an item that is an exact combination of others (a total, a mean of sub-items) makes the correlation
        # matrix singular: determinant 0, no KMO, no Bartlett; such items carry nothing of their own
        corr = np.corrcoef(complete.to_numpy(dtype="float64"), rowvar=False)
        vif = np.diag(np.linalg.inv(corr + 1e-9 * np.eye(len(corr))))  # pinv would hide exact dependencies
        if not np.isfinite(vif).all() or vif.max() <= 100:
            break
        worst = complete.columns[int(np.argmax(vif))]
        redundant.append(worst)
        complete = complete.drop(columns=worst)
    det = float(np.linalg.det(np.corrcoef(complete.to_numpy(dtype="float64"), rowvar=False))) if n > 2 else float("nan")
    if redundant:
        step(1, "Redundant variables?", "removed as (nearly) exact combinations of the others (VIF > 100): "
             + ", ".join(map(str, redundant)), "removed")
        p = complete.shape[1]
    step(1, "Determinant of the correlation matrix", f"{det:.2e}" + (" (> 0.00001: no singularity)" if det > 1e-5 else
         " (< 0.00001: strong multicollinearity, loadings less stable)"), bool(det > 1e-5))
    out.update(redundant=redundant, determinant=det)
    same = _same_scale(complete) if n else False
    step(1, "Variables on the same scale?", "yes: one common integer range, items kept as they are" if same else
         "no: items standardised (z-scores) for the reliability analysis and the scores; extraction uses the "
         "correlation matrix either way", True)
    enough = n >= 100 and n >= 10 * p
    step(1, "Sample size sufficient?", f"{n:,} complete rows (listwise) for {p} variables: {n / p:.0f} per variable "
         f"(rule: at least 100 and about 10 per variable)", enough)
    out.update(n=n, n_variables=p, same_scale=same, sample_ok=enough)
    if n < max(30, 3 * p):
        out.update(verdict="not applicable", reason=f"only {n} complete rows for {p} variables")
        return out

    current = list(complete.columns)
    initial = _solution(complete, None, cfg.random_state)
    chi2, bart_p = bartlett_sphericity(initial["corr"], n)
    kmo_all, _ = kmo(initial["corr"])
    step(2, "Extraction", "principal components of the correlation matrix, eigenvalues shown with the scree plot "
         "against those of random data (parallel analysis); "
         "varimax rotation (Kaiser normalisation); loadings sorted by size", None)
    step(3, "KMO > .50?", f"KMO = {kmo_all:.2f} on the {p} variables", kmo_all > KMO_MIN)
    step(3, "Bartlett Sig. < .05?", f"chi2 = {chi2:,.1f}, p = {fmt_p(bart_p)}", bart_p < 0.05)
    out.update(kmo=kmo_all, bartlett_chi2=chi2, bartlett_p=bart_p)
    if not (kmo_all > KMO_MIN and bart_p < 0.05):  # step 3 gates everything after it
        out.update(verdict="not useful", reason=f"KMO {kmo_all:.2f} and Bartlett p {fmt_p(bart_p)}: the variables do "
                                                "not share enough variance for factors")
        return out
    sol, forced = initial, None
    while len(current) > 3:
        sizes = sol["rotated"].abs().idxmax(axis=1).value_counts().reindex(sol["rotated"].columns, fill_value=0)
        if sol["k"] > 1 and sizes.min() < 2:  # a factor carried by one item is not a construct
            fewer = _solution(complete[current], sol["k"] - 1, cfg.random_state)
            out["coherence"].append(f"{sol['k']} factors leave one resting on a single item "
                                f"({sol['rotated'].abs().idxmax(axis=1).eq(sizes.idxmin()).idxmax()}): "
                                    f"{fewer['k']} kept for coherence")
            sol, forced = fewer, fewer["k"]
            continue
        problems = _problems(sol)
        if not problems:
            break
        if len(problems) >= 2 and sol["k"] > 1:
            # several items at fault at once usually means too many factors, not bad items
            fewer = _solution(complete[current], sol["k"] - 1, cfg.random_state)
            if len(_problems(fewer)) < len(problems):
                out["coherence"].append(f"{sol['k']} factors leave {len(problems)} items to examine, {fewer['k']} leave "
                                    f"{len(_problems(fewer))}: {fewer['k']} kept for coherence")
                sol, forced = fewer, fewer["k"]
                continue
        item, why = problems[0]
        out["removed"].append({"item": item, "why": why, "k": sol["k"]})
        current.remove(item)
        sol = _solution(complete[current], forced, cfg.random_state)
    x = complete[current]
    pertinent = True
    if out["removed"]:  # the retained items must still pass step 3
        chi2, bart_p = bartlett_sphericity(sol["corr"], n)
        kmo_all, _ = kmo(sol["corr"])
        pertinent = kmo_all > KMO_MIN and bart_p < 0.05
        step(3, "KMO and Bartlett on the retained items", f"KMO = {kmo_all:.2f}, Bartlett p = {fmt_p(bart_p)} on "
             f"{len(current)} items", pertinent)
        out.update(kmo=kmo_all, bartlett_chi2=chi2, bartlett_p=bart_p)
    for r in out["removed"]:
        step(4 if r["why"].startswith("communality") else 6, f"Examine {r['item']}",
             f"{r['why']} -> removed, analysis run again", "removed")
    h = sol["communality"]
    step(4, "Communalities (extraction) all >= .40?", f"yes: lowest {h.min():.2f} ({h.idxmin()})"
         if h.min() >= COMMUNALITY_MIN else f"no: lowest {h.min():.2f} ({h.idxmin()}), kept (3 items left)",
         bool(h.min() >= COMMUNALITY_MIN))
    others = {"eigenvalue > 1": sol["k_kaiser"], "cumulative variance >= 60%": sol["k_variance"],
              "scree elbow": sol["k_scree"]}
    differ = [name for name, v in others.items() if v != sol["k_parallel"]]
    step(5, "How many factors?", f"parallel analysis (Horn): {sol['k_parallel']} -> {sol['k']} "
         f"({100 * sol['cumulative'][sol['k'] - 1]:.0f}% of the variance); "
         + "; ".join(f"{name}: {v}" for name, v in others.items())
         + (f" ({', '.join(differ)} {'disagree' if len(differ) > 1 else 'disagrees'}: these criteria tend to "
            "overestimate)" if differ else " (all agree)")
         + "".join(f"; {c}" for c in out["coherence"]), not out["coherence"])
    rot = sol["rotated"]
    assign = rot.abs().idxmax(axis=1)
    for f in rot.columns:  # orient each factor like most of its items, so that the minority is reversed
        if rot.loc[assign == f, f].sum() < 0:
            rot[f] *= -1
    order = sorted(rot.index, key=lambda c: (list(rot.columns).index(assign[c]), -abs(rot.loc[c, assign[c]])))
    rot = rot.loc[order]
    factors = {}
    for f in rot.columns:
        members = [c for c in order if assign[c] == f]
        if members:
            factors[f] = members
    named = {f: " / ".join(m[:3]) + (" ..." if len(m) > 3 else "") for f, m in factors.items()}
    step(6, "Rotated component matrix", "; ".join(f"{f} = {', '.join(m)}" for f, m in factors.items()), True)
    out.update(initial=initial, solution=sol, loadings=rot, unrotated=sol["unrotated"].loc[order],
               communality=sol["communality"].loc[order].to_dict(), factors=factors, names=named,
               n_factors=sol["k"], items=current)
    reliability = {}
    for f, members in factors.items():
        rel = _reliability(x[members], rot.loc[members, f], same, (float(x.min().min()), float(x.max().max())))
        reliability[f] = rel
        if len(members) < 2:
            step(7, f"Reliability of {f}", f"a single item ({members[0]}): no alpha, not a scale", None)
            continue
        text = (f"two items: Spearman-Brown = {rel['spearman_brown']:.2f} ({rel['label']}; r = {rel['r']:.2f}"
                + (f"; alpha {rel['alpha']:.2f} understates it: the two items vary unequally)"
                   if rel["alpha"] < rel["spearman_brown"] - 0.005 else ")") if rel["spearman_brown"] is not None
                else f"alpha = {rel['alpha']:.2f} ({rel['label']}) on {len(members)} items")
        if rel["reversed"]:
            text += f"; reversed: {', '.join(rel['reversed'])}"
        for c, a in rel["proposals"]:
            text += f"; deleting {c} would raise it to {a:.2f} (only if it makes sense for the construct)"
        step(7, f"Reliability of {f}", text, rel["reliability"] >= 0.60)
    out["reliability"] = {f: {k: v for k, v in r.items() if k != "scored"} for f, r in reliability.items()}
    scores, descriptives, code = {}, {}, []
    for f, rel in reliability.items():
        if len(rel["items"]) < 2 or not np.isfinite(rel["reliability"]) or rel["reliability"] < 0.60:
            continue
        name = f"{f}_" + "_".join(str(i) for i in rel["items"][:2])
        s = rel["scored"].mean(axis=1)
        scores[name] = s
        descriptives[name] = {"items": rel["items"], "n": int(s.notna().sum()), "mean": float(s.mean()),
                              "median": float(s.median()), "mode": _mode(s), "std": float(s.std())}
        cols = rel["items"]
        lines = [f"items = df[{cols!r}].copy()"]
        codes = sorted({v for c in cols for v in prep["declared_missing"].get(c, [])})
        if codes:
            lines.append(f"items = items.replace({codes!r}, np.nan)  # declared missing codes")
        if any(prep["kinds"][c].startswith("Likert labels") for c in cols):
            lines.insert(0, "# ordered labels first mapped to their ranks (1 = lowest), as in the report")
        if not same:
            lines.append("items = (items - items.mean()) / items.std()  # different scales: z-scores")
        for c in rel["reversed"]:
            flip = f"{rel['low'] + rel['high']:g} - items[{c!r}]" if same else f"-items[{c!r}]"
            lines.append(f"items[{c!r}] = {flip}  # reversed item (negative loading)")
        lines.append(f"df[{name!r}] = items.mean(axis=1)  # MEAN: the items answered")
        lines.append(f"# MEAN.{len(cols)} (every item required): items.mean(axis=1).where(items.notna().all(axis=1))")
        code.append("\n".join(lines))
    step(8, "Create the factors", ", ".join(scores) + " (mean of the items)" if scores
         else "none: no factor reached an acceptable alpha (>= .60)", bool(scores))
    if descriptives:
        step(9, "Descriptives of the new factors", "; ".join(
            f"{k}: mean {d['mean']:.2f}, median {d['median']:.2f}, mode {d['mode']:.2f}, sd {d['std']:.2f}"
            for k, d in descriptives.items()), None)
    out.update(scores=pd.DataFrame(scores) if scores else None, descriptives=descriptives, code="\n\n".join(code))
    k = sol["k"]
    explained = 100 * sol["cumulative"][k - 1]
    reliable = len(scores)
    if not pertinent:
        out.update(verdict="not useful", reason=f"after removing {len(out['removed'])} item{'s' if len(out['removed']) > 1 else ''}, KMO {kmo_all:.2f} and "
                                                f"Bartlett p {fmt_p(bart_p)}: the retained items no longer share "
                                                "enough variance")
    elif not reliable:
        best = max((r["alpha"] for r in reliability.values() if np.isfinite(r["alpha"])), default=float("nan"))
        out.update(verdict="not useful", reason=f"{k} factor{'s' if k > 1 else ''} ({explained:.0f}% of the variance), "
                                                f"but no reliable scale (best alpha {best:.2f} < .60)")
    elif not enough:
        out.update(verdict="indicative", reason=f"{k} factor{'s' if k > 1 else ''} ({explained:.0f}% of the variance), "
                                                f"but the sample is small ({n} rows, {n / p:.0f} per variable)")
    else:
        out.update(verdict="recommended", reason=f"{k} factor{'s' if k > 1 else ''} explain {explained:.0f}% of the "
                                                 f"variance of {len(current)} variables (KMO {kmo_all:.2f}, Bartlett p "
                                                 f"{fmt_p(bart_p)}); {reliable} reliable scale"
                                                 f"{'s' if reliable != 1 else ''} created")
    steps.sort(key=lambda st: st["step"])  # stable: removals sit with their step, in the order they happened
    out["defined_by"] = factors
    out["weak_columns"] = [r["item"] for r in out["removed"]]
    return out


__all__ = ["alpha_label", "cronbach_alpha", "factor_analysis", "scree_elbow"]
