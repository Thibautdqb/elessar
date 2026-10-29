"""Exploratory factor analysis, step by step, the way it is taught and reported.

1. **Preparing the variables**: quantitative or Likert items only (ordered labels become their
   ranks; identifiers, booleans, dates and free labels are left out); declared missing codes
   (``-999``...) removed; same scale for every item (a Likert battery) or not (then items are
   standardised for the reliability analysis and the scores); listwise deletion; a sample of at
   least 100 complete rows and about 10 per variable.
2. **Extraction**: principal components of the correlation matrix, unrotated solution, scree
   plot; varimax rotation with Kaiser normalisation; loadings sorted by size.
3. **Is it worth it?** Kaiser-Meyer-Olkin above .50 and Bartlett's test of sphericity below .05.
4. **Communalities** (extraction): an item explained below .50 is examined, removed, and the
   analysis is run again - one item at a time, the weakest first.
5. **How many factors?** Eigenvalue above 1 (Kaiser), cumulative variance of at least 60%, and the
   elbow of the scree plot. When two criteria agree that number is kept; otherwise the scree plot
   decides, reduced if needed so that no factor rests on a single item (coherence).
6. **Rotated component matrix**: each item goes to the factor of its highest loading; an item whose
   highest loading is below .50, or that loads at least .40 on a second factor within .20 of the
   first (a cross-loading), is examined, removed, and the analysis is run again. Factors are named
   after their items.
7. **Cronbach's alpha, one per factor**, on that factor's items (negatively loading items reversed),
   with the alpha if each item is deleted. Excellent >= .90, very good >= .80, good >= .70,
   acceptable >= .60, poor >= .50, else unacceptable. Deleting an item that would raise alpha by .05
   or more is *proposed*, never done: it must also make sense for the construct.
8. **Creating the factors**, only for factors whose alpha is at least acceptable: the mean of their
   items (raw for a common scale, z-scores otherwise; reversed items flipped).
9. **Descriptives** of the new factors: mean, median, mode, standard deviation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._utils import fmt_num, fmt_p, quiet, sample
from .config import StoneConfig
from .multivariate import bartlett_sphericity, kmo, varimax
from .types import NUMERIC_TYPES, as_numeric, ordinal_order

KMO_MIN = 0.50
COMMUNALITY_MIN = 0.50
LOADING_MIN = 0.50
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

    cols, kinds, declared = {}, {}, {}
    for c in df.columns:
        stype = types.get(c)
        if c == target or stype in ("id", "boolean", "constant", "empty"):
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
    return frame, {"kinds": {c: kinds[c] for c in frame.columns}, "declared_missing": declared}


def _same_scale(items: pd.DataFrame) -> bool:
    """Every item integer-valued on the same range (a Likert battery)."""
    ranges = {(float(items[c].min()), float(items[c].max())) for c in items}
    integer = all(np.all(np.mod(items[c].dropna(), 1) == 0) for c in items)
    return integer and len(ranges) == 1


def _solution(x: pd.DataFrame, k: int | None) -> dict[str, Any]:
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
    votes = [k_kaiser, k_var, k_scree]
    agreed = next((v for v in votes if votes.count(v) >= 2), None)
    chosen = k if k is not None else (agreed if agreed is not None else k_scree)
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
        "k_kaiser": k_kaiser, "k_variance": k_var, "k_scree": k_scree, "agreed": agreed is not None, "k": chosen,
        "unrotated": pd.DataFrame(unrot, index=x.columns, columns=names),
        "rotated": pd.DataFrame(rot, index=x.columns, columns=names),
        "communality": pd.Series((unrot**2).sum(axis=1), index=x.columns),
    }


def _problem_item(sol: dict[str, Any]) -> tuple[str, str] | None:
    """The one item to examine and remove before re-running (steps 4 and 6), or None."""
    h = sol["communality"]
    weak = h[h < COMMUNALITY_MIN]
    if len(weak):
        c = weak.idxmin()
        return c, f"communality {h[c]:.2f} < {COMMUNALITY_MIN:.2f}: poorly explained by the factors"
    load = sol["rotated"].abs()
    worst, why, gap_min = None, "", np.inf
    for c, row in load.iterrows():
        top = row.sort_values(ascending=False).to_numpy()
        second = top[1] if len(top) > 1 else 0.0
        if top[0] < LOADING_MIN and top[0] - second < gap_min:
            worst, gap_min, why = c, top[0] - second, f"highest loading {top[0]:.2f} < {LOADING_MIN:.2f}"
        elif second >= CROSS_LOADING and top[0] - second < CROSS_GAP and top[0] - second < gap_min:
            worst, gap_min = c, top[0] - second
            why = f"cross-loading: {top[0]:.2f} and {second:.2f} on two factors (gap < {CROSS_GAP:.2f})"
    return (worst, why) if worst is not None else None


def _reliability(items: pd.DataFrame, loadings: pd.Series, same_scale: bool) -> dict[str, Any]:
    """Step 7 for one factor: alpha on its items (negative loadings reversed), alpha if item deleted."""
    x = items.copy()
    reversed_items = [c for c in x if loadings[c] < 0]
    lo, hi = float(items.min().min()), float(items.max().max())
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
    return {"items": list(x.columns), "alpha": alpha, "label": alpha_label(alpha) if np.isfinite(alpha) else "n/a",
            "if_deleted": deleted, "reversed": reversed_items, "proposals": proposals, "scored": x, "low": lo, "high": hi,
            "standardised": not same_scale}


def _mode(x: pd.Series) -> float:
    m = x.round(2).mode()
    return float(m.iloc[0]) if len(m) else float("nan")


def factor_analysis(df: pd.DataFrame, types: dict[str, str], cfg: StoneConfig, target: str | None = None) -> dict[str, Any]:
    """Run the nine steps (see the module docstring); every step records its question and answer."""
    out: dict[str, Any] = {"steps": [], "removed": []}
    steps = out["steps"]

    def step(n: int, question: str, answer: str, ok: bool | str | None = None) -> None:
        steps.append({"step": n, "question": question, "answer": answer, "ok": ok})

    data = sample(df, cfg.sample_size, cfg.random_state)
    items, prep = _items(data, types, target, cfg)
    out.update(prep)
    p = items.shape[1]
    step(1, "Quantitative or Likert variables?", f"{p} usable: " + ", ".join(f"{c} ({k})" for c, k in prep["kinds"].items())
         if p else "none", p >= 3)
    if p < 3:
        out.update(verdict="not applicable", reason="factor analysis needs at least 3 quantitative or Likert variables")
        return out
    if prep["declared_missing"]:
        step(1, "Missing values correctly declared?",
             "codes declared missing before the analysis: " + "; ".join(f"{c}: {v}" for c, v in prep["declared_missing"].items()),
             True)
    else:
        step(1, "Missing values correctly declared?", "yes: no missing-value code among the items", True)
    complete = items.dropna()
    n = len(complete)
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
    initial = _solution(complete, None)
    chi2, bart_p = bartlett_sphericity(initial["corr"], n)
    kmo_all, _ = kmo(initial["corr"])
    step(2, "Extraction", "principal components of the correlation matrix, eigenvalues > 1 shown with the scree plot; "
         "varimax rotation (Kaiser normalisation); loadings sorted by size", None)
    step(3, "KMO > .50?", f"KMO = {kmo_all:.2f} on the {p} variables", kmo_all > KMO_MIN)
    step(3, "Bartlett Sig. < .05?", f"chi2 = {chi2:,.1f}, p = {fmt_p(bart_p)}", bart_p < 0.05)
    out.update(kmo=kmo_all, bartlett_chi2=chi2, bartlett_p=bart_p)
    if not (kmo_all > KMO_MIN and bart_p < 0.05):  # step 3 gates everything after it
        out.update(verdict="not useful", reason=f"KMO {kmo_all:.2f} and Bartlett p {fmt_p(bart_p)}: the variables do "
                                                "not share enough variance for factors")
        return out
    sol = initial
    while True:
        problem = _problem_item(sol) if len(current) > 3 else None
        if problem is None:
            break
        out["removed"].append({"item": problem[0], "why": problem[1], "k": sol["k"]})
        current.remove(problem[0])
        sol = _solution(complete[current], None)
    x = complete[current]
    scree_k = sol["k"]
    while not sol["agreed"] and sol["k"] > 1 and sol["rotated"].abs().idxmax(axis=1).value_counts().reindex(
            sol["rotated"].columns, fill_value=0).min() < 2:
        sol = _solution(x, sol["k"] - 1)  # coherence: a factor carried by a single item is not a construct
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
    step(4, "Communalities (extraction) all >= .50?", f"yes: lowest {h.min():.2f} ({h.idxmin()})"
         if h.min() >= COMMUNALITY_MIN else f"no: lowest {h.min():.2f} ({h.idxmin()}), kept (3 items left)",
         bool(h.min() >= COMMUNALITY_MIN))
    rule = ("the three criteria" if len({sol["k_kaiser"], sol["k_variance"], sol["k_scree"]}) == 1
            else "two criteria agree" if sol["agreed"] else "no two criteria agree: the scree plot decides")
    step(5, "How many factors?", f"eigenvalue > 1: {sol['k_kaiser']}; cumulative variance >= 60%: {sol['k_variance']}; "
         f"scree elbow: {sol['k_scree']} -> {sol['k']} ({rule}; {100 * sol['cumulative'][sol['k'] - 1]:.0f}% of the "
         "variance)" + (f"; reduced from {scree_k} for coherence: no factor may rest on a single item"
                        if sol["k"] < scree_k else ""), sol["agreed"])
    rot = sol["rotated"]
    assign = rot.abs().idxmax(axis=1)
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
        rel = _reliability(x[members], rot.loc[members, f], same)
        reliability[f] = rel
        if len(members) < 2:
            step(7, f"Cronbach's alpha of {f}", f"a single item ({members[0]}): no alpha, not a scale", None)
            continue
        text = f"alpha = {rel['alpha']:.2f} ({rel['label']}) on {len(members)} items"
        if rel["reversed"]:
            text += f"; reversed: {', '.join(rel['reversed'])}"
        for c, a in rel["proposals"]:
            text += f"; deleting {c} would raise it to {a:.2f} (only if it makes sense for the construct)"
        step(7, f"Cronbach's alpha of {f}", text, rel["alpha"] >= 0.60)
    out["reliability"] = {f: {k: v for k, v in r.items() if k != "scored"} for f, r in reliability.items()}
    scores, descriptives, code = {}, {}, []
    for f, rel in reliability.items():
        if len(rel["items"]) < 2 or not np.isfinite(rel["alpha"]) or rel["alpha"] < 0.60:
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
