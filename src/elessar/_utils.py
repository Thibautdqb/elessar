"""Small shared helpers (private)."""

from __future__ import annotations

import math
import re
import warnings
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import pandas as pd

UNHASHABLE = (list, dict, set, np.ndarray)
REPR_TYPES = (*UNHASHABLE, bytes, bytearray)


@contextmanager
def quiet() -> Iterator[None]:
    """Silence numpy/scipy/pandas warnings inside the block."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def is_text_dtype(s: pd.Series) -> bool:
    """True for object columns and pandas string dtypes."""
    return pd.api.types.is_object_dtype(s.dtype) or isinstance(s.dtype, pd.StringDtype)


def hashable(s: pd.Series) -> pd.Series:
    """Return ``s`` with unhashable cells (lists, dicts, ...) and raw bytes replaced by their repr.

    Bytes are included because pandas decodes them as UTF-8 when casting to str, which can fail.
    """
    if not pd.api.types.is_object_dtype(s.dtype):
        return s
    if not s.map(lambda v: isinstance(v, REPR_TYPES)).any():
        return s
    return s.map(lambda v: repr(v) if isinstance(v, REPR_TYPES) else v)


def sample(obj: pd.Series | pd.DataFrame, n: int, random_state: int) -> pd.Series | pd.DataFrame:
    """Deterministic row sample that preserves the original row order."""
    if len(obj) <= n:
        return obj
    rng = np.random.default_rng(random_state)
    pos = np.sort(rng.choice(len(obj), size=n, replace=False))
    return obj.iloc[pos]


def sample_array(x: np.ndarray, n: int, random_state: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(random_state)
    return x[np.sort(rng.choice(len(x), size=n, replace=False))]


def finite(x: Any) -> float | None:
    """Convert to a plain float, mapping NaN/inf/errors to None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / pandas objects to JSON-serialisable Python."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [jsonable(r) for r in obj.reset_index().to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return finite(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def dedupe_names(names: list[Any]) -> list[str]:
    """Stringify column names and make them unique (``a``, ``a.1``, ...)."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for name in names:
        base = "|".join(map(str, name)) if isinstance(name, tuple) else str(name)
        key = base
        while key in seen:
            seen[base] += 1
            key = f"{base}.{seen[base]}"
        seen.setdefault(key, 0)
        out.append(key)
    return out


def _as_frame(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        return df.to_frame(name="value" if df.name is None else df.name)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a pandas DataFrame or Series, got {type(df).__name__}")
    return df


def prepare(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Accept a Series or DataFrame; return a DataFrame safe to analyse.

    Column names become unique strings (``a``, ``a.1``, tuples joined with ``|``), sparse columns
    are densified, and a non-unique row index is replaced by row positions.
    """
    df = _as_frame(df)
    names = dedupe_names(list(df.columns))
    if names != list(df.columns):
        df = df.set_axis(names, axis=1)
    sparse = [c for c in df.columns if isinstance(df[c].dtype, pd.SparseDtype)]
    if sparse:
        df = df.assign(**{c: df[c].sparse.to_dense() for c in sparse})
    if not df.index.is_unique:
        df = df.reset_index(drop=True)
    return df


def _same_label(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        if bool(a == b):
            return True
    except (TypeError, ValueError):
        pass
    try:
        return bool(pd.isna(a) and pd.isna(b))
    except (TypeError, ValueError):
        return False


def resolve_target(df: pd.DataFrame | pd.Series, target: Any) -> str | None:
    """Map a target given as an original column label (any hashable) to its name after :func:`prepare`."""
    if target is None:
        return None
    cols = list(_as_frame(df).columns)
    names = dedupe_names(cols)
    for label, name in zip(cols, names):
        if _same_label(label, target):
            return name
    if str(target) in names:
        return str(target)
    raise KeyError(f"target column {target!r} not found")


def fmt_num(x: Any, digits: int = 4) -> str:
    """Compact human formatting for numbers (thousands separators, sig. digits)."""
    if x is None:
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    f = finite(x)
    if f is None:
        return str(x)
    if f == 0:
        return "0"
    if abs(f) >= 1e15 or abs(f) < 1e-3:
        return f"{f:.{digits - 1}e}"
    if f.is_integer() and abs(f) < 1e15:
        return f"{int(f):,}"
    return f"{f:,.{digits}g}" if abs(f) < 1000 else f"{f:,.1f}"


def fmt_p(p: Any) -> str:
    """Compact p-value: 0.032, 4e-07, < 1e-300 (exact zero after underflow), n/a."""
    f = finite(p)
    if f is None:
        return "n/a"
    if f == 0:
        return "< 1e-300"
    return f"{f:.2g}" if f >= 1e-3 else f"{f:.0e}"


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def safe_name(name: Any) -> str:
    """Filesystem-safe version of a column name (used in plot file names)."""
    return re.sub(r"[^\w.-]+", "_", str(name)).strip("_")[:40] or "col"


def truncate(text: Any, width: int) -> str:
    s = str(text).replace("\n", " ")
    if width <= 1:
        return s[:width]
    return s if len(s) <= width else s[: width - 1] + "…"
