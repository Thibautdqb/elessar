"""Choose the target column: ranked suggestions, a small window to click one, or a terminal prompt.

elessar never guesses the target silently: :func:`suggest_targets` only ranks candidates (name
hints, cardinality, position) so the most likely ones come first when a person picks.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Callable

import pandas as pd

from ._utils import hashable, quiet, truncate

_STRONG = re.compile(r"^(target|label|labels|class|y|outcome|response|dependent|ground_?truth|gt)$|"
                     r"(^|[_\s.-])(target|label|outcome)($|[_\s.-])", re.I)
_DOMAIN = re.compile(r"churn|survived|surviv|default|fraud|spam|diagnos|disease|cancer|malignant|defect|failure|"
                     r"attrition|converted|conversion|clicked|click|purchased|approved|accepted|won|winner|"
                     r"price|salary|income|revenue|sales|charges|cost|quality|rating|grade|sentiment|"
                     r"severity|risk|readmit|status|result|species|category$|^is_|^has_|^will_", re.I)
_ID = re.compile(r"(^|[_\s.-])(id|uuid|guid|key|index|idx|pk)($|[_\s.-])|^#$|^unnamed", re.I)


def suggest_targets(df: pd.DataFrame, n_rows: int = 50_000) -> list[dict[str, Any]]:
    """Every column ranked by how plausible a target it is: ``{column, score, reason, ...}``.

    Higher is more plausible; the score only orders the list, it is never used to pick.
    """
    head = df.head(n_rows) if len(df) > n_rows else df
    out = []
    n_cols = head.shape[1]
    with quiet():
        for pos, col in enumerate(head.columns):
            s = head.iloc[:, pos]
            name = str(col)
            nonnull = hashable(s.dropna())
            n = len(nonnull)
            distinct = int(nonnull.nunique()) if n else 0
            missing = 100.0 * (1 - n / len(s)) if len(s) else 0.0
            numeric = pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)
            score, reasons = 0.0, []
            if _STRONG.search(name):
                score, reasons = score + 4, reasons + ["named like a target"]
            elif _DOMAIN.search(name):
                score, reasons = score + 2, reasons + ["usual target name"]
            if n == 0 or distinct <= 1:
                score, reasons = score - 6, reasons + ["empty or constant"]
            elif _ID.search(name) or (distinct / n > 0.95 and not pd.api.types.is_float_dtype(s.dtype)):
                score, reasons = score - 5, reasons + ["looks like an identifier"]
            elif not numeric and distinct > 50:
                score, reasons = score - 2, reasons + ["many distinct labels"]
            elif distinct == 2:
                score, reasons = score + 1.5, reasons + ["two classes"]
            elif distinct <= 10:
                score, reasons = score + 1, reasons + [f"{distinct} classes"]
            elif numeric:
                score += 0.3
            if missing > 50:
                score, reasons = score - 2, reasons + [f"{missing:.0f}% missing"]
            if pos == n_cols - 1 and n_cols > 1:
                score, reasons = score + 1, reasons + ["last column"]
            examples = ", ".join(truncate(v, 18) for v in nonnull.astype(str).drop_duplicates().head(3))
            out.append({"column": col, "score": round(score, 2), "reason": ", ".join(reasons), "dtype": str(s.dtype),
                        "distinct": distinct, "pct_missing": missing, "examples": examples, "position": pos})
    return sorted(out, key=lambda r: (-r["score"], r["position"]))


def _has_display() -> bool:
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def pick_target(df: pd.DataFrame, how: str = "auto") -> Any:
    """Ask a person which column is the target; returns its label, or None for no target.

    ``how``: ``"window"`` (a small Tk window), ``"terminal"`` (numbered prompt) or ``"auto"``
    (the window when a display and Tk are available, else the terminal).
    """
    ranked = suggest_targets(df)
    if not ranked:
        return None
    if how in ("auto", "window") and _has_display():
        try:
            return _window(ranked, title=f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as exc:  # noqa: BLE001 - no Tk, or the display refused the window
            if how == "window":
                raise RuntimeError(f"cannot open the target window: {exc}") from exc
    if not sys.stdin or not sys.stdin.isatty():
        raise RuntimeError("target='ask' needs a display or an interactive terminal; pass target='<column>'")
    return _terminal(ranked)


# ------------------------------------------------------------------------------------------- terminal
def _terminal(ranked: list[dict[str, Any]], read: Callable[[str], str] = input,
              write: Callable[[str], Any] = print) -> Any:
    by_position = sorted(ranked, key=lambda r: r["position"])
    best = ranked[0] if ranked[0]["score"] >= 2 else None
    write("\nWhich column is the target (the one the others should explain)?\n")
    width = max(len(str(r["column"])) for r in by_position)
    for i, r in enumerate(by_position, start=1):
        star = "★" if best is not None and r is best else " "
        write(f"  {i:>3} {star} {truncate(str(r['column']), 40):<{min(width, 40)}}  {r['dtype']:<10} "
              f"{r['distinct']:>7,} distinct  {r['pct_missing']:5.1f}% missing   {r['examples']}")
    default = f"Enter = {best['column']} (suggested)" if best else "Enter = no target"
    while True:
        answer = read(f"\nNumber or name ({default}, 0 = no target): ").strip()
        if not answer:
            return best["column"] if best else None
        if answer in ("0", "-", "none"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(by_position):
            return by_position[int(answer) - 1]["column"]
        named = [r["column"] for r in by_position if str(r["column"]) == answer]
        if named:
            return named[0]
        write(f"  '{answer}' is neither a number from 1 to {len(by_position)} nor a column name.")


# ------------------------------------------------------------------------------------------- window
_BG, _SURFACE, _INK, _INK2, _MUTED, _LINE, _BLUE, _PALE = (
    "#f4f3ef", "#fcfcfb", "#1b1b19", "#52514e", "#8f8e88", "#e6e4de", "#2a78d6", "#dbe9fb")


def _window(ranked: list[dict[str, Any]], title: str = "", on_ready: Callable[[Any], Any] | None = None) -> Any:
    """A Tk window listing the columns (suggestions first); returns the clicked column or None."""
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import ttk

    root = tk.Tk()
    root.title("elessar · choose the target column")
    root.configure(bg=_BG)
    base = tkfont.nametofont("TkDefaultFont")
    base.configure(size=11)
    bold = base.copy()
    bold.configure(weight="bold")
    big = base.copy()
    big.configure(size=15, weight="bold")

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=_BG, foreground=_INK, font=base)
    style.configure("Treeview", background=_SURFACE, fieldbackground=_SURFACE, foreground=_INK, rowheight=30,
                    borderwidth=0)
    style.configure("Treeview.Heading", background=_BG, foreground=_MUTED, relief="flat", font=bold, padding=(8, 6))
    style.map("Treeview", background=[("selected", _BLUE)], foreground=[("selected", "#ffffff")])
    style.configure("Accent.TButton", background=_BLUE, foreground="#ffffff", padding=(16, 8), borderwidth=0, font=bold)
    style.map("Accent.TButton", background=[("active", "#1f62b3")])
    style.configure("Plain.TButton", background=_SURFACE, foreground=_INK, padding=(16, 8), borderwidth=1)
    style.configure("TEntry", fieldbackground=_SURFACE, padding=6)

    chosen: dict[str, Any] = {"value": None}
    outer = ttk.Frame(root, padding=18)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="Which column is the target?", font=big).pack(anchor="w")
    ttk.Label(outer, text="The column the others should explain. Suggestions come first (★); "
                          "double-click a row or press Enter.", foreground=_INK2).pack(anchor="w", pady=(2, 10))
    if title:
        ttk.Label(outer, text=title, foreground=_MUTED).pack(anchor="w", pady=(0, 8))

    query = tk.StringVar()
    search = ttk.Entry(outer, textvariable=query)
    search.pack(fill="x", pady=(0, 8))

    frame = ttk.Frame(outer)
    frame.pack(fill="both", expand=True)
    cols = ("column", "type", "distinct", "missing", "examples")
    tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="browse")
    widths = {"column": 220, "type": 90, "distinct": 90, "missing": 80, "examples": 280}
    for c in cols:
        tree.heading(c, text=c.upper(), anchor="w")
        tree.column(c, width=widths[c], anchor="e" if c in ("distinct", "missing") else "w",
                    stretch=c in ("column", "examples"))
    tree.tag_configure("suggested", background=_PALE)
    tree.tag_configure("unlikely", foreground=_MUTED)
    bar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")

    rows = {}

    def fill(*_: Any) -> None:
        text = query.get().strip().lower()
        tree.delete(*tree.get_children())
        rows.clear()
        for r in ranked:
            if text and text not in str(r["column"]).lower():
                continue
            tag = "suggested" if r["score"] >= 2 else "unlikely" if r["score"] < 0 else ""
            label = ("★ " if tag == "suggested" else "   ") + str(r["column"])
            iid = tree.insert("", "end", values=(label, r["dtype"], f"{r['distinct']:,}", f"{r['pct_missing']:.1f}%",
                                                 r["examples"]), tags=(tag,) if tag else ())
            rows[iid] = r["column"]
        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            tree.focus(children[0])

    def choose(*_: Any) -> None:
        sel = tree.selection()
        if sel:
            chosen["value"] = rows[sel[0]]
            root.destroy()

    def no_target(*_: Any) -> None:
        chosen["value"] = None
        root.destroy()

    query.trace_add("write", fill)
    tree.bind("<Double-1>", choose)
    root.bind("<Return>", choose)
    root.bind("<Escape>", no_target)
    root.protocol("WM_DELETE_WINDOW", no_target)

    buttons = ttk.Frame(outer)
    buttons.pack(fill="x", pady=(12, 0))
    ttk.Button(buttons, text="Analyse with this target", style="Accent.TButton", command=choose).pack(side="right")
    ttk.Button(buttons, text="No target", style="Plain.TButton", command=no_target).pack(side="right", padx=(0, 8))
    fill()

    root.update_idletasks()
    w, h = 820, min(640, 230 + 30 * len(ranked))
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 3
    root.geometry(f"{w}x{max(h, 380)}+{x}+{y}")
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    search.focus_set()
    if on_ready is not None:
        root.after(200, lambda: on_ready(root))
    root.mainloop()
    return chosen["value"]


__all__ = ["pick_target", "suggest_targets"]
