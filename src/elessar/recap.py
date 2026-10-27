"""The recap: one document (Markdown and HTML) with key findings first and every chart in place.

The recap is built once as a list of blocks, then rendered twice. Inline text is a list of
``(style, text)`` spans (``""``, ``"code"``, ``"b"`` or ``"i"``), never pre-formatted markup, so column
names and values are escaped correctly for each format whatever characters they contain.
"""

from __future__ import annotations

import datetime as dt
import html
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, Union
from urllib.parse import quote

import pandas as pd

from ._utils import fmt_bytes
from .report import _detail_parts, _overview_md, _p, _pair_rows, _target_rows

if TYPE_CHECKING:
    from .report import Vision

Span = tuple[str, str]
Inline = Union[str, list[Span]]  # a plain string is one plain span
Block = tuple[Any, ...]
SECTIONS = {"findings": "Key findings", "target": "Target analysis", "quality": "Data quality",
            "associations": "Associations", "columns": "Columns"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "info": 2}


def _strength(value: float) -> str:
    """Words for an association on a 0-1 scale (Cohen-style rule of thumb)."""
    a = abs(value)
    return "strongly" if a >= 0.5 else "moderately" if a >= 0.3 else "weakly" if a >= 0.1 else "very weakly"


def code(text: Any) -> Span:
    return ("code", str(text))


def bold(text: Any) -> Span:
    return ("b", str(text))


def _spans(inline: Inline) -> list[Span]:
    return [("", inline)] if isinstance(inline, str) else inline


# ------------------------------------------------------------------------------------------- building
class _Recap:
    """Collects blocks and knows where each chart is (paths relative to the recap file)."""

    def __init__(self, vision: Vision, base: Path) -> None:
        self.v = vision
        self.base = base.resolve()
        self.blocks: list[Block] = []

    def rel(self, path: Path) -> str:
        return Path(os.path.relpath(Path(path).resolve(), self.base)).as_posix()

    def charts(self, kinds: tuple[str, ...], column: str | None = None, any_column: bool = False) -> list[tuple[str, str]]:
        out = []
        for rec in self.v.plot_index:
            if rec["kind"] in kinds and (any_column or rec["column"] == column):
                label = rec["kind"] if rec["column"] is None else " vs ".join(
                    str(c) for c in (rec["column"], rec.get("other")) if c is not None)
                out.append((self.rel(rec["path"]), label))
        return out

    def add(self, *block: Any) -> None:
        self.blocks.append(block)

    # sections ---------------------------------------------------------------------------------
    def build(self, title: str) -> list[Block]:
        v, o = self.v, self.v.overview
        self.add("h", 1, title, "top")
        self.add("p", [("i", f"{o['n_rows']:,} rows × {o['n_columns']} columns · generated "
                             f"{dt.datetime.now():%Y-%m-%d %H:%M} by elessar")])
        toc = ["findings"] + (["target"] if v.target else []) + ["quality", "associations", "columns"]
        self.add("toc", [(key, SECTIONS[key]) for key in toc])
        self.add("table", ["", ""], [[[bold(k.capitalize())], v_] for k, v_ in _overview_md(o).items()], "kv")
        self.findings()
        if v.target:
            self.target_section()
        self.quality_section()
        self.association_section()
        self.column_section()
        return self.blocks

    def findings(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["findings"], "findings")
        items: list[Inline] = []
        assoc = v.correlations.get("target")
        if v.target and assoc is not None and not assoc.empty:
            sig = assoc[assoc["significant"]]
            for r in sig.head(3).itertuples(index=False):
                extra = f", AUC {r.auc:.2f}" if pd.notna(r.auc) else ""
                items.append([code(r.feature), ("", f" is {_strength(r.association)} associated with "), code(v.target),
                              ("", f" ({r.method} {r.association:.3f}{extra}, p_adj {_p(r.p_adjusted)})")])
            if len(assoc) > len(sig):
                names = ", ".join(assoc[~assoc["significant"]]["feature"].astype(str).head(6))
                n_none = len(assoc) - len(sig)
                items.append([("", f"{n_none} feature{' shows' if n_none == 1 else 's show'} no significant link "
                                   "with "), code(v.target), ("", f": {names}")])
        ranked = sorted(v.alerts, key=lambda a: _SEVERITY_ORDER.get(a["severity"], 9))
        for a in [a for a in ranked if a["severity"] != "info"][:12]:
            prefix = [(f"sev-{a['severity']}", a["severity"].upper()), ("", " ")] + (
                [code(a["column"]), ("", ": ")] if a["column"] else [])
            items.append(prefix + [("", a["message"])])
        if not items:
            items.append("No issue found, and nothing stands out.")
        self.add("ul", items)
        self.figures(self.charts(("overview",)))

    def target_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["target"], "target")
        self.add("p", [("", "Target: "), code(v.target), ("", f" ({v.columns[v.target]['semantic_type']})")])
        self.figures(self.charts(("target_ranking",), v.target))
        assoc = v.correlations.get("target")
        if assoc is not None and not assoc.empty:
            headers, rows = _target_rows(assoc)
            self.add("table", headers, rows, "data")
        mt = v.correlations.get("missing_target")
        if mt is not None and not mt.empty:
            self.add("h", 3, "Is the target different when a feature is missing?", None)
            self.add("table", ["feature", "% missing", "significant", "p_adj", "detail"],
                     [[[code(r.feature)], f"{r.pct_missing:.1f}", "yes" if r.significant else "no", _p(r.p_adjusted),
                       r.detail] for r in mt.itertuples(index=False)], "data")
        self.figures(self.charts(("target",), v.target), grid=True)

    def quality_section(self) -> None:
        v, q = self.v, self.v.quality
        self.add("h", 2, SECTIONS["quality"], "quality")
        if v.alerts:
            ranked = sorted(v.alerts, key=lambda a: _SEVERITY_ORDER.get(a["severity"], 9))
            self.add("table", ["severity", "column", "message"],
                     [[a["severity"], [code(a["column"])] if a["column"] else "-", a["message"]] for a in ranked], "alerts")
        else:
            self.add("p", "No data-quality issue found.")
        self.figures(self.charts(("missing", "nullity_corr")))
        miss = v.correlations.get("missingness")
        if miss is not None and not miss.empty and miss["significant"].any():
            self.add("h", 3, "Missing values that are not random", None)
            sig = miss[miss["significant"]].drop_duplicates("column").head(20)
            self.add("table", ["column", "% missing", "depends on", "detail", "p_adj"],
                     [[[code(r.column)], f"{r.pct_missing:.1f}", [code(r.depends_on)], r.detail, _p(r.p_adjusted)]
                      for r in sig.itertuples(index=False)], "data")
        opts = q.get("dtype_optimizations") or []
        if opts:
            self.add("h", 3, f"Memory: suggested dtypes save {fmt_bytes(q.get('potential_saving_bytes') or 0)}", None)
            self.add("table", ["column", "current", "suggested", "saving", "code"],
                     [[[code(o_["column"])], o_["current_dtype"], o_["suggested_dtype"],
                       f"{fmt_bytes(o_['saving_bytes'])} ({o_['saving_pct']:.0f}%)", [code(o_["code"])]] for o_ in opts],
                     "data")

    def association_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["associations"], "associations")
        pairs, top = v.correlations.get("pairs"), v.correlations.get("top_pairs")
        if pairs is not None and not pairs.empty:
            self.add("p", f"{len(pairs):,} pairs tested, {int(pairs['significant'].sum()):,} significant after "
                          f"Benjamini-Hochberg correction (alpha {v.alpha:g}).")
        self.figures(self.charts(("association", "corr_pearson", "corr_spearman", "corr_cramers_v")), grid=True)
        if top is not None and not top.empty:
            self.add("h", 3, "Strongest associations", None)
            headers, rows = _pair_rows(top.head(30))
            self.add("table", headers, rows, "data")
        deps = v.correlations.get("linear_dependencies")
        if deps is not None and not deps.empty:
            self.add("h", 3, "Exact linear dependencies", None)
            self.add("ul", [[code(r.equation)] for r in deps.itertuples(index=False)])
        vif = v.correlations.get("vif")
        if vif is not None and not vif.empty:
            self.add("h", 3, "Multicollinearity (VIF)", None)
            self.add("table", ["column", "VIF"],
                     [[[code(k)], "inf" if val == float("inf") else f"{val:.2f}"] for k, val in vif["vif"].items()], "data")
        self.figures(self.charts(("scatter",), any_column=True), grid=True)

    def column_section(self) -> None:
        v = self.v
        self.add("h", 2, SECTIONS["columns"], "columns")
        rows = []
        for i, (name, d) in enumerate(v.columns.items()):
            st = d.get("stats", {})
            rows.append([[("link", f"col-{i}"), code(name)], d.get("semantic_type", ""),
                         f"{st.get('pct_missing', 0):.1f}%", f"{st.get('n_unique', 0):,}",
                         str(len(d.get("alerts", [])))])
        self.add("table", ["column", "type", "missing", "distinct", "alerts"], rows, "data")
        for i, (name, d) in enumerate(v.columns.items()):
            body: list[Block] = [("p", [("i", f"{d.get('pandas_dtype')} · {d.get('reason')}")]),
                                 ("ul", [" · ".join(p for p in parts if p) for parts in _detail_parts(d) if any(parts)])]
            for path, label in self.charts(("dist", "bar", "time", "text"), name):
                body.append(("img", path, f"{name}: {label}"))
            summary = [code(name), ("", f" · {d.get('semantic_type')}")]
            n_alerts = len(d.get("alerts", []))
            if n_alerts:
                summary.append(("", f" · {n_alerts} alert{'s' if n_alerts > 1 else ''}"))
            self.add("details", summary, body, f"col-{i}")

    def figures(self, charts: list[tuple[str, str]], grid: bool = False) -> None:
        if charts:
            self.add("figures", charts, grid)


# ------------------------------------------------------------------------------------------- markdown
def _md_escape(text: str, table: bool = False) -> str:
    text = re.sub(r"([\\`*_\[\]<>#])", r"\\\1", text)
    return text.replace("|", "\\|").replace("\n", " ") if table else text


def _md_code(text: str, table: bool = False) -> str:
    fence = "`" * (max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    body = text.replace("|", "\\|").replace("\n", " ") if table else text
    pad = " " if body.startswith("`") or body.endswith("`") or not body.strip() else ""
    return f"{fence}{pad}{body}{pad}{fence}"


def _md_inline(inline: Inline, table: bool = False) -> str:
    out = []
    for style, text in _spans(inline):
        if style == "code":
            out.append(_md_code(text, table))
        elif style == "b":
            out.append(f"**{_md_escape(text, table)}**")
        elif style == "i":
            out.append(f"*{_md_escape(text, table)}*")
        elif style.startswith("sev-"):
            out.append(f"**{_md_escape(text, table)}**")
        elif style != "link":
            out.append(_md_escape(text, table))
    return "".join(out)


def _md_url(path: str) -> str:
    return quote(path, safe="/._-~")


def render_markdown(blocks: list[Block]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            lines += ["#" * block[1] + " " + _md_escape(block[2]), ""]
        elif kind == "p":
            lines += [_md_inline(block[1]), ""]
        elif kind == "toc":
            lines += [" · ".join(f"[{_md_escape(label)}](#{key_slug(label)})" for _, label in block[1]), ""]
        elif kind == "ul":
            lines += [f"- {_md_inline(item)}" for item in block[1]] + [""]
        elif kind == "table":
            headers, rows = block[1], block[2]
            lines.append("| " + " | ".join(_md_escape(h, True) or " " for h in headers) + " |")
            lines.append("|" + "|".join("---" for _ in headers) + "|")
            lines += ["| " + " | ".join(_md_inline(c, True) for c in row) + " |" for row in rows]
            lines.append("")
        elif kind == "img":
            lines += [f"![{_md_escape(block[2])}]({_md_url(block[1])})", ""]
        elif kind == "figures":
            lines += [line for path, label in block[1] for line in (f"![{_md_escape(label)}]({_md_url(path)})", "")]
        elif kind == "details":
            lines += ["<details>", f"<summary>{_html_inline(block[1])}</summary>", ""]
            lines += render_markdown(block[2]).splitlines()
            lines += ["</details>", ""]
    return "\n".join(lines).rstrip() + "\n"


def key_slug(title: str) -> str:
    """GitHub's heading anchor for the fixed section titles."""
    return re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")


# ------------------------------------------------------------------------------------------- html
def _html_inline(inline: Inline) -> str:
    out, link = [], None
    for style, text in _spans(inline):
        if style == "link":
            link = text
            continue
        t = html.escape(text)
        piece = {"code": f"<code>{t}</code>", "b": f"<strong>{t}</strong>", "i": f"<em>{t}</em>"}.get(style, t)
        if style.startswith("sev-"):
            piece = f'<span class="chip {html.escape(style)}">{t}</span>'
        if link:
            piece, link = f'<a href="#{html.escape(link)}">{piece}</a>', None
        out.append(piece)
    return "".join(out)


def render_html(blocks: list[Block], title: str) -> str:
    return _PAGE.format(title=html.escape(title), body=_html_body(blocks))


def _html_body(blocks: list[Block]) -> str:
    body: list[str] = []
    for block in blocks:
        kind = block[0]
        if kind == "h":
            anchor = f' id="{block[3]}"' if block[3] else ""
            body.append(f"<h{block[1]}{anchor}>{html.escape(block[2])}</h{block[1]}>")
        elif kind == "p":
            body.append(f"<p>{_html_inline(block[1])}</p>")
        elif kind == "toc":
            body.append('<nav class="toc">' + "".join(f'<a href="#{key}">{html.escape(label)}</a>'
                                                     for key, label in block[1]) + "</nav>")
        elif kind == "ul":
            body.append("<ul>" + "".join(f"<li>{_html_inline(i)}</li>" for i in block[1]) + "</ul>")
        elif kind == "table":
            headers, rows, cls = block[1], block[2], block[3]
            head = "" if cls == "kv" else "<thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead>"
            trs = []
            for row in rows:
                attrs = f' class="sev-{row[0]}"' if cls == "alerts" else ""
                trs.append(f"<tr{attrs}>" + "".join(f"<td>{_html_inline(c)}</td>" for c in row) + "</tr>")
            body.append(f'<div class="table-wrap"><table class="{cls}">{head}<tbody>{"".join(trs)}</tbody></table></div>')
        elif kind == "img":
            body.append(_figure(block[1], block[2]))
        elif kind == "figures":
            cls = "figures grid" if block[2] and len(block[1]) > 1 else "figures"
            body.append(f'<div class="{cls}">' + "".join(_figure(p, label) for p, label in block[1]) + "</div>")
        elif kind == "details":
            body.append(f'<details id="{block[3]}"><summary>{_html_inline(block[1])}</summary>'
                        f"{_html_body(block[2])}</details>")
    return "\n".join(body)


def _figure(path: str, label: str) -> str:
    src = html.escape(quote(path, safe="/._-~"))
    return f'<figure><a href="{src}"><img src="{src}" alt="{html.escape(label)}" loading="lazy"></a></figure>'


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg:#f4f3ef; --surface:#fcfcfb; --ink:#1b1b19; --ink2:#52514e; --muted:#8f8e88; --line:#e6e4de;
  --blue:#2a78d6; --high:#c9302c; --medium:#b7791f; --info:#2a78d6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 Inter, Lato, "Helvetica Neue", Arial, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 20px 80px; }}
h1 {{ font-size:30px; margin:8px 0 2px; letter-spacing:-0.01em; }}
h2 {{ font-size:22px; margin:44px 0 12px; padding-top:8px; border-top:1px solid var(--line); }}
h3 {{ font-size:16px; margin:26px 0 8px; }}
p {{ color:var(--ink2); }}
code {{ font:13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#ebe9e3; padding:1px 5px; border-radius:4px; }}
a {{ color:var(--blue); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
nav.toc {{ position:sticky; top:0; z-index:5; display:flex; gap:18px; flex-wrap:wrap; padding:10px 14px; margin:14px 0 18px;
  background:var(--surface); border:1px solid var(--line); border-radius:10px; font-weight:600; }}
ul {{ padding-left:20px; }} li {{ margin:4px 0; }}
.table-wrap {{ overflow-x:auto; margin:10px 0 18px; border:1px solid var(--line); border-radius:10px; background:var(--surface); }}
table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
th, td {{ text-align:left; padding:7px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); background:#f7f6f2; }}
tr:last-child td {{ border-bottom:none; }}
table.kv td:first-child {{ width:190px; color:var(--ink2); }}
tr.sev-high td:first-child {{ color:var(--high); font-weight:700; }}
tr.sev-medium td:first-child {{ color:var(--medium); font-weight:700; }}
tr.sev-info td:first-child {{ color:var(--info); }}
.chip {{ display:inline-block; font-size:11px; font-weight:700; letter-spacing:.04em; padding:1px 7px; border-radius:99px; }}
.chip.sev-high {{ background:#fde2e1; color:var(--high); }} .chip.sev-medium {{ background:#fdf0d5; color:var(--medium); }}
.chip.sev-info {{ background:#dbe9fb; color:var(--info); }}
figure {{ margin:14px 0; }}
.figures:not(.grid) figure img, details figure img {{ width:auto; max-width:100%; max-height:760px; margin:0 auto; }}
figure img {{ width:100%; height:auto; display:block; border:1px solid var(--line); border-radius:10px; background:#fcfcfb; }}
.figures.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(480px, 1fr)); gap:14px; }}
.figures.grid figure {{ margin:0; }}
details {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; margin:8px 0; padding:0 16px; }}
details[open] {{ padding-bottom:12px; }}
summary {{ cursor:pointer; padding:11px 0; font-weight:600; }}
details > ul {{ font-size:13.5px; color:var(--ink2); }}
@media (max-width: 640px) {{ main {{ padding:16px; }} .figures.grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
{body}
</main>
<script>
function openTarget() {{ var el = document.getElementById(location.hash.slice(1)); if (el && el.tagName === "DETAILS") el.open = true; }}
window.addEventListener("hashchange", openTarget); openTarget();
</script>
</body>
</html>
"""


# ------------------------------------------------------------------------------------------- entry points
def build(vision: Vision, base: Path, title: str | None = None) -> list[Block]:
    return _Recap(vision, base).build(title or vision.name or "elessar report")


def write(vision: Vision, out_dir: str | Path, title: str | None = None) -> dict[str, Path]:
    """Write ``report.md`` and ``report.html`` in ``out_dir``; returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks = build(vision, out, title)
    md, page = out / "report.md", out / "report.html"
    md.write_text(render_markdown(blocks), encoding="utf-8")
    page.write_text(render_html(blocks, title or vision.name or "elessar report"), encoding="utf-8")
    return {"md": md, "html": page}


def maybe_open(vision: Vision, open_report: bool | str | None, stream: TextIO | None = None) -> Path | None:
    """Open the recap per ``open_report`` (see :func:`elessar.see`) and say where it is on ``stream``."""
    if not vision.recap or open_report is False or (open_report is None and not should_open()):
        return None
    path = vision.recap["md" if open_report == "md" else "html"].resolve()
    opened = open_file(path)
    if stream is not None:
        print(f"\nRecap: {path}" + ("" if opened else "  (open it in a browser)"), file=stream)
    return path


def should_open() -> bool:
    """Open automatically only for a person at a terminal: never in CI, tests or pipes."""
    return sys.stdout.isatty() and not os.environ.get("CI")


def open_file(path: Path) -> bool:
    """Open ``path`` with the system default application; False when that is not possible."""
    path = Path(path).resolve()
    try:
        if sys.platform == "darwin":
            cmd = ["open", str(path)]
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True
        elif os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or "microsoft" in platform.release().lower():
            cmd = ["xdg-open", str(path)]
        else:
            return False
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return True
    except (OSError, AttributeError):
        return False
