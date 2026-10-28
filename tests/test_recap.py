from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import pytest

from elessar import see
from elessar import recap as recap_module
from elessar.recap import render_markdown, should_open


def _md_images(text: str) -> list[str]:
    return [unquote(m) for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)]


def _html_images(text: str) -> list[str]:
    return [unquote(m) for m in re.findall(r'<img src="([^"]+)"', text)]


def test_recap_written_with_every_chart(messy_df: pd.DataFrame, tmp_path: Path) -> None:
    v = see(messy_df, target="target", out_dir=tmp_path / "out", verbose=False, name="messy.csv")
    md, page = v.recap["md"], v.recap["html"]
    assert md.parent == page.parent == tmp_path / "out"
    md_text, html_text = md.read_text(), page.read_text()
    for section in ("Key findings", "Overall statistics", "Target analysis", "Feature selection", "Outliers",
                    "Data quality", "Multivariate statistics", "Preparation plan", "Attribute statistics", "Run"):
        assert f"## {section}" in md_text and f">{section}</h2>" in html_text
    assert md_text.startswith("# messy.csv")
    referenced = set(_md_images(md_text))
    assert referenced == set(_html_images(html_text))
    assert all((md.parent / p).exists() for p in referenced)
    assert {p.name for p in v.plots} == {Path(p).name for p in referenced}  # every chart is in the recap


def test_hostile_names_are_escaped(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    names = ["a|b", "`tick`", "*star*_x", "<script>alert(1)</script>", "[link](x)", "# head"]
    df = pd.DataFrame({n: rng.normal(size=80) for n in names}).assign(g=rng.choice(["u|v", "<b>"], 80))
    v = see(df, target="g", out_dir=tmp_path, verbose=False)
    md_text, html_text = v.recap["md"].read_text(), v.recap["html"].read_text()
    assert "<script>alert(1)</script>" not in html_text
    assert "&lt;script&gt;" in html_text
    lines = md_text.splitlines()
    start = lines.index(next(line for line in lines if line.startswith("| column | type | missing")))
    table = [line for line in lines[start:start + len(names) + 4] if line.startswith("|")]

    def cells(line: str) -> int:
        return len(re.findall(r"(?<!\\)\|", line))

    assert len(table) == len(names) + 1 + 2  # header, rule, one row per column (target "g" included)
    assert len({cells(line) for line in table}) == 1  # "|" inside names never adds a cell


def test_markdown_code_spans_survive_backticks() -> None:
    text = render_markdown([("p", [("code", "a`b")]), ("p", [("code", "``x``")])])
    assert "``a`b``" in text and "``` ``x`` ```" in text


def test_open_logic(monkeypatch: pytest.MonkeyPatch, messy_df: pd.DataFrame, tmp_path: Path) -> None:
    assert should_open() is False  # pytest captures stdout: never auto-open in tests
    monkeypatch.setenv("CI", "1")
    assert should_open() is False
    opened: list[Path] = []
    monkeypatch.setattr(recap_module, "open_file", lambda p: opened.append(Path(p)) or True)
    v = see(messy_df, out_dir=tmp_path, verbose=False, open_report=False)
    assert opened == []
    recap_module.maybe_open(v, True)
    recap_module.maybe_open(v, "md")
    recap_module.maybe_open(v, None)
    assert [p.name for p in opened] == ["report.html", "report.md"]


def test_no_recap_without_plots(messy_df: pd.DataFrame, tmp_path: Path) -> None:
    v = see(messy_df, plots=False, verbose=False, out_dir=tmp_path / "nothing")
    assert v.recap == {} and not (tmp_path / "nothing").exists()
    md = v.to_markdown(tmp_path / "solo" / "r.md")
    assert md.exists() and "## Key findings" in md.read_text()
    assert v.to_html(tmp_path / "solo" / "r.html").read_text().startswith("<!doctype html>")
