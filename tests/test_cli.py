from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar.cli import main


@pytest.fixture
def csv_file(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=200), "g": rng.choice(["a", "b"], 200), "y": rng.integers(0, 2, 200)})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_cli_end_to_end(csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "rep"
    assert main([str(csv_file), "-t", "y", "-o", str(out), "--json", "--markdown", "--csv", "--width", "100"]) == 0
    printed = capsys.readouterr()
    assert "TARGET ANALYSIS" in printed.out and "wrote" in printed.err
    assert json.loads((out / "report.json").read_text())["overview"]["target"] == "y"
    assert (out / "report.md").exists() and (out / "summary.csv").exists()
    assert list(out.glob("*.png"))


def test_cli_quiet_no_plots_and_bad_target(csv_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(csv_file), "-q", "--no-plots", "-o", str(tmp_path / "o"), "--sample", "50", "--alpha", "0.01"]) == 0
    assert capsys.readouterr().out == ""
    assert main([str(csv_file), "-t", "nope", "--no-plots"]) == 2


def test_cli_reads_any_format_and_streams(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=3000), "y": rng.integers(0, 2, 3000)})
    df.to_csv(tmp_path / "d.csv.gz", sep=";", index=False)
    out = tmp_path / "o"
    assert main([str(tmp_path / "d.csv.gz"), "-t", "y", "-q", "--no-plots", "--max-rows", "1000", "-o", str(out),
                 "--json"]) == 0
    overview = json.loads((out / "report.json").read_text())["overview"]
    assert overview["n_rows"] == 3000 and overview["n_rows_profiled"] == 1000 and overview["sampled"]
    assert main([str(tmp_path / "missing.csv"), "--no-plots"]) == 2
    assert "no such file" in capsys.readouterr().err


def test_cli_never_asks_when_not_interactive(csv_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # stdin is not a terminal under pytest and there is no display: no picker, no target
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    assert main([str(csv_file), "-q", "--no-plots", "-o", str(tmp_path / "o")]) == 0
    assert main([str(csv_file), "-q", "--no-plots", "-t", "ask"]) == 2  # asked explicitly, but nobody can answer


def test_cli_type_option(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(2)
    path = tmp_path / "shops.csv"
    pd.DataFrame({"store": rng.integers(1, 30, 300), "basket": rng.lognormal(3, 1, 300)}).to_csv(path, index=False)
    out = tmp_path / "out"
    assert main([str(path), "--no-target", "--no-plots", "-o", str(out), "--json", "--open", "none",
                 "--type", "store=numeric_discrete"]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["columns"]["store"]["semantic_type"] == "numeric_discrete"
    assert main([str(path), "--no-target", "-q", "--type", "store=banana"]) == 2
    assert main([str(path), "--no-target", "-q", "--type", "store"]) == 2
    assert "choose one of" in capsys.readouterr().err


def test_cli_descriptions_and_dictionary(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    path = tmp_path / "loans.csv"
    pd.DataFrame({"amount": rng.lognormal(10, 1, 200).round(), "status": rng.choice(list("ABCD"), 200)}).to_csv(path, index=False)
    described = tmp_path / "describe.csv"
    pd.DataFrame({"column": ["amount", "status", "nope"],
                  "description": ["loan amount (CZK)", "A finished OK, B finished unpaid", "x"]}).to_csv(described, index=False)
    out = tmp_path / "out"
    assert main([str(path), "--no-target", "-q", "--no-plots", "-o", str(out), "--describe", str(described),
                 "--dictionary", "--open", "none"]) == 0
    table = pd.read_csv(out / "dictionary.csv")
    assert table.set_index("column").loc["amount", "description"] == "loan amount (CZK)"
    assert "to" in table.set_index("column").loc["amount", "values"] and "A (" in table.set_index("column").loc["status", "values"]
    md = (out / "dictionary.md").read_text()
    assert md.startswith("| column | description |") and "A finished OK, B finished unpaid" in md
    assert main([str(path), "--no-target", "-q", "--describe", str(tmp_path / "missing.csv")]) == 2


def test_cli_several_targets(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 300
    x = rng.normal(size=n)
    path = tmp_path / "clients.csv"
    pd.DataFrame({"x": x, "z": rng.normal(size=n), "loan": (x + rng.normal(size=n) > 0).astype(int),
                  "card": rng.integers(0, 2, n)}).to_csv(path, index=False)
    out = tmp_path / "out"
    assert main([str(path), "-t", "loan", "-t", "card", "-q", "--no-plots", "-o", str(out), "--json", "--open", "none"]) == 0
    for t in ("loan", "card"):
        report = json.loads((out / f"target-{t}" / "report.json").read_text())
        assert report["target"] == t
    assert main([str(path), "-t", "loan", "-t", "nope", "-q", "--no-plots"]) == 2
