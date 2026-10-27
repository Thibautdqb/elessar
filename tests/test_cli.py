from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from elessar.cli import main, read_table


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


def test_read_table_formats(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_csv(tmp_path / "t.tsv", sep="\t", index=False)
    df.to_csv(tmp_path / "t.csv.gz", index=False)
    df.to_json(tmp_path / "t.json")
    df.to_csv(tmp_path / "semi.txt", sep=";", index=False)
    for name in ("t.tsv", "t.csv.gz", "t.json", "semi.txt"):
        assert read_table(tmp_path / name).shape == (3, 2), name
    assert len(read_table(tmp_path / "t.tsv", nrows=2)) == 2
    with pytest.raises(SystemExit):
        read_table(tmp_path / "t.unknown")
