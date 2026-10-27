from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from elessar import see_plot


def test_see_plot_writes_informative_pngs(messy_df: pd.DataFrame, tmp_path: Path) -> None:
    paths = see_plot(messy_df, tmp_path, target="target")
    names = [p.name for p in paths]
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)
    assert names == sorted(names)
    assert any(n.endswith("_dataset_missing.png") for n in names)
    assert any(n.endswith("_income_dist.png") for n in names)
    assert any(n.endswith("_city_bar.png") for n in names)
    assert any(n.endswith("_when_time.png") for n in names)
    assert any(n.endswith("_scatter.png") for n in names)
    assert any(n.endswith("_target.png") for n in names)
    for kind in ("nullity_corr", "corr_pearson", "corr_spearman", "corr_cramers_v", "association", "target_ranking"):
        assert any(n.endswith(f"_{kind}.png") for n in names), kind
    for skipped in ("id", "uuid", "constant", "all_nan"):
        assert not any(n.split("_", 1)[1].startswith(skipped + "_") for n in names)
    assert "matplotlib.pyplot" not in sys.modules or not sys.modules["matplotlib.pyplot"].get_fignums()


def test_skips_uninformative_plots(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2})
    names = [p.name for p in see_plot(df, tmp_path)]
    assert not any("missing" in n for n in names)  # no missing values
    assert not any("corr" in n for n in names)  # fewer than 3 numeric columns


def test_replaces_previous_plots_only(tmp_path: Path) -> None:
    keep = tmp_path / "notes.txt"
    keep.write_text("mine")
    stale = tmp_path / "99_old_dist.png"
    stale.write_bytes(b"x")
    see_plot(pd.DataFrame({"x": np.random.default_rng(0).normal(size=100)}), tmp_path)
    assert keep.exists() and not stale.exists()


def test_see_plot_edge(edge_df: pd.DataFrame, tmp_path: Path) -> None:
    for p in see_plot(edge_df, tmp_path):
        assert p.exists()


def test_text_plot(tmp_path: Path) -> None:
    words = "the quick brown fox jumps over a lazy dog again and again".split()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"comment": [" ".join(rng.choice(words, rng.integers(3, 15))) for _ in range(300)]})
    assert [p.name for p in see_plot(df, tmp_path)] == ["00_comment_text.png"]


def test_overview_and_headers(messy_df: pd.DataFrame, tmp_path: Path) -> None:
    from elessar.plots import _compact

    names = [p.name for p in see_plot(messy_df, tmp_path)]
    assert names[0] == "00_dataset_overview.png"
    assert [_compact(v) for v in (0, 2020, 12_500, 3.4e6, 2e9, 0.05, 1e-5)] == [
        "0", "2020", "12.5k", "3.4M", "2B", "0.05", "1e-05"]


def test_log_axes_never_show_mathtext(tmp_path: Path) -> None:
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from elessar.plots import STYLE, _set_log

    with mpl.rc_context(STYLE):
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot()
        v = np.r_[0.0, np.geomspace(1, 1e5, 50)]
        ax.plot(v, v)
        _set_log(ax, v, "xy")
        fig.canvas.draw()
        labels = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
    assert labels and not any("$" in t or "mathdefault" in t for t in labels)
