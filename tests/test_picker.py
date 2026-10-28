from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elessar import see, suggest_targets
from elessar.picker import _terminal, pick_target


@pytest.fixture
def churn() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure": rng.integers(1, 72, n),
        "monthly": rng.normal(70, 20, n),
        "contract": rng.choice(["month", "year"], n),
        "Churn": rng.choice(["Yes", "No"], n),
        "notes": [f"free text number {i} with words" for i in range(n)],
    })


def test_suggestions_rank_the_target_first(churn: pd.DataFrame) -> None:
    ranked = suggest_targets(churn)
    assert ranked[0]["column"] == "Churn"
    assert ranked[-1]["column"] in ("customer_id", "notes")
    assert {r["column"] for r in ranked} == set(churn.columns)
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [0, 1, 0, 1], "label": ["x", "y", "x", "y"]})
    assert suggest_targets(df)[0]["column"] == "label"


def test_terminal_prompt(churn: pd.DataFrame) -> None:
    ranked = suggest_targets(churn)
    shown: list[str] = []

    def answers(*values: str):  # noqa: ANN202
        it = iter(values)
        return lambda _prompt: next(it)

    assert _terminal(ranked, answers(""), shown.append) == "Churn"  # Enter = the suggestion
    assert _terminal(ranked, answers("2"), shown.append) == "tenure"  # numbers follow the column order
    assert _terminal(ranked, answers("monthly"), shown.append) == "monthly"
    assert _terminal(ranked, answers("99", "zzz", "0"), shown.append) is None
    assert any("neither a number" in line for line in shown)
    assert any("★" in line and "Churn" in line for line in shown)


def test_ask_without_display_or_terminal(churn: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    with pytest.raises(RuntimeError, match="interactive terminal"):
        pick_target(churn)


def test_see_ask_uses_the_picker(churn: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    monkeypatch.setattr("elessar.core.pick_target", lambda df: "Churn")
    v = see(churn, target="ask", plots=False, verbose=False)
    assert v.target == "Churn"
