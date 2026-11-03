"""Release gates for visible report statements and adaptive grouping."""
import inspect
import numpy as np
import pandas as pd
from elessar import recap
from elessar.config import StoneConfig
from elessar.preparation import group_categories, plan
from elessar.report import _target_rows


def test_selection_report_describes_shadow_votes():
    source = inspect.getsource(recap._Recap.selection_section)
    assert "shadow-feature coefficients" in source
    assert "10% of the largest" not in source
    assert "non-zero coefficient" not in source


def test_supervised_bucketing_snippet_is_self_contained():
    df = pd.DataFrame({"x": np.arange(200.), "y": (np.arange(200) > 99).astype(int)})
    details = {"x": {"semantic_type": "numeric_continuous", "stats": {"heaping_index": 1}},
               "y": {"semantic_type": "boolean", "stats": {}}}
    steps = plan(df, details, {}, {}, "y", StoneConfig())["steps"]
    code = next(s["code"] for s in steps if s["action"] == "Bucketing")
    context = {"df": df.copy(), "pd": pd}
    exec(code, context)
    assert context["df"]["x_bin"].notna().all()


def test_category_grouping_controls_independent_null_and_detects_signal():
    rng = np.random.default_rng(891)
    labels = np.tile([f"category_{j}" for j in range(12)], 80)
    false_splits = sum(group_categories(pd.Series(labels),
                                       pd.Series(rng.normal(size=len(labels))), True,
                                       StoneConfig(random_state=42)) is not None for _ in range(80))
    assert false_splits <= 12
    y = np.array([int(label.split("_")[1]) // 4 * 2 for label in labels]) + rng.normal(size=len(labels)) * .25
    assert group_categories(pd.Series(labels), pd.Series(y), True, StoneConfig(random_state=42)) is not None


def test_eta_effect_is_not_signed():
    association = pd.DataFrame([{"feature": "x", "feature_type": "numeric_continuous",
        "method": "eta", "association": .853, "test": "kruskal-wallis",
        "n": 178, "p_adjusted": .001, "significant": True, "auc": np.nan,
        "mutual_info": .1, "leakage_suspect": False}])
    _, rows = _target_rows(association)
    assert rows[0][3] == "0.853"
