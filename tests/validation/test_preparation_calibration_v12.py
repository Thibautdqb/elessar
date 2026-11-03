"""Pre-registered false-split diagnostic on iid no-signal categorical data."""
import numpy as np
import pandas as pd
import pytest
from elessar.config import StoneConfig
from elessar.preparation import group_categories


@pytest.mark.validation
def test_adaptive_category_grouping_iid_null_calibration(record):
    rng = np.random.default_rng(901)
    labels = pd.Series(np.tile([f"g{i}" for i in range(10)], 50))
    cfg = StoneConfig(random_state=11)
    n = 500
    false_splits = sum(group_categories(labels,
        pd.Series(rng.normal(size=len(labels))), True, cfg) is not None
        for _ in range(n))
    # Conservative 3-sigma binomial allowance above the nominal 5% bound.
    upper = .05 + 3 * np.sqrt(.05 * .95 / n)
    record(design="category_grouping_iid_null", n=n, false_splits=false_splits,
           rate=false_splits / n, upper=upper)
    assert false_splits / n <= upper
