"""Statistical validation: simulations with a known truth, run apart from the unit tests.

    python -m pytest -m validation tests/validation                   # run them (a few minutes)
    ELESSAR_VALIDATION_OUT=results.json python -m pytest -m validation tests/validation

Every test measures a rate (type I error, detection, coverage, right number of factors...) on data whose
truth is known and checks it against a bound that leaves room for simulation noise: the binomial
standard error of the rate, never a bound tuned to pass. With ``ELESSAR_VALIDATION_OUT`` the measures are
written to that JSON file: VALIDATION.md is built from it.
"""

from __future__ import annotations

import json
import os
import platform
import time
from typing import Any

import numpy as np
import pytest

_RESULTS: dict[str, Any] = {}


@pytest.fixture
def record(request: pytest.FixtureRequest):  # noqa: ANN201
    """``record(**measures)``: keep the measures of this test for the JSON report."""
    def keep(**measures: Any) -> None:
        _RESULTS.setdefault(request.node.name, {}).update(
            {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in measures.items()})
    return keep


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    out = os.environ.get("ELESSAR_VALIDATION_OUT")
    if not out or not _RESULTS:
        return
    import elessar
    import pandas
    import scipy
    import sklearn

    meta = {"elessar": elessar.__version__, "python": platform.python_version(), "pandas": pandas.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
            "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"), "exit_status": int(exitstatus)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": _RESULTS}, fh, indent=2, default=str)

