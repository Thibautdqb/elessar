"""Verify a clean wheel/sdist installation through the public API and CLI."""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import elessar

assert importlib.metadata.version("elessar") == elessar.__version__ == "1.2.0"
installed = Path(elessar.__file__).resolve()
assert Path(sys.prefix).resolve() in installed.parents, "smoke test must import the clean installation"
cli = subprocess.run([sys.executable, "-I", "-m", "elessar", "--version"], check=True, capture_output=True, text=True)
assert elessar.__version__ in cli.stdout
rng = np.random.default_rng(645001)
df = pd.DataFrame({"x": rng.normal(size=80), "z": rng.normal(size=80)})
df["y"] = df.x + .5 * rng.normal(size=len(df))
with tempfile.TemporaryDirectory() as folder:
    report = elessar.see(df, target="y", plots=True, verbose=False, open_report=False,
                         config=elessar.StoneConfig(n_jobs=1, candidates_continuous=("norm",)), out_dir=folder)
    assert report.selection is not None
    assert (Path(folder) / "report.html").is_file()
    assert report.selection["consensus_evaluation"]["status"] == "ok"
print(json.dumps({"version": elessar.__version__, "installed_module": str(installed), "cli_and_report": "pass"}))
