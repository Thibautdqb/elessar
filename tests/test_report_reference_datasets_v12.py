"""Integration tests on two public reference datasets for the final report contract."""
from pathlib import Path
import json
import numpy as np
import pytest
from sklearn.datasets import load_iris, load_wine
from elessar import see


@pytest.mark.parametrize("dataset,target,n", [
    ("wine", "cultivar", 178),
    ("iris", "species", 150),
])
def test_reference_report_coheres_with_selected_model_and_documentation(tmp_path: Path, dataset, target, n):
    raw = (load_wine if dataset == "wine" else load_iris)(as_frame=True)
    frame = raw.frame.rename(columns={"target": target})
    report = see(frame, target=target, verbose=False, plots=True,
                 open_report=False, out_dir=tmp_path / dataset)
    assert report.overview["n_rows"] == n
    assert report.selection is not None
    chosen = report.selection["consensus_evaluation"]
    assert chosen["status"] == "ok"
    assert np.isfinite(chosen["score"])
    assert chosen["n_test"] > 0
    assert set(chosen["features"]) == set(report.selection["keep"])
    assert chosen["n_model_fit"] > 0
    assert report.to_dict()["selection"]["consensus_evaluation"]["status"] == "ok"
    html_path = report.recap["html"]
    html = html_path.read_text(encoding="utf-8")
    assert "shadow-feature coefficients" in html
    assert "final-test outcomes never determine votes" in html
    assert "Exploratory component scores" in html if report.multivariate.get("factor", {}).get("descriptives") else True
    assert "non-zero coefficient at the penalty" not in html
    assert "10% of the largest" not in html
    assert "independent final consensus" in html.lower()
    report.to_json(tmp_path / f"{dataset}.json")
    exported = json.loads((tmp_path / f"{dataset}.json").read_text())
    assert exported["overview"]["n_rows"] == n
