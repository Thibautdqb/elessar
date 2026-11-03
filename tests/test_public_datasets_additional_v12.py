"""Offline public scikit-learn datasets: descriptive, reporting and reference checks."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits

import elessar
from elessar.correlations import compute_correlations


@pytest.fixture
def config():
    return elessar.StoneConfig(
        n_jobs=1, random_state=42, fit_timeout=.15,
        candidates_continuous=("norm",), candidates_discrete=("poisson",),
    )


@pytest.mark.parametrize("dataset", ("breast_cancer", "diabetes", "digits"))
def test_public_dataset_report_and_numeric_references(dataset, tmp_path, config):
    if dataset == "breast_cancer":
        frame = load_breast_cancer(as_frame=True).frame.iloc[:, list(range(8)) + [-1]].copy()
    elif dataset == "diabetes":
        frame = load_diabetes(as_frame=True).frame.copy()
    else:
        all_digits = load_digits(as_frame=True).frame
        cols = [c for c in all_digits if c != "target" and all_digits[c].nunique() > 1][:8]
        frame = all_digits.iloc[:180][cols + ["target"]].copy()

    # Digits is an unsupervised profiling smoke test: the full multiclass
    # selection path is covered by other validation cases.
    target = None if dataset == "digits" else "target"
    report = elessar.see(frame, target=target, config=config, plots=True,
                         verbose=False, open_report=False, out_dir=tmp_path)
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.md").exists()
    assert len(report.summary) == len(frame.columns)
    if dataset != "digits":
        assert report.selection is not None
        assert report.selection.get("n_final_test", 0) > 0
    assert {"selection", "correlations", "outliers", "preparation"}.issubset(report.to_dict())

    for column in frame.columns[:3]:
        v = frame[column].dropna().to_numpy(dtype=float)
        row = report.summary.loc[column]
        assert int(row["count"]) == len(v)
        assert np.isclose(float(row["mean"]), np.mean(v), rtol=1e-9, atol=1e-9)
        assert np.isclose(float(row["std"]), np.std(v, ddof=1), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("dataset", ("breast_cancer", "diabetes", "digits"))
def test_public_dataset_correlations_match_scipy(dataset, config):
    if dataset == "breast_cancer":
        frame = load_breast_cancer(as_frame=True).frame[["mean radius", "mean texture"]]
    elif dataset == "diabetes":
        frame = load_diabetes(as_frame=True).frame[["bmi", "bp"]]
    else:
        frame = load_digits(as_frame=True).frame[["pixel_2_2", "pixel_2_3"]]

    a, b = list(frame.columns)
    output = compute_correlations(
        frame, {a: "numeric_continuous", b: "numeric_continuous"}, config=config
    )
    for key, reference in (
        ("pearson", stats.pearsonr),
        ("spearman", stats.spearmanr),
        ("kendall", stats.kendalltau),
    ):
        actual = float(output[key].loc[a, b])
        expected = float(reference(frame[a], frame[b]).statistic)
        assert np.isclose(actual, expected, rtol=0, atol=2e-12), (
            dataset, key, actual, expected,
        )
