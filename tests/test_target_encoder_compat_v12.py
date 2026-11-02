"""Test TargetEncoder compatibility with legacy and current sklearn APIs."""
import sklearn
import pytest
from sklearn.model_selection import KFold, StratifiedKFold
from elessar.preparation import make_target_encoder


def test_target_encoder_legacy_cv(monkeypatch):
    monkeypatch.setattr(sklearn, "__version__", "1.8.0")
    encoder = make_target_encoder("binary", 17)
    assert encoder.cv == 5
    assert encoder.random_state == 17


@pytest.mark.parametrize("kind,expected", [
    ("continuous", KFold),
    ("binary", StratifiedKFold),
    ("multiclass", StratifiedKFold),
])
def test_target_encoder_modern_cv(monkeypatch, kind, expected):
    monkeypatch.setattr(sklearn, "__version__", "1.9.1")
    encoder = make_target_encoder(kind, 17)
    assert isinstance(encoder.cv, expected)
    assert encoder.cv.random_state == 17
    assert encoder.cv.shuffle
