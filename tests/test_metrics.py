"""Tests for evaluation metrics."""

import numpy as np

from src.model.metrics import compute_all, mae, mape, rmse


def test_metrics_perfect_prediction():
    y = np.array([100.0, 110.0, 120.0])
    result = compute_all(y, y)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["mape"] == 0.0


def test_metrics_with_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    assert mae(y_true, y_pred) == 15.0
    assert rmse(y_true, y_pred) > 0
    assert mape(y_true, y_pred) > 0
