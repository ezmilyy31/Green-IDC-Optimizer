"""
domain/forecasting/metrics.py 단위 테스트.

대상: mae, rmse, mape — 순수 numpy 함수.
"""
import math

import numpy as np
import pytest

from domain.forecasting.metrics import mae, mape, rmse


class TestMAE:
    def test_perfect_prediction_returns_zero(self):
        assert mae([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_known_value(self):
        assert mae([0, 0], [1, 1]) == pytest.approx(1.0)

    def test_symmetric(self):
        # 방향과 무관: MAE(a,b) == MAE(b,a)
        assert mae([3], [1]) == mae([1], [3])

    def test_returns_float(self):
        assert isinstance(mae([1], [2]), float)

    @pytest.mark.parametrize("y_true,y_pred,expected", [
        ([10], [12], 2.0),
        ([0, 4], [2, 2], 2.0),
        ([-1, 1], [-2, 2], 1.0),
    ])
    def test_parametrized_values(self, y_true, y_pred, expected):
        assert mae(y_true, y_pred) == pytest.approx(expected)


class TestRMSE:
    def test_perfect_prediction_returns_zero(self):
        assert rmse([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_known_value(self):
        assert rmse([0], [3]) == pytest.approx(3.0)

    def test_returns_float(self):
        assert isinstance(rmse([1], [2]), float)

    def test_rmse_geq_mae_for_outlier(self):
        # 이상치(큰 오차)에 제곱 패널티가 적용되므로 RMSE >= MAE
        y_true = [1, 10]
        y_pred = [2, 5]
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)

    @pytest.mark.parametrize("y_true,y_pred,expected", [
        ([0, 0], [3, 4], math.sqrt(12.5)),
        ([1], [1], 0.0),
    ])
    def test_parametrized_values(self, y_true, y_pred, expected):
        assert rmse(y_true, y_pred) == pytest.approx(expected, rel=1e-4)


class TestMAPE:
    def test_known_value(self):
        assert mape([100], [110]) == pytest.approx(10.0)

    def test_zero_true_uses_eps_denominator(self):
        # y_true=0이면 abs(y_true) < eps → eps로 분모 대체, 0 나눗셈 방지
        result = mape([0], [1], eps=1e-6)
        assert math.isfinite(result)

    def test_perfect_prediction_returns_zero(self):
        assert mape([5, 10], [5, 10]) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(mape([1], [2]), float)

    @pytest.mark.parametrize("y_true,y_pred,expected", [
        ([100, 200], [110, 200], 5.0),
        ([50], [25], 50.0),
    ])
    def test_parametrized_values(self, y_true, y_pred, expected):
        assert mape(y_true, y_pred) == pytest.approx(expected)
