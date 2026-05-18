"""
domain/forecasting/intervals.py 단위 테스트.

요구사항: 단위 테스트 Coverage 60% 이상
"""
import numpy as np
import pytest

from domain.forecasting.intervals import (
    build_quantile_interval,
    build_residual_interval,
    calculate_coverage,
    calculate_mean_interval_width,
)


def test_build_quantile_interval():
    lower = np.array([100, 200, 300])
    upper = np.array([150, 190, 350])  # 190은 역전됨

    safe_low, safe_up = build_quantile_interval(lower, upper)
    assert safe_low[1] == 190
    assert safe_up[1] == 200


def test_calculate_coverage():
    y_true = np.array([120, 195, 320])
    lower = np.array([100, 190, 300])
    upper = np.array([150, 200, 350])

    cov = calculate_coverage(y_true, lower, upper)
    assert cov == 1.0  # 모두 구간 안에 포함됨


def test_build_residual_interval_returns_tuple_of_ndarrays():
    preds = np.array([10., 20.])
    residuals = np.array([-1., 0., 1.])

    result = build_residual_interval(preds, residuals)

    assert isinstance(result, tuple) and len(result) == 2
    lower, upper = result
    assert isinstance(lower, np.ndarray)
    assert isinstance(upper, np.ndarray)


def test_build_residual_interval_shifts_preds_by_quantiles():
    # alpha=0.2 → 하위 10%, 상위 90% 분위수 오차 사용
    residuals = np.array([-2., -1., 0., 1., 2.])
    preds = np.array([10., 10.])

    lower, upper = build_residual_interval(preds, residuals, alpha=0.2)

    expected_lower_err = float(np.quantile(residuals, 0.1))  # -1.6
    expected_upper_err = float(np.quantile(residuals, 0.9))  # 1.6
    assert lower == pytest.approx([10. + expected_lower_err] * 2)
    assert upper == pytest.approx([10. + expected_upper_err] * 2)


def test_calculate_coverage_partial():
    # 첫 번째(100)와 세 번째(300)만 구간 내 포함, 두 번째(200)는 미포함
    y_true = np.array([100., 200., 300.])
    lower = np.array([90., 210., 290.])
    upper = np.array([110., 220., 310.])

    cov = calculate_coverage(y_true, lower, upper)
    assert cov == pytest.approx(2 / 3)


def test_calculate_mean_interval_width_known_value():
    lower = np.array([0., 0.])
    upper = np.array([4., 6.])  # 너비: 4, 6 → 평균 5.0

    assert calculate_mean_interval_width(lower, upper) == pytest.approx(5.0)