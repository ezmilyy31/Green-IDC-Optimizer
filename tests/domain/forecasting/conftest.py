import sys
from pathlib import Path

# pytest pythonpath 미설정 우회: 프로젝트 루트를 이 테스트 디렉토리 한정으로 sys.path 에 주입
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from domain.forecasting.lgbm_model import LGBMForecaster
from domain.forecasting.moving_avg import MovingAverageForecaster


@pytest.fixture
def tiny_train_df():
    """20행 결정론적 학습 데이터 (feat_a, feat_b, it_load_kw)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "feat_a": rng.uniform(0, 1, 20),
        "feat_b": rng.uniform(10, 30, 20),
        "it_load_kw": rng.uniform(50, 150, 20),
    })


@pytest.fixture
def fast_lgbm_params():
    """테스트용 초소형 파라미터 — 50ms 이내 학습 완료.
    min_data_in_leaf=1: 20행 데이터에서 기본값(20)이면 분기 불가 → 1로 낮춰 스플릿 허용.
    """
    return {"n_estimators": 5, "num_leaves": 4, "n_jobs": 1, "min_data_in_leaf": 1}


@pytest.fixture
def fitted_lgbm(tiny_train_df, fast_lgbm_params):
    """학습 완료 LGBMForecaster 인스턴스."""
    forecaster = LGBMForecaster(
        "it_load_kw", ["feat_a", "feat_b"], params=fast_lgbm_params
    )
    forecaster.fit(train_df=tiny_train_df)
    return forecaster


@pytest.fixture
def simple_series_short():
    """simple MA (window=3) 테스트용 5개 시계열."""
    return np.array([1., 2., 3., 4., 5.])


@pytest.fixture
def seasonal_series():
    """seasonal MA (window=2, period=4) 테스트용 12개 시계열.
    tile([10,20,30,40], 3) → 잔차 계산 시 4개 잔차 생성, 위상 정렬 검증 가능.
    """
    return np.tile([10., 20., 30., 40.], 3)


@pytest.fixture
def fitted_simple_ma(simple_series_short):
    """학습 완료 MovingAverageForecaster (simple, window=3)."""
    m = MovingAverageForecaster("val", kind="simple", window=3)
    m.fit(series=simple_series_short)
    return m


@pytest.fixture
def fitted_seasonal_ma(seasonal_series):
    """학습 완료 MovingAverageForecaster (seasonal, window=2, period=4)."""
    m = MovingAverageForecaster("val", kind="seasonal", window=2, seasonal_period=4)
    m.fit(series=seasonal_series)
    return m
