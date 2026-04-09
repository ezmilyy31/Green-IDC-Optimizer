import numpy as np
from typing import Tuple

"""
예측 구간(Prediction Interval) 계산 및 평가 로직.

이 파일의 책임:
- 점 예측(point forecast)을 불확실성 정보와 함께 확장 (Interval 생성)
- Quantile 역전 현상 방지 및 보수적 마진(Margin) 적용
- interval 품질 평가 지표 (Coverage, Width) 제공
"""

def build_residual_interval(
    preds: np.ndarray, 
    residuals: np.ndarray, 
    alpha: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    (참고용) 분위수 회귀(Quantile Regression) 모델이 없을 때, 
    과거 오차(Residuals) 분포를 기반으로 신뢰 구간을 생성합니다.
    
    - 90% 구간이라면 alpha=0.1 -> 하위 5%, 상위 95% 분위수 오차를 사용
    """
    # 잔차의 하위/상위 분위수 계산
    lower_bound_error = np.quantile(residuals, alpha / 2)
    upper_bound_error = np.quantile(residuals, 1 - (alpha / 2))
    
    lower_preds = preds + lower_bound_error
    upper_preds = preds + upper_bound_error
    
    return lower_preds, upper_preds


def build_quantile_interval(
    lower_preds: np.ndarray, 
    upper_preds: np.ndarray,
    margin_ratio: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Quantile Regression 출력값을 받아 역전 현상(Crossing)을 방지하고,
    안전성을 위해 선택적으로 마진(여유 폭)을 추가합니다.
    """
    # 1. 역전 현상 방지 (하한선이 상한선보다 큰 경우 교정)
    safe_lower = np.minimum(lower_preds, upper_preds)
    safe_upper = np.maximum(lower_preds, upper_preds)
    
    # 2. 보수적 마진 추가 (loader.py의 interval_config 설정 반영)
    if margin_ratio > 0.0:
        # 현재 구간의 폭을 계산
        width = safe_upper - safe_lower
        
        # 위아래로 폭의 margin_ratio 만큼을 더 벌려줌 (안전성 확보)
        safe_lower = safe_lower - (width * margin_ratio)
        safe_upper = safe_upper + (width * margin_ratio)
        
    return safe_lower, safe_upper


def calculate_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float: 
    """
    명세서 요구사항인 "90% 예측 구간 Coverage가 85% 이상"을 확인하기 위한 함수.
    실제값이 예측 구간 안에 포함되는 비율(Coverage)을 0.0 ~ 1.0 사이로 반환합니다.
    """ 
    is_covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(is_covered))


def calculate_mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    예측 구간의 평균 너비(Width)를 계산합니다.
    Coverage를 만족하는 선에서 이 너비가 좁을수록 모델의 불확실성이 낮고 우수함을 의미합니다.
    """
    width = upper - lower
    return float(np.mean(width))