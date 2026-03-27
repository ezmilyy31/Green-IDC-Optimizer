# intervals.py
"""
예측 구간(Prediction Interval) 계산 로직.

이 파일에서 구현할 것
- 90% prediction interval 계산
- residual 기반 interval 생성
- quantile 기반 lower/upper interval 생성
- coverage 계산
- average interval width 계산

이 파일의 책임
- 점 예측(point forecast)을 불확실성 정보와 함께 확장
- interval 품질 평가 지표 제공
"""

def build_residual_interval(preds, residuals, alpha: float = 0.1): ...
def calculate_coverage(y_true, lower, upper): ...
def calculate_mean_interval_width(lower, upper): ...
def build_quantile_interval(lower_preds, upper_preds): ...