# lgbm_model.py
"""
LightGBM 기반 시계열 예측 모델 로직.

이 파일에서 구현할 것
- LightGBM regressor 생성
- 학습 데이터(X, y) 기반 모델 학습
- IT 부하 예측
- 냉각 수요 예측
- 재귀적 multi-step forecast
- feature importance 확인

이 파일의 책임
- 모델 자체의 fit / predict 로직
- 모델 하이퍼파라미터 관리
- 학습/추론 인터페이스 제공
"""

class LGBMForecaster:
    def __init__(self, params: dict | None = None): ...
    def fit(self, X, y): ...
    def predict(self, X): ...
    def forecast_recursive(self, history_df, horizon: int): ...
    def get_feature_importance(self): ...