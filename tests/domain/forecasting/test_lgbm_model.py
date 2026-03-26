import pandas as pd
import numpy as np
from domain.forecasting.lgbm_model import LGBMForecaster

"""
학습용 가짜 데이터를 사용하여, 
lgbm_model이 잘 작동하는지 확인

<테스트 명령어>
uv run python -m tests.domain.forecasting.test_lgbm_model
"""

# 1. 학습용 가짜 데이터 생성 (100행)
train_data = {
    "timestamp": pd.date_range("2024-01-01", periods=100, freq="H"),
    "temp": np.random.uniform(20, 30, 100),
    "humidity": np.random.uniform(40, 60, 100),
    "it_load_kw": np.random.uniform(100, 500, 100) # 정답(Target)
}
train_df = pd.DataFrame(train_data)

# 2. 모델 초기화 및 학습
# feature_columns를 명시하지 않으면 target을 제외한 나머지를 자동으로 피처
# 사용할 숫자형 피처만 명시적으로 지정
model = LGBMForecaster(
    target_name="it_load_kw",
    feature_columns=["temp", "humidity"] 
)
model.fit(train_df=train_df)

# 3. 예측용 가짜 데이터 생성 (future_features.csv 역할)
# 주의: 학습 때 쓴 피처(temp, humidity)가 똑같이 들어있어야 합니다.
future_data = {
    "timestamp": pd.date_range("2024-01-05", periods=5, freq="H"),
    "temp": [25, 26, 27, 28, 29],
    "humidity": [50, 51, 52, 53, 54]
}
future_features = pd.DataFrame(future_data)

# 4. 예측 실행
pred_df = model.predict_frame(future_features, timestamp_col="timestamp")
print("--- 예측 결과 ---")
print(pred_df)

# 5. 저장 테스트
model.save("data/models/it_load_lgbm.joblib")
print("\n모델 저장 완료!")