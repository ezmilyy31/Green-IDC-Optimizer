import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.lgbm_model import LGBMForecaster
from domain.forecasting.features.builder import build_train_features

"""
IT 부하 예측용 LGBM 학습 실행 파일

- 주어진 데이터를 필요에 맞게 가공 & 필요 파생변수 생성
- train data와 test data로 나누어 모델의 성능을 평가
- 평가 완료 후, 전체 데이터로 재학습하여 모델을 .joblib로 저장

<joblib를 생성하기 위한 명령어>
uv run python -m domain.forecasting.train.train_lgbm_it_load
"""

# =========================================================
# 1. 데이터 로드 및 Feature Engineering
# =========================================================

data_path = "data/processed/synthetic_idc_1year.parquet"

if not Path(data_path).exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

raw_df = pd.read_parquet(data_path)
print(f"원본 데이터 로드 완료: {raw_df.shape}")

# builder.py의 로직을 통해 파생 변수가 일괄 추가 및 결측치 제거
processed_df = build_train_features(raw_df)
print(f"파생 변수 생성 완료 (결측치 제거 후): {processed_df.shape}")

# 데이터 확인 (선택 사항)
print(f"데이터 로드 완료: {processed_df.shape}")
print(processed_df.head())


# =========================================================
# 2. 사용할 파생 변수(Features) 정의
# =========================================================
selected_features = [
    "cpu_utilization", 
    "outdoor_temp_c",      # builder.py에서 컬럼명이 변경됨
    "outdoor_humidity",    # builder.py에서 컬럼명이 변경됨
    "hour_sin", 
    "hour_cos", 
    "dow_sin", 
    "dow_cos", 
    "is_weekend",
    "temp_above_15c", 
    "temp_below_15c", 
    "humidity_temp_index",
    "it_power_kw_lag_1", 
    "it_power_kw_lag_2", 
    "it_power_kw_lag_12",
    "it_power_diff_1", 
    "it_power_diff_12",
    "it_power_x_outdoor_temp", 
    "it_power_x_humidity"
]

# =========================================================
# 3. 시계열 분리 및 모델 평가 (Validation)
# =========================================================

# 테스트 데이터 분리 (마지막 2주를 테스트용으로 사용)
# 5분 단위 데이터 -> 1시간=12개, 하루=288개, 14일=4032개
test_size = 288 * 14
train_df = processed_df.iloc[:-test_size]
test_df = processed_df.iloc[-test_size:]

print(f"\n데이터 분리 완료:")
print(f" - Train: {train_df.shape[0]} rows (학습용)")
print(f" - Test:  {test_df.shape[0]} rows (평가용)")

# 모델 초기화 및 Train 데이터로만 1차 학습
model = LGBMForecaster(
    target_name="it_power_kw",
    feature_columns=selected_features
)
model.fit(train_df=train_df)

# Test 데이터로 예측 (정답 가리기)
X_test = test_df.drop(columns=["it_power_kw"]) # 정답 가리기
y_test = test_df["it_power_kw"].values         # 실제 정답 추출
pred_df = model.predict_frame(X_test, timestamp_col="timestamp")
y_pred = pred_df["predicted_it_power_kw"].values

# 평가 지표 계산
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

print("\n=== 모델 평가 결과 (Test Data) ===")
print(f"MAE  (평균 절대 오차): {mae:.2f} kW")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f} kW")
print(f"MAPE (평균 오차율):    {mape:.2f} %")

# =========================================================
# 4. 실서비스 투입용 전체 데이터 재학습 (Retrain) 및 저장
# =========================================================

print("\n[안내] 평가 완료! 실제 서비스 투입을 위해 전체 데이터로 모델을 재학습합니다...")

prod_model = LGBMForecaster(
    target_name="it_power_kw",
    feature_columns=selected_features
)
prod_model.fit(train_df=processed_df)

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True) 

model_path = save_dir / "it_load_lgbm.joblib"
prod_model.save(str(model_path))

print(f"\n최종 배포용 모델 저장 완료: {model_path}")
print("학습에 사용된 Feature 목록:", model.feature_columns)