import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.lgbm_model import LGBMForecaster
from domain.forecasting.features.builder import build_train_features

"""
냉방 수요 예측용 LGBM 학습 실행 파일

<joblib를 생성하기 위한 명령어>
uv run python -m domain.forecasting.train.train_lgbm_cooling_demand
"""

# =========================================================
# 1. 데이터 로드 및 Feature Engineering
# =========================================================
data_path = "data/processed/synthetic_idc_1year.parquet"
TARGET_COL = "chiller_power_kw" 

if not Path(data_path).exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

raw_df = pd.read_parquet(data_path)
print(f"원본 데이터 로드 완료: {raw_df.shape}")

# ★ builder에 target_col을 명시하여 chiller_power_kw의 Lag 변수를 생성합니다.
processed_df = build_train_features(raw_df, target_col=TARGET_COL)
print(f"파생 변수 생성 완료 (최종 데이터 크기): {processed_df.shape}\n")

# =========================================================
# 2. 사용할 파생 변수(Features) 정의
# =========================================================
selected_features = [
    "predicted_it_power_kw",  # ★ 냉방 예측의 핵심: IT 부하량
    "outdoor_temp_c",      
    "outdoor_humidity",    
    "hour_sin", 
    "hour_cos", 
    "dow_sin", 
    "dow_cos", 
    "is_weekend",
    "temp_above_15c", 
    "temp_below_15c", 
    "free_cooling_available",
    f"{TARGET_COL}_lag_1",    # 5분 전 칠러 전력
    f"{TARGET_COL}_lag_2",    # 10분 전 칠러 전력
    f"{TARGET_COL}_lag_12",   # 1시간 전 칠러 전력
    f"{TARGET_COL}_diff_1",   
    "it_power_x_outdoor_temp", 
    "free_cooling_x_it_power"
]

# =========================================================
# 3. 시계열 분리 및 모델 평가 (Validation)
# =========================================================
test_size = 288 * 14
train_df = processed_df.iloc[:-test_size]
test_df = processed_df.iloc[-test_size:]

print(f"데이터 분리 완료:")
print(f" - Train: {train_df.shape[0]} rows (학습 및 검증용)")
print(f" - Test:  {test_df.shape[0]} rows (평가용)")

eval_model = LGBMForecaster(
    target_name=TARGET_COL,
    feature_columns=selected_features
)
eval_model.fit(train_df=train_df)

X_test = test_df.drop(columns=[TARGET_COL]) 
y_test = test_df[TARGET_COL].values         

pred_df = eval_model.predict_frame(X_test, timestamp_col="timestamp")
y_pred = pred_df[f"predicted_{TARGET_COL}"].values

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

print("\n=== 냉방 수요 모델 평가 결과 (Test Data) ===")
print(f"MAE  (평균 절대 오차): {mae:.2f} kW")
print(f"RMSE (평균 제곱근 오차): {rmse:.2f} kW")
print(f"MAPE (평균 오차율):    {mape:.2f} %")

# =========================================================
# 4. 실서비스 투입용 전체 데이터 재학습 (Retrain) 및 저장
# =========================================================
print("\n[안내] 평가 완료! 실제 서비스 투입을 위해 전체 데이터로 모델을 재학습합니다...")

prod_model = LGBMForecaster(
    target_name=TARGET_COL,
    feature_columns=selected_features
)
prod_model.fit(train_df=processed_df)

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True) 

model_path = save_dir / "cooling_demand_lgbm.joblib"
prod_model.save(str(model_path))

print(f"\n최종 배포용 모델 저장 완료: {model_path}")
print("학습에 사용된 Feature 목록:", prod_model.feature_columns)