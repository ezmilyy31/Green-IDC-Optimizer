import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.lgbm_model import LGBMForecaster
from domain.forecasting.features.builder import build_train_features

"""
냉방 수요 구간 예측(Interval Forecasting)용 LGBM Quantile 학습 실행 파일

- 5%(하한), 50%(중심), 95%(상한) 3개의 모델을 학습합니다.
- 평가 완료 후, 전체 데이터로 재학습하여 모델을 .pkl로 저장합니다.

<실행 명령어>
uv run python -m domain.forecasting.train.train_lgbm_quantile_cooling_demand
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

# --- 평가용 모델 초기화 및 학습 ---
print(f"\n[진행] 평가용 Quantile 모델 3개 학습 시작 (Target: {TARGET_COL})...")

# 1. Lower Bound (5%)
eval_model_lower = LGBMForecaster(
    target_name=TARGET_COL,
    feature_columns=selected_features,
    params={'objective': 'quantile', 'alpha': 0.05, 'n_estimators': 200}
)
eval_model_lower.fit(train_df=train_df)

# 2. Point Prediction (50%)
eval_model_point = LGBMForecaster(
    target_name=TARGET_COL,
    feature_columns=selected_features,
    params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 200}
)
eval_model_point.fit(train_df=train_df)

# 3. Upper Bound (95%)
eval_model_upper = LGBMForecaster(
    target_name=TARGET_COL,
    feature_columns=selected_features,
    params={'objective': 'quantile', 'alpha': 0.95, 'n_estimators': 200}
)
eval_model_upper.fit(train_df=train_df)

# --- Test 데이터로 예측 (정답 가리기) ---
X_test = test_df.drop(columns=[TARGET_COL]) 
y_test = test_df[TARGET_COL].values         

y_pred_lower = eval_model_lower.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values
y_pred_point = eval_model_point.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values
y_pred_upper = eval_model_upper.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values

# --- 평가 지표 계산 ---
mae = mean_absolute_error(y_test, y_pred_point)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_point))
# 분모 0 방지
mape = np.mean(np.abs((y_test - y_pred_point) / (y_test + 1e-8))) * 100

is_covered = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
coverage_rate = np.mean(is_covered) * 100
mean_interval_width = np.mean(y_pred_upper - y_pred_lower)

print("\n=== 냉방 수요 구간 예측 모델 평가 결과 (Test Data) ===")
print(f"Point MAE  (평균 절대 오차): {mae:.2f} kW")
print(f"Point RMSE (평균 제곱근 오차): {rmse:.2f} kW")
print(f"Point MAPE (평균 오차율):    {mape:.2f} %")
print("-" * 35)
print(f"Interval Coverage (포함 비율): {coverage_rate:.2f} % (목표: 85% 이상)")
print(f"Interval Mean Width (평균 폭): {mean_interval_width:.2f} kW")

# =========================================================
# 4. 실서비스 투입용 전체 데이터 재학습 (Retrain) 및 저장
# =========================================================
print("\n[안내] 평가 완료! 실제 서비스 투입을 위해 전체 데이터로 3개 모델을 재학습합니다...")

prod_model_lower = LGBMForecaster(
    target_name=TARGET_COL, feature_columns=selected_features,
    params={'objective': 'quantile', 'alpha': 0.05, 'n_estimators': 200}
)
prod_model_point = LGBMForecaster(
    target_name=TARGET_COL, feature_columns=selected_features,
    params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 200}
)
prod_model_upper = LGBMForecaster(
    target_name=TARGET_COL, feature_columns=selected_features,
    params={'objective': 'quantile', 'alpha': 0.95, 'n_estimators': 200}
)

prod_model_lower.fit(train_df=processed_df)
prod_model_point.fit(train_df=processed_df)
prod_model_upper.fit(train_df=processed_df)

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True) 

# loader.py에서 지정한 정확한 이름으로 저장
path_lower = save_dir / "lgbm_quantile_cooling_demand_lower.pkl"
path_point = save_dir / "lgbm_quantile_cooling_demand_point.pkl"
path_upper = save_dir / "lgbm_quantile_cooling_demand_upper.pkl"

prod_model_lower.save(str(path_lower))
prod_model_point.save(str(path_point))
prod_model_upper.save(str(path_upper))

print(f"\n최종 배포용 Quantile 모델 저장 완료:")
print(f" 1) {path_lower}")
print(f" 2) {path_point}")
print(f" 3) {path_upper}")