import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

from domain.forecasting.lgbm_model import LGBMForecaster
from domain.forecasting.features.builder import build_train_features

"""
IT 부하 구간 예측(Interval Forecasting)용 LGBM Quantile 학습 실행 파일

- 5%(하한), 50%(중심), 95%(상한) 3개의 모델을 학습합니다.
- 평가 완료 후, 전체 데이터로 재학습하여 모델을 .pkl로 저장합니다.
- loader.py에서 이 3개의 모델을 하나의 번들(bundle)로 로드하게 됩니다.

<실행 명령어>
uv run python -m domain.forecasting.train.train_lgbm_quantile_it_load
"""

# =========================================================
# 1. 데이터 로드 및 Feature Engineering
# =========================================================

data_path = "data/processed/synthetic_idc_1year.parquet"
TARGET_COL = "it_power_kw" 

if not Path(data_path).exists():
    raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")

raw_df = pd.read_parquet(data_path)
print(f"원본 데이터 로드 완료: {raw_df.shape}")

# builder.py의 로직을 통해 파생 변수가 일괄 추가 및 결측치 제거
processed_df = build_train_features(raw_df, target_col=TARGET_COL)
print(f"파생 변수 생성 완료 (결측치 제거 후): {processed_df.shape}")

# =========================================================
# 2. 사용할 파생 변수(Features) 정의
# =========================================================
selected_features = [
    "cpu_utilization", 
    "outdoor_temp_c",      
    "outdoor_humidity",    
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
test_size = 288 * 14
train_df = processed_df.iloc[:-test_size]
test_df = processed_df.iloc[-test_size:]

print(f"\n데이터 분리 완료:")
print(f" - Train: {train_df.shape[0]} rows (학습용)")
print(f" - Test:  {test_df.shape[0]} rows (평가용)")

# --- 평가용 모델 초기화 및 학습 ---
print("\n[진행] 평가용 Quantile 모델 3개 학습 시작...")

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
# Point 예측(50%)에 대한 기본 오차율 계산
mae = mean_absolute_error(y_test, y_pred_point)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_point))
mape = np.mean(np.abs((y_test - y_pred_point) / (y_test + 1e-8))) * 100

# 구간 예측(Interval)의 핵심 지표: Coverage 계산 (목표: 85% 이상)
is_covered = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
coverage_rate = np.mean(is_covered) * 100
mean_interval_width = np.mean(y_pred_upper - y_pred_lower)

print("\n=== 모델 평가 결과 (Test Data) ===")
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

# 전체 데이터로 학습
prod_model_lower.fit(train_df=processed_df)
prod_model_point.fit(train_df=processed_df)
prod_model_upper.fit(train_df=processed_df)

save_dir = Path("data/models")
save_dir.mkdir(parents=True, exist_ok=True) 

# loader.py에서 지정한 정확한 이름으로 저장 (.pkl 확장자 사용)
path_lower = save_dir / "lgbm_quantile_it_load_lower.pkl"
path_point = save_dir / "lgbm_quantile_it_load_point.pkl"
path_upper = save_dir / "lgbm_quantile_it_load_upper.pkl"

prod_model_lower.save(str(path_lower))
prod_model_point.save(str(path_point))
prod_model_upper.save(str(path_upper))

print(f"\n최종 배포용 Quantile 모델 저장 완료:")
print(f" 1) {path_lower}")
print(f" 2) {path_point}")
print(f" 3) {path_upper}")