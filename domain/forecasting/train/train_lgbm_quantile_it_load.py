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

data_path = "data/processed/synthetic_idc_1year_noisy.parquet"
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
    # --- 새로 추가된 열역학 피처 ---
    "theoretical_cooling_load",
    "theoretical_cop",
    "free_cooling_efficiency",
    # -----------------------------
    "it_power_kw_lag_1", 
    "it_power_kw_lag_2", 
    "it_power_kw_lag_12",
    "it_power_diff_1", 
    "it_power_diff_12",
    "it_power_x_outdoor_temp", 
    "it_power_x_humidity"
]


# =========================================================
# 3. 1~12월 시계열 분리 및 모델 평가 (Validation)
# =========================================================
print("\n[진행] 1월 ~ 12월 월별 교차 검증을 시작합니다.")
print("       (Quantile 모델 3개 x 12개월 = 총 36회 학습이 진행되므로 시간이 소요됩니다...)")

monthly_results = []

for month in range(1, 13):
    month_mask = (processed_df['timestamp'].dt.month == month)
    test_df = processed_df[month_mask]
    train_df = processed_df[~month_mask]

    if test_df.empty:
        continue

    print(f"\n[{month:02d}월 평가] Train: {train_df.shape[0]} rows / Test: {test_df.shape[0]} rows")

    # 모델 3개 학습
    eval_model_lower = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features, params={'objective': 'quantile', 'alpha': 0.05, 'n_estimators': 200})
    eval_model_point = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features, params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 200})
    eval_model_upper = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features, params={'objective': 'quantile', 'alpha': 0.95, 'n_estimators': 200})
    
    eval_model_lower.fit(train_df=train_df)
    eval_model_point.fit(train_df=train_df)
    eval_model_upper.fit(train_df=train_df)

    # 예측
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values

    y_pred_lower = eval_model_lower.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values
    y_pred_point = eval_model_point.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values
    y_pred_upper = eval_model_upper.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values

    # Point 지표 (50%)
    mae_p = mean_absolute_error(y_test, y_pred_point)
    mape_24h = float(np.mean(np.abs((y_test[:288] - y_pred_point[:288]) / (y_test[:288] + 1e-8))) * 100)
    end_168h = min(2016, len(y_test))
    mape_168h = float(np.mean(np.abs((y_test[:end_168h] - y_pred_point[:end_168h]) / (y_test[:end_168h] + 1e-8))) * 100)

    # Interval 지표
    is_covered = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
    coverage_rate = np.mean(is_covered) * 100
    mean_interval_width = np.mean(y_pred_upper - y_pred_lower)

    monthly_results.append((month, mae_p, mape_24h, mape_168h, coverage_rate, mean_interval_width))
    print(f"  > 완료! Point MAPE(24h): {mape_24h:.2f}%, Coverage: {coverage_rate:.2f}%")

# =========================================================
# 3-1. 1~12월 모델 평가 비교 요약 출력
# =========================================================
print("\n=== 1~12월 모델 평가 비교 요약 (LGBM Quantile Bundle) ===")
print(f"{'월 (Month)':<7} {'Point MAE':>10} {'MAPE (24h)':>15} {'MAPE (168h)':>15} {'Coverage':>12} {'Mean Width':>12}")
print("-" * 80)

total_mae, total_mape_24h, total_coverage = [], [], []

for month, mae, mape_24h, mape_168h, cov, width in monthly_results:
    total_mae.append(mae)
    total_mape_24h.append(mape_24h)
    total_coverage.append(cov)
    print(f"{month:02d}월{' ':<6} {mae:>7.2f} kW {mape_24h:>13.2f} % {mape_168h:>13.2f} % {cov:>10.2f} % {width:>9.2f} kW")

print("-" * 80)
print(f"연간 평균 Point MAE : {np.mean(total_mae):.2f} kW")
print(f"연간 평균 MAPE(24h) : {np.mean(total_mape_24h):.2f} %")
print(f"연간 평균 Coverage  : {np.mean(total_coverage):.2f} %  (목표: 85% 이상)\n")


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