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
data_path = "data/weather/synthetic_idc_1year_noisy.parquet"
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
    # --- 새로 추가된 열역학 피처 ---
    "theoretical_cooling_load",
    "theoretical_cop",
    "free_cooling_efficiency",
    # -----------------------------
    f"{TARGET_COL}_lag_1",    # 5분 전 칠러 전력
    f"{TARGET_COL}_lag_2",    # 10분 전 칠러 전력
    f"{TARGET_COL}_lag_12",   # 1시간 전 칠러 전력
    f"{TARGET_COL}_diff_1",   
    "it_power_x_outdoor_temp", 
    "free_cooling_x_it_power"
]
# =========================================================
# 3. 1~12월 시계열 분리 및 모델 평가 (Validation)
# =========================================================
print("\n[진행] 1월 ~ 12월 월별 교차 검증을 시작합니다.")
print("       (Quantile 모델 3개 x 12개월 = 총 36회 학습이 진행되므로 시간이 소요됩니다...)")

monthly_results = []

for month in range(1, 13):
    # 1. 월별 데이터 분리
    month_mask = (processed_df['timestamp'].dt.month == month)
    test_df = processed_df[month_mask]
    train_df = processed_df[~month_mask]

    if test_df.empty:
        continue

    print(f"\n[{month:02d}월 평가] Train: {train_df.shape[0]} rows / Test: {test_df.shape[0]} rows")

    # 2. 평가용 모델 3개 학습 (5%, 50%, 95%)
    eval_model_lower = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features, params={'objective': 'quantile', 'alpha': 0.05, 'n_estimators': 200})
    eval_model_lower.fit(train_df=train_df)

    eval_model_point = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features, params={'objective': 'quantile', 'alpha': 0.50, 'n_estimators': 200})
    eval_model_point.fit(train_df=train_df)

    eval_model_upper = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features, params={'objective': 'quantile', 'alpha': 0.95, 'n_estimators': 200})
    eval_model_upper.fit(train_df=train_df)

    # 3. Test 데이터 예측
    X_test = test_df.drop(columns=[TARGET_COL]) 
    y_test = test_df[TARGET_COL].values         

    y_pred_lower = eval_model_lower.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values
    y_pred_point = eval_model_point.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values
    y_pred_upper = eval_model_upper.predict_frame(X_test, timestamp_col="timestamp")[f"predicted_{TARGET_COL}"].values

    # 4. Point(50%) 지표 계산
    mae_p = mean_absolute_error(y_test, y_pred_point)
    mean_act = float(np.mean(y_test))
    nmae_p = (mae_p / mean_act * 100) if mean_act > 0 else 0.0

    nonzero_mask = y_test > 1.0
    if nonzero_mask.sum() > 0:
        mape_nonzero_p = np.mean(np.abs((y_test[nonzero_mask] - y_pred_point[nonzero_mask]) / y_test[nonzero_mask])) * 100
    else:
        mape_nonzero_p = 0.0

    # 5. Interval(구간) 지표 계산
    is_covered = (y_test >= y_pred_lower) & (y_test <= y_pred_upper)
    coverage_rate = np.mean(is_covered) * 100
    mean_interval_width = np.mean(y_pred_upper - y_pred_lower)

    # 결과 저장 (y_test 포함)
    monthly_results.append((month, mae_p, nmae_p, mape_nonzero_p, coverage_rate, mean_interval_width, y_test))
    print(f"  > 완료! Point nMAE: {nmae_p:.2f}%, Coverage: {coverage_rate:.2f}%")


# =========================================================
# 3-1. 1~12월 모델 평가 비교 요약 출력 
# =========================================================
print("\n=== 1~12월 모델 평가 비교 요약 (LGBM Quantile Bundle) ===")
# 표 헤더에 Coverage와 Width를 추가하여 Quantile 특성 반영
print(f"{'월 (Month)':<7} {'Point MAE':>10} {'nMAE (전체)':>16} {'MAPE (비영)':>13} {'Coverage':>10} {'Mean Width':>12} {'칠러 미가동률':>12}")
print("-" * 90)

total_mae = []
valid_nmae_list = [] 
total_coverage = []

for month, mae, nmae, mape_nonzero, cov, width, y_test in monthly_results:  
    total_mae.append(mae)
    total_coverage.append(cov)
    
    # 칠러가 사실상 가동되지 않은(1.0kW 미만) 시간의 비율 계산
    free_cooling_ratio = (y_test < 1.0).mean()
    
    # 해당 월의 90% 이상 시간 동안 칠러가 꺼져 있었다면 프리쿨링 달로 처리
    if free_cooling_ratio >= 0.90:
        nmae_str = "(free cooling)"
        mape_str = "- "
    else:
        nmae_str = f"{nmae:.2f} %"
        mape_str = f"{mape_nonzero:.2f} %"
        valid_nmae_list.append(nmae)
    
    # 출력 포맷팅
    print(f"{month:02d}월{' ':<6} {mae:>7.2f} kW {nmae_str:>16} {mape_str:>13} {cov:>9.2f} % {width:>9.2f} kW {free_cooling_ratio:>12.1%}")

print("-" * 90)
print(f"연간 평균 Point MAE : {np.mean(total_mae):.2f} kW")
print(f"연간 평균 Coverage  : {np.mean(total_coverage):.2f} %  (목표: 85% 이상)")

if valid_nmae_list:
    print(f"유효 nMAE 평균      : {np.mean(valid_nmae_list):.2f} % (프리쿨링 월 제외)")
print()

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

# =========================================================
# 5. [추가 검증] point model 피처 중요도 (Feature Importance) 출력
# =========================================================
print("\n=== 모델 피처 중요도 ===")

# 사용하는 LGBMForecaster 내부 구조에 따라 모델 객체 접근 (보통 .model 에 저장됨)
try:
    importances = prod_model_point.model.feature_importances_
    features = prod_model_point.feature_columns
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 중요도를 비율(%)로 변환
    importance_df['Importance(%)'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    
    print(importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"피처 중요도 추출 실패: {e}")