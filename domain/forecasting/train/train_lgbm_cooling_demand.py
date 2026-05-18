import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
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
# 3. 1~12월 월별 시계열 분리 및 모델 평가 (Validation)
# =========================================================
print("\n[진행] 1월 ~ 12월 월별 교차 검증을 시작합니다. (시간이 다소 소요될 수 있습니다...)")

monthly_results = []

for month in range(1, 13):
    # 1. 월별 데이터 분리 (해당 월은 Test, 나머지 11개월은 Train)
    month_mask = (processed_df['timestamp'].dt.month == month)
    test_df = processed_df[month_mask]
    train_df = processed_df[~month_mask]

    if test_df.empty:
        continue

    print(f"\n[{month}월 평가] Train: {train_df.shape[0]} rows / Test: {test_df.shape[0]} rows")

    # 2. 모델 초기화 및 학습
    eval_model = LGBMForecaster(
        target_name=TARGET_COL,
        feature_columns=selected_features
    )
    eval_model.fit(train_df=train_df)

    # 3. Test 데이터로 예측
    X_test = test_df.drop(columns=[TARGET_COL]) 
    y_test = test_df[TARGET_COL].values         

    pred_df = eval_model.predict_frame(X_test, timestamp_col="timestamp")
    y_pred = pred_df[f"predicted_{TARGET_COL}"].values

    # 4. 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mean_actual = float(np.mean(y_test))
    
    # nMAE 계산
    nmae = (mae / mean_actual * 100) if mean_actual > 0 else 0.0

    # 비영 구간 MAPE 계산
    nonzero_mask = y_test > 1.0  # 1kW 이상만 실질적 사용으로 간주
    if nonzero_mask.sum() > 0:
        mape_nonzero = np.mean(np.abs((y_test[nonzero_mask] - y_pred[nonzero_mask]) / y_test[nonzero_mask])) * 100
    else:
        mape_nonzero = 0.0

    # 결과 임시 저장
    monthly_results.append((month, mae, nmae, mape_nonzero, y_test))
    print(f"  > 완료! nMAE: {nmae:.2f} %")


# =========================================================
# 3-1. 1~12월 모델 평가 비교 요약 출력 
# =========================================================
print("\n=== 1~12월 모델 평가 비교 요약 (LGBM Forecaster) ===")
print(f"{'월 (Month)':<7} {'MAE':>10} {'nMAE (전체)':>18} {'MAPE (비영)':>15} {'칠러 미가동률':>15}")
print("-" * 80)

total_mae = []
valid_nmae_list = [] # 프리쿨링이 아닌 달의 nMAE만 모아서 연간 평균을 내기 위함

for month, mae, nmae, mape_nonzero, y_test in monthly_results:  
    total_mae.append(mae)
    
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
    
    print(f"{month:02d}월{' ':<8} {mae:>7.2f} kW {nmae_str:>18} {mape_str:>15}  {free_cooling_ratio:>15.1%}")

print("-" * 80)
print(f"연간 평균 MAE : {np.mean(total_mae):.2f} kW")

if valid_nmae_list:
    print(f"유효 nMAE 평균 : {np.mean(valid_nmae_list):.2f} % (프리쿨링 월 제외)")
print()

# CV 결과를 JSON으로 저장 — 프리쿨링 월(칠러 평균 ~0kW) 제외한 유효 nMAE만 KPI로 사용
eval_dir = Path("data/eval")
eval_dir.mkdir(parents=True, exist_ok=True)
(eval_dir / "cv_lgbm_cooling_demand.json").write_text(json.dumps({
    "generated_at":  datetime.utcnow().isoformat() + "Z",
    "model":         "lgbm_cooling_demand",
    "method":        "monthly_cv",
    "n_months":      len(monthly_results),
    "n_valid_months": len(valid_nmae_list),
    "mae_kw":        round(float(np.mean(total_mae)), 4),
    "nmae_valid_pct": round(float(np.mean(valid_nmae_list)), 4) if valid_nmae_list else None,
    "spec_nmae_pct": 10.0,
    "monthly": [
        {"month": m, "mae_kw": round(mae, 4),
         "nmae_pct": round(nmae, 4),
         "mape_nonzero_pct": round(mape_nz, 4),
         "free_cooling_ratio": round(float((yt < 1.0).mean()), 4)}
        for m, mae, nmae, mape_nz, yt in monthly_results
    ],
}, indent=2, ensure_ascii=False))
print(f"[eval] CV 결과 저장: {eval_dir / 'cv_lgbm_cooling_demand.json'}")


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


# =========================================================
# 5. [추가 검증] 피처 중요도 (Feature Importance) 출력
# =========================================================
print("\n=== 모델 피처 중요도 ===")

# 사용하는 LGBMForecaster 내부 구조에 따라 모델 객체 접근 (보통 .model 에 저장됨)
try:
    importances = prod_model.model.feature_importances_
    features = prod_model.feature_columns
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 중요도를 비율(%)로 변환
    importance_df['Importance(%)'] = (importance_df['Importance'] / importance_df['Importance'].sum()) * 100
    
    print(importance_df.head(10).to_string(index=False))
except Exception as e:
    print(f"피처 중요도 추출 실패: {e}")