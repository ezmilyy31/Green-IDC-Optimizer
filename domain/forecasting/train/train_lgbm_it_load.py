import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
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

data_path = "data/weather/synthetic_idc_1year_noisy.parquet"
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
# 3. 1~12월 월별 시계열 분리 및 모델 평가 (Validation)
# =========================================================
print("\n[진행] 1월 ~ 12월 월별 교차 검증을 시작합니다. (시간이 다소 소요됩니다...)")

monthly_results = []

for month in range(1, 13):
    # 1. 월별 데이터 분리
    month_mask = (processed_df['timestamp'].dt.month == month)
    test_df = processed_df[month_mask]
    train_df = processed_df[~month_mask]

    if test_df.empty:
        continue

    print(f"\n[{month:02d}월 평가] Train: {train_df.shape[0]} rows / Test: {test_df.shape[0]} rows")

    # 2. 모델 학습
    eval_model = LGBMForecaster(target_name=TARGET_COL, feature_columns=selected_features)
    eval_model.fit(train_df=train_df)

    # 3. Test 데이터 예측
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL].values
    
    pred_df = eval_model.predict_frame(X_test, timestamp_col="timestamp")
    y_pred = pred_df[f"predicted_{TARGET_COL}"].values

    # 4. 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mape_all = float(np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100)
    
    mape_24h = float(np.mean(np.abs((y_test[:288] - y_pred[:288]) / (y_test[:288] + 1e-8))) * 100)
    end_168h = min(2016, len(y_test))
    mape_168h = float(np.mean(np.abs((y_test[:end_168h] - y_pred[:end_168h]) / (y_test[:end_168h] + 1e-8))) * 100)

    monthly_results.append((month, mae, mape_all, mape_24h, mape_168h))
    print(f"  > 완료! MAPE(24h): {mape_24h:.2f} %")


# =========================================================
# 3-1. 1~12월 모델 평가 비교 요약 출력
# =========================================================
print("\n=== 1~12월 모델 평가 비교 요약 (LGBM Point Forecaster) ===")
print(f"{'월 (Month)':<10} {'MAE':>10} {'MAPE (전체)':>15} {'MAPE (24h)':>15} {'MAPE (168h)':>15}")
print("-" * 73)

total_mae, total_mape_24h, total_mape_168h = [], [], []

for month, mae, mape_all, mape_24h, mape_168h in monthly_results:
    total_mae.append(mae)
    total_mape_24h.append(mape_24h)
    total_mape_168h.append(mape_168h)
    print(f"{month:02d}월{' ':<8} {mae:>7.2f} kW {mape_all:>13.2f} % {mape_24h:>13.2f} % {mape_168h:>13.2f} %")

print("-" * 73)
print(f"연간 평균 MAE       : {np.mean(total_mae):.2f} kW")
print(f"연간 평균 MAPE(24h) : {np.mean(total_mape_24h):.2f} %  (요구사항: 5% 이내)")
print(f"연간 평균 MAPE(168h): {np.mean(total_mape_168h):.2f} %  (요구사항: 8% 이내)\n")

# CV 결과를 JSON으로 저장 — 대시보드 KPI는 누수 없는 이 값을 우선 사용한다.
eval_dir = Path("data/eval")
eval_dir.mkdir(parents=True, exist_ok=True)
(eval_dir / "cv_lgbm_it_load.json").write_text(json.dumps({
    "generated_at":  datetime.utcnow().isoformat() + "Z",
    "model":         "lgbm_it_load",
    "method":        "monthly_cv",
    "n_months":      len(monthly_results),
    "mae_kw":        round(float(np.mean(total_mae)), 4),
    "mape_24h_pct":  round(float(np.mean(total_mape_24h)), 4),
    "mape_168h_pct": round(float(np.mean(total_mape_168h)), 4),
    "spec_24h_pct":  5.0,
    "spec_168h_pct": 8.0,
    "monthly": [
        {"month": m, "mae_kw": round(mae, 4),
         "mape_24h_pct": round(m24, 4), "mape_168h_pct": round(m168, 4)}
        for m, mae, _all, m24, m168 in monthly_results
    ],
}, indent=2, ensure_ascii=False))
print(f"[eval] CV 결과 저장: {eval_dir / 'cv_lgbm_it_load.json'}")


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

model_path = save_dir / "it_load_lgbm.joblib"
prod_model.save(str(model_path))

print(f"\n최종 배포용 모델 저장 완료: {model_path}")
print("학습에 사용된 Feature 목록:", prod_model.feature_columns)